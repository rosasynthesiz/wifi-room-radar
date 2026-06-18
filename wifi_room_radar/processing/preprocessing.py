"""CSI preprocessing: phase cleaning, background subtraction, outlier
rejection and dimensionality reduction.

Why CSI phase needs cleaning
----------------------------
Commodity WiFi NICs do not share a clock between transmitter and receiver,
so every received packet carries nuisance phase terms on top of the
geometric channel response ``H[rx, tx, sub]``:

* **CFO / LO phase** -- the residual carrier-frequency offset left by the
  receiver's correction loop puts a random common phase ``exp(j*phi_t)`` on
  each packet. It is constant across subcarriers *and* across RX chains
  (all chains share one downconversion LO), but it random-walks from packet
  to packet, scrambling the phase of any single antenna's CSI over time.
* **STO / SFO timing offset** -- packet-detection jitter and sampling-clock
  drift shift the FFT window by a small delay ``dt`` per packet. A time
  shift is a *linear phase ramp across subcarriers*,
  ``exp(-j*2*pi*f_k*dt)``, and is again identical on every RX chain because
  the chains share one ADC clock.

Everything here either cancels those terms (:func:`csi_ratio`,
:func:`sanitize_phase`, :func:`align_to_background`) or operates on the
cleaned signal (:class:`BackgroundSubtractor`, :class:`HampelFilter`,
:class:`CSIBuffer`, :func:`pca_first_component`).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

_EPS = 1e-12


def csi_ratio(csi: np.ndarray) -> np.ndarray:
    """Cross-antenna conjugate CSI ratio.

    For RX chain ``i`` the measured CSI is
    ``H_i * exp(j*phi_t) * exp(-j*2*pi*f_k*dt)`` where ``H_i`` is the true
    channel, ``phi_t`` the common CFO/LO phase of this packet and ``dt`` the
    common timing offset (a linear phase slope across subcarrier frequencies
    ``f_k``). Because *all* RX chains of one NIC share the same LO and ADC
    clock, both nuisance terms are **identical across chains**. Multiplying
    chain ``i >= 1`` by the conjugate of chain ``0`` therefore cancels them
    exactly::

        r_i = H_i * conj(H_0) * |exp(j*phi_t)|^2 * |exp(-j*2*pi*f_k*dt)|^2
            = H_i * conj(H_0)

    leaving a quantity whose phase is the *inter-antenna* channel phase
    difference -- stable over time and still sensitive to moving reflectors
    (a reflector that moves changes ``H_i`` and ``H_0`` differently because
    the antennas sit at different positions). Dividing by ``|H_0|^2``
    normalises out the reference chain's amplitude so the ratio behaves
    like ``H_i / H_0``.

    Args:
        csi: Complex CSI matrix, shape ``[n_rx, n_tx, n_sub]``.

    Returns:
        Conjugate ratio, shape ``[n_rx - 1, n_tx, n_sub]``:
        ``csi[1:] * conj(csi[0]) / (|csi[0]|**2 + eps)``.
    """
    csi = np.asarray(csi)
    if csi.ndim != 3:
        raise ValueError(f"expected [n_rx, n_tx, n_sub] CSI, got shape {csi.shape}")
    if csi.shape[0] < 2:
        raise ValueError("csi_ratio needs at least 2 RX chains")
    ref = csi[0]
    return csi[1:] * np.conj(ref)[None, ...] / (np.abs(ref)[None, ...] ** 2 + _EPS)


def sanitize_phase(csi: np.ndarray) -> np.ndarray:
    """Remove the per-(rx, tx) linear phase ramp + offset across subcarriers.

    The STO/SFO timing error appears as a linear phase slope over subcarrier
    index and the CFO residual as a constant offset. For every (rx, tx) pair
    this fits ``phase[k] ~ slope*k + offset`` to the *unwrapped* phase by
    ordinary least squares over the subcarrier axis and subtracts the fit,
    keeping the measured amplitude untouched.

    Note the classic caveat: the true propagation delay also contributes a
    linear phase ramp, so this sanitisation destroys absolute time-of-flight
    information. What survives is the *non-linear* phase structure across
    subcarriers (frequency-selective multipath), which is what motion and
    occupancy sensing rely on.

    Args:
        csi: Complex CSI, shape ``[..., n_sub]`` (typically
            ``[n_rx, n_tx, n_sub]``); the fit runs along the last axis
            independently for every leading index.

    Returns:
        CSI with the same amplitude and detrended phase, same shape/dtype
        family as the input (complex128).
    """
    csi = np.asarray(csi)
    n_sub = csi.shape[-1]
    amp = np.abs(csi)
    phase = np.unwrap(np.angle(csi), axis=-1)

    # Centred subcarrier index -> sum(k) == 0, so the OLS solution decouples:
    # slope = sum(k * phi) / sum(k^2), offset = mean(phi).
    k = np.arange(n_sub, dtype=float) - (n_sub - 1) / 2.0
    denom = float(np.sum(k * k))
    if denom > 0.0:
        slope = (phase @ k) / denom  # shape = leading dims
    else:  # single subcarrier: only an offset to remove
        slope = np.zeros(csi.shape[:-1])
    offset = phase.mean(axis=-1)
    fit = slope[..., None] * k + offset[..., None]
    return amp * np.exp(1j * (phase - fit))


def align_to_background(csi: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Phase-align one raw CSI frame to a static background estimate.

    Each frame ``csi`` differs from the background ``B`` by (at least) the
    per-packet common phase offset ``b`` (CFO random walk) and common linear
    phase slope ``a`` across subcarriers (STO jitter), both shared by every
    (rx, tx) element::

        csi[r, t, k] ~ B[r, t, k] * exp(j * (a*k + b)) + dynamic part

    Subtracting an unaligned frame from the background would be dominated by
    these nuisance rotations rather than by actual scene change. This
    function estimates ``(a, b)`` by weighted least squares on the wrapped
    per-element phase error ``angle(csi * conj(background))`` (the product's
    angle handles the 2*pi wrap), with weights ``|background|`` so strong,
    reliable subcarriers dominate, then removes ``exp(j*(a*k + b))`` from
    ``csi``. After alignment, ``csi - background`` is a geometrically
    meaningful dynamic component.

    To stay robust when the offset sits near +-pi, the offset is first
    coarse-estimated as the angle of the weighted circular mean of the
    phase errors; the WLS then only fits the small residual.

    Args:
        csi: Single frame, shape ``[n_rx, n_tx, n_sub]``.
        background: Static background estimate, same shape.

    Returns:
        Aligned copy of ``csi``, same shape.
    """
    csi = np.asarray(csi)
    background = np.asarray(background)
    if csi.shape != background.shape:
        raise ValueError(f"shape mismatch: csi {csi.shape} vs background {background.shape}")
    n_sub = csi.shape[-1]

    p = csi * np.conj(background)  # angle(p) = wrapped phase error per element
    w = np.abs(background)
    sw = float(np.sum(w))
    if sw <= _EPS:
        return csi.copy()

    # Coarse common offset via the weighted circular mean (wrap-safe even
    # when the offset is near +-pi).
    u = p / (np.abs(p) + _EPS)  # unit phasors of the phase error
    b0 = float(np.angle(np.sum(w * u)))

    # Residual wrapped phase, now centred near zero so plain WLS is valid.
    phi = np.angle(u * np.exp(-1j * b0))

    # Weighted least squares of phi ~ a*(k - kbar) + db over all elements,
    # with the subcarrier index k broadcast along the last axis.
    k = np.arange(n_sub, dtype=float)
    kk = np.broadcast_to(k, csi.shape)
    kbar = float(np.sum(w * kk)) / sw
    kc = kk - kbar
    skk = float(np.sum(w * kc * kc))
    a = float(np.sum(w * kc * phi)) / skk if skk > _EPS else 0.0
    db = float(np.sum(w * phi)) / sw

    correction = np.exp(-1j * (a * (k - kbar) + b0 + db))  # [n_sub]
    return csi * correction


class BackgroundSubtractor:
    """Exponential-moving-average estimate of the static CSI background.

    The static channel (walls, furniture, direct path) dominates raw CSI;
    people add a small time-varying perturbation on top. An EMA over
    *phase-aligned* complex frames converges to the static part because the
    zero-mean dynamic perturbations average out, and ``aligned - background``
    isolates the moving-reflector component used by all downstream sensing.

    Alignment before averaging is essential: the per-packet CFO random walk
    rotates each raw frame by an arbitrary common phase, so an EMA of raw
    frames would average phasors with random orientation toward zero instead
    of converging to the true static channel.

    Args:
        alpha: EMA coefficient in (0, 1];
            ``background <- (1 - alpha)*background + alpha*aligned``.
            Small alpha (e.g. 0.01) tracks slow drift (temperature, slight
            furniture moves) while staying insensitive to people moving
            through the scene.
    """

    def __init__(self, alpha: float, n_nodes: int = 1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
        self.alpha = float(alpha)
        #: Receiver-node count for multistatic frames: a frame's rows are the
        #: stacked elements of ``n_nodes`` independent radio chains, each with
        #: its own CFO/STO, so the phase alignment must be fitted per node
        #: block rather than once across all rows.
        self.n_nodes = int(n_nodes)
        self._background: Optional[np.ndarray] = None

    @property
    def background(self) -> Optional[np.ndarray]:
        """Copy of the current background estimate (None before the first frame)."""
        return None if self._background is None else self._background.copy()

    def update(self, csi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Consume one frame; return ``(dynamic, background_copy)``.

        Cold start: the first frame initialises the background and the
        dynamic component is all zeros. From the second frame on, the frame
        is phase-aligned to the current background
        (:func:`align_to_background`), the EMA is updated with the *aligned*
        frame, and ``dynamic = aligned - background`` (against the freshly
        updated background) is returned.

        Args:
            csi: Single frame ``[n_rx, n_tx, n_sub]``.

        Returns:
            ``(dynamic, background_copy)``, both the same shape as ``csi``.
        """
        csi = np.asarray(csi, dtype=np.complex128)
        if self._background is None:
            self._background = csi.copy()
            return np.zeros_like(csi), self._background.copy()

        if self.n_nodes <= 1:
            aligned = align_to_background(csi, self._background)
        else:
            # Independent radio chains: fit phase offset + slope per node.
            per = csi.shape[0] // self.n_nodes
            aligned = np.empty_like(csi)
            for i in range(self.n_nodes):
                sl = slice(i * per, (i + 1) * per)
                aligned[sl] = align_to_background(csi[sl], self._background[sl])
        self._background = (1.0 - self.alpha) * self._background + self.alpha * aligned
        dynamic = aligned - self._background
        return dynamic, self._background.copy()

    @staticmethod
    def motion_energy(dynamic: np.ndarray) -> float:
        """Mean squared magnitude of the dynamic component (motion power)."""
        return float(np.mean(np.abs(dynamic) ** 2))


class HampelFilter:
    """Streaming Hampel (rolling median / MAD) outlier rejector.

    CSI streams contain impulsive glitches: AGC re-locks, packet decode
    errors, interference bursts. A Hampel filter keeps a trailing window of
    the last ``window`` samples and, elementwise across the feature
    dimension, replaces any value further than ``n_sigma`` robust standard
    deviations from the rolling median with that median. The robust sigma is
    ``1.4826 * MAD`` (the factor makes the median absolute deviation a
    consistent estimator of the Gaussian standard deviation), so isolated
    spikes are removed while genuine signal trends pass through.

    Complex-valued samples are filtered on real and imaginary parts
    independently (the median of complex numbers is not well defined).

    Args:
        window: Trailing window length (includes the current sample).
        n_sigma: Rejection threshold in robust standard deviations.
    """

    #: MAD -> Gaussian sigma consistency constant: 1 / Phi^-1(3/4).
    _MAD_SCALE = 1.4826

    def __init__(self, window: int = 11, n_sigma: float = 3.0):
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        self.window = int(window)
        self.n_sigma = float(n_sigma)
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)

    def __len__(self) -> int:
        return len(self._buf)

    def _filter_real(self, stack: np.ndarray, current: np.ndarray) -> np.ndarray:
        med = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - med), axis=0)
        sigma = self._MAD_SCALE * mad
        out = current.copy()
        mask = np.abs(current - med) > self.n_sigma * sigma + _EPS
        out[mask] = med[mask]
        return out

    def push(self, x: np.ndarray) -> np.ndarray:
        """Add one sample; return it with outlier elements replaced.

        Until the window is full the sample is stored and returned
        unchanged (not enough history for a robust median).

        Args:
            x: Sample of any fixed shape (vector, matrix, ...); filtering is
                elementwise across that shape.

        Returns:
            Filtered sample, same shape as ``x``.
        """
        x = np.asarray(x)
        self._buf.append(x.copy())
        if len(self._buf) < self.window:
            return x

        stack = np.stack(self._buf, axis=0)  # [window, ...feature shape]
        if np.iscomplexobj(stack):
            real = self._filter_real(stack.real, np.real(x).copy())
            imag = self._filter_real(stack.imag, np.imag(x).copy())
            return real + 1j * imag
        return self._filter_real(stack, x)


class CSIBuffer:
    """Fixed-capacity ring buffer of ``(timestamp, feature_vector)`` pairs.

    Used to assemble sliding windows of CSI features for batch analyses
    (PCA, occupancy mapping, vitals spectra) while the pipeline keeps
    streaming. Once full, the oldest entries are overwritten. The feature
    length and dtype are fixed by the first push.

    Args:
        capacity: Maximum number of samples retained.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._ts = np.zeros(self.capacity, dtype=np.float64)
        self._X: Optional[np.ndarray] = None
        self._head = 0  # next write index
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def push(self, t: float, x: np.ndarray) -> None:
        """Append one sample, overwriting the oldest when full.

        Args:
            t: Timestamp in seconds.
            x: Feature vector (flattened to 1-D on storage).
        """
        x = np.ravel(np.asarray(x))
        if self._X is None:
            dtype = x.dtype if np.issubdtype(x.dtype, np.inexact) else np.float64
            self._X = np.zeros((self.capacity, x.size), dtype=dtype)
        elif x.size != self._X.shape[1]:
            raise ValueError(
                f"feature length changed: buffer holds {self._X.shape[1]}, got {x.size}"
            )
        self._ts[self._head] = float(t)
        self._X[self._head] = x
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def array(self) -> tuple[np.ndarray, np.ndarray]:
        """Valid contents in time order.

        Returns:
            ``(ts, X)`` with ``ts`` shape ``[T]`` and ``X`` shape ``[T, F]``
            (copies; safe to mutate). Empty arrays when nothing was pushed.
        """
        if self._count == 0 or self._X is None:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        idx = (self._head - self._count + np.arange(self._count)) % self.capacity
        return self._ts[idx].copy(), self._X[idx].copy()


def pca_first_component(X: np.ndarray) -> np.ndarray:
    """Project a CSI feature window onto its dominant principal component.

    A single moving reflector (a breathing chest, a walking person)
    modulates all CSI features coherently -- each subcarrier/antenna sees a
    scaled, rotated copy of the same underlying path-length variation. The
    first principal component therefore concentrates the motion signal while
    averaging down independent per-feature noise, and the projected time
    series is what vitals/Doppler analysis runs on.

    The dominant right-singular direction of the mean-removed ``X = U S V^H``
    is computed via the smaller of the two Gram matrices for efficiency:
    if ``F <= T``, ``eigh`` on the ``F x F`` covariance ``Xc^H Xc`` gives
    ``v1`` and the series is ``Xc @ v1``; otherwise ``eigh`` on the
    ``T x T`` Gram ``Xc Xc^H`` gives ``u1`` and the identical series is
    ``u1 * s1``. The result is deterministic up to sign, which is fixed by
    requiring the series to start with a non-negative real part.

    Args:
        X: Complex feature matrix, shape ``[T, F]`` (time by feature).

    Returns:
        Complex time series of length ``T``: the dominant motion component.
    """
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 2:
        raise ValueError(f"expected [T, F] matrix, got shape {X.shape}")
    n_t, n_f = X.shape
    if n_t == 0:
        return np.zeros(0, dtype=np.complex128)

    Xc = X - X.mean(axis=0, keepdims=True)
    if n_f <= n_t:
        cov = Xc.conj().T @ Xc  # [F, F] Hermitian PSD
        vals, vecs = np.linalg.eigh(cov)
        if vals[-1] <= _EPS:
            return np.zeros(n_t, dtype=np.complex128)
        series = Xc @ vecs[:, -1]
    else:
        gram = Xc @ Xc.conj().T  # [T, T] Hermitian PSD
        vals, vecs = np.linalg.eigh(gram)
        top = float(np.real(vals[-1]))
        if top <= _EPS:
            return np.zeros(n_t, dtype=np.complex128)
        series = vecs[:, -1] * np.sqrt(top)

    if series[0].real < 0.0:
        series = -series
    return series
