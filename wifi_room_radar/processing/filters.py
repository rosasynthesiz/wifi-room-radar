"""Streaming time-domain filters for CSI-derived signals.

The pipeline pushes small chunks of samples (a few frames at a time at
200 Hz), so every filter here carries its own internal state across calls:
the IIR delay line of :class:`StreamingSOSFilter` and the sample counter of
:class:`Decimator` make chunk-wise processing bit-identical to filtering the
whole stream in one call.

Typical uses in wifi_room_radar:

* band-passing the motion time series into the breathing (0.08-0.6 Hz),
  heartbeat (0.8-2.2 Hz) and subvocal (15-80 Hz) bands,
* decimating the 200 Hz CSI-rate motion signal down to ``vitals_fs``
  (20 Hz) before the slow vitals analysis,
* light smoothing / envelope extraction for display.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def design_bandpass(low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Design a Butterworth band-pass filter in second-order sections.

    Butterworth gives a maximally flat passband (no ripple to masquerade as
    periodicity when we later peak-pick a vitals spectrum) and the SOS form
    keeps the high-order cascade numerically stable -- important for the
    very narrow normalised bands a 0.08 Hz edge implies at fs = 200 Hz.

    The band edges are clipped to lie safely inside the open interval
    ``(0, fs/2)`` so a configuration that nominally touches DC or Nyquist
    (e.g. a 15-80 Hz subvocal band at a low sample rate) still yields a
    valid, stable design instead of raising inside scipy.

    Args:
        low: Lower band edge in Hz.
        high: Upper band edge in Hz.
        fs: Sample rate in Hz.
        order: Filter order (of the underlying lowpass prototype).

    Returns:
        SOS array of shape ``[n_sections, 6]`` for use with
        :func:`scipy.signal.sosfilt` / :class:`StreamingSOSFilter`.
    """
    nyq = fs / 2.0
    lo = max(float(low), 1e-6 * nyq)
    hi = min(float(high), 0.999 * nyq)
    if not lo < hi:
        raise ValueError(
            f"invalid band ({low}, {high}) Hz at fs={fs} Hz: clips to ({lo}, {hi})"
        )
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


class StreamingSOSFilter:
    """Chunk-wise IIR filter with persistent second-order-section state.

    Wraps :func:`scipy.signal.sosfilt` and keeps the ``zi`` delay-line state
    between calls, so feeding the signal in arbitrary chunk sizes produces
    exactly the same output as filtering it in one pass. On the first chunk
    the state is initialised to the filter's step-response steady state
    scaled by the first sample (:func:`scipy.signal.sosfilt_zi`), which
    suppresses the start-up transient that a zero state would ring with.

    Works on real or complex 1-D data (complex is the common case here:
    the dynamic CSI component is a complex phasor).

    Args:
        sos: Second-order sections, shape ``[n_sections, 6]``.
    """

    def __init__(self, sos: np.ndarray):
        sos = np.asarray(sos, dtype=np.float64)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError(f"expected sos of shape [n_sections, 6], got {sos.shape}")
        self.sos = sos
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        """Forget all filter state (next chunk starts a fresh stream)."""
        self._zi = None

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Filter one chunk, continuous with all previously pushed chunks.

        Args:
            chunk: 1-D array of samples (real or complex). May be empty.

        Returns:
            Filtered samples, same length and dtype family as ``chunk``.
        """
        x = np.asarray(chunk)
        if x.ndim != 1:
            raise ValueError(f"expected 1-D chunk, got shape {x.shape}")
        if x.size == 0:
            return x.copy()
        if self._zi is None:
            zi0 = signal.sosfilt_zi(self.sos)  # [n_sections, 2], unit-step state
            self._zi = zi0.astype(np.result_type(zi0.dtype, x.dtype)) * x[0]
        y, self._zi = signal.sosfilt(self.sos, x, zi=self._zi)
        return y


class Decimator:
    """Integer-factor streaming decimator with anti-alias lowpass.

    Downsampling by ``q`` folds everything above the new Nyquist
    ``fs_out/2`` back into the band, so the input is first lowpassed with a
    streaming Butterworth filter (order 8, cutoff at ``0.8 * fs_out/2``, the
    same margin :func:`scipy.signal.decimate` uses) and then every ``q``-th
    sample is kept.

    Phase continuity across chunks: a global input-sample counter decides
    which samples survive (those whose absolute index is a multiple of
    ``q``), so splitting the stream into chunks of any size — including
    chunks shorter than the decimation factor — yields exactly the same
    output sequence as decimating the whole stream at once.

    Args:
        fs_in: Input sample rate in Hz.
        fs_out: Output sample rate in Hz; ``fs_in / fs_out`` must be a
            positive integer.
    """

    _AA_ORDER = 8  # anti-alias Butterworth order

    def __init__(self, fs_in: float, fs_out: float):
        ratio = float(fs_in) / float(fs_out)
        q = int(round(ratio))
        if q < 1 or abs(ratio - q) > 1e-9:
            raise ValueError(
                f"fs_in/fs_out must be a positive integer, got {fs_in}/{fs_out} = {ratio}"
            )
        self.fs_in = float(fs_in)
        self.fs_out = float(fs_out)
        self.factor = q
        if q > 1:
            cutoff = 0.8 * (self.fs_out / 2.0)
            sos = signal.butter(self._AA_ORDER, cutoff, btype="lowpass", fs=self.fs_in, output="sos")
            self._lpf: StreamingSOSFilter | None = StreamingSOSFilter(sos)
        else:
            self._lpf = None  # factor 1: filtering would only distort
        self._n_in = 0  # input samples consumed so far, modulo factor

    def reset(self) -> None:
        """Forget all state (filter memory and sample-phase counter)."""
        if self._lpf is not None:
            self._lpf.reset()
        self._n_in = 0

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Consume one chunk; return the decimated samples it produces.

        Args:
            chunk: 1-D array of samples (real or complex). May be empty or
                shorter than the decimation factor — output then may be
                empty and the deficit carries into the next chunk.

        Returns:
            1-D array of 0 or more output samples at ``fs_out``.
        """
        x = np.asarray(chunk)
        if x.ndim != 1:
            raise ValueError(f"expected 1-D chunk, got shape {x.shape}")
        if x.size == 0:
            return x.copy()
        y = self._lpf.push(x) if self._lpf is not None else x
        # First sample in this chunk whose *global* index is divisible by q.
        offset = (-self._n_in) % self.factor
        out = y[offset :: self.factor]
        self._n_in = (self._n_in + x.size) % self.factor
        return np.array(out, copy=True)


def savgol_smooth(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay smoothing with safe edge handling.

    Fits a degree-``poly`` polynomial in a sliding ``window`` and evaluates
    it at the centre: preserves peak positions and low-order waveform shape
    (good for breathing waveforms shown on the dashboard) far better than a
    plain moving average of the same bandwidth.

    The window is clipped to the signal length and forced odd; if the signal
    is too short for the requested polynomial order the input is returned
    unchanged. Complex input is smoothed on real and imaginary parts
    independently.

    Args:
        x: 1-D signal.
        window: Nominal window length in samples.
        poly: Polynomial order (must be < effective window).

    Returns:
        Smoothed signal, same length as ``x``.
    """
    x = np.asarray(x)
    n = x.size
    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w <= poly or w < 3:
        return np.array(x, copy=True)
    if np.iscomplexobj(x):
        return (
            signal.savgol_filter(x.real, w, poly)
            + 1j * signal.savgol_filter(x.imag, w, poly)
        )
    return signal.savgol_filter(x, w, poly)


def moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    """Moving root-mean-square envelope.

    Square-law detection followed by a boxcar average: the standard
    short-time energy envelope, used e.g. to score subvocal micro-motion
    bursts. Edges use the partial window actually available (so the output
    is unbiased at the boundaries rather than tapering to zero). Magnitude
    squared is used, so complex input yields its (real) RMS envelope.

    Args:
        x: 1-D signal (real or complex).
        window: Averaging window length in samples (clipped to ``len(x)``).

    Returns:
        Real RMS envelope, same length as ``x``.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    w = max(1, min(int(window), x.size))
    power = np.abs(x).astype(np.float64) ** 2
    kernel = np.ones(w, dtype=np.float64)
    num = np.convolve(power, kernel, mode="same")
    den = np.convolve(np.ones(x.size, dtype=np.float64), kernel, mode="same")
    return np.sqrt(num / den)
