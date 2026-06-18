"""Heartbeat-rate estimation from a phase-derived chest-motion signal.

Physics background
------------------
The ballistocardiographic motion of the chest surface from the heartbeat is
~0.2-0.5 mm — roughly 15x smaller than breathing displacement, hence a power
ratio of ~200x against the breathing line. At 5 GHz (lambda ~5.8 cm) that is
only a few hundredths of a radian of path phase, so the heartbeat sits barely
above the phase-noise floor of even a well-sanitised CSI stream. Two
consequences shape this module:

1. **Harmonic leakage**: breathing is not sinusoidal (inhale/exhale
   asymmetry), so its harmonics (2f, 3f, 4f, ...) extend well into the
   cardiac band (0.8-2.2 Hz, i.e. 48-132 bpm) at amplitudes that can dwarf
   the true cardiac line. Given the currently estimated breathing frequency
   (``breathing_hz``), PSD bins within +/-0.08 Hz of every integer multiple
   of it are clamped to the in-band median before peak picking, so the peak
   search cannot lock onto a breathing harmonic.
2. **Honest confidence**: even under ideal conditions the cardiac peak is
   weak, so the confidence is never allowed to reach 1.0 (ceiling 0.85), is
   derated when many band bins had to be suppressed as harmonics, and is
   derated by gross body motion starting at a much lower motion level than
   for breathing (0.3 instead of 0.5). Downstream consumers should treat
   anything below ~0.3 as unreliable, per :class:`~wifi_room_radar.types.VitalSign`.

The signal path is otherwise identical to
:class:`~wifi_room_radar.vitals.breathing.BreathingEstimator` (whose ring-buffer
and parabolic-interpolation helpers are reused here): streaming Butterworth
band-pass, ring buffer, Welch PSD, in-band peak with parabolic refinement,
peak-power-ratio confidence.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..types import VitalSign
from .breathing import _parabolic_peak, _RingBuffer

__all__ = ["HeartbeatEstimator"]


class HeartbeatEstimator:
    """Streaming heart-rate estimator over a decimated scalar series.

    Same interface as :class:`~wifi_room_radar.vitals.breathing.BreathingEstimator`;
    :meth:`update` additionally accepts the current breathing frequency for
    harmonic suppression.

    Args:
        fs: Sample rate (Hz) of the incoming scalar series (~20 Hz from the
            pipeline; must be > 2 * band[1]).
        band: Analysis band in Hz; default 0.8-2.2 Hz (48-132 bpm).
        window_sec: Analysis history length in seconds.
        filter_order: Butterworth band-pass design order (per edge).
        harmonic_guard_hz: Half-width (Hz) of the notch applied around each
            integer multiple of ``breathing_hz``.
    """

    #: Minimum buffered duration before any estimate is produced.
    min_analysis_sec: float = 12.0
    #: Duration of the display waveform.
    waveform_sec: float = 15.0
    #: Maximum number of points in the display waveform.
    waveform_max_points: int = 200
    #: Hard ceiling on confidence — a single-antenna commodity-WiFi cardiac
    #: estimate should never be presented as certain.
    confidence_ceiling: float = 0.85
    #: Peak candidates within this many Hz of either band edge are excluded
    #: from the peak search: the band-pass skirt lets the (200x stronger)
    #: breathing tail leak across the low edge, where whitening then re-boosts
    #: it into a spurious rising ramp whose argmax sits exactly on the edge
    #: bin. Edge bins still contribute to the confidence denominator.
    edge_guard_hz: float = 0.1
    #: Highest breathing harmonic that is notched. Chest displacement is
    #: quasi-sinusoidal, so the phase-modulation comb decays like Bessel
    #: J_n of a ~1.3 rad index: by n = 5 the line power is ~60 dB below the
    #: fundamental — far weaker than a real cardiac line. Notching without
    #: this cap erases the true heart rate whenever it lands near an exact
    #: multiple of the breathing rate (e.g. 70 bpm on 14 bpm breathing).
    max_harmonic: int = 4
    #: Minimum prominence (dB) for a spectral line to be considered in the
    #: peak search (see the prominence-based picking in ``_spectral_peak``).
    min_prominence_db: float = 3.0

    def __init__(
        self,
        fs: float,
        band: tuple[float, float] = (0.8, 2.2),
        window_sec: float = 30.0,
        filter_order: int = 2,
        harmonic_guard_hz: float = 0.08,
    ) -> None:
        lo, hi = float(band[0]), float(band[1])
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if not (0.0 < lo < hi < fs / 2.0):
            raise ValueError(f"band {band} must satisfy 0 < lo < hi < fs/2 ({fs / 2.0})")
        if window_sec <= self.min_analysis_sec:
            raise ValueError(f"window_sec must exceed {self.min_analysis_sec}s, got {window_sec}")
        self.fs = float(fs)
        self.band = (lo, hi)
        self.window_sec = float(window_sec)
        self.harmonic_guard_hz = float(harmonic_guard_hz)

        self._sos = signal.butter(filter_order, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        self._zi = signal.sosfilt_zi(self._sos)
        self._zi_primed = False
        self._ring = _RingBuffer(int(round(self.window_sec * self.fs)))

        self._seg_len = max(16, int(round(self.fs * min(self.min_analysis_sec, self.window_sec / 2.0))))
        self._nfft = max(4096, 1 << int(np.ceil(np.log2(self._seg_len))))
        self._whiten_cache: tuple[np.ndarray, np.ndarray] | None = None  # (fb, 1/|H|^2)

    def reset(self) -> None:
        """Forget the buffered signal and the band-pass filter state."""
        self._ring.clear()
        self._zi = signal.sosfilt_zi(self._sos)
        self._zi_primed = False

    # ------------------------------------------------------------------ update
    def update(
        self,
        chunk: np.ndarray,
        timestamp: float,
        motion_level: float = 0.0,
        breathing_hz: float | None = None,
    ) -> VitalSign | None:
        """Ingest a chunk of the scalar series; return the current estimate.

        Args:
            chunk: 1-D array of new samples at ``fs`` (any length, including
                empty). Non-finite samples are zeroed defensively.
            timestamp: Timestamp (seconds) of the end of the chunk; kept for
                interface symmetry, the estimator is sample-count driven.
            motion_level: Current 0..1 gross-motion level. Derating starts at
                0.3 — the cardiac line is so weak that even fidgeting buries
                it — and reaches zero confidence by 0.8.
            breathing_hz: Currently estimated breathing fundamental in Hz
                (e.g. ``breathing.rate_bpm / 60``). When provided (and
                physically plausible, >= 0.05 Hz), PSD bins near its
                harmonics are suppressed before peak picking.

        Returns:
            A :class:`~wifi_room_radar.types.VitalSign`, or ``None`` while fewer
            than ``min_analysis_sec`` seconds of data are buffered, or when
            harmonic suppression leaves too few clean bins to pick a peak.
        """
        x = np.asarray(chunk, dtype=np.float64).ravel()
        if x.size:
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if not self._zi_primed:
                self._zi = self._zi * x[0]
                self._zi_primed = True
            y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
            self._ring.extend(y)

        n = len(self._ring)
        if n < int(round(self.min_analysis_sec * self.fs)):
            return None

        data = self._ring.view()
        peak = self._spectral_peak(data, breathing_hz)
        if peak is None:
            return None
        peak_hz, conf, clean_frac = peak

        # Short-buffer derate (same rationale as the breathing estimator).
        dur = n / self.fs
        short = np.clip((dur - self.min_analysis_sec) / (self.window_sec - self.min_analysis_sec), 0.0, 1.0)
        conf *= 0.5 + 0.5 * float(short)

        # Derate when breathing harmonics forced us to discard a large slice
        # of the band: the surviving peak had less competition, so the
        # peak-power ratio is optimistically biased.
        conf *= 0.5 + 0.5 * clean_frac

        # Aggressive motion derate: zero confidence by motion_level 0.8.
        m = float(np.clip(motion_level, 0.0, 1.0))
        if m > 0.3:
            conf *= float(np.clip((0.8 - m) / 0.5, 0.0, 1.0))

        conf = min(conf, self.confidence_ceiling)

        return VitalSign(
            rate_bpm=float(peak_hz * 60.0),
            confidence=float(np.clip(conf, 0.0, 1.0)),
            waveform=self._waveform(data),
            band=self.band,
        )

    # ---------------------------------------------------------------- internals
    def _spectral_peak(
        self, data: np.ndarray, breathing_hz: float | None
    ) -> tuple[float, float, float] | None:
        """Welch PSD with harmonic suppression.

        Returns ``(peak_hz, raw_confidence, clean_bin_fraction)`` or ``None``.
        """
        nperseg = min(data.size, self._seg_len)
        # Blackman-Harris rather than Hann: its first sidelobe is -92 dB
        # (vs -31 dB), so the 20+ dB-stronger breathing-harmonic lines do not
        # smear leakage skirts across the cardiac band. The wider mainlobe
        # (~2x Hann) costs resolution we can spare: +/-0.1 Hz is +/-6 bpm of
        # window width but the parabolic peak interpolation recovers the line
        # centre far more finely than the mainlobe width.
        freqs, psd = signal.welch(
            data,
            fs=self.fs,
            window="blackmanharris",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=max(self._nfft, nperseg),
            detrend="constant",
        )
        in_band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if np.count_nonzero(in_band) < 3:
            return None
        fb = freqs[in_band]
        # Whiten by the band-pass response so the Butterworth passband shape
        # cannot masquerade as a spectral peak (see the breathing estimator).
        pb = psd[in_band] * self._whitening(fb)
        if not np.isfinite(pb.sum()) or pb.sum() <= 0.0:
            return None

        # --- breathing harmonic suppression ---------------------------------
        clean_frac = 1.0
        if breathing_hz is not None and breathing_hz >= 0.05:
            # Distance of each bin to its nearest integer multiple of the
            # breathing fundamental. Within the cardiac band every bin's
            # nearest multiple has n >= 1, so n = 0 needs no special casing.
            # Only multiples up to ``max_harmonic`` are notched (see the class
            # attribute): higher breathing harmonics are negligible while a
            # coincident true cardiac line is not.
            ratio = fb / breathing_hz
            n_mult = np.round(ratio)
            dist = np.abs(ratio - n_mult) * breathing_hz
            harmonic = (dist <= self.harmonic_guard_hz) & (n_mult <= self.max_harmonic)
            clean = ~harmonic
            n_clean = int(np.count_nonzero(clean))
            if n_clean < 3:
                return None  # the breathing comb covers the whole band
            # Clamp (never raise) contaminated bins to the clean median so
            # they cannot win the peak search yet still contribute a sane
            # amount to the band-power denominator of the confidence.
            fill = float(np.median(pb[clean]))
            pb[harmonic] = np.minimum(pb[harmonic], fill)
            clean_frac = n_clean / pb.size

        # Peak search restricted to the band interior (see edge_guard_hz);
        # the full band still forms the confidence denominator.
        interior = (fb >= self.band[0] + self.edge_guard_hz) & (
            fb <= self.band[1] - self.edge_guard_hz
        )
        if not np.any(interior):
            return None

        # Pick by *prominence*, not height. The breathing comb's broad Welch
        # skirts (window leakage of lines 20+ dB stronger than the cardiac
        # one) routinely out-power the true cardiac line even after notching,
        # but a skirt is smooth — it has no prominence — while the cardiac
        # component is a narrow line standing ~10 dB above its local floor.
        db = 10.0 * np.log10(pb + 1e-300)
        pk_idx, props = signal.find_peaks(db, prominence=self.min_prominence_db)
        if pk_idx.size:
            ok = interior[pk_idx]
            pk_idx, prom = pk_idx[ok], props["prominences"][ok]
        else:
            prom = np.empty(0)
        if pk_idx.size:
            best = int(np.argmax(prom))
            idx = int(pk_idx[best])
            # Confidence from the line's prominence — the statistic the pick
            # is actually based on. The classic peak-power/band-power ratio
            # is meaningless here: the band total is dominated by breathing
            # harmonic skirts 20+ dB above any credible cardiac line, so a
            # perfectly clean cardiac line would still score near zero.
            # ~min_prominence_db maps to 0 and a 12 dB-prominent line to 1.
            conf = float(
                np.clip((float(prom[best]) - self.min_prominence_db) / 9.0, 0.0, 1.0)
            )
        else:
            # No prominent line anywhere: fall back to the tallest interior
            # bin, scored by the (pessimistic) power-ratio confidence so the
            # estimate is kept appropriately humble.
            idx = int(np.argmax(np.where(interior, pb, -np.inf)))
            peak_hz0 = float(fb[idx])
            conf = self._peak_confidence(fb, pb, peak_hz0, nperseg)
        peak_hz, _ = _parabolic_peak(fb, pb, idx)
        peak_hz = float(np.clip(peak_hz, self.band[0], self.band[1]))
        return peak_hz, conf, float(clean_frac)

    def _whitening(self, fb: np.ndarray) -> np.ndarray:
        """Cached ``1 / |H(f)|^2`` of the analysis band-pass on the grid ``fb``."""
        cache = self._whiten_cache
        if cache is None or cache[0].size != fb.size or cache[0][0] != fb[0]:
            _, h = signal.sosfreqz(self._sos, worN=fb, fs=self.fs)
            mag2 = np.maximum(np.abs(h) ** 2, 0.25)
            self._whiten_cache = (fb.copy(), 1.0 / mag2)
        return self._whiten_cache[1]

    def _peak_confidence(self, fb: np.ndarray, pb: np.ndarray, peak_hz: float, nperseg: int) -> float:
        """Peak-mainlobe power over total band power, mapped to 0..1.

        Identical construction to the breathing estimator; because the
        cardiac band (1.4 Hz) is much wider than the Hann mainlobe, the flat
        baseline is low and the ratio discriminates well.
        """
        t_seg = nperseg / self.fs
        halfwidth = max(0.06, 1.5 / t_seg)
        near = np.abs(fb - peak_hz) <= halfwidth
        ratio = float(pb[near].sum() / (pb.sum() + 1e-300))
        flat = float(np.clip(2.0 * halfwidth / (self.band[1] - self.band[0]), 0.02, 0.90))
        conf = (ratio - flat) / max(0.05, 0.95 - flat)
        return float(np.clip(conf, 0.0, 1.0))

    def _waveform(self, data: np.ndarray) -> list[float]:
        """Last ~15 s of the band-passed signal, <= 200 pts, roughly -1..1.

        Note: at 200 points / 15 s the display rate is ~13.3 Hz, comfortably
        above twice the band edge (2.2 Hz), so strided decimation of the
        already band-limited signal does not alias.
        """
        w = data[-int(round(self.waveform_sec * self.fs)) :]
        if w.size == 0:
            return []
        stride = int(np.ceil(w.size / self.waveform_max_points))
        w = w[::stride]
        scale = float(np.percentile(np.abs(w), 98)) + 1e-12
        return [float(v) for v in np.clip(w / scale, -1.0, 1.0)]
