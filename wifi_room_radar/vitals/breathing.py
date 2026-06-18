"""Breathing-rate estimation from a phase-derived chest-motion signal.

Physics background
------------------
Breathing moves the chest wall by roughly 4-12 mm. At 5 GHz the carrier
wavelength is ~5.8 cm, so a reflection path bouncing off the chest changes
length by up to ~2.4 cm per breath (the path covers the displacement twice),
i.e. a phase swing of a couple of radians — easily visible in sanitised CSI
phase. The pipeline projects the CSI onto a scalar "chest motion" series
(e.g. the dominant dynamic phase component) and decimates it to a low rate
(~20 Hz, plenty for a < 0.6 Hz signal). This module ingests that scalar
stream in small chunks and produces a :class:`~wifi_room_radar.types.VitalSign`.

Method
------
1. **Streaming band-pass** (Butterworth, second-order sections) to the
   breathing band, default 0.08-0.6 Hz (4.8-36 breaths/min). The filter state
   is carried across chunks so the output is identical to filtering one long
   recording.
2. **Ring buffer** of the last ``window_sec`` seconds of the filtered signal.
3. Once at least ~12 s of data is buffered (enough for >= 1-2 breath cycles
   even at slow rates), estimate the **Welch PSD** and pick the largest peak
   inside the band. Welch averaging (Hann windows, 50% overlap) trades a
   little resolution for variance reduction; a long zero-padded FFT plus
   **parabolic interpolation** of log-power around the peak recovers
   sub-bin rate resolution (well under 0.5 breaths/min).
4. **Confidence** = fraction of in-band power concentrated under the peak's
   spectral mainlobe, renormalised so that a flat (noise) spectrum maps to
   ~0 and a clean sinusoid maps to ~1. The confidence is derated when the
   buffer is still short (fewer Welch averages -> spurious peaks) and when
   large-body motion dominates (walking modulates the channel across the
   whole band and swamps the breathing line; pass ``motion_level``).

The returned waveform is the last ~15 s of the band-passed signal,
downsampled to <= 200 points and normalised to roughly -1..1 for display.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..types import VitalSign

__all__ = ["BreathingEstimator"]


class _RingBuffer:
    """Fixed-capacity float64 ring buffer with chronological readout.

    Shared by the breathing and heartbeat estimators (heartbeat imports it
    from this module); kept private to the vitals package.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf = np.zeros(int(capacity), dtype=np.float64)
        self._n = 0  # number of valid samples (<= capacity)
        self._pos = 0  # next write index

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._buf.size

    def clear(self) -> None:
        self._n = 0
        self._pos = 0

    def extend(self, x: np.ndarray) -> None:
        """Append samples, discarding the oldest once full. O(len(x))."""
        x = np.asarray(x, dtype=np.float64).ravel()
        cap = self._buf.size
        if x.size == 0:
            return
        if x.size >= cap:
            self._buf[:] = x[-cap:]
            self._pos = 0
            self._n = cap
            return
        end = self._pos + x.size
        if end <= cap:
            self._buf[self._pos:end] = x
        else:
            k = cap - self._pos
            self._buf[self._pos:] = x[:k]
            self._buf[: end - cap] = x[k:]
        self._pos = end % cap
        self._n = min(cap, self._n + x.size)

    def view(self) -> np.ndarray:
        """Return a chronological copy of the buffered samples."""
        if self._n < self._buf.size:
            # Until the first wrap, data occupies [0:_n) in order.
            return self._buf[: self._n].copy()
        return np.concatenate((self._buf[self._pos:], self._buf[: self._pos]))


def _parabolic_peak(freqs: np.ndarray, power: np.ndarray, idx: int) -> tuple[float, float]:
    """Refine a PSD peak location by parabolic interpolation in log-power.

    Fits a parabola through ``log(power)`` at bins ``idx-1, idx, idx+1`` and
    returns ``(refined_freq_hz, refined_power)``. Log-power is used because a
    windowed sinusoid's spectral mainlobe is close to Gaussian, i.e.
    parabolic in the log domain, which makes the three-point fit nearly
    unbiased. Falls back to the raw bin at the array edges or for degenerate
    (flat) neighbourhoods.
    """
    if idx <= 0 or idx >= power.size - 1:
        return float(freqs[idx]), float(power[idx])
    eps = 1e-300
    y0, y1, y2 = np.log(power[idx - 1 : idx + 2] + eps)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[idx]), float(power[idx])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    df = float(freqs[1] - freqs[0])
    refined_f = float(freqs[idx]) + delta * df
    refined_p = float(np.exp(y1 - 0.25 * (y0 - y2) * delta))
    return refined_f, refined_p


class BreathingEstimator:
    """Streaming breathing-rate estimator over a decimated scalar series.

    Args:
        fs: Sample rate (Hz) of the incoming scalar series (the pipeline
            supplies ~20 Hz; must be > 2 * band[1]).
        band: Analysis band in Hz; default 0.08-0.6 Hz (4.8-36 bpm).
        window_sec: Analysis history length in seconds.
        filter_order: Butterworth band-pass design order (per edge; the
            resulting SOS filter has twice this order).
    """

    #: Minimum buffered duration before any estimate is produced.
    min_analysis_sec: float = 12.0
    #: Duration of the display waveform.
    waveform_sec: float = 15.0
    #: Maximum number of points in the display waveform.
    waveform_max_points: int = 200

    def __init__(
        self,
        fs: float,
        band: tuple[float, float] = (0.08, 0.6),
        window_sec: float = 30.0,
        filter_order: int = 2,
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

        self._sos = signal.butter(filter_order, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        self._zi = signal.sosfilt_zi(self._sos)  # unit-step steady state; scaled on first chunk
        self._zi_primed = False
        self._ring = _RingBuffer(int(round(self.window_sec * self.fs)))

        # Welch segment length: ~12 s segments give 50%-overlap averaging
        # over a 30 s window while keeping enough resolution for the band.
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
    ) -> VitalSign | None:
        """Ingest a chunk of the scalar series; return the current estimate.

        Args:
            chunk: 1-D array of new samples at ``fs`` (any length, including
                empty). Non-finite samples are zeroed defensively.
            timestamp: Timestamp (seconds) of the end of the chunk; kept for
                interface symmetry, the estimator is sample-count driven.
            motion_level: Current 0..1 gross-motion level from the motion
                detector. Above 0.5 the confidence is progressively derated:
                whole-body motion produces broadband channel modulation that
                buries the breathing line, so any spectral peak found while
                walking is untrustworthy.

        Returns:
            A :class:`~wifi_room_radar.types.VitalSign`, or ``None`` while fewer
            than ``min_analysis_sec`` seconds of data are buffered (or if no
            in-band spectral estimate can be formed).
        """
        x = np.asarray(chunk, dtype=np.float64).ravel()
        if x.size:
            if not np.all(np.isfinite(x)):
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if not self._zi_primed:
                # Scale the unit-step steady state by the first sample so a
                # DC offset in the input does not ring through the band-pass.
                self._zi = self._zi * x[0]
                self._zi_primed = True
            y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
            self._ring.extend(y)

        n = len(self._ring)
        if n < int(round(self.min_analysis_sec * self.fs)):
            return None

        data = self._ring.view()
        peak = self._spectral_peak(data)
        if peak is None:
            return None
        peak_hz, conf = peak

        # Derate while the buffer is short: with a single Welch segment the
        # peak-power ratio is noisy, so cap confidence at 0.5 at the minimum
        # duration, ramping to 1.0 once the full window is buffered.
        dur = n / self.fs
        short = np.clip((dur - self.min_analysis_sec) / (self.window_sec - self.min_analysis_sec), 0.0, 1.0)
        conf *= 0.5 + 0.5 * float(short)

        # Derate under gross body motion (see docstring).
        m = float(np.clip(motion_level, 0.0, 1.0))
        if m > 0.5:
            conf *= float(np.clip((1.0 - m) / 0.5, 0.0, 1.0))

        return VitalSign(
            rate_bpm=float(peak_hz * 60.0),
            confidence=float(np.clip(conf, 0.0, 1.0)),
            waveform=self._waveform(data),
            band=self.band,
        )

    # ---------------------------------------------------------------- internals
    def _spectral_peak(self, data: np.ndarray) -> tuple[float, float] | None:
        """Welch PSD -> (peak frequency Hz, raw 0..1 confidence), or None."""
        nperseg = min(data.size, self._seg_len)
        freqs, psd = signal.welch(
            data,
            fs=self.fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=max(self._nfft, nperseg),
            detrend="constant",
        )
        in_band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if np.count_nonzero(in_band) < 3:
            return None
        fb = freqs[in_band]
        # Whiten by the band-pass response: the buffered data was Butterworth
        # filtered, so even white input noise has a mid-band PSD bump that
        # would otherwise masquerade as a "peak" and inflate the confidence.
        # Dividing by |H(f)|^2 restores a flat in-band noise spectrum while a
        # genuine breathing line is barely affected.
        pb = psd[in_band] * self._whitening(fb)
        total = float(pb.sum())
        if not np.isfinite(total) or total <= 0.0:
            return None

        # Exclude a small guard region at the band edges from the peak
        # search: out-of-band energy (gross-motion random walk below the
        # band, gait Doppler above it) leaks through the filter skirt and
        # piles up at the edge bins, forming spurious "peaks" pinned exactly
        # at the band limit. Edge bins still count toward the confidence
        # denominator, keeping the confidence honest about that energy.
        guard = 0.03  # Hz
        interior = (fb >= self.band[0] + guard) & (fb <= self.band[1] - guard)
        if not np.any(interior):
            return None
        idx = int(np.argmax(np.where(interior, pb, -np.inf)))
        peak_hz, _ = _parabolic_peak(fb, pb, idx)
        peak_hz = float(np.clip(peak_hz, self.band[0], self.band[1]))
        return peak_hz, self._peak_confidence(fb, pb, peak_hz, nperseg)

    def _whitening(self, fb: np.ndarray) -> np.ndarray:
        """Cached ``1 / |H(f)|^2`` of the analysis band-pass on the grid ``fb``.

        The in-band magnitude is >= 0.5 by construction (-3 dB edges); the
        0.25 floor merely bounds amplification against numerical edge cases.
        """
        cache = self._whiten_cache
        if cache is None or cache[0].size != fb.size or cache[0][0] != fb[0]:
            _, h = signal.sosfreqz(self._sos, worN=fb, fs=self.fs)
            mag2 = np.maximum(np.abs(h) ** 2, 0.25)
            self._whiten_cache = (fb.copy(), 1.0 / mag2)
        return self._whiten_cache[1]

    def _peak_confidence(self, fb: np.ndarray, pb: np.ndarray, peak_hz: float, nperseg: int) -> float:
        """Map (peak mainlobe power / total band power) to a 0..1 confidence.

        The "peak power" is integrated over the Hann mainlobe half-width
        (~1.5 / T_segment Hz) around the refined peak. For a flat noise
        spectrum this ratio approaches ``2 * halfwidth / bandwidth``; for a
        clean sinusoid it approaches 1. The affine map below sends the flat
        baseline to 0 and ~0.95 to 1.
        """
        t_seg = nperseg / self.fs
        halfwidth = max(0.06, 1.5 / t_seg)
        near = np.abs(fb - peak_hz) <= halfwidth
        ratio = float(pb[near].sum() / (pb.sum() + 1e-300))
        flat = float(np.clip(2.0 * halfwidth / (self.band[1] - self.band[0]), 0.02, 0.90))
        conf = (ratio - flat) / max(0.05, 0.95 - flat)
        return float(np.clip(conf, 0.0, 1.0))

    def _waveform(self, data: np.ndarray) -> list[float]:
        """Last ~15 s of the band-passed signal, <= 200 pts, roughly -1..1."""
        w = data[-int(round(self.waveform_sec * self.fs)) :]
        if w.size == 0:
            return []
        stride = int(np.ceil(w.size / self.waveform_max_points))
        # Plain decimation by striding is alias-free here: the signal is
        # already band-limited to band[1] Hz, far below the decimated
        # Nyquist rate (>= waveform_max_points / waveform_sec / 2 Hz).
        w = w[::stride]
        scale = float(np.percentile(np.abs(w), 98)) + 1e-12
        return [float(v) for v in np.clip(w / scale, -1.0, 1.0)]
