"""Experimental subvocalization / micro-vibration burst detector.

Honesty statement
-----------------
Detecting subvocalization (silent speech: laryngeal and articulator
micro-movements without audible sound) from commodity WiFi CSI is **not an
established capability**. The laryngeal surface displacement involved is on
the order of tens of micrometres — about two orders of magnitude below
breathing motion and well below the phase-noise floor of typical commodity
hardware at realistic ranges. Published demonstrations of related ideas
(e.g. WiFi-based throat/lip motion sensing) rely on very short ranges,
directional antennas, or specialised radios, and none establish robust
room-scale subvocalization detection.

This module is therefore a **research harness**, not a product feature. The
wifi_room_radar simulator models subvocalization as bursty micro-vibration in the
15-80 Hz band (:class:`wifi_room_radar.config.PersonSpec.subvocal`), and this
detector is built to find exactly that signature, providing an end-to-end
scaffold (band-pass bank -> energy tracking -> adaptive floor -> burst
discrimination) on which future work with better hardware can iterate. Every
emitted :class:`~wifi_room_radar.types.SubvocalState` carries
``note="experimental"`` and downstream consumers must surface that caveat.

Method
------
The input is a scalar CSI-derived series at the FULL frame rate (200 Hz),
so the 15-80 Hz analysis band is observable (Nyquist 100 Hz).

1. **Sub-band energy bank**: the band is split into ``n_subbands`` equal
   slices, each with its own streaming Butterworth band-pass (SOS, state
   carried across chunks). Per sub-band, the squared output is smoothed by a
   one-pole IIR with a ~0.5 s time constant — a streaming RMS tracker.
2. **Adaptive noise floor**: per sub-band, the smoothed energy (in dB) is
   snapshotted at ~20 Hz into a ~20 s history; the 10th percentile of that
   history is the floor. As with motion detection, this self-calibrates
   against hardware gain and steady interferers (a fan raises the floor of
   the bands it occupies and is thereafter ignored).
3. **Activity score** = mean over sub-bands of the dB exceedance above the
   floor (after a small deadband that absorbs the floor-estimation jitter of
   stationary noise), clipped and scaled so that a ~10 dB mean exceedance
   saturates the score at 1. Requiring exceedance across multiple sub-bands
   favours the broadband signature of articulation bursts over narrowband
   interference.
4. **Burstiness gate**: speech-like activity is amplitude-modulated at the
   syllabic rate (~2-8 Hz), so the broadband energy envelope over the last
   ~2 s has a high coefficient of variation (std/mean), whereas fan hum or
   mains interference is stationary (CV near 0). Crucially this statistic is
   computed on a *fast* (~50 ms) envelope — the 0.5 s RMS used for scoring
   would smooth the syllabic modulation away. ``active`` requires the score
   to clear an on/off hysteresis AND the fast-envelope CV to exceed a
   threshold.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

from ..types import SubvocalState

__all__ = ["SubvocalDetector"]

_DB_EPS = 1e-30  # power floor before log10, ~ -300 dB


class SubvocalDetector:
    """Streaming detector for bursty micro-vibration energy in 15-80 Hz CSI.

    Call :meth:`update` with consecutive chunks of the full-rate scalar
    series; chunk length is arbitrary and state is carried across calls.

    Args:
        fs: Sample rate (Hz) of the input series — the full CSI frame rate,
            typically 200. Must exceed ``2 * band[1]``.
        band: Overall analysis band in Hz.
        n_subbands: Number of equal-width sub-bands the band is split into.
        rms_sec: Time constant of the per-sub-band RMS energy smoother.
        floor_window_sec: History length for the adaptive per-sub-band noise
            floor.
        floor_quantile: Quantile (0..1) of the energy history used as the
            noise floor.
        hysteresis: ``(on, off)`` thresholds on the activity score for the
            ``active`` flag.
        burst_window_sec: Length of the envelope history used for the
            burstiness (std/mean) statistic.
        burst_rms_sec: Time constant of the *fast* broadband envelope used
            for the burstiness statistic. Must be short enough to preserve
            syllabic-rate (~2-8 Hz) amplitude modulation.
        burst_cv_threshold: Minimum envelope coefficient of variation for
            activity to count as burst-like (speech-like) rather than a
            steady tone/hum.
        exceed_scale_db: Mean dB exceedance over the floor at which the
            activity score saturates at 1.
        score_bias_db: Deadband subtracted from the mean exceedance before
            scaling; absorbs the ~1-2 dB jitter of the smoothed energy of
            stationary noise relative to its low-quantile floor.
    """

    def __init__(
        self,
        fs: float,
        band: tuple[float, float] = (15.0, 80.0),
        n_subbands: int = 4,
        rms_sec: float = 0.5,
        floor_window_sec: float = 20.0,
        floor_quantile: float = 0.10,
        hysteresis: tuple[float, float] = (0.35, 0.20),
        burst_window_sec: float = 2.0,
        burst_rms_sec: float = 0.05,
        burst_cv_threshold: float = 0.35,
        exceed_scale_db: float = 10.0,
        score_bias_db: float = 1.5,
    ) -> None:
        lo, hi = float(band[0]), float(band[1])
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if not (0.0 < lo < hi < fs / 2.0):
            raise ValueError(f"band {band} must satisfy 0 < lo < hi < fs/2 ({fs / 2.0})")
        if n_subbands < 1:
            raise ValueError(f"n_subbands must be >= 1, got {n_subbands}")
        on_thr, off_thr = hysteresis
        if not (0.0 < off_thr <= on_thr <= 1.0):
            raise ValueError(f"hysteresis must satisfy 0 < off <= on <= 1, got {hysteresis}")

        self.fs = float(fs)
        self.band = (lo, hi)
        self.n_subbands = int(n_subbands)
        self.hysteresis = (float(on_thr), float(off_thr))
        self.burst_cv_threshold = float(burst_cv_threshold)
        self.exceed_scale_db = float(exceed_scale_db)
        self.score_bias_db = float(score_bias_db)
        self.floor_quantile = float(floor_quantile)

        # --- parallel sub-band band-pass bank --------------------------------
        edges = np.linspace(lo, hi, self.n_subbands + 1)
        self.subband_edges: list[tuple[float, float]] = [
            (float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])
        ]
        self._sos: list[np.ndarray] = []
        self._zi: list[np.ndarray] = []
        self._zi_unit: list[np.ndarray] = []  # unit-step steady states, for priming
        for a, b in self.subband_edges:
            sos = signal.butter(4, [a, b], btype="bandpass", fs=self.fs, output="sos")
            self._sos.append(sos)
            self._zi_unit.append(signal.sosfilt_zi(sos))
            self._zi.append(np.zeros_like(self._zi_unit[-1]))
        self._primed = False

        # --- streaming RMS (one-pole IIR on the squared signal) --------------
        alpha = 1.0 - np.exp(-1.0 / max(1e-6, self.fs * rms_sec))
        self._ema_b = np.array([alpha])
        self._ema_a = np.array([1.0, alpha - 1.0])
        self._ema_zi = [np.zeros(1) for _ in range(self.n_subbands)]
        # Fast broadband envelope for the burstiness statistic only.
        alpha_f = 1.0 - np.exp(-1.0 / max(1e-6, self.fs * burst_rms_sec))
        self._fast_b = np.array([alpha_f])
        self._fast_a = np.array([1.0, alpha_f - 1.0])
        self._fast_zi = np.zeros(1)

        # --- snapshotting for floor / burst tracking --------------------------
        # The smoothed energy varies on a >= rms_sec scale, so sampling it at
        # ~20 Hz loses nothing while keeping the histories small.
        self._snap_every = max(1, int(round(self.fs * 0.05)))
        snap_rate = self.fs / self._snap_every
        self._snap_phase = 0  # samples until the next snapshot, within the next chunk
        hist_len = max(8, int(round(floor_window_sec * snap_rate)))
        self._floor_hist: list[deque[float]] = [deque(maxlen=hist_len) for _ in range(self.n_subbands)]
        self._floor_min_snaps = max(8, int(round(2.0 * snap_rate)))  # ~2 s warm-up
        self._floor_update_snaps = max(1, int(round(snap_rate / 2.0)))  # refresh every ~0.5 s
        self._snaps_since_floor = 0
        self._floors: np.ndarray | None = None  # dB, per sub-band
        self._burst_hist: deque[float] = deque(maxlen=max(8, int(round(burst_window_sec * snap_rate))))
        self._burst_min_snaps = max(8, int(round(1.0 * snap_rate)))  # need ~1 s for a CV

        # --- decision state ----------------------------------------------------
        self._score_latched = False
        self._active = False
        self._gated_until = float("-inf")  # motion-gate re-arm deadline
        self._last_state = SubvocalState(
            active=False, activity_score=0.0, band_energy=[0.0] * self.n_subbands, note="experimental"
        )

    def reset(self) -> None:
        """Forget all filter, floor and decision state."""
        for i in range(self.n_subbands):
            self._zi[i] = np.zeros_like(self._zi_unit[i])
            self._ema_zi[i] = np.zeros(1)
            self._floor_hist[i].clear()
        self._fast_zi = np.zeros(1)
        self._primed = False
        self._snap_phase = 0
        self._snaps_since_floor = 0
        self._floors = None
        self._burst_hist.clear()
        self._score_latched = False
        self._active = False
        self._gated_until = float("-inf")
        self._last_state = SubvocalState(
            active=False, activity_score=0.0, band_energy=[0.0] * self.n_subbands, note="experimental"
        )

    #: Gross-motion level (0..1) above which the detector reports inactive:
    #: gait and gesture micro-Doppler land squarely in the 15-80 Hz band with
    #: a bursty, speech-like envelope, so micro-vibration sensing on a moving
    #: person is hopeless (and real subvocalization work assumes a still
    #: subject). The filters keep running so the floor stays warm.
    motion_gate_level: float = 0.35
    #: Seconds the detector stays suppressed after the motion level last
    #: exceeded the gate: the band-pass ring-down and RMS-tracker tail of a
    #: gait burst outlive the motion flag itself and would otherwise fire the
    #: detector right after every pause in walking.
    rearm_sec: float = 3.0

    # ------------------------------------------------------------------ update
    def update(
        self, chunk: np.ndarray, timestamp: float, motion_level: float = 0.0
    ) -> SubvocalState:
        """Ingest a chunk of the full-rate scalar series; return the state.

        Args:
            chunk: 1-D array of new samples at ``fs`` (any length; an empty
                chunk returns the previous state unchanged). Non-finite
                samples are zeroed defensively.
            timestamp: Timestamp (seconds) of the end of the chunk; kept for
                interface symmetry, the detector is sample-count driven.
            motion_level: Current gross-motion level (0..1) from the motion
                detector; above ``motion_gate_level`` the detector is
                suppressed (see the class attribute). During suppressed
                stretches the energy snapshots are also kept out of the
                noise-floor history so gait energy cannot inflate the floor.

        Returns:
            The current :class:`~wifi_room_radar.types.SubvocalState`. Until ~2 s
            of noise-floor history has accumulated the detector reports
            inactive with score 0. ``band_energy`` holds the per-sub-band
            dB exceedance over the adaptive floor.
        """
        x = np.asarray(chunk, dtype=np.float64).ravel()
        if x.size == 0:
            return self._last_state
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        t = float(timestamp)
        if float(motion_level) > self.motion_gate_level:
            self._gated_until = t + self.rearm_sec
        gated = t < self._gated_until

        if not self._primed:
            # Prime each band-pass with the steady state for a constant input
            # at x[0], suppressing the DC step transient on the first chunk.
            for i in range(self.n_subbands):
                self._zi[i] = self._zi_unit[i] * x[0]
            self._primed = True
            prime_ema = True
        else:
            prime_ema = False

        # --- snapshot indices for this chunk (phase carried across chunks) ----
        snap_idx = np.arange(self._snap_phase, x.size, self._snap_every)
        self._snap_phase = int((self._snap_phase - x.size) % self._snap_every)

        # --- filter bank + RMS tracking ---------------------------------------
        band_db_now = np.empty(self.n_subbands)
        y2_broadband = np.zeros(x.size)
        alpha = float(self._ema_b[0])
        for i in range(self.n_subbands):
            y, self._zi[i] = signal.sosfilt(self._sos[i], x, zi=self._zi[i])
            y2 = y * y
            y2_broadband += y2
            if prime_ema:
                # Start the RMS tracker at the first chunk's mean power so the
                # floor history is not polluted by a from-zero ramp.
                self._ema_zi[i] = np.array([(1.0 - alpha) * float(y2.mean())])
            p, self._ema_zi[i] = signal.lfilter(self._ema_b, self._ema_a, y2, zi=self._ema_zi[i])
            band_db_now[i] = 10.0 * np.log10(p[-1] + _DB_EPS)
            if snap_idx.size and not gated:
                self._floor_hist[i].extend(10.0 * np.log10(p[snap_idx] + _DB_EPS))
        # Fast broadband envelope feeds the burstiness statistic only: the
        # slow (rms_sec) tracker above would iron out the syllabic on/off
        # structure that distinguishes speech-like bursts from steady hum.
        if prime_ema:
            alpha_f = float(self._fast_b[0])
            self._fast_zi = np.array([(1.0 - alpha_f) * float(y2_broadband.mean())])
        pf, self._fast_zi = signal.lfilter(self._fast_b, self._fast_a, y2_broadband, zi=self._fast_zi)
        if snap_idx.size and not gated:
            self._burst_hist.extend(float(v) for v in pf[snap_idx])

        # --- adaptive per-sub-band noise floor ---------------------------------
        self._snaps_since_floor += int(snap_idx.size)
        if (
            self._floors is None or self._snaps_since_floor >= self._floor_update_snaps
        ) and len(self._floor_hist[0]) >= self._floor_min_snaps:
            self._floors = np.array(
                [
                    np.percentile(np.fromiter(h, dtype=np.float64, count=len(h)), self.floor_quantile * 100.0)
                    for h in self._floor_hist
                ]
            )
            self._snaps_since_floor = 0

        if self._floors is None:
            # Warm-up: no calibrated floor yet, report quiescent.
            self._last_state = SubvocalState(
                active=False,
                activity_score=0.0,
                band_energy=[0.0] * self.n_subbands,
                note="experimental",
            )
            return self._last_state

        if gated:
            # Gross motion: micro-vibration sensing is meaningless, report
            # quiescent (band_energy stays informative for debugging).
            self._score_latched = False
            self._active = False
            self._last_state = SubvocalState(
                active=False,
                activity_score=0.0,
                band_energy=[float(v) for v in np.maximum(0.0, band_db_now - self._floors)],
                note="experimental",
            )
            return self._last_state

        # --- activity score -----------------------------------------------------
        exceed_db = np.maximum(0.0, band_db_now - self._floors)
        score = float(
            np.clip((float(exceed_db.mean()) - self.score_bias_db) / self.exceed_scale_db, 0.0, 1.0)
        )

        # --- burstiness (speech is bursty, fan hum is not) ----------------------
        if len(self._burst_hist) >= self._burst_min_snaps:
            env = np.fromiter(self._burst_hist, dtype=np.float64, count=len(self._burst_hist))
            mean = float(env.mean())
            cv = float(env.std() / (mean + _DB_EPS)) if mean > 0.0 else 0.0
        else:
            cv = 0.0
        # Mild hysteresis on the burstiness gate too, so the AND of the two
        # conditions does not chatter at the boundary.
        cv_needed = self.burst_cv_threshold if not self._active else 0.7 * self.burst_cv_threshold
        bursty = cv >= cv_needed

        # --- score hysteresis + final decision ----------------------------------
        on_thr, off_thr = self.hysteresis
        if self._score_latched:
            if score <= off_thr:
                self._score_latched = False
        else:
            if score >= on_thr:
                self._score_latched = True
        self._active = self._score_latched and bursty

        self._last_state = SubvocalState(
            active=self._active,
            activity_score=score,
            band_energy=[float(v) for v in exceed_db],
            note="experimental",
        )
        return self._last_state
