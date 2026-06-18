"""Motion detection from dynamic CSI energy.

Physics background
------------------
A person moving through a room perturbs the WiFi multipath channel: the
reflection off the moving body adds a time-varying component on top of the
otherwise static channel response. Upstream processing subtracts a slowly
adapting estimate of the static background channel and collapses the residual
("dynamic") CSI to a single non-negative energy scalar per frame, e.g. the
power of the background-subtracted CSI averaged over RX antennas and
subcarriers. This module turns that raw scalar stream into a calibrated
0..1 motion level and a debounced boolean decision.

Why an adaptive noise floor: the absolute scale of the dynamic energy depends
on transmit power, link distance, hardware gain and (in simulation) the SNR
setting, so fixed thresholds do not transfer between setups. We instead track
a running low quantile (default: 10th percentile over the last ~30 s) of the
energy itself. In a quiet room essentially every frame is noise, so the low
quantile converges to the noise level; while somebody moves, motion frames
live in the upper tail of the distribution and barely influence a 10th
percentile. The motion level is then a soft saturating-exponential
normalisation of the energy *excess* over this floor: "barely above noise"
maps near 0, "many times the noise floor" saturates towards 1, with no hard
clipping artefacts in between. A short exponential smoother (time constant
``window_sec``) removes per-frame jitter, and a two-threshold hysteresis
keeps the boolean output from flickering when the level rides a threshold.

Known limitation: if motion is sustained for much longer than the floor
window (someone pacing for minutes on end), the low quantile slowly absorbs
some motion energy and sensitivity degrades gracefully. This is the standard
trade-off of self-calibrating detectors and is acceptable for room sensing.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionResult:
    """Output of :class:`MotionDetector.update` for one frame.

    Attributes:
        level: Smoothed, noise-floor-normalised motion level in 0..1.
        detected: Debounced boolean motion decision (hysteresis applied).
        raw_energy: The unmodified input energy for this frame (useful for
            logging / debugging the calibration).
    """

    level: float
    detected: bool
    raw_energy: float


class MotionDetector:
    """Streaming motion detector over a per-frame dynamic-CSI energy scalar.

    Call :meth:`update` exactly once per CSI frame (i.e. at the full frame
    rate ``fs``, typically 200 Hz).

    Args:
        fs: Frame rate in Hz at which ``update`` is called.
        window_sec: Time constant of the exponential smoother applied to the
            instantaneous motion level.
        hysteresis: ``(on_threshold, off_threshold)`` applied to the smoothed
            level. ``detected`` turns on when the level rises to or above the
            on-threshold and only turns off again once it falls to or below
            the off-threshold.
        floor_window_sec: Length of the history used for the adaptive noise
            floor estimate.
        floor_quantile: Quantile (0..1) of the energy history used as the
            noise floor. Low values make the floor robust to long stretches
            of motion in the window.
        sensitivity: Scale of the soft normalisation, in multiples of the
            noise floor. The level reaches ~0.63 when the energy exceeds the
            floor by ``sensitivity * floor``; smaller values are more
            sensitive.
    """

    def __init__(
        self,
        fs: float,
        window_sec: float = 1.0,
        hysteresis: tuple[float, float] = (0.25, 0.15),
        floor_window_sec: float = 30.0,
        floor_quantile: float = 0.10,
        sensitivity: float = 3.0,
    ) -> None:
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        on_thr, off_thr = hysteresis
        if not (0.0 < off_thr <= on_thr <= 1.0):
            raise ValueError(f"hysteresis must satisfy 0 < off <= on <= 1, got {hysteresis}")
        self.fs = float(fs)
        self.window_sec = float(window_sec)
        self.hysteresis = (float(on_thr), float(off_thr))
        self.floor_quantile = float(floor_quantile)
        self.sensitivity = float(sensitivity)

        # Bounded energy history for the quantile-based noise floor.
        self._energy_hist: deque[float] = deque(maxlen=max(8, int(round(floor_window_sec * self.fs))))
        # Recomputing a percentile over the full history every frame would be
        # wasteful at 200 fps; refresh ~4x per second instead.
        self._floor_update_every = max(1, int(round(self.fs / 4.0)))
        self._frames_since_floor = 0
        # Require ~1 s of history before trusting the quantile estimate.
        self._min_floor_samples = max(8, int(round(self.fs)))
        self._floor: float | None = None
        self._median: float = 0.0  # long-run median, used to regularise the soft-norm scale

        # One-pole smoother: alpha chosen so the step response time constant
        # is window_sec at the given frame rate.
        self._alpha = 1.0 - math.exp(-1.0 / max(1e-6, self.fs * self.window_sec))
        self._level = 0.0
        self._detected = False

    def reset(self) -> None:
        """Forget all adaptive state (noise floor, smoothing, hysteresis)."""
        self._energy_hist.clear()
        self._frames_since_floor = 0
        self._floor = None
        self._median = 0.0
        self._level = 0.0
        self._detected = False

    def update(self, energy: float, timestamp: float) -> MotionResult:
        """Ingest one frame's dynamic-CSI energy and return the motion state.

        Args:
            energy: Non-negative dynamic CSI energy for this frame. Negative
                or non-finite inputs are clamped to 0.
            timestamp: Frame timestamp in seconds (kept for interface
                symmetry / logging; the detector is sample-count driven).

        Returns:
            A :class:`MotionResult` with the smoothed level, the debounced
            detection flag and the raw input energy.
        """
        e = float(energy)
        if not math.isfinite(e) or e < 0.0:
            e = 0.0
        self._energy_hist.append(e)

        # --- adaptive noise floor -------------------------------------------------
        self._frames_since_floor += 1
        if (self._floor is None or self._frames_since_floor >= self._floor_update_every) and len(
            self._energy_hist
        ) >= self._min_floor_samples:
            hist = np.fromiter(self._energy_hist, dtype=np.float64, count=len(self._energy_hist))
            self._floor = float(np.percentile(hist, self.floor_quantile * 100.0))
            self._median = float(np.median(hist))
            self._frames_since_floor = 0

        if self._floor is None:
            # Warm-up: be conservative (assume everything so far is noise).
            floor = max(self._energy_hist) if self._energy_hist else e
        else:
            floor = self._floor

        # --- soft normalisation ---------------------------------------------------
        # level_inst = 1 - exp(-excess / scale). The 5% median term keeps the
        # scale sane when the low quantile is ~0 (e.g. an unnaturally clean
        # simulated stream), avoiding a divide-by-epsilon blow-up.
        excess = max(0.0, e - floor)
        scale = self.sensitivity * floor + 0.05 * self._median + 1e-12
        level_inst = 1.0 - math.exp(-excess / scale)
        level_inst = min(1.0, max(0.0, level_inst))

        # --- temporal smoothing + hysteresis -------------------------------------
        self._level += self._alpha * (level_inst - self._level)
        on_thr, off_thr = self.hysteresis
        if self._detected:
            if self._level <= off_thr:
                self._detected = False
        else:
            if self._level >= on_thr:
                self._detected = True

        return MotionResult(level=float(self._level), detected=self._detected, raw_energy=e)
