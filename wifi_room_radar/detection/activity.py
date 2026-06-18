"""Room-level activity classification from Doppler + motion features.

Deliberately classical: the four activity classes are separable with two
robust features, no trained model required.

* ``idle``      — no presence.
* ``micro``     — presence without gross motion (someone still; breathing
                  micro-motion only).
* ``walking``   — sustained gross motion with broadband Doppler: locomotion
                  produces continuous limb/torso returns over tens of Hz for
                  seconds at a time.
* ``gesturing`` — intermittent gross motion: arm/hand movements arrive as
                  short bursts with rest between them, so the motion duty
                  cycle over a few seconds sits well below walking's.

A minimum-dwell hysteresis stops the label flapping at class boundaries.
"""
from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["ActivityClassifier"]


class ActivityClassifier:
    """Streaming activity labeller; call :meth:`update` at the publish rate.

    Args:
        window_sec: History length for the motion duty-cycle feature.
        motion_on: Motion level treated as "moving" for the duty cycle.
        walk_duty: Duty cycle at/above which sustained motion reads as
            walking (below it, bursts read as gesturing).
        broadband_hz: |Doppler| above this counts as broadband (locomotion)
            energy for the walking check.
        min_dwell_sec: Minimum time a new label must persist before the
            output switches (debounce).
    """

    def __init__(
        self,
        window_sec: float = 4.0,
        motion_on: float = 0.3,
        walk_duty: float = 0.35,
        broadband_hz: float = 8.0,
        min_dwell_sec: float = 1.5,
    ) -> None:
        self.window_sec = float(window_sec)
        self.motion_on = float(motion_on)
        self.walk_duty = float(walk_duty)
        self.broadband_hz = float(broadband_hz)
        self.min_dwell_sec = float(min_dwell_sec)
        self._hist: deque[tuple[float, bool]] = deque()
        self._bb_hist: deque[tuple[float, bool]] = deque()
        self._label = "idle"
        self._candidate = "idle"
        self._candidate_since = float("-inf")

    def update(
        self,
        timestamp: float,
        motion_level: float,
        presence: bool,
        doppler_db: np.ndarray | None,
        doppler_freqs: np.ndarray | None,
    ) -> str:
        """Classify the current instant; returns the (debounced) label."""
        t = float(timestamp)
        self._hist.append((t, float(motion_level) >= self.motion_on))
        while self._hist and self._hist[0][0] < t - self.window_sec:
            self._hist.popleft()
        duty = float(np.mean([m for _, m in self._hist])) if self._hist else 0.0

        # Windowed Doppler-line presence: a walker's return is a narrow line
        # at their current radial velocity, present most of the time but
        # vanishing at waypoint pauses; gestures produce such lines only in
        # brief flashes. Window the boolean so neither sampling instant
        # decides the label by itself.
        self._bb_hist.append((t, self._fast_line_present(doppler_db, doppler_freqs)))
        while self._bb_hist and self._bb_hist[0][0] < t - self.window_sec:
            self._bb_hist.popleft()
        line_frac = float(np.mean([b for _, b in self._bb_hist])) if self._bb_hist else 0.0

        if not presence:
            raw = "idle"
        elif duty < 0.08:
            raw = "micro"
        else:
            raw = (
                "walking"
                if (duty >= self.walk_duty and line_frac >= 0.4)
                else "gesturing"
            )

        # Minimum-dwell debounce: a raw label must persist before adoption.
        if raw == self._label:
            self._candidate = raw
        elif raw != self._candidate:
            self._candidate = raw
            self._candidate_since = t
        elif t - self._candidate_since >= self.min_dwell_sec:
            self._label = raw
        return self._label

    def _fast_line_present(
        self, doppler_db: np.ndarray | None, freqs: np.ndarray | None
    ) -> bool:
        """True when a strong Doppler line exists beyond ``broadband_hz``.

        Locomotion shows up as a narrow line at the body's current radial
        velocity (well beyond breathing/clutter), so the test is the MAX of
        the fast bins against the column's own median floor — a fraction-of-
        bins test would miss it, because one moving body only ever lights a
        few bins at a time.
        """
        if doppler_db is None or freqs is None or len(doppler_db) == 0:
            return True  # no spectrum available: defer to the duty cycle
        col = np.asarray(doppler_db, dtype=float)
        f = np.asarray(freqs, dtype=float)
        out = np.abs(f) >= self.broadband_hz
        if not np.any(out):
            return True
        floor = float(np.median(col))
        return float(np.max(col[out])) > floor + 10.0
