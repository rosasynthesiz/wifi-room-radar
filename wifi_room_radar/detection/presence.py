"""Presence detection fusing motion and vital-sign evidence.

Why fusion is needed: dynamic-CSI motion energy is excellent at catching
walking and gesturing, but a person sitting perfectly still produces almost
no broadband motion energy — their only channel signature is the tiny
quasi-periodic chest displacement from breathing (millimetres) and the even
smaller cardiac pulse. A presence detector based on motion alone would
therefore declare the room empty whenever its occupant relaxes.

The fix is the classic occupancy-sensing fusion: *presence = recent motion OR
a credible periodic vitals signal*. The breathing estimator's confidence is a
peak-to-band power ratio, so a confident breathing estimate means there is a
genuinely periodic 0.08-0.6 Hz component in the channel — fans and HVAC do
not breathe at human rates with human regularity, and an empty room's noise
spectrum is flat in that band.

Both evidence sources are intermittent (a still person stops "moving", a
walking person's breathing estimate is swamped by gross motion), so the
output is latched: presence persists for ``hold_sec`` after the most recent
positive evidence, which bridges the hand-off gap between the two detectors
and prevents the dashboard from blinking.
"""
from __future__ import annotations

from .motion import MotionResult


class PresenceDetector:
    """Latched presence decision from motion + breathing-confidence evidence.

    Args:
        hold_sec: How long presence stays asserted after the last positive
            evidence (motion detection or credible breathing).
        breathing_conf_threshold: Minimum breathing confidence (0..1) that
            counts as evidence of a (possibly motionless) person. The default
            0.45 sits well above the ~0.3 "unreliable" floor documented on
            :class:`wifi_room_radar.types.VitalSign` so spurious spectral peaks in
            an empty room do not latch presence.
        breathing_debounce_sec: Breathing confidence must stay at or above
            the threshold for this long *continuously* before it counts as
            evidence. An empty room's Welch peak ratio occasionally spikes
            above any sane threshold for a moment, and with ``hold_sec``
            latching, even rare spikes inflate the presence duty cycle badly;
            a real motionless person's breathing peak is stable for minutes,
            so the debounce costs only a small detection delay.
    """

    def __init__(
        self,
        hold_sec: float = 5.0,
        breathing_conf_threshold: float = 0.45,
        breathing_debounce_sec: float = 2.5,
    ) -> None:
        if hold_sec < 0:
            raise ValueError(f"hold_sec must be non-negative, got {hold_sec}")
        if breathing_debounce_sec < 0:
            raise ValueError(
                f"breathing_debounce_sec must be non-negative, got {breathing_debounce_sec}"
            )
        self.hold_sec = float(hold_sec)
        self.breathing_conf_threshold = float(breathing_conf_threshold)
        self.breathing_debounce_sec = float(breathing_debounce_sec)
        self._last_evidence_t: float | None = None
        self._breath_ok_since: float | None = None
        self._present = False

    def reset(self) -> None:
        """Forget the evidence latch."""
        self._last_evidence_t = None
        self._breath_ok_since = None
        self._present = False

    @property
    def present(self) -> bool:
        """The most recently computed presence state."""
        return self._present

    def update(self, motion: MotionResult, breathing_conf: float, timestamp: float) -> bool:
        """Fuse one step of evidence and return the latched presence state.

        Args:
            motion: Current output of :class:`~wifi_room_radar.detection.motion.MotionDetector`.
            breathing_conf: Confidence (0..1) of the current breathing
                estimate; pass 0.0 when no estimate is available yet.
            timestamp: Current time in seconds (same clock as previous calls;
                used to age the evidence latch).

        Returns:
            ``True`` if a person is considered present at ``timestamp``.
        """
        t = float(timestamp)
        # Breathing evidence is debounced: the confidence must hold above the
        # threshold continuously for breathing_debounce_sec.
        if breathing_conf >= self.breathing_conf_threshold:
            if self._breath_ok_since is None:
                self._breath_ok_since = t
            breath_evidence = (t - self._breath_ok_since) >= self.breathing_debounce_sec
        else:
            self._breath_ok_since = None
            breath_evidence = False

        evidence = motion.detected or breath_evidence
        if evidence:
            self._last_evidence_t = t

        if self._last_evidence_t is None:
            self._present = False
        else:
            self._present = (float(timestamp) - self._last_evidence_t) <= self.hold_sec
        return self._present
