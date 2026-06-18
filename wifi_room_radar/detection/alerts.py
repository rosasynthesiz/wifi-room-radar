"""Safety alerts: fall detection and breathing-cessation (apnea-style) alarm.

Both detectors are built from signals the pipeline already computes — no new
DSP, just temporal logic over them — and both are deliberately conservative:
an alert that cries wolf gets disabled by its users.

Fall signature (radar literature's classic two-phase pattern):
    1. a brief, strong high-Doppler burst (the body accelerating downward
       produces tens of Hz of bistatic Doppler for a fraction of a second),
    2. followed almost immediately by *stillness* (the person is on the
       floor) — which is exactly what separates a fall from sitting down
       fast (motion continues) or from walking (no sustained stillness).

Breathing-cessation signature: a person whose breathing line was credibly
tracked stops producing it while remaining present and motionless. Gross
motion legitimately destroys the breathing estimate, so motion anywhere in
the window vetoes the alarm rather than firing it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..types import VitalSign

__all__ = ["FallDetector", "ApneaDetector", "AlertManager"]


@dataclass
class _ActiveAlert:
    type: str
    message: str
    since: float

    def to_dict(self) -> dict:
        return {"type": self.type, "message": self.message, "since": self.since}


class FallDetector:
    """High-Doppler burst followed by sustained stillness.

    Args:
        burst_band_hz: |Doppler| at/above this is "fall band" energy. The
            default sits above any walking Doppler on purpose: at 5 GHz a
            0.9 m/s walker tops out near 2v/lambda ~ 31 Hz, while a falling
            body's ~2 m/s of path-length rate produces ~70 Hz, so a 35 Hz
            band edge separates falls from gait physically — walking then
            cannot trip the detector no matter how abruptly it stops
            (sitting down, pausing at a waypoint).
        burst_excess_db: How far above its adaptive floor the fall-band
            energy must spike to count as a candidate impact.
        still_level: Motion level below which the person counts as still.
        still_within_sec: The stillness must begin within this long of the
            burst.
        still_for_sec: ... and persist this long before the alert fires.
        clear_motion: Motion level that clears an active alert (they moved /
            got up).
        clear_after_sec: Active alerts auto-expire after this long.
        warmup_sec: Bursts within this long of the first update are ignored:
            the pipeline's background EMA and spectral floors are still
            converging, and the settling transient reads as a high-Doppler
            burst followed by stillness — a textbook phantom fall.
    """

    warmup_sec: float = 10.0

    def __init__(
        self,
        burst_band_hz: float = 35.0,
        burst_excess_db: float = 10.0,
        still_level: float = 0.12,
        still_within_sec: float = 2.5,
        still_for_sec: float = 1.2,
        clear_motion: float = 0.3,
        clear_after_sec: float = 60.0,
    ) -> None:
        self.burst_band_hz = float(burst_band_hz)
        self.burst_excess_db = float(burst_excess_db)
        self.still_level = float(still_level)
        self.still_within_sec = float(still_within_sec)
        self.still_for_sec = float(still_for_sec)
        self.clear_motion = float(clear_motion)
        self.clear_after_sec = float(clear_after_sec)

        self._floor_db: Optional[float] = None  # EMA of quiescent band energy
        self._burst_t: Optional[float] = None
        self._still_since: Optional[float] = None
        self._alert: Optional[_ActiveAlert] = None
        self._first_t: Optional[float] = None

    def update(
        self,
        timestamp: float,
        doppler_db: np.ndarray | None,
        doppler_freqs: np.ndarray | None,
        motion_level: float,
    ) -> Optional[_ActiveAlert]:
        """Advance one tick; returns the active alert (or None)."""
        t = float(timestamp)
        if self._first_t is None:
            self._first_t = t
        in_warmup = (t - self._first_t) < self.warmup_sec

        # --- clear an active alert -----------------------------------------
        if self._alert is not None:
            if motion_level >= self.clear_motion or t - self._alert.since > self.clear_after_sec:
                self._alert = None
            else:
                return self._alert

        # --- fall-band energy vs adaptive floor -----------------------------
        band_db = self._band_energy_db(doppler_db, doppler_freqs)
        if band_db is not None:
            if self._floor_db is None:
                self._floor_db = band_db
            burst = band_db > self._floor_db + self.burst_excess_db
            if not burst:
                # Quiescent: slow EMA floor tracking (frozen during bursts so
                # the burst cannot raise its own threshold).
                self._floor_db += 0.02 * (band_db - self._floor_db)
            elif in_warmup:
                # Converging floors: track the transient quickly instead of
                # arming on it (see warmup_sec).
                self._floor_db += 0.2 * (band_db - self._floor_db)
            else:
                self._burst_t = t

        # --- burst followed by stillness ------------------------------------
        if self._burst_t is not None:
            if t - self._burst_t > self.still_within_sec + self.still_for_sec + 1.0:
                self._burst_t = None  # window expired without a confirmed fall
                self._still_since = None
            elif motion_level <= self.still_level:
                if self._still_since is None:
                    self._still_since = t
                started_in_time = self._still_since - self._burst_t <= self.still_within_sec
                if started_in_time and t - self._still_since >= self.still_for_sec:
                    self._alert = _ActiveAlert(
                        type="fall",
                        message="High-impact motion followed by stillness — possible fall",
                        since=self._burst_t,
                    )
                    self._burst_t = None
                    self._still_since = None
            else:
                self._still_since = None
        return self._alert

    def _band_energy_db(
        self, doppler_db: np.ndarray | None, freqs: np.ndarray | None
    ) -> Optional[float]:
        if doppler_db is None or freqs is None or len(doppler_db) == 0:
            return None
        col = np.asarray(doppler_db, dtype=float)
        f = np.asarray(freqs, dtype=float)
        band = np.abs(f) >= self.burst_band_hz
        if not np.any(band):
            return None
        # dB columns: mean in linear power, back to dB (a single strong bin
        # should dominate the way it does physically).
        return float(10.0 * np.log10(np.mean(10.0 ** (col[band] / 10.0)) + 1e-30))


class ApneaDetector:
    """Credible breathing that stops while the person stays present and still.

    Args:
        healthy_conf: Breathing confidence that counts as a healthy track.
        lost_conf: Confidence below which breathing counts as lost.
        no_breath_sec: How long breathing must stay lost (with presence and
            stillness throughout) before the alarm fires.
        recent_healthy_sec: The alarm only arms if breathing was healthy
            within this window (prevents firing on someone never tracked).
        still_level: Motion above this vetoes/clears the alarm — gross
            motion legitimately hides breathing.
    """

    def __init__(
        self,
        healthy_conf: float = 0.5,
        lost_conf: float = 0.2,
        no_breath_sec: float = 12.0,
        recent_healthy_sec: float = 120.0,
        still_level: float = 0.15,
    ) -> None:
        self.healthy_conf = float(healthy_conf)
        self.lost_conf = float(lost_conf)
        self.no_breath_sec = float(no_breath_sec)
        self.recent_healthy_sec = float(recent_healthy_sec)
        self.still_level = float(still_level)

        self._last_healthy_t: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._alert: Optional[_ActiveAlert] = None

    #: Tail/head RMS ratio of the display waveform below which breathing
    #: counts as stopped. The estimator's spectral confidence lags a
    #: cessation by tens of seconds (the Welch window still contains the old
    #: breathing), but the ~15 s display waveform is time-ordered: a few
    #: seconds after the chest stops, its last third is residual noise while
    #: its first third still holds full-swing breathing, so this ratio
    #: collapses long before the confidence does.
    waveform_tail_ratio: float = 0.25

    def update(
        self,
        timestamp: float,
        presence: bool,
        motion_level: float,
        breathing: Optional[VitalSign],
    ) -> Optional[_ActiveAlert]:
        """Advance one tick; returns the active alert (or None).

        Note ``presence`` is deliberately NOT required for arming: for a
        motionless person presence is itself breathing-driven, so requiring
        it would disarm the alarm at exactly the moment it should fire.
        Recent healthy breathing + no motion since is the arming condition —
        the person demonstrably was there and cannot have left unseen.
        """
        t = float(timestamp)
        conf = float(breathing.confidence) if breathing is not None else 0.0

        if conf >= self.healthy_conf and not self._waveform_collapsed(breathing):
            self._last_healthy_t = t

        moving = float(motion_level) > self.still_level
        recovered = conf >= 0.4 and not self._waveform_collapsed(breathing)
        if self._alert is not None:
            if recovered or moving:
                self._alert = None
                self._lost_since = None
            return self._alert

        armed = (
            not moving
            and self._last_healthy_t is not None
            and t - self._last_healthy_t <= self.recent_healthy_sec
        )
        lost = conf < self.lost_conf or self._waveform_collapsed(breathing)
        if not armed or not lost:
            self._lost_since = None
            return None
        if self._lost_since is None:
            self._lost_since = t
        if t - self._lost_since >= self.no_breath_sec:
            self._alert = _ActiveAlert(
                type="breathing_stopped",
                message="Tracked breathing stopped while the person remains present and still",
                since=self._lost_since,
            )
        return self._alert

    def _waveform_collapsed(self, breathing: Optional[VitalSign]) -> bool:
        """True when the waveform's recent tail has died relative to its head."""
        if breathing is None or len(breathing.waveform) < 30:
            return False
        w = np.asarray(breathing.waveform, dtype=float)
        third = w.size // 3
        head = float(np.sqrt(np.mean(w[:third] ** 2)))
        tail = float(np.sqrt(np.mean(w[-third:] ** 2)))
        return tail < self.waveform_tail_ratio * (head + 1e-12)


class AlertManager:
    """Fuses all alert detectors into the SensingState ``alerts`` list."""

    def __init__(self) -> None:
        self.fall = FallDetector()
        self.apnea = ApneaDetector()

    def update(
        self,
        timestamp: float,
        doppler_db: np.ndarray | None,
        doppler_freqs: np.ndarray | None,
        motion_level: float,
        presence: bool,
        breathing: Optional[VitalSign],
    ) -> list[dict]:
        """Run every detector; return active alerts as JSON-ready dicts."""
        active = [
            self.fall.update(timestamp, doppler_db, doppler_freqs, motion_level),
            self.apnea.update(timestamp, presence, motion_level, breathing),
        ]
        return [a.to_dict() for a in active if a is not None]
