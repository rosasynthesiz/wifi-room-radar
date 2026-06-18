"""Multi-person tracking from occupancy-map peak detections.

Takes the noisy, intermittent ``(x, y, strength)`` detections produced by
:meth:`wifi_room_radar.mapping.room_mapper.RoomMapper.detect_peaks` (typically at a
low ~2 Hz rate) and turns them into stable, identity-preserving person tracks.

Estimator
---------
Each track runs an independent constant-velocity (CV) Kalman filter with
state ``[x, y, vx, vy]``:

* **Process model**: a discrete white-noise-acceleration CV model. The
  process noise covariance ``Q`` is the standard DWNA form scaled by
  ``accel_std**2`` with ``accel_std ~ 0.5 m/s^2`` -- people change speed and
  direction gently compared to the update interval, so the filter trusts its
  straight-line prediction across short detection dropouts.
* **Measurement model**: position-only, ``z = [x, y] + v`` with
  ``meas_std ~ 0.3 m`` -- roughly one occupancy-grid cell plus the ridge-like
  range ambiguity of the 20 MHz mapper. Velocity is never measured; it is
  inferred from the position sequence by the filter.

Because ``Q`` and the state transition are built from the actual ``dt``
between calls, the tracker is rate-agnostic: it behaves the same whether it
is stepped at 2 Hz or 10 Hz, and tolerates irregular timestamps.

Data association
----------------
Detections are assigned to predicted track positions with the Hungarian
algorithm (``scipy.optimize.linear_sum_assignment``) on Euclidean distance,
gated at ``gate_m``: a detection further than the gate from every track can
never be claimed by one, which is what lets new people spawn their own tracks
instead of yanking an existing track across the room.

Track lifecycle (M-of-N style)
------------------------------
* Unmatched detections spawn *tentative* tracks (never reported).
* A tentative track is *confirmed* after ``confirm_hits`` total hits, which
  suppresses one-off ghost peaks from the mapper's side lobes.
* Any track that goes ``max_misses_sec`` of wall-clock time without a hit is
  deleted (coasting on prediction in between).
* Track ids are assigned from a forever-increasing counter, so an id is never
  reused even after its track dies.

The whole update is deterministic: identical detection/timestamp sequences
produce identical outputs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import TrackState

# Position-only measurement matrix z = H @ [x, y, vx, vy].
_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)

#: Cost placed on gated-out (track, detection) pairs so the Hungarian solver
#: only picks them when nothing else is feasible; such picks are re-rejected
#: after assignment anyway.
_GATED_COST = 1.0e6

#: Hard cap on inferred speed (m/s). Position-only measurements at ~2 Hz can
#: briefly launch the velocity estimate when a detection jumps along the
#: mapper's range ridge; people do not move at 3+ m/s indoors.
_MAX_SPEED = 3.0


class _Track:
    """Internal per-person Kalman filter and lifecycle bookkeeping."""

    __slots__ = (
        "track_id",
        "x",
        "P",
        "hits",
        "hit_score",
        "last_strength",
        "last_hit_time",
        "time",
        "confirmed",
        "confirm_time",
    )

    def __init__(
        self,
        track_id: int,
        x: float,
        y: float,
        strength: float,
        timestamp: float,
        meas_var: float,
        init_vel_std: float,
    ) -> None:
        self.track_id = track_id
        #: State [x, y, vx, vy]; velocity starts unknown-at-zero with a wide
        #: prior so the first few updates can establish it quickly.
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([meas_var, meas_var, init_vel_std**2, init_vel_std**2])
        self.hits = 1
        #: EMA of the hit/miss history in 0..1 (rate-of-recent-hits proxy).
        self.hit_score = 0.5
        self.last_strength = float(strength)
        self.last_hit_time = float(timestamp)
        #: Epoch the state estimate refers to.
        self.time = float(timestamp)
        self.confirmed = False
        self.confirm_time = float(timestamp)  # meaningful once confirmed


class MultiPersonTracker:
    """Streaming multi-target tracker over occupancy-map detections.

    Args:
        max_people: Maximum number of confirmed tracks ever reported (and a
            soft cap on internal track count -- at most ``2 * max_people``
            live tracks are kept so stale tentatives cannot accumulate).
        gate_m: Association gate, metres: a detection may only update a track
            whose *predicted* position lies within this distance.
        confirm_hits: Total detections required before a tentative track is
            confirmed and starts being reported.
        max_misses_sec: A track is deleted after this much time without a
            matched detection. At a 2 Hz update rate the default 2.5 s allows
            about five consecutive missed frames before giving up.
        room_size: Optional ``(width_m, depth_m)``. When given, *reported*
            positions are clamped into ``[0, width] x [0, depth]`` (the
            internal filter state is left untouched so velocity estimation is
            not biased at the walls).
        accel_std: White-noise acceleration density for the CV process model,
            m/s^2. ~0.5 suits walking humans.
        meas_std: 1-sigma position measurement noise, metres.

    All tuning is done in physical units, so the tracker behaves consistently
    across update rates.
    """

    def __init__(
        self,
        max_people: int = 4,
        gate_m: float = 1.2,
        confirm_hits: int = 3,
        max_misses_sec: float = 2.5,
        room_size: Optional[tuple[float, float]] = None,
        accel_std: float = 0.5,
        meas_std: float = 0.3,
    ) -> None:
        self.max_people = int(max_people)
        self.gate_m = float(gate_m)
        self.confirm_hits = int(confirm_hits)
        self.max_misses_sec = float(max_misses_sec)
        self.room_size = room_size
        self.accel_std = float(accel_std)
        self.meas_var = float(meas_std) ** 2
        self._R = self.meas_var * np.eye(2)
        self._init_vel_std = 1.0  # m/s prior std on unknown initial velocity

        self._tracks: list[_Track] = []
        self._next_id = 1
        #: EMA coefficient for each track's hit/miss score.
        self._hit_beta = 0.3

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: Sequence[tuple[float, float, float]],
        timestamp: float,
    ) -> list[TrackState]:
        """Step the tracker to ``timestamp`` with this frame's detections.

        Args:
            detections: ``(x_m, y_m, strength)`` tuples from
                ``RoomMapper.detect_peaks`` (may be empty).
            timestamp: Seconds (same clock as previous calls); must be
                non-decreasing for meaningful prediction. A repeated or
                earlier timestamp is treated as ``dt = 0`` (no prediction).

        Returns:
            Confirmed tracks only, as :class:`~wifi_room_radar.types.TrackState`,
            at most ``max_people`` of them, ordered by track id (oldest
            first) for stable presentation.
        """
        timestamp = float(timestamp)

        # 1. Predict every track forward to this timestamp.
        for track in self._tracks:
            self._predict(track, timestamp)

        # 2. Gated Hungarian association of detections to predicted tracks.
        det_xy = np.array([[d[0], d[1]] for d in detections], dtype=float).reshape(-1, 2)
        det_strength = [float(d[2]) for d in detections]
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if self._tracks and det_xy.shape[0]:
            predicted = np.array([t.x[:2] for t in self._tracks])  # [n_t, 2]
            dist = np.linalg.norm(
                predicted[:, None, :] - det_xy[None, :, :], axis=2
            )  # [n_t, n_d]
            cost = np.where(dist <= self.gate_m, dist, _GATED_COST)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if dist[r, c] <= self.gate_m:
                    self._correct(self._tracks[r], det_xy[c], det_strength[c], timestamp)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        # 3. Misses: decay the hit score of unmatched tracks.
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                track.hit_score *= 1.0 - self._hit_beta

        # 4. Delete tracks that have coasted too long without evidence.
        self._tracks = [
            t
            for t in self._tracks
            if timestamp - t.last_hit_time <= self.max_misses_sec
        ]

        # 5. Spawn tentative tracks from unmatched detections (strongest
        #    first, bounded so clutter cannot grow the track list forever).
        spawn_order = sorted(
            (c for c in range(det_xy.shape[0]) if c not in matched_dets),
            key=lambda c: (-det_strength[c], c),
        )
        for c in spawn_order:
            if len(self._tracks) >= 2 * self.max_people:
                break
            self._tracks.append(
                _Track(
                    self._next_id,
                    float(det_xy[c, 0]),
                    float(det_xy[c, 1]),
                    det_strength[c],
                    timestamp,
                    self.meas_var,
                    self._init_vel_std,
                )
            )
            self._next_id += 1

        # 6. Confirm tentative tracks that have collected enough hits.
        for track in self._tracks:
            if not track.confirmed and track.hits >= self.confirm_hits:
                track.confirmed = True
                track.confirm_time = timestamp

        # 7. Report confirmed tracks (at most max_people, by confidence).
        confirmed = [t for t in self._tracks if t.confirmed]
        if len(confirmed) > self.max_people:
            confirmed = sorted(
                confirmed, key=lambda t: (-self._confidence(t), t.track_id)
            )[: self.max_people]
        confirmed.sort(key=lambda t: t.track_id)
        return [self._to_state(t, timestamp) for t in confirmed]

    @property
    def n_tracks(self) -> int:
        """Number of live internal tracks (tentative + confirmed)."""
        return len(self._tracks)

    # ------------------------------------------------------------------ #
    # Kalman filter internals                                            #
    # ------------------------------------------------------------------ #

    def _predict(self, track: _Track, timestamp: float) -> None:
        """Propagate a track's CV state and covariance to ``timestamp``."""
        dt = timestamp - track.time
        track.time = timestamp
        if dt <= 0.0:
            return
        f = np.eye(4)
        f[0, 2] = dt
        f[1, 3] = dt
        # Discrete white-noise-acceleration Q (per-axis blocks interleaved
        # into the [x, y, vx, vy] ordering).
        q2 = self.accel_std**2
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        q = q2 * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ]
        )
        track.x = f @ track.x
        track.P = f @ track.P @ f.T + q

    def _correct(
        self, track: _Track, z: np.ndarray, strength: float, timestamp: float
    ) -> None:
        """Standard KF measurement update with a position observation."""
        innovation = z - _H @ track.x
        s = _H @ track.P @ _H.T + self._R  # 2x2 innovation covariance
        k = track.P @ _H.T @ np.linalg.inv(s)  # 4x2 Kalman gain
        track.x = track.x + k @ innovation
        # Joseph-free covariance update is fine at this conditioning.
        track.P = (np.eye(4) - k @ _H) @ track.P

        # Physical sanity: cap the speed so a detection jumping along the
        # mapper's range ridge cannot launch the track across the room.
        speed = float(np.hypot(track.x[2], track.x[3]))
        if speed > _MAX_SPEED:
            track.x[2:] *= _MAX_SPEED / speed

        track.hits += 1
        track.hit_score += self._hit_beta * (1.0 - track.hit_score)
        track.last_strength = float(strength)
        track.last_hit_time = timestamp

    # ------------------------------------------------------------------ #
    # reporting                                                          #
    # ------------------------------------------------------------------ #

    def _confidence(self, track: _Track) -> float:
        """Blend recent hit ratio with the latest detection strength."""
        return float(
            np.clip(0.65 * track.hit_score + 0.35 * track.last_strength, 0.0, 1.0)
        )

    def _to_state(self, track: _Track, timestamp: float) -> TrackState:
        """Convert an internal track to the public, JSON-friendly TrackState."""
        x, y = float(track.x[0]), float(track.x[1])
        if self.room_size is not None:
            width, depth = self.room_size
            x = min(max(x, 0.0), float(width))
            y = min(max(y, 0.0), float(depth))
        return TrackState(
            track_id=track.track_id,
            x=x,
            y=y,
            vx=float(track.x[2]),
            vy=float(track.x[3]),
            confidence=self._confidence(track),
            age=max(0.0, timestamp - track.confirm_time),
        )
