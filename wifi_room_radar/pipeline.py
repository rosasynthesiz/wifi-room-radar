"""End-to-end streaming sensing pipeline.

:class:`SensingPipeline` wires every wifi_room_radar component into one
frame-driven loop. It consumes :class:`~wifi_room_radar.types.CSIFrame` objects
from any :class:`~wifi_room_radar.capture.base.CSISource` and keeps a thread-safe
"latest" :class:`~wifi_room_radar.types.SensingState` that the web server polls.

Signal flow per frame (all rates derived from ``source.radio.sample_rate``):

1. **Background subtraction** — :class:`BackgroundSubtractor` phase-aligns
   the raw frame to the running static-channel estimate and returns the
   dynamic (moving-reflector) residual.
2. **Motion** — the Doppler-projected sample (stage 3) is high-passed at
   ~2 Hz and its power fed to the adaptive
   :class:`~wifi_room_radar.detection.motion.MotionDetector`, so gross body motion
   registers but sub-Hz breathing micro-motion does not.
3. **Micro-Doppler** — the flattened dynamic CSI is projected onto a
   slowly-tracked dominant eigenvector (warm-started power iteration over the
   last ``stft_size`` frames, refreshed every ``stft_hop`` frames; the warm
   start keeps the projection phase continuous between refreshes). The
   resulting complex series feeds :class:`StreamingSTFT`, whose newest column
   becomes the dashboard Doppler display (~``sample_rate / stft_hop``
   columns/sec).
4. **Vitals** — the cross-antenna conjugate ratio (:func:`csi_ratio`)
   cancels CFO/STO; an amplitude-weighted complex mean across (rx pairs,
   subcarriers) collapses it to one robust scalar per frame, whose
   streaming-unwrapped phase is the chest-motion series. The full-rate
   series feeds the experimental :class:`SubvocalDetector` (its 15-80 Hz
   band needs the full frame rate); a :class:`Decimator` takes it down to
   ``vitals_fs`` for the :class:`BreathingEstimator` and
   :class:`HeartbeatEstimator` (which receives the current breathing
   fundamental for harmonic suppression and the motion level for derating).

   Deliberately *not* applied here: :func:`sanitize_phase`. It removes the
   common phase offset across subcarriers per frame — which is exactly the
   component breathing lives in. The conjugate ratio already cancels the
   CFO/STO nuisance terms, so only a non-finite guard is needed.
5. **Mapping / tracking** at ``map_update_rate`` — the last
   ``map_window_sec`` of dynamic CSI ``[T, n_rx, n_sub]`` goes through
   :class:`RoomMapper` (Bartlett matched field), peaks through
   :class:`MultiPersonTracker`.
6. **Presence** — motion OR credible breathing, latched by
   :class:`PresenceDetector`.
7. At ``update_rate`` a :class:`SensingState` is assembled from plain Python
   floats/lists (the JSON contract) and stored under a lock.

Performance: all hot-path work is O(n_rx * n_sub) numpy ops on small arrays;
the only heavier jobs (eigenvector refresh, STFT column, Welch PSDs, Bartlett
matmul) run at their own low cadences. On a typical laptop the pipeline
free-runs at well over 1000 frames/sec — comfortably faster than the 200 Hz
realtime stream it has to keep up with.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy import signal as _signal

from .capture.base import CSISource
from .config import PipelineConfig
from .detection.activity import ActivityClassifier
from .detection.alerts import AlertManager
from .detection.motion import MotionDetector
from .detection.presence import PresenceDetector
from .mapping.room_mapper import RoomMapper
from .processing.filters import Decimator
from .processing.preprocessing import BackgroundSubtractor, CSIBuffer, csi_ratio
from .processing.spectrogram import StreamingSTFT
from .tracking.tracker import MultiPersonTracker
from .types import CSIFrame, SensingState, VitalSign
from .vitals.breathing import BreathingEstimator
from .vitals.heartbeat import HeartbeatEstimator
from .vitals.subvocal import SubvocalDetector

__all__ = ["SensingPipeline"]

#: Minimum breathing confidence before the estimate is offered to the
#: heartbeat estimator as a harmonic-suppression hint (a wrong fundamental
#: would notch the wrong bins).
_BREATH_HINT_CONF = 0.2

#: Power-iteration sweeps per eigenvector refresh. Warm-started from the
#: previous vector, two sweeps track the slowly rotating dominant subspace.
_POWER_ITERS = 2

#: High-pass corner (Hz) applied to the Doppler-projected series before it
#: feeds the motion detector. Gross body motion (walking, gesturing) has
#: Doppler content of several to tens of Hz, while breathing modulates the
#: channel below ~1 Hz (plus weak low harmonics). Without this split a
#: motionless breathing person saturates the "motion" level — which both
#: reads wrong on the dashboard and wrongly derates the vitals confidence.
_MOTION_HP_HZ = 2.0

#: Seconds a per-track vitals extractor survives after its track disappears
#: (bridges brief tracker dropouts without losing the vitals buffers).
_TRACK_VITALS_GRACE_SEC = 5.0


class _TrackVitals:
    """Spatially-filtered vital-sign state for one confirmed track.

    The pipeline projects each frame's dynamic CSI onto the matched-filter
    steering vector of the track's grid cell (on the receiver node that sees
    the track best), giving a scalar complex series dominated by motion *at
    that position*; its unwrapped phase feeds this track's own breathing and
    heartbeat estimators. This is what makes vitals per-person: each track's
    spatial filter attenuates the other occupants by the array's beam
    pattern, instead of every chest competing in one room-wide series.

    Note: when the track (hence cell) moves, the projection vector changes
    and the phase series picks up a step transient — harmless, because a
    moving person's vitals are motion-derated anyway; for the still person
    the cell is stable and the series is clean.
    """

    def __init__(self, fs: float, vitals_fs: float, cfg: PipelineConfig) -> None:
        self.dec = Decimator(fs, vitals_fs)
        self.breath = BreathingEstimator(vitals_fs, band=cfg.breath_band)
        self.heart = HeartbeatEstimator(vitals_fs, band=cfg.heart_band)
        self.node: int = 0
        self.vec: Optional[np.ndarray] = None  # conj steering row [F_node]
        self.breathing_vs: Optional[VitalSign] = None
        self.heart_vs: Optional[VitalSign] = None
        self.last_seen: float = 0.0
        self.speed: float = 0.0
        # Streaming phase unwrap state.
        self._z_prev: Optional[complex] = None
        self._phi_last = 0.0
        self._phi_acc: list[float] = []

    def push(self, z: complex) -> None:
        """Accumulate one projected sample into the unwrapped phase series."""
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            z = self._z_prev if self._z_prev is not None else complex(1e-12, 0.0)
        if self._z_prev is None:
            self._phi_last = float(np.angle(z)) if z != 0 else 0.0
        else:
            self._phi_last += float(np.angle(z * np.conj(self._z_prev)))
        self._z_prev = z
        self._phi_acc.append(self._phi_last)

    def drain(self) -> np.ndarray:
        """Return and clear the accumulated phase samples."""
        out = np.array(self._phi_acc, dtype=np.float64)
        self._phi_acc.clear()
        return out


class SensingPipeline:
    """Fuse a CSI stream into live sensing states.

    Satisfies the duck-typed provider contract of :mod:`wifi_room_radar.server`:
    ``latest_state()`` and ``info``.

    Args:
        source: Any CSI source (simulator, replay, hardware).
        cfg: Pipeline tuning knobs.
        room_size: ``(width_m, depth_m)`` extent of the sensed room; must
            match the geometry in ``source.radio`` for mapping to make sense.
    """

    def __init__(
        self,
        source: CSISource,
        cfg: PipelineConfig,
        room_size: tuple[float, float],
    ) -> None:
        self.source = source
        self.cfg = cfg
        self.room_size = (float(room_size[0]), float(room_size[1]))

        radio = source.radio
        fs = float(radio.sample_rate)
        if fs <= 0:
            raise ValueError(f"source sample_rate must be positive, got {fs}")
        self.fs = fs
        self._n_nodes = int(radio.n_nodes)
        self._per_rx = int(radio.n_rx)  # elements per node
        self._total_rx = int(radio.total_rx)  # stacked rows in a frame
        self._n_sub = int(radio.n_subcarriers)
        n_feat = self._total_rx * self._n_sub
        self._node_feat = self._per_rx * self._n_sub  # features per node

        # --- stage 1+2: background + motion --------------------------------
        self._background = BackgroundSubtractor(
            alpha=cfg.background_alpha, n_nodes=self._n_nodes
        )
        self._motion = MotionDetector(fs=fs, hysteresis=cfg.motion_hysteresis)
        self._presence = PresenceDetector(hold_sec=cfg.presence_hold_sec)
        self._activity = ActivityClassifier()
        self._alerts = AlertManager()
        # Streaming high-pass separating gross motion from vitals-band
        # micro-motion (see _MOTION_HP_HZ). Applied per (rx, subcarrier)
        # feature of the dynamic CSI — real and imaginary parts as separate
        # real channels — and the energies averaged across all features.
        # Averaging matters as much as the filtering: a single projected
        # sample's power is exponentially distributed (its 10th-percentile
        # noise floor sits far below its mean, so an adaptive-floor detector
        # fires on pure noise), while the mean over 2 * n_rx * n_sub channels
        # concentrates tightly around the true noise power.
        if fs > 4.0 * _MOTION_HP_HZ:
            self._mot_sos = _signal.butter(
                2, _MOTION_HP_HZ, btype="highpass", fs=fs, output="sos"
            )
            n_sections = self._mot_sos.shape[0]
            self._mot_zi = np.zeros((n_sections, 2 * n_feat, 2))
        else:
            self._mot_sos = None
            self._mot_zi = None

        # --- stage 3: micro-Doppler ----------------------------------------
        self._stft = StreamingSTFT(cfg.stft_size, cfg.stft_hop, fs)
        self._doppler_freqs: list[float] = [float(f) for f in self._stft.freqs]
        self._doppler_col: Optional[np.ndarray] = None
        self._evec = np.full(n_feat, 1.0 / np.sqrt(n_feat), dtype=np.complex128)
        self._dopp_acc: list[complex] = []

        # Shared rolling window of flattened dynamic CSI (Doppler PCA window
        # and mapping window are both sliced from it).
        self._map_frames = max(2, int(round(cfg.map_window_sec * fs)))
        self._buf = CSIBuffer(capacity=max(int(cfg.stft_size), self._map_frames))

        # --- stage 4: vitals -------------------------------------------------
        # Decimator needs an integer factor; snap vitals_fs to the nearest
        # exact divisor of the frame rate (e.g. 200/20 -> q=10, fs_out=20).
        q = max(1, int(round(fs / cfg.vitals_fs)))
        self._vitals_fs = fs / q
        self._vitals_dec = Decimator(fs, self._vitals_fs)
        self._breath = BreathingEstimator(self._vitals_fs, band=cfg.breath_band)
        self._heart = HeartbeatEstimator(self._vitals_fs, band=cfg.heart_band)
        # Subvocal needs its band below Nyquist of the *full* frame rate;
        # disabled (None) on sources too slow to observe it.
        sub_lo, sub_hi = cfg.subvocal_band
        self._subvocal: Optional[SubvocalDetector] = (
            SubvocalDetector(fs, band=cfg.subvocal_band)
            if 0.0 < sub_lo < sub_hi < 0.5 * fs
            else None
        )
        self._phi_acc: list[float] = []
        self._phi_last = 0.0
        self._z_prev: Optional[complex] = None

        # --- stage 5: mapping + tracking ------------------------------------
        self._mapper = RoomMapper(
            radio,
            room_width=self.room_size[0],
            room_depth=self.room_size[1],
            grid_resolution=cfg.grid_resolution,
        )
        self._tracker = MultiPersonTracker(
            max_people=cfg.max_people, room_size=self.room_size
        )

        # --- cadence ----------------------------------------------------------
        self._frame_i = 0
        self._publish_every = max(1, int(round(fs / cfg.update_rate)))
        self._map_every = max(1, int(round(fs / cfg.map_update_rate)))
        self._hop = max(1, int(cfg.stft_hop))

        # --- latest results ---------------------------------------------------
        self._breathing_vs = None
        self._heart_vs = None
        self._sub_state = None
        self._tracks: list = []
        self._track_vitals: dict[int, _TrackVitals] = {}
        self._activity_label = "idle"
        self._alert_list: list[dict] = []
        self._doppler_freqs_arr = np.asarray(self._doppler_freqs, dtype=float)

        # --- published state + stats -----------------------------------------
        self._lock = threading.Lock()
        self._latest: Optional[SensingState] = None
        self._frames_done = 0
        self._wall_start: Optional[float] = None

    # ------------------------------------------------------------------ #
    # provider contract                                                  #
    # ------------------------------------------------------------------ #

    def latest_state(self) -> Optional[SensingState]:
        """Most recently published state (thread-safe), or None."""
        with self._lock:
            return self._latest

    def apply_profile(self, profile: dict) -> None:
        """Preload adaptive noise floors from a calibration profile.

        Profiles are produced by ``scripts/calibrate.py`` against an empty
        room; loading one makes cold starts honest (without it, the first
        seconds of data set the floors — mis-calibrated if someone is
        already in the room when the pipeline starts). Fields that do not
        match this pipeline's configuration are ignored individually.
        """
        if profile.get("motion_floor") is not None:
            self._motion._floor = float(profile["motion_floor"])
            self._motion._median = float(profile.get("motion_median") or 0.0)
        floors = profile.get("mapper_noise_floor")
        if floors is not None and len(floors) == self._mapper.n_nodes:
            self._mapper._noise_floor = np.asarray(floors, dtype=float)

    @property
    def info(self) -> dict:
        """Source metadata + room geometry + package version + live stats."""
        from . import __version__  # local import avoids a circular import

        radio = self.source.radio
        info = dict(self.source.info)
        info.update(
            {
                "room_size": list(self.room_size),
                "tx_pos": [float(v) for v in radio.tx_pos],
                "rx_positions": radio.rx_positions().tolist(),
                "version": __version__,
            }
        )
        info.update(self.stats())
        return info

    def stats(self) -> dict:
        """Processed-frame counter and measured wall-clock throughput."""
        with self._lock:
            frames = self._frames_done
            start = self._wall_start
        elapsed = (time.perf_counter() - start) if start is not None else 0.0
        fps = frames / elapsed if elapsed > 0 else 0.0
        return {"frames_processed": frames, "measured_fps": float(fps)}

    # ------------------------------------------------------------------ #
    # driving                                                            #
    # ------------------------------------------------------------------ #

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Consume ``source.frames()`` until exhausted or ``stop_event`` set.

        Intended to run in a daemon thread next to the dashboard server.
        """
        try:
            for frame in self.source.frames():
                if stop_event is not None and stop_event.is_set():
                    break
                self.step_frame(frame)
        finally:
            self.source.close()

    def step_frame(self, frame: CSIFrame) -> Optional[SensingState]:
        """Process one frame; return the new state on publish ticks else None."""
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        cfg = self.cfg
        t = float(frame.timestamp)
        csi = np.asarray(frame.csi)

        # ---- 1. background subtraction --------------------------------------
        dynamic, _bg = self._background.update(csi)
        dyn = dynamic[:, 0, :]  # [n_rx, n_sub] (co-located TX chains are identical)

        # ---- 2. doppler projection -------------------------------------------
        flat = dyn.ravel()
        self._buf.push(t, flat)
        proj = complex(np.vdot(self._evec, flat))
        self._dopp_acc.append(proj)

        # ---- 3. motion --------------------------------------------------------
        # Gross motion only: per-feature high-pass at _MOTION_HP_HZ, then the
        # power averaged over all features, so sub-Hz breathing modulation
        # does not register and the noise floor stays tightly concentrated.
        if self._mot_sos is not None:
            xs = np.concatenate([flat.real, flat.imag])  # [2 * n_feat]
            y, self._mot_zi = _signal.sosfilt(
                self._mot_sos, xs[:, None], axis=1, zi=self._mot_zi
            )
            energy = float(np.mean(y[:, 0] ** 2))
        else:
            energy = BackgroundSubtractor.motion_energy(dynamic)
        motion = self._motion.update(energy, t)

        # ---- 4. vitals scalar -------------------------------------------------
        self._push_vitals_sample(csi)

        # ---- 4b. per-track spatial projections (cheap dot products) ----------
        # Each confirmed track has a matched-filter vector for its grid cell
        # on its best-viewing node; projecting the dynamic CSI onto it gives
        # that person's own chest-motion series (see _TrackVitals).
        for tv in self._track_vitals.values():
            if tv.vec is not None:
                sl = slice(tv.node * self._per_rx, (tv.node + 1) * self._per_rx)
                tv.push(complex(np.dot(tv.vec, dyn[sl].ravel())))

        self._frame_i += 1

        # Eigenvector refresh at the STFT hop cadence.
        if self._frame_i % self._hop == 0 and len(self._buf) >= 16:
            self._refresh_eigenvector()

        # ---- 5. mapping / tracking -------------------------------------------
        if self._frame_i % self._map_every == 0 and len(self._buf) >= self._map_frames:
            _, X = self._buf.array()
            window = X[-self._map_frames :].reshape(-1, self._total_rx, self._n_sub)
            self._mapper.update(window)
            # Mesh maps are point-like but a mover's EMA trail leaves a lobe
            # behind them that clears the relative gate; a wider separation
            # absorbs it (single-link maps keep the bearing gate instead).
            peaks = self._mapper.detect_peaks(
                cfg.max_people,
                min_separation=1.3 if self._n_nodes > 1 else 0.8,
            )
            self._tracks = self._tracker.update(peaks, t)
            self._refresh_track_vitals(window, t)

        # ---- 6. presence ------------------------------------------------------
        b_conf = self._breathing_vs.confidence if self._breathing_vs is not None else 0.0
        present = self._presence.update(motion, b_conf, t)

        with self._lock:
            self._frames_done += 1

        # ---- 7. publish ---------------------------------------------------------
        if self._frame_i % self._publish_every != 0:
            return None

        self._process_vitals_chunk(t, motion.level)
        self._process_track_vitals(t)
        self._process_doppler_chunk()

        # Activity + safety alerts (room level). Alerts use the most
        # confident breathing estimate available — a per-track one when a
        # spatial filter has locked on, else the room-wide series.
        self._activity_label = self._activity.update(
            t, motion.level, present, self._doppler_col, self._doppler_freqs_arr
        )
        breath_for_alerts = self._breathing_vs
        for tv in self._track_vitals.values():
            if tv.breathing_vs is not None and (
                breath_for_alerts is None
                or tv.breathing_vs.confidence > breath_for_alerts.confidence
            ):
                breath_for_alerts = tv.breathing_vs
        self._alert_list = self._alerts.update(
            t,
            self._doppler_col,
            self._doppler_freqs_arr,
            motion.level,
            present,
            breath_for_alerts,
        )

        # Attach per-track vitals to the published track states.
        tracks_out = []
        for tr in self._tracks:
            tv = self._track_vitals.get(tr.track_id)
            tracks_out.append(
                replace(
                    tr,
                    breathing=tv.breathing_vs if tv is not None else None,
                    heartbeat=tv.heart_vs if tv is not None else None,
                )
            )

        state = SensingState(
            timestamp=t,
            presence=bool(present),
            motion_level=float(motion.level),
            motion_detected=bool(motion.detected),
            doppler_freqs=self._doppler_freqs if self._doppler_col is not None else [],
            doppler_column=(
                [float(v) for v in self._doppler_col]
                if self._doppler_col is not None
                else []
            ),
            room_size=self.room_size,
            occupancy_grid=self._mapper.grid.tolist(),
            tracks=tracks_out,
            breathing=self._breathing_vs,
            heartbeat=self._heart_vs,
            subvocal=self._sub_state,
            activity=self._activity_label,
            alerts=list(self._alert_list),
            ground_truth=frame.ground_truth,
        )
        with self._lock:
            self._latest = state
        return state

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _refresh_track_vitals(self, window: np.ndarray, t: float) -> None:
        """Re-aim every confirmed track's spatial filter; prune dead tracks.

        Runs at the map cadence: for each confirmed track, look up its grid
        cell, pick the receiver node whose matched-filter projection carries
        the most energy over the current window (that node sees this person
        best), and store the projection vector used per-frame in
        :meth:`step_frame`. Extractors of vanished tracks are kept for a
        grace period to bridge tracker dropouts, then dropped.
        """
        n_frames = window.shape[0]
        for tr in self._tracks:
            tv = self._track_vitals.get(tr.track_id)
            if tv is None:
                tv = _TrackVitals(self.fs, self._vitals_fs, self.cfg)
                self._track_vitals[tr.track_id] = tv
            cell = self._mapper.cell_index(tr.x, tr.y)
            best_node, best_score = 0, -1.0
            for node in range(self._n_nodes):
                vec = self._mapper.matched_filter(node, cell)
                node_flat = window[
                    :, node * self._per_rx : (node + 1) * self._per_rx, :
                ].reshape(n_frames, -1)
                cell_energy = float(np.mean(np.abs(node_flat @ vec) ** 2))
                total_energy = float(np.mean(np.abs(node_flat) ** 2)) + 1e-30
                # Signal-to-interference, not raw energy: the node where this
                # track's cell holds the largest SHARE of the dynamic power
                # isolates this person best — raw energy would pick whichever
                # node hears the loudest *other* mover.
                score = cell_energy / total_energy
                if score > best_score:
                    best_node, best_score = node, score
            tv.node = best_node
            tv.vec = self._mapper.matched_filter(best_node, cell)
            tv.last_seen = t
            tv.speed = math.hypot(tr.vx, tr.vy)
        confirmed = {tr.track_id for tr in self._tracks}
        for tid in list(self._track_vitals):
            tv = self._track_vitals[tid]
            if tid not in confirmed and t - tv.last_seen > _TRACK_VITALS_GRACE_SEC:
                del self._track_vitals[tid]

    def _process_track_vitals(self, t: float) -> None:
        """Feed each track's accumulated phase into its own estimators."""
        for tv in self._track_vitals.values():
            phi = tv.drain()
            if phi.size == 0:
                continue
            dec = tv.dec.push(phi)
            # A moving person's own gait dominates their spatial filter, so
            # derate by track speed rather than the room motion level: the
            # whole point is that someone ELSE walking must not destroy a
            # still person's per-track vitals.
            motion_hint = min(1.0, tv.speed / 0.5)
            bv = tv.breath.update(dec, t, motion_level=motion_hint)
            if bv is not None:
                tv.breathing_vs = bv
            hint = None
            if tv.breathing_vs is not None and tv.breathing_vs.confidence >= _BREATH_HINT_CONF:
                hint = tv.breathing_vs.rate_bpm / 60.0
            hv = tv.heart.update(dec, t, motion_level=motion_hint, breathing_hz=hint)
            if hv is not None:
                tv.heart_vs = hv

    def _push_vitals_sample(self, csi: np.ndarray) -> None:
        """Collapse one frame to a complex scalar; extend the phase series.

        With >= 2 RX chains the conjugate cross-antenna ratio cancels CFO/STO
        exactly; the amplitude-weighted mean across (rx pair, tx, subcarrier)
        — weights ``|csi[0]|^2``, i.e. the self-weighted product
        ``csi[i] * conj(csi[0])`` — is a robust single phasor whose phase
        wobbles with chest motion. Only node 0's elements are used: the
        ratio is valid within one radio chain, and mixing phasors from
        different node geometries would smear the modulation. (Per-track
        vitals use spatial filtering instead and see every node.) Single-
        chain sources get a degraded fallback (raw frame vs. background,
        CFO not cancelled).
        """
        if self._per_rx >= 2:
            csi0 = csi[: self._per_rx]  # node 0 rows
            ratio = csi_ratio(csi0)  # [per_rx-1, n_tx, n_sub]
            w = np.abs(csi0[0]) ** 2  # [n_tx, n_sub]
            z = complex(np.sum(ratio * w[None, :, :]))
        else:
            bg = self._background.background
            z = complex(np.sum(csi * np.conj(bg))) if bg is not None else 0.0
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            z = self._z_prev if self._z_prev is not None else 0.0

        # Streaming phase unwrap: accumulate the wrapped successive
        # difference, identical to np.unwrap over the whole series.
        if self._z_prev is None:
            self._phi_last = float(np.angle(z)) if z != 0 else 0.0
        else:
            self._phi_last += float(np.angle(z * np.conj(self._z_prev)))
        self._z_prev = z
        self._phi_acc.append(self._phi_last)

    def _refresh_eigenvector(self) -> None:
        """Warm-started power iteration on the trailing dynamic-CSI window.

        Tracking (rather than recomputing from scratch with an arbitrary
        global phase) keeps the projected Doppler series phase-continuous, so
        the STFT does not see artificial jumps at refresh boundaries.
        """
        _, X = self._buf.array()
        if X.shape[0] > self.cfg.stft_size:
            X = X[-self.cfg.stft_size :]
        Xc = X - X.mean(axis=0, keepdims=True)
        v = self._evec
        for _ in range(_POWER_ITERS):
            u = Xc @ v  # [T]
            v2 = Xc.conj().T @ u  # [F]
            n = float(np.linalg.norm(v2))
            if n < 1e-18:
                return  # no dynamic energy: keep the previous direction
            v = v2 / n
        self._evec = v

    def _process_vitals_chunk(self, t: float, motion_level: float) -> None:
        """Feed accumulated full-rate phase into subvocal + decimated vitals."""
        if not self._phi_acc:
            return
        phi = np.asarray(self._phi_acc, dtype=np.float64)
        self._phi_acc.clear()

        if self._subvocal is not None:
            self._sub_state = self._subvocal.update(phi, t, motion_level=motion_level)

        dec = self._vitals_dec.push(phi)
        bv = self._breath.update(dec, t, motion_level=motion_level)
        if bv is not None:
            self._breathing_vs = bv
        hint = None
        if (
            self._breathing_vs is not None
            and self._breathing_vs.confidence >= _BREATH_HINT_CONF
        ):
            hint = self._breathing_vs.rate_bpm / 60.0
        hv = self._heart.update(
            dec, t, motion_level=motion_level, breathing_hz=hint
        )
        if hv is not None:
            self._heart_vs = hv

    def _process_doppler_chunk(self) -> None:
        """Flush projected samples into the STFT; keep the newest column."""
        if not self._dopp_acc:
            return
        chunk = np.asarray(self._dopp_acc, dtype=np.complex128)
        self._dopp_acc.clear()
        cols = self._stft.push(chunk)
        if cols:
            self._doppler_col = cols[-1]
