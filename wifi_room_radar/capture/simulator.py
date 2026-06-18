"""Physics-based CSI simulator.

Forward model
=============
The simulated channel is a coherent sum of discrete propagation paths. For a
path of total geometric length ``L`` (metres), the complex gain seen on
subcarrier with absolute RF frequency ``f`` is::

    h(f) = g * exp(-1j * 2*pi * f * L / c)

i.e. a free-space phase rotation of ``2*pi*L/lambda(f)`` at amplitude ``g``.
Amplitudes follow the bistatic radar equation in its simplest form:

* line-of-sight (TX -> RX element):           ``g = 1 / L``
* specular wall reflection (image method):    ``g = wall_loss / L_image``
* point scatterer (TX -> p -> RX element):    ``g = reflectivity / (L1 * L2)``
  where ``L1 = |TX - p|`` and ``L2 = |p - RX_m|``.

Crucially, ``L2`` is computed *per RX array element* from
:meth:`RadioConfig.rx_positions`, so the inter-element phase gradient of each
path encodes its true angle of arrival — downstream AoA / music-style
estimators have real geometry to estimate, not a synthetic steering vector.

Static channel
--------------
Precomputed once at construction: the LoS path, the four first-order wall
reflections (TX mirrored across each wall of the ``[0..room_width] x
[0..room_depth]`` room, amplitude scaled by ``sim.wall_reflection_loss``), and
``sim.static_scatterers`` random furniture-like point reflectors. Per-frame
work is only the dynamic person paths plus hardware impairments, which keeps
the simulator far faster than realtime (target >= 2000 frames/sec free-run).

Dynamic paths (people)
----------------------
Each :class:`~wifi_room_radar.config.PersonSpec` contributes one moving point
scatterer. Walkers wander between random waypoints (kept 0.4 m off the
walls) at ``spec.speed`` with occasional 2-5 s pauses; "still" people sit at
``(spec.x, spec.y)``. Chest motion is modelled as a displacement ``d(t)``
along the radial (person -> RX-centre) direction; because both the TX->chest
and chest->RX legs shorten by ~``d``, the *effective* path length is
``L1 + L2 + 2*d(t)``:

* breathing:  ``breathing_amp * sin(2*pi * bpm/60 * t)``
* heartbeat:  raised-cosine (Hann) pulses at ``heart_bpm/60`` Hz, duty ~30%
  of the cycle, zero-mean — a sharp periodic waveform with harmonics, unlike
  a pure sine, so a matched detector has structure to lock onto.
* subvocalization (optional): ~5e-5 m of 15-80 Hz band-limited noise,
  amplitude-modulated by a speech-like burst envelope (bursts 0.3-1.5 s,
  gaps 0.5-2 s).

Magnitude sanity check (defaults, 5.18 GHz, lambda ~= 5.79 cm)
--------------------------------------------------------------
The phase deviation of a path whose length is modulated by ``2*d`` is
``4*pi*d/lambda``:

* breathing  (d = 6 mm):   4*pi*0.006/0.0579  ~= 1.3 rad  — clearly visible
  in a single subcarrier phase trace.
* heartbeat  (d = 0.4 mm): 4*pi*0.0004/0.0579 ~= 0.09 rad — needs averaging
  across subcarriers/antennas and time to pull out of the noise.
* subvocal   (d = 5e-5 m): ~0.011 rad — detectable only via narrow-band
  energy integration in 15-80 Hz.

Path-gain budget: with the default geometry the LoS amplitude is ~0.19,
walls ~0.06 each, furniture ~0.02-0.07 each, giving a mean static channel
power around -14 dB (linear ~0.036). A person at mid-room with ``rcs = 1``
has amplitude ``0.4 / (2.7*2.7) ~= 0.055`` -> power ~3e-3, i.e. **~11 dB
below the static sum** (off-centre positions land at 11-17 dB below —
realistically buried) but **~14 dB above the AWGN floor** at the default
``snr_db = 25`` — comfortably recoverable after background subtraction.

Hardware impairments (applied per packet to the summed channel)
---------------------------------------------------------------
* **CFO/LO phase**: a random-walk common phase (std
  ``sim.cfo_phase_walk_std`` rad/packet), *identical across RX elements and
  subcarriers* — all RX chains share one oscillator, so downstream code can
  cancel it with a cross-antenna conjugate ratio.
* **STO**: per-packet linear phase slope across subcarriers (jitter std
  ``sim.sto_slope_std`` rad/subcarrier), identical across RX elements.
* **AGC drift**: ~1% slow multiplicative amplitude wander (AR(1) process on
  a log scale).
* **AWGN** at ``sim.snr_db`` relative to the mean static channel power.
* **Packet drops** with probability ``sim.packet_drop_prob``: sequence
  numbers are skipped but the time grid (and all random state evolution) is
  preserved.

Everything is deterministic for a fixed ``sim.seed`` (single
``numpy.random.Generator``); the ``realtime`` flag affects pacing only, never
the random stream.
"""
from __future__ import annotations

import math
import time
from typing import Iterator, Optional

import numpy as np
from scipy import signal

from ..config import SPEED_OF_LIGHT, PersonSpec, RadioConfig, SimConfig
from ..types import CSIFrame
from .base import CSISource

#: Reflectivity of a human torso relative to the (unit) LoS excitation.
#: Chosen so a mid-room person sits ~10-20 dB below the static channel sum
#: but well above the noise floor at the default snr_db = 25 (see module
#: docstring for the budget).
PERSON_REFLECTIVITY: float = 0.4

#: Displacement scale (metres) of the subvocalization micro-vibration.
#: Deliberately optimistic: real laryngeal micro-motion is ~1e-5..1e-4 m,
#: which at snr_db = 25 sits well below the CSI phase-noise floor in the
#: 15-80 Hz band — i.e. genuinely undetectable, matching the honesty notes in
#: :mod:`wifi_room_radar.vitals.subvocal`. The simulator uses a strong-signal value
#: so the end-to-end detection harness has a real (if idealised) signature to
#: find; treat any detection as a best-case upper bound, not a capability.
SUBVOCAL_AMP: float = 8e-4

#: Minimum clearance (metres) walkers keep from the walls.
WALL_MARGIN: float = 0.4

#: dB offset turning mean |H|^2 into a plausible RSSI figure in dBm.
_RSSI_REF_DB: float = -30.0

#: Block size for incremental band-limited noise synthesis (amortizes the
#: per-call overhead of scipy's IIR filter across many frames).
_NOISE_BLOCK: int = 256


class _SubvocalGenerator:
    """Streaming generator of speech-like micro-vibration displacement.

    Produces band-limited Gaussian noise (4th-order Butterworth bandpass,
    15-80 Hz) normalized to unit standard deviation, then amplitude-modulated
    by an on/off burst envelope: "syllable" bursts lasting 0.3-1.5 s separated
    by 0.5-2 s of silence, smoothed with a ~20 ms one-pole attack/release so
    the envelope itself does not splatter energy across the band.

    The IIR filter state persists between blocks, so the noise spectrum is
    seamless across block boundaries; everything draws from the simulator's
    single :class:`numpy.random.Generator` for determinism.
    """

    def __init__(self, rng: np.random.Generator, sample_rate: float) -> None:
        self._rng = rng
        lo = 15.0
        hi = min(80.0, 0.45 * sample_rate)  # keep below Nyquist
        self._enabled = hi > lo * 1.2
        if self._enabled:
            self._sos = signal.butter(
                4, [lo, hi], btype="bandpass", fs=sample_rate, output="sos"
            )
            self._zi = np.zeros((self._sos.shape[0], 2))
            # Empirically calibrate the filter's noise gain so the output has
            # unit std before envelope/amplitude scaling (deterministic: uses
            # the shared rng).
            warm = signal.sosfilt(self._sos, rng.standard_normal(8192))
            std = float(np.std(warm[2048:]))
            self._gain = 1.0 / std if std > 1e-12 else 0.0
        self._buf = np.zeros(0)
        self._idx = 0
        # Burst envelope state: start silent.
        self._burst_on = False
        self._state_end = 0.0
        self._env = 0.0
        self._env_alpha = 1.0 - math.exp(-(1.0 / sample_rate) / 0.020)

    def sample(self, t: float) -> float:
        """Return the micro-vibration displacement (metres) at time ``t``."""
        if not self._enabled:
            return 0.0
        if self._idx >= self._buf.shape[0]:
            white = self._rng.standard_normal(_NOISE_BLOCK)
            self._buf, self._zi = signal.sosfilt(self._sos, white, zi=self._zi)
            self._idx = 0
        x = float(self._buf[self._idx]) * self._gain
        self._idx += 1
        if t >= self._state_end:
            self._burst_on = not self._burst_on
            dur = (
                self._rng.uniform(0.3, 1.5)
                if self._burst_on
                else self._rng.uniform(0.5, 2.0)
            )
            self._state_end = t + float(dur)
        target = 1.0 if self._burst_on else 0.0
        self._env += self._env_alpha * (target - self._env)
        return SUBVOCAL_AMP * self._env * x


class _Person:
    """Internal kinematic + physiological state of one simulated person.

    Motion model ("walk" mode): waypoint wander. A random waypoint is drawn
    uniformly inside the room shrunk by :data:`WALL_MARGIN`; the person walks
    straight toward it at ``spec.speed``. On arrival a new waypoint is drawn
    and, with 35% probability, the person first stands still for 2-5 s
    (velocity exactly zero — gives the Doppler processor a clean
    motion/no-motion contrast). "still" mode pins the person at
    ``(spec.x, spec.y)`` forever.

    Physiology: breathing and heartbeat phases are randomized per person so
    multiple people are mutually incoherent; see the module docstring for the
    waveforms.
    """

    def __init__(
        self,
        spec: PersonSpec,
        sim: SimConfig,
        rng: np.random.Generator,
        sample_rate: float,
    ) -> None:
        self.spec = spec
        self._sim = sim
        self._rng = rng
        lo_x, hi_x = WALL_MARGIN, max(WALL_MARGIN, sim.room_width - WALL_MARGIN)
        lo_y, hi_y = WALL_MARGIN, max(WALL_MARGIN, sim.room_depth - WALL_MARGIN)
        self._bounds = (lo_x, hi_x, lo_y, hi_y)
        self.pos = np.array(
            [min(max(spec.x, lo_x), hi_x), min(max(spec.y, lo_y), hi_y)], dtype=float
        )
        self.vel = np.zeros(2)
        self._waypoint = self._draw_waypoint()
        self._pause_until = 0.0
        # Random initial physiological phases (cycles).
        self._breath_phase0 = float(rng.uniform(0.0, 1.0))
        self._heart_phase0 = float(rng.uniform(0.0, 1.0))
        self._subvocal: Optional[_SubvocalGenerator] = (
            _SubvocalGenerator(rng, sample_rate) if spec.subvocal else None
        )
        self._fallen = False  # set once spec.falls_at has elapsed

    def _draw_waypoint(self) -> np.ndarray:
        lo_x, hi_x, lo_y, hi_y = self._bounds
        return np.array(
            [self._rng.uniform(lo_x, hi_x), self._rng.uniform(lo_y, hi_y)]
        )

    def _heart_pulse(self, t: float) -> float:
        """Zero-mean raised-cosine pulse train at ``heart_bpm/60`` Hz.

        Each cardiac cycle contains one Hann-shaped bump occupying 30% of the
        cycle (sharper than a sine -> harmonic-rich, like a real
        ballistocardiogram); the DC component (0.5 * duty) is subtracted so
        the heartbeat does not bias the mean path length.
        """
        phase = (self.spec.heart_bpm / 60.0 * t + self._heart_phase0) % 1.0
        duty = 0.30
        if phase < duty:
            bump = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase / duty))
        else:
            bump = 0.0
        return bump - 0.5 * duty

    def step(self, t: float, dt: float) -> float:
        """Advance kinematics by ``dt`` and return chest displacement (m).

        The returned displacement is along the radial (person -> RX centre)
        direction and is added (doubled, for the two path legs) to the
        effective TX->person->RX path length by the caller.
        """
        # --- fall event (see PersonSpec.falls_at) -------------------------
        # Two-phase signature for the fall detector to find: ~0.45 s of
        # rapid path-length change (the body dropping produces ~70 Hz of
        # bistatic Doppler at 5 GHz), then permanent stillness on the floor.
        fall_disp = 0.0
        if self.spec.falls_at is not None and t >= self.spec.falls_at:
            FALL_DURATION, FALL_DROP = 0.45, 0.9  # s, metres of path proxy
            progress = min(1.0, (t - self.spec.falls_at) / FALL_DURATION)
            # Smooth ramp (half-cosine) so the burst is band-limited-ish.
            fall_disp = -FALL_DROP * 0.5 * (1.0 - math.cos(math.pi * progress))
            if not self._fallen and progress >= 1.0:
                self._fallen = True
            self.vel[:] = 0.0

        if self.spec.mode == "walk" and not (
            self.spec.falls_at is not None and t >= self.spec.falls_at
        ):
            if t < self._pause_until:
                self.vel[:] = 0.0
            else:
                dx = self._waypoint[0] - self.pos[0]
                dy = self._waypoint[1] - self.pos[1]
                dist = math.hypot(dx, dy)
                step_len = self.spec.speed * dt
                if dist <= max(step_len, 1e-9):
                    self.pos[0], self.pos[1] = self._waypoint
                    self.vel[:] = 0.0
                    self._waypoint = self._draw_waypoint()
                    if self._rng.random() < 0.35:
                        self._pause_until = t + float(self._rng.uniform(2.0, 5.0))
                else:
                    scale = self.spec.speed / dist
                    self.vel[0] = dx * scale
                    self.vel[1] = dy * scale
                    self.pos[0] += self.vel[0] * dt
                    self.pos[1] += self.vel[1] * dt
        # "still": position and velocity stay fixed at the spec values / zero.
        breathing_active = not (
            self.spec.breath_stops_at is not None and t >= self.spec.breath_stops_at
        )
        breath = (
            self.spec.breathing_amp
            * math.sin(
                2.0 * math.pi * (self.spec.breathing_bpm / 60.0 * t + self._breath_phase0)
            )
            if breathing_active
            else 0.0
        )
        heart = self.spec.heart_amp * self._heart_pulse(t)
        sub = self._subvocal.sample(t) if self._subvocal is not None else 0.0
        self._breathing_active = breathing_active
        return breath + heart + sub + fall_disp

    def ground_truth(self) -> dict:
        """Plain-float ground-truth dict per the CSIFrame contract."""
        return {
            "x": float(self.pos[0]),
            "y": float(self.pos[1]),
            "vx": float(self.vel[0]),
            "vy": float(self.vel[1]),
            "mode": "still" if self._fallen else self.spec.mode,
            "breathing_bpm": float(self.spec.breathing_bpm),
            "heart_bpm": float(self.spec.heart_bpm),
            "subvocal": bool(self.spec.subvocal),
            "fallen": bool(self._fallen),
            "breathing": bool(getattr(self, "_breathing_active", True)),
        }


class SimulatedCSISource(CSISource):
    """Multipath ray-style CSI simulator (see module docstring for physics).

    Deterministic for a fixed ``sim.seed``. When ``sim.realtime`` is True,
    :meth:`frames` paces itself to wall-clock time at
    ``radio.sample_rate`` packets/sec; otherwise it free-runs as fast as the
    consumer pulls (>= 2000 frames/sec on a typical laptop).
    """

    def __init__(self, radio: RadioConfig, sim: SimConfig) -> None:
        super().__init__(radio)
        self.sim = sim
        self._rng = np.random.default_rng(sim.seed)
        self._closed = False

        # --- geometry, precomputed --------------------------------------
        self._tx = np.asarray(radio.tx_pos, dtype=float)
        rx = radio.rx_positions()  # [n_rx, 2]
        self._rx_x = np.ascontiguousarray(rx[:, 0])
        self._rx_y = np.ascontiguousarray(rx[:, 1])
        # Wavenumber per subcarrier: phase = -k(f) * L with k = 2*pi*f/c.
        self._k = 2.0 * np.pi * radio.subcarrier_freqs() / SPEED_OF_LIGHT  # [n_sub]
        # Centred subcarrier index for the STO phase slope (zero-mean so the
        # slope does not also act like a common phase).
        self._sub_idx = (
            np.arange(radio.n_subcarriers) - (radio.n_subcarriers - 1) / 2.0
        )

        # --- static channel, precomputed once ----------------------------
        self._h_static = self._build_static_channel()  # [n_rx, n_sub]
        self._static_power = float(np.mean(np.abs(self._h_static) ** 2))
        noise_power = self._static_power * 10.0 ** (-sim.snr_db / 10.0)
        self._noise_std = math.sqrt(noise_power / 2.0)  # per real/imag part

        # --- people -------------------------------------------------------
        self._people = [
            _Person(spec, sim, self._rng, radio.sample_rate) for spec in sim.people
        ]

        # --- impairment state ---------------------------------------------
        # One independent LO/AGC state per receiver NODE: physically separate
        # mesh nodes have separate oscillators and gain control, so CFO/STO
        # are common only across the elements of one node. (Downstream
        # cross-antenna ratios must therefore stay within a node.)
        n_nodes = radio.n_nodes
        self._cfo_phase = np.zeros(n_nodes)
        # AR(1) log-amplitude AGC wander: rho close to 1 -> slow drift;
        # innovation std chosen for a ~1% stationary amplitude std.
        self._agc_rho = 0.995
        self._agc_sigma = 0.01 * math.sqrt(1.0 - self._agc_rho**2)
        self._agc_log = np.zeros(n_nodes)

    # ------------------------------------------------------------------
    def _build_static_channel(self) -> np.ndarray:
        """Sum LoS + 4 wall images + random fixed scatterers.

        Returns the static channel matrix, shape ``[n_rx, n_subcarriers]``,
        complex128. Uses the image method for first-order specular wall
        reflections: mirroring the TX across a wall gives a virtual source
        whose straight-line distance to each RX element equals the true
        reflected path length; amplitude is free-space ``1/L`` further scaled
        by ``sim.wall_reflection_loss``.
        """
        tx = self._tx
        sim = self.sim
        rng = self._rng
        lengths: list[np.ndarray] = []  # each entry [n_rx]
        amps: list[np.ndarray] = []

        # Line of sight.
        d_los = np.hypot(self._rx_x - tx[0], self._rx_y - tx[1])
        lengths.append(d_los)
        amps.append(1.0 / d_los)

        # First-order wall reflections (image method, all 4 walls).
        images = (
            (-tx[0], tx[1]),  # wall x = 0
            (2.0 * sim.room_width - tx[0], tx[1]),  # wall x = room_width
            (tx[0], -tx[1]),  # wall y = 0
            (tx[0], 2.0 * sim.room_depth - tx[1]),  # wall y = room_depth
        )
        for ix, iy in images:
            d = np.hypot(self._rx_x - ix, self._rx_y - iy)
            lengths.append(d)
            amps.append(sim.wall_reflection_loss / d)

        # Random furniture-like fixed point scatterers.
        for _ in range(sim.static_scatterers):
            px = float(rng.uniform(0.3, max(0.3, sim.room_width - 0.3)))
            py = float(rng.uniform(0.3, max(0.3, sim.room_depth - 0.3)))
            refl = float(rng.uniform(0.2, 0.6))
            l1 = math.hypot(px - tx[0], py - tx[1])
            l2 = np.hypot(self._rx_x - px, self._rx_y - py)
            lengths.append(l1 + l2)
            amps.append(refl / (l1 * l2))

        length = np.stack(lengths)  # [n_paths, n_rx]
        amp = np.stack(amps)  # [n_paths, n_rx]
        # h[p, m, s] = a[p, m] * exp(-1j * k[s] * L[p, m]); sum over paths p.
        h = np.sum(
            amp[:, :, None] * np.exp(-1j * length[:, :, None] * self._k[None, None, :]),
            axis=0,
        )
        return h

    # ------------------------------------------------------------------
    def frames(self) -> Iterator[CSIFrame]:
        """Yield CSI frames on a fixed ``1/sample_rate`` time grid.

        Per frame: copy the precomputed static channel, add one path per
        person (exact per-RX-element geometry), apply CFO/STO/AGC
        impairments, add AWGN, and possibly drop the packet (sequence number
        skipped, time grid and random state preserved).
        """
        radio = self.radio
        sim = self.sim
        rng = self._rng
        fs = radio.sample_rate
        dt = 1.0 / fs
        n_tx, n_sub = radio.n_tx, radio.n_subcarriers
        n_nodes, per_node = radio.n_nodes, radio.n_rx
        total_rx = radio.total_rx  # stacked rows: node 0 elements first
        tx_x, tx_y = float(self._tx[0]), float(self._tx[1])
        k = self._k
        n = 0
        start_wall = time.perf_counter()

        while not self._closed:
            t = n * dt

            # ---- dynamic person paths ---------------------------------
            h = self._h_static.copy()
            gt_people = []
            for person in self._people:
                disp = person.step(t, dt)
                px, py = person.pos[0], person.pos[1]
                l1 = math.hypot(px - tx_x, py - tx_y)
                l2 = np.hypot(self._rx_x - px, self._rx_y - py)  # [n_rx]
                # Both legs shorten by ~disp -> effective length + 2*disp.
                length = l1 + l2 + 2.0 * disp
                amp = (PERSON_REFLECTIVITY * person.spec.rcs) / (l1 * l2)
                h += amp[:, None] * np.exp(-1j * length[:, None] * k[None, :])
                gt_people.append(person.ground_truth())

            # ---- hardware impairments ----------------------------------
            # Per-NODE LO/CFO random-walk phase: identical on the elements
            # and subcarriers of one node (single radio chain per node) but
            # independent between nodes (separate oscillators).
            self._cfo_phase += rng.normal(0.0, sim.cfo_phase_walk_std, size=n_nodes)
            # Per-packet sampling-time-offset: linear phase across
            # subcarriers, identical across one node's elements.
            sto_slope = rng.normal(0.0, sim.sto_slope_std, size=n_nodes)
            # Slow AGC-like log-amplitude wander (~1% scale), per node.
            self._agc_log = self._agc_rho * self._agc_log + rng.normal(
                0.0, self._agc_sigma, size=n_nodes
            )
            common = np.exp(self._agc_log)[:, None] * np.exp(
                1j
                * (
                    self._cfo_phase[:, None]
                    + sto_slope[:, None] * self._sub_idx[None, :]
                )
            )  # [n_nodes, n_sub]
            h *= np.repeat(common, per_node, axis=0)  # expand nodes -> element rows
            # AWGN at snr_db relative to mean static channel power.
            h += self._noise_std * (
                rng.standard_normal((total_rx, n_sub))
                + 1j * rng.standard_normal((total_rx, n_sub))
            )

            # Drop decision is drawn every grid slot regardless of the
            # probability so the random stream (hence everything else) is
            # invariant to packet_drop_prob.
            dropped = float(rng.random()) < sim.packet_drop_prob

            if not dropped:
                mean_power = float(np.mean(h.real**2 + h.imag**2))
                rssi = (
                    10.0 * math.log10(mean_power) + _RSSI_REF_DB
                    if mean_power > 0.0
                    else -100.0
                )
                if n_tx == 1:
                    csi = h[:, None, :]
                else:
                    # Multiple co-located TX antennas: identical channel per
                    # TX chain (single tx_pos in the geometry model).
                    csi = np.tile(h[:, None, :], (1, n_tx, 1))
                if sim.realtime:
                    delay = (start_wall + t) - time.perf_counter()
                    if delay > 0.0:
                        time.sleep(delay)
                yield CSIFrame(
                    timestamp=t,
                    seq=n,
                    csi=csi,
                    rssi=rssi,
                    ground_truth={"people": gt_people},
                )
            n += 1

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the frame generator at its next iteration. Idempotent."""
        self._closed = True

    @property
    def info(self) -> dict:
        base = super().info
        base.update(
            {
                "room": (self.sim.room_width, self.sim.room_depth),
                "n_people": len(self._people),
                "snr_db": self.sim.snr_db,
                "realtime": self.sim.realtime,
                "seed": self.sim.seed,
            }
        )
        return base
