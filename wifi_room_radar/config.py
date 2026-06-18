"""Configuration dataclasses for the wifi_room_radar package.

The geometry of the radio link (TX position, RX array position/orientation)
lives in :class:`RadioConfig` because both the simulator (forward model) and
the room mapper (inverse model) must agree on it. Room coordinates are in
metres, origin at the south-west corner; x runs along the room width, y along
the room depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass
class RadioConfig:
    """Radio / link parameters shared by capture sources and the pipeline."""

    carrier_freq: float = 5.21e9  # Hz (5 GHz, ~channel 42 centre of an 80 MHz block)
    # 80 MHz (WiFi 5/6 VHT80) rather than legacy 20 MHz: bistatic range
    # resolution is c/(2B) — ~1.9 m at 80 MHz vs ~7.5 m at 20 MHz. At 20 MHz
    # the occupancy map degenerates into a flat ridge along the person's
    # bearing (range unobservable inside a normal room); at 80 MHz the
    # across-band delay slope localises along the ridge too. 64 simulated
    # subcarriers spread over the band keep compute light (real VHT80 has 234
    # usable tones; a uniform 64-tone subset carries the same geometry).
    bandwidth: float = 80e6  # Hz
    n_subcarriers: int = 64
    n_rx: int = 3
    n_tx: int = 1
    sample_rate: float = 200.0  # CSI packets per second
    tx_pos: tuple[float, float] = (0.3, 2.5)  # metres, room coordinates
    rx_center: tuple[float, float] = (5.7, 2.5)  # metres, centre of the RX array
    rx_axis: tuple[float, float] = (0.0, 1.0)  # unit vector along which RX elements are spaced
    rx_spacing: Optional[float] = None  # metres between adjacent RX elements; None -> half wavelength
    # Multistatic mesh: centres of ADDITIONAL receiver nodes beyond rx_center
    # (each an n_rx-element array with the same axis/spacing). Every node is
    # an independent radio chain — independent CFO/STO, no shared clock — so
    # processing may be coherent only WITHIN a node; fusion across nodes is
    # incoherent (per-node power maps summed). Multiple viewing geometries
    # are what resolve the single-link range ambiguity along the bearing.
    extra_rx_centers: list[tuple[float, float]] = field(default_factory=list)

    @property
    def wavelength(self) -> float:
        """Carrier wavelength in metres."""
        return SPEED_OF_LIGHT / self.carrier_freq

    @property
    def element_spacing(self) -> float:
        """Actual RX element spacing in metres (resolves the half-wavelength default)."""
        return self.rx_spacing if self.rx_spacing is not None else self.wavelength / 2.0

    @property
    def n_nodes(self) -> int:
        """Number of receiver nodes (1 + extra_rx_centers)."""
        return 1 + len(self.extra_rx_centers)

    @property
    def total_rx(self) -> int:
        """Total RX elements across all nodes (rows of a CSI frame)."""
        return self.n_nodes * self.n_rx

    def node_centers(self) -> list[tuple[float, float]]:
        """Array centre of every node, node 0 first."""
        return [tuple(self.rx_center)] + [tuple(c) for c in self.extra_rx_centers]

    def node_slice(self, node: int) -> slice:
        """Row slice of node ``node``'s elements within a stacked CSI frame."""
        return slice(node * self.n_rx, (node + 1) * self.n_rx)

    def rx_positions(self) -> np.ndarray:
        """Positions of every RX element across all nodes, ``[total_rx, 2]``.

        Node order matches :meth:`node_centers`; within a node, elements are
        centred on the node centre and spaced ``element_spacing`` apart along
        ``rx_axis``. CSI frames stack node element rows in this same order.
        """
        axis = np.asarray(self.rx_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        offsets = (np.arange(self.n_rx) - (self.n_rx - 1) / 2.0) * self.element_spacing
        blocks = [
            np.asarray(c, dtype=float)[None, :] + offsets[:, None] * axis[None, :]
            for c in self.node_centers()
        ]
        return np.concatenate(blocks, axis=0)

    def subcarrier_freqs(self) -> np.ndarray:
        """Absolute RF frequency of each subcarrier, shape ``[n_subcarriers]`` (Hz)."""
        k = np.arange(self.n_subcarriers) - (self.n_subcarriers - 1) / 2.0
        return self.carrier_freq + k * (self.bandwidth / self.n_subcarriers)


@dataclass
class PersonSpec:
    """One simulated person."""

    x: float
    y: float
    mode: str = "walk"  # "walk" | "still"
    speed: float = 0.8  # m/s while walking
    breathing_bpm: float = 15.0
    heart_bpm: float = 68.0
    breathing_amp: float = 0.006  # metres of chest displacement
    heart_amp: float = 0.0004  # metres
    subvocal: bool = False  # add speech-like micro-vibration bursts (15-80 Hz)
    rcs: float = 1.0  # relative reflectivity (radar cross-section scale)
    falls_at: Optional[float] = None  # sim-seconds: rapid fall, then lying still
    breath_stops_at: Optional[float] = None  # sim-seconds: breathing amplitude -> 0


@dataclass
class SimConfig:
    """Parameters of the simulated environment."""

    room_width: float = 6.0  # metres (x extent)
    room_depth: float = 5.0  # metres (y extent)
    people: list[PersonSpec] = field(default_factory=list)
    snr_db: float = 25.0  # AWGN level relative to mean static channel power
    static_scatterers: int = 6  # random furniture-like fixed reflectors
    wall_reflection_loss: float = 0.4  # amplitude factor applied to first-order wall images
    cfo_phase_walk_std: float = 0.5  # rad/packet random-walk std of common LO phase
    sto_slope_std: float = 0.02  # rad/subcarrier per-packet timing-offset jitter
    packet_drop_prob: float = 0.0
    realtime: bool = True  # pace frames() to wall-clock; False = generate as fast as possible
    seed: Optional[int] = 0


@dataclass
class PipelineConfig:
    """Tuning knobs for the processing pipeline."""

    update_rate: float = 10.0  # SensingState publishes per second
    stft_size: int = 256
    stft_hop: int = 32
    background_alpha: float = 0.01  # EMA coefficient for the static-background estimate
    breath_band: tuple[float, float] = (0.08, 0.6)  # Hz
    heart_band: tuple[float, float] = (0.8, 2.2)  # Hz
    subvocal_band: tuple[float, float] = (15.0, 80.0)  # Hz
    vitals_fs: float = 20.0  # Hz, decimated rate used for vitals analysis
    map_window_sec: float = 0.5  # CSI window per occupancy-map update
    map_update_rate: float = 2.0  # occupancy map / tracker updates per second
    grid_resolution: float = 0.25  # metres per occupancy-grid cell
    max_people: int = 4
    motion_hysteresis: tuple[float, float] = (0.25, 0.15)  # (on, off) thresholds on motion_level
    presence_hold_sec: float = 5.0  # keep presence after last evidence for this long


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    ws_fps: float = 10.0  # max state pushes per second per websocket client


@dataclass
class AppConfig:
    """Top-level application configuration."""

    radio: RadioConfig = field(default_factory=RadioConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
