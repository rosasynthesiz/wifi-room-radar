"""Core data types shared across the wifi_room_radar package.

These are the contracts between the capture layer, the processing pipeline,
and the dashboard server. Keep fields JSON-friendly (plain floats / lists)
except for `CSIFrame.csi`, which is a numpy array and never leaves the
pipeline process.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class CSIFrame:
    """One channel-state-information measurement (one received packet).

    Attributes:
        timestamp: Seconds since stream start.
        seq: Packet sequence number (monotonically increasing; gaps allowed
            when packets are dropped).
        csi: Complex channel matrix, shape ``[n_rx, n_tx, n_subcarriers]``,
            dtype complex128.
        rssi: Received signal strength in dBm (0.0 if unknown).
        ground_truth: Simulator-only truth for evaluation. ``None`` on real
            hardware. When present, a dict like::

                {
                    "people": [
                        {"x": float, "y": float, "vx": float, "vy": float,
                         "mode": "walk"|"still", "breathing_bpm": float,
                         "heart_bpm": float, "subvocal": bool},
                        ...
                    ]
                }
    """

    timestamp: float
    seq: int
    csi: np.ndarray
    rssi: float = 0.0
    ground_truth: Optional[dict] = None


@dataclass
class VitalSign:
    """An estimated periodic vital sign (breathing or heartbeat)."""

    rate_bpm: float
    confidence: float  # 0..1; treat < 0.3 as unreliable
    waveform: list[float] = field(default_factory=list)  # recent filtered samples for display (<= 200 points)
    band: tuple[float, float] = (0.0, 0.0)  # Hz band the estimate came from


@dataclass
class TrackState:
    """One tracked person, in room coordinates (metres).

    ``breathing`` / ``heartbeat`` are per-person vitals extracted from the
    dynamic CSI spatially filtered toward this track's position (None until
    the per-track estimators have warmed up or when unreliable).
    """

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    confidence: float  # 0..1
    age: float  # seconds since the track was confirmed
    breathing: Optional[VitalSign] = None
    heartbeat: Optional[VitalSign] = None


@dataclass
class SubvocalState:
    """Experimental subvocalization / micro-motion detector output."""

    active: bool
    activity_score: float  # 0..1
    band_energy: list[float] = field(default_factory=list)  # per analysis sub-band, dB-ish relative units
    note: str = "experimental"


@dataclass
class SensingState:
    """Everything the pipeline knows at one instant; serialised to the dashboard.

    All numeric fields must be plain Python floats/ints/lists (no numpy
    scalars/arrays) so ``to_json_dict`` is directly JSON-serialisable.
    """

    timestamp: float
    presence: bool = False
    motion_level: float = 0.0  # 0..1 normalised motion energy
    motion_detected: bool = False
    doppler_freqs: list[float] = field(default_factory=list)  # Hz, fft-shifted (negative..positive)
    doppler_column: list[float] = field(default_factory=list)  # dB magnitudes, same length as doppler_freqs
    room_size: tuple[float, float] = (0.0, 0.0)  # (width_m, depth_m)
    occupancy_grid: list[list[float]] = field(default_factory=list)  # [rows=depth][cols=width], values 0..1
    tracks: list[TrackState] = field(default_factory=list)
    breathing: Optional[VitalSign] = None
    heartbeat: Optional[VitalSign] = None
    subvocal: Optional[SubvocalState] = None
    activity: str = "idle"  # "idle" | "micro" | "walking" | "gesturing" (room-level)
    alerts: list[dict] = field(default_factory=list)  # [{"type": "fall"|"breathing_stopped", "message": str, "since": float}]
    ground_truth: Optional[dict] = None  # passed through from the source when simulating

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this state."""
        return asdict(self)
