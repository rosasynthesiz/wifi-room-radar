"""Canned simulation scenarios for demos, benchmarking and recording.

Each entry of :data:`SCENARIOS` is a zero-argument callable returning a fresh
:class:`~wifi_room_radar.config.AppConfig` with the simulated people configured.
:func:`build_scenario` resolves a name into a ready-to-run
``(SimulatedCSISource, AppConfig)`` pair.
"""
from __future__ import annotations

from typing import Callable

from .capture.simulator import SimulatedCSISource
from .config import AppConfig, PersonSpec

__all__ = ["SCENARIOS", "build_scenario"]


def _empty() -> AppConfig:
    """Nobody in the room — the false-positive baseline."""
    return AppConfig()


def _still_person(subvocal: bool = False) -> PersonSpec:
    # Placed well off the TX->RX baseline deliberately: a scatterer ON the
    # baseline is in forward-scatter geometry, where the bistatic path length
    # barely changes with range (the iso-delay ellipse degenerates onto the
    # baseline), so even wideband CSI cannot localise along the bearing.
    # Off-axis, the delay ellipse cuts the bearing ray transversally and the
    # map localises in both dimensions. Real deployments have the same
    # blind spot — it is a property of the physics, not of this simulator.
    return PersonSpec(
        x=4.2,
        y=3.8,
        mode="still",
        breathing_bpm=14.0,
        heart_bpm=70.0,
        subvocal=subvocal,
    )


def _one_still() -> AppConfig:
    """One motionless person at (4.2, 3.8): 14 bpm breathing, 70 bpm heart."""
    cfg = AppConfig()
    cfg.sim.people = [_still_person()]
    return cfg


def _one_walking() -> AppConfig:
    """One person wandering the room at 0.8 m/s."""
    cfg = AppConfig()
    cfg.sim.people = [PersonSpec(x=2.0, y=3.5, mode="walk", speed=0.8)]
    return cfg


def _two_people() -> AppConfig:
    """One walker plus one still (breathing) person."""
    cfg = AppConfig()
    cfg.sim.people = [
        PersonSpec(x=2.0, y=3.5, mode="walk", speed=0.8),
        _still_person(),
    ]
    return cfg


def _breathing_demo() -> AppConfig:
    """Alias of one-still tuned for a responsive realtime dashboard."""
    cfg = _one_still()
    cfg.pipeline.update_rate = 10.0
    cfg.pipeline.map_update_rate = 2.0
    cfg.server.ws_fps = 10.0
    return cfg


def _subvocal_demo() -> AppConfig:
    """One still person emitting speech-like 15-80 Hz micro-vibrations."""
    cfg = AppConfig()
    cfg.sim.people = [_still_person(subvocal=True)]
    return cfg


def _fall_demo() -> AppConfig:
    """One walker who falls at t = 25 s and stays down (breathing)."""
    cfg = AppConfig()
    cfg.sim.people = [
        PersonSpec(x=2.0, y=3.5, mode="walk", speed=0.9, falls_at=25.0)
    ]
    return cfg


def _apnea_demo() -> AppConfig:
    """One still person whose breathing stops at t = 40 s."""
    cfg = AppConfig()
    person = _still_person()
    person.breath_stops_at = 40.0
    cfg.sim.people = [person]
    return cfg


#: Name -> AppConfig factory. Each call returns a fresh config (safe to mutate).
SCENARIOS: dict[str, Callable[[], AppConfig]] = {
    "empty": _empty,
    "one-still": _one_still,
    "one-walking": _one_walking,
    "two-people": _two_people,
    "breathing-demo": _breathing_demo,
    "subvocal-demo": _subvocal_demo,
    "fall-demo": _fall_demo,
    "apnea-demo": _apnea_demo,
}

#: Additional receiver-node positions used when a mesh is requested
#: (node 0 stays at RadioConfig.rx_center on the east wall). Spread across
#: different walls so every spot in the room is seen from well-separated
#: bearings — the geometric diversity that collapses single-link range
#: ambiguity. Element offsets (half-wavelength, ~3 cm) stay inside the room.
MESH_RX_CENTERS: list[tuple[float, float]] = [
    (3.0, 4.7),  # north wall
    (3.0, 0.3),  # south wall
    (0.3, 0.7),  # south-west corner (4th node, used when nodes >= 4)
]


def build_scenario(
    name: str, realtime: bool, seed: int = 0, nodes: int = 1
) -> tuple[SimulatedCSISource, AppConfig]:
    """Instantiate a named scenario.

    Args:
        name: One of :data:`SCENARIOS` (e.g. ``"two-people"``).
        realtime: Pace frames to wall clock (dashboard) or free-run (offline).
        seed: Simulator RNG seed (deterministic per seed).
        nodes: Total receiver nodes (1 = classic single link; 2-4 = add
            mesh nodes from :data:`MESH_RX_CENTERS` for multistatic fusion).

    Returns:
        ``(source, cfg)`` — a fresh simulator and the full app config.

    Raises:
        KeyError: For an unknown scenario name (message lists valid names).
        ValueError: For an unsupported node count.
    """
    try:
        factory = SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}"
        ) from None
    if not 1 <= nodes <= 1 + len(MESH_RX_CENTERS):
        raise ValueError(f"nodes must be in 1..{1 + len(MESH_RX_CENTERS)}, got {nodes}")
    cfg = factory()
    cfg.sim.realtime = bool(realtime)
    cfg.sim.seed = int(seed)
    cfg.radio.extra_rx_centers = list(MESH_RX_CENTERS[: nodes - 1])
    source = SimulatedCSISource(cfg.radio, cfg.sim)
    return source, cfg
