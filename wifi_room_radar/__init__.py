"""wifi_room_radar — WiFi CSI-based room sensing.

Passive sensing from WiFi channel state information: motion and presence
detection, room occupancy mapping, multi-person tracking, and vital-sign
(breathing / heart-rate) extraction, with a live web dashboard.
"""
from .types import CSIFrame, SensingState, SubvocalState, TrackState, VitalSign
from .config import AppConfig, PersonSpec, PipelineConfig, RadioConfig, ServerConfig, SimConfig

__version__ = "0.1.0"

# Imported after __version__ so pipeline.info can resolve it lazily without
# a circular-import hazard.
from .pipeline import SensingPipeline
from .scenarios import SCENARIOS, build_scenario

__all__ = [
    "CSIFrame",
    "SensingState",
    "SubvocalState",
    "TrackState",
    "VitalSign",
    "AppConfig",
    "PersonSpec",
    "PipelineConfig",
    "RadioConfig",
    "ServerConfig",
    "SimConfig",
    "SensingPipeline",
    "SCENARIOS",
    "build_scenario",
    "__version__",
]
