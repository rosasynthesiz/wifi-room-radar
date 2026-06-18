"""Abstract CSI source interface.

Every way of obtaining CSI — the physics simulator, file replay, or real
hardware (ESP32-CSI-Tool, Nexmon, Intel 5300) — implements
:class:`CSISource`. The pipeline only ever sees this interface, so real
hardware can replace simulation without touching anything downstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from ..config import RadioConfig
from ..types import CSIFrame


class CSISource(ABC):
    """A stream of :class:`~wifi_room_radar.types.CSIFrame` measurements."""

    def __init__(self, radio: RadioConfig):
        self.radio = radio

    @abstractmethod
    def frames(self) -> Iterator[CSIFrame]:
        """Yield CSI frames in time order.

        Real-time sources (hardware, realtime simulation) pace themselves to
        wall-clock time; offline sources (replay, non-realtime simulation)
        yield as fast as the consumer pulls. The iterator ends when the
        source is exhausted or closed.
        """

    def close(self) -> None:
        """Release any underlying resources. Idempotent."""

    @property
    def info(self) -> dict:
        """Human-readable description of this source for the dashboard."""
        return {
            "type": self.__class__.__name__,
            "sample_rate": self.radio.sample_rate,
            "n_rx": self.radio.n_rx,
            "n_tx": self.radio.n_tx,
            "n_nodes": self.radio.n_nodes,
            "n_subcarriers": self.radio.n_subcarriers,
            "carrier_freq_ghz": self.radio.carrier_freq / 1e9,
            "bandwidth_mhz": self.radio.bandwidth / 1e6,
        }
