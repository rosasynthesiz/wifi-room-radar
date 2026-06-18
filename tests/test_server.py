"""Server tests with a fake provider — no sockets, no pipeline, no realtime."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from wifi_room_radar.server import create_app
from wifi_room_radar.types import SensingState, TrackState, VitalSign


class FakeProvider:
    """Minimal duck-typed provider: each latest_state() call advances time."""

    def __init__(self) -> None:
        self._t = 0.0

    @property
    def info(self) -> dict:
        return {"type": "fake", "sample_rate": 200.0, "room_size": [6.0, 5.0]}

    def latest_state(self) -> SensingState:
        self._t += 0.1
        return SensingState(
            timestamp=round(self._t, 3),
            presence=True,
            motion_level=0.4,
            motion_detected=True,
            doppler_freqs=[-10.0, 0.0, 10.0],
            doppler_column=[-50.0, -20.0, -45.0],
            room_size=(6.0, 5.0),
            occupancy_grid=[[0.0, 0.1], [0.9, 0.2]],
            tracks=[TrackState(track_id=1, x=2.0, y=3.0, vx=0.1, vy=0.0, confidence=0.8, age=4.2)],
            breathing=VitalSign(rate_bpm=14.5, confidence=0.9, waveform=[0.0, 1.0], band=(0.08, 0.6)),
        )


def _client() -> TestClient:
    return TestClient(create_app(FakeProvider(), ws_fps=50.0))


def test_index_and_static_assets():
    client = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "app.js" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_api_info():
    client = _client()
    r = client.get("/api/info")
    assert r.status_code == 200
    assert r.json()["info"]["type"] == "fake"


def test_websocket_envelope_and_state_stream():
    client = _client()
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "info"
        assert hello["data"]["sample_rate"] == 200.0
        timestamps = []
        for _ in range(3):
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "state"
            data = msg["data"]
            for key in ("presence", "motion_level", "doppler_column", "occupancy_grid", "tracks"):
                assert key in data
            assert data["tracks"][0]["track_id"] == 1
            timestamps.append(data["timestamp"])
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)
