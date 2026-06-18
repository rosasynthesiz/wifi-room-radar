"""End-to-end pipeline tests over short offline simulations."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from wifi_room_radar import SensingPipeline, build_scenario


def _run(scenario: str, seconds: float, seed: int = 1):
    source, cfg = build_scenario(scenario, realtime=False, seed=seed)
    pipe = SensingPipeline(source, cfg.pipeline, room_size=(cfg.sim.room_width, cfg.sim.room_depth))
    n = int(seconds * source.radio.sample_rate)
    states = []
    t0 = time.perf_counter()
    for frame in itertools.islice(source.frames(), n):
        st = pipe.step_frame(frame)
        if st is not None:
            states.append(st)
    return pipe, states, n / (time.perf_counter() - t0)


def test_one_still_end_to_end():
    pipe, states, fps = _run("one-still", 25.0)
    cfg_rate = 10.0
    assert len(states) > 0.8 * 25.0 * cfg_rate
    last = states[-1]
    assert last.presence  # breathing person must register
    assert 0.0 <= last.motion_level <= 1.0
    assert not last.motion_detected  # ... but a still person is not "moving"
    assert len(last.doppler_column) == len(last.doppler_freqs) > 0
    assert np.all(np.isfinite(last.doppler_column))
    rows = len(last.occupancy_grid)
    cols = len(last.occupancy_grid[0])
    assert rows == int(np.ceil(5.0 / 0.25)) and cols == int(np.ceil(6.0 / 0.25))
    assert last.breathing is not None
    assert abs(last.breathing.rate_bpm - 14.0) < 2.0
    # the JSON contract: a published state must serialise as-is
    json.dumps(last.to_json_dict())
    assert fps > 200, f"pipeline too slow for realtime: {fps:.0f} fps"


def test_empty_room_stays_quiet():
    _, states, _ = _run("empty", 15.0)
    tail = [s for s in states if s.timestamp > 5.0]
    assert np.mean([s.presence for s in tail]) < 0.1
    assert all(len(s.tracks) == 0 for s in tail)
    assert np.mean([s.motion_detected for s in tail]) < 0.1


def test_provider_contract_for_server():
    pipe, states, _ = _run("one-still", 6.0)
    st = pipe.latest_state()
    assert st is not None and st.timestamp == states[-1].timestamp
    info = pipe.info
    assert info["room_size"] == [6.0, 5.0]
    assert "tx_pos" in info and "rx_positions" in info
    json.dumps(info)
