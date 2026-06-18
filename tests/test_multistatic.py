"""Multistatic mesh: simulator, mapper fusion, and per-track vitals."""
from __future__ import annotations

import itertools
import json

import numpy as np

from wifi_room_radar import SensingPipeline, build_scenario
from wifi_room_radar.config import SPEED_OF_LIGHT, RadioConfig
from wifi_room_radar.mapping.room_mapper import RoomMapper

RNG = np.random.default_rng(31)
ROOM = (6.0, 5.0)
MESH = [(3.0, 4.7), (3.0, 0.3)]


def _mesh_radio() -> RadioConfig:
    return RadioConfig(extra_rx_centers=list(MESH))


def test_simulator_mesh_shapes_and_independent_cfo():
    src, cfg = build_scenario("one-still", realtime=False, seed=4, nodes=3)
    frames = list(itertools.islice(src.frames(), 200))
    radio = src.radio
    assert radio.n_nodes == 3 and radio.total_rx == 9
    assert frames[0].csi.shape == (9, 1, radio.n_subcarriers)
    # CFO must be common WITHIN a node (cross-antenna ratio stable) but
    # independent BETWEEN nodes (cross-node ratio random-walks).
    within = np.array([f.csi[1, 0, 0] * np.conj(f.csi[0, 0, 0]) for f in frames])
    across = np.array([f.csi[3, 0, 0] * np.conj(f.csi[0, 0, 0]) for f in frames])
    step_within = np.abs(np.angle(within[1:] * np.conj(within[:-1])))
    step_across = np.abs(np.angle(across[1:] * np.conj(across[:-1])))
    assert np.median(step_across) > 5 * np.median(step_within)


def _scatterer_window(radio: RadioConfig, pos, n_frames=80, noise=0.02):
    """Smooth-amplitude scatterer response across ALL nodes (same forward
    model as tests/test_mapping_tracking.py, generalised to stacked nodes)."""
    tx = np.asarray(radio.tx_pos)
    rx = radio.rx_positions()  # [total_rx, 2]
    freqs = radio.subcarrier_freqs()
    d = np.linalg.norm(tx - pos) + np.linalg.norm(rx - pos[None, :], axis=1)
    resp = np.exp(-2j * np.pi * freqs[None, :] * d[:, None] / SPEED_OF_LIGHT)
    t = np.arange(n_frames)
    amp = np.exp(1j * 1.2 * np.sin(2 * np.pi * t / n_frames)) * (
        1.0 + 0.2 * np.sin(2 * np.pi * 2 * t / n_frames + 0.7)
    )
    window = amp[:, None, None] * resp[None, :, :]
    window += noise * (
        RNG.standard_normal(window.shape) + 1j * RNG.standard_normal(window.shape)
    )
    return window


def test_mesh_mapper_localises_tighter_than_worst_single_node():
    radio = _mesh_radio()
    mapper = RoomMapper(radio, room_width=ROOM[0], room_depth=ROOM[1])
    truth = np.array([2.6, 2.2])
    for _ in range(5):
        mapper.update(_scatterer_window(radio, truth))
    peaks = mapper.detect_peaks(4)
    assert peaks, "mesh mapper found no peaks"
    px, py, _ = peaks[0]
    assert np.hypot(px - truth[0], py - truth[1]) <= 0.8
    # near-radio cells must be masked out of the map entirely
    grid_flat = mapper.grid.ravel()
    assert float(np.max(grid_flat[mapper._radio_mask])) == 0.0


def test_mesh_pipeline_end_to_end_with_per_track_vitals():
    src, cfg = build_scenario("one-still", realtime=False, seed=2, nodes=3)
    pipe = SensingPipeline(src, cfg.pipeline, room_size=ROOM)
    last = None
    track_vitals = []  # per-track breathing seen in any post-warmup state
    for frame in itertools.islice(src.frames(), int(45 * src.radio.sample_rate)):
        st = pipe.step_frame(frame)
        if st is not None:
            last = st
            if st.timestamp > 25.0:
                track_vitals.extend(
                    tr.breathing for tr in st.tracks if tr.breathing is not None
                )
    assert last is not None
    json.dumps(last.to_json_dict())  # numpy must not leak through new fields
    assert last.presence
    assert last.activity == "micro"
    assert last.alerts == []
    assert track_vitals, "expected per-track breathing on the still person's track"
    best = max(track_vitals, key=lambda v: v.confidence)
    assert abs(best.rate_bpm - 14.0) < 2.0
    assert best.confidence > 0.3


def test_profile_roundtrip_preloads_floors():
    src, cfg = build_scenario("empty", realtime=False, seed=6, nodes=2)
    pipe = SensingPipeline(src, cfg.pipeline, room_size=ROOM)
    for frame in itertools.islice(src.frames(), int(8 * src.radio.sample_rate)):
        pipe.step_frame(frame)
    src.close()
    profile = {
        "motion_floor": pipe._motion._floor,
        "motion_median": pipe._motion._median,
        "mapper_noise_floor": pipe._mapper._noise_floor.tolist(),
    }
    src2, cfg2 = build_scenario("empty", realtime=False, seed=7, nodes=2)
    pipe2 = SensingPipeline(src2, cfg2.pipeline, room_size=ROOM)
    pipe2.apply_profile(profile)
    assert pipe2._motion._floor == pipe._motion._floor
    np.testing.assert_allclose(pipe2._mapper._noise_floor, pipe._mapper._noise_floor)
