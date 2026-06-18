"""Room mapper localisation and multi-person tracker behaviour."""
from __future__ import annotations

import numpy as np

from wifi_room_radar.config import SPEED_OF_LIGHT, RadioConfig
from wifi_room_radar.mapping.room_mapper import RoomMapper
from wifi_room_radar.tracking.tracker import MultiPersonTracker

RNG = np.random.default_rng(21)
ROOM = (6.0, 5.0)


def _scatterer_window(radio: RadioConfig, pos, n_frames=80, noise=0.02):
    """Synthesise dynamic CSI for a scatterer at ``pos`` using the same
    bistatic forward model the mapper inverts (independent reimplementation).

    The complex amplitude varies *smoothly* over the window (slowly rotating
    phasor, breathing-like), because the mapper's activity gate uses lag-1
    temporal coherence to tell people from noise — a temporally white
    amplitude is correctly classified as noise and suppressed.
    """
    tx = np.asarray(radio.tx_pos)
    rx = radio.rx_positions()  # [n_rx, 2]
    freqs = radio.subcarrier_freqs()  # [n_sub]
    d = np.linalg.norm(tx - pos) + np.linalg.norm(rx - pos[None, :], axis=1)  # [n_rx]
    resp = np.exp(-2j * np.pi * freqs[None, :] * d[:, None] / SPEED_OF_LIGHT)  # [n_rx, n_sub]
    t = np.arange(n_frames)
    amp = np.exp(1j * 1.2 * np.sin(2 * np.pi * t / n_frames)) * (
        1.0 + 0.2 * np.sin(2 * np.pi * 2 * t / n_frames + 0.7)
    )
    window = amp[:, None, None] * resp[None, :, :]
    window += noise * (
        RNG.standard_normal(window.shape) + 1j * RNG.standard_normal(window.shape)
    )
    return window


def test_mapper_localises_scatterer():
    radio = RadioConfig()
    mapper = RoomMapper(radio, room_width=ROOM[0], room_depth=ROOM[1])
    truth = np.array([4.2, 3.6])
    for _ in range(5):
        mapper.update(_scatterer_window(radio, truth))
    idx = int(np.argmax(mapper.grid))
    cell = mapper.cell_centres[idx]
    assert np.hypot(cell[0] - truth[0], cell[1] - truth[1]) <= 1.0
    peaks = mapper.detect_peaks(4)
    assert peaks, "expected at least one peak"
    px, py, strength = peaks[0]
    assert np.hypot(px - truth[0], py - truth[1]) <= 1.0
    assert strength > 0.5


def test_mapper_noise_only_stays_quiet():
    radio = RadioConfig()
    mapper = RoomMapper(radio, room_width=ROOM[0], room_depth=ROOM[1])
    shape = (80, radio.n_rx, radio.n_subcarriers)
    # calibrate the noise floor with a few noise windows first
    for _ in range(6):
        noise = 1e-3 * (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape))
        mapper.update(noise)
    assert mapper.grid.max() < 0.3
    assert mapper.detect_peaks(4) == []


def test_tracker_confirms_and_follows_straight_line():
    tr = MultiPersonTracker(room_size=ROOM)
    tracks = []
    for k in range(20):  # 2 Hz updates, walker moving +x at 0.5 m/s
        t = 0.5 * k
        det = (1.0 + 0.25 * t + RNG.normal(0, 0.05), 2.5 + RNG.normal(0, 0.05), 0.9)
        tracks = tr.update([det], t)
    assert len(tracks) == 1
    tk = tracks[0]
    assert abs(tk.x - (1.0 + 0.25 * 9.5)) < 0.4
    assert abs(tk.vx - 0.25) < 0.15
    assert abs(tk.vy) < 0.15
    assert tk.confidence > 0.5


def test_tracker_survives_short_gap_dies_after_long_one():
    tr = MultiPersonTracker(room_size=ROOM, max_misses_sec=2.5)
    for k in range(10):
        tracks = tr.update([(2.0, 2.0, 0.9)], 0.5 * k)
    tid = tracks[0].track_id
    # 1 s gap: survives
    tracks = tr.update([], 5.5)
    tracks = tr.update([(2.0, 2.0, 0.9)], 6.0)
    assert [t.track_id for t in tracks] == [tid]
    # >2.5 s with no detections: dies
    for t in np.arange(6.5, 10.5, 0.5):
        tracks = tr.update([], float(t))
    assert tracks == []


def test_tracker_keeps_two_separate_people():
    tr = MultiPersonTracker(room_size=ROOM)
    for k in range(16):
        t = 0.5 * k
        d1 = (1.0 + 0.2 * t, 1.0, 0.9)
        d2 = (5.0 - 0.2 * t, 4.0, 0.8)
        tracks = tr.update([d1, d2], t)
    assert len(tracks) == 2
    ids = {tk.track_id for tk in tracks}
    assert len(ids) == 2
