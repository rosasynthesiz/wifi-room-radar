"""Simulator physics tests: shapes, determinism, vitals modulation, CFO."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import signal

from wifi_room_radar.capture.simulator import SimulatedCSISource
from wifi_room_radar.config import PersonSpec, RadioConfig, SimConfig


def _take(source: SimulatedCSISource, n: int):
    return list(itertools.islice(source.frames(), n))


def _radio(fs: float = 50.0) -> RadioConfig:
    return RadioConfig(sample_rate=fs)


def _still_sim(**kw) -> SimConfig:
    return SimConfig(
        people=[PersonSpec(x=4.2, y=3.8, mode="still", breathing_bpm=15.0, heart_bpm=72.0)],
        realtime=False,
        seed=11,
        **kw,
    )


def test_frame_shapes_and_timing():
    radio = _radio()
    frames = _take(SimulatedCSISource(radio, _still_sim()), 100)
    assert len(frames) == 100
    for f in frames[:5]:
        assert f.csi.shape == (radio.n_rx, radio.n_tx, radio.n_subcarriers)
        assert np.iscomplexobj(f.csi)
        assert np.all(np.isfinite(f.csi))
    dts = np.diff([f.timestamp for f in frames])
    assert np.allclose(dts, 1.0 / radio.sample_rate, atol=1e-9)
    assert frames[0].ground_truth is not None
    person = frames[0].ground_truth["people"][0]
    assert person["mode"] == "still"
    assert person["breathing_bpm"] == 15.0


def test_determinism_per_seed():
    a = _take(SimulatedCSISource(_radio(), _still_sim()), 50)
    b = _take(SimulatedCSISource(_radio(), _still_sim()), 50)
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(fa.csi, fb.csi)


def test_breathing_visible_in_cross_antenna_ratio():
    """The conjugate cross-antenna ratio must carry a spectral line at the
    breathing rate (the simulator's core promise to the vitals pipeline)."""
    fs = 50.0
    frames = _take(SimulatedCSISource(_radio(fs), _still_sim()), int(30 * fs))
    z = np.array([np.sum(f.csi[1] * np.conj(f.csi[0])) for f in frames])
    phase = np.unwrap(np.angle(z))
    f, p = signal.welch(phase - phase.mean(), fs=fs, nperseg=len(phase) // 2, nfft=8192)
    band = (f >= 0.1) & (f <= 0.6)
    peak_hz = f[band][np.argmax(p[band])]
    assert abs(peak_hz - 15.0 / 60.0) < 0.05


def test_cfo_wild_per_antenna_but_cancelled_by_ratio():
    """Raw per-antenna phase random-walks packet to packet (CFO/LO drift);
    the cross-antenna ratio must be orders of magnitude more stable."""
    fs = 50.0
    frames = _take(SimulatedCSISource(_radio(fs), _still_sim()), int(10 * fs))
    raw = np.array([f.csi[0, 0, 0] for f in frames])
    ratio = np.array([f.csi[1, 0, 0] * np.conj(f.csi[0, 0, 0]) for f in frames])
    raw_step = np.abs(np.angle(raw[1:] * np.conj(raw[:-1])))
    ratio_step = np.abs(np.angle(ratio[1:] * np.conj(ratio[:-1])))
    assert np.median(raw_step) > 10 * np.median(ratio_step)


def test_packet_drops_skip_seq_but_keep_time_grid():
    sim = _still_sim()
    sim.packet_drop_prob = 0.3
    frames = _take(SimulatedCSISource(_radio(), sim), 200)
    seqs = np.array([f.seq for f in frames])
    assert np.all(np.diff(seqs) >= 1)
    assert np.any(np.diff(seqs) > 1)  # at 30% drop, gaps are certain in 200 frames
    dt = np.diff([f.timestamp for f in frames])
    base = 1.0 / 50.0
    assert np.allclose(dt / base, np.round(dt / base), atol=1e-6)
