"""Detector and vital-sign estimator tests on synthetic scalar streams."""
from __future__ import annotations

import numpy as np
from scipy import signal

from wifi_room_radar.detection.motion import MotionDetector, MotionResult
from wifi_room_radar.detection.presence import PresenceDetector
from wifi_room_radar.vitals.breathing import BreathingEstimator
from wifi_room_radar.vitals.heartbeat import HeartbeatEstimator
from wifi_room_radar.vitals.subvocal import SubvocalDetector

RNG = np.random.default_rng(9)


def _drive_motion(det: MotionDetector, energies, fs):
    out = []
    for i, e in enumerate(energies):
        out.append(det.update(float(e), i / fs))
    return out


def test_motion_detector_step_response_and_hysteresis():
    fs = 200.0
    det = MotionDetector(fs=fs)
    quiet = 1.0 + 0.05 * RNG.standard_normal(int(20 * fs))
    loud = 20.0 + 1.0 * RNG.standard_normal(int(5 * fs))
    res_quiet = _drive_motion(det, quiet, fs)
    assert not res_quiet[-1].detected
    assert res_quiet[-1].level < 0.2
    res_loud = _drive_motion(det, loud, fs)
    assert res_loud[-1].detected
    assert res_loud[-1].level > 0.6
    # detection within ~1 s of the step
    first_on = next(i for i, r in enumerate(res_loud) if r.detected)
    assert first_on < 1.5 * fs
    res_release = _drive_motion(det, 1.0 + 0.05 * RNG.standard_normal(int(20 * fs)), fs)
    assert not res_release[-1].detected


def test_presence_latch_and_breathing_debounce():
    pres = PresenceDetector(hold_sec=5.0, breathing_debounce_sec=2.5)
    moving = MotionResult(level=0.8, detected=True, raw_energy=1.0)
    still = MotionResult(level=0.0, detected=False, raw_energy=0.0)
    assert pres.update(moving, 0.0, timestamp=0.0)
    assert pres.update(still, 0.0, timestamp=4.0)  # held
    assert not pres.update(still, 0.0, timestamp=6.0)  # hold expired
    # a momentary confidence spike must NOT count (debounce) ...
    assert not pres.update(still, 0.9, timestamp=7.0)
    assert not pres.update(still, 0.0, timestamp=8.0)
    # ... but sustained confidence must
    for k, t in enumerate(np.arange(10.0, 13.5, 0.5)):
        latched = pres.update(still, 0.9, timestamp=float(t))
    assert latched


def test_breathing_estimator_accuracy():
    fs = 20.0
    est = BreathingEstimator(fs)
    t = np.arange(int(40 * fs)) / fs
    x = np.sin(2 * np.pi * 0.25 * t) + 0.2 * RNG.standard_normal(t.size)  # 15 bpm
    vs = None
    for chunk in np.split(x, 40):
        out = est.update(chunk, float(t[-1]))
        vs = out if out is not None else vs
    assert vs is not None
    assert abs(vs.rate_bpm - 15.0) < 1.0
    assert vs.confidence > 0.5
    assert 0 < len(vs.waveform) <= 200


def test_heartbeat_estimator_rejects_breathing_harmonics():
    fs = 20.0
    est = HeartbeatEstimator(fs)
    t = np.arange(int(40 * fs)) / fs
    breath = 1.0 * np.sin(2 * np.pi * 0.25 * t)
    # non-sinusoidal breathing -> strong harmonics at 0.5/0.75/1.0 Hz
    breath = breath + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.15 * np.sin(2 * np.pi * 0.75 * t)
    heart = 0.05 * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
    x = breath + heart + 0.01 * RNG.standard_normal(t.size)
    vs = None
    for chunk in np.split(x, 40):
        out = est.update(chunk, float(t[-1]), motion_level=0.0, breathing_hz=0.25)
        vs = out if out is not None else vs
    assert vs is not None
    assert abs(vs.rate_bpm - 72.0) < 6.0


def _bandlimited_noise(n, fs, lo, hi):
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, RNG.standard_normal(n))


def test_subvocal_bursty_fires_steady_hum_does_not():
    fs = 200.0
    n = int(30 * fs)
    t = np.arange(n) / fs
    base = 0.02 * RNG.standard_normal(n)

    burst_env = ((t % 2.0) < 0.8).astype(float)
    bursty = base + 1.0 * burst_env * _bandlimited_noise(n, fs, 25, 70)
    det = SubvocalDetector(fs)
    states = [det.update(c, float(i)) for i, c in enumerate(np.split(bursty, 30))]
    assert any(s.active for s in states[10:])

    hum = base + 1.0 * np.sin(2 * np.pi * 50.0 * t)  # steady mains-like tone
    det2 = SubvocalDetector(fs)
    states2 = [det2.update(c, float(i)) for i, c in enumerate(np.split(hum, 30))]
    assert not any(s.active for s in states2[10:])


def test_subvocal_motion_gate_suppresses():
    fs = 200.0
    n = int(20 * fs)
    t = np.arange(n) / fs
    burst_env = ((t % 2.0) < 0.8).astype(float)
    x = 1.0 * burst_env * _bandlimited_noise(n, fs, 25, 70)
    det = SubvocalDetector(fs)
    states = [
        det.update(c, float(i), motion_level=0.9) for i, c in enumerate(np.split(x, 20))
    ]
    assert not any(s.active for s in states)
    assert all(s.activity_score == 0.0 for s in states)
