"""Fall / apnea alert detectors and the activity classifier."""
from __future__ import annotations

import numpy as np

from wifi_room_radar.detection.activity import ActivityClassifier
from wifi_room_radar.detection.alerts import ApneaDetector, FallDetector
from wifi_room_radar.types import VitalSign

FREQS = np.fft.fftshift(np.fft.fftfreq(256, 1.0 / 200.0))


def _column(line_hz: float | None = None, line_db: float = -20.0) -> np.ndarray:
    """Synthetic doppler column: -55 dB floor + optional line at line_hz."""
    col = -55.0 + 0.5 * np.random.default_rng(1).standard_normal(FREQS.size)
    col += 25.0 * np.exp(-(FREQS**2) / (2 * 1.5**2))  # DC clutter
    if line_hz is not None:
        col += (line_db + 55.0) * np.exp(-((FREQS - line_hz) ** 2) / (2 * 3.0**2))
    return col


def test_fall_burst_then_stillness_fires():
    det = FallDetector()
    t = 0.0
    for _ in range(100):  # 10 s quiet floor calibration @ 10 Hz
        assert det.update(t, _column(), FREQS, motion_level=0.05) is None
        t += 0.1
    for _ in range(4):  # 0.4 s of 70 Hz fall burst
        det.update(t, _column(line_hz=70.0), FREQS, motion_level=0.9)
        t += 0.1
    alert = None
    for _ in range(30):  # stillness after the impact
        alert = det.update(t, _column(), FREQS, motion_level=0.02)
        t += 0.1
    assert alert is not None and alert.type == "fall"
    # getting up (motion) clears it
    assert det.update(t, _column(), FREQS, motion_level=0.6) is None


def test_fall_ignores_walking_band_and_busy_aftermath():
    det = FallDetector()
    t = 0.0
    for _ in range(100):
        det.update(t, _column(), FREQS, motion_level=0.05)
        t += 0.1
    # walking doppler (~25 Hz, below the 35 Hz fall band) + pause: no alert
    for _ in range(20):
        assert det.update(t, _column(line_hz=25.0), FREQS, motion_level=0.8) is None
        t += 0.1
    for _ in range(40):
        assert det.update(t, _column(), FREQS, motion_level=0.03) is None
        t += 0.1
    # a real burst NOT followed by stillness (person kept moving): no alert
    for _ in range(4):
        det.update(t, _column(line_hz=70.0), FREQS, motion_level=0.9)
        t += 0.1
    for _ in range(40):
        assert det.update(t, _column(line_hz=20.0), FREQS, motion_level=0.7) is None
        t += 0.1


def _breathing(conf: float, waveform: np.ndarray | None = None) -> VitalSign:
    wave = list(waveform) if waveform is not None else list(np.sin(np.linspace(0, 20, 90)))
    return VitalSign(rate_bpm=14.0, confidence=conf, waveform=wave, band=(0.08, 0.6))


def test_apnea_fires_on_waveform_collapse_and_clears_on_recovery():
    det = ApneaDetector()
    t = 0.0
    healthy = _breathing(0.9)
    for _ in range(60):  # 30 s of healthy breathing @ 2 Hz
        assert det.update(t, presence=True, motion_level=0.02, breathing=healthy) is None
        t += 0.5
    # breathing stops: tail of the display waveform collapses
    dead_wave = np.concatenate([np.sin(np.linspace(0, 12, 60)), 0.01 * np.ones(30)])
    stopped = _breathing(0.6, dead_wave)  # conf lags; waveform tells the truth
    alert = None
    for _ in range(40):  # 20 s
        alert = det.update(t, presence=True, motion_level=0.02, breathing=stopped)
        t += 0.5
    assert alert is not None and alert.type == "breathing_stopped"
    # recovery clears
    assert det.update(t, presence=True, motion_level=0.02, breathing=healthy) is None


def test_apnea_vetoed_by_motion():
    det = ApneaDetector()
    t = 0.0
    for _ in range(60):
        det.update(t, presence=True, motion_level=0.02, breathing=_breathing(0.9))
        t += 0.5
    # breathing "lost" but the person is moving: walking hides breathing
    for _ in range(60):
        assert det.update(t, presence=True, motion_level=0.7, breathing=_breathing(0.05)) is None
        t += 0.5


def test_activity_labels():
    clf = ActivityClassifier()
    t = 0.0
    label = "idle"
    for _ in range(80):  # nobody
        label = clf.update(t, 0.0, False, _column(), FREQS)
        t += 0.1
    assert label == "idle"
    for _ in range(80):  # someone still (breathing only)
        label = clf.update(t, 0.05, True, _column(), FREQS)
        t += 0.1
    assert label == "micro"
    for _ in range(120):  # sustained motion with a fast doppler line
        label = clf.update(t, 0.8, True, _column(line_hz=22.0), FREQS)
        t += 0.1
    assert label == "walking"
    # intermittent low-velocity bursts: gesturing (1 tick in 5 moving, no fast line)
    for i in range(150):
        moving = (i % 5) == 0
        label = clf.update(t, 0.8 if moving else 0.05, True, _column(), FREQS)
        t += 0.1
    assert label == "gesturing"
