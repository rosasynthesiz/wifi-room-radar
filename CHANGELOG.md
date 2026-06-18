# Changelog

All notable changes to **wifi-room-radar** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-06-13

Initial public release — root commit on the default branch.

The first complete cut of wifi-room-radar: passive room sensing from WiFi
channel-state information (CSI), end to end, verified against simulation
ground truth.

### Added — sensing capabilities

- **CSI capture** behind a hardware-agnostic `CSISource` interface
  - Physics-based multipath **simulator**: line-of-sight + image-method wall
    reflections + random scatterers, with per-node CFO/STO/AGC impairments and
    breathing / heartbeat / subvocal chest modulation.
  - **ESP32-CSI-Tool** serial adapter (fully implemented), plus documented
    Nexmon and Intel-5300 stubs.
  - `.npz` record / replay.
- **Signal processing** — cross-antenna conjugate ratio (CFO/STO cancellation),
  background subtraction with per-node phase alignment, streaming Butterworth
  filters, decimation, streaming STFT, PCA projection.
- **Motion & presence** — adaptive-noise-floor motion detector with hysteresis;
  presence fusion (motion ∪ credible breathing) that catches motionless people.
- **Room mapping** — Bartlett matched-field beamforming over a room grid.
- **Multistatic multi-node fusion** — 1–4 receiver nodes, coherent per-node
  maps fused in the log domain (geometric ridge intersection).
- **Multi-person tracking** — constant-velocity Kalman filters with
  Hungarian-assignment data association.
- **Vital signs** — Welch-PSD breathing estimator with band-pass whitening;
  harmonic-suppressed, prominence-picked heart-rate estimator.
- **Per-person vital signs** — spatial filter aimed at each track's map cell,
  best-isolating node selected by signal-to-interference ratio.
- **Activity classification** — idle / micro / walking / gesturing (no ML).
- **Safety alerts** — fall detection (high-Doppler burst → stillness) and
  breathing-cessation alarm (waveform-tail collapse, motion-vetoed).
- **Subvocalization detector** (experimental) — 15–80 Hz burst detector;
  research scaffold, every output flagged `experimental`.

### Added — interfaces & tooling

- **Live dashboard** — FastAPI + WebSocket server with a dependency-free
  vanilla-JS canvas UI: status cards, Doppler waterfall, room map with track
  trails / per-track vitals labels / multi-node markers, vitals strip, rolling
  timeline, and a pulsing alert banner.
- **`scripts/run_dashboard.py`** — realtime pipeline + dashboard.
- **`scripts/run_headless.py`** — offline accuracy harness, ground-truth scored.
- **`scripts/check_accuracy.py`** — sensing regression gate (CI firewall).
- **`scripts/calibrate.py`** — empty-room calibration profiles.
- **`scripts/record_csi.py`** — recording, consent-gated for live capture.
- **`scripts/mqtt_bridge.py`** — Home Assistant MQTT bridge (aggregates only).
- **CI** — `.github/workflows/ci.yml` runs pytest + the accuracy gate.

### Verified

- **38 / 38** unit + system tests passing.
- Accuracy gate **green** across 8 scenario configurations (29 thresholds):

  | Metric | Result |
  |---|---|
  | Breathing error (still person) | 0.017 bpm |
  | Heart-rate error (still person) | 0.042 bpm |
  | Per-track breathing error | 0.019 bpm |
  | Empty-room false positives | 0 presence / 0 motion / 0 tracks / 0 alerts |
  | Tracking error, single node | 1.18 m |
  | Tracking error, 3-node mesh | 0.59 m |
  | Fall detection | ~3.7 s latency, 0 false alerts |
  | Breathing-cessation alarm | ~18 s latency, 0 false alerts |
  | Subvocalization (experimental) | 82% burst detection, ~0% false positives |
  | Throughput | 800–7,300 frames/s (4–36× realtime) |

### Stack

Python 3.10+ · numpy · scipy · FastAPI · uvicorn · vanilla-JS canvas.
No machine-learning dependencies — entirely classical signal processing.

### Committed file set

```
.github/workflows/ci.yml      pyproject.toml         README.md
.gitignore                    demo_preview.py        PROJECT_REPORT.md

wifi_room_radar/        types, config, pipeline, scenarios, server, __init__
  capture/        base, simulator, replay, hardware
  processing/     preprocessing, filters, spectrogram
  detection/      motion, presence, activity, alerts
  mapping/        room_mapper
  tracking/       tracker
  vitals/         breathing, heartbeat, subvocal
  dashboard/      index.html, app.js, style.css

scripts/          run_dashboard, run_headless, check_accuracy,
                  calibrate, record_csi, mqtt_bridge
tests/            8 files, 38 tests
```

Excluded via `.gitignore`: `.venv/`, `__pycache__/`, `*.npz`,
`room-profile.json`, `.pytest_cache/`, `.claude/launch.json`.

[0.1.0]: initial release
