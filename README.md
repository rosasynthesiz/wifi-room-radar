# wifi-room-radar — WiFi CSI room sensing

Passive room sensing from WiFi channel state information (CSI): motion and
presence detection, occupancy mapping, **multistatic multi-node fusion**,
multi-person tracking with **per-person vital signs** (breathing, heart
rate), **activity classification**, **fall + breathing-cessation alarms**,
experimental subvocalization — with a live web dashboard. Ships with a
physics-based CSI simulator; real hardware drops in behind one interface.

```
┌────────────────────────────  capture  ────────────────────────────┐
│  SimulatedCSISource │ ReplayCSISource │ ESP32 / Nexmon / Intel    │
│        (all CSISource; 1-4 receiver nodes stacked per frame)      │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼  CSIFrame [total_rx, n_tx, n_sub] @ 200 Hz
┌──────────────────────────  SensingPipeline  ──────────────────────┐
│ BackgroundSubtractor (per-node align + EMA)  →  dynamic CSI       │
│   ├─ >2 Hz high-pass power ───────→ MotionDetector → Presence     │
│   ├─ eigenvector projection ──────→ StreamingSTFT → Doppler ──┐   │
│   ├─ rolling window → RoomMapper (per-node Bartlett, log-domain│   │
│   │     mesh fusion) → Tracker ─→ per-track spatial filters    │   │
│   │                                  └→ per-track Breath/Heart │   │
│   ├─ csi_ratio phase ─┬─ decimate → room Breathing / Heartbeat │   │
│   │                   └─ full rate → SubvocalDetector          │   │
│   └─ Doppler + motion + vitals → Activity / FallDetector / ←──┘   │
│                                   ApneaDetector                   │
└────────────────────────────────┬──────────────────────────────────┘
                                 ▼  SensingState (JSON) @ 10 Hz
        FastAPI + WebSocket  →  canvas dashboard  /  MQTT bridge
```

## The physics in two paragraphs

A WiFi receiver estimates the complex channel response (amplitude and phase)
on every OFDM subcarrier of every packet — the CSI. Each measurement is the
coherent sum of all propagation paths: the direct path, wall reflections, and
reflections off people. When a body moves, its path length changes and that
path's phase rotates through the sum: walking sweeps the phase at tens of Hz
(micro-Doppler), while a motionless person's chest displaces the reflection
path by millimetres, phase-modulating the channel at the breathing rate
(~4π·6 mm/λ ≈ 1.3 rad at 5 GHz) and, far more faintly, at the heart rate
(~0.09 rad). Cross-antenna conjugate ratios cancel the oscillator's phase
noise (CFO/STO are common to all RX chains), background subtraction isolates
the moving component, and spectral analysis pulls motion, breathing, and
cardiac lines out of it. With an RX antenna array, beamforming the dynamic
component over a room grid (Bartlett matched field) localises the reflector.

Honest caveats, also enforced in code: bearing comes from a 3-element array
(~1 rad beamwidth — coarse); range comes from the delay slope across the
band, so the default simulates 80 MHz (WiFi 5/6) — at 20 MHz range is
unobservable indoors and the map degenerates to a bearing ridge. A scatterer
on the TX→RX baseline is in forward-scatter geometry where range information
vanishes entirely. Heart-rate extraction works in simulation (the cardiac
line is ~10 dB above the floor after harmonic notching and prominence-based
line picking) but is genuinely marginal on real commodity hardware.
Subvocalization detection from commodity WiFi is **not an established
capability**: the simulator uses a deliberately optimistic micro-vibration
amplitude so the detection harness has something real to find; treat that
whole path as a research scaffold (`note="experimental"` on every output).

## Quickstart

```powershell
# environment (already created at .venv if you followed the build)
python -m venv .venv
.venv\Scripts\pip install -e . pytest httpx websockets

# live dashboard — 3-node multistatic mesh, two people, falls/alerts UI
.venv\Scripts\python scripts\run_dashboard.py --scenario two-people --nodes 3 --port 8787
# → open http://127.0.0.1:8787

# offline accuracy harness (ground-truth scored, non-realtime)
.venv\Scripts\python scripts\run_headless.py --scenario one-still --duration 60
.venv\Scripts\python scripts\run_headless.py --scenario fall-demo --duration 45

# the sensing regression firewall (run after ANY DSP change; CI runs it too)
.venv\Scripts\python scripts\check_accuracy.py

# calibrate an empty room into a reusable profile, then load it
.venv\Scripts\python scripts\calibrate.py --duration 30 --out room.json
.venv\Scripts\python scripts\run_dashboard.py --profile room.json

# record a scenario to .npz (live capture requires --i-have-consent)
.venv\Scripts\python scripts\record_csi.py --scenario one-walking --duration 30 --out walk.npz

# publish presence/vitals/alerts to Home Assistant over MQTT
.venv\Scripts\python scripts\mqtt_bridge.py --broker 192.168.1.10

# tests
.venv\Scripts\python -m pytest tests -q
```

`demo_preview.py` serves the dashboard on synthetic (non-physics) data —
useful for UI work without running the pipeline.

## Multistatic mesh (`--nodes 2..4`)

One link cannot range: with a single TX→RX pair the occupancy map is a ridge
along the person's bearing (and on the TX→RX baseline, forward-scatter
geometry destroys range information entirely). Adding receiver nodes on
other walls gives every spot in the room several viewing geometries; each
node runs its own coherent Bartlett matched-field map (nodes share no
clock — cross-node phase is unobservable, exactly like real mesh hardware,
which the simulator models with independent per-node CFO/STO), and the
per-node likelihood maps are fused in the log domain so a cell must be
supported by every node that sees signal. Measured effect on the walker
scenario: **median tracking error 1.18 m (1 node) → 0.59 m (3 nodes)**.

## Safety alerts & activity

- **Fall detection**: a >35 Hz Doppler burst (a falling body; walking tops
  out near 31 Hz at 5 GHz, so gait physically cannot trip it) followed by
  sustained stillness. Measured: detected at ~3.7 s latency, zero false
  alerts across every other scenario.
- **Breathing-cessation alarm**: credible breathing that stops while the
  person stays still (waveform-tail collapse for fast response; motion
  vetoes rather than fires — gait legitimately hides breathing). Measured:
  detected ~18 s after cessation, zero false alerts.
- **Activity label**: idle / micro (breathing only) / walking / gesturing,
  from motion duty cycle + Doppler line presence. No ML.
- **Per-track vitals**: each confirmed track gets its own breathing/heart
  estimators fed by a spatial filter aimed at its map cell (best-isolating
  node selected by signal-to-interference). A still person's per-track
  breathing scores ~0.02 bpm median error in a quiet room; with an active
  walker nearby the 3-element beams are too wide for full isolation and the
  estimates honestly flag themselves unreliable.

## Scenarios

| name            | contents                                            |
|-----------------|-----------------------------------------------------|
| `empty`         | nobody — the false-positive baseline                |
| `one-still`     | motionless person, 14 bpm breathing / 70 bpm heart  |
| `one-walking`   | one walker at 0.8 m/s with pauses                   |
| `two-people`    | one walker + one still person                       |
| `breathing-demo`| `one-still` tuned for dashboard responsiveness      |
| `subvocal-demo` | still person emitting speech-like micro-vibrations  |
| `fall-demo`     | walker falls at t=25 s and stays down               |
| `apnea-demo`    | still person whose breathing stops at t=40 s        |

All scenarios accept `--nodes 1..4` (single link vs multistatic mesh) and
`--profile room.json` (preloaded calibration).

Measured on the default seeds (60 s, `run_headless`): breathing within
~0.02 bpm and heart rate within ~0.05 bpm of truth for a still person
(confidence-flagged honest: both report unreliable while gross motion
dominates); empty-room presence/motion/tracks all exactly zero; walker
tracked to ~1 m median error; subvocal demo detects ~80% of burst time with
~0% false positives on non-subvocal scenes.

## Module map

| path                              | contents                                      |
|-----------------------------------|-----------------------------------------------|
| `wifi_room_radar/types.py`              | `CSIFrame`, `SensingState`, … (the contracts) |
| `wifi_room_radar/config.py`             | radio geometry, sim, pipeline, server configs |
| `wifi_room_radar/capture/`              | `CSISource` ABC, simulator, replay, hardware  |
| `wifi_room_radar/processing/`           | ratio/align/background, filters, STFT         |
| `wifi_room_radar/detection/`            | motion (adaptive floor), presence (latched)   |
| `wifi_room_radar/mapping/room_mapper.py`| Bartlett matched-field occupancy grid         |
| `wifi_room_radar/tracking/tracker.py`   | Kalman + Hungarian multi-person tracker       |
| `wifi_room_radar/vitals/`               | breathing, heartbeat, subvocal (experimental) |
| `wifi_room_radar/pipeline.py`           | wires everything, publishes `SensingState`    |
| `wifi_room_radar/server.py` + `dashboard/` | FastAPI/WebSocket + canvas dashboard       |
| `scripts/`                        | run_dashboard, run_headless, record_csi       |

## Real hardware

Everything downstream of `CSISource` is hardware-agnostic. To go live,
implement `frames()` for your capture device and pass it to
`SensingPipeline` instead of the simulator:

- **ESP32 (easiest, ~$10)** — flash [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool);
  `wifi_room_radar/capture/hardware.py::ESP32CSISource` already parses its serial
  CSV format (`pip install pyserial`, point it at the COM port). Single
  antenna: motion/presence/vitals work, mapping/tracking need an array.
- **Nexmon CSI (Broadcom, e.g. Raspberry Pi 4 / ASUS RT-AC86U)** — finish
  `NexmonCSISource` (UDP frame parser documented in its docstring).
  Up to 4×4 antennas at 80 MHz — full mapping support.
- **Intel 5300 (the classic research card)** — finish `IntelCSISource`
  against the linux-80211n-csitool log format. 3 RX antennas, 30 subcarrier
  groups.

Calibration notes for real captures: keep `csi_ratio`-based processing (CFO
on real radios is far worse than the simulator's), expect to retune
`PipelineConfig.background_alpha` and the motion hysteresis, and measure your
actual TX/RX positions into `RadioConfig` — the mapper's steering matrix is
exactly as good as that geometry.

## Tuning guide (`PipelineConfig`)

- `background_alpha` — static-channel EMA rate. Higher adapts faster but
  starts absorbing slow human motion into the "background".
- `motion_hysteresis` — (on, off) thresholds on the 0..1 motion level.
- `breath_band` / `heart_band` — physiological search bands (Hz).
- `map_window_sec`, `map_update_rate`, `grid_resolution` — mapping cadence
  and grid granularity; cost scales with cells × window frames.
- `max_people` — tracker capacity and peak-picker budget.
- `update_rate` — `SensingState` publish rate (dashboard FPS ceiling).

## Ethics & legal

WiFi sensing observes **people**, including through interior walls, without
cameras and without their devices participating. That power carries real
privacy weight:

- **Consent**: only sense spaces whose occupants have explicitly agreed.
  "It's my router" does not make monitoring housemates or guests okay.
- **Vitals are health data**: breathing and heart-rate traces can be subject
  to health-data regulation (HIPAA/GDPR-special-category and similar).
- **Law varies**: electronic-surveillance statutes in many jurisdictions
  cover passive RF sensing. Check before deploying anywhere people live or
  work.
- This project exists for education and research on your own premises with
  the knowledge and consent of everyone present. The simulator exists
  precisely so the full stack can be developed and demonstrated with nobody
  surveilled at all.
