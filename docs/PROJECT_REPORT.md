# wifi-room-radar — Project Report

**WiFi CSI-based passive room sensing**
Version 0.1.0 · Report generated 2026-06-12 · ~8,000 lines across 49 source files

---

## 1. Executive summary

**wifi-room-radar** turns ordinary WiFi radio signals into a real-time spatial
sensing system. Using only the channel-state information (CSI) that a
receiver already estimates on every packet, it detects presence and motion,
maps where people are in a room, tracks multiple people simultaneously,
extracts per-person breathing and heart rate, classifies activity, and
raises fall and breathing-cessation alarms — through walls, in the dark, with
no cameras and no wearables.

The system is built around a **hardware-agnostic capture interface**: it ships
with a rigorous physics-based CSI simulator (so the entire stack can be
developed, demonstrated, and regression-tested with nobody actually
surveilled), and real hardware — ESP32, Nexmon, Intel 5300 — drops in behind
the same interface without touching anything downstream.

Every claim in this report is **measured against simulation ground truth** by
an automated accuracy harness, not asserted. The headline numbers:

| Capability | Measured result (default seed) |
|---|---|
| Breathing rate (still person) | **0.017 bpm** median error, 100% reliable |
| Heart rate (still person) | **0.042 bpm** median error |
| Per-track breathing | **0.019 bpm** median error |
| Empty-room false positives | **0** presence, **0** motion, **0** tracks, **0** alerts |
| Single-node tracking | **1.18 m** median position error |
| **3-node mesh tracking** | **0.59 m** median position error |
| Fall detection | detected at **~3.7 s** latency, 0 false alerts |
| Breathing-cessation alarm | detected at **~18 s**, 0 false alerts |
| Subvocalization (experimental) | 82% burst detection, ~0% false positives |
| Throughput | **800–7,000 frames/s** (4–35× realtime) |
| Test suite | **38/38 passing**; full accuracy gate green |

---

## 2. Capabilities

| Capability | Status | How |
|---|---|---|
| CSI capture (simulated) | ✅ Production | Physics-based multipath simulator with hardware impairments |
| CSI capture (real hardware) | ✅ ESP32 ready; Nexmon/Intel stubbed | `CSISource` interface; ESP32 serial parser fully implemented |
| Signal processing pipeline | ✅ | Background subtraction, phase sanitisation, STFT, streaming filters |
| Motion detection | ✅ | Adaptive noise-floor, hysteresis-debounced |
| Presence detection | ✅ | Motion ∪ credible-breathing fusion (catches motionless people) |
| Room occupancy mapping | ✅ | Bartlett matched-field beamforming over a room grid |
| **Multistatic mesh fusion** | ✅ | 1–4 receiver nodes, log-domain per-node map fusion |
| Multi-person tracking | ✅ | Kalman filters + Hungarian assignment |
| Breathing extraction | ✅ | Welch PSD with band-pass whitening |
| Heart-rate extraction | ✅ | Harmonic-suppressed, prominence-based line picking |
| **Per-person vital signs** | ✅ | Spatial filter aimed at each track's cell |
| **Activity classification** | ✅ | idle / micro / walking / gesturing |
| **Fall detection** | ✅ | High-Doppler burst → stillness |
| **Breathing-cessation alarm** | ✅ | Waveform-tail collapse, motion-vetoed |
| Subvocalization detection | ⚗️ Experimental | 15–80 Hz burst detector (research scaffold) |
| Live dashboard | ✅ | FastAPI + WebSocket + vanilla-JS canvas |
| MQTT / Home Assistant bridge | ✅ | Auto-discovery, privacy-conscious export |
| Room calibration | ✅ | Empty-room baseline profiles |
| Accuracy regression gate | ✅ | CI-enforced sensing thresholds |

---

## 3. The physics

A WiFi receiver estimates the complex channel response — amplitude and phase —
on every OFDM subcarrier of every packet. This CSI is the coherent sum of all
propagation paths between transmitter and receiver: the direct line-of-sight
path, reflections off walls and furniture (the *static* channel), and
reflections off people (the *dynamic* channel).

When a body moves, the length of its reflected path changes, and that path's
phase rotates through the sum. Three regimes matter:

- **Gross motion** (walking, gesturing) sweeps the reflected phase at tens of
  Hz — micro-Doppler.
- **Breathing** displaces the chest by millimetres, phase-modulating the
  channel at the respiration rate. At 5 GHz (λ ≈ 5.8 cm) a ~6 mm chest
  excursion produces a phase swing of `4π·6mm/λ ≈ 1.3 rad` — large and clearly
  visible.
- **Heartbeat** displaces the chest surface by ~0.2–0.5 mm — roughly 15× smaller,
  giving only ~0.09 rad of phase. It sits barely above the phase-noise floor
  and requires harmonic-suppressed spectral averaging to recover.

The core processing chain is:

1. **Cancel oscillator noise.** The transmitter and receiver run on separate
   clocks, so every packet carries a random common phase (carrier frequency
   offset, CFO) and a linear phase slope across subcarriers (sampling timing
   offset, STO). Because all antennas of one radio share the same clock, a
   **cross-antenna conjugate ratio** cancels both exactly.
2. **Subtract the static channel.** An exponential moving average of
   phase-aligned frames converges to the walls-and-furniture channel; the
   residual is the moving-reflector ("dynamic") component.
3. **Analyse the dynamic component.** Spectral analysis of its Doppler content
   gives motion; spatial beamforming over a room grid gives location; the
   phase of an antenna ratio gives breathing and heartbeat.

### Honest physical limits (enforced in code, not hidden)

- **Range resolution is poor.** Range comes from the delay slope across the
  band: `c/(2B)`. At 20 MHz that is ~7.5 m — wider than a room. wifi-room-radar
  therefore defaults to **80 MHz** (WiFi 5/6), giving ~1.9 m, and even then
  most localisation power comes from **bearing** (angle of arrival across the
  antenna array), which is why a single link produces a *ridge* of likely
  positions rather than a point.
- **The TX→RX baseline is blind.** A person standing on the line between
  transmitter and receiver is in forward-scatter geometry, where the bistatic
  path length barely changes with range — range information vanishes
  entirely. This is a property of the physics, not a bug, and it is the
  primary motivation for the multistatic mesh (§7).
- **Heartbeat is marginal on real hardware.** It works in simulation
  (the cardiac line is ~10 dB above the floor after harmonic notching), but on
  commodity radios the phase-noise floor makes it genuinely hard.
- **Subvocalization is unproven.** Detecting silent speech from commodity WiFi
  is not an established capability; the simulator uses a deliberately
  optimistic micro-vibration amplitude so the detection *harness* has a real
  signature to find. Every output is flagged `experimental`.

---

## 4. Architecture

```
┌────────────────────────────  capture  ────────────────────────────┐
│  SimulatedCSISource │ ReplayCSISource │ ESP32 / Nexmon / Intel     │
│        (all CSISource; 1-4 receiver nodes stacked per frame)       │
└────────────────────────────────┬───────────────────────────────────┘
                                 ▼  CSIFrame [total_rx, n_tx, n_sub] @ 200 Hz
┌──────────────────────────  SensingPipeline  ───────────────────────┐
│ BackgroundSubtractor (per-node align + EMA)  →  dynamic CSI        │
│   ├─ >2 Hz high-pass power ────────→ MotionDetector → Presence     │
│   ├─ eigenvector projection ───────→ StreamingSTFT → Doppler ──┐   │
│   ├─ rolling window → RoomMapper (per-node Bartlett, log-domain │   │
│   │     mesh fusion) → MultiPersonTracker ─→ per-track spatial  │   │
│   │                                  filters → per-track vitals │   │
│   ├─ csi_ratio phase ─┬─ decimate → room Breathing / Heartbeat  │   │
│   │                   └─ full rate → SubvocalDetector           │   │
│   └─ Doppler + motion + vitals → Activity / Fall / Apnea ←──────┘   │
└────────────────────────────────┬───────────────────────────────────┘
                                 ▼  SensingState (JSON) @ 10 Hz
        FastAPI + WebSocket  →  canvas dashboard  /  MQTT bridge
```

### Design principles

- **One contract, swappable backends.** Everything downstream of `CSISource`
  is hardware-agnostic. The pipeline only ever sees the abstract frame
  interface, so a simulator, a file replay, or an ESP32 are interchangeable.
- **Duck-typed presentation edge.** The server never imports the pipeline; it
  talks to any object exposing `latest_state()` and `info`. This is why the
  same dashboard serves both the real pipeline and the synthetic UI demo.
- **Streaming everywhere.** Every DSP block keeps its own state and processes
  small chunks incrementally — no block re-initialises per call, no buffer
  grows unbounded. The whole pipeline runs many times faster than realtime.
- **Plain-data wire format.** `SensingState` is a dataclass of pure
  floats/lists; `asdict()` is directly JSON-serialisable, with a numpy-leak
  guard at the server boundary.

---

## 5. Module-by-module breakdown

### Package `wifi_room_radar/` — core library (~4,400 lines)

| Module | Lines | Responsibility |
|---|---|---|
| `types.py` | 90 | Data contracts: `CSIFrame`, `SensingState`, `TrackState`, `VitalSign`, `SubvocalState` |
| `config.py` | 137 | `RadioConfig` (geometry, multi-node), `SimConfig`, `PersonSpec`, `PipelineConfig`, `ServerConfig`, `AppConfig` |
| `pipeline.py` | 573 | `SensingPipeline` — wires every component into one frame-driven loop |
| `scenarios.py` | 125 | 8 canned scenarios + `build_scenario(name, realtime, seed, nodes)` |
| `server.py` | 136 | FastAPI app, `/ws` WebSocket stream, static dashboard host |
| **capture/** | | |
| `base.py` | 38 | `CSISource` abstract interface |
| `simulator.py` | 487 | Physics-based multipath CSI simulator |
| `replay.py` | 170 | `.npz` record/replay |
| `hardware.py` | 336 | ESP32 (complete), Nexmon & Intel 5300 (documented stubs) |
| **processing/** | | |
| `preprocessing.py` | 355 | `csi_ratio`, `align_to_background`, `BackgroundSubtractor`, `HampelFilter`, `CSIBuffer`, `pca_first_component` |
| `filters.py` | 194 | `design_bandpass`, `StreamingSOSFilter`, `Decimator`, smoothing |
| `spectrogram.py` | 115 | `StreamingSTFT`, `doppler_bins` |
| **detection/** | | |
| `motion.py` | 157 | Adaptive-floor motion detector with hysteresis |
| `presence.py` | 95 | Latched motion ∪ breathing fusion |
| `activity.py` | 109 | idle / micro / walking / gesturing classifier |
| `alerts.py` | 250 | `FallDetector`, `ApneaDetector`, `AlertManager` |
| **mapping/** | | |
| `room_mapper.py` | 402 | Bartlett matched-field map, per-node mesh fusion, peak detection |
| **tracking/** | | |
| `tracker.py` | 301 | `MultiPersonTracker` — Kalman + Hungarian association |
| **vitals/** | | |
| `breathing.py` | 288 | Welch-PSD breathing estimator with band-pass whitening |
| `heartbeat.py` | 296 | Harmonic-suppressed, prominence-picked heart-rate estimator |
| `subvocal.py` | 325 | Experimental 15–80 Hz burst detector |

### `wifi_room_radar/dashboard/` — web UI (~1,400 lines)

| File | Lines | Contents |
|---|---|---|
| `app.js` | 934 | State store, WebSocket client, `Spectrogram` / `RoomMap` / `LineChart` / `Timeline` renderers, alert banner, trails, per-track labels |
| `style.css` | 374 | Dark control-room theme, alert-pulse animation, responsive grid |
| `index.html` | 96 | Panel layout |

### `scripts/` — entry points (~660 lines)

| Script | Lines | Purpose |
|---|---|---|
| `run_dashboard.py` | 52 | Launch realtime pipeline + dashboard |
| `run_headless.py` | 260 | Offline accuracy harness (ground-truth scored) |
| `check_accuracy.py` | 107 | Sensing regression gate (CI firewall) |
| `calibrate.py` | 70 | Empty-room calibration profiles |
| `record_csi.py` | 62 | Record scenarios to `.npz` (consent-gated for live) |
| `mqtt_bridge.py` | 105 | Home Assistant MQTT publisher |

### `tests/` — verification (~540 lines, 38 tests)

| File | Lines | Covers |
|---|---|---|
| `test_simulator.py` | 68 | Frame shapes, determinism, breathing visibility, CFO cancellation |
| `test_processing.py` | 77 | Ratio, alignment, background subtraction, PCA, STFT, decimation |
| `test_detection_vitals.py` | 102 | Motion, presence, breathing, heartbeat, subvocal |
| `test_mapping_tracking.py` | 90 | Localisation, peak detection, Kalman tracking |
| `test_pipeline.py` | 50 | End-to-end, JSON contract, throughput |
| `test_server.py` | 58 | HTTP routes, WebSocket envelope |
| `test_alerts_activity.py` | 102 | Fall, apnea, activity classification |
| `test_multistatic.py` | 94 | Mesh simulator, fusion, per-track vitals, profiles |

### Root & ops

| File | Lines | Purpose |
|---|---|---|
| `README.md` | 183 | User-facing documentation |
| `demo_preview.py` | 109 | Synthetic-data dashboard (UI work without the pipeline) |
| `pyproject.toml` | 23 | Package metadata, dependencies |
| `.github/workflows/ci.yml` | 23 | pytest + accuracy gate on push/PR |

---

## 6. The signal-processing pipeline in detail

Per frame (driven at the 200 Hz CSI rate), `SensingPipeline.step_frame`:

1. **Background subtraction.** `BackgroundSubtractor.update(csi)` phase-aligns
   the raw frame to the running static-channel estimate (per receiver node,
   since nodes have independent clocks) and returns the dynamic residual.
2. **Motion.** The Doppler-projected sample is high-passed at 2 Hz (so sub-Hz
   breathing does not register as motion) and its power, averaged across all
   features, feeds the adaptive-floor `MotionDetector`.
3. **Micro-Doppler.** The flattened dynamic CSI is projected onto a slowly
   tracked dominant eigenvector (warm-started power iteration), and the
   resulting complex series feeds a `StreamingSTFT` whose newest column is the
   dashboard's Doppler waterfall.
4. **Room vitals.** The cross-antenna conjugate ratio (`csi_ratio`, node 0)
   cancels CFO/STO; an amplitude-weighted complex mean collapses it to one
   scalar per frame; its unwrapped phase is the chest-motion series.
   Decimated to 20 Hz it feeds the room `BreathingEstimator` and
   `HeartbeatEstimator`; at full rate it feeds the `SubvocalDetector`.
5. **Per-track projections.** Each confirmed track's matched-filter steering
   vector projects the dynamic CSI to a per-person chest-motion series (§8).
6. **Mapping & tracking** (at 2 Hz): a rolling window of dynamic CSI goes
   through `RoomMapper`; its peaks feed `MultiPersonTracker`.
7. **Presence, activity, alerts.** Fused from the above.
8. **Publish** (at 10 Hz): a `SensingState` is assembled from pure Python
   floats and stored under a lock for the server to poll.

### Vital-sign estimator design (the subtle part)

Both estimators share a careful spectral pipeline:

- **Band-pass whitening.** The buffered signal was Butterworth-filtered, so
  even white noise has a mid-band PSD bump that would masquerade as a peak.
  Dividing the PSD by `|H(f)|²` restores a flat noise floor.
- **Edge guards.** Out-of-band energy leaks through the filter skirt and piles
  up at band-edge bins; the peak search excludes a guard region while the edge
  bins still count toward the confidence denominator.
- **Heartbeat harmonic suppression.** Breathing is not sinusoidal, so its
  harmonics (2f, 3f, …) extend into the cardiac band — and at 70 bpm on 14 bpm
  breathing, the heart rate sits *exactly* on the 5th breathing harmonic.
  Bins within ±0.08 Hz of breathing multiples (up to the 4th, beyond which
  Bessel-comb harmonics are negligible) are clamped before peak picking.
- **Prominence-based picking + Blackman-Harris window.** The cardiac line is
  chosen by spectral *prominence*, not height (a breathing skirt can be taller
  but has no prominence), and the −92 dB-sidelobe window prevents the
  200×-stronger breathing line from leaking across the band.
- **Honest confidence.** Heartbeat confidence is capped at 0.85, derated by
  harmonic contamination and by motion (zero by motion level 0.8). Estimates
  below 0.3 confidence are treated as unreliable by contract.

---

## 7. Multistatic mesh fusion

A single TX→RX link cannot resolve range inside a room (§3). The mesh adds
receiver nodes on other walls, giving every location several viewing
geometries.

**Simulator side.** `RadioConfig.extra_rx_centers` adds nodes; each is an
independent radio chain with its **own** CFO/STO/AGC random walks. Frames stack
all nodes' antenna rows: `[total_rx, n_tx, n_sub]`. This faithfully models real
mesh hardware, which shares no clock between nodes.

**Processing side.** Because nodes share no clock, cross-node absolute phase is
unobservable — matched-field processing is **coherent only within a node**.
Each node runs its own Bartlett map, and the per-node likelihood maps are fused
in the **log domain** (a weighted product, weighted by each node's relative
dynamic power). A product requires a cell to score well at *every* node that
sees signal — the geometric intersection of the per-node bearing ridges, which
is exactly the point of having a mesh. With one node this reduces to the plain
Bartlett map.

Additional mesh-specific machinery:
- **Per-node noise floors.** Each node's breathing-power troughs occur at
  different times, so the *summed* power never dips to the noise level while
  each node's own power does. Floors are tracked per node.
- **Near-field masking.** Cells within ~0.5 m of the TX or any RX node have
  quasi-degenerate steering vectors and are excluded from the map.
- **Gate disabling.** The single-link bearing-suppression gate is *disabled* in
  mesh mode, since mesh maps are already point-like and the gate would wrongly
  suppress a second person sharing node 0's bearing.

**Measured win:** walker tracking median error **1.18 m → 0.59 m** (1 → 3
nodes), verified as a hard threshold in the accuracy gate.

---

## 8. Per-track spatially-filtered vitals

The room-wide vitals series mixes every occupant's chest motion; when a walker
is present it swamps a still person's breathing. The per-track approach beams
toward each person:

- At the map cadence, each confirmed track's grid cell is looked up, and the
  receiver node with the best **signal-to-interference ratio** at that cell is
  selected (not raw energy — that would pick whichever node hears the loudest
  *other* mover).
- That node's matched-filter steering vector is stored; per frame, the dynamic
  CSI is projected onto it, yielding a complex series dominated by motion *at
  that position*. Its unwrapped phase feeds the track's own breathing/heart
  estimators.
- A moving person's own gait dominates their filter, so vitals are derated by
  **track speed** rather than room motion — the whole point is that someone
  else walking must not destroy a still person's vitals.
- Extractors survive a 5 s grace period after a track vanishes, bridging
  tracker dropouts.

**Measured:** a still person's per-track breathing scores **0.019 bpm** median
error in a quiet room. With an active walker nearby, the 3-element beams (~60°
wide) cannot fully isolate the two, and the estimates **honestly flag
themselves unreliable** rather than reporting garbage.

---

## 9. Activity classification & safety alerts

### Activity (`detection/activity.py`)

Four classes from two robust features — no ML:
- **idle** — no presence.
- **micro** — present, no gross motion (breathing only).
- **walking** — sustained motion duty cycle + a persistent fast Doppler line
  (locomotion shows up as a narrow line at the body's radial velocity).
- **gesturing** — intermittent motion bursts without a persistent line.

A minimum-dwell hysteresis prevents label flapping. Measured: the walking
scenario reads 84% walking; the still scenario reads 100% micro.

### Fall detection (`detection/alerts.py::FallDetector`)

The classic radar two-phase signature: a brief **>35 Hz Doppler burst** (a
falling body produces ~70 Hz at 5 GHz; walking tops out near 31 Hz, so gait
*physically cannot* trip it) followed by **sustained stillness**. A 10 s warmup
guard ignores the pipeline's settling transient (which otherwise reads as a
phantom fall). Motion clears an active alert. **Measured: detected at ~3.7 s
latency, zero false alerts across every other scenario.**

### Breathing-cessation alarm (`detection/alerts.py::ApneaDetector`)

Credible breathing that stops while the person stays present and still. The
key insight: the estimator's spectral *confidence* lags a cessation by tens of
seconds (the Welch window still holds old breathing), but the ~15 s display
**waveform is time-ordered** — its last third collapses to noise long before
the confidence does. The alarm watches the tail/head RMS ratio. Gross motion
vetoes (gait legitimately hides breathing). **Measured: detected ~18 s after
cessation, zero false alerts.**

---

## 10. The live dashboard

A dark, control-room-aesthetic single page, vanilla JS + canvas, **zero
external dependencies** (works fully offline):

- **Alert banner** — full-width, pulsing red, unmissable; flashes the browser
  tab title; stacks multiple alerts with elapsed-time counters.
- **Status cards** — presence (big on/off), motion (level bar + detected
  badge), **activity** (glyph per state), breathing (bpm + confidence), heart
  rate, subvocal (with `experimental` tag).
- **Doppler waterfall** — scrolling spectrogram, viridis colormap, +Hz
  approaching / −Hz receding.
- **Room map** — occupancy heatmap; **track trails** (fading 20 s history);
  **per-track vitals labels** (`♥ 64 | ⌁ 14 bpm`); **multi-node RX markers**
  (clustered, labelled RX1…RXn); ground-truth "ghost" markers when simulating.
- **Vitals strip** — breathing waveform + heart-rate sparkline.
- **Timeline** — rolling 5-minute history: presence band, motion area,
  breathing line.

Auto-reconnect with backoff; graceful degradation when new fields are absent
(older server); every new field access guarded.

---

## 11. Scenarios

| name | contents |
|---|---|
| `empty` | nobody — the false-positive baseline |
| `one-still` | motionless person, 14 bpm breathing / 70 bpm heart |
| `one-walking` | one walker at 0.8 m/s with pauses |
| `two-people` | one walker + one still person |
| `breathing-demo` | `one-still` tuned for dashboard responsiveness |
| `subvocal-demo` | still person emitting speech-like micro-vibrations |
| `fall-demo` | walker falls at t=25 s and stays down |
| `apnea-demo` | still person whose breathing stops at t=40 s |

All accept `--nodes 1..4` and `--profile room.json`.

---

## 12. Verification — measured results

The accuracy gate (`scripts/check_accuracy.py`) runs 8 scenario configurations
and enforces 29 hard thresholds. **Final run: all green.**

```
  PASS  empty[n=1]          presence_fraction = 0.0          (want <= 0.1)
  PASS  empty[n=1]          motion_fraction = 0.0            (want <= 0.1)
  PASS  empty[n=1]          mean_confirmed_tracks = 0.0      (want <= 0.2)
  PASS  empty[n=1]          false_alert_states = 0           (want <= 0)
  PASS  one-still[n=1]      presence_fraction = 1.0          (want >= 0.9)
  PASS  one-still[n=1]      motion_fraction = 0.0            (want <= 0.1)
  PASS  one-still[n=1]      breathing err = 0.017 bpm        (want <= 2.0)
  PASS  one-still[n=1]      breathing reliable = 1.0         (want >= 0.8)
  PASS  one-still[n=1]      heartbeat err = 0.042 bpm        (want <= 8.0)
  PASS  one-still[n=1]      track_breathing err = 0.019 bpm  (want <= 2.0)
  PASS  one-still[n=1]      tracking pos err = 0.95 m        (want <= 1.5)
  PASS  one-still[n=1]      subvocal active = 0.0            (want <= 0.1)
  PASS  one-still[n=1]      false_alert_states = 0           (want <= 0)
  PASS  one-walking[n=1]    presence_fraction = 1.0          (want >= 0.9)
  PASS  one-walking[n=1]    tracking pos err = 1.18 m        (want <= 1.5)
  PASS  one-walking[n=1]    mean_confirmed_tracks = 1.57     (want >= 0.6)
  PASS  one-walking[n=1]    subvocal active = 0.025          (want <= 0.15)
  PASS  one-walking[n=1]    false_alert_states = 0           (want <= 0)
  PASS  one-walking[n=3]    tracking pos err = 0.59 m        (want <= 0.9)  ← mesh win
  PASS  one-walking[n=3]    false_alert_states = 0           (want <= 0)
  PASS  two-people[n=3]     mean_confirmed_tracks = 1.82     (want >= 1.2)
  PASS  two-people[n=3]     tracking pos err = 1.01 m        (want <= 1.6)
  PASS  two-people[n=3]     false_alert_states = 0           (want <= 0)
  PASS  subvocal-demo[n=1]  subvocal active = 0.82           (want >= 0.3)
  PASS  subvocal-demo[n=1]  breathing err = 0.080 bpm        (want <= 2.0)
  PASS  fall-demo[n=1]      fall_detected = True             (want == True)
  PASS  fall-demo[n=1]      false_alert_states = 0           (want <= 0)
  PASS  apnea-demo[n=1]     apnea_detected = True            (want == True)
  PASS  apnea-demo[n=1]     false_alert_states = 0           (want <= 0)

ACCURACY GATE PASSED
```

Plus **38/38 unit + system tests passing** in ~17 s.

### Throughput

| Configuration | frames/s | × realtime |
|---|---|---|
| Single node | 3,700–7,300 | 18–36× |
| 3-node mesh | 800–1,450 | 4–7× |

Comfortably faster than the 200 Hz stream in every case.

---

## 13. Operations tooling

- **`calibrate.py`** — streams an empty room, lets adaptive floors converge,
  freezes them into a JSON profile. `--profile room.json` preloads it so cold
  starts are honest even if someone is already present.
- **`check_accuracy.py`** — the sensing regression firewall. pytest proves the
  code *runs*; this proves the *sensing still works*. Any threshold violation
  exits non-zero.
- **`mqtt_bridge.py`** — publishes presence, motion, activity, people count,
  breathing/heart bpm, and alerts to MQTT with Home Assistant auto-discovery.
  **Privacy by structure:** only derived aggregates leave the process — never
  raw CSI, never individual positions.
- **`record_csi.py`** — records scenarios to `.npz`; live capture is gated
  behind an explicit `--i-have-consent` flag.
- **CI** — `.github/workflows/ci.yml` runs pytest + the quick accuracy gate on
  every push and PR.

---

## 14. Real-hardware path

Everything downstream of `CSISource` is hardware-agnostic. To go live,
implement `frames()` for your device:

- **ESP32-S3 / C6 (~$9, easiest)** — `ESP32CSISource` is **fully implemented**,
  including the serial CSV parser for ESP32-CSI-Tool. Single antenna:
  motion/presence/vitals work; mapping/tracking need an array or mesh.
- **Nexmon CSI (Raspberry Pi 4 / ASUS routers)** — `NexmonCSISource` is a
  documented stub (UDP wire format described); up to 4×4 antennas at 80 MHz —
  full mapping.
- **Intel 5300 (research card)** — `IntelCSISource` stub against the
  linux-80211n-csitool log format.

Calibration notes are in the README: real CFO is far worse than the simulator's
(keep `csi_ratio` processing), retune `background_alpha` and motion hysteresis,
and measure actual TX/RX positions into `RadioConfig` — the steering matrix is
exactly as good as that geometry.

---

## 15. Honest limitations

| Limitation | Why | Mitigation |
|---|---|---|
| Two close people defeat per-track vitals | 3-element beams ~60° wide | Estimates self-flag unreliable |
| ~1 m residual still-person range bias | Bearing-dominated single-link geometry | Multistatic mesh |
| Heartbeat marginal on real hardware | Phase-noise floor vs ~0.09 rad signal | Works in sim; honest confidence |
| Subvocalization unproven | Below commodity phase-noise floor | Flagged experimental; research scaffold |
| Sustained pacing degrades sensitivity | Adaptive floor slowly absorbs motion | Standard self-calibration trade-off |
| Everything beyond sim needs HW validation | No physical hardware yet | ESP32 path is ready |

---

## 16. Ethics & legal

WiFi sensing observes **people**, including through interior walls, without
cameras and without their devices participating. The project takes an explicit
stance:

- **Consent required** — only sense spaces whose occupants have agreed.
- **Vitals are health data** — breathing/heart traces can fall under
  HIPAA/GDPR special-category rules *regardless of sensor type*. (Note: this is
  the opposite of the common — and legally wrong — "no camera means no privacy
  regulation" claim.)
- **Law varies** — electronic-surveillance statutes in many jurisdictions cover
  passive RF sensing.
- **The simulator exists precisely** so the full stack can be developed and
  demonstrated with nobody surveilled at all.
- **Structural enforcement** — `record_csi` consent-gates live capture; the
  MQTT bridge exports only aggregates.

---

## 17. How it was built

The project was developed in phases:

1. **Foundation** — I hand-authored the shared contracts (`types.py`,
   `config.py`, the `CSISource` interface), then fanned out the five core
   modules (simulator, DSP, detectors, mapper+tracker, server+dashboard) across
   a parallel multi-agent workflow with adversarial verification.
2. **Integration & hardening** — when the workflow hit a session limit, I
   finished integration, tests, and docs inline, debugging each issue grounded
   in the physics (motion/breathing separation, heartbeat harmonic capture,
   mapper normalisation, bandwidth choice).
3. **Live previews** — stood up the dashboard on synthetic then real data with
   browser-screenshot verification.
4. **Competitive analysis** — analysed the 73k-star RuView project; its
   classical DSP independently validates this design, and its best
   architectural idea (multistatic mesh) targeted exactly the single-link
   limitation this project's ground-truth verification had *measured*.
5. **Major upgrade** — added multistatic fusion, per-track vitals, activity
   classification, fall/apnea alarms, the dashboard overhaul (delegated to a
   background agent), and ops tooling (calibration, MQTT, CI gates), with the
   full accuracy gate kept green throughout.

The disciplined verification culture — every capability scored against
simulation ground truth, an accuracy gate that fails the build on regression —
is the project's main structural advantage over comparable work.

---

## 18. Future work

- **Joint angle-Doppler mapping** — separate two people who share a bearing but
  move differently; also kills the residual phantom-track tendency.
- **Doppler-assisted track association** — velocity-aware gating.
- **Sleep-stage / posture inference** — from breathing-waveform morphology.
- **Real ESP32 mesh deployment** — validate the simulated mesh gains on
  hardware.
- **802.11bf / beamforming-feedback sensing** — use BFI from stock consumer APs
  without firmware patching.
- **Learned refinement** — a small model on top of the classical features for
  pose/activity, trained on simulator-labelled data (kept honest about the
  camera-free accuracy ceiling).

---

## 19. File manifest

```
wifi mapping/
├── pyproject.toml                  package metadata
├── README.md                       user documentation
├── PROJECT_REPORT.md               this report
├── demo_preview.py                 synthetic-data dashboard
├── .github/workflows/ci.yml        pytest + accuracy gate
├── wifi_room_radar/                      core library (~4,400 lines)
│   ├── types.py  config.py  pipeline.py  scenarios.py  server.py  __init__.py
│   ├── capture/   base, simulator, replay, hardware
│   ├── processing/ preprocessing, filters, spectrogram
│   ├── detection/  motion, presence, activity, alerts
│   ├── mapping/    room_mapper
│   ├── tracking/   tracker
│   ├── vitals/     breathing, heartbeat, subvocal
│   └── dashboard/  index.html, app.js, style.css
├── scripts/                        run_dashboard, run_headless, check_accuracy,
│                                   calibrate, record_csi, mqtt_bridge
└── tests/                          8 files, 38 tests
```

**Total: ~8,000 lines of production code, tests, and documentation.**

---

*Stack: Python 3.10+ · numpy · scipy · FastAPI · uvicorn · vanilla-JS canvas.
No machine-learning dependencies — the entire sensing chain is classical signal
processing and estimation theory, every stage explainable and every result
verified against physics-based ground truth.*
