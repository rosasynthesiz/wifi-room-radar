"""UI preview server: drives the wifi_room_radar dashboard with synthetic data.

This exists so the dashboard can be seen live before (or without) running the
full physics pipeline — it fabricates a plausible, animated SensingState on
every poll. It exercises every dashboard panel: presence, motion, doppler
waterfall, room map with tracks + ground-truth ghosts, breathing waveform,
heart rate, and subvocal bursts.

Run:  python demo_preview.py [--port 8800]
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from wifi_room_radar.config import ServerConfig
from wifi_room_radar.server import serve
from wifi_room_radar.types import SensingState, SubvocalState, TrackState, VitalSign

ROOM_W, ROOM_D = 6.0, 5.0
GRID_RES = 0.25
N_FFT, FS = 128, 200.0
BREATH_BPM, HEART_BPM = 14.2, 68.0


class SyntheticProvider:
    """Duck-typed provider generating animated demo states from wall-clock time."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.freqs = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1.0 / FS))
        self.rows = int(round(ROOM_D / GRID_RES))
        self.cols = int(round(ROOM_W / GRID_RES))
        yy, xx = np.mgrid[0:self.rows, 0:self.cols]
        self.cell_x = (xx + 0.5) * GRID_RES
        self.cell_y = (yy + 0.5) * GRID_RES
        self.rng = np.random.default_rng(7)

    @property
    def info(self) -> dict:
        return {
            "type": "SyntheticPreview (UI demo — not the physics pipeline)",
            "sample_rate": FS,
            "n_rx": 3,
            "n_tx": 1,
            "n_subcarriers": 56,
            "carrier_freq_ghz": 5.18,
            "bandwidth_mhz": 20.0,
            "room_size": [ROOM_W, ROOM_D],
            "tx_pos": [0.3, 2.5],
            "rx_positions": [[5.7, 2.44], [5.7, 2.5], [5.7, 2.56]],
        }

    def _people(self, t: float) -> list[dict]:
        # one walker on a slow lissajous loop + one still person
        wx = 3.0 + 1.7 * math.sin(0.25 * t)
        wy = 2.6 + 1.3 * math.sin(0.17 * t + 1.2)
        vx = 1.7 * 0.25 * math.cos(0.25 * t)
        vy = 1.3 * 0.17 * math.cos(0.17 * t + 1.2)
        return [
            {"x": wx, "y": wy, "vx": vx, "vy": vy, "mode": "walk",
             "breathing_bpm": 16.0, "heart_bpm": 80.0, "subvocal": False},
            {"x": 4.4, "y": 1.3, "vx": 0.0, "vy": 0.0, "mode": "still",
             "breathing_bpm": BREATH_BPM, "heart_bpm": HEART_BPM, "subvocal": True},
        ]

    def latest_state(self) -> SensingState:
        t = time.perf_counter() - self.t0
        people = self._people(t)

        # occupancy: gaussian blob per person
        grid = np.zeros((self.rows, self.cols))
        for p in people:
            d2 = (self.cell_x - p["x"]) ** 2 + (self.cell_y - p["y"]) ** 2
            grid += math.hypot(p["vx"], p["vy"]) * 0.6 + 0.55 * np.exp(-d2 / (2 * 0.45**2))
        grid = np.clip(grid / max(grid.max(), 1e-9), 0, 1)

        # doppler column: clutter ridge at 0 Hz + walker bump at its radial speed
        speed = math.hypot(people[0]["vx"], people[0]["vy"])
        fd = 2 * speed / 0.0579 * math.cos(0.4 * t)  # wandering radial component
        col = -58.0 + 2.5 * self.rng.standard_normal(N_FFT)
        col += 26.0 * np.exp(-(self.freqs - 0.0) ** 2 / (2 * 1.2**2))
        col += 34.0 * np.exp(-(self.freqs - fd) ** 2 / (2 * 2.8**2))
        col += 10.0 * np.exp(-(self.freqs - 2 * fd) ** 2 / (2 * 4.0**2))

        # breathing waveform: last 15 s at ~13 Hz
        ts = np.linspace(t - 15.0, t, 200)
        breath_wave = np.sin(2 * math.pi * BREATH_BPM / 60.0 * ts) * (0.9 + 0.1 * np.sin(0.1 * ts))

        burst = (t % 7.0) < 2.5  # speech-like subvocal bursts
        sub_score = (0.65 + 0.25 * math.sin(9 * t)) if burst else 0.06
        motion = min(1.0, max(0.0, 0.45 + 0.4 * math.sin(0.4 * t) + 0.3 * speed))

        tracks = [
            TrackState(track_id=1, x=people[0]["x"] + 0.12, y=people[0]["y"] - 0.1,
                       vx=people[0]["vx"], vy=people[0]["vy"], confidence=0.92, age=t),
            TrackState(track_id=2, x=people[1]["x"] - 0.08, y=people[1]["y"] + 0.11,
                       vx=0.0, vy=0.0, confidence=0.71, age=max(0.0, t - 4.0)),
        ]
        return SensingState(
            timestamp=round(t, 3),
            presence=True,
            motion_level=round(motion, 3),
            motion_detected=motion > 0.25,
            doppler_freqs=[float(f) for f in self.freqs],
            doppler_column=[float(c) for c in col],
            room_size=(ROOM_W, ROOM_D),
            occupancy_grid=[[float(v) for v in row] for row in grid],
            tracks=tracks,
            breathing=VitalSign(rate_bpm=BREATH_BPM, confidence=0.86,
                                waveform=[float(v) for v in breath_wave], band=(0.08, 0.6)),
            heartbeat=VitalSign(rate_bpm=HEART_BPM + 1.5 * math.sin(0.05 * t),
                                confidence=0.42, waveform=[], band=(0.8, 2.2)),
            subvocal=SubvocalState(active=burst, activity_score=round(sub_score, 3),
                                   band_energy=[round(6 + 3 * math.sin(3 * t + i), 2) for i in range(4)]),
            ground_truth={"people": people},
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    print(f"UI preview (synthetic data): http://127.0.0.1:{args.port}")
    serve(SyntheticProvider(), ServerConfig(port=args.port))
