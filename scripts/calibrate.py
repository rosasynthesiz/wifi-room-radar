"""Learn-this-room calibration: capture an empty-room baseline profile.

Run this once with the room EMPTY (or against a recording of the empty
room). It streams CSI for ``--duration`` seconds, lets every adaptive stage
(background EMA, motion noise floor, mapper per-node floors) converge with
nobody present, then freezes the result into a JSON profile:

    .venv/Scripts/python scripts/calibrate.py --duration 30 --out room.json

A profile makes cold starts honest: without one, the detectors must guess
their noise floors from the first seconds of data — which mis-calibrates if
someone is already in the room when the pipeline starts. `run_dashboard.py`
and `run_headless.py` accept ``--profile room.json`` to preload it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wifi_room_radar import SCENARIOS, SensingPipeline, build_scenario


def calibrate(scenario: str, duration: float, seed: int, nodes: int) -> dict:
    """Stream an (assumed empty) scenario; return the converged-floor profile."""
    source, cfg = build_scenario(scenario, realtime=False, seed=seed, nodes=nodes)
    pipeline = SensingPipeline(
        source, cfg.pipeline, room_size=(cfg.sim.room_width, cfg.sim.room_depth)
    )
    n = int(duration * source.radio.sample_rate)
    for i, frame in enumerate(source.frames()):
        if i >= n:
            break
        pipeline.step_frame(frame)
    source.close()

    mapper = pipeline._mapper
    motion = pipeline._motion
    profile = {
        "created_unix": time.time(),
        "duration_s": float(duration),
        "n_nodes": int(source.radio.n_nodes),
        "sample_rate": float(source.radio.sample_rate),
        # Converged adaptive floors (the calibration payload).
        "motion_floor": motion._floor,
        "motion_median": motion._median,
        "mapper_noise_floor": (
            mapper._noise_floor.tolist() if mapper._noise_floor is not None else None
        ),
        # Environment fingerprint, useful for "has the room changed?" checks.
        "mean_grid_power": float(np.mean(mapper.grid)),
    }
    return profile


def apply_profile(pipeline: SensingPipeline, profile: dict) -> None:
    """Preload a pipeline's adaptive floors from a saved profile.

    Thin wrapper kept for symmetry; the logic lives on
    :meth:`wifi_room_radar.SensingPipeline.apply_profile`.
    """
    pipeline.apply_profile(profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", default="empty", choices=sorted(SCENARIOS),
        help="source to calibrate against (use 'empty'; sim-only for now)",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("room-profile.json"))
    args = parser.parse_args()

    profile = calibrate(args.scenario, args.duration, args.seed, args.nodes)
    args.out.write_text(json.dumps(profile, indent=2))
    print(f"profile written: {args.out}")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
