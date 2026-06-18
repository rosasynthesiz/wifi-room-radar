"""Accuracy gate: run the scenario suite and FAIL if any sensing metric regresses.

This is the project's regression firewall — pytest proves the code runs;
this proves the *sensing* still works. Each scenario is scored against the
simulator's ground truth and checked against hard thresholds; any violation
exits non-zero (CI-friendly). Run it after any change to capture, processing,
detection, mapping, tracking, or vitals code:

    .venv/Scripts/python scripts/check_accuracy.py            # full gate (~3 min)
    .venv/Scripts/python scripts/check_accuracy.py --quick    # reduced durations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_headless import run  # noqa: E402  (sibling script import)

# (scenario, duration_s, nodes, [(metric_path, op, threshold), ...])
# Thresholds are deliberately looser than typical measured values (see
# README) so seed-to-seed jitter does not flake CI, while real regressions
# (broken estimator, dead tracker, false-alarm storm) always trip.
GATES = [
    ("empty", 40, 1, [
        ("presence_fraction", "<=", 0.10),
        ("motion_fraction", "<=", 0.10),
        ("tracking.mean_confirmed_tracks", "<=", 0.2),
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("one-still", 60, 1, [
        ("presence_fraction", ">=", 0.90),
        ("motion_fraction", "<=", 0.10),
        ("breathing.median_abs_err_bpm", "<=", 2.0),
        ("breathing.reliable_fraction", ">=", 0.8),
        ("heartbeat.median_abs_err_bpm", "<=", 8.0),
        ("track_breathing.median_abs_err_bpm", "<=", 2.0),
        ("tracking.median_pos_err_m", "<=", 1.5),
        ("subvocal.active_fraction", "<=", 0.1),
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("one-walking", 60, 1, [
        ("presence_fraction", ">=", 0.90),
        ("tracking.median_pos_err_m", "<=", 1.5),
        ("tracking.mean_confirmed_tracks", ">=", 0.6),
        ("subvocal.active_fraction", "<=", 0.15),
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("one-walking", 60, 3, [
        ("tracking.median_pos_err_m", "<=", 0.9),  # the multistatic win
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("two-people", 60, 3, [
        ("tracking.mean_confirmed_tracks", ">=", 1.2),
        ("tracking.median_pos_err_m", "<=", 1.6),
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("subvocal-demo", 40, 1, [
        ("subvocal.active_fraction", ">=", 0.3),
        ("breathing.median_abs_err_bpm", "<=", 2.0),
    ]),
    ("fall-demo", 45, 1, [
        ("alerts.fall_detected", "==", True),
        ("alerts.false_alert_states", "<=", 0),
    ]),
    ("apnea-demo", 80, 1, [
        ("alerts.apnea_detected", "==", True),
        ("alerts.false_alert_states", "<=", 0),
    ]),
]

OPS = {
    "<=": lambda v, t: v is not None and v <= t,
    ">=": lambda v, t: v is not None and v >= t,
    "==": lambda v, t: v == t,
}


def _lookup(metrics: dict, path: str):
    node = metrics
    for key in path.split("."):
        if node is None:
            return None
        node = node.get(key)
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="halve durations (smoke gate)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    failures: list[str] = []
    for scenario, duration, nodes, checks in GATES:
        dur = max(30, duration // 2) if args.quick else duration
        metrics = run(scenario, float(dur), args.seed, nodes=nodes)
        tag = f"{scenario}[n={nodes}]"
        for path, op, threshold in checks:
            value = _lookup(metrics, path)
            ok = OPS[op](value, threshold)
            mark = "PASS" if ok else "FAIL"
            print(f"  {mark}  {tag:24s} {path} = {value!r}  (want {op} {threshold!r})")
            if not ok:
                failures.append(f"{tag}: {path} = {value!r}, want {op} {threshold!r}")
        if metrics.get("proc_fps", 0) < 200:
            failures.append(f"{tag}: proc_fps {metrics['proc_fps']:.0f} < 200 (realtime broken)")

    print()
    if failures:
        print(f"ACCURACY GATE FAILED — {len(failures)} violation(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ACCURACY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
