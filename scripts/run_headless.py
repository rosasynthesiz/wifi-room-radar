"""Offline accuracy harness: run a scenario through the pipeline, score it.

Pulls non-realtime simulated frames for ``--duration`` simulated seconds,
steps :class:`wifi_room_radar.SensingPipeline` frame by frame, collects every
published :class:`~wifi_room_radar.types.SensingState`, then prints a metrics dict
(pretty by default, raw JSON with ``--json``).

Metrics are computed honestly against the simulator's per-frame ground truth:
vitals are scored only on post-warmup states and only when a still person
actually exists; an empty room must show a LOW presence fraction.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from wifi_room_radar import SCENARIOS, SensingPipeline, build_scenario

#: Seconds of stream discarded before accuracy metrics are scored (filters,
#: noise floors and vitals buffers are still converging before this).
WARMUP_SEC = 20.0


def run(
    scenario: str, duration: float, seed: int, nodes: int = 1,
    profile: dict | None = None,
) -> dict:
    """Run one scenario offline and return the metrics dict."""
    source, cfg = build_scenario(scenario, realtime=False, seed=seed, nodes=nodes)
    room = (cfg.sim.room_width, cfg.sim.room_depth)
    pipeline = SensingPipeline(source, cfg.pipeline, room_size=room)
    if profile is not None:
        pipeline.apply_profile(profile)

    states = []
    n_frames = 0
    t_wall = time.perf_counter()
    for frame in source.frames():
        if frame.timestamp >= duration:
            break
        state = pipeline.step_frame(frame)
        n_frames += 1
        if state is not None:
            states.append(state)
    elapsed = time.perf_counter() - t_wall
    source.close()

    metrics = compute_metrics(scenario, duration, n_frames, elapsed, states)
    metrics["n_nodes"] = int(nodes)
    return metrics


def _truth_people(states: list) -> list[dict]:
    """Ground-truth person list from the last state that carries one."""
    for state in reversed(states):
        gt = state.ground_truth
        if gt and gt.get("people") is not None:
            return gt["people"]
    return []


#: Estimates below this confidence are excluded from the rate-accuracy score.
#: This matches the reliability floor documented on
#: :class:`wifi_room_radar.types.VitalSign`: the estimators themselves flag
#: estimates as unreliable (e.g. while gross motion swamps the vitals band),
#: and a consumer that ignores the flag gets garbage by contract. The
#: ``reliable_fraction`` field reports how often a usable estimate existed.
CONF_FLOOR = 0.3


def _vital_metrics(states: list, attr: str, truth_key: str, truth: list[dict]) -> dict | None:
    """Score one vital sign over post-warmup states; None if no still person."""
    still = [p for p in truth if p.get("mode") == "still"]
    if not still:
        return None
    truth_bpm = float(still[0][truth_key])
    ests, confs = [], []
    for state in states:
        vs = getattr(state, attr)
        if vs is not None:
            confs.append(float(vs.confidence))
            if vs.confidence >= CONF_FLOOR:
                ests.append(float(vs.rate_bpm))
    if not ests:
        return {
            "truth_bpm": truth_bpm,
            "median_est_bpm": None,
            "median_abs_err_bpm": None,
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "reliable_fraction": 0.0,
        }
    ests_arr = np.asarray(ests)
    return {
        "truth_bpm": truth_bpm,
        "median_est_bpm": float(np.median(ests_arr)),
        "median_abs_err_bpm": float(np.median(np.abs(ests_arr - truth_bpm))),
        "mean_confidence": float(np.mean(confs)),
        "reliable_fraction": float(len(ests) / max(1, len(states))),
    }


def _track_vital_metrics(states: list, truth: list[dict]) -> dict | None:
    """Score the PER-TRACK breathing of the still person (spatial filtering).

    Finds, in each state, the confirmed track nearest the still truth person
    (within 1.5 m) and scores that track's own breathing estimate — the
    quantity the room-wide estimate cannot deliver once a second person
    moves around.
    """
    still = [p for p in truth if p.get("mode") == "still"]
    if not still:
        return None
    truth_bpm = float(still[0]["breathing_bpm"])
    sx, sy = float(still[0]["x"]), float(still[0]["y"])
    ests, confs = [], []
    for state in states:
        best, best_d = None, 1.5
        for tr in state.tracks:
            d = math.hypot(tr.x - sx, tr.y - sy)
            if d < best_d:
                best, best_d = tr, d
        if best is not None and best.breathing is not None:
            confs.append(float(best.breathing.confidence))
            if best.breathing.confidence >= CONF_FLOOR:
                ests.append(float(best.breathing.rate_bpm))
    if not ests:
        return {
            "truth_bpm": truth_bpm,
            "median_est_bpm": None,
            "median_abs_err_bpm": None,
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "reliable_fraction": 0.0,
        }
    arr = np.asarray(ests)
    return {
        "truth_bpm": truth_bpm,
        "median_est_bpm": float(np.median(arr)),
        "median_abs_err_bpm": float(np.median(np.abs(arr - truth_bpm))),
        "mean_confidence": float(np.mean(confs)),
        "reliable_fraction": float(len(ests) / max(1, len(states))),
    }


def _alert_metrics(states: list) -> dict:
    """Detection + latency for fall / breathing-stopped events vs ground truth."""
    fall_truth_t = apnea_truth_t = None
    for s in states:
        for p in (s.ground_truth or {}).get("people") or []:
            if p.get("fallen") and fall_truth_t is None:
                fall_truth_t = s.timestamp
            if p.get("breathing") is False and apnea_truth_t is None:
                apnea_truth_t = s.timestamp

    def first_alert(kind: str) -> float | None:
        for s in states:
            if any(a.get("type") == kind for a in (s.alerts or [])):
                return s.timestamp
        return None

    fall_t = first_alert("fall")
    apnea_t = first_alert("breathing_stopped")
    # Any alert active before its truth event (or in a scenario without one)
    # is a false positive state.
    false_states = 0
    for s in states:
        for a in s.alerts or []:
            truth_t = fall_truth_t if a.get("type") == "fall" else apnea_truth_t
            if truth_t is None or s.timestamp < truth_t:
                false_states += 1
    return {
        "fall_truth_t": fall_truth_t,
        "fall_detected": fall_t is not None,
        "fall_latency_s": (fall_t - fall_truth_t) if fall_t and fall_truth_t else None,
        "apnea_truth_t": apnea_truth_t,
        "apnea_detected": apnea_t is not None,
        "apnea_latency_s": (apnea_t - apnea_truth_t) if apnea_t and apnea_truth_t else None,
        "false_alert_states": false_states,
    }


def _tracking_metrics(states: list) -> dict:
    """Mean confirmed-track count and median truth-to-nearest-track error."""
    n_truth = 0
    track_counts = []
    errors = []
    for state in states:
        gt_people = (state.ground_truth or {}).get("people") or []
        n_truth = max(n_truth, len(gt_people))
        track_counts.append(len(state.tracks))
        if not gt_people or not state.tracks:
            continue
        txy = np.asarray([[tr.x, tr.y] for tr in state.tracks])
        for person in gt_people:
            d = np.hypot(txy[:, 0] - person["x"], txy[:, 1] - person["y"])
            errors.append(float(np.min(d)))
    if n_truth == 0:
        return {
            "n_truth_people": 0,
            "mean_confirmed_tracks": float(np.mean(track_counts)) if track_counts else 0.0,
            "median_pos_err_m": None,
        }
    return {
        "n_truth_people": n_truth,
        "mean_confirmed_tracks": float(np.mean(track_counts)) if track_counts else 0.0,
        "median_pos_err_m": float(np.median(errors)) if errors else None,
    }


def compute_metrics(
    scenario: str, duration: float, n_frames: int, elapsed: float, states: list
) -> dict:
    """Assemble the full metrics dict from collected states."""
    post = [s for s in states if s.timestamp >= WARMUP_SEC]
    truth = _truth_people(states)

    # Presence/motion are steady-state behaviours: scored post-warmup like
    # everything else (the first WARMUP_SEC are spent filling vitals buffers
    # and noise floors, during which a presence "miss" is by design).
    presence_fraction = (
        float(np.mean([s.presence for s in post])) if post else 0.0
    )
    motion_fraction = (
        float(np.mean([s.motion_detected for s in post])) if post else 0.0
    )

    breathing = _vital_metrics(post, "breathing", "breathing_bpm", truth)
    heartbeat = _vital_metrics(post, "heartbeat", "heart_bpm", truth)
    tracking = _tracking_metrics(post) if truth else {
        "n_truth_people": 0,
        "mean_confirmed_tracks": float(np.mean([len(s.tracks) for s in post])) if post else 0.0,
        "median_pos_err_m": None,
    }

    sub_states = [s.subvocal for s in post if s.subvocal is not None]
    active_fraction = (
        float(np.mean([sv.active for sv in sub_states])) if sub_states else 0.0
    )

    labels = [s.activity for s in post]
    activity_fractions = {
        label: float(labels.count(label) / len(labels)) for label in sorted(set(labels))
    } if labels else {}

    return {
        "scenario": scenario,
        "duration_s": float(duration),
        "frames_processed": int(n_frames),
        "proc_fps": float(n_frames / elapsed) if elapsed > 0 else math.inf,
        "presence_fraction": presence_fraction,
        "motion_fraction": motion_fraction,
        "breathing": breathing,
        "heartbeat": heartbeat,
        "track_breathing": _track_vital_metrics(post, truth),
        "tracking": tracking,
        "subvocal": {"active_fraction": active_fraction},
        "activity": activity_fractions,
        "alerts": _alert_metrics(states),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="two-people", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--duration", type=float, default=60.0, help="simulated seconds to process"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nodes", type=int, default=1,
        help="receiver nodes (1 = single link, 2-4 = multistatic mesh)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print raw JSON instead of pretty output"
    )
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="calibration profile JSON from scripts/calibrate.py",
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text()) if args.profile else None
    metrics = run(
        args.scenario, args.duration, args.seed, nodes=args.nodes, profile=profile
    )
    if args.json:
        print(json.dumps(metrics))
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
