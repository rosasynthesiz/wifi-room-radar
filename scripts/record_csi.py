"""Record a simulated CSI stream to a .npz file for later replay.

Generates ``--duration`` simulated seconds of a scenario as fast as possible
(non-realtime) and saves them with :func:`wifi_room_radar.capture.replay.save_recording`.
Play back with ``wifi_room_radar.capture.replay.ReplayCSISource``.

Privacy: persisted CSI is a recording of how people moved, breathed, and
behaved in a space — treat it like camera footage. Recording a LIVE (non
simulated) source therefore requires the explicit ``--i-have-consent`` flag,
asserting that everyone in the sensed space has agreed to the recording.
Simulated scenarios involve nobody and need no flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wifi_room_radar import SCENARIOS, build_scenario
from wifi_room_radar.capture.replay import save_recording

#: Scenario names are simulator-only today; flip to True if this script is
#: ever pointed at live hardware.
LIVE_CAPTURE = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="two-people", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--duration", type=float, default=60.0, help="simulated seconds to record"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=Path("recording.npz"), help="output .npz file"
    )
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="assert that every person in the sensed space consented to this "
        "recording (required for live-hardware capture; unused for simulation)",
    )
    args = parser.parse_args()

    if LIVE_CAPTURE and not args.i_have_consent:
        print(
            "Refusing to record a live space without --i-have-consent.\n"
            "CSI recordings capture how real people move and breathe; get\n"
            "everyone's agreement first (see README: Ethics & legal).",
            file=sys.stderr,
        )
        sys.exit(2)

    source, _cfg = build_scenario(
        args.scenario, realtime=False, seed=args.seed, nodes=args.nodes
    )
    frames = []
    for frame in source.frames():
        if frame.timestamp >= args.duration:
            break
        frames.append(frame)
    source.close()

    path = save_recording(frames, args.out)
    size_mb = path.stat().st_size / 1e6
    print(
        f"Saved {len(frames)} frames ({args.duration:g} s of '{args.scenario}', "
        f"seed {args.seed}) to {path} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
