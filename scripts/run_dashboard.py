"""Launch the live wifi_room_radar dashboard against a simulated scenario.

Starts the realtime simulator + sensing pipeline in a daemon thread, then
serves the web dashboard (blocking) on the requested host/port. Ctrl-C stops
the server, signals the pipeline thread, and exits cleanly.
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from wifi_room_radar import SCENARIOS, SensingPipeline, build_scenario
from wifi_room_radar.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="two-people", choices=sorted(SCENARIOS))
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nodes", type=int, default=1,
        help="receiver nodes (1 = single link, 2-4 = multistatic mesh)",
    )
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="calibration profile JSON from scripts/calibrate.py",
    )
    args = parser.parse_args()

    source, cfg = build_scenario(
        args.scenario, realtime=True, seed=args.seed, nodes=args.nodes
    )
    cfg.server.host = args.host
    cfg.server.port = args.port
    room = (cfg.sim.room_width, cfg.sim.room_depth)
    pipeline = SensingPipeline(source, cfg.pipeline, room_size=room)
    if args.profile is not None:
        pipeline.apply_profile(json.loads(args.profile.read_text()))

    stop = threading.Event()
    worker = threading.Thread(
        target=pipeline.run, args=(stop,), name="wifi_room_radar-pipeline", daemon=True
    )
    worker.start()

    print(f"Dashboard: http://{args.host}:{args.port}")
    try:
        serve(pipeline, cfg.server)  # blocks; uvicorn handles Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        source.close()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    main()
