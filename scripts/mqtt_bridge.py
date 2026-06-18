"""MQTT bridge: publish wifi_room_radar state for Home Assistant & friends.

Runs a simulated (or, later, hardware) pipeline and publishes a compact
state summary to an MQTT broker at ~1 Hz, plus Home Assistant MQTT-discovery
config messages so the entities appear automatically:

    wifi_room_radar/<id>/state         {"presence": ..., "motion_level": ..., ...}
    wifi_room_radar/<id>/alert         latest alert (retained while active)
    homeassistant/.../config     discovery payloads (retained)

Privacy by structure: only derived, aggregate quantities leave this process
(presence flag, motion level, rates, activity label, alert flags) — never
raw CSI, never positions of individual tracks. Edit ``_summary`` if you
disagree, but know what you are exporting.

Requires: pip install paho-mqtt   (optional dependency, lazily imported)
"""
from __future__ import annotations

import argparse
import json
import threading
import time

from wifi_room_radar import SCENARIOS, SensingPipeline, build_scenario


def _summary(state) -> dict:
    """Reduce a SensingState to the privacy-conscious export payload."""
    breathing = state.breathing
    heartbeat = state.heartbeat
    return {
        "timestamp": state.timestamp,
        "presence": state.presence,
        "motion_level": round(state.motion_level, 3),
        "motion_detected": state.motion_detected,
        "activity": state.activity,
        "people_count": len(state.tracks),
        "breathing_bpm": round(breathing.rate_bpm, 1) if breathing and breathing.confidence >= 0.3 else None,
        "heart_bpm": round(heartbeat.rate_bpm, 1) if heartbeat and heartbeat.confidence >= 0.3 else None,
        "alerts": [a["type"] for a in state.alerts],
    }


def _discovery_payloads(node_id: str) -> list[tuple[str, dict]]:
    """Home Assistant MQTT-discovery configs for the exported entities."""
    base = f"wifi_room_radar/{node_id}"
    device = {"identifiers": [f"wifi_room_radar_{node_id}"], "name": f"wifi_room_radar {node_id}",
              "manufacturer": "wifi_room_radar", "model": "CSI room sensor"}
    ent = lambda kind, name, extra: (  # noqa: E731
        f"homeassistant/{kind}/wifi_room_radar_{node_id}_{name}/config",
        {"name": f"wifi_room_radar {name}", "state_topic": f"{base}/state",
         "unique_id": f"wifi_room_radar_{node_id}_{name}", "device": device, **extra},
    )
    return [
        ent("binary_sensor", "presence",
            {"value_template": "{{ 'ON' if value_json.presence else 'OFF' }}",
             "device_class": "occupancy"}),
        ent("binary_sensor", "motion",
            {"value_template": "{{ 'ON' if value_json.motion_detected else 'OFF' }}",
             "device_class": "motion"}),
        ent("sensor", "activity", {"value_template": "{{ value_json.activity }}"}),
        ent("sensor", "people", {"value_template": "{{ value_json.people_count }}"}),
        ent("sensor", "breathing",
            {"value_template": "{{ value_json.breathing_bpm }}",
             "unit_of_measurement": "bpm"}),
        ent("sensor", "heart_rate",
            {"value_template": "{{ value_json.heart_bpm }}",
             "unit_of_measurement": "bpm"}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="two-people", choices=sorted(SCENARIOS))
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--node-id", default="room1")
    parser.add_argument("--rate", type=float, default=1.0, help="publishes per second")
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise SystemExit("mqtt_bridge needs paho-mqtt:  pip install paho-mqtt")

    source, cfg = build_scenario(args.scenario, realtime=True, seed=args.seed, nodes=args.nodes)
    pipeline = SensingPipeline(source, cfg.pipeline, room_size=(cfg.sim.room_width, cfg.sim.room_depth))
    stop = threading.Event()
    threading.Thread(target=pipeline.run, args=(stop,), daemon=True).start()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port)
    client.loop_start()
    for topic, payload in _discovery_payloads(args.node_id):
        client.publish(topic, json.dumps(payload), retain=True)
    state_topic = f"wifi_room_radar/{args.node_id}/state"
    alert_topic = f"wifi_room_radar/{args.node_id}/alert"
    print(f"publishing to {args.broker}:{args.port} {state_topic} at {args.rate} Hz")

    try:
        while True:
            time.sleep(1.0 / max(args.rate, 0.1))
            state = pipeline.latest_state()
            if state is None:
                continue
            client.publish(state_topic, json.dumps(_summary(state)))
            if state.alerts:
                client.publish(alert_topic, json.dumps(state.alerts[0]), retain=True)
            else:
                client.publish(alert_topic, "", retain=True)  # clear
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        source.close()
        client.loop_stop()


if __name__ == "__main__":
    main()
