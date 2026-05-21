#!/usr/bin/env python3
"""
Bridge Arduino testbed serial data into simulation decision logic.

This script is designed for the provided Arduino firmware that:
1) waits for a timestamp line from serial, then
2) emits one CSV line with sensor and actuator states.

It evaluates each sample with the same simulation layers:
- ML prediction
- Comfort/attention zone classification
- Decision fusion

It logs all records to CSV for Excel analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation.classroom_sim import (  # noqa: E402
    ATTENTION_THRESHOLDS,
    COMFORT_THRESHOLDS,
    compute_agreement_score,
    evaluate_features_zone,
    fuse_model_zone_status,
)
from simulation.ml_integration import predict_environment  # noqa: E402
from validation.hardware_test import normalize_serial_port, open_serial  # noqa: E402


@dataclass
class ArduinoTestbedSample:
    timestamp: datetime
    dht_temperature: float
    lm35_temperature: float
    humidity: float
    light: float
    co2: float
    green_led: str
    red_led: str
    white_led: str
    buzzer: str
    reason: str
    temp_response_ms: int
    light_response_ms: int
    co2_response_ms: int

    def as_features(self, occupancy_count: int) -> Dict[str, float]:
        return {
            "temperature": self.dht_temperature,
            "humidity": self.humidity,
            "co2": self.co2,
            "light": self.light,
            "occupancy_count": occupancy_count,
            "occupancy": occupancy_count,
        }


def parse_arduino_csv_line(line: str) -> Optional[ArduinoTestbedSample]:
    line = line.strip()
    if not line:
        return None

    tokens = [token.strip() for token in line.split(",")]
    if len(tokens) < 13:
        return None

    try:
        dht_temp = float(tokens[0])
        lm35_temp = float(tokens[1])
        humidity = float(tokens[2])
        light = float(tokens[3])
        co2 = float(tokens[4])

        green_led = tokens[5]
        red_led = tokens[6]
        white_led = tokens[7]
        buzzer = tokens[8]
        reason = tokens[9]

        temp_response_ms = int(float(tokens[10]))
        light_response_ms = int(float(tokens[11]))
        co2_response_ms = int(float(tokens[12]))
    except (TypeError, ValueError):
        return None

    if any(math.isnan(v) for v in (dht_temp, lm35_temp, humidity, light, co2)):
        return None

    return ArduinoTestbedSample(
        timestamp=datetime.now(),
        dht_temperature=dht_temp,
        lm35_temperature=lm35_temp,
        humidity=humidity,
        light=light,
        co2=co2,
        green_led=green_led,
        red_led=red_led,
        white_led=white_led,
        buzzer=buzzer,
        reason=reason,
        temp_response_ms=temp_response_ms,
        light_response_ms=light_response_ms,
        co2_response_ms=co2_response_ms,
    )


def bridge_recommendations(features: Dict[str, float], zone_state: Dict[str, object]) -> List[str]:
    recs: List[str] = []

    temperature = float(features["temperature"])
    humidity = float(features["humidity"])
    co2 = float(features["co2"])
    light = float(features["light"])

    if co2 > COMFORT_THRESHOLDS["co2"]["high"]:
        recs.append("VENTILATION_HIGH")
    elif co2 > ATTENTION_THRESHOLDS["co2"]["high"]:
        recs.append("VENTILATION_LOW")

    if temperature > COMFORT_THRESHOLDS["temperature"]["high"]:
        recs.append("COOLING_HIGH")
    elif temperature > ATTENTION_THRESHOLDS["temperature"]["high"]:
        recs.append("COOLING_LOW")
    elif temperature < COMFORT_THRESHOLDS["temperature"]["low"]:
        recs.append("HEATING_HIGH")
    elif temperature < ATTENTION_THRESHOLDS["temperature"]["low"]:
        recs.append("HEATING_LOW")

    if humidity > COMFORT_THRESHOLDS["humidity"]["high"]:
        recs.append("DEHUMIDIFIER_ON")
    elif humidity < ATTENTION_THRESHOLDS["humidity"]["low"]:
        recs.append("HUMIDIFIER_ON")

    if light < ATTENTION_THRESHOLDS["light"]["low"]:
        recs.append("LIGHTS_ON")
    elif light > ATTENTION_THRESHOLDS["light"]["high"]:
        recs.append("BLINDS_ADJUST")

    if zone_state.get("overall_zone") == "optimal" and not recs:
        recs.append("NO_ACTION")

    return recs


def summarize(records: List[Dict[str, object]]) -> None:
    if not records:
        print("No records collected.")
        return

    total = len(records)
    final_conducive = sum(1 for row in records if row["final_status"] == "conducive")
    disagreements = sum(1 for row in records if bool(row["model_zone_disagreement"]))
    avg_conf = statistics.fmean(float(row["model_confidence"]) for row in records)
    avg_agreement = statistics.fmean(float(row["agreement_score"]) for row in records)

    print("\n" + "=" * 72)
    print("TESTBED -> SIMULATION BRIDGE SUMMARY")
    print("=" * 72)
    print(f"Samples                       : {total}")
    print(f"Final conducive               : {final_conducive}/{total} ({final_conducive / total:.1%})")
    print(f"Model-zone disagreements      : {disagreements}/{total} ({disagreements / total:.1%})")
    print(f"Average model confidence      : {avg_conf:.1%}")
    print(f"Average agreement score       : {avg_agreement:.2f}")


def export_csv(records: List[Dict[str, object]], output_path: str) -> None:
    if not records:
        return

    fieldnames = [
        "timestamp",
        "dht_temperature",
        "lm35_temperature",
        "humidity",
        "co2",
        "light",
        "occupancy_count",
        "arduino_green_led",
        "arduino_red_led",
        "arduino_white_led",
        "arduino_buzzer",
        "arduino_reason",
        "arduino_temp_response_ms",
        "arduino_light_response_ms",
        "arduino_co2_response_ms",
        "model_prediction",
        "model_confidence",
        "overall_zone",
        "final_status",
        "model_zone_disagreement",
        "decision_rationale",
        "agreement_score",
        "acceptable_factors",
        "non_conducive_factors",
        "bridge_recommendations",
        "bridge_timestamp_sent",
        "raw_line",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved bridge log to: {output_path}")


def run_bridge(args: argparse.Namespace) -> int:
    if not args.port:
        print("Error: --port is required.")
        return 2

    serial_conn = open_serial(args.port, args.baud, timeout=args.timeout)
    print(f"Connected to serial device at {normalize_serial_port(args.port)} @ {args.baud} baud")

    records: List[Dict[str, object]] = []
    started = time.time()
    max_runtime = args.duration

    try:
        while (time.time() - started) < max_runtime:
            now = datetime.now()
            ts_payload = now.isoformat()
            serial_conn.write((ts_payload + "\n").encode("utf-8"))

            raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            sample = parse_arduino_csv_line(raw)
            if sample is None:
                print(f"Skipping malformed Arduino line: {raw}")
                time.sleep(max(args.interval, 0.05))
                continue

            features = sample.as_features(occupancy_count=args.occupancy_count)
            prediction, confidence = predict_environment(
                features,
                context={
                    "datetime": sample.timestamp,
                    "current_minute": int(time.time() - started) // 60,
                    "room_size": args.room_size,
                    "start_hour": args.start_hour,
                },
            )

            zone_state = evaluate_features_zone(features)
            fused = fuse_model_zone_status(
                prediction,
                zone_state,
                confidence,
                low_confidence=args.low_confidence,
            )
            agreement_score = compute_agreement_score(prediction, zone_state)
            recs = bridge_recommendations(features, zone_state)

            status = "OK" if fused["final_status"] == "conducive" else "ALERT"
            print(
                f"[{sample.timestamp.strftime('%H:%M:%S')}] {status:<5} "
                f"zone={zone_state['overall_zone']:<13} final={fused['final_status']:<13} "
                f"model={prediction:<13} conf={float(confidence):.1%} "
                f"agreement={agreement_score:.2f}"
            )

            records.append(
                {
                    "timestamp": sample.timestamp.isoformat(),
                    "dht_temperature": round(sample.dht_temperature, 3),
                    "lm35_temperature": round(sample.lm35_temperature, 3),
                    "humidity": round(sample.humidity, 3),
                    "co2": round(sample.co2, 3),
                    "light": round(sample.light, 3),
                    "occupancy_count": args.occupancy_count,
                    "arduino_green_led": sample.green_led,
                    "arduino_red_led": sample.red_led,
                    "arduino_white_led": sample.white_led,
                    "arduino_buzzer": sample.buzzer,
                    "arduino_reason": sample.reason,
                    "arduino_temp_response_ms": sample.temp_response_ms,
                    "arduino_light_response_ms": sample.light_response_ms,
                    "arduino_co2_response_ms": sample.co2_response_ms,
                    "model_prediction": prediction,
                    "model_confidence": round(float(confidence), 6),
                    "overall_zone": zone_state.get("overall_zone", "unknown"),
                    "final_status": fused.get("final_status", "unknown"),
                    "model_zone_disagreement": bool(fused.get("disagreement", False)),
                    "decision_rationale": fused.get("rationale", ""),
                    "agreement_score": round(float(agreement_score), 6),
                    "acceptable_factors": "; ".join(zone_state.get("acceptable_factors", [])),
                    "non_conducive_factors": "; ".join(zone_state.get("non_conducive_factors", [])),
                    "bridge_recommendations": ";".join(recs),
                    "bridge_timestamp_sent": ts_payload,
                    "raw_line": raw,
                }
            )

            time.sleep(max(args.interval, 0.05))

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        serial_conn.close()

    summarize(records)
    export_csv(records, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Arduino testbed data into simulation decision logic")
    parser.add_argument("--port", type=str, default="COM7", help="Serial port, e.g. COM7")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout in seconds")
    parser.add_argument("--duration", type=int, default=180, help="Run duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Loop interval in seconds")
    parser.add_argument("--room-size", type=int, default=100, help="Room size context for model")
    parser.add_argument("--start-hour", type=int, default=datetime.now().hour, help="Start hour context for model")
    parser.add_argument("--occupancy-count", type=int, default=30, help="Occupancy context injected into model")
    parser.add_argument("--low-confidence", type=float, default=0.6, help="Low confidence threshold for fusion")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "validation" / "testbed_simulation_bridge.csv"),
        help="CSV output path",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_bridge(args)


if __name__ == "__main__":
    raise SystemExit(main())
