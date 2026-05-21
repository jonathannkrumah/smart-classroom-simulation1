#!/usr/bin/env python3
"""
Bridge Arduino testbed serial data into simulation decision logic (v2).

Supports two Arduino payload styles:
1) Minimal payload (recommended):
   temperature,humidity,co2,light[,occupancy]
2) Legacy payload:
   dht_temp,lm35_temp,humidity,light,co2,green,red,white,buzzer,reason,temp_rt,light_rt,co2_rt

For each sample, this script computes model prediction + zone/fusion outputs,
writes a final CSV, and optionally forwards rows in near real-time to a
CSV feed that Streamlit can display directly.
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
from validation.hardware_test import open_serial  # noqa: E402


def normalize_serial_port(port: str) -> str:
    port = str(port).strip()
    return port.upper() if port else "COM7"


def baseline_label(features: Dict[str, float]) -> str:
    zone_state = evaluate_features_zone(features)
    overall = zone_state.get("overall_zone", "optimal")
    if overall == "optimal":
        return "conducive"
    if overall == "acceptable":
        return "acceptable"
    return "non-conducive"


@dataclass
class ArduinoTestbedSample:
    timestamp: datetime
    dht_temperature: float
    lm35_temperature: float
    humidity: float
    light: float
    co2: float
    occupancy_count: int
    green_led: str = ""
    red_led: str = ""
    white_led: str = ""
    buzzer: str = ""
    reason: str = ""
    temp_response_ms: int = 0
    light_response_ms: int = 0
    co2_response_ms: int = 0

    def as_features(self) -> Dict[str, float]:
        return {
            "temperature": self.dht_temperature,
            "humidity": self.humidity,
            "co2": self.co2,
            "light": self.light,
            "occupancy_count": self.occupancy_count,
            "occupancy": self.occupancy_count,
        }


def parse_arduino_csv_line(line: str, default_occupancy: int) -> Optional[ArduinoTestbedSample]:
    line = line.strip()
    if not line:
        return None

    tokens = [token.strip() for token in line.split(",")]
    if len(tokens) < 4:
        return None

    # Legacy payload
    if len(tokens) >= 13:
        try:
            dht_temp = float(tokens[0])
            lm35_temp = float(tokens[1])
            humidity = float(tokens[2])
            light = float(tokens[3])
            co2 = float(tokens[4])
            temp_rt = int(float(tokens[10]))
            light_rt = int(float(tokens[11]))
            co2_rt = int(float(tokens[12]))
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
            occupancy_count=default_occupancy,
            green_led=tokens[5],
            red_led=tokens[6],
            white_led=tokens[7],
            buzzer=tokens[8],
            reason=tokens[9],
            temp_response_ms=temp_rt,
            light_response_ms=light_rt,
            co2_response_ms=co2_rt,
        )

    # Minimal payload: temp,humidity,co2,light[,occupancy]
    try:
        dht_temp = float(tokens[0])
        humidity = float(tokens[1])
        co2 = float(tokens[2])
        light = float(tokens[3])
        occupancy = int(float(tokens[4])) if len(tokens) >= 5 else default_occupancy
    except (TypeError, ValueError):
        return None

    if any(math.isnan(v) for v in (dht_temp, humidity, co2, light)):
        return None

    return ArduinoTestbedSample(
        timestamp=datetime.now(),
        dht_temperature=dht_temp,
        lm35_temperature=dht_temp,
        humidity=humidity,
        light=light,
        co2=co2,
        occupancy_count=occupancy,
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


def _write_header_if_needed(path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()


def append_live_forward_row(path: str, row: Dict[str, object]) -> None:
    fieldnames = [
        "timestamp",
        "temperature",
        "humidity",
        "co2",
        "co2_raw_line",
        "co2_used_for_model",
        "light",
        "occupancy_count",
        "model_prediction",
        "confidence",
        "overall_zone",
        "final_status",
        "disagreement",
        "rationale",
        "baseline_prediction",
        "agreement_score",
        "recommendations",
        "status",
        "source",
    ]
    _write_header_if_needed(path, fieldnames)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


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
        "model_prediction",
        "model_confidence",
        "baseline_prediction",
        "overall_zone",
        "final_status",
        "model_zone_disagreement",
        "decision_rationale",
        "agreement_score",
        "bridge_recommendations",
        "acceptable_factors",
        "non_conducive_factors",
        "raw_line",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved bridge log to: {output_path}")


def run_bridge(args: argparse.Namespace) -> int:
    serial_conn = open_serial(args.port, args.baud, timeout=args.timeout)
    print(f"Connected to serial device at {normalize_serial_port(args.port)} @ {args.baud} baud")

    records: List[Dict[str, object]] = []
    started = time.time()

    if args.reset_live_feed and args.forward_live and os.path.exists(args.forward_live):
        os.remove(args.forward_live)

    try:
        while (time.time() - started) < args.duration:
            ts_payload = datetime.now().isoformat()
            serial_conn.write((ts_payload + "\n").encode("utf-8"))

            raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            sample = parse_arduino_csv_line(raw, default_occupancy=args.occupancy_count)
            if sample is None:
                print(f"Skipping malformed Arduino line: {raw}")
                time.sleep(max(args.interval, 0.05))
                continue

            features = sample.as_features()
            raw_co2_line = None
            try:
                raw_co2_line = float(raw.split(",")[2].strip())
            except Exception:
                raw_co2_line = None

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
            baseline = baseline_label(features)
            recs = bridge_recommendations(features, zone_state)

            status = "OK" if fused["final_status"] == "conducive" else "ALERT"
            print(
                f"[{sample.timestamp.strftime('%H:%M:%S')}] {status:<5} "
                f"T={sample.dht_temperature:.1f}C H={sample.humidity:.1f}% "
                f"CO2={sample.co2:.1f}ppm L={sample.light:.1f}lux "
                f"zone={zone_state['overall_zone']:<13} final={fused['final_status']:<13} "
                f"model={prediction:<13} conf={float(confidence):.1%} agreement={agreement_score:.2f}"
            )

            row = {
                "timestamp": sample.timestamp.isoformat(),
                "dht_temperature": round(sample.dht_temperature, 3),
                "lm35_temperature": round(sample.lm35_temperature, 3),
                "humidity": round(sample.humidity, 3),
                "co2": round(sample.co2, 3),
                "co2_raw_line": round(float(raw_co2_line), 3) if raw_co2_line is not None else None,
                "co2_used_for_model": round(float(features["co2"]), 3),
                "light": round(sample.light, 3),
                "occupancy_count": sample.occupancy_count,
                "model_prediction": prediction,
                "model_confidence": round(float(confidence), 6),
                "baseline_prediction": baseline,
                "overall_zone": zone_state.get("overall_zone", "unknown"),
                "final_status": fused.get("final_status", "unknown"),
                "model_zone_disagreement": bool(fused.get("disagreement", False)),
                "decision_rationale": fused.get("rationale", ""),
                "agreement_score": round(float(agreement_score), 6),
                "bridge_recommendations": ";".join(recs),
                "acceptable_factors": "; ".join(zone_state.get("acceptable_factors", [])),
                "non_conducive_factors": "; ".join(zone_state.get("non_conducive_factors", [])),
                "raw_line": raw,
            }
            records.append(row)

            if args.forward_live:
                append_live_forward_row(
                    args.forward_live,
                    {
                        "timestamp": row["timestamp"],
                        "temperature": row["dht_temperature"],
                        "humidity": row["humidity"],
                        "co2": row["co2"],
                        "co2_raw_line": row["co2_raw_line"],
                        "co2_used_for_model": row["co2_used_for_model"],
                        "light": row["light"],
                        "occupancy_count": row["occupancy_count"],
                        "model_prediction": row["model_prediction"],
                        "confidence": row["model_confidence"],
                        "overall_zone": row["overall_zone"],
                        "final_status": row["final_status"],
                        "disagreement": row["model_zone_disagreement"],
                        "rationale": row["decision_rationale"],
                        "baseline_prediction": row["baseline_prediction"],
                        "agreement_score": row["agreement_score"],
                        "recommendations": row["bridge_recommendations"],
                        "status": "ok",
                        "source": "bridge",
                    },
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
    parser = argparse.ArgumentParser(description="Bridge Arduino testbed data into simulation decision logic (v2)")
    default_live_feed = str(ROOT_DIR / "validation" / "live_bridge_feed.csv")
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
        "--forward-live",
        type=str,
        nargs="?",
        const=default_live_feed,
        default=None,
        metavar="PATH",
        help=(
            "Enable live-forward feed for dashboard. "
            f"Use without value to default to {default_live_feed}, or provide a custom PATH."
        ),
    )
    parser.add_argument("--reset-live-feed", action="store_true", help="Reset live-forward CSV at start")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "validation" / "testbed_simulation_bridge_v2.csv"),
        help="Final CSV output path",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_bridge(args)


if __name__ == "__main__":
    raise SystemExit(main())
