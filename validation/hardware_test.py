#!/usr/bin/env python3
"""
Hardware-in-the-loop validation for smart classroom model.
Now aligned with simulation three-zone + ML fusion framework.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation.ml_integration import predict_environment  # noqa: E402

# Import SAME logic from simulation (critical for consistency)
from simulation.classroom_sim import (
    evaluate_features_zone,
    fuse_model_zone_status,
    compute_agreement_score,
)

REQUIRED_FEATURES = ("temperature", "humidity", "co2", "light")


def normalize_serial_port(port: str) -> str:
    port = str(port).strip()
    if not port:
        return "COM7"
    # Normalize common typo like COMP7 -> COM7.
    return port.upper().replace("COMP", "COM")


def apply_refined_calibration(features: Dict[str, float]) -> Dict[str, float]:
    """Apply the same refined calibration used in the dashboard HIL pipeline."""
    calibrated = dict(features)
    calibrated["temperature"] = float(features["temperature"]) - 1.0
    calibrated["humidity"] = float(features["humidity"]) * 0.6
    calibrated["co2"] = float(features["co2"]) - 120.0
    calibrated_light = float(features["light"]) * 0.8
    calibrated["light"] = max(300.0, min(650.0, calibrated_light))
    return calibrated


# -----------------------------
# DATA STRUCTURE
# -----------------------------
@dataclass
class SensorSample:
    timestamp: datetime
    temperature: float
    humidity: float
    co2: float
    light: float
    occupancy_count: int = 30

    def as_features(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "co2": self.co2,
            "light": self.light,
            "occupancy_count": self.occupancy_count,
        }


# -----------------------------
# SERIAL / MOCK PARSING
# -----------------------------
def parse_sample_line(line: str) -> Optional[SensorSample]:
    line = line.strip()
    if not line:
        return None

    payload: Dict[str, object]

    if line.startswith("{"):
        payload = json.loads(line)
    elif "=" in line:
        payload = {}
        for item in line.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            payload[key.strip()] = value.strip()
    else:
        tokens = line.split(",")
        if len(tokens) < 4:
            return None
        payload = {
            "temperature": tokens[0],
            "humidity": tokens[1],
            "co2": tokens[2],
            "light": tokens[3],
        }
        if len(tokens) > 4:
            payload["occupancy_count"] = tokens[4]

    try:
        return SensorSample(
            timestamp=datetime.now(),
            temperature=float(payload["temperature"]),
            humidity=float(payload["humidity"]),
            co2=float(payload["co2"]),
            light=float(payload["light"]),
            occupancy_count=int(payload.get("occupancy_count", 30)),
        )
    except Exception:
        return None


# -----------------------------
# MOCK STREAM (SIMULATION-LIKE)
# -----------------------------
def generate_mock_stream(duration: int, interval: float):
    steps = int(duration / interval)

    for i in range(steps):
        drift = i / steps

        yield SensorSample(
            timestamp=datetime.now(),
            temperature=22 + random.uniform(-1, 1) + drift * 3,
            humidity=50 + random.uniform(-5, 5),
            co2=500 + drift * 400 + random.uniform(-50, 50),
            light=450 + random.uniform(-100, 100),
            occupancy_count=random.randint(20, 35),
        )

        time.sleep(interval)


# -----------------------------
# SERIAL CONNECTION
# -----------------------------
def open_serial(port: str, baudrate: int, timeout: float):
    try:
        import serial
    except ImportError:
        raise RuntimeError("Install pyserial: pip install pyserial")

    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


# -----------------------------
# MAIN LOOP
# -----------------------------
def run_hil_test(args: argparse.Namespace) -> int:
    records = []

    stream = generate_mock_stream(args.duration, args.interval) if args.mock else None
    serial_conn = None
    empty_reads = 0
    malformed_reads = 0
    max_empty_reads = max(20, int(args.duration / max(args.interval, 0.2)) * 3)

    if not args.mock:
        port = normalize_serial_port(args.port)
        serial_conn = open_serial(port, args.baud, args.timeout)
        print(f"Connected to serial device at {port} @ {args.baud} baud")

    start = time.time()

    try:
        while time.time() - start < args.duration:

            if args.mock:
                sample = next(stream)
            else:
                raw = serial_conn.readline().decode(errors="ignore")
                if not raw.strip():
                    # Request-response fallback for Arduino firmware that waits for a ping.
                    serial_conn.write((datetime.now().isoformat() + "\n").encode("utf-8"))
                    raw = serial_conn.readline().decode(errors="ignore")
                sample = parse_sample_line(raw)
                if not sample:
                    if raw.strip():
                        malformed_reads += 1
                        if malformed_reads <= 5 or malformed_reads % 20 == 0:
                            print(f"Skipping malformed line: {raw.strip()}")
                    else:
                        empty_reads += 1
                        if empty_reads in (10, 30, 60):
                            print(
                                f"No serial data yet from {normalize_serial_port(args.port)}. "
                                "Check COM port and baud (common values: 9600 or 9700)."
                            )
                        if empty_reads >= max_empty_reads:
                            print("Stopping early due to repeated empty reads from serial device.")
                            break
                    continue
                empty_reads = 0

            raw_features = sample.as_features()
            features = apply_refined_calibration(raw_features) if not args.no_calibration else raw_features

            # -----------------------------
            # ML prediction
            # -----------------------------
            prediction, confidence = predict_environment(features)

            # -----------------------------
            # UNIFIED SIMULATION LOGIC
            # -----------------------------
            zone_state = evaluate_features_zone(features)
            fused = fuse_model_zone_status(prediction, zone_state, confidence)

            final_status = fused["final_status"]
            agreement = compute_agreement_score(prediction, zone_state)

            # -----------------------------
            # INTERVENTION RULES (simple hardware-side trigger)
            # -----------------------------
            interventions = []
            if zone_state["overall_zone"] == "non-conducive":
                interventions.append("CRITICAL_ADJUSTMENT")

            # -----------------------------
            # LOGGING
            # -----------------------------
            records.append({
                "time": sample.timestamp.isoformat(),
                "temp_raw": sample.temperature,
                "humidity_raw": sample.humidity,
                "co2_raw": sample.co2,
                "light_raw": sample.light,
                "temp": features["temperature"],
                "humidity": features["humidity"],
                "co2": features["co2"],
                "light": features["light"],
                "prediction": prediction,
                "confidence": float(confidence),
                "final_status": final_status,
                "zone": zone_state["overall_zone"],
                "agreement": agreement,
                "interventions": ";".join(interventions),
                "calibration_mode": "raw" if args.no_calibration else "refined",
            })

            if args.no_calibration:
                sensor_summary = (
                    f"T={features['temperature']:.1f}C "
                    f"H={features['humidity']:.1f}% "
                    f"CO2={features['co2']:.0f}ppm "
                    f"L={features['light']:.0f}lux"
                )
            else:
                sensor_summary = (
                    f"T={features['temperature']:.1f}C(raw {sample.temperature:.1f}) "
                    f"H={features['humidity']:.1f}%(raw {sample.humidity:.1f}) "
                    f"CO2={features['co2']:.0f}ppm(raw {sample.co2:.0f}) "
                    f"L={features['light']:.0f}lux(raw {sample.light:.0f})"
                )

            print(
                f"[{sample.timestamp.strftime('%H:%M:%S')}] "
                f"{sensor_summary} "
                f"model={prediction:<12} zone={zone_state['overall_zone']:<15} "
                f"final={final_status:<15} conf={confidence:.2f}"
            )

            if not args.mock:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("Stopped.")

    finally:
        if serial_conn:
            serial_conn.close()

    # -----------------------------
    # SUMMARY
    # -----------------------------
    print("\n=== HIL SUMMARY ===")
    print(f"Samples: {len(records)}")
    if records:
        avg_conf = statistics.fmean(r["confidence"] for r in records)
        print(f"Avg confidence: {avg_conf:.2f}")
    else:
        print("Avg confidence: N/A (no valid samples)")
        if not args.mock:
            print(
                "Hint: verify COM port/baud and sensor payload format. "
                "For many Arduino sketches, try --baud 9600."
            )

    if args.output and records:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved log to: {args.output}")
    elif args.output:
        print("No output CSV written because no valid samples were captured.")

    return 0


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    p.add_argument("--port", default="COM7")
    p.add_argument("--baud", type=int, default=9700)
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--output", default=str(ROOT_DIR / "hil_log.csv"))
    p.add_argument("--no-calibration", action="store_true", help="Use raw sensor values instead of refined calibrated values")
    return p


def main():
    args = build_parser().parse_args()
    return run_hil_test(args)


if __name__ == "__main__":
    raise SystemExit(main())