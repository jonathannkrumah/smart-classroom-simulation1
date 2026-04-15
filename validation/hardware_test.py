#!/usr/bin/env python3
"""
Hardware-in-the-loop validation for smart classroom model.

Supports two modes:
1) Serial mode: read live sensor packets from MCU/edge device.
2) Mock mode: generate synthetic sensor stream for end-to-end verification.

Windows setup notes:
- Serial port example: COM7
- Baud rate example: 9700

Packet input formats accepted from serial line (one sample per line):
- JSON: {"temperature":22.4,"humidity":48,"co2":650,"light":420,"occupancy_count":28}
- CSV:  22.4,48,650,420,28
- Key-value: temperature=22.4,humidity=48,co2=650,light=420,occupancy_count=28
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
from typing import Dict, Iterable, List, Optional, Tuple


# Ensure root import works when run as: python validation/hardware_test.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from simulation.ml_integration import predict_environment  # noqa: E402


REQUIRED_FEATURES = ("temperature", "humidity", "co2", "light")


def normalize_serial_port(port: str) -> str:
	"""Normalize common Windows COM-port typos and preserve non-Windows ports."""
	port = port.strip()
	upper_port = port.upper()
	if upper_port.startswith("COMP") and len(port) > 4:
		# Accept accidental 'COMP7' and normalize to 'COM7'.
		return f"COM{port[4:]}"
	return port


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
			"occupancy": self.occupancy_count,
		}


def baseline_label(features: Dict[str, float]) -> str:
	"""Simple threshold baseline for verification agreement checks."""
	conducive = True

	if features["co2"] > 800:
		conducive = False
	if features["temperature"] > 27 or features["temperature"] < 18:
		conducive = False
	if features["light"] < 250 or features["light"] > 800:
		conducive = False
	if features["humidity"] < 30 or features["humidity"] > 70:
		conducive = False

	return "conducive" if conducive else "non-conducive"


def intervention_recommendations(features: Dict[str, float]) -> List[str]:
	recommendations: List[str] = []
	if features["co2"] > 800:
		recommendations.append("VENT_ON")
	if features["temperature"] > 27:
		recommendations.append("COOLING_ON")
	elif features["temperature"] < 18:
		recommendations.append("HEATING_ON")
	if features["light"] < 250:
		recommendations.append("LIGHTS_ON")
	elif features["light"] > 800:
		recommendations.append("DIM_LIGHTS")
	return recommendations


def _safe_float(value: object, default: float = math.nan) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _safe_int(value: object, default: int = 30) -> int:
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return default


def parse_sample_line(line: str) -> Optional[SensorSample]:
	line = line.strip()
	if not line:
		return None

	payload: Dict[str, object]

	# JSON payload
	if line.startswith("{") and line.endswith("}"):
		try:
			payload = json.loads(line)
		except json.JSONDecodeError:
			return None
	else:
		# key=value pairs
		if "=" in line:
			payload = {}
			for token in line.split(","):
				if "=" not in token:
					continue
				key, value = token.split("=", 1)
				payload[key.strip()] = value.strip()
		else:
			# CSV fallback: temp,humidity,co2,light,occupancy(optional)
			tokens = [t.strip() for t in line.split(",")]
			if len(tokens) < 5:
				if len(tokens) < 4:
					return None
			payload = {
				"temperature": tokens[0],
				"humidity": tokens[1],
				"co2": tokens[2],
				"light": tokens[3],
			}
			if len(tokens) >= 5:
				payload["occupancy_count"] = tokens[4]

	temperature = _safe_float(payload.get("temperature"))
	humidity = _safe_float(payload.get("humidity"))
	co2 = _safe_float(payload.get("co2"))
	light = _safe_float(payload.get("light"))
	occupancy_count = _safe_int(payload.get("occupancy_count", payload.get("occupancy", 30)), 30)

	if any(math.isnan(v) for v in (temperature, humidity, co2, light)):
		return None

	return SensorSample(
		timestamp=datetime.now(),
		temperature=temperature,
		humidity=humidity,
		co2=co2,
		light=light,
		occupancy_count=occupancy_count,
	)


def generate_mock_stream(duration_seconds: int, interval_seconds: float, seed: int) -> Iterable[SensorSample]:
	random.seed(seed)
	samples = max(1, int(duration_seconds / max(interval_seconds, 0.1)))

	for i in range(samples):
		drift = i / max(samples, 1)
		temp = 22 + random.uniform(-1.2, 1.2) + 4.5 * max(0, drift - 0.45)
		humidity = 50 + random.uniform(-6, 6)
		co2 = 500 + (420 * drift) + random.uniform(-80, 80)
		light = 420 + random.uniform(-140, 140)
		occupancy = random.randint(20, 35)

		yield SensorSample(
			timestamp=datetime.now(),
			temperature=max(14, min(35, temp)),
			humidity=max(20, min(85, humidity)),
			co2=max(350, min(2200, co2)),
			light=max(80, min(1200, light)),
			occupancy_count=occupancy,
		)

		time.sleep(interval_seconds)


def open_serial(port: str, baudrate: int, timeout: float):
	try:
		import serial  # type: ignore
	except ImportError as exc:
		raise RuntimeError("pyserial is required for serial mode. Install with: pip install pyserial") from exc

	port = normalize_serial_port(port)
	return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)


def summarize(records: List[Dict[str, object]]) -> None:
	if not records:
		print("No records collected.")
		return

	total = len(records)
	agreements = sum(1 for r in records if r["agrees_with_baseline"])
	conducive = sum(1 for r in records if r["model_prediction"] == "conducive")
	non_conducive = total - conducive
	avg_conf = statistics.fmean(float(r["confidence"]) for r in records)

	print("\n" + "=" * 68)
	print("HARDWARE-IN-THE-LOOP VERIFICATION SUMMARY")
	print("=" * 68)
	print(f"Samples                     : {total}")
	print(f"Model conducive count       : {conducive}")
	print(f"Model non-conducive count   : {non_conducive}")
	print(f"Agreement vs baseline rules : {agreements}/{total} ({agreements / total:.1%})")
	print(f"Average confidence          : {avg_conf:.1%}")


def export_csv(records: List[Dict[str, object]], output_path: str) -> None:
	if not records:
		return

	fieldnames = [
		"timestamp",
		"temperature",
		"humidity",
		"co2",
		"light",
		"occupancy_count",
		"model_prediction",
		"confidence",
		"baseline_prediction",
		"agrees_with_baseline",
		"recommendations",
	]

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(records)

	print(f"\nSaved verification log to: {output_path}")


def run_hil_test(args: argparse.Namespace) -> int:
	records: List[Dict[str, object]] = []

	if args.mock:
		stream = generate_mock_stream(
			duration_seconds=args.duration,
			interval_seconds=args.interval,
			seed=args.seed,
		)
		serial_conn = None
		print("Running in MOCK mode (no hardware serial required).")
	else:
		if not args.port:
			print("Error: --port is required in serial mode.")
			return 2
		serial_conn = open_serial(args.port, args.baud, timeout=args.timeout)
		stream = None
		print(f"Connected to serial device at {args.port} @ {args.baud} baud")

	started = time.time()
	max_runtime = args.duration

	try:
		while (time.time() - started) < max_runtime:
			if args.mock:
				try:
					sample = next(stream)  # type: ignore[arg-type]
				except StopIteration:
					break
			else:
				raw = serial_conn.readline().decode("utf-8", errors="ignore").strip()
				if not raw:
					continue
				sample = parse_sample_line(raw)
				if sample is None:
					print(f"Skipping malformed line: {raw}")
					continue

			features = sample.as_features()
			prediction, confidence = predict_environment(
				features,
				context={
					"datetime": sample.timestamp,
					"current_minute": int(time.time() - started) // 60,
					"room_size": args.room_size,
					"start_hour": args.start_hour,
				},
			)

			baseline = baseline_label(features)
			agrees = prediction == baseline
			recs = intervention_recommendations(features)

			ts = sample.timestamp.strftime("%H:%M:%S")
			status = "✅" if prediction == "conducive" else "⚠️"
			agree_text = "match" if agrees else "diff"
			print(
				f"[{ts}] {status} model={prediction:<13} conf={confidence:.1%} "
				f"baseline={baseline:<13} ({agree_text}) | "
				f"T={sample.temperature:.1f}°C CO2={sample.co2:.0f}ppm "
				f"H={sample.humidity:.0f}% L={sample.light:.0f}lux"
			)

			if args.send_actuation and recs and serial_conn is not None:
				for command in recs:
					serial_conn.write((command + "\n").encode("utf-8"))

			records.append(
				{
					"timestamp": sample.timestamp.isoformat(),
					"temperature": round(sample.temperature, 3),
					"humidity": round(sample.humidity, 3),
					"co2": round(sample.co2, 3),
					"light": round(sample.light, 3),
					"occupancy_count": sample.occupancy_count,
					"model_prediction": prediction,
					"confidence": round(float(confidence), 6),
					"baseline_prediction": baseline,
					"agrees_with_baseline": agrees,
					"recommendations": ";".join(recs),
				}
			)

			if not args.mock:
				time.sleep(max(args.interval, 0.05))

	except KeyboardInterrupt:
		print("\nStopped by user.")
	finally:
		if serial_conn is not None:
			serial_conn.close()

	summarize(records)

	output_file = args.output
	if output_file:
		export_csv(records, output_file)

	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Hardware-in-the-loop model verification")
	parser.add_argument("--mock", action="store_true", help="Use synthetic sensor stream instead of serial hardware")
	parser.add_argument("--port", type=str, default="COM7", help="Serial port, e.g. COM7 on Windows or /dev/ttyUSB0 on Linux")
	parser.add_argument("--baud", type=int, default=9700, help="Serial baud rate (use the same value as the Arduino sketch)")
	parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout in seconds")
	parser.add_argument("--duration", type=int, default=180, help="Run duration in seconds")
	parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
	parser.add_argument("--room-size", type=int, default=100, help="Room size in m² for model context")
	parser.add_argument("--start-hour", type=int, default=datetime.now().hour, help="Start hour context for model")
	parser.add_argument("--send-actuation", action="store_true", help="Send actuator command strings to serial device")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for mock mode")
	parser.add_argument(
		"--output",
		type=str,
		default=str(ROOT_DIR / "validation" / "hil_verification_log.csv"),
		help="CSV output path",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	return run_hil_test(args)


if __name__ == "__main__":
	raise SystemExit(main())