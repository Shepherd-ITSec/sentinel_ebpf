#!/usr/bin/env python3
"""Run BETH warmup-on-train then evaluate on test split."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

if __package__ is None or __package__ == "":
  sys.path.append(str(Path(__file__).resolve().parent))

from convert_beth_to_evt1 import convert
from evaluate_beth_replay import evaluate
from replay_logs import replay


def _wait_for_detector(target: str, timeout_s: float) -> None:
  deadline = time.time() + timeout_s
  last_error = ""
  while time.time() < deadline:
    try:
      channel = grpc.insecure_channel(target)
      stub = health_pb2_grpc.HealthStub(channel)
      resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=2.0)
      if resp.status == health_pb2.HealthCheckResponse.SERVING:
        return
      last_error = f"health status={resp.status}"
    except Exception as exc:
      last_error = str(exc)
    time.sleep(0.5)
  raise RuntimeError(f"detector at {target} did not become ready: {last_error}")


def _start_detector(port: int, anomaly_log_path: Path) -> subprocess.Popen:
  env = os.environ.copy()
  env["DETECTOR_PORT"] = str(port)
  env["ANOMALY_LOG_PATH"] = str(anomaly_log_path)
  env["DETECTOR_QUIET"] = "1"  # no per-event anomaly/processed logs during eval
  cmd = [sys.executable, "-m", "detector.server"]
  return subprocess.Popen(
    cmd,
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )


def _stop_detector(proc: subprocess.Popen) -> None:
  if proc.poll() is not None:
    return
  proc.terminate()
  try:
    proc.wait(timeout=10)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait(timeout=5)


def main() -> None:
  ap = argparse.ArgumentParser(description="Warm up detector on BETH train split, then evaluate on test split.")
  ap.add_argument("--train-csv", default="test_data/beth/labelled_training_data.csv", help="BETH training CSV")
  ap.add_argument("--test-csv", default="test_data/beth/labelled_testing_data.csv", help="BETH test CSV")
  ap.add_argument("--out-dir", default="test_data/beth/eval", help="Output directory for converted files and metrics")
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port for local eval run")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast", help="Replay speed for both splits")
  ap.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for detector readiness")
  ap.add_argument("--train-limit", type=int, default=0, help="Optional row cap for train conversion (0=all)")
  ap.add_argument("--test-limit", type=int, default=0, help="Optional row cap for test conversion (0=all)")
  ap.add_argument(
    "--label-mode",
    choices=["evil_only", "sus_or_evil"],
    default="sus_or_evil",
    help="Eval positive = evil>0 only (evil_only) or sus>0 or evil>0 (sus_or_evil, default)",
  )
  args = ap.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  train_evt = out_dir / "train.evt1"
  train_labels = out_dir / "train.labels.ndjson"
  test_evt = out_dir / "test.evt1"
  test_labels = out_dir / "test.labels.ndjson"
  anomalies = out_dir / "anomalies.jsonl"
  metrics_out = out_dir / "metrics.test_only.json"

  train_count = convert(
    Path(args.train_csv),
    train_evt,
    train_labels,
    limit=args.train_limit,
    event_id_prefix="beth-train",
  )
  test_count = convert(
    Path(args.test_csv),
    test_evt,
    test_labels,
    limit=args.test_limit,
    event_id_prefix="beth-test",
  )

  if anomalies.exists():
    anomalies.unlink()

  print(f"Starting detector on port {args.detector_port}")
  detector = _start_detector(args.detector_port, anomalies)
  target = f"localhost:{args.detector_port}"
  try:
    _wait_for_detector(target, timeout_s=args.startup_timeout)
    print(f"Replaying train events from {train_evt} to {target}")
    replay(str(train_evt), target, args.pace, start_ms=None, end_ms=None, total=train_count, label="Train")
    print(f"Replaying test events from {test_evt} to {target}")
    replay(str(test_evt), target, args.pace, start_ms=None, end_ms=None, total=test_count, label="Test")
  finally:
    _stop_detector(detector)

  result = evaluate(test_labels, anomalies, label_mode=args.label_mode)
  result["train_samples"] = train_count
  result["test_samples_converted"] = test_count
  result["anomalies_path"] = str(anomalies)
  result["labels_path"] = str(test_labels)
  metrics_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  print(f"metrics_file={metrics_out}")


if __name__ == "__main__":
  main()

