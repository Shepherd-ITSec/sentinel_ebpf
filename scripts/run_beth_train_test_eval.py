#!/usr/bin/env python3
"""Run BETH warmup-on-train then evaluate on test split.

Run-all mode (survives SSH disconnect):
  Run under nohup or inside tmux/screen so the process keeps running after you log out:
    nohup uv run python scripts/run_beth_train_test_eval.py --run-all --pace fast > run_all.log 2>&1 &
  Or: tmux new -s beth && uv run python scripts/run_beth_train_test_eval.py --run-all --pace fast
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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


def _start_detector(port: int, anomaly_log_path: Path, env_overrides: Optional[Dict[str, str]] = None) -> subprocess.Popen:
  env = os.environ.copy()
  env["DETECTOR_PORT"] = str(port)
  env["ANOMALY_LOG_PATH"] = str(anomaly_log_path)
  env["DETECTOR_QUIET"] = "1"
  if env_overrides:
    env.update(env_overrides)
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


def run_one(
  out_dir: Path,
  train_evt: Path,
  test_evt: Path,
  test_labels: Path,
  train_count: int,
  test_count: int,
  pace: str,
  detector_port: int,
  startup_timeout: float,
  label_mode: str,
  env_overrides: Optional[Dict[str, str]] = None,
  quiet: bool = False,
) -> Dict:
  """Run one eval: start detector with env_overrides, replay train+test, evaluate. Returns metrics dict."""
  anomalies = out_dir / "anomalies.jsonl"
  if anomalies.exists():
    anomalies.unlink()
  if not quiet:
    print(f"Starting detector on port {detector_port} (env overrides: {env_overrides or {}})")
  detector = _start_detector(detector_port, anomalies, env_overrides=env_overrides)
  target = f"localhost:{detector_port}"
  try:
    _wait_for_detector(target, timeout_s=startup_timeout)
    if not quiet:
      print(f"Replaying train events from {train_evt} to {target}")
    replay(str(train_evt), target, pace, start_ms=None, end_ms=None, total=train_count, label="Train")
    if not quiet:
      print(f"Replaying test events from {test_evt} to {target}")
    replay(str(test_evt), target, pace, start_ms=None, end_ms=None, total=test_count, label="Test")
  finally:
    _stop_detector(detector)
  result = evaluate(test_labels, anomalies, label_mode=label_mode)
  result["train_samples"] = train_count
  result["test_samples_converted"] = test_count
  result["anomalies_path"] = str(anomalies)
  result["labels_path"] = str(test_labels)
  return result


def _run_all_matrix() -> List[tuple]:
  """Yield (run_id, label_mode, env_overrides) for the full overnight matrix."""
  algorithms = ["halfspacetrees", "loda", "kitnet", "memstream"]
  label_modes = ["evil_only", "sus_or_evil"]
  thresholds = [0.3, 0.5, 0.7]
  runs: List[tuple] = []
  for algo in algorithms:
    for label_mode in label_modes:
      for th in thresholds:
        run_id = f"{algo}_th{th}_{label_mode}"
        env = {
          "DETECTOR_MODEL_ALGORITHM": algo,
          "DETECTOR_THRESHOLD": str(th),
        }
        runs.append((run_id, label_mode, env))
  return runs


def main() -> None:
  ap = argparse.ArgumentParser(
    description="Warm up detector on BETH train split, then evaluate on test split.",
    epilog="For overnight run-all, use: nohup uv run python scripts/run_beth_train_test_eval.py --run-all --pace fast > run_all.log 2>&1 &",
  )
  ap.add_argument("--train-csv", default="test_data/beth/labelled_training_data.csv", help="BETH training CSV")
  ap.add_argument("--test-csv", default="test_data/beth/labelled_testing_data.csv", help="BETH test CSV")
  ap.add_argument("--out-dir", default="test_data/beth/eval", help="Output directory (or base dir for --run-all)")
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port for local eval run")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast", help="Replay speed for both splits")
  ap.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for detector readiness")
  ap.add_argument("--train-limit", type=int, default=0, help="Optional row cap for train conversion (0=all)")
  ap.add_argument("--test-limit", type=int, default=0, help="Optional row cap for test conversion (0=all)")
  ap.add_argument(
    "--label-mode",
    choices=["evil_only", "sus_or_evil"],
    default="sus_or_evil",
    help="Eval positive = evil>0 only (evil_only) or sus>0 or evil>0 (sus_or_evil). Ignored if --run-all.",
  )
  ap.add_argument(
    "--run-all",
    action="store_true",
    help="Run full matrix: all algorithms × label_modes × thresholds; write each run to a subdir and a manifest. Use nohup/tmux so it survives SSH disconnect.",
  )
  args = ap.parse_args()

  if args.run_all:
    base_dir = Path(args.out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_all_dir = base_dir.parent / f"run_all_{ts}"
    run_all_dir.mkdir(parents=True, exist_ok=True)
    train_evt = run_all_dir / "train.evt1"
    train_labels = run_all_dir / "train.labels.ndjson"
    test_evt = run_all_dir / "test.evt1"
    test_labels = run_all_dir / "test.labels.ndjson"
    print(f"Run-all: converting once into {run_all_dir}")
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
    matrix = _run_all_matrix()
    manifest: List[Dict] = []
    for i, (run_id, label_mode, env_overrides) in enumerate(matrix):
      run_dir = run_all_dir / run_id
      run_dir.mkdir(parents=True, exist_ok=True)
      print(f"\n[{i+1}/{len(matrix)}] Run: {run_id} (label_mode={label_mode})")
      start = time.time()
      try:
        result = run_one(
          out_dir=run_dir,
          train_evt=train_evt,
          test_evt=test_evt,
          test_labels=test_labels,
          train_count=train_count,
          test_count=test_count,
          pace=args.pace,
          detector_port=args.detector_port,
          startup_timeout=args.startup_timeout,
          label_mode=label_mode,
          env_overrides=env_overrides,
          quiet=True,
        )
        elapsed = time.time() - start
        metrics_path = run_dir / "metrics.test_only.json"
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        manifest.append({
          "run_id": run_id,
          "label_mode": label_mode,
          "config": env_overrides,
          "status": "ok",
          "elapsed_seconds": round(elapsed, 1),
          "metrics_path": str(metrics_path),
          "precision": result.get("precision"),
          "recall": result.get("recall"),
          "f1": result.get("f1"),
        })
        print(f"  ok in {elapsed:.0f}s  precision={result.get('precision', 0):.3f} recall={result.get('recall', 0):.3f} f1={result.get('f1', 0):.3f}")
      except Exception as e:
        manifest.append({
          "run_id": run_id,
          "label_mode": label_mode,
          "config": env_overrides,
          "status": "error",
          "error": str(e),
        })
        print(f"  error: {e}")
    manifest_path = run_all_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nRun-all done. Manifest: {manifest_path}")
    return

  # Single run
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  train_evt = out_dir / "train.evt1"
  train_labels = out_dir / "train.labels.ndjson"
  test_evt = out_dir / "test.evt1"
  test_labels = out_dir / "test.labels.ndjson"
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

  result = run_one(
    out_dir=out_dir,
    train_evt=train_evt,
    test_evt=test_evt,
    test_labels=test_labels,
    train_count=train_count,
    test_count=test_count,
    pace=args.pace,
    detector_port=args.detector_port,
    startup_timeout=args.startup_timeout,
    label_mode=args.label_mode,
    env_overrides=None,
    quiet=False,
  )
  metrics_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2))
  print(f"metrics_file={metrics_out}")


if __name__ == "__main__":
  main()

