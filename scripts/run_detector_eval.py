#!/usr/bin/env python3
"""Run detector eval: BETH (train+test) or single-stream (e.g. synthetic EVT1).

BETH mode (default): convert train/test CSV, warm up on train, replay test, evaluate.
Single-stream: pass --evt1 and --labels to replay one EVT1 and evaluate (e.g. synthetic data).
Run-all: full matrix; use nohup/tmux for overnight runs.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

if __package__ is None or __package__ == "":
  sys.path.append(str(Path(__file__).resolve().parent))

from convert_beth_to_evt1 import convert
from evaluate_beth_replay import evaluate
from replay_logs import replay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(Path(__file__).stem)


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
  test_evt: Path,
  test_labels: Path,
  test_count: int,
  pace: str,
  detector_port: int,
  startup_timeout: float,
  label_mode: str,
  env_overrides: Optional[Dict[str, str]] = None,
  quiet: bool = False,
  train_evt: Optional[Path] = None,
  train_count: int = 0,
) -> Dict:
  """Run one eval: start detector, optionally replay train then replay test, evaluate test labels. Returns metrics dict."""
  anomalies = out_dir / "anomalies.jsonl"
  if anomalies.exists():
    anomalies.unlink()
  if not quiet:
    log.info(f"Run one: Starting detector on port {detector_port} (env overrides: {env_overrides or {}})")
  detector = _start_detector(detector_port, anomalies, env_overrides=env_overrides)
  target = f"localhost:{detector_port}"
  try:
    _wait_for_detector(target, timeout_s=startup_timeout)
    if train_evt is not None and train_count > 0:
      if not quiet:
        log.info(f"Run one: Replaying train events from {train_evt} to {target}")
      replay(str(train_evt), target, pace, start_ms=None, end_ms=None, total=train_count, label="Train")
    if not quiet:
      log.info(f"Run one: Replaying test events from {test_evt} to {target}")
    replay(str(test_evt), target, pace, start_ms=None, end_ms=None, total=test_count, label="Test")
  finally:
    _stop_detector(detector)
  result = evaluate(test_labels, anomalies, label_mode=label_mode)
  result["train_samples"] = train_count
  result["test_samples_converted"] = test_count
  result["anomalies_path"] = str(anomalies)
  result["labels_path"] = str(test_labels)
  return result


def _count_labels(path: Path) -> int:
  """Count non-empty lines in a labels NDJSON file."""
  n = 0
  with path.open("r", encoding="utf-8") as f:
    line_iter = iter(f)
    if tqdm:
      line_iter = tqdm(line_iter, desc="Count labels", unit=" line", file=sys.stderr)
    for line in line_iter:
      if line.strip():
        n += 1
  return n


def _run_all_matrix(single_stream: bool = False) -> List[tuple]:
  """Yield (run_id, label_mode, env_overrides). BETH: 28 runs (both label_modes). Single-stream (e.g. synthetic): 14 runs (evil_only only)."""
  algorithms = ["kitnet", "halfspacetrees", "loda_ema", "memstream", "zscore", "knn", "freq1d", "gausscop", "copulatree", "latentcluster"]
  thresholds = [0.7, 0.5, 0.3]
  label_modes = ["evil_only"] if single_stream else ["evil_only", "sus_or_evil"]
  runs: List[tuple] = []
  log.info(f"Run matrix: algorithms={algorithms} label_modes={label_modes} single_stream={single_stream}")
  for algo in algorithms:
    for label_mode in label_modes:
      for th in thresholds:
        run_id = f"{algo}_th{th}_{label_mode}"
        env = {
          "DETECTOR_MODEL_ALGORITHM": algo,
          "DETECTOR_THRESHOLD": str(th),
        }
        runs.append((run_id, label_mode, env))
  log.info(f"Run matrix: Total runs: {len(runs)}")
  return runs


def main() -> None:
  ap = argparse.ArgumentParser(
    description="Detector eval: BETH (train+test) or single-stream EVT1. Warm up, replay, evaluate.",
    epilog="Overnight: nohup uv run python scripts/run_detector_eval.py --run-all --pace fast > run_all.log 2>&1 &",
  )
  ap.add_argument("--train-csv", default=None, help="BETH training CSV (defaults to test_data/beth/... if not using --evt1/--labels)")
  ap.add_argument("--test-csv", default=None, help="BETH test CSV")
  ap.add_argument("--evt1", default=None, help="Single-stream: path to EVT1 log (e.g. synthetic). Use with --labels; then no train/test CSV.")
  ap.add_argument("--labels", default=None, help="Single-stream: path to labels NDJSON matching --evt1.")
  ap.add_argument("--out-dir", default="test_data/beth/eval", help="Output directory (or base dir for --run-all)")
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port for local eval run")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast", help="Replay speed for both splits")
  ap.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for detector readiness")
  ap.add_argument("--train-limit", type=int, default=0, help="Optional row cap for train conversion (0=all)")
  ap.add_argument("--test-limit", type=int, default=0, help="Optional row cap for test conversion (0=all)")
  ap.add_argument(
    "--score-mode",
    choices=["raw", "scaled"],
    default="raw",
    help="Detector score space: raw (model.score_*_raw) or scaled (bounded [0,1] scores). Default: raw.",
  )
  ap.add_argument(
    "--label-mode",
    choices=["evil_only", "sus_or_evil"],
    default="sus_or_evil",
    help="Eval positive = evil>0 only (evil_only) or sus>0 or evil>0 (sus_or_evil). Ignored if --run-all.",
  )
  ap.add_argument(
    "--run-all",
    action="store_true",
    help="Run full matrix; use nohup/tmux so it survives SSH disconnect.",
  )
  args = ap.parse_args()

  single_stream = args.evt1 is not None and args.labels is not None
  if not single_stream:
    args.train_csv = args.train_csv or "test_data/beth/labelled_training_data.csv"
    args.test_csv = args.test_csv or "test_data/beth/labelled_testing_data.csv"

  if args.run_all:
    base_dir = Path(args.out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_all_dir = base_dir.parent / f"run_all_{ts}"
    run_all_dir.mkdir(parents=True, exist_ok=True)

    if single_stream:
      test_evt = Path(args.evt1)
      test_labels = Path(args.labels)
      test_count = _count_labels(test_labels)
      train_evt, train_count = None, 0
      log.info("Run-all (single-stream): %s / %s (%d events)", test_evt, test_labels, test_count)
    else:
      train_evt = run_all_dir / "train.evt1"
      train_labels = run_all_dir / "train.labels.ndjson"
      test_evt = run_all_dir / "test.evt1"
      test_labels = run_all_dir / "test.labels.ndjson"
      log.info("Run-all: converting once into %s", run_all_dir)
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

    matrix = _run_all_matrix(single_stream=single_stream)
    manifest: List[Dict] = []
    for i, (run_id, label_mode, env_overrides) in enumerate(matrix):
      # Ensure score_mode is propagated for every run
      env_overrides = dict(env_overrides)
      env_overrides["DETECTOR_SCORE_MODE"] = args.score_mode
      run_dir = run_all_dir / run_id
      run_dir.mkdir(parents=True, exist_ok=True)
      log.info("[%d/%d] Run: %s (label_mode=%s)", i + 1, len(matrix), run_id, label_mode)
      start = time.time()
      try:
        result = run_one(
          out_dir=run_dir,
          test_evt=test_evt,
          test_labels=test_labels,
          test_count=test_count,
          pace=args.pace,
          detector_port=args.detector_port,
          startup_timeout=args.startup_timeout,
          label_mode=label_mode,
          env_overrides=env_overrides,
          quiet=True,
          train_evt=train_evt,
          train_count=train_count,
        )
        elapsed = time.time() - start
        metrics_path = run_dir / "metrics.json"
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
        log.info("  ok in %.0fs  precision=%.3f recall=%.3f f1=%.3f", elapsed, result.get("precision", 0) or 0, result.get("recall", 0) or 0, result.get("f1", 0) or 0)
      except Exception as e:
        manifest.append({
          "run_id": run_id,
          "label_mode": label_mode,
          "config": env_overrides,
          "status": "error",
          "error": str(e),
        })
        log.error("  error: %s", e)
    manifest_path = run_all_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Run-all done. Manifest: %s", manifest_path)
    return

  # Single run
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  metrics_out = out_dir / "metrics.json"

  if single_stream:
    test_evt = Path(args.evt1)
    test_labels = Path(args.labels)
    test_count = _count_labels(test_labels)
    train_evt, train_count = None, 0
  else:
    train_evt = out_dir / "train.evt1"
    train_labels = out_dir / "train.labels.ndjson"
    test_evt = out_dir / "test.evt1"
    test_labels = out_dir / "test.labels.ndjson"
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
    test_evt=test_evt,
    test_labels=test_labels,
    test_count=test_count,
    pace=args.pace,
    detector_port=args.detector_port,
    startup_timeout=args.startup_timeout,
    label_mode=args.label_mode,
    env_overrides={"DETECTOR_SCORE_MODE": args.score_mode},
    quiet=False,
    train_evt=train_evt,
    train_count=train_count,
  )
  metrics_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  log.info("metrics: %s", json.dumps(result, indent=2))
  log.info("metrics_file=%s", metrics_out)


if __name__ == "__main__":
  main()
