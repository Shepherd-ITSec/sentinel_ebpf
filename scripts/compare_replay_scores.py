#!/usr/bin/env python3
"""Replay a slice of detector-events JSONL and compare model scores with original log.

Compare: original scores (model had seen events 0..N-1) vs cold-model scores (model had
NOT seen events before --start-event). Slice = events [start, start+limit). Replay the
full slice with online score-and-learn; compare every event's scores.
"""

import argparse
import gzip
import json
import logging
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

if __package__ is None or __package__ == "":
  here = Path(__file__).resolve()
  scripts_dir = here.parent
  repo_root = scripts_dir.parent
  # Ensure both the scripts directory (for replay_logs) and repo root (for detector.*) are on sys.path.
  sys.path.insert(0, str(repo_root))
  sys.path.insert(0, str(scripts_dir))

from replay_logs import replay
from detector import config as detector_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(Path(__file__).stem)


def _open_text_lines(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rt", encoding="utf-8")
  return path.open("r", encoding="utf-8")


def _count_and_load_slice(
  path: Path,
  start_event: int = 0,
  start_ms: Optional[int] = None,
  limit: int = 0,
) -> Tuple[int, Dict[str, float]]:
  """Count events from start, load original scores for full slice. Returns (total_in_slice, {event_id: score})."""
  total_in_slice = 0
  cap = limit if limit > 0 else None
  n_seen = 0
  with _open_text_lines(path) as f:
    line_iter = iter(f)
    if tqdm:
      line_iter = tqdm(line_iter, desc="Count slice", unit=" line", file=sys.stderr)
    for line in line_iter:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "event_id" not in obj:
        continue
      if start_ms is not None:
        ts_ns = obj.get("ts_unix_nano", 0)
        ts_ms = ts_ns // 1_000_000
        if ts_ms < start_ms:
          continue
      elif n_seen < start_event:
        n_seen += 1
        continue
      total_in_slice += 1
      if cap is not None and total_in_slice >= cap:
        break

  total_in_slice = min(total_in_slice, cap) if cap else total_in_slice
  scores: Dict[str, float] = {}
  n_in_slice = 0
  n_seen = 0
  with _open_text_lines(path) as f:
    line_iter = iter(f)
    if tqdm:
      line_iter = tqdm(line_iter, desc="Load scores", unit=" line", file=sys.stderr)
    for line in line_iter:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "event_id" not in obj:
        continue
      if start_ms is not None:
        ts_ns = obj.get("ts_unix_nano", 0)
        ts_ms = ts_ns // 1_000_000
        if ts_ms < start_ms:
          continue
      elif n_seen < start_event:
        n_seen += 1
        continue
      eid = str(obj.get("event_id", ""))
      if "score" in obj:
        scores[eid] = float(obj["score"])
      n_in_slice += 1
      if n_in_slice >= total_in_slice:
        break

  return total_in_slice, scores


def _load_replay_scores(dump_path: Path, expected_count: int) -> Dict[str, float]:
  """Load scores from event dump. The last expected_count lines are from the replay."""
  lines: list[str] = []
  with dump_path.open("r", encoding="utf-8") as f:
    line_iter = iter(f)
    if tqdm:
      line_iter = tqdm(line_iter, desc="Load dump", unit=" line", file=sys.stderr)
    for line in line_iter:
      if line.strip():
        lines.append(line)

  # Last expected_count entries are from second-half replay
  scores: Dict[str, float] = {}
  score_iter = lines[-expected_count:]
  if tqdm:
    score_iter = tqdm(score_iter, desc="Parse scores", unit=" rec", file=sys.stderr)
  for line in score_iter:
    try:
      obj = json.loads(line)
    except json.JSONDecodeError:
      continue
    eid = str(obj.get("event_id", ""))
    if "score" in obj:
      scores[eid] = float(obj["score"])
  return scores


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


def _port_in_use(port: int) -> bool:
  """Return True if something is already serving on the given port (e.g. leftover detector)."""
  try:
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = health_pb2_grpc.HealthStub(channel)
    stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=1.0)
    return True
  except Exception:
    return False


def _start_detector(port: int, env_overrides: Optional[Dict[str, str]] = None) -> subprocess.Popen:
  if _port_in_use(port):
    raise RuntimeError(
      f"Port {port} already in use (another detector?). Stop it or use --detector-port to pick a different port."
    )
  env = os.environ.copy()
  env["DETECTOR_PORT"] = str(port)
  env["DETECTOR_QUIET"] = "1"
  if env_overrides:
    env.update(env_overrides)
  cmd = [sys.executable, "-m", "detector.server"]
  # For replay/debug runs we want to see detector logs instead of discarding them,
  # so send stdout/stderr to a file under test_data.
  debug_log = Path(__file__).resolve().parent.parent / "test_data" / "detector_replay_debug.log"
  debug_log.parent.mkdir(parents=True, exist_ok=True)
  log_file = debug_log.open("a", encoding="utf-8")
  return subprocess.Popen(
    cmd,
    env=env,
    cwd=Path(__file__).resolve().parent.parent,
    stdout=log_file,
    stderr=log_file,
    start_new_session=True,
  )


def _stop_detector(proc: subprocess.Popen) -> None:
  if proc.poll() is not None:
    return
  # Detector runs in its own process group (start_new_session=True). Kill the whole group
  # so we don't leave orphaned children (e.g. if detector ever spawns subprocesses).
  try:
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
  except (ProcessLookupError, OSError):
    proc.terminate()
  try:
    proc.wait(timeout=10)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
      proc.kill()
    proc.wait(timeout=5)


def _build_detector_config(env_overrides: Dict[str, str], detector_port: int) -> Dict[str, object]:
  """Return the effective DetectorConfig as a dict, given our env overrides.

  We mirror what _start_detector() does by applying overrides on top of the
  current environment, call detector.config.load_config(), then restore the
  original environment.
  """
  # Mirror the env we pass to the detector process
  cfg_env = {
    "DETECTOR_PORT": str(detector_port),
    "DETECTOR_QUIET": "1",
  }
  cfg_env.update(env_overrides)

  orig_env = os.environ.copy()
  try:
    os.environ.update(cfg_env)
    cfg = detector_config.load_config()
  finally:
    os.environ.clear()
    os.environ.update(orig_env)

  return asdict(cfg)


def main() -> None:
  ap = argparse.ArgumentParser(
    description="Replay slice of detector-events JSONL and compare model scores with original log.",
  )
  ap.add_argument("events", default="events_09_03_26.jsonl", nargs="?", help="Path to detector-events JSONL")
  ap.add_argument("--out-dir", default="test_data/compare_replay", help="Output directory for dump and report")
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast", help="Replay pace")
  ap.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for detector")
  ap.add_argument("--algorithm", default=None, help="Model algorithm (kitnet, loda_ema, halfspacetrees, memstream, zscore, knn, freq1d, gausscop, copulatree, latentcluster)")
  ap.add_argument("--threshold", default=None, help="Anomaly threshold (e.g. 0.7)")
  ap.add_argument(
    "--score-mode",
    choices=["raw", "scaled"],
    default="raw",
    help="Detector score space: raw (model.score_*_raw) or scaled (bounded [0,1] scores). Default: raw.",
  )
  ap.add_argument("--limit", type=int, default=0, help="Cap total events in slice for quick testing (0=all)")
  ap.add_argument("--start-event", type=int, default=0, help="Skip first N events; replay slice from there")
  ap.add_argument("--start-ms", type=int, default=None, help="Start from timestamp (ms since epoch); overrides --start-event")
  args = ap.parse_args()

  events_path = Path(args.events)
  if not events_path.exists():
    log.error("Events file not found: %s", events_path)
    sys.exit(1)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  dump_path = out_dir / "replay_dump.jsonl"
  if dump_path.exists():
    dump_path.unlink()

  log.info("Loading slice (start_event=%s, start_ms=%s, limit=%s)...", args.start_event, args.start_ms, args.limit or "all")
  total, original_scores = _count_and_load_slice(
    events_path,
    start_event=args.start_event,
    start_ms=args.start_ms,
    limit=args.limit,
  )
  skip = args.start_event if args.start_ms is None else 0
  log.info("Slice: %d events (cold from start), replay with online score-and-learn", total)

  env_overrides: Dict[str, str] = {"EVENT_DUMP_PATH": str(dump_path), "DETECTOR_SCORE_MODE": args.score_mode}
  if args.algorithm:
    env_overrides["DETECTOR_MODEL_ALGORITHM"] = args.algorithm
  if args.threshold:
    env_overrides["DETECTOR_THRESHOLD"] = args.threshold

  detector_cfg = _build_detector_config(env_overrides, args.detector_port)

  log.info("Starting detector (EVENT_DUMP_PATH=%s)...", dump_path)
  detector = _start_detector(args.detector_port, env_overrides=env_overrides)
  target = f"localhost:{args.detector_port}"

  try:
    _wait_for_detector(target, timeout_s=args.startup_timeout)
    log.info("Replay slice (%d events)...", total)
    replay(
      str(events_path),
      target,
      args.pace,
      start_ms=args.start_ms,
      end_ms=None,
      total=total,
      label="Replay",
      max_events=total,
      skip=skip,
    )
  finally:
    _stop_detector(detector)

  log.info("Loading replay scores from dump...")
  replay_scores = _load_replay_scores(dump_path, total)

  # Compare
  common = set(original_scores.keys()) & set(replay_scores.keys())
  if not common:
    log.error("No common event_ids between original and replay")
    sys.exit(1)

  diffs = []
  for eid in common:
    orig = original_scores[eid]
    rep = replay_scores[eid]
    diffs.append(abs(orig - rep))

  mae = statistics.mean(diffs) if diffs else 0.0
  max_diff = max(diffs) if diffs else 0.0
  try:
    orig_vals = [original_scores[e] for e in common]
    rep_vals = [replay_scores[e] for e in common]
    corr = statistics.correlation(orig_vals, rep_vals) if len(common) > 1 else 1.0
  except Exception:
    corr = float("nan")

  report = {
    "events_path": str(events_path),
    "start_event": args.start_event,
    "start_ms": args.start_ms,
    "total_events_in_slice": total,
    "common_event_ids": len(common),
    "mae": round(mae, 6),
    "max_abs_diff": round(max_diff, 6),
    "correlation": round(corr, 6) if not math.isnan(corr) else None,
    "algorithm": args.algorithm or "default",
    "threshold": args.threshold or "default",
    "detector_config": detector_cfg,
  }
  report_path = out_dir / "compare_report.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  log.info("Report: %s", json.dumps(report, indent=2))
  log.info("Report saved to %s", report_path)


if __name__ == "__main__":
  main()
