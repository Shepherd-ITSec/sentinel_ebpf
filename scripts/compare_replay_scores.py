#!/usr/bin/env python3
"""Replay a slice of detector-events JSONL and compare model scores with original log.

Define a starting point (--start-event or --start-ms). From there, the reduced dataset
is split: first half = warmup, second half = replay. Compares replay scores with the
scores originally written into the log.
"""

import argparse
import gzip
import json
import math
import statistics
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

if __package__ is None or __package__ == "":
  sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_logs import replay

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
) -> Tuple[int, int, Dict[str, float]]:
  """Count events from start, load original scores for second half of slice. Returns (total_in_slice, half, {event_id: score})."""
  # First pass: count events in the slice (from start_event or start_ms onwards)
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
  half = total_in_slice // 2
  second_half_scores: Dict[str, float] = {}
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
      if n_in_slice < half:
        n_in_slice += 1
        continue
      eid = str(obj.get("event_id", ""))
      if "score" in obj:
        second_half_scores[eid] = float(obj["score"])
      n_in_slice += 1
      if n_in_slice >= total_in_slice:
        break

  return total_in_slice, half, second_half_scores


def _load_replay_scores(dump_path: Path, expected_count: int) -> Dict[str, float]:
  """Load scores from event dump. The last expected_count lines are from the second-half replay."""
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


def _start_detector(port: int, env_overrides: Optional[Dict[str, str]] = None) -> subprocess.Popen:
  env = os.environ.copy()
  env["DETECTOR_PORT"] = str(port)
  env["DETECTOR_QUIET"] = "1"
  if env_overrides:
    env.update(env_overrides)
  cmd = [sys.executable, "-m", "detector.server"]
  return subprocess.Popen(
    cmd,
    env=env,
    cwd=Path(__file__).resolve().parent.parent,
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
  ap = argparse.ArgumentParser(
    description="Replay second half of detector-events JSONL and compare model scores with original log.",
  )
  ap.add_argument("events", default="events_09_03_26.jsonl", nargs="?", help="Path to detector-events JSONL")
  ap.add_argument("--out-dir", default="test_data/compare_replay", help="Output directory for dump and report")
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast", help="Replay pace")
  ap.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for detector")
  ap.add_argument("--algorithm", default=None, help="Model algorithm (kitnet, loda, halfspacetrees, memstream)")
  ap.add_argument("--threshold", default=None, help="Anomaly threshold (e.g. 0.7)")
  ap.add_argument("--limit", type=int, default=0, help="Cap total events in slice for quick testing (0=all)")
  ap.add_argument("--start-event", type=int, default=0, help="Skip first N events; warmup+replay on the rest")
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
  total, half, original_scores = _count_and_load_slice(
    events_path,
    start_event=args.start_event,
    start_ms=args.start_ms,
    limit=args.limit,
  )
  second_half_count = total - half
  log.info("Slice: %d events, warmup=%d, replay=%d", total, half, second_half_count)

  env_overrides: Dict[str, str] = {"EVENT_DUMP_PATH": str(dump_path)}
  if args.algorithm:
    env_overrides["DETECTOR_MODEL_ALGORITHM"] = args.algorithm
  if args.threshold:
    env_overrides["DETECTOR_THRESHOLD"] = args.threshold

  log.info("Starting detector (EVENT_DUMP_PATH=%s)...", dump_path)
  detector = _start_detector(args.detector_port, env_overrides=env_overrides)
  target = f"localhost:{args.detector_port}"
  warmup_skip = 0 if args.start_ms is not None else args.start_event
  replay_skip = half if args.start_ms is not None else args.start_event + half

  try:
    _wait_for_detector(target, timeout_s=args.startup_timeout)
    log.info("Warmup: first half of slice (%d events)...", half)
    replay(
      str(events_path),
      target,
      args.pace,
      start_ms=args.start_ms,
      end_ms=None,
      total=half,
      label="Warmup",
      max_events=half,
      skip=warmup_skip,
    )
    log.info("Replay: second half of slice (%d events)...", second_half_count)
    replay(
      str(events_path),
      target,
      args.pace,
      start_ms=args.start_ms,
      end_ms=None,
      total=second_half_count,
      label="SecondHalf",
      skip=replay_skip,
      max_events=second_half_count,
    )
  finally:
    _stop_detector(detector)

  log.info("Loading replay scores from dump...")
  replay_scores = _load_replay_scores(dump_path, second_half_count)

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
    "first_half_warmup": half,
    "second_half_replay": second_half_count,
    "common_event_ids": len(common),
    "mae": round(mae, 6),
    "max_abs_diff": round(max_diff, 6),
    "correlation": round(corr, 6) if not math.isnan(corr) else None,
    "algorithm": args.algorithm or "default",
    "threshold": args.threshold or "default",
  }
  report_path = out_dir / "compare_report.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  log.info("Report: %s", json.dumps(report, indent=2))
  log.info("Report saved to %s", report_path)


if __name__ == "__main__":
  main()
