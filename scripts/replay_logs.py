#!/usr/bin/env python3
import argparse
import gzip
import json
import logging
import struct
import sys
import time
from pathlib import Path

import grpc
try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

import events_pb2
import events_pb2_grpc

MAGIC = b"EVT1"


def open_stream(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rb")
  return path.open("rb")


def _detect_format(path: Path) -> str:
  """Return 'evt1' if file starts with EVT1 magic, else 'jsonl' (detector-events.jsonl style)."""
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    with gzip.open(path, "rb") as zf:
      first = zf.read(4)
    return "evt1" if first == MAGIC else "jsonl"
  with path.open("rb") as f:
    first = f.read(4)
  return "evt1" if first == MAGIC else "jsonl"


def _open_text_lines(path: Path):
  """Open path for reading text lines (supports .gz)."""
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rt", encoding="utf-8")
  return path.open("r", encoding="utf-8")


def iter_events(path: Path, start_ms=None, end_ms=None, max_events=None, skip=None):
  """Yield event dicts from an EVT1 file. Optional: start_ms, end_ms (timestamp filter), max_events (stop after N events), skip (skip first N events)."""
  n = 0
  skipped = 0
  with open_stream(path) as f:
    while True:
      magic = f.read(4)
      if not magic:
        break
      if magic != MAGIC:
        raise ValueError(f"bad magic at offset {f.tell()-4}")
      raw_len = f.read(4)
      if len(raw_len) < 4:
        break
      (length,) = struct.unpack("<I", raw_len)
      payload = f.read(length)
      if len(payload) < length:
        break
      obj = json.loads(payload.decode("utf-8"))
      ts_ms = obj.get("ts_unix_nano", 0) // 1_000_000
      if start_ms is not None and ts_ms < start_ms:
        continue
      if end_ms is not None and ts_ms > end_ms:
        break
      if skip is not None and skipped < skip:
        skipped += 1
        continue
      if max_events is not None and n >= max_events:
        return
      n += 1
      yield obj


def iter_events_jsonl(path: Path, start_ms=None, end_ms=None, max_events=None, skip=None):
  """Yield event dicts from a detector-events.jsonl file (lossless dump). Skips lines that are not event records (e.g. no event_id)."""
  n = 0
  skipped = 0
  with _open_text_lines(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "event_id" not in obj:
        continue
      ts_ns = obj.get("ts_unix_nano", 0)
      ts_ms = ts_ns // 1_000_000
      if start_ms is not None and ts_ms < start_ms:
        continue
      if end_ms is not None and ts_ms > end_ms:
        continue
      if skip is not None and skipped < skip:
        skipped += 1
        continue
      if max_events is not None and n >= max_events:
        return
      n += 1
      yield obj


def replay(path, target, pace, start_ms, end_ms, total=None, label="Replay", max_events=None, skip=None):
  path = Path(path)
  fmt = _detect_format(path)
  extra = []
  if max_events is not None:
    extra.append(f"max_events={max_events}")
  if skip is not None:
    extra.append(f"skip={skip}")
  log.info("replay: %s -> %s (format=%s, pace=%s%s)", path, target, fmt, pace, f" {' '.join(extra)}" if extra else "")
  channel = grpc.insecure_channel(target)
  stub = events_pb2_grpc.DetectorServiceStub(channel)

  event_iter = iter_events_jsonl(path, start_ms, end_ms, max_events, skip) if fmt == "jsonl" else iter_events(path, start_ms, end_ms, max_events, skip)

  def gen():
    first_ts = None
    start_wall = None
    for obj in event_iter:
      data_field = obj.get("data", [])
      if not isinstance(data_field, list):
        raise ValueError("invalid event record: 'data' must be an ordered list")
      # event_name = syscall name; event_type = category. Support old payloads that only had event_type (syscall name).
      if "event_name" in obj:
        event_name = obj.get("event_name", "") or (data_field[0] if data_field else "")
        event_type = obj.get("event_type", "")
      else:
        event_name = obj.get("event_type", "") or (data_field[0] if data_field else "")
        event_type = ""
      # JSONL dump uses pod_name; EVT1 uses pod
      pod_name = obj.get("pod_name", obj.get("pod", ""))
      env = events_pb2.EventEnvelope(
        event_id=obj.get("event_id", ""),
        hostname=obj.get("hostname", ""),
        pod_name=pod_name,
        namespace=obj.get("namespace", ""),
        container_id=obj.get("container_id", ""),
        ts_unix_nano=int(obj.get("ts_unix_nano", 0)),
        event_name=event_name,
        event_type=event_type,
        data=data_field,
        attributes=dict(obj.get("attributes", {}) or {}),
      )
      if pace == "realtime":
        ts = env.ts_unix_nano / 1_000_000_000
        if first_ts is None:
          first_ts = ts
          start_wall = time.time()
        else:
          elapsed_sim = ts - first_ts
          target_wall = start_wall + elapsed_sim
          now = time.time()
          if target_wall > now:
            time.sleep(target_wall - now)
      yield env

  total = total or 0
  show_progress = total > 0 and tqdm is not None
  if total > 0 and tqdm is None:
    log.info("tqdm not installed; replaying without progress bar")
  pbar = (
    tqdm(total=total, desc=label, unit=" evt", file=sys.stderr, leave=True)
    if show_progress
    else None
  )
  n = 0
  for resp in stub.StreamEvents(gen()):
    n += 1
    if pbar:
      pbar.update(1)
    elif (n % 10000) == 0:
      log.info("Replay: %d events sent", n)
  if pbar:
    pbar.close()
  log.info("Replay: Done: %d events sent", n)


def main():
  ap = argparse.ArgumentParser(description="Replay event logs to DetectorService.StreamEvents. Supports EVT1 (events.bin) or detector-events.jsonl (lossless dump).")
  ap.add_argument("logfile", help="Path to events.bin (EVT1) or detector-events.jsonl (supports .gz for EVT1)")
  ap.add_argument("--target", default="localhost:50051", help="Detector gRPC endpoint")
  ap.add_argument("--pace", choices=["realtime", "fast"], default="fast", help="Pace replay in realtime or as fast as possible")
  ap.add_argument("--start-ms", type=int, default=None, help="Start timestamp filter (ms since epoch)")
  ap.add_argument("--end-ms", type=int, default=None, help="End timestamp filter (ms since epoch)")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after replaying this many events (for warmup)")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events before replaying (e.g. for second-half replay)")
  args = ap.parse_args()

  replay(args.logfile, args.target, args.pace, args.start_ms, args.end_ms, max_events=args.max_events, skip=args.skip)


if __name__ == "__main__":
  main()
