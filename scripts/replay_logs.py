#!/usr/bin/env python3
import argparse
import gzip
import json
import logging
import struct
import time
from pathlib import Path

import grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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


def iter_events(path: Path, start_ms=None, end_ms=None):
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
      yield obj


def replay(path, target, pace, start_ms, end_ms):
  log.info("replay: %s -> %s (pace=%s)", path, target, pace)
  channel = grpc.insecure_channel(target)
  stub = events_pb2_grpc.DetectorServiceStub(channel)

  def gen():
    first_ts = None
    start_wall = None
    for obj in iter_events(Path(path), start_ms, end_ms):
      data_field = obj.get("data", [])
      if not isinstance(data_field, list):
        raise ValueError("invalid event record: 'data' must be an ordered list")
      env = events_pb2.EventEnvelope(
        event_id=obj.get("event_id", ""),
        hostname=obj.get("hostname", ""),
        pod_name=obj.get("pod", ""),
        namespace=obj.get("namespace", ""),
        container_id=obj.get("container_id", ""),
        ts_unix_nano=int(obj.get("ts_unix_nano", 0)),
        event_type=obj.get("event_type", ""),
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

  n = 0
  for resp in stub.StreamEvents(gen()):
    n += 1
    if (n % 10000) == 0:
      log.info("replay: %d events sent", n)
  log.info("replay done: %d events sent", n)


def main():
  ap = argparse.ArgumentParser(description="Replay EVT1 logs to DetectorService.StreamEvents")
  ap.add_argument("logfile", help="Path to events.bin (EVT1), supports .gz")
  ap.add_argument("--target", default="localhost:50051", help="Detector gRPC endpoint")
  ap.add_argument("--pace", choices=["realtime", "fast"], default="fast", help="Pace replay in realtime or as fast as possible")
  ap.add_argument("--start-ms", type=int, default=None, help="Start timestamp filter (ms since epoch)")
  ap.add_argument("--end-ms", type=int, default=None, help="End timestamp filter (ms since epoch)")
  args = ap.parse_args()

  replay(args.logfile, args.target, args.pace, args.start_ms, args.end_ms)


if __name__ == "__main__":
  main()
