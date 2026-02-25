#!/usr/bin/env python3
"""Convert BETH CSV rows to EVT1 records used by replay_logs.py."""

import argparse
import ast
import csv
import json
import struct
import time
from pathlib import Path
from typing import Dict, List, Tuple

MAGIC = b"EVT1"


def _parse_int(raw: str, default: int = 0) -> int:
  try:
    return int(raw)
  except (TypeError, ValueError):
    return default


def _parse_float(raw: str, default: float = 0.0) -> float:
  try:
    return float(raw)
  except (TypeError, ValueError):
    return default


def _parse_args(raw: str) -> List[Dict[str, object]]:
  if not raw:
    return []
  try:
    obj = ast.literal_eval(raw)
  except (SyntaxError, ValueError):
    return []
  if not isinstance(obj, list):
    return []
  return [it for it in obj if isinstance(it, dict)]


def _extract_path_and_flags(args: List[Dict[str, object]]) -> Tuple[str, str]:
  path = ""
  flags = ""
  for arg in args:
    name = str(arg.get("name", ""))
    value = arg.get("value", "")
    if name in ("pathname", "filename", "oldname", "newname", "path"):
      path = str(value)
    if name in ("flags", "type"):
      flags = str(value)
  return path, flags


def _build_evt(row: Dict[str, str], row_idx: int, base_ts_ns: int, event_id_prefix: str) -> Tuple[Dict[str, object], Dict[str, object]]:
  ts_rel_s = _parse_float(row.get("timestamp", "0"))
  ts_ns = base_ts_ns + int(ts_rel_s * 1_000_000_000)
  event_name = (row.get("eventName") or "").strip() or "unknown"
  event_id = _parse_int(row.get("eventId", "0"))
  pid = _parse_int(row.get("processId", "0"))
  tid = _parse_int(row.get("threadId", "0"))
  uid = _parse_int(row.get("userId", "0"))
  comm = (row.get("processName") or "").strip()
  host = (row.get("hostName") or "").strip()
  ret = _parse_int(row.get("returnValue", "0"))

  args = _parse_args(row.get("args", ""))
  path, flags = _extract_path_and_flags(args)
  arg0 = str(args[0].get("value", "")) if len(args) > 0 else ""
  arg1 = str(args[1].get("value", "")) if len(args) > 1 else ""
  out_event_id = f"{event_id_prefix}-{row_idx}-{pid}-{tid}"

  payload = {
    "event_id": out_event_id,
    "ts_unix_nano": ts_ns,
    "hostname": host,
    "pod": "",
    "namespace": "",
    "container_id": "",
    "event_type": event_name,
    "data": [
      event_name,
      str(event_id),
      comm,
      str(pid),
      str(tid),
      str(uid),
      arg0,
      arg1,
      path,
      flags,
    ],
    "attributes": {
      "mount_namespace": str(row.get("mountNamespace", "")),
      "return_value": str(ret),
      "args_num": str(row.get("argsNum", "")),
      "sus": str(row.get("sus", "0")),
      "evil": str(row.get("evil", "0")),
    },
  }
  label_row = {
    "event_id": out_event_id,
    "sus": _parse_int(row.get("sus", "0")),
    "evil": _parse_int(row.get("evil", "0")),
    "event_type": event_name,
  }
  return payload, label_row


def convert(csv_file: Path, evt1_out: Path, labels_out: Path, limit: int = 0, event_id_prefix: str = "beth") -> int:
  base_ts_ns = int(time.time() * 1_000_000_000)
  written = 0
  evt1_out.parent.mkdir(parents=True, exist_ok=True)
  labels_out.parent.mkdir(parents=True, exist_ok=True)

  with csv_file.open("r", encoding="utf-8", newline="") as src, evt1_out.open("wb") as evtf, labels_out.open("w", encoding="utf-8") as labf:
    reader = csv.DictReader(src)
    for idx, row in enumerate(reader):
      if limit > 0 and written >= limit:
        break
      payload, label_row = _build_evt(row, idx, base_ts_ns, event_id_prefix=event_id_prefix)
      blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
      evtf.write(MAGIC + struct.pack("<I", len(blob)) + blob)
      labf.write(json.dumps(label_row) + "\n")
      written += 1
  return written


def main() -> None:
  ap = argparse.ArgumentParser(description="Convert BETH CSV into EVT1 + labels NDJSON")
  ap.add_argument("csv_file", help="Input BETH CSV file")
  ap.add_argument("--evt1-out", default="test_data/beth/events_from_beth.bin", help="Output EVT1 path")
  ap.add_argument("--labels-out", default="test_data/beth/events_from_beth.labels.ndjson", help="Output labels path")
  ap.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all rows)")
  ap.add_argument("--event-id-prefix", default="beth", help="Prefix for event_id values (use different prefixes for train/test)")
  args = ap.parse_args()

  count = convert(
    Path(args.csv_file),
    Path(args.evt1_out),
    Path(args.labels_out),
    limit=args.limit,
    event_id_prefix=args.event_id_prefix,
  )
  print(f"converted_rows={count}")
  print(f"evt1={args.evt1_out}")
  print(f"labels={args.labels_out}")


if __name__ == "__main__":
  main()

