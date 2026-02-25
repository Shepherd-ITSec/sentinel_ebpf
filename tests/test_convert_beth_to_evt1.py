"""Tests for scripts/convert_beth_to_evt1.py."""

import json
import struct
from pathlib import Path

from scripts.convert_beth_to_evt1 import MAGIC, _build_evt, convert


def test_build_evt_uses_event_id_prefix():
  row = {
    "timestamp": "1.0",
    "processId": "10",
    "threadId": "11",
    "userId": "0",
    "mountNamespace": "4026531840",
    "processName": "bash",
    "hostName": "node-a",
    "eventId": "257",
    "eventName": "openat",
    "argsNum": "1",
    "returnValue": "0",
    "args": "[]",
    "sus": "0",
    "evil": "0",
  }
  payload, label = _build_evt(row, row_idx=7, base_ts_ns=1_700_000_000_000_000_000, event_id_prefix="beth-train")
  assert payload["event_id"] == "beth-train-7-10-11"
  assert label["event_id"] == "beth-train-7-10-11"


def test_convert_writes_prefixed_ids(temp_dir):
  csv_path = Path(temp_dir) / "in.csv"
  out_evt = Path(temp_dir) / "out.evt1"
  out_labels = Path(temp_dir) / "out.labels.ndjson"

  csv_path.write_text(
    "timestamp,processId,threadId,parentProcessId,userId,mountNamespace,processName,hostName,eventId,eventName,stackAddresses,argsNum,returnValue,args,sus,evil\n"
    "1.0,10,10,1,0,4026531840,bash,node,59,execve,[],1,0,[],0,0\n",
    encoding="utf-8",
  )

  n = convert(csv_path, out_evt, out_labels, event_id_prefix="beth-test")
  assert n == 1

  blob = out_evt.read_bytes()
  assert blob.startswith(MAGIC)
  payload_len = struct.unpack("<I", blob[4:8])[0]
  payload = json.loads(blob[8 : 8 + payload_len].decode("utf-8"))
  assert payload["event_id"].startswith("beth-test-")

  label_line = out_labels.read_text(encoding="utf-8").strip()
  label = json.loads(label_line)
  assert label["event_id"].startswith("beth-test-")
