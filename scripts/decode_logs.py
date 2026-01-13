#!/usr/bin/env python3
import gzip
import json
import struct
import sys
from pathlib import Path

MAGIC = b"EVT1"


def open_stream(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rb")
  return path.open("rb")


def decode(path: Path, out):
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
      out.write(json.dumps(obj))
      out.write("\n")


def main():
  if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <events.bin or .gz> [output.ndjson]", file=sys.stderr)
    sys.exit(1)
  path = Path(sys.argv[1])
  if len(sys.argv) > 2:
    with open(sys.argv[2], "w", encoding="utf-8") as out:
      decode(path, out)
  else:
    decode(path, sys.stdout)


if __name__ == "__main__":
  main()
