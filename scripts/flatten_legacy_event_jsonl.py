#!/usr/bin/env python3
"""Rewrite JSONL events: expand legacy top-level ``data`` array into envelope fields (see legacy_event_jsonl)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from legacy_event_jsonl import expand_legacy_data_field


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("path", type=Path, help="Input JSONL path")
  ap.add_argument("--in-place", action="store_true", help="Atomically replace the input file")
  ap.add_argument("-o", "--output", type=Path, default=None, help="Output JSONL (required if not --in-place)")
  args = ap.parse_args()
  if not args.in_place and args.output is None:
    ap.error("use --in-place or -o OUTPUT")
  inp = args.path
  if args.in_place:
    out_path = inp.with_name(inp.name + ".tmp")
  else:
    out_path = args.output

  n_in = n_out = n_skipped = 0
  with inp.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
      n_in += 1
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        n_skipped += 1
        print(f"skip line {n_in}: JSONDecodeError", file=sys.stderr)
        continue
      if not isinstance(obj, dict):
        n_skipped += 1
        print(f"skip line {n_in}: not a dict", file=sys.stderr)
        continue
      fixed = expand_legacy_data_field(obj)
      fout.write(json.dumps(fixed, ensure_ascii=False, separators=(",", ":")) + "\n")
      n_out += 1

  if args.in_place:
    os.replace(out_path, inp)
  print(f"lines_read={n_in} records_written={n_out} skipped={n_skipped}", file=sys.stderr)


if __name__ == "__main__":
  main()
