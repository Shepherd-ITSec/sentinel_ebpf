#!/usr/bin/env python3
"""Split a combined replay_dump.jsonl (from multiple runs appending to same file) into per-run files."""

import sys
from pathlib import Path

# Source: the bloated dump with 4 runs concatenated
SRC = Path("test_data/compare_replay_freq1d_09_scaled/replay_dump.jsonl")
EVENTS_PER_RUN = 5_114_947

OUTPUTS = [
  ("test_data/compare_replay_freq1d_09_scaled/replay_dump_freq1d.jsonl", "freq1d"),
  ("test_data/compare_replay_loda_09_scaled/replay_dump_loda.jsonl", "loda"),
  ("test_data/compare_replay_kitnet_09_AE2/replay_dump_kitnet_AE2.jsonl", "kitnet_AE2"),
  ("test_data/compare_replay_kitnet_09_scaled_AE2/replay_dump_kitnet_scaled_AE2.jsonl", "kitnet_scaled_AE2"),
]


def main():
  if not SRC.exists():
    print(f"Error: {SRC} not found", file=sys.stderr)
    sys.exit(1)

  out_paths = [Path(p) for p, _ in OUTPUTS]
  for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)

  files = [p.open("w", encoding="utf-8") for p in out_paths]
  try:
    line_num = 0
    chunk = 0
    with SRC.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        line_num += 1
        idx = (line_num - 1) // EVENTS_PER_RUN
        if idx >= len(files):
          print(f"Warning: extra lines after chunk 4 (line {line_num}), skipping", file=sys.stderr)
          continue
        files[idx].write(line + "\n")
        if line_num % 1_000_000 == 0:
          print(f"  {line_num:,} lines...", file=sys.stderr)
  finally:
    for f in files:
      f.close()

  for (path, name) in OUTPUTS:
    p = Path(path)
    n = sum(1 for _ in p.open("r", encoding="utf-8")) if p.exists() else 0
    print(f"{path}: {n:,} lines")


if __name__ == "__main__":
  main()
