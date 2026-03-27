#!/usr/bin/env python3
"""Add is_one_off label to events JSONL.

A one-off event is one whose pattern (syscall_name, comm, path) appears exactly
once in the full dataset. Two-pass streaming: first count pattern frequencies,
then write each event with is_one_off: true/false.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _event_signature(obj: dict) -> tuple | None:
    """Return (syscall_name, comm, path) for frequency counting. None if not an event."""
    if "_meta" in obj or "event_id" not in obj:
        return None
    syscall_name = obj.get("syscall_name") or ""
    comm = obj.get("comm") or ""
    path = obj.get("path") or ""
    return (str(syscall_name), str(comm), str(path))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add is_one_off field to events JSONL (pattern appears exactly once)"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="artifacts/datasets/events_17_03_26_1M.jsonl",
        help="Input JSONL (default: artifacts/datasets/events_17_03_26_1M.jsonl)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL path (default: input with _one_off suffix before .jsonl)",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if args.output:
        out = Path(args.output)
    else:
        stem = inp.stem
        out = inp.parent / f"{stem}_one_off.jsonl"

    if not inp.exists():
        print(f"Error: input {inp} not found", file=sys.stderr)
        sys.exit(1)

    # Pass 1: count pattern frequencies
    counts: Counter[tuple] = Counter()
    n_events = 0
    with inp.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"json decode error: {e}", file=sys.stderr)
                continue
            sig = _event_signature(obj)
            if sig is not None:
                counts[sig] += 1
                n_events += 1
            if n_events > 0 and n_events % 200_000 == 0:
                print(f"Pass 1: counted {n_events:,} events...", file=sys.stderr)

    one_offs = {s for s, c in counts.items() if c == 1}
    print(f"Pass 1 done: {n_events:,} events, {len(counts):,} unique patterns, {len(one_offs):,} one-offs", file=sys.stderr)

    # Pass 2: write output with is_one_off
    written = 0
    with inp.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"json decode error: {e}", file=sys.stderr)
                continue
            sig = _event_signature(obj)
            if sig is None:
                # Keep metadata / non-event lines as-is (no is_one_off)
                fout.write(line + "\n")
                continue
            obj["is_one_off"] = sig in one_offs
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            if written % 200_000 == 0:
                print(f"Pass 2: written {written:,} events...", file=sys.stderr)

    print(f"Wrote {written:,} events to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
