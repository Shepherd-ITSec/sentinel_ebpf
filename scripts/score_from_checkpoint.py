#!/usr/bin/env python3
"""Load a detector checkpoint and score a JSONL/EVT1 event stream."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl
from scripts.train_detector_checkpoint import _dict_to_event_envelope, _make_detector

log = logging.getLogger(Path(__file__).stem)


def _iter_event_dicts(path: Path, *, max_events: int | None = None, skip: int | None = None):
  fmt = _detect_format(path)
  if fmt == "jsonl":
    yield from iter_events_jsonl(path, max_events=max_events, skip=skip)
    return
  yield from iter_events(path, max_events=max_events, skip=skip)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  ap = argparse.ArgumentParser(description="Load checkpoint and score a JSONL/EVT1 stream.")
  ap.add_argument("logfile", type=Path, help="Path to detector-events.jsonl or EVT1 events.bin (supports .gz).")
  ap.add_argument("--algorithm", default="sequence_mlp", help="Detector algorithm (default: sequence_mlp).")
  ap.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.pkl) to load before scoring.")
  ap.add_argument("--out", type=Path, required=True, help="Output JSONL path with per-event scores.")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (for quick smoke tests).")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events.")
  args = ap.parse_args()

  if not args.logfile.exists():
    raise SystemExit(f"File not found: {args.logfile}")
  if not args.checkpoint.exists():
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

  detector, feature_fn = _make_detector(args.algorithm)
  detector.load_checkpoint(args.checkpoint)

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  n = 0
  event_iter = _iter_event_dicts(args.logfile, max_events=args.max_events, skip=args.skip)
  if tqdm is not None:
    total = args.max_events if args.max_events is not None else None
    event_iter = tqdm(event_iter, total=total, desc="Score events", unit=" evt", file=sys.stderr, leave=True)
  else:
    log.info("tqdm not installed; scoring without progress bar")
  with out_path.open("w", encoding="utf-8") as f:
    for obj in event_iter:
      evt = _dict_to_event_envelope(obj)
      raw, scaled = detector.score_and_learn_event(evt, feature_fn=feature_fn)
      f.write(
        json.dumps(
          {
            "event_id": evt.event_id,
            "event_group": evt.event_group,
            "syscall_name": evt.syscall_name,
            "score_raw": float(raw),
            "score_scaled": float(scaled),
          }
        )
        + "\n"
      )
      n += 1
      if (n % 10000) == 0:
        log.info("Scored %d events...", n)

  log.info("Wrote %d scored events to %s", n, out_path)


if __name__ == "__main__":
  main()

