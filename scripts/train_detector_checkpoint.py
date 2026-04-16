#!/usr/bin/env python3
"""Train a composed pipeline IDS on EVT1/JSONL and save a building_blocks checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import events_pb2
try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None
from detector.building_blocks import OnlineIDS, save_pipeline_checkpoint
from detector.building_blocks.core.base import DecisionOutput
from detector.config import detector_config_to_dict, load_config
from detector.pipelines import build_final_bb
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl

log = logging.getLogger(Path(__file__).stem)


def _dict_to_event_envelope(obj: dict) -> events_pb2.EventEnvelope:
  from event_envelope_codec import envelope_from_dict

  return envelope_from_dict(obj)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  ap = argparse.ArgumentParser(description="Train composed detector pipeline offline and save a checkpoint.")
  ap.add_argument("logfile", type=Path, help="Path to detector-events.jsonl or EVT1 events.bin (supports .gz).")
  ap.add_argument("--out", type=Path, required=True, help="Output checkpoint path (.pkl).")
  ap.add_argument("--score-dump", type=Path, default=None, help="Optional JSONL path to write per-event training scores.")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (for quick smoke tests).")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events.")
  args = ap.parse_args()

  path = Path(args.logfile)
  if not path.exists():
    log.error("File not found: %s", path)
    raise SystemExit(1)

  cfg = load_config()
  pipeline_id = str(getattr(cfg, "pipeline_id", "")).strip()
  if not pipeline_id:
    raise SystemExit("DETECTOR_PIPELINE_ID must be set to a registered pipeline id")
  final_bb = build_final_bb(cfg)
  ids = OnlineIDS(final_bb, pipeline_id=pipeline_id)

  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path, max_events=args.max_events, skip=args.skip) if fmt == "jsonl" else iter_events(path, max_events=args.max_events, skip=args.skip)

  if tqdm is not None:
    total = args.max_events if args.max_events is not None else None
    event_iter = tqdm(event_iter, total=total, desc="Train", unit=" evt", file=sys.stderr, leave=True)
  else:
    log.info("tqdm not installed; training without progress bar")

  n = 0
  dump_file = None
  if args.score_dump is not None:
    args.score_dump.parent.mkdir(parents=True, exist_ok=True)
    dump_file = args.score_dump.open("w", encoding="utf-8")
  try:
    for obj in event_iter:
      evt = _dict_to_event_envelope(obj)
      out = ids.run_event(evt)
      if not isinstance(out, DecisionOutput):
        raise TypeError("final_bb must write DecisionOutput, got %r" % type(out))
      raw = float(out.raw)
      scaled = float(out.scaled)
      if dump_file is not None:
        dump_file.write(
          '{"event_id": "%s", "event_group": "%s", "syscall_name": "%s", "score": %.10g, "score_raw": %.10g, "score_scaled": %.10g}\n'
          % (
            evt.event_id.replace('"', '\\"'),
            evt.event_group.replace('"', '\\"'),
            evt.syscall_name.replace('"', '\\"'),
            float(scaled),
            float(raw),
            float(scaled),
          )
        )
      n += 1
      if tqdm is None:
        if (n % 10000) == 0:
          log.info("Trained on %d events...", n)
  finally:
    if dump_file is not None:
      dump_file.close()

  if n == 0:
    log.error("No events processed; nothing to save.")
    raise SystemExit(2)

  save_pipeline_checkpoint(
    args.out,
    pipeline_id=pipeline_id,
    manager=ids.manager,
    checkpoint_index=n,
    extra={"detector_config": detector_config_to_dict(cfg)},
  )
  log.info("Saved checkpoint: %s (events learned: %d)", args.out, n)


if __name__ == "__main__":
  main()

