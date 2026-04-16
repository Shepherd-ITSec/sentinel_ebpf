#!/usr/bin/env python3
"""Load a detector checkpoint and score a JSONL/EVT1 event stream."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Iterator

try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None

from detector.building_blocks import OnlineIDS, load_pipeline_checkpoint
from detector.building_blocks.core.base import DecisionOutput
from detector.config import detector_config_from_dict
from detector.pipelines import build_final_bb
from scripts.performance_measurement import PerformanceEvent, PerformanceMeasurement
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl
from scripts.train_detector_checkpoint import _dict_to_event_envelope

log = logging.getLogger(Path(__file__).stem)


def _label_from_row(obj: dict[str, Any]) -> bool | None:
  malicious = obj.get("malicious")
  if isinstance(malicious, bool):
    return malicious
  return None


def _default_summary_path(out_path: Path) -> Path:
  return out_path.with_name(f"{out_path.stem}.summary.json")


def _iter_event_dicts(
  path: Path,
  *,
  max_events: int | None = None,
  max_recordings: int | None = None,
  skip: int | None = None,
) -> Iterator[dict]:
  fmt = _detect_format(path)
  recording_names_seen: set[str] = set()
  recording_count = 0
  if fmt == "jsonl":
    base_iter = iter_events_jsonl(path, max_events=max_events, skip=skip)
  else:
    base_iter = iter_events(path, max_events=max_events, skip=skip)

  if max_recordings is None:
    yield from base_iter
    return

  for obj in base_iter:
    recording_name = obj.get("lidds_recording_name")
    if not isinstance(recording_name, str) or not recording_name:
      raise ValueError("max_recordings requires each event to have a non-empty lidds_recording_name")
    if recording_name not in recording_names_seen:
      if recording_count >= max_recordings:
        return
      recording_names_seen.add(recording_name)
      recording_count += 1
    yield obj


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  ap = argparse.ArgumentParser(description="Load checkpoint and score a JSONL/EVT1 stream.")
  ap.add_argument("logfile", type=Path, help="Path to detector-events.jsonl or EVT1 events.bin (supports .gz).")
  ap.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.pkl) to load before scoring.")
  ap.add_argument("--out", type=Path, required=True, help="Output JSONL path with per-event scores.")
  ap.add_argument(
    "--summary-out",
    type=Path,
    default=None,
    help="Optional output JSON path for aggregate metrics. Defaults to <out>.summary.json.",
  )
  ap.add_argument("--threshold", type=float, default=None, help="Override anomaly threshold. Defaults to detector config threshold.")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (for quick smoke tests).")
  ap.add_argument("--max-recordings", type=int, default=None, help="Stop after scoring N distinct lidds_recording_name values.")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events.")
  args = ap.parse_args()

  if not args.logfile.exists():
    raise SystemExit(f"File not found: {args.logfile}")
  if not args.checkpoint.exists():
    raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

  with Path(args.checkpoint).open("rb") as f:
    blob = pickle.load(f)
  if not (isinstance(blob, dict) and blob.get("format") == "building_blocks_v1"):
    raise SystemExit("Checkpoint must be in building_blocks_v1 format")
  ck_pid = (blob.get("pipeline_id") or "").strip().lower()
  extra = blob.get("extra")
  if not isinstance(extra, dict):
    raise SystemExit("Checkpoint missing extra metadata")
  cfg_blob = extra.get("detector_config")
  if not isinstance(cfg_blob, dict):
    raise SystemExit("Checkpoint missing detector_config snapshot")
  cfg = detector_config_from_dict(cfg_blob)
  pipeline_id = str(getattr(cfg, "pipeline_id", "")).strip().lower()
  if ck_pid and ck_pid != pipeline_id:
    raise SystemExit(f"Checkpoint pipeline_id={ck_pid!r} does not match checkpoint detector_config={pipeline_id!r}")
  if not pipeline_id:
    raise SystemExit("Checkpoint detector_config must define a registered pipeline_id")
  if args.threshold is not None:
    cfg.threshold = float(args.threshold)
  threshold = float(cfg.threshold)
  final_bb = build_final_bb(cfg)
  ids = OnlineIDS(final_bb, pipeline_id=pipeline_id)
  load_pipeline_checkpoint(Path(args.checkpoint), ids.manager)

  warmup_events = int(getattr(cfg, "warmup_events", 0))
  warmup_suppress = bool(getattr(cfg, "suppress_anomalies_during_warmup", False))
  score_mode = str(getattr(cfg, "score_mode", "raw"))

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  summary_path = Path(args.summary_out) if args.summary_out is not None else _default_summary_path(out_path)
  summary_path.parent.mkdir(parents=True, exist_ok=True)

  n = 0
  anomaly_count = 0
  labeled_rows = 0
  perf = PerformanceMeasurement()
  perf.set_threshold(threshold)
  event_iter = _iter_event_dicts(
    args.logfile,
    max_events=args.max_events,
    max_recordings=args.max_recordings,
    skip=args.skip,
  )
  if tqdm is not None:
    total = args.max_events if args.max_events is not None else None
    event_iter = tqdm(event_iter, total=total, desc="Score events", unit=" evt", file=sys.stderr, leave=True)
  else:
    log.info("tqdm not installed; scoring without progress bar")
  with out_path.open("w", encoding="utf-8") as f:
    for obj in event_iter:
      evt = _dict_to_event_envelope(obj)
      out = ids.run_event(evt)
      if not isinstance(out, DecisionOutput):
        raise TypeError("final_bb must write DecisionOutput, got %r" % type(out))
      raw = float(out.raw)
      scaled = float(out.scaled)
      score_primary = float(out.primary)
      predicted_anomaly = bool(out.anomaly)
      suppress_primary = bool(out.suppressed)
      score_mode = str(out.score_mode)
      threshold = float(out.threshold)
      expected_anomaly = _label_from_row(obj)
      if predicted_anomaly:
        anomaly_count += 1
      if expected_anomaly is not None:
        labeled_rows += 1
        recording_name = obj.get("lidds_recording_name")
        perf.add_event(
          PerformanceEvent(
            predicted_anomaly=predicted_anomaly,
            expected_anomaly=expected_anomaly,
            recording_name=recording_name if isinstance(recording_name, str) and recording_name else None,
            ts_unix_nano=int(obj["ts_unix_nano"]) if isinstance(obj.get("ts_unix_nano"), int) else None,
          )
        )
      f.write(
        json.dumps(
          {
            "event_id": evt.event_id,
            "event_group": evt.event_group,
            "syscall_name": evt.syscall_name,
            "score_raw": float(raw),
            "score_scaled": float(scaled),
            "score_primary": float(score_primary),
            "score_mode": score_mode,
            "suppress_primary": suppress_primary,
            "threshold": threshold,
            "anomaly": predicted_anomaly,
            "expected_anomaly": expected_anomaly,
          }
        )
        + "\n"
      )
      n += 1
      if tqdm is None:
        if (n % 10000) == 0:
          log.info("Scored %d events...", n)

  summary_payload: dict[str, Any] = {
    "logfile": str(args.logfile),
    "pipeline_id": pipeline_id,
    "checkpoint": str(args.checkpoint),
    "out": str(out_path),
    "summary_out": str(summary_path),
    "threshold": threshold,
    "score_mode": score_mode,
    "warmup_events": warmup_events,
    "suppress_anomalies_during_warmup": warmup_suppress,
    "events_scored": n,
    "predicted_anomalies": anomaly_count,
    "labeled_events": labeled_rows,
  }
  log.info("Wrote %d scored events to %s (threshold=%.6f, predicted_anomalies=%d)", n, out_path, threshold, anomaly_count)
  if labeled_rows:
    performance = perf.get_results()
    summary_payload["performance"] = performance
    log.info("LID-DS-style performance evaluation: %s", json.dumps(performance, sort_keys=True))
  else:
    summary_payload["performance"] = None
    log.info("Evaluation skipped: no boolean malicious labels found in scored rows")
  summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  log.info("Wrote summary metrics to %s", summary_path)


if __name__ == "__main__":
  main()

