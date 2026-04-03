#!/usr/bin/env python3
"""Load a detector checkpoint and score a JSONL/EVT1 event stream."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator
try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None
from detector.config import load_config
from detector.model import OnlinePercentileCalibrator
from detector.scoring import (
  anomaly_from_primary,
  compute_primary_score,
  event_group_key,
  get_or_create_percentile_calibrator,
)
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl
from scripts.train_detector_checkpoint import _dict_to_event_envelope, _make_detector

log = logging.getLogger(Path(__file__).stem)


def _label_from_row(obj: dict[str, Any]) -> bool | None:
  malicious = obj.get("malicious")
  if isinstance(malicious, bool):
    return malicious
  return None


def _confusion_summary(rows: list[tuple[bool, bool]]) -> dict[str, float | int]:
  tp = fp = tn = fn = 0
  for predicted_anomaly, expected_anomaly in rows:
    if predicted_anomaly and expected_anomaly:
      tp += 1
    elif predicted_anomaly and not expected_anomaly:
      fp += 1
    elif (not predicted_anomaly) and (not expected_anomaly):
      tn += 1
    else:
      fn += 1

  total = len(rows)
  flagged_samples = tp + fp
  precision = tp / (tp + fp) if (tp + fp) else 0.0
  recall = tp / (tp + fn) if (tp + fn) else 0.0
  f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
  accuracy = (tp + tn) / total if total else 0.0
  return {
    "samples": total,
    "flagged_samples": flagged_samples,
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "flagged_rate": (flagged_samples / total) if total else 0.0,
  }


def _recording_detection_summary(
  rows: list[tuple[str, bool, bool]],
) -> dict[str, float | int]:
  by_recording: dict[str, dict[str, bool]] = {}
  for recording_name, predicted_anomaly, expected_anomaly in rows:
    entry = by_recording.setdefault(
      recording_name,
      {
        "has_attack_region": False,
        "detected_attack": False,
        "has_any_alarm": False,
      },
    )
    entry["has_any_alarm"] = entry["has_any_alarm"] or predicted_anomaly
    entry["has_attack_region"] = entry["has_attack_region"] or expected_anomaly
    if predicted_anomaly and expected_anomaly:
      entry["detected_attack"] = True

  attack_recordings = sum(1 for entry in by_recording.values() if entry["has_attack_region"])
  benign_recordings = sum(1 for entry in by_recording.values() if not entry["has_attack_region"])
  detected_attack_recordings = sum(1 for entry in by_recording.values() if entry["detected_attack"])
  missed_attack_recordings = attack_recordings - detected_attack_recordings
  benign_recordings_with_alarm = sum(
    1
    for entry in by_recording.values()
    if (not entry["has_attack_region"]) and entry["has_any_alarm"]
  )

  return {
    "recordings": len(by_recording),
    "attack_recordings": attack_recordings,
    "benign_recordings": benign_recordings,
    "detected_attack_recordings": detected_attack_recordings,
    "missed_attack_recordings": missed_attack_recordings,
    "benign_recordings_with_alarm": benign_recordings_with_alarm,
    "detection_rate": (detected_attack_recordings / attack_recordings) if attack_recordings else 0.0,
    "benign_recording_alarm_rate": (
      benign_recordings_with_alarm / benign_recordings
    ) if benign_recordings else 0.0,
  }


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
  ap.add_argument("--algorithm", default="sequence_mlp", help="Detector algorithm (default: sequence_mlp).")
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

  cfg = load_config()
  threshold = float(args.threshold) if args.threshold is not None else float(cfg.threshold)
  detector, feature_fn = _make_detector(args.algorithm)
  detector.load_checkpoint(args.checkpoint)

  percentiles: dict[str, OnlinePercentileCalibrator] = {}
  warmup_counts: dict[str, int] = {}
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
  confusion_rows: list[tuple[bool, bool]] = []
  recording_rows: list[tuple[str, bool, bool]] = []
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
      key = event_group_key(evt.event_group or "")
      warmup_counts[key] = warmup_counts.get(key, 0) + 1
      suppress_primary = warmup_suppress and warmup_counts[key] <= warmup_events
      raw, scaled = detector.score_and_learn_event(evt, feature_fn=feature_fn)
      percentile_cal = None
      if (not suppress_primary) and score_mode == "percentile":
        percentile_cal = get_or_create_percentile_calibrator(percentiles, key, cfg)
      score_primary = compute_primary_score(
        raw,
        scaled,
        score_mode=score_mode,
        suppress_primary=suppress_primary,
        percentile_cal=percentile_cal,
      )
      predicted_anomaly = anomaly_from_primary(
        score_primary,
        threshold,
        suppress_primary=suppress_primary,
      )
      expected_anomaly = _label_from_row(obj)
      if predicted_anomaly:
        anomaly_count += 1
      if expected_anomaly is not None:
        labeled_rows += 1
        confusion_rows.append((predicted_anomaly, expected_anomaly))
        recording_name = obj.get("lidds_recording_name")
        if isinstance(recording_name, str) and recording_name:
          recording_rows.append((recording_name, predicted_anomaly, expected_anomaly))
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
    "algorithm": str(args.algorithm),
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
    if recording_rows:
      recording_summary = _recording_detection_summary(recording_rows)
      summary_payload["recording_metrics"] = recording_summary
      log.info("Recording-level evaluation: %s", json.dumps(recording_summary, sort_keys=True))
    event_summary = _confusion_summary(confusion_rows)
    summary_payload["event_metrics"] = event_summary
    log.info("Event-level evaluation against attack-region labels: %s", json.dumps(event_summary, sort_keys=True))
  else:
    summary_payload["recording_metrics"] = None
    summary_payload["event_metrics"] = None
    log.info("Evaluation skipped: no boolean malicious labels found in scored rows")
  summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  log.info("Wrote summary metrics to %s", summary_path)


if __name__ == "__main__":
  main()

