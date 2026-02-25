#!/usr/bin/env python3
"""Evaluate detector anomaly outputs against BETH labels."""

import argparse
import json
from pathlib import Path
from typing import Dict, Literal, Set

LabelMode = Literal["evil_only", "sus_or_evil"]


def _load_labels(path: Path, label_mode: LabelMode = "sus_or_evil") -> Dict[str, int]:
  """Load labels ndjson; positive = 1 by label_mode: evil_only (evil>0) or sus_or_evil (sus>0 or evil>0)."""
  labels: Dict[str, int] = {}
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      eid = str(row.get("event_id", ""))
      sus = int(row.get("sus", 0))
      evil = int(row.get("evil", 0))
      if label_mode == "evil_only":
        positive = 1 if evil > 0 else 0
      else:
        positive = 1 if (sus > 0 or evil > 0) else 0
      labels[eid] = positive
  return labels


def _load_flagged(path: Path) -> Set[str]:
  flagged: Set[str] = set()
  if not path.exists():
    return flagged
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      eid = str(row.get("event_id", ""))
      if eid:
        flagged.add(eid)
  return flagged


def evaluate(labels_path: Path, anomalies_path: Path, label_mode: LabelMode = "sus_or_evil") -> Dict[str, float]:
  labels = _load_labels(labels_path, label_mode=label_mode)
  flagged = _load_flagged(anomalies_path)
  ids = set(labels.keys())

  tp = fp = tn = fn = 0
  for eid in ids:
    y = labels[eid]
    p = 1 if eid in flagged else 0
    if p == 1 and y == 1:
      tp += 1
    elif p == 1 and y == 0:
      fp += 1
    elif p == 0 and y == 0:
      tn += 1
    else:
      fn += 1

  precision = tp / (tp + fp) if (tp + fp) else 0.0
  recall = tp / (tp + fn) if (tp + fn) else 0.0
  f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
  accuracy = (tp + tn) / len(ids) if ids else 0.0
  flagged_rate = len(flagged & ids) / len(ids) if ids else 0.0

  return {
    "samples": len(ids),
    "flagged_samples": len(flagged & ids),
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": accuracy,
    "flagged_rate": flagged_rate,
    "label_mode": label_mode,
  }


def main() -> None:
  ap = argparse.ArgumentParser(description="Evaluate replay anomalies against BETH labels")
  ap.add_argument("--labels", required=True, help="Path to labels ndjson from convert_beth_to_evt1.py")
  ap.add_argument("--anomalies", required=True, help="Path to detector anomaly log jsonl")
  ap.add_argument(
    "--label-mode",
    choices=["evil_only", "sus_or_evil"],
    default="sus_or_evil",
    help="Positive = evil>0 only (evil_only) or sus>0 or evil>0 (sus_or_evil, default)",
  )
  args = ap.parse_args()

  result = evaluate(Path(args.labels), Path(args.anomalies), label_mode=args.label_mode)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

