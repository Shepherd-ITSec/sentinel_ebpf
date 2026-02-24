#!/usr/bin/env python3
"""Evaluate detector anomaly outputs against BETH labels."""

import argparse
import json
from pathlib import Path
from typing import Dict, Set


def _load_labels(path: Path) -> Dict[str, int]:
  labels: Dict[str, int] = {}
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      eid = str(row.get("event_id", ""))
      sus = int(row.get("sus", 0))
      evil = int(row.get("evil", 0))
      labels[eid] = 1 if (sus > 0 or evil > 0) else 0
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


def evaluate(labels_path: Path, anomalies_path: Path) -> Dict[str, float]:
  labels = _load_labels(labels_path)
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
  }


def main() -> None:
  ap = argparse.ArgumentParser(description="Evaluate replay anomalies against BETH labels")
  ap.add_argument("--labels", required=True, help="Path to labels ndjson from convert_beth_to_evt1.py")
  ap.add_argument("--anomalies", required=True, help="Path to detector anomaly log jsonl")
  args = ap.parse_args()

  result = evaluate(Path(args.labels), Path(args.anomalies))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()

