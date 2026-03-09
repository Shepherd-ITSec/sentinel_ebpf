#!/usr/bin/env python3
"""Compute GradCAM-like feature attribution for anomaly detector events.

Replays events from the start of the file (as the original detector did), then computes
perturbation-based attribution for a selected event. Model state matches the original
stream: events 0..N-1 are learned before attributing event N.
"""

import argparse
import gzip
import json
import logging
import struct
import sys
from pathlib import Path

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

if __package__ is None or __package__ == "":
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import events_pb2
from detector.features import extract_feature_dict
from detector.model import OnlineAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(Path(__file__).stem)

MAGIC = b"EVT1"


def _open_text_lines(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rt", encoding="utf-8")
  return path.open("r", encoding="utf-8")


def _detect_format(path: Path) -> str:
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    with gzip.open(path, "rb") as zf:
      first = zf.read(4)
    return "evt1" if first == MAGIC else "jsonl"
  with path.open("rb") as f:
    first = f.read(4)
  return "evt1" if first == MAGIC else "jsonl"


def _dict_to_event_envelope(obj: dict) -> events_pb2.EventEnvelope:
  data_field = obj.get("data", [])
  if not isinstance(data_field, list):
    data_field = []
  if "event_name" in obj:
    event_name = obj.get("event_name", "") or (data_field[0] if data_field else "")
    event_type = obj.get("event_type", "")
  else:
    event_name = obj.get("event_type", "") or (data_field[0] if data_field else "")
    event_type = ""
  pod_name = obj.get("pod_name", obj.get("pod", ""))
  return events_pb2.EventEnvelope(
    event_id=obj.get("event_id", ""),
    hostname=obj.get("hostname", ""),
    pod_name=pod_name,
    namespace=obj.get("namespace", ""),
    container_id=obj.get("container_id", ""),
    ts_unix_nano=int(obj.get("ts_unix_nano", 0)),
    event_name=event_name,
    event_type=event_type,
    data=data_field,
    attributes=dict(obj.get("attributes", {}) or {}),
  )


def _open_stream(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rb")
  return path.open("rb")


def iter_events(path: Path, max_events=None):
  n = 0
  with _open_stream(path) as f:
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
      n += 1
      yield obj
      if max_events is not None and n >= max_events:
        return


def iter_events_jsonl(path: Path, max_events=None):
  n = 0
  with _open_text_lines(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "event_id" not in obj:
        continue
      n += 1
      yield obj
      if max_events is not None and n >= max_events:
        return


def _count_events(path: Path) -> int:
  """Count events in file (streaming, no storage)."""
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path) if fmt == "jsonl" else iter_events(path)
  return sum(1 for _ in event_iter)


def _replay_and_get_target_features(
  path: Path,
  event_index: int,
  detector: OnlineAnomalyDetector,
):
  """Replay events 0..event_index-1 through score_and_learn, return features for event_index."""
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path, max_events=event_index + 1) if fmt == "jsonl" else iter_events(path, max_events=event_index + 1)
  it = event_iter
  if tqdm:
    it = tqdm(it, desc="Replay", unit=" evt", total=event_index + 1, file=sys.stderr)
  target_features = None
  for i, obj in enumerate(it):
    evt = _dict_to_event_envelope(obj)
    features = extract_feature_dict(evt)
    if i < event_index:
      detector.score_and_learn(features)
    else:
      target_features = features
  return target_features


def main():
  ap = argparse.ArgumentParser(
    description="Compute GradCAM-like feature attribution for anomaly detector events.",
  )
  ap.add_argument(
    "logfile",
    help="Path to detector-events JSONL or EVT1",
  )
  ap.add_argument(
    "--algorithm",
    default="kitnet",
    choices=["kitnet", "memstream", "loda", "halfspacetrees"],
    help="Detector algorithm (default: kitnet). KitNet needs ~50k events to exit grace.",
  )
  ap.add_argument(
    "--event-index",
    type=int,
    default=None,
    help="Event index to attribute (0-based). Default: last event in file.",
  )
  ap.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="Perturbation epsilon for finite difference (default: 0.01)",
  )
  ap.add_argument(
    "--top-k",
    type=int,
    default=20,
    help="Show top K features by |attribution| (default: 20)",
  )
  ap.add_argument(
    "--out",
    type=Path,
    default=Path("attribution.png"),
    help="Output path for bar chart (default: attribution.png)",
  )
  ap.add_argument(
    "--json",
    type=Path,
    default=None,
    help="Optional: write full attribution dict to JSON file",
  )
  args = ap.parse_args()

  path = Path(args.logfile)
  if not path.exists():
    log.error("File not found: %s", path)
    sys.exit(1)

  event_index = args.event_index
  if event_index is None:
    log.info("Counting events in %s...", path)
    total = _count_events(path)
    if total == 0:
      log.error("No events in file")
      sys.exit(1)
    event_index = total - 1
    log.info("Using last event: index %d (of %d)", event_index, total)

  if event_index < 0:
    log.error("Event index must be >= 0")
    sys.exit(1)

  detector = OnlineAnomalyDetector(algorithm=args.algorithm)
  log.info("Replaying events 0..%d (learn), then attribute event %d", event_index - 1, event_index)
  features = _replay_and_get_target_features(path, event_index, detector)
  if features is None:
    log.error("Event index %d beyond file bounds", event_index)
    sys.exit(1)

  log.info("Computing attribution for event index %d", event_index)
  score, attribution = detector.compute_feature_attribution(features, epsilon=args.epsilon)

  max_attr = max(abs(v) for v in attribution.values()) if attribution else 0.0
  if score == 0.0 and args.algorithm == "kitnet":
    log.warning(
      "KitNet returned score 0 (likely in grace period). Needs ~50k events. Try --algorithm loda or higher event_index."
    )
  if max_attr < 1e-9:
    log.warning(
      "All attributions near zero. KitNet may be in grace; try --algorithm loda. Or score is flat (no gradient)."
    )

  sorted_attrs = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
  top_k = sorted_attrs[: args.top_k]

  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w", encoding="utf-8") as f:
      json.dump({"score": score, "event_index": event_index, "attribution": attribution}, f, indent=2)
    log.info("Wrote %s", args.json)

  try:
    import matplotlib.pyplot as plt
  except ImportError:
    log.error("matplotlib not installed; run: uv sync --extra dev")
    sys.exit(1)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  names = [n for n, _ in top_k]
  vals = [v for _, v in top_k]
  colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
  fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
  y_pos = range(len(names))
  ax.barh(y_pos, vals, color=colors)
  ax.set_yticks(y_pos)
  ax.set_yticklabels(names, fontsize=9)
  ax.set_xlabel("Attribution (gradient × input)")
  ax.set_title(f"Feature attribution (score={score:.4f}, event_index={event_index}, algorithm={args.algorithm})")
  ax.axvline(0, color="black", linewidth=0.5)
  if max_attr < 1e-9 and names:
    ax.text(
      0.5, 0.5, "All attributions zero.\nKitNet needs ~50k events; try --algorithm loda.",
      transform=ax.transAxes, fontsize=12, ha="center", va="center",
      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
  fig.tight_layout()
  fig.savefig(args.out, dpi=150)
  plt.close(fig)
  log.info("Wrote %s", args.out)
  top = top_k[0] if top_k else ("N/A", 0.0)
  print(f"Score: {score:.4f} | Top contributor: {top[0]}={top[1]:.4f}")


if __name__ == "__main__":
  main()
