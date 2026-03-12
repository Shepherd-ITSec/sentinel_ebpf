#!/usr/bin/env python3
"""Compute GradCAM-like feature attribution for anomaly detector events.

Replays events from the start of the file (as the original detector did), then computes
perturbation-based attribution for a selected event. Model state matches the original
stream: events 0..N-1 are learned before attributing event N.

How the attribution picture is created
--------------------------------------
1. Load events from JSONL or EVT1, build EventEnvelope, extract features via
   extract_feature_dict().

2. Replay events 0..event_index-1 through score_and_learn() to match the original
   detector state. For event_index, call compute_feature_attribution() (no learn).

3. Attribution per feature:
   - Binary features (value in [0, 0.01] or [0.99, 1]): use flip instead of ±ε.
     attribution = (score(feature=1) - score(feature=0)) × value.
     Avoids invalid interpolation and threshold effects from tiny perturbations.
   - Continuous features: finite difference with ±ε.
     attribution = (s_plus - s_minus) / (2ε) × value.

4. Sort features by |attribution| descending, take top K (default 20).

5. Horizontal bar chart: feature names on Y-axis, attribution on X-axis.
   - Red bars: positive (feature pushes score up, more anomalous).
   - Blue bars: negative (feature pushes score down).
   - Black line at 0. Numeric labels at bar ends.
   - Title: score, event_index, algorithm.
"""

import argparse
import gzip
import json
import logging
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List

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
_JSON_DEFAULT = object()  # sentinel: --json with no path


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


def _find_event_index_by_id(path: Path, event_id: str) -> int | None:
  """Return 0-based index of first event whose event_id equals event_id, or None."""
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path) if fmt == "jsonl" else iter_events(path)
  want = str(event_id)
  for i, obj in enumerate(event_iter):
    if str(obj.get("event_id", "")) == want:
      return i
  return None


def _replay_and_get_target_features(
  path: Path,
  event_index: int,
  detector: OnlineAnomalyDetector,
  *,
  start_from: int = 0,
  checkpoint_at: int | None = None,
  checkpoint_path: Path | None = None,
):
  """Replay events start_from..event_index-1 through score_and_learn, return (features, event_id) for event_index.
  If start_from > 0, detector is assumed pre-loaded from checkpoint (events 0..start_from-1 already learned)."""
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path, max_events=event_index + 1) if fmt == "jsonl" else iter_events(path, max_events=event_index + 1)
  target_features = None
  event_id = ""
  start_i = 0

  # Prime detector before tqdm when starting from scratch (avoids lazy-init log mid-progress bar)
  if start_from == 0 and event_index > 0:
    first = next(event_iter, None)
    if first is not None:
      evt = _dict_to_event_envelope(first)
      features = extract_feature_dict(evt)
      detector.score_and_learn(features)
      if checkpoint_at == 1 and checkpoint_path is not None:
        detector.save_checkpoint(checkpoint_path, 1)
        log.info("Saved checkpoint at index 1")
      start_i = 1

  it = event_iter
  if tqdm:
    it = tqdm(it, desc="Replay", unit=" evt", total=event_index + 1, initial=start_i, file=sys.stderr)
  for j, obj in enumerate(it):
    i = start_i + j
    evt = _dict_to_event_envelope(obj)
    features = extract_feature_dict(evt)
    if i < event_index:
      if i >= start_from:
        detector.score_and_learn(features)
      if checkpoint_at is not None and checkpoint_path is not None and i == checkpoint_at - 1:
        detector.save_checkpoint(checkpoint_path, checkpoint_at)
        log.info("Saved checkpoint at index %d", checkpoint_at)
    else:
      target_features = features
      event_id = str(obj.get("event_id", ""))
  return target_features, event_id


def _compute_sanity_report(
  detector: OnlineAnomalyDetector,
  features: Dict[str, float],
  sorted_attrs: list,
  epsilons: List[float],
  sanity_top_k: int,
  *,
  attribution_space: str,
  binary_threshold: float = 0.01,
) -> Dict[str, object]:
  report: Dict[str, object] = {
    "binary_threshold": binary_threshold,
    "epsilons": epsilons,
    "score_space": attribution_space,
    "score_scaled_transform": "score = 1 - exp(-max(0, raw))",
    "score_lograw_transform": "score_lograw = log(1 + max(0, raw))",
    "features": [],
  }
  for name, attr in sorted_attrs[:sanity_top_k]:
    val = float(features[name])
    is_binary = val <= binary_threshold or val >= (1.0 - binary_threshold)
    item: Dict[str, object] = {
      "name": name,
      "value": val,
      "attribution": float(attr),
      "is_binary": is_binary,
      "eps_diagnostics": {},
    }
    if is_binary:
      f0 = dict(features)
      f1 = dict(features)
      f0[name] = 0.0
      f1[name] = 1.0
      s0_raw = detector.score_only_raw(f0)
      s1_raw = detector.score_only_raw(f1)
      s0_lograw = math.log1p(max(0.0, float(s0_raw)))
      s1_lograw = math.log1p(max(0.0, float(s1_raw)))
      s0 = detector.score_only(f0)
      s1 = detector.score_only(f1)
      item["flip"] = {
        "s0_raw": s0_raw,
        "s1_raw": s1_raw,
        "delta_raw": s1_raw - s0_raw,
        "s0_lograw": s0_lograw,
        "s1_lograw": s1_lograw,
        "delta_lograw": s1_lograw - s0_lograw,
        "s0": s0,
        "s1": s1,
        "delta": s1 - s0,
      }
    for eps in epsilons:
      f_plus = dict(features)
      f_minus = dict(features)
      f_plus[name] = max(0.0, min(1.0, val + eps))
      f_minus[name] = max(0.0, min(1.0, val - eps))
      s_plus_raw = detector.score_only_raw(f_plus)
      s_minus_raw = detector.score_only_raw(f_minus)
      s_plus_lograw = math.log1p(max(0.0, float(s_plus_raw)))
      s_minus_lograw = math.log1p(max(0.0, float(s_minus_raw)))
      s_plus = detector.score_only(f_plus)
      s_minus = detector.score_only(f_minus)
      item["eps_diagnostics"][str(eps)] = {
        "s_plus_raw": s_plus_raw,
        "s_minus_raw": s_minus_raw,
        "delta_raw": s_plus_raw - s_minus_raw,
        "s_plus_lograw": s_plus_lograw,
        "s_minus_lograw": s_minus_lograw,
        "delta_lograw": s_plus_lograw - s_minus_lograw,
        "s_plus": s_plus,
        "s_minus": s_minus,
        "delta": s_plus - s_minus,
      }
    report["features"].append(item)
  return report


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
    choices=["kitnet", "memstream", "loda", "halfspacetrees", "zscore", "knn", "freq1d"],
    help="Detector algorithm (default: kitnet). KitNet needs ~50k events to exit grace.",
  )
  g = ap.add_mutually_exclusive_group()
  g.add_argument(
    "--event-index",
    type=int,
    default=None,
    help="Event index to attribute (0-based). Default: last event in file.",
  )
  g.add_argument(
    "--event-id",
    type=str,
    default=None,
    metavar="ID",
    help="Event ID to attribute (first matching event_id in file).",
  )
  ap.add_argument(
    "--epsilon",
    type=float,
    default=0.01,
    help="Perturbation epsilon for finite difference (default: 0.01)",
  )
  ap.add_argument(
    "--attribution-space",
    choices=["raw", "lograw", "scaled"],
    default="raw",
    help="Score space for attribution: raw, lograw=log(1+raw) (percent-like), or scaled (default: raw).",
  )
  ap.add_argument(
    "--offset",
    type=int,
    default=0,
    metavar="K",
    help="Relative offset from the selected event before attributing. "
    "Negative = earlier event, positive = later (default: 0).",
  )
  ap.add_argument(
    "--num-events",
    type=int,
    default=1,
    metavar="N",
    help="Attribute N consecutive events starting from the selected one (default: 1).",
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
    default=None,
    help="Output path for bar chart (default: attribution_<event_id>.png)",
  )
  ap.add_argument(
    "--json",
    nargs="?",
    type=lambda x: Path(x) if x else _JSON_DEFAULT,
    default=None,
    const=_JSON_DEFAULT,
    metavar="PATH",
    help="Write attribution + sanity to JSON. With no PATH, uses attribution_<event_id>.json",
  )
  ap.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    metavar="PATH",
    help="Checkpoint file: load if exists (skip replay up to checkpoint), save when reaching --checkpoint-at.",
  )
  ap.add_argument(
    "--checkpoint-at",
    type=int,
    default=None,
    metavar="N",
    help="Save checkpoint after learning events 0..N-1. Requires --checkpoint.",
  )
  args = ap.parse_args()

  path = Path(args.logfile)
  if not path.exists():
    log.error("File not found: %s", path)
    sys.exit(1)

  if args.num_events <= 0:
    log.error("--num-events must be >= 1 (got %d)", args.num_events)
    sys.exit(1)

  event_index = args.event_index
  total_events: int | None = None
  if args.event_id is not None:
    log.info("Resolving event_id %r...", args.event_id)
    event_index = _find_event_index_by_id(path, args.event_id)
    if event_index is None:
      log.error("No event with event_id=%r in file", args.event_id)
      sys.exit(1)
    log.info("Found event_id at index %d", event_index)
  elif event_index is None:
    log.info("Counting events in %s...", path)
    total_events = _count_events(path)
    if total_events == 0:
      log.error("No events in file")
      sys.exit(1)
    event_index = total_events - 1
    log.info("Using last event: index %d (of %d)", event_index, total_events)

  # Apply relative offset (distance) from the selected event_id/index.
  if args.offset != 0:
    if total_events is None:
      log.info("Counting events in %s for --offset...", path)
      total_events = _count_events(path)
      if total_events == 0:
        log.error("No events in file")
        sys.exit(1)
    target_index = event_index + args.offset
    if target_index < 0 or target_index >= total_events:
      log.error(
        "Offset moved target out of bounds: base_index=%d offset=%d target_index=%d total_events=%d",
        event_index,
        args.offset,
        target_index,
        total_events,
      )
      sys.exit(1)
    log.info("Applying offset %d: attributing from event_index=%d instead of %d", args.offset, target_index, event_index)
    event_index = target_index

  if event_index < 0:
    log.error("Event index must be >= 0")
    sys.exit(1)

  if args.checkpoint_at is not None and args.checkpoint is None:
    log.error("--checkpoint-at requires --checkpoint")
    sys.exit(1)

  detector = OnlineAnomalyDetector(algorithm=args.algorithm)
  start_from = 0
  checkpoint_at = args.checkpoint_at
  checkpoint_path = args.checkpoint

  if checkpoint_path is not None and checkpoint_path.exists():
    try:
      start_from = detector.load_checkpoint(checkpoint_path)
      log.info("Resuming from checkpoint: replay events %d..%d", start_from, event_index - 1)
    except ValueError as e:
      log.error("%s", e)
      sys.exit(1)
  elif start_from == 0:
    log.info("Replaying events 0..%d (learn), then attribute event %d", event_index - 1, event_index)

  # Single-event mode: keep original behavior (replay up to event_index once).
  if args.num_events == 1:
    features, event_id = _replay_and_get_target_features(
      path,
      event_index,
      detector,
      start_from=start_from,
      checkpoint_at=checkpoint_at,
      checkpoint_path=checkpoint_path,
    )
    if features is None:
      log.error("Event index %d beyond file bounds", event_index)
      sys.exit(1)

    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in (event_id or "unknown"))
    base_name = f"attribution_{event_index:08d}_{safe_id}_{args.algorithm}_{args.attribution_space}"

    if args.out is not None:
      out_path = args.out
      if out_path.suffix == "":
        out_path = out_path / f"{base_name}.png"
    else:
      out_path = Path("test_data") / "attribution" / f"{base_name}.png"

    json_path = None
    if args.json is not None:
      if args.json is _JSON_DEFAULT:
        json_path = Path("test_data") / "attribution" / f"{base_name}.json"
      else:
        json_path = args.json

    log.info("Computing attribution for event index %d", event_index)
    score_scaled = detector.score_only(features)
    score_raw = detector.score_only_raw(features)
    score_attr, attribution = detector.compute_feature_attribution(
      features,
      epsilon=args.epsilon,
      score_mode=args.attribution_space,
    )

    max_attr = max(abs(v) for v in attribution.values()) if attribution else 0.0
    if score_scaled == 0.0 and args.algorithm == "kitnet":
      log.warning(
        "KitNet returned score 0 (likely in grace period). Needs ~50k events. Try --algorithm loda or higher event_index."
      )
    if max_attr < 1e-9:
      log.warning(
        "All attributions near zero. This can happen if the scaled score saturates near 1.0 (exp squash), "
        "or if the model is in grace / score is flat."
      )
    if args.attribution_space == "raw" and score_raw > 1e9:
      log.warning("Raw score is extremely large; try --attribution-space lograw for more interpretable magnitudes.")

    sorted_attrs = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
    top_k = sorted_attrs[: args.top_k]

    if json_path is not None:
      json_path.parent.mkdir(parents=True, exist_ok=True)
      sanity_report = _compute_sanity_report(
        detector=detector,
        features=features,
        sorted_attrs=sorted_attrs,
        epsilons=[0.001, 0.005, 0.01],
        sanity_top_k=10,
        attribution_space=args.attribution_space,
      )
      payload = {
        "score": score_scaled,
        "score_raw": score_raw,
        "score_attribution_space": args.attribution_space,
        "score_attribution": score_attr,
        "attribution_score_space": args.attribution_space,
        "event_index": event_index,
        "event_number": event_index + 1,
        "event_id": event_id,
        "attribution": attribution,
        "sanity": sanity_report,
      }
      with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
      log.info("Wrote %s", json_path)

    try:
      import matplotlib.pyplot as plt
    except ImportError:
      log.error("matplotlib not installed; run: uv sync --extra dev")
      sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in top_k]
    vals = [v for _, v in top_k]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=2, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"Attribution (d {args.attribution_space} score × input)")
    if args.attribution_space == "raw":
      score_attr_str = f"{score_attr:.3g}"
    else:
      score_attr_str = f"{score_attr:.4f}"
    title = (
      f"Feature attribution (score={score_scaled:.4f}, {args.attribution_space}={score_attr_str}, "
      f"event_index={event_index}, algorithm={args.algorithm})"
    )
    if event_id:
      title += f"\nevent_id={event_id}"
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    if max_attr < 1e-9 and names:
      ax.text(
        0.5,
        0.5,
        "All attributions zero.\nKitNet needs ~50k events; try --algorithm loda.",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
      )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    top = top_k[0] if top_k else ("N/A", 0.0)
    print(
      f"Score: {score_scaled:.4f} | raw={score_raw:.3g} | {args.attribution_space}={score_attr_str} | "
      f"event_index: {event_index} | event_id: {event_id} | Top contributor: {top[0]}={top[1]:.4f}"
    )
    return

  # Multi-event mode: single replay pass, attribute several targets, and learn between them.
  if total_events is None:
    log.info("Counting events in %s for --num-events...", path)
    total_events = _count_events(path)
    if total_events == 0:
      log.error("No events in file")
      sys.exit(1)

  max_index = min(event_index + args.num_events - 1, total_events - 1)
  if max_index < event_index:
    log.error("Invalid target range [%d, %d]", event_index, max_index)
    sys.exit(1)

  target_indices = set(range(event_index, max_index + 1))
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path, max_events=max_index + 1) if fmt == "jsonl" else iter_events(
    path, max_events=max_index + 1
  )
  it = event_iter
  if tqdm:
    it = tqdm(it, desc="Replay (multi-attr)", unit=" evt", total=max_index + 1, file=sys.stderr)

  for i, obj in enumerate(it):
    if i < start_from:
      continue

    evt = _dict_to_event_envelope(obj)
    features = extract_feature_dict(evt)
    event_id = str(obj.get("event_id", ""))

    is_target = i in target_indices
    did_learn = False

    if is_target:
      safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in (event_id or "unknown"))
      base_name = f"attribution_{i:08d}_{safe_id}_{args.algorithm}_{args.attribution_space}"

      if args.out is not None:
        out_path = args.out
        if out_path.suffix == "":
          out_path = out_path / f"{base_name}.png"
      else:
        out_path = Path("test_data") / "attribution" / f"{base_name}.png"

      json_path = None
      if args.json is not None:
        if args.json is _JSON_DEFAULT:
          json_path = Path("test_data") / "attribution" / f"{base_name}.json"
        else:
          if args.json.suffix != ".json":
            json_path = args.json / f"{base_name}.json"
          else:
            json_path = args.json

      log.info("Computing attribution for event index %d", i)
      score_scaled = detector.score_only(features)
      score_raw = detector.score_only_raw(features)
      score_attr, attribution = detector.compute_feature_attribution(
        features,
        epsilon=args.epsilon,
        score_mode=args.attribution_space,
      )

      max_attr = max(abs(v) for v in attribution.values()) if attribution else 0.0
      if score_scaled == 0.0 and args.algorithm == "kitnet":
        log.warning(
          "KitNet returned score 0 (likely in grace period). Needs ~50k events. Try --algorithm loda or higher event_index."
        )
      if max_attr < 1e-9:
        log.warning(
          "All attributions near zero. This can happen if the scaled score saturates near 1.0 (exp squash), "
          "or if the model is in grace / score is flat."
        )
      if args.attribution_space == "raw" and score_raw > 1e9:
        log.warning("Raw score is extremely large; try --attribution-space lograw for more interpretable magnitudes.")

      sorted_attrs = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
      top_k = sorted_attrs[: args.top_k]

      if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_report = _compute_sanity_report(
          detector=detector,
          features=features,
          sorted_attrs=sorted_attrs,
          epsilons=[0.001, 0.005, 0.01],
          sanity_top_k=10,
          attribution_space=args.attribution_space,
        )
        payload = {
          "score": score_scaled,
          "score_raw": score_raw,
          "score_attribution_space": args.attribution_space,
          "score_attribution": score_attr,
          "attribution_score_space": args.attribution_space,
          "event_index": i,
          "event_number": i + 1,
          "event_id": event_id,
          "attribution": attribution,
          "sanity": sanity_report,
        }
        with json_path.open("w", encoding="utf-8") as f:
          json.dump(payload, f, indent=2)
        log.info("Wrote %s", json_path)

      try:
        import matplotlib.pyplot as plt
      except ImportError:
        log.error("matplotlib not installed; run: uv sync --extra dev")
        sys.exit(1)

      out_path.parent.mkdir(parents=True, exist_ok=True)
      names = [n for n, _ in top_k]
      vals = [v for _, v in top_k]
      colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
      fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
      y_pos = range(len(names))
      bars = ax.barh(y_pos, vals, color=colors)
      ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=2, fontsize=8)
      ax.set_yticks(y_pos)
      ax.set_yticklabels(names, fontsize=9)
      ax.set_xlabel(f"Attribution (d {args.attribution_space} score × input)")
      if args.attribution_space == "raw":
        score_attr_str = f"{score_attr:.3g}"
      else:
        score_attr_str = f"{score_attr:.4f}"
      title = (
        f"Feature attribution (score={score_scaled:.4f}, {args.attribution_space}={score_attr_str}, "
        f"event_index={i}, algorithm={args.algorithm})"
      )
      if event_id:
        title += f"\nevent_id={event_id}"
      ax.set_title(title)
      ax.axvline(0, color="black", linewidth=0.5)
      if max_attr < 1e-9 and names:
        ax.text(
          0.5,
          0.5,
          "All attributions zero.\nKitNet needs ~50k events; try --algorithm loda.",
          transform=ax.transAxes,
          fontsize=12,
          ha="center",
          va="center",
          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
      fig.tight_layout()
      fig.savefig(out_path, dpi=150)
      plt.close(fig)
      log.info("Wrote %s", out_path)
      top = top_k[0] if top_k else ("N/A", 0.0)
      print(
        f"Score: {score_scaled:.4f} | raw={score_raw:.3g} | {args.attribution_space}={score_attr_str} | "
        f"event_index: {i} | event_id: {event_id} | Top contributor: {top[0]}={top[1]:.4f}"
      )

      # After attributing, learn from this event so later targets see the same state as the live detector.
      detector.score_and_learn(features)
      did_learn = True
    else:
      detector.score_and_learn(features)
      did_learn = True

    if (
      did_learn
      and checkpoint_at is not None
      and checkpoint_path is not None
      and i == checkpoint_at - 1
    ):
      detector.save_checkpoint(checkpoint_path, checkpoint_at)
      log.info("Saved checkpoint at index %d", checkpoint_at)


if __name__ == "__main__":
  main()
