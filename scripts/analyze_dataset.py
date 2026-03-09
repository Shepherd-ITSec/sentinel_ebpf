#!/usr/bin/env python3
"""Plot loss (anomaly score) over samples from detector log. Reads scores from existing log, no model run."""

import argparse
import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None


def _open_lines(path: Path):
  """Open path for reading text lines (supports .gz)."""
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rt", encoding="utf-8")
  return path.open("r", encoding="utf-8")


def _record_hash(obj: dict) -> str:
  """Canonical hash of a record for duplicate detection."""
  return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _record_without(obj: dict, exclude: tuple[str, ...]) -> dict:
  """Copy record without given keys."""
  return {k: v for k, v in obj.items() if k not in exclude}


def count_duplicates(records: list[dict], fields: list[str], max_diffs: int) -> tuple[int | None, int | None, int | None, int | None]:
  """
  Count duplicates: exact (0), 1-field diff, 2-field diff, 3-field diff.
  Returns (exact, one_field, two_fields, three_fields); None for levels not computed.
  max_diffs: 0=skip all, 1=exact only, 2=exact+1-field, 3=exact+1+2-field, 4=all.
  """
  if max_diffs <= 0:
    return (None, None, None, None)

  exact_count = one_count = two_count = three_count = None
  rec_iter = tqdm(records, desc="Full hash", unit=" rec") if tqdm else records
  full_hashes = [_record_hash(r) for r in rec_iter]
  full_counts: dict[str, int] = {}
  for h in full_hashes:
    full_counts[h] = full_counts.get(h, 0) + 1

  if max_diffs >= 1:
    exact_count = sum(c - 1 for c in full_counts.values() if c > 1)

  if max_diffs < 2:
    return (exact_count, one_count, two_count, three_count)

  one_field_recs: set[str] = set()
  field_list = [f for f in fields if records and f in records[0]]
  for f in (tqdm(field_list, desc="1-field diff", unit=" field") if tqdm else field_list):
    groups: dict[str, set[str]] = {}
    for r, fh in zip(records, full_hashes):
      reduced = _record_without(r, (f,))
      h = _record_hash(reduced)
      groups.setdefault(h, set()).add(fh)
    for members in groups.values():
      if len(members) > 1:
        one_field_recs.update(members)
  one_count = sum(1 for h in full_hashes if h in one_field_recs)

  if max_diffs < 3:
    return (exact_count, one_count, two_count, three_count)

  two_field_recs: set[str] = set()
  pairs = [(f1, f2) for f1, f2 in itertools.combinations(fields, 2)
           if records and f1 in records[0] and f2 in records[0]]
  for f1, f2 in (tqdm(pairs, desc="2-field diff", unit=" pair") if tqdm else pairs):
    groups = {}
    for r, fh in zip(records, full_hashes):
      reduced = _record_without(r, (f1, f2))
      h = _record_hash(reduced)
      groups.setdefault(h, set()).add(fh)
    for members in groups.values():
      if len(members) > 1:
        two_field_recs.update(members)
  two_count = sum(1 for h in full_hashes if h in two_field_recs)

  if max_diffs < 4:
    return (exact_count, one_count, two_count, three_count)

  three_field_recs: set[str] = set()
  triples = [(f1, f2, f3) for f1, f2, f3 in itertools.combinations(fields, 3)
             if records and all(f in records[0] for f in (f1, f2, f3))]
  for f1, f2, f3 in (tqdm(triples, desc="3-field diff", unit=" triple") if tqdm else triples):
    groups = {}
    for r, fh in zip(records, full_hashes):
      reduced = _record_without(r, (f1, f2, f3))
      h = _record_hash(reduced)
      groups.setdefault(h, set()).add(fh)
    for members in groups.values():
      if len(members) > 1:
        three_field_recs.update(members)
  three_count = sum(1 for h in full_hashes if h in three_field_recs)

  return (exact_count, one_count, two_count, three_count)


def load_records(path: Path, max_events: int | None = None) -> tuple[list[dict], list[float]]:
  """Load full records and scores from JSONL."""
  records: list[dict] = []
  scores: list[float] = []
  with _open_lines(path) as f:
    line_iter = iter(f)
    if tqdm:
      line_iter = tqdm(line_iter, desc="Loading", unit=" lines")
    for line in line_iter:
      line = line.strip()
      if not line:
        continue
      if max_events is not None and len(records) >= max_events:
        break
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      if "score" not in obj:
        continue
      try:
        scores.append(float(obj["score"]))
        records.append(obj)
      except (TypeError, ValueError):
        continue
  return records, scores


def main():
  ap = argparse.ArgumentParser(
    description="Plot loss (anomaly score) over samples from detector log. Expects JSONL with 'score' per line.",
  )
  ap.add_argument(
    "logfile",
    nargs="?",
    default="events_05_03_26.jsonl",
    help="Path to detector event dump JSONL (default: events_05_03_26.jsonl)",
  )
  ap.add_argument(
    "--max-events",
    type=int,
    default=None,
    help="Limit number of events to load and plot",
  )
  ap.add_argument(
    "--out",
    type=Path,
    default=Path("loss_over_samples.png"),
    help="Output path for plot (default: loss_over_samples.png)",
  )
  ap.add_argument(
    "--diffs",
    type=int,
    default=4,
    choices=[0, 1, 2, 3, 4],
    metavar="N",
    help="Field-diff levels to compute: 0=none, 1=exact, 2=+1-field, 3=+2-field, 4=all (default: 4)",
  )
  args = ap.parse_args()

  path = Path(args.logfile)
  if not path.exists():
    print(f"Error: {path} not found", file=sys.stderr)
    sys.exit(1)

  print("Loading records...", file=sys.stderr)
  records, scores = load_records(path, max_events=args.max_events)
  if not scores:
    print("No scores found in log (expect 'score' field per JSONL line)", file=sys.stderr)
    sys.exit(1)

  n_events = len(records)
  n_anomalies = sum(1 for r in records if r.get("anomaly") is True)

  if args.diffs > 0:
    print("Counting duplicates (all records)...", file=sys.stderr)
    fields = list(records[0].keys()) if records else []
    exact_dup, one_field, two_fields, three_fields = count_duplicates(records, fields, args.diffs)
  else:
    exact_dup = one_field = two_fields = three_fields = None

  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not installed; run: uv sync --extra dev", file=sys.stderr)
    sys.exit(1)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(24, 10))
  ax.plot(range(len(scores)), scores, linewidth=0.3, alpha=0.9)
  ax.set_xlabel("Sample index")
  ax.set_ylabel("Loss (anomaly score)")
  ax.set_title(f"Loss over samples ({n_events} events)")
  ax.grid(True, alpha=0.3, which="major")
  ax.minorticks_on()
  ax.grid(True, alpha=0.15, which="minor", linestyle=":")

  stats_lines = [
    f"Events: {n_events:,}",
    f"Anomalies: {n_anomalies:,}",
    f"Mean score: {sum(scores)/len(scores):.4f}",
  ]
  if exact_dup is not None:
    stats_lines.append(f"Exact duplicates: {exact_dup:,}")
  if one_field is not None:
    stats_lines.append(f"1-field diff duplicates: {one_field:,}")
  if two_fields is not None:
    stats_lines.append(f"2-field diff duplicates: {two_fields:,}")
  if three_fields is not None:
    stats_lines.append(f"3-field diff duplicates: {three_fields:,}")
  stats_text = "\n".join(stats_lines)
  ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
          verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
          family="monospace")

  fig.tight_layout()
  fig.savefig(args.out, dpi=200)
  plt.close(fig)

  print(f"Plotted {n_events} scores -> {args.out}")
  dup_parts = []
  if exact_dup is not None:
    dup_parts.append(f"Exact: {exact_dup:,}")
  if one_field is not None:
    dup_parts.append(f"1-field: {one_field:,}")
  if two_fields is not None:
    dup_parts.append(f"2-field: {two_fields:,}")
  if three_fields is not None:
    dup_parts.append(f"3-field: {three_fields:,}")
  dup_str = " | " + " | ".join(dup_parts) if dup_parts else ""
  print(f"Events: {n_events:,} | Anomalies: {n_anomalies:,}{dup_str}")


if __name__ == "__main__":
  main()
