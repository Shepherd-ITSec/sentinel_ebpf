#!/usr/bin/env python3
"""
Build a subset EVT1 dataset plus labels for detector evaluation.

Samples rows from labelled CSV files (eventName, evil, sus, etc.), without
replacement, and writes:
- <out_prefix>.evt1
- <out_prefix>.labels.ndjson

Dataset shape controls:
- many negatives + few positives via --positive-fraction
- warmup prefix with no positives via --warmup-fraction
- default filter to network-relevant syscall rows only

If requested constraints cannot be satisfied with available rows, the script
exits with a clear error.
"""

import argparse
import csv
import json
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

MAGIC = b"EVT1"


def _build_evt(row: Dict[str, str], out_idx: int, start_ts_unix_nano: int, event_id_prefix: str) -> Tuple[dict, dict]:
  """Convert a labelled CSV row to (evt dict, label dict) for EVT1/labels NDJSON."""
  event_name = (row.get("eventName") or row.get("syscall_name") or "unknown").strip()
  event_id = f"{event_id_prefix}-{out_idx}"
  ts = start_ts_unix_nano + out_idx * 1_000_000

  try:
    syscall_nr = int(row.get("syscall_nr") or 0)
  except (TypeError, ValueError):
    syscall_nr = 0
  if syscall_nr < 0:
    syscall_nr = 0

  arg0 = str(row.get("arg0", "0"))
  arg1 = str(row.get("arg1", "0"))

  attrs: Dict[str, str] = {}
  if row.get("sin_port") or row.get("sin_addr") or row.get("sa_family"):
    attrs["sin_port"] = str(row.get("sin_port", ""))
    attrs["sin_addr"] = str(row.get("sin_addr", ""))
    attrs["sa_family"] = str(row.get("sa_family", ""))
  elif row.get("sockaddr"):
    arg1 = str(row["sockaddr"])

  evt: dict = {
    "event_id": event_id,
    "syscall_name": event_name,
    "event_group": "network",
    "ts_unix_nano": ts,
    "syscall_nr": syscall_nr,
    "comm": (row.get("comm") or "").strip(),
    "pid": str(row.get("pid", "0")),
    "tid": str(row.get("tid", "0")),
    "uid": str(row.get("uid", "0")),
    "arg0": arg0,
    "arg1": arg1,
    "path": (row.get("path") or "").strip(),
  }
  if attrs:
    evt["attributes"] = attrs

  label: dict = {
    "event_id": event_id,
    "sus": int(row.get("sus", 0) or 0),
    "evil": int(row.get("evil", 0) or 0),
  }
  return evt, label
NETWORK_EVENT_NAMES = {
  "accept",
  "accept4",
  "bind",
  "connect",
  "getpeername",
  "getsockname",
  "getsockopt",
  "listen",
  "recvfrom",
  "recvmsg",
  "recvmmsg",
  "sendmmsg",
  "sendmsg",
  "sendto",
  "setsockopt",
  "shutdown",
  "socket",
  "socketpair",
}


@dataclass(frozen=True)
class SourceRow:
  source_file: str
  source_row_idx: int
  row: Dict[str, str]


def _parse_int(raw: str, default: int = 0) -> int:
  try:
    return int(raw)
  except (TypeError, ValueError):
    return default


def _load_csv_rows(csv_paths: List[Path]) -> List[SourceRow]:
  all_rows: List[SourceRow] = []
  for csv_path in csv_paths:
    if not csv_path.exists():
      raise ValueError(f"CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as src:
      reader = csv.DictReader(src)
      row_iter = enumerate(reader)
      if tqdm:
        row_iter = tqdm(row_iter, desc=f"Load {csv_path.name}", unit=" row", file=sys.stderr)
      for row_idx, row in row_iter:
        all_rows.append(SourceRow(source_file=str(csv_path), source_row_idx=row_idx, row=row))
  if not all_rows:
    raise ValueError("No rows loaded from CSV input files.")
  return all_rows


def _is_network_row(row: Dict[str, str]) -> bool:
  event_name = (row.get("eventName") or "").strip().lower()
  return event_name in NETWORK_EVENT_NAMES


def _select_rows(
  rows: List[SourceRow],
  total_events: int,
  positive_fraction: float,
  warmup_fraction: float,
  rng: random.Random,
) -> List[SourceRow]:
  if total_events <= 0:
    raise ValueError("total_events must be > 0")
  if not (0.0 <= positive_fraction <= 1.0):
    raise ValueError("positive_fraction must be in [0, 1]")
  if not (0.0 <= warmup_fraction < 1.0):
    raise ValueError("warmup_fraction must be in [0, 1)")

  positives = [r for r in rows if _parse_int(r.row.get("evil", "0")) > 0]
  negatives = [r for r in rows if _parse_int(r.row.get("evil", "0")) <= 0]

  n_pos = int(round(total_events * positive_fraction))
  n_neg = total_events - n_pos
  warmup_len = int(total_events * warmup_fraction)
  warmup_len = max(0, min(warmup_len, total_events))
  tail_len = total_events - warmup_len

  if n_pos > len(positives):
    raise ValueError(
      f"Not doable: requested positives={n_pos} but only {len(positives)} evil>0 rows available in source CSVs."
    )
  if n_neg > len(negatives):
    raise ValueError(
      f"Not doable: requested negatives={n_neg} but only {len(negatives)} non-evil rows available in source CSVs."
    )
  if total_events > len(rows):
    raise ValueError(
      f"Not doable: requested total_events={total_events} but only {len(rows)} source rows available (no duplicates allowed)."
    )
  if warmup_len > n_neg:
    raise ValueError(
      "Not doable: warmup requires all-negative prefix, "
      f"but warmup_len={warmup_len} > available selected negatives={n_neg}. "
      "Reduce warmup_fraction or positive_fraction, or increase total events/source pool."
    )
  if n_pos > tail_len:
    raise ValueError(
      "Not doable: positives must be placed after warmup, "
      f"but positives={n_pos} > post_warmup_slots={tail_len}. "
      "Reduce positive_fraction or warmup_fraction."
    )

  selected_pos = rng.sample(positives, n_pos)
  selected_neg = rng.sample(negatives, n_neg)

  warmup_neg = selected_neg[:warmup_len]
  tail = selected_neg[warmup_len:] + selected_pos
  rng.shuffle(tail)
  selected = warmup_neg + tail

  if len(selected) != total_events:
    raise RuntimeError("Internal selection error: selected count mismatch.")
  # Safety check: no duplicated source rows in output.
  source_keys = {(r.source_file, r.source_row_idx) for r in selected}
  if len(source_keys) != len(selected):
    raise RuntimeError("Internal selection error: duplicate source rows selected.")
  return selected


def _build_output_from_selected(
  selected_rows: List[SourceRow],
  start_ts_unix_nano: int,
  event_id_prefix: str,
) -> Tuple[List[dict], List[dict]]:
  events: List[dict] = []
  labels: List[dict] = []
  for out_idx, src in enumerate(selected_rows):
    evt, label = _build_evt(src.row, out_idx, start_ts_unix_nano, event_id_prefix=event_id_prefix)
    evt["event_group"] = "network"
    # Keep provenance to make debugging easier while preserving original label semantics.
    evt.setdefault("attributes", {})
    evt["attributes"]["source_file"] = src.source_file
    evt["attributes"]["source_row_idx"] = str(src.source_row_idx)
    events.append(evt)
    labels.append(label)
  return events, labels


def _write_evt1(path: Path, events: List[dict]) -> None:
  """Write events to an EVT1 binary file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f:
    evt_iter = events
    if tqdm:
      evt_iter = tqdm(events, desc="Write EVT1", unit=" evt", file=sys.stderr)
    for evt in evt_iter:
      payload = json.dumps(evt, separators=(",", ":")).encode("utf-8")
      f.write(MAGIC)
      f.write(struct.pack("<I", len(payload)))
      f.write(payload)


def _write_labels_ndjson(path: Path, labels: List[dict]) -> None:
  """Write labels to NDJSON."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    label_iter = labels
    if tqdm:
      label_iter = tqdm(labels, desc="Write labels", unit=" row", file=sys.stderr)
    for row in label_iter:
      f.write(json.dumps(row) + "\n")


def main() -> None:
  ap = argparse.ArgumentParser(description="Build an EVT1 subset dataset from labelled CSVs (no fabricated events/labels).")
  ap.add_argument(
    "--source-csv",
    nargs="+",
    default=[
      "test_data/synthetic/labelled_training_data.csv",
      "test_data/synthetic/labelled_testing_data.csv",
    ],
    help="One or more labelled CSV inputs (eventName, evil, sus, etc.) to sample from.",
  )
  ap.add_argument(
    "--out-prefix",
    required=True,
    help="Output prefix (e.g., test_data/synthetic/run1). Writes <out-prefix>.evt1 and <out-prefix>.labels.ndjson.",
  )
  ap.add_argument(
    "--total-events",
    type=int,
    default=100_000,
    help="Total number of sampled events in output dataset.",
  )
  ap.add_argument(
    "--positive-fraction",
    type=float,
    default=0.01,
    help="Target fraction of evil>0 events in output dataset.",
  )
  ap.add_argument(
    "--warmup-fraction",
    type=float,
    default=0.75,
    help="Fraction of output stream prefix forced to contain only non-evil events.",
  )
  ap.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
  )
  ap.add_argument(
    "--start-ts-unix-ms",
    type=int,
    default=None,
    help="Optional epoch-ms base for EVT1 conversion timing. If omitted, uses current time.",
  )
  ap.add_argument(
    "--event-id-prefix",
    default="synth",
    help="Prefix for generated event_ids in output EVT1/labels.",
  )
  ap.add_argument(
    "--network-only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep only network syscall rows (default: true). Use --no-network-only to disable.",
  )

  args = ap.parse_args()
  print(f"Starting synthetic EVT1 dataset generation...")
  rng = random.Random(args.seed)
  if args.start_ts_unix_ms is None:
    start_ts_ms = int(time.time() * 1000)
  else:
    start_ts_ms = args.start_ts_unix_ms
  start_ts_unix_nano = start_ts_ms * 1_000_000

  source_csvs = [Path(p) for p in args.source_csv]
  source_rows = _load_csv_rows(source_csvs)
  total_source_rows = len(source_rows)
  if args.network_only:
    source_rows = [r for r in source_rows if _is_network_row(r.row)]
    if not source_rows:
      raise ValueError("No network rows found in source CSVs. Try --no-network-only to inspect all events.")
    print(
      f"Filtered source rows to network events: {len(source_rows)} / {total_source_rows} "
      f"({len(source_rows) / total_source_rows:.4f})"
    )

  selected_rows = _select_rows(
    rows=source_rows,
    total_events=args.total_events,
    positive_fraction=args.positive_fraction,
    warmup_fraction=args.warmup_fraction,
    rng=rng,
  )
  events, labels = _build_output_from_selected(
    selected_rows=selected_rows,
    start_ts_unix_nano=start_ts_unix_nano,
    event_id_prefix=args.event_id_prefix,
  )

  out_prefix = Path(args.out_prefix)
  evt1_path = out_prefix.with_suffix(".evt1")
  labels_path = out_prefix.with_suffix(".labels.ndjson")

  _write_evt1(evt1_path, events)
  _write_labels_ndjson(labels_path, labels)

  n_pos = sum(1 for r in labels if int(r.get("evil", 0)) > 0)
  n_neg = len(labels) - n_pos
  warmup_len = int(args.total_events * args.warmup_fraction)
  warmup_positives = sum(1 for r in labels[:warmup_len] if int(r.get("evil", 0)) > 0)

  print(f"Wrote EVT1 log: {evt1_path}")
  print(f"Wrote labels:  {labels_path}")
  print(f"Source CSVs:   {', '.join(str(p) for p in source_csvs)}")
  print(f"Total events:  {len(events)}")
  print(f"Positives:     {n_pos} ({n_pos / len(events):.6f})")
  print(f"Negatives:     {n_neg} ({n_neg / len(events):.6f})")
  print(f"Warmup len:    {warmup_len}")
  print(f"Warmup pos:    {warmup_positives}")


if __name__ == "__main__":
  main()

