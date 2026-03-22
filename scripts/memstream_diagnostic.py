#!/usr/bin/env python3
"""Run a lightweight MemStream diagnostic on a replay stream."""

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ is None or __package__ == "":
  here = Path(__file__).resolve()
  scripts_dir = here.parent
  repo_root = scripts_dir.parent
  sys.path.insert(0, str(repo_root))
  sys.path.insert(0, str(scripts_dir))

import events_pb2
from detector.config import load_config
from detector.features import extract_feature_dict, feature_view_for_algorithm
from detector.model import OnlineAnomalyDetector, OnlinePercentileCalibrator
from replay_logs import _detect_format, iter_events, iter_events_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(Path(__file__).stem)

DEFAULT_LIMIT = 500_000
DEFAULT_OUT_DIR = Path("test_data") / "memstream_diagnostic"


def _dict_to_event_envelope(obj: dict) -> events_pb2.EventEnvelope:
  data_field = obj.get("data", [])
  if not isinstance(data_field, list):
    data_field = []
  if "event_name" in obj:
    event_name = obj.get("event_name", "") or (data_field[0] if data_field else "")
    event_group = obj.get("event_group", "")
  else:
    event_name = obj.get("event_name", "") or (data_field[0] if data_field else "")
    event_group = ""
  pod_name = obj.get("pod_name", obj.get("pod", ""))
  return events_pb2.EventEnvelope(
    event_id=obj.get("event_id", ""),
    hostname=obj.get("hostname", ""),
    pod_name=pod_name,
    namespace=obj.get("namespace", ""),
    container_id=obj.get("container_id", ""),
    ts_unix_nano=int(obj.get("ts_unix_nano", 0)),
    event_name=event_name,
    event_group=event_group,
    data=data_field,
    attributes=dict(obj.get("attributes", {}) or {}),
  )


def _iter_event_dicts(path: Path, limit: int) -> Iterable[dict]:
  fmt = _detect_format(path)
  if fmt == "jsonl":
    yield from iter_events_jsonl(path, max_events=limit)
  else:
    yield from iter_events(path, max_events=limit)


def _round_float(value: float | None, digits: int = 6) -> float | None:
  if value is None or not math.isfinite(value):
    return None
  return round(float(value), digits)


def _safe_corr(a: list[float], b: list[float]) -> float | None:
  if len(a) < 2 or len(b) < 2:
    return None
  a_arr = np.asarray(a, dtype=np.float64)
  b_arr = np.asarray(b, dtype=np.float64)
  if np.allclose(a_arr, a_arr[0]) or np.allclose(b_arr, b_arr[0]):
    return None
  corr = np.corrcoef(a_arr, b_arr)[0, 1]
  return None if not np.isfinite(corr) else float(corr)


def _summarize_values(values: list[float]) -> dict[str, Any]:
  arr = np.asarray(values, dtype=np.float64)
  quantiles = {
    "p000": 0.0,
    "p001": 0.001,
    "p010": 0.01,
    "p050": 0.05,
    "p100": 0.10,
    "p250": 0.25,
    "p500": 0.50,
    "p750": 0.75,
    "p900": 0.90,
    "p950": 0.95,
    "p990": 0.99,
    "p999": 0.999,
    "p1000": 1.0,
  }
  return {
    "min": _round_float(float(arr.min())),
    "max": _round_float(float(arr.max())),
    "mean": _round_float(float(arr.mean())),
    "std": _round_float(float(arr.std())),
    "quantiles": {
      name: _round_float(float(np.quantile(arr, q)))
      for name, q in quantiles.items()
    },
  }


def _finalize_window_rates(
  windows: list[dict[str, Any]],
  threshold_counts: dict[str, int],
  chunk_index: int,
  chunk_start: int,
  chunk_events: int,
  thresholds: list[float],
) -> None:
  if chunk_events <= 0:
    return
  rates = {
    str(threshold): _round_float(threshold_counts[str(threshold)] / chunk_events)
    for threshold in thresholds
  }
  windows.append(
    {
      "chunk_index": chunk_index,
      "start_event_index": chunk_start,
      "end_event_index": chunk_start + chunk_events - 1,
      "events": chunk_events,
      "rates": rates,
    }
  )


def _select_sample_events(
  timeseries_path: Path,
  *,
  analysis_start: int,
  score_key: str,
  low_target: float,
  mid_target: float,
  high_target: float,
) -> dict[str, dict[str, Any]]:
  candidate_lists: dict[str, list[tuple[float, dict[str, Any]]]] = {
    "low": [],
    "mid": [],
    "high": [],
  }
  target_scores = {"low": low_target, "mid": mid_target, "high": high_target}
  with timeseries_path.open("r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      obj = json.loads(line)
      idx = int(obj["event_index"])
      if idx < analysis_start:
        continue
      score_value = float(obj[score_key])
      for name, target in target_scores.items():
        gap = abs(score_value - target)
        candidate_lists[name].append((gap, obj))
        candidate_lists[name].sort(key=lambda item: item[0])
        del candidate_lists[name][8:]
  selected: dict[str, dict[str, Any]] = {}
  used_indices: set[int] = set()
  for name in ("low", "mid", "high"):
    record = None
    for _, candidate in candidate_lists[name]:
      idx = int(candidate["event_index"])
      if idx not in used_indices:
        record = candidate
        used_indices.add(idx)
        break
    if record is None and candidate_lists[name]:
      record = candidate_lists[name][0][1]
      used_indices.add(int(record["event_index"]))
    if record is None:
      continue
    selected[name] = {
      f"target_{score_key}": _round_float(target_scores[name]),
      "event_index": int(record["event_index"]),
      "event_id": record["event_id"],
      "event_name": record["event_name"],
      "event_group": record["event_group"],
      "score_raw": _round_float(float(record["score_raw"])),
      "score_scaled": _round_float(float(record["score_scaled"])),
    }
  return selected


def _nearest_checkpoint_path(checkpoints_dir: Path, checkpoint_interval: int, event_index: int) -> Path | None:
  checkpoint_index = (event_index // checkpoint_interval) * checkpoint_interval
  if checkpoint_index <= 0:
    return None
  path = checkpoints_dir / f"checkpoint_{checkpoint_index:08d}.pkl"
  return path if path.exists() else None


def _run_attribution_samples(
  *,
  events_path: Path,
  sample_events: dict[str, dict[str, Any]],
  checkpoints_dir: Path,
  checkpoint_interval: int,
  out_dir: Path,
) -> None:
  samples_dir = out_dir / "feature_contribution_samples"
  samples_dir.mkdir(parents=True, exist_ok=True)
  for old_file in samples_dir.glob("*"):
    if old_file.is_file():
      old_file.unlink()
  script_path = Path(__file__).resolve().parent / "feature_attribution.py"
  for sample_name, sample in sample_events.items():
    json_out = samples_dir / f"{sample_name}.json"
    png_out = samples_dir / f"{sample_name}.png"
    cmd = [
      sys.executable,
      str(script_path),
      str(events_path),
      "--algorithm",
      "memstream",
      "--event-index",
      str(sample["event_index"]),
      "--attribution-space",
      "raw",
      "--top-k",
      "20",
      "--json",
      str(json_out),
      "--out",
      str(png_out),
    ]
    checkpoint = _nearest_checkpoint_path(checkpoints_dir, checkpoint_interval, int(sample["event_index"]))
    if checkpoint is not None:
      cmd.extend(["--checkpoint", str(checkpoint)])
    log.info("Attribution sample %s at event %d", sample_name, sample["event_index"])
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, check=True)


def _load_attribution_notes(samples_dir: Path) -> dict[str, dict[str, Any]]:
  notes: dict[str, dict[str, Any]] = {}
  for path in sorted(samples_dir.glob("*.json")):
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
    attrs = payload.get("attribution", {})
    ordered = sorted(attrs.items(), key=lambda item: abs(float(item[1])), reverse=True)
    top = ordered[:5]
    notes[path.stem] = {
      "event_index": payload.get("event_index"),
      "event_id": payload.get("event_id"),
      "score_raw": _round_float(float(payload.get("score_raw", 0.0))),
      "score_scaled": _round_float(float(payload.get("score", 0.0))),
      "top_features": [
        {"name": name, "attribution": _round_float(float(value))}
        for name, value in top
      ],
    }
  return notes


def _build_report(
  *,
  events_path: Path,
  limit: int,
  summary: dict[str, Any],
  memory: dict[str, Any],
  sample_events: dict[str, dict[str, Any]],
  attribution_notes: dict[str, dict[str, Any]],
  top_tail_events: list[dict[str, Any]],
) -> str:
  scaled = summary["scaled"]
  p50 = scaled["quantiles"]["p500"]
  p99 = scaled["quantiles"]["p990"]
  scaled_std = scaled["std"]
  accepted_rate = memory["accepted_update_rate"]
  rejected_rate = memory["rejected_update_rate"]
  overwrite_rate = memory["overwrite_rate_among_accepted"]
  score_memory_corr = memory["score_vs_memory_error_corr"]
  tail_gap = None if p50 is None or p99 is None else float(p99 - p50)

  issues: list[str] = []
  positives: list[str] = []
  if scaled_std is not None and scaled_std < 0.03:
    issues.append(f"scaled-score spread is very narrow (`std={scaled_std:.4f}`)")
  else:
    positives.append(f"scaled scores show non-trivial spread (`std={scaled_std:.4f}`)")
  if tail_gap is not None and tail_gap < 0.08:
    issues.append(f"upper tail is weak (`p99-p50={tail_gap:.4f}`)")
  else:
    positives.append(f"score tail exists (`p99-p50={tail_gap:.4f}`)")
  if accepted_rate is not None and (accepted_rate < 0.05 or accepted_rate > 0.98):
    issues.append(f"memory gating is degenerate (`accepted={accepted_rate:.3f}`, `rejected={rejected_rate:.3f}`)")
  else:
    positives.append(f"memory gating is active (`accepted={accepted_rate:.3f}`, `rejected={rejected_rate:.3f}`)")
  if score_memory_corr is None or abs(score_memory_corr) < 0.2:
    issues.append("memory distance has weak correlation with final score")
  else:
    positives.append(f"memory distance contributes meaningfully (`corr={score_memory_corr:.3f}`)")

  if len(issues) >= 3 and len(positives) <= 2:
    verdict = "Current MemStream looks weak as implemented; the diagnostic does not justify more AE-family work without changing the input representation."
  elif len(positives) >= 3:
    verdict = "Current MemStream shows meaningful internal signals; the bottleneck looks more like calibration or representation mismatch than total AE-family collapse."
  else:
    verdict = "Current MemStream is mixed: there is some internal signal, but not enough evidence yet to call the current raw-feature setup healthy."

  lines = [
    "# MemStream Diagnostic Report",
    "",
    "## Verdict",
    "",
    verdict,
    "",
    "## Run",
    "",
    f"- Events file: `{events_path}`",
    f"- Replay limit: `{limit}`",
    f"- Scored events: `{summary['events_processed']}`",
    f"- Threshold: `{summary['threshold']}`",
    "",
    "## Evidence For Concern",
    "",
  ]
  if issues:
    lines.extend([f"- {item}" for item in issues])
  else:
    lines.append("- No strong failure signal was detected in this diagnostic.")
  lines.extend([
    "",
    "## Evidence For Viability",
    "",
  ])
  if positives:
    lines.extend([f"- {item}" for item in positives])
  else:
    lines.append("- No clearly positive internal signal stood out.")
  lines.extend([
    "",
    "## Sampled Events",
    "",
    "Samples were chosen by raw score because the scaled score is saturated for most of the post-warmup stream.",
    "",
  ])
  for sample_name, sample in sample_events.items():
    lines.append(
      f"- `{sample_name}`: event `{sample['event_index']}` (`{sample['event_name']}` / `{sample['event_group'] or 'default'}`) "
      f"score=`{sample['score_scaled']}` raw=`{sample['score_raw']}`"
    )
  lines.extend([
    "",
    "## Attribution Notes",
    "",
  ])
  if attribution_notes:
    for stem, note in attribution_notes.items():
      top_names = ", ".join(item["name"] for item in note["top_features"][:3]) or "n/a"
      lines.append(f"- `{stem}`: top features = {top_names}")
  else:
    lines.append("- Attribution artifacts were not produced.")
  lines.extend([
    "",
    "## Highest-Scoring Tail",
    "",
  ])
  if top_tail_events:
    for item in top_tail_events[:10]:
      lines.append(
        f"- event `{item['event_index']}` `{item['event_name']}` / `{item['event_group'] or 'default'}` "
        f"scaled=`{item['score_scaled']}` raw=`{item['score_raw']}` update=`{item['update_allowed']}`"
      )
  else:
    lines.append("- No tail events were collected.")
  lines.extend([
    "",
    "## Interpretation",
    "",
    "This diagnostic is unlabeled, so it cannot prove benchmark usefulness on its own.",
    "It does answer whether the current implementation produces a score tail and whether memory gating is active.",
    "",
  ])
  return "\n".join(lines) + "\n"


def main() -> None:
  ap = argparse.ArgumentParser(description="Run a MemStream diagnostic replay and write compact artifacts.")
  ap.add_argument("events", nargs="?", default="events_17_03_26.jsonl", help="Replay file (JSONL or EVT1)")
  ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max events to replay")
  ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Artifact directory")
  ap.add_argument("--checkpoint-interval", type=int, default=100000, help="Save replay checkpoints every N events")
  ap.add_argument("--window-size", type=int, default=10000, help="Window size for anomaly-rate-over-time summaries")
  args = ap.parse_args()

  events_path = Path(args.events)
  if not events_path.exists():
    raise SystemExit(f"Events file not found: {events_path}")

  cfg = load_config()
  out_dir = args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  checkpoints_dir = out_dir / "checkpoints"
  checkpoints_dir.mkdir(parents=True, exist_ok=True)
  timeseries_path = out_dir / "score_timeseries.jsonl"
  if timeseries_path.exists():
    timeseries_path.unlink()

  detector = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=cfg.mem_memory_size,
    mem_lr=cfg.mem_lr,
    mem_beta=cfg.mem_beta,
    mem_k=cfg.mem_k,
    mem_gamma=cfg.mem_gamma,
    mem_input_mode=cfg.mem_input_mode,
    mem_warmup_accept=cfg.mem_warmup_accept,
    freq1d_bins=cfg.freq1d_bins,
    freq1d_alpha=cfg.freq1d_alpha,
    freq1d_decay=cfg.freq1d_decay,
    freq1d_max_categories=cfg.freq1d_max_categories,
    model_device=cfg.model_device,
    seed=cfg.model_seed,
  )

  score_mode = getattr(cfg, "score_mode", "raw").strip().lower()
  percentile_calibrators: dict[str, OnlinePercentileCalibrator] = {}

  def _get_percentile(event_group: str) -> OnlinePercentileCalibrator:
    key = (event_group or "").strip().lower() or "__default__"
    if key not in percentile_calibrators:
      percentile_calibrators[key] = OnlinePercentileCalibrator(
        window_size=getattr(cfg, "percentile_window_size", 2048),
        warmup=getattr(cfg, "percentile_warmup", 128),
      )
    return percentile_calibrators[key]

  scaled_scores: list[float] = []
  raw_scores: list[float] = []
  recon_errors: list[float] = []
  memory_errors: list[float] = []
  thresholds = sorted({0.5, 0.7, 0.9, 0.99, float(cfg.threshold)})
  overall_threshold_counts = {str(th): 0 for th in thresholds}
  window_threshold_counts = {str(th): 0 for th in thresholds}
  threshold_windows: list[dict[str, Any]] = []
  chunk_start = 0
  chunk_events = 0
  chunk_index = 0
  first_memory_full_at: int | None = None
  top_tail_events: list[dict[str, Any]] = []
  overall_event_groups: Counter[str] = Counter()
  overall_event_names: Counter[str] = Counter()

  with timeseries_path.open("w", encoding="utf-8") as timeseries_file:
    for i, obj in enumerate(_iter_event_dicts(events_path, args.limit)):
      evt = _dict_to_event_envelope(obj)
      features = extract_feature_dict(
        evt,
        feature_view=feature_view_for_algorithm(cfg.model_algorithm),
      )
      raw, scaled = detector.score_and_learn(features)
      debug = detector.get_last_debug()
      scaled_scores.append(float(scaled))
      raw_scores.append(float(raw))
      recon_errors.append(float(debug.get("recon_error", 0.0)))
      memory_errors.append(float(debug.get("memory_error", 0.0)))
      event_group_key = (evt.event_group or "").strip().lower() or "__default__"
      overall_event_groups[event_group_key] += 1
      overall_event_names[(evt.event_name or "").strip().lower() or "__unknown__"] += 1

      if score_mode == "percentile":
        cal = _get_percentile(evt.event_group or "")
        score_primary = float(cal.percentile_prequential(float(raw)))
      elif score_mode == "scaled":
        score_primary = float(scaled)
      else:
        score_primary = float(raw)

      for threshold in thresholds:
        key = str(threshold)
        if score_primary >= threshold:
          overall_threshold_counts[key] += 1
          window_threshold_counts[key] += 1

      record = {
        "event_index": i,
        "event_id": evt.event_id,
        "event_name": evt.event_name,
        "event_group": evt.event_group,
        "score_raw": round(float(raw), 6),
        "score_scaled": round(float(scaled), 6),
        "score_primary": round(score_primary, 6),
        "anomaly": bool(score_primary >= cfg.threshold),
        "recon_error": _round_float(float(debug.get("recon_error", 0.0))),
        "memory_error": _round_float(float(debug.get("memory_error", 0.0))),
        "update_allowed": debug.get("update_allowed"),
        "update_reason": debug.get("update_reason"),
        "mem_filled_after": debug.get("mem_filled_after"),
        "memory_overwrite": debug.get("memory_overwrite"),
      }
      timeseries_file.write(json.dumps(record) + "\n")

      top_tail_events.append(record)
      sort_key = "score_primary" if score_mode in ("percentile", "scaled") else "score_raw"
      top_tail_events.sort(key=lambda item: float(item.get(sort_key, item["score_scaled"])), reverse=True)
      del top_tail_events[20:]

      chunk_events += 1
      if chunk_events >= args.window_size:
        _finalize_window_rates(
          threshold_windows,
          window_threshold_counts,
          chunk_index,
          chunk_start,
          chunk_events,
          thresholds,
        )
        chunk_index += 1
        chunk_start = i + 1
        chunk_events = 0
        window_threshold_counts = {str(th): 0 for th in thresholds}

      impl = detector.impl
      if (
        getattr(impl, "algorithm", "") == "memstream"
        and first_memory_full_at is None
        and getattr(impl, "_mem_filled", 0) >= getattr(impl, "memory_size", 0)
      ):
        first_memory_full_at = i

      processed = i + 1
      if args.checkpoint_interval > 0 and processed % args.checkpoint_interval == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_{processed:08d}.pkl"
        detector.save_checkpoint(checkpoint_path, processed)
        log.info("Saved checkpoint %s", checkpoint_path.name)

      if processed % 10000 == 0:
        log.info("Processed %d events", processed)

  total_events = len(scaled_scores)
  if total_events == 0:
    raise SystemExit("No events were processed")

  _finalize_window_rates(
    threshold_windows,
    window_threshold_counts,
    chunk_index,
    chunk_start,
    chunk_events,
    thresholds,
  )

  impl = detector.impl
  if getattr(impl, "algorithm", "") != "memstream":
    raise RuntimeError("MemStream diagnostic expected a MemStream implementation")

  state = impl.get_state()
  analysis_start = min(total_events - 1, max(1000, int(impl.memory_size) * 2))
  post_warmup_raw_scores = raw_scores[analysis_start:] if analysis_start < total_events else raw_scores
  low_target = float(np.quantile(post_warmup_raw_scores, 0.10))
  mid_target = float(np.quantile(post_warmup_raw_scores, 0.50))
  high_target = float(np.quantile(post_warmup_raw_scores, 0.995))

  score_summary = {
    "events_path": str(events_path),
    "events_processed": total_events,
    "threshold": float(cfg.threshold),
    "score_mode": score_mode,
    "config": {
      "mem_memory_size": cfg.mem_memory_size,
      "mem_lr": cfg.mem_lr,
      "mem_beta": cfg.mem_beta,
      "mem_k": cfg.mem_k,
      "mem_gamma": cfg.mem_gamma,
      "model_device": cfg.model_device,
      "model_seed": cfg.model_seed,
    },
    "detector_config": asdict(cfg),
    "raw": _summarize_values(raw_scores),
    "scaled": _summarize_values(scaled_scores),
    "anomaly_rates": {
      str(threshold): _round_float(overall_threshold_counts[str(threshold)] / total_events)
      for threshold in thresholds
    },
    "anomaly_rates_by_window": {
      "window_size": args.window_size,
      "thresholds": thresholds,
      "windows": threshold_windows,
    },
  }
  (out_dir / "score_summary.json").write_text(json.dumps(score_summary, indent=2) + "\n", encoding="utf-8")

  memory_diagnostics = {
    "events_processed": total_events,
    "memory_size": int(impl.memory_size),
    "warmup_accept": int(impl._warmup_accept),
    "final_mem_filled": int(impl._mem_filled),
    "final_mem_index": int(impl._mem_index),
    "first_memory_full_at_event": first_memory_full_at,
    "accepted_updates": int(state.get("accepted_updates", 0)),
    "rejected_updates": int(state.get("rejected_updates", 0)),
    "overwrite_updates": int(state.get("overwrite_updates", 0)),
    "accepted_update_rate": _round_float(int(state.get("accepted_updates", 0)) / total_events),
    "rejected_update_rate": _round_float(int(state.get("rejected_updates", 0)) / total_events),
    "overwrite_rate_among_accepted": _round_float(
      int(state.get("overwrite_updates", 0)) / max(1, int(state.get("accepted_updates", 0)))
    ),
    "beta": _round_float(float(impl.beta)),
    "mean_recon_error": _round_float(statistics.fmean(recon_errors)),
    "mean_memory_error": _round_float(statistics.fmean(memory_errors)),
    "score_vs_recon_error_corr": _round_float(_safe_corr(scaled_scores, recon_errors)),
    "score_vs_memory_error_corr": _round_float(_safe_corr(scaled_scores, memory_errors)),
    "recon_vs_memory_error_corr": _round_float(_safe_corr(recon_errors, memory_errors)),
    "last_debug": detector.get_last_debug(),
    "event_group_counts": dict(overall_event_groups.most_common(20)),
    "event_name_counts": dict(overall_event_names.most_common(20)),
  }
  (out_dir / "memory_diagnostics.json").write_text(json.dumps(memory_diagnostics, indent=2) + "\n", encoding="utf-8")

  sample_events = _select_sample_events(
    timeseries_path,
    analysis_start=analysis_start,
    score_key="score_raw",
    low_target=low_target,
    mid_target=mid_target,
    high_target=high_target,
  )
  (out_dir / "sample_events.json").write_text(json.dumps(sample_events, indent=2) + "\n", encoding="utf-8")

  _run_attribution_samples(
    events_path=events_path,
    sample_events=sample_events,
    checkpoints_dir=checkpoints_dir,
    checkpoint_interval=args.checkpoint_interval,
    out_dir=out_dir,
  )

  attribution_notes = _load_attribution_notes(out_dir / "feature_contribution_samples")
  report = _build_report(
    events_path=events_path,
    limit=args.limit,
    summary=score_summary,
    memory=memory_diagnostics,
    sample_events=sample_events,
    attribution_notes=attribution_notes,
    top_tail_events=top_tail_events,
  )
  (out_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")

  log.info("Artifacts written to %s", out_dir)


if __name__ == "__main__":
  main()
