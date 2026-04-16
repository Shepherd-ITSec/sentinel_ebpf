#!/usr/bin/env python3
"""Run a compact unlabeled diagnostic for memstream or loda_ema on a replay stream."""

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
try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None

if __package__ is None or __package__ == "":
  here = Path(__file__).resolve()
  scripts_dir = here.parent
  repo_root = scripts_dir.parent
  sys.path.insert(0, str(repo_root))
  sys.path.insert(0, str(scripts_dir))

import events_pb2
from detector.config import load_config
from detector.building_blocks.primitives.models.factory import new_model_impl, scaled_score_for_algorithm
from detector.building_blocks.primitives.features import extract_feature_dict
from detector.pipelines.feature_sets import feature_names_for_algorithm
from replay_logs import _detect_format, iter_events, iter_events_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger(Path(__file__).stem)

DEFAULT_LIMIT = 500_000
DEFAULT_OUT_DIR = Path("test_data") / "model_diagnostic"


def _dict_to_event_envelope(obj: dict) -> events_pb2.EventEnvelope:
  from event_envelope_codec import envelope_from_dict

  return envelope_from_dict(obj)


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
      "syscall_name": record["syscall_name"],
      "event_group": record["event_group"],
      "score_raw": _round_float(float(record["score_raw"])),
      "score_scaled": _round_float(float(record["score_scaled"])),
    }
  return selected


def _select_top_tail_events(
  timeseries_path: Path,
  *,
  analysis_start: int,
  score_key: str,
  limit: int = 20,
) -> list[dict[str, Any]]:
  top_tail: list[dict[str, Any]] = []
  with timeseries_path.open("r", encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      obj = json.loads(line)
      if int(obj["event_index"]) < analysis_start:
        continue
      top_tail.append(obj)
      top_tail.sort(key=lambda item: float(item[score_key]), reverse=True)
      del top_tail[limit:]
  return top_tail


def _nearest_checkpoint_path(checkpoints_dir: Path, checkpoint_interval: int, event_index: int) -> Path | None:
  if checkpoint_interval <= 0:
    return None
  checkpoint_index = (event_index // checkpoint_interval) * checkpoint_interval
  if checkpoint_index <= 0:
    return None
  path = checkpoints_dir / f"checkpoint_{checkpoint_index:08d}.pkl"
  return path if path.exists() else None


class _DetectorAdapter:
  def __init__(self, algorithm: str, impl: Any) -> None:
    self.algorithm = algorithm
    self.impl = impl

  def score_and_learn(self, features: dict[str, float], *, meta=None) -> tuple[float, float]:
    raw = float(self.impl.score_and_learn_raw(features, meta=meta))
    return raw, scaled_score_for_algorithm(self.algorithm, raw)

  def score_only(self, features: dict[str, float], *, meta=None) -> tuple[float, float]:
    raw = float(self.impl.score_only_raw(features, meta=meta))
    return raw, scaled_score_for_algorithm(self.algorithm, raw)

  def get_last_debug(self) -> dict[str, Any]:
    getter = getattr(self.impl, "get_last_debug", None)
    if getter is None:
      return {}
    val = getter()
    return val if isinstance(val, dict) else {}


def _build_detector(cfg: Any, algorithm: str) -> _DetectorAdapter:
  return _DetectorAdapter(algorithm, new_model_impl(algorithm, cfg))


def _run_attribution_samples(
  *,
  algorithm: str,
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
      algorithm,
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


def _build_model_diagnostics(
  *,
  algorithm: str,
  impl: Any,
  state: dict[str, Any],
  total_events: int,
  analysis_start: int,
  score_raw: list[float],
  score_scaled: list[float],
  model_signal: list[float],
  first_model_ready_at: int | None,
  last_debug: dict[str, Any],
  overall_event_groups: Counter[str],
  overall_syscall_names: Counter[str],
  update_allowed: list[bool] | None = None,
  memory_overwrite: list[bool] | None = None,
) -> dict[str, Any]:
  analyzed_raw = score_raw[analysis_start:]
  analyzed_scaled = score_scaled[analysis_start:]
  analyzed_signal = model_signal[analysis_start:]
  analyzed_events = len(analyzed_raw)

  if algorithm == "memstream":
    accepted_total = int(state.get("accepted_updates", 0))
    rejected_total = int(state.get("rejected_updates", 0))
    overwrite_total = int(state.get("overwrite_updates", 0))
    analyzed_updates = update_allowed[analysis_start:] if update_allowed is not None else []
    analyzed_overwrites = memory_overwrite[analysis_start:] if memory_overwrite is not None else []
    accepted_analyzed = int(sum(1 for item in analyzed_updates if item))
    rejected_analyzed = int(sum(1 for item in analyzed_updates if not item))
    return {
      "algorithm": algorithm,
      "events_processed": total_events,
      "analysis_start_event": analysis_start,
      "events_analyzed": analyzed_events,
      "memory_size": int(impl.memory_size),
      "warmup_accept": int(impl._warmup_accept),
      "final_mem_filled": int(impl._mem_filled),
      "final_mem_index": int(impl._mem_index),
      "first_model_ready_at_event": first_model_ready_at,
      "accepted_updates": accepted_total,
      "rejected_updates": rejected_total,
      "overwrite_updates": overwrite_total,
      "accepted_update_rate": _round_float(accepted_analyzed / max(1, analyzed_events)),
      "rejected_update_rate": _round_float(rejected_analyzed / max(1, analyzed_events)),
      "overwrite_rate_among_accepted": _round_float(
        int(sum(1 for item in analyzed_overwrites if item)) / max(1, accepted_analyzed)
      ),
      "beta": _round_float(float(impl.beta)),
      "k": int(impl.k),
      "gamma": _round_float(float(impl.gamma)),
      "mean_model_signal": _round_float(statistics.fmean(analyzed_signal)),
      "score_vs_model_signal_corr": _round_float(_safe_corr(analyzed_raw, analyzed_signal)),
      "last_debug": last_debug,
      "event_group_counts": dict(overall_event_groups.most_common(20)),
      "syscall_name_counts": dict(overall_syscall_names.most_common(20)),
    }

  counts = np.asarray(state.get("counts"))
  var = np.asarray(state.get("var"))
  totals = counts.sum(axis=1) if counts.size else np.asarray([], dtype=np.float64)
  nonzero_ratio = float(np.count_nonzero(counts) / counts.size) if counts.size else 0.0
  result: dict[str, Any] = {
    "algorithm": algorithm,
    "events_processed": total_events,
    "analysis_start_event": analysis_start,
    "events_analyzed": analyzed_events,
    "effective_n_projections": int(state.get("effective_n_projections", 0)),
    "feature_dim": len(state.get("feature_names", [])),
    "bins": int(impl.bins),
    "hist_decay": _round_float(float(impl.hist_decay)),
    "ema_alpha": _round_float(float(impl.ema_alpha)),
    "first_model_ready_at_event": first_model_ready_at,
    "histogram_nonzero_ratio": _round_float(nonzero_ratio),
    "mean_total_count_per_projection": _round_float(float(totals.mean())) if totals.size else None,
    "std_total_count_per_projection": _round_float(float(totals.std())) if totals.size else None,
    "min_projection_variance": _round_float(float(var.min())) if var.size else None,
    "mean_projection_variance": _round_float(float(var.mean())) if var.size else None,
    "max_projection_variance": _round_float(float(var.max())) if var.size else None,
    "mean_model_signal": _round_float(statistics.fmean(analyzed_signal)),
    "score_vs_model_signal_corr": _round_float(_safe_corr(analyzed_raw, analyzed_signal)),
    "event_group_counts": dict(overall_event_groups.most_common(20)),
    "syscall_name_counts": dict(overall_syscall_names.most_common(20)),
  }
  if algorithm == "loda_ema":
    result["last_debug"] = last_debug
  return result


def _build_report(
  *,
  algorithm: str,
  events_path: Path,
  limit: int,
  summary: dict[str, Any],
  model_diagnostics: dict[str, Any],
  sample_events: dict[str, dict[str, Any]],
  attribution_notes: dict[str, dict[str, Any]],
  top_tail_events: list[dict[str, Any]],
) -> str:
  scaled = summary["post_warmup"]["scaled"]
  p50 = scaled["quantiles"]["p500"]
  p99 = scaled["quantiles"]["p990"]
  scaled_std = scaled["std"]
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

  if algorithm == "memstream":
    accepted_rate = model_diagnostics.get("accepted_update_rate")
    rejected_rate = model_diagnostics.get("rejected_update_rate")
    signal_corr = model_diagnostics.get("score_vs_model_signal_corr")
    if accepted_rate is not None and (accepted_rate < 0.05 or accepted_rate > 0.98):
      issues.append(f"memory gating is degenerate (`accepted={accepted_rate:.3f}`, `rejected={rejected_rate:.3f}`)")
    else:
      positives.append(f"memory gating is active (`accepted={accepted_rate:.3f}`, `rejected={rejected_rate:.3f}`)")
    if signal_corr is None or abs(signal_corr) < 0.2:
      issues.append("memory distance has weak correlation with score")
    else:
      positives.append(f"memory distance contributes meaningfully (`corr={signal_corr:.3f}`)")
  else:
    nonzero_ratio = model_diagnostics.get("histogram_nonzero_ratio")
    proj_var = model_diagnostics.get("mean_projection_variance")
    if nonzero_ratio is not None and nonzero_ratio < 0.001:
      issues.append(f"LODA histograms are extremely sparse (`nonzero_ratio={nonzero_ratio:.6f}`)")
    elif nonzero_ratio is not None:
      positives.append(f"projection histograms are populated (`nonzero_ratio={nonzero_ratio:.4f}`)")
    if proj_var is not None and proj_var < 1e-4:
      issues.append(f"projection variance looks collapsed (`mean_var={proj_var:.6f}`)")
    elif proj_var is not None:
      positives.append(f"projection variance looks healthy (`mean_var={proj_var:.4f}`)")

  if len(issues) >= 3 and len(positives) <= 2:
    verdict = (
      f"Current `{algorithm}` looks weak on this stream; the present feature/view setup is not producing a strong, "
      "interpretable anomaly signal."
    )
  elif len(positives) >= 3:
    verdict = (
      f"Current `{algorithm}` shows meaningful internal signal on this stream; if benchmark results disappoint, "
      "the next suspect is calibration or feature choice rather than total model collapse."
    )
  else:
    verdict = (
      f"Current `{algorithm}` is mixed on this stream: some signal exists, but not enough yet to call the feature setup healthy."
    )

  lines = [
    f"# {algorithm} Diagnostic Report",
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
    f"- Analysis start event: `{summary['analysis_start_event']}`",
    f"- Post-warmup events analyzed: `{summary['events_analyzed']}`",
    f"- Requested features: `{len(summary['requested_features'])}`",
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
    "Samples were chosen by raw score after the initial warmup region so the report shows low/mid/high behavior on a trained model state.",
    "",
  ])
  for sample_name, sample in sample_events.items():
    lines.append(
      f"- `{sample_name}`: event `{sample['event_index']}` (`{sample['syscall_name']}` / `{sample['event_group'] or 'default'}`) "
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
        f"- event `{item['event_index']}` `{item['syscall_name']}` / `{item['event_group'] or 'default'}` "
        f"scaled=`{item['score_scaled']}` raw=`{item['score_raw']}`"
      )
  else:
    lines.append("- No tail events were collected.")
  lines.extend([
    "",
    "## Interpretation",
    "",
    "This diagnostic is unlabeled, so it cannot prove benchmark usefulness on its own.",
    "It does answer whether the current implementation produces a score tail and whether the model internals look alive rather than collapsed.",
    "",
  ])
  return "\n".join(lines) + "\n"


def main() -> None:
  ap = argparse.ArgumentParser(description="Run a compact unlabeled diagnostic for memstream or loda_ema.")
  ap.add_argument("events", nargs="?", default="artifacts/datasets/events_17_03_26.jsonl", help="Replay file (JSONL or EVT1)")
  ap.add_argument("--algorithm", choices=["memstream", "loda_ema"], default="memstream")
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

  detector = _build_detector(cfg, args.algorithm)
  requested_features = feature_names_for_algorithm(cfg, args.algorithm)
  scaled_scores: list[float] = []
  raw_scores: list[float] = []
  model_signal: list[float] = []
  thresholds = sorted({0.5, 0.7, 0.9, 0.99, float(cfg.threshold)})
  overall_threshold_counts = {str(th): 0 for th in thresholds}
  window_threshold_counts = {str(th): 0 for th in thresholds}
  threshold_windows: list[dict[str, Any]] = []
  chunk_start = 0
  chunk_events = 0
  chunk_index = 0
  first_model_ready_at: int | None = None
  top_tail_events: list[dict[str, Any]] = []
  memstream_update_allowed: list[bool] = []
  memstream_memory_overwrite: list[bool] = []
  overall_event_groups: Counter[str] = Counter()
  overall_syscall_names: Counter[str] = Counter()

  with timeseries_path.open("w", encoding="utf-8") as timeseries_file:
    event_iter = _iter_event_dicts(events_path, args.limit)
    if tqdm is not None:
      event_iter = tqdm(event_iter, total=int(args.limit), desc="Diagnostic replay", unit=" evt", file=sys.stderr, leave=True)
    else:
      log.info("tqdm not installed; running diagnostic without progress bar")
    for i, obj in enumerate(event_iter):
      evt = _dict_to_event_envelope(obj)
      features, meta = extract_feature_dict(evt, requested_features=requested_features)
      raw, scaled = detector.score_and_learn(features, meta=meta)
      debug = detector.get_last_debug()
      impl = detector.impl
      if args.algorithm == "memstream":
        signal_value = float(debug.get("memory_error", raw))
      elif args.algorithm == "loda_ema":
        signal_value = float(debug.get("max_projection_excess", raw))
      else:
        signal_value = float(raw)
      scaled_scores.append(float(scaled))
      raw_scores.append(float(raw))
      model_signal.append(signal_value)
      event_group = (evt.event_group or "").strip().lower()
      overall_event_groups[event_group or "__default__"] += 1
      overall_syscall_names[(evt.syscall_name or "").strip().lower() or "__unknown__"] += 1

      for threshold in thresholds:
        key = str(threshold)
        if scaled >= threshold:
          overall_threshold_counts[key] += 1
          window_threshold_counts[key] += 1

      record = {
        "event_index": i,
        "event_id": evt.event_id,
        "syscall_name": evt.syscall_name,
        "event_group": evt.event_group,
        "score_raw": round(float(raw), 6),
        "score_scaled": round(float(scaled), 6),
        "anomaly": bool(scaled >= cfg.threshold),
        "model_signal": _round_float(signal_value),
      }
      if args.algorithm == "memstream":
        record.update({
          "recon_error": _round_float(float(debug.get("recon_error", 0.0))),
          "memory_error": _round_float(float(debug.get("memory_error", 0.0))),
          "update_allowed": debug.get("update_allowed"),
          "update_reason": debug.get("update_reason"),
          "mem_filled_after": debug.get("mem_filled_after"),
          "memory_overwrite": debug.get("memory_overwrite"),
        })
        memstream_update_allowed.append(bool(debug.get("update_allowed", False)))
        memstream_memory_overwrite.append(bool(debug.get("memory_overwrite", False)))
      elif args.algorithm == "loda_ema":
        record.update({
          "max_projection_excess": _round_float(float(debug.get("max_projection_excess", 0.0))),
          "mean_projection_excess": _round_float(float(debug.get("mean_projection_excess", 0.0))),
        })
      timeseries_file.write(json.dumps(record) + "\n")

      top_tail_events.append(record)
      top_tail_events.sort(key=lambda item: float(item["score_scaled"]), reverse=True)
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

      if args.algorithm == "memstream":
        if first_model_ready_at is None and getattr(impl, "_mem_filled", 0) >= getattr(impl, "memory_size", 0):
          first_model_ready_at = i
      elif first_model_ready_at is None and getattr(impl, "_counts", None) is not None:
        first_model_ready_at = i

      processed = i + 1
      if args.checkpoint_interval > 0 and processed % args.checkpoint_interval == 0:
        checkpoint_path = checkpoints_dir / f"checkpoint_{processed:08d}.pkl"
        # model_diagnostic is primarily for score debugging; feature state is optional here.
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
  state = impl.get_state()
  analysis_start = min(
    total_events - 1,
    max(
      1000,
      int(getattr(impl, "memory_size", 0)) * 2 if args.algorithm == "memstream" else 1000,
      int(first_model_ready_at or 0),
    ),
  )
  post_warmup_raw_scores = raw_scores[analysis_start:] if analysis_start < total_events else raw_scores
  post_warmup_scaled_scores = scaled_scores[analysis_start:] if analysis_start < total_events else scaled_scores
  post_warmup_model_signal = model_signal[analysis_start:] if analysis_start < total_events else model_signal
  low_target = float(np.quantile(post_warmup_raw_scores, 0.10))
  mid_target = float(np.quantile(post_warmup_raw_scores, 0.50))
  high_target = float(np.quantile(post_warmup_raw_scores, 0.995))

  top_tail_events = _select_top_tail_events(
    timeseries_path,
    analysis_start=analysis_start,
    score_key="score_scaled",
    limit=20,
  )

  score_summary = {
    "algorithm": args.algorithm,
    "requested_features": list(requested_features),
    "events_path": str(events_path),
    "events_processed": total_events,
    "analysis_start_event": analysis_start,
    "events_analyzed": len(post_warmup_raw_scores),
    "threshold": float(cfg.threshold),
    "config": asdict(cfg),
    "raw": _summarize_values(raw_scores),
    "scaled": _summarize_values(scaled_scores),
    "model_signal": _summarize_values(model_signal),
    "post_warmup": {
      "raw": _summarize_values(post_warmup_raw_scores),
      "scaled": _summarize_values(post_warmup_scaled_scores),
      "model_signal": _summarize_values(post_warmup_model_signal),
    },
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

  model_diagnostics = _build_model_diagnostics(
    algorithm=args.algorithm,
    impl=impl,
    state=state,
    total_events=total_events,
    analysis_start=analysis_start,
    score_raw=raw_scores,
    score_scaled=scaled_scores,
    model_signal=model_signal,
    first_model_ready_at=first_model_ready_at,
    last_debug=detector.get_last_debug(),
    overall_event_groups=overall_event_groups,
    overall_syscall_names=overall_syscall_names,
    update_allowed=memstream_update_allowed if args.algorithm == "memstream" else None,
    memory_overwrite=memstream_memory_overwrite if args.algorithm == "memstream" else None,
  )
  (out_dir / "model_diagnostics.json").write_text(json.dumps(model_diagnostics, indent=2) + "\n", encoding="utf-8")

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
    algorithm=args.algorithm,
    events_path=events_path,
    sample_events=sample_events,
    checkpoints_dir=checkpoints_dir,
    checkpoint_interval=args.checkpoint_interval,
    out_dir=out_dir,
  )

  attribution_notes = _load_attribution_notes(out_dir / "feature_contribution_samples")
  report = _build_report(
    algorithm=args.algorithm,
    events_path=events_path,
    limit=args.limit,
    summary=score_summary,
    model_diagnostics=model_diagnostics,
    sample_events=sample_events,
    attribution_notes=attribution_notes,
    top_tail_events=top_tail_events,
  )
  (out_dir / "diagnostic_report.md").write_text(report, encoding="utf-8")
  log.info("Artifacts written to %s", out_dir)


if __name__ == "__main__":
  main()
