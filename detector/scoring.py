"""Primary score and anomaly flag used for thresholding (matches detector server logic)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from detector.model import OnlinePercentileCalibrator

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def event_group_key(event_group: str) -> str:
  """Normalize event_group the same way as detector.server.DeterministicScorer._key_for_type."""
  t = (event_group or "").strip().lower()
  return t or "__default__"


def compute_primary_score(
  score_raw: float,
  score_scaled: float,
  *,
  score_mode: str,
  suppress_primary: bool,
  percentile_cal: OnlinePercentileCalibrator | None,
) -> float:
  """Return the score compared against ``cfg.threshold`` (same as server ``score_event``)."""
  if suppress_primary:
    return 0.0
  mode = (score_mode or "raw").strip().lower()
  if mode == "scaled":
    return float(score_scaled)
  if mode == "percentile":
    if percentile_cal is None:
      raise ValueError("percentile score_mode requires an OnlinePercentileCalibrator")
    return float(percentile_cal.percentile_prequential(float(score_raw)))
  return float(score_raw)


def anomaly_from_primary(
  score_primary: float,
  threshold: float,
  *,
  suppress_primary: bool,
) -> bool:
  return (not suppress_primary) and float(score_primary) >= float(threshold)


def get_or_create_percentile_calibrator(
  registry: dict[str, OnlinePercentileCalibrator],
  key: str,
  cfg: DetectorConfig,
) -> OnlinePercentileCalibrator:
  cal = registry.get(key)
  if cal is None:
    cal = OnlinePercentileCalibrator(
      window_size=int(getattr(cfg, "percentile_window_size", 2048)),
      warmup=int(getattr(cfg, "percentile_warmup", 128)),
    )
    registry[key] = cal
  return cal
