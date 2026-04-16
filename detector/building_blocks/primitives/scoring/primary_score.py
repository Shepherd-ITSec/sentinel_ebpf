from __future__ import annotations

from detector.building_blocks.core.base import ScoreOutput
from detector.building_blocks.primitives.scoring.calibration import OnlinePercentileCalibrator


def event_group_key(event_group: str) -> str:
  """Normalize event_group the same way as detector.server.DeterministicScorer._key_for_type."""
  t = (event_group or "").strip().lower()
  return t or "__default__"


def compute_primary_score(
  score: ScoreOutput,
  *,
  score_mode: str,
  suppress_primary: bool,
  percentile_cal: OnlinePercentileCalibrator | None,
) -> float:
  """Return the score compared against ``cfg.threshold``."""
  if suppress_primary:
    return 0.0
  mode = (score_mode or "raw").strip().lower()
  if mode == "scaled":
    return float(score.scaled)
  if mode == "percentile":
    if percentile_cal is None:
      raise ValueError("percentile score_mode requires an OnlinePercentileCalibrator")
    return float(percentile_cal.percentile_prequential(float(score.raw)))
  return float(score.raw)

