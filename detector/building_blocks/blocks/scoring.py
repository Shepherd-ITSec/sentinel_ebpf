from __future__ import annotations

from typing import TYPE_CHECKING, Any

from detector.building_blocks.core.base import (
  BlockContext,
  BuildingBlock,
  DecisionOutput,
  PrimaryScoreOutput,
  ScoreOutput,
)
from detector.building_blocks.primitives.scoring.calibration import OnlinePercentileCalibrator
from detector.building_blocks.primitives.scoring.decision import anomaly_from_primary
from detector.building_blocks.primitives.scoring.primary_score import compute_primary_score, event_group_key

if TYPE_CHECKING:
  from detector.config import DetectorConfig


class PrimaryScoreBlock(BuildingBlock):
  """Apply score-mode selection, warmup suppression, and percentile calibration."""

  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._score_mode = str(getattr(cfg, "score_mode", "raw"))
    self._warmup_events = int(getattr(cfg, "warmup_events", 0))
    self._warmup_suppress = bool(getattr(cfg, "suppress_anomalies_during_warmup", False))
    self._percentile_window_size = int(getattr(cfg, "percentile_window_size", 2048))
    self._percentile_warmup = int(getattr(cfg, "percentile_warmup", 128))
    self._warmup_counts: dict[str, int] = {}
    self._percentiles: dict[str, OnlinePercentileCalibrator] = {}

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def _get_percentile(self, key: str) -> OnlinePercentileCalibrator:
    cal = self._percentiles.get(key)
    if cal is None:
      cal = OnlinePercentileCalibrator(
        window_size=self._percentile_window_size,
        warmup=self._percentile_warmup,
      )
      self._percentiles[key] = cal
    return cal

  def forward(self, ctx: BlockContext) -> None:
    parent_out = ctx.get_parent_output(self._parent)
    if not isinstance(parent_out, ScoreOutput):
      raise TypeError("PrimaryScoreBlock parent must output ScoreOutput")

    key = event_group_key(getattr(ctx.event, "event_group", "") or "")
    self._warmup_counts[key] = self._warmup_counts.get(key, 0) + 1
    suppress_primary = self._warmup_suppress and self._warmup_counts[key] <= self._warmup_events

    percentile_cal = None
    if (not suppress_primary) and self._score_mode.strip().lower() == "percentile":
      percentile_cal = self._get_percentile(key)
    primary = compute_primary_score(
      parent_out,
      score_mode=self._score_mode,
      suppress_primary=suppress_primary,
      percentile_cal=percentile_cal,
    )
    ctx.outputs[id(self)] = PrimaryScoreOutput(
      raw=float(parent_out.raw),
      scaled=float(parent_out.scaled),
      primary=float(primary),
      score_mode=self._score_mode.strip().lower(),
      suppressed=bool(suppress_primary),
    )

  def get_state(self) -> dict[str, Any]:
    return {
      "warmup_counts": dict(self._warmup_counts),
      "percentiles": {k: cal.get_state() for k, cal in self._percentiles.items()},
    }

  def set_state(self, state: dict[str, Any]) -> None:
    self._warmup_counts = {
      str(k): int(v)
      for k, v in (state.get("warmup_counts", {}) or {}).items()
    }
    self._percentiles = {}
    for key, cal_state in (state.get("percentiles", {}) or {}).items():
      if isinstance(cal_state, dict):
        cal = OnlinePercentileCalibrator(
          window_size=self._percentile_window_size,
          warmup=self._percentile_warmup,
        )
        cal.set_state(cal_state)
        self._percentiles[str(key)] = cal


class ThresholdDecisionBlock(BuildingBlock):
  """Apply anomaly threshold to a primary score."""

  def __init__(self, parent: BuildingBlock, threshold: float, *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._threshold = float(threshold)

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    parent_out = ctx.get_parent_output(self._parent)
    if not isinstance(parent_out, PrimaryScoreOutput):
      raise TypeError("ThresholdDecisionBlock parent must output PrimaryScoreOutput")
    anomaly = anomaly_from_primary(
      parent_out.primary,
      self._threshold,
      suppress_primary=parent_out.suppressed,
    )
    ctx.outputs[id(self)] = DecisionOutput(
      raw=float(parent_out.raw),
      scaled=float(parent_out.scaled),
      primary=float(parent_out.primary),
      score_mode=str(parent_out.score_mode),
      suppressed=bool(parent_out.suppressed),
      threshold=float(self._threshold),
      anomaly=bool(anomaly),
    )

