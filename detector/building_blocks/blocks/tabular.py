from __future__ import annotations

from typing import TYPE_CHECKING, Any

from detector.building_blocks.blocks.model_blocks import ZScoreBlock
from detector.building_blocks.core.base import BlockContext, BuildingBlock

if TYPE_CHECKING:
  from detector.config import DetectorConfig
  from detector.building_blocks.primitives.features.extractor import FeatureExtractor


class FeatureDictExtractorBlock(BuildingBlock):
  """Feature side: run ``FeatureExtractor.extract_feature_dict`` for one explicit feature list."""

  def __init__(
    self,
    extractor: FeatureExtractor,
    requested_features: tuple[str, ...] | list[str],
    *,
    block_uid: str | None = None,
  ) -> None:
    super().__init__(block_uid=block_uid)
    self._extractor = extractor
    self._requested_features = tuple(str(name) for name in requested_features)

  def depends_on(self) -> list[BuildingBlock]:
    return []

  def forward(self, ctx: BlockContext) -> None:
    feats, meta = self._extractor.extract_feature_dict(ctx.event, requested_features=self._requested_features)
    ctx.outputs[id(self)] = (feats, meta)

  def get_state(self) -> dict[str, Any]:
    return {"extractor": self._extractor.get_state()}

  def set_state(self, state: dict[str, Any]) -> None:
    ex = state.get("extractor")
    if isinstance(ex, dict):
      self._extractor.set_state(ex)


class ZScoreOnlineBlock(BuildingBlock):
  """Tabular z-score; parent must output ``(dict[str,float], Meta|None)``."""

  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._delegate = ZScoreBlock(parent, cfg, block_uid=block_uid)

  def depends_on(self) -> list[BuildingBlock]:
    return self._delegate.depends_on()

  def forward(self, ctx: BlockContext) -> None:
    self._delegate.forward(ctx)
    ctx.outputs[id(self)] = ctx.outputs[id(self._delegate)]

  def get_state(self) -> dict[str, Any]:
    return self._delegate.get_state()

  def set_state(self, state: dict[str, Any]) -> None:
    self._delegate.set_state(state)

