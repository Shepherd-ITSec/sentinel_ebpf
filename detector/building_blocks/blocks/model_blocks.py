from __future__ import annotations

from typing import TYPE_CHECKING, Any

from detector.building_blocks.core.base import BlockContext, BuildingBlock, ScoreOutput
from detector.building_blocks.primitives.models.factory import new_model_impl, scaled_score_for_algorithm

if TYPE_CHECKING:
  from detector.config import DetectorConfig


class DictFeatureModelBlock(BuildingBlock):
  """Model-side block: parent outputs dict or (dict, meta)."""

  def __init__(
    self,
    parent: BuildingBlock,
    cfg: "DetectorConfig",
    *,
    algorithm: str,
    block_uid: str | None = None,
  ) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._algorithm = str(algorithm).strip().lower()
    self._impl = new_model_impl(self._algorithm, cfg)

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    payload = ctx.get_parent_output(self._parent)
    if isinstance(payload, tuple) and len(payload) >= 2:
      features, meta = payload[0], payload[1]
    elif isinstance(payload, dict):
      features, meta = payload, None
    else:
      raise TypeError("Model block parent must output dict or (dict, meta)")
    raw = float(self._impl.score_and_learn_raw(features, meta=meta))
    ctx.outputs[id(self)] = ScoreOutput(raw=raw, scaled=scaled_score_for_algorithm(self._algorithm, raw))

  def get_state(self) -> dict[str, Any]:
    return {"algorithm": self._algorithm, "impl": self._impl.get_state()}

  def set_state(self, state: dict[str, Any]) -> None:
    impl = state.get("impl")
    if isinstance(impl, dict):
      self._impl.set_state(impl)


class HalfSpaceTreesBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="halfspacetrees", block_uid=block_uid)


class LodaEmaBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="loda_ema", block_uid=block_uid)


class KitNetBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="kitnet", block_uid=block_uid)


class MemStreamBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="memstream", block_uid=block_uid)


class ZScoreBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="zscore", block_uid=block_uid)


class KnnBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="knn", block_uid=block_uid)


class Freq1DBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="freq1d", block_uid=block_uid)


class CopulaTreeBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="copulatree", block_uid=block_uid)


class LatentClusterBlock(DictFeatureModelBlock):
  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(parent, cfg, algorithm="latentcluster", block_uid=block_uid)
