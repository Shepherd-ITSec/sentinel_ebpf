"""Composable score squash / calibration (model side)."""

from __future__ import annotations

import math

from detector.building_blocks.core.base import BlockContext, BuildingBlock, ScoreOutput


class ScalingBlock(BuildingBlock):
  """Parent must output :class:`ScoreOutput`. Emits transformed ``ScoreOutput``."""

  def __init__(self, parent: BuildingBlock, mode: str, *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._mode = (mode or "identity").strip().lower()

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    so = ctx.get_parent_output(self._parent)
    if not isinstance(so, ScoreOutput):
      raise TypeError("ScalingBlock parent must output ScoreOutput")
    raw = float(so.raw)
    if self._mode == "identity":
      scaled = float(so.scaled)
    elif self._mode == "exp1m":
      scaled = 1.0 - float(math.exp(-max(0.0, raw)))
    elif self._mode == "raw_as_scaled":
      scaled = raw
    else:
      raise ValueError(f"Unknown scaling mode: {self._mode!r}")
    ctx.outputs[id(self)] = ScoreOutput(raw=raw, scaled=scaled)

