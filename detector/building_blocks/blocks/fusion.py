"""Multi-parent score fusion (model side): stack parent :class:`ScoreOutput` — not feature Concat."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from detector.building_blocks.core.base import BlockContext, BuildingBlock, ScoreOutput


class FusionMeanBlock(BuildingBlock):
  """Mean of parent ``ScoreOutput`` (raw and scaled separately)."""

  def __init__(self, parents: list[BuildingBlock], *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    if len(parents) < 1:
      raise ValueError("FusionMeanBlock needs at least one parent")
    self._parents = list(parents)

  def depends_on(self) -> list[BuildingBlock]:
    return list(self._parents)

  def forward(self, ctx: BlockContext) -> None:
    raws: list[float] = []
    scaled: list[float] = []
    for p in self._parents:
      so = ctx.get_parent_output(p)
      if not isinstance(so, ScoreOutput):
        raise TypeError("FusionMeanBlock parents must output ScoreOutput")
      raws.append(float(so.raw))
      scaled.append(float(so.scaled))
    n = len(raws)
    ctx.outputs[id(self)] = ScoreOutput(raw=sum(raws) / n, scaled=sum(scaled) / n)


class FusionLinearBlock(BuildingBlock):
  """Stack parent **scaled** scores into a vector, apply ``Linear(n,1)`` + sigmoid (inference)."""

  def __init__(self, parents: list[BuildingBlock], *, seed: int = 42, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    if len(parents) < 1:
      raise ValueError("FusionLinearBlock needs at least one parent")
    self._parents = list(parents)
    torch.manual_seed(int(seed))
    n = len(self._parents)
    self._lin = nn.Linear(n, 1, bias=True)
    nn.init.constant_(self._lin.weight, 1.0 / n)
    nn.init.constant_(self._lin.bias, 0.0)

  def depends_on(self) -> list[BuildingBlock]:
    return list(self._parents)

  def forward(self, ctx: BlockContext) -> None:
    xs: list[float] = []
    raws: list[float] = []
    for p in self._parents:
      so = ctx.get_parent_output(p)
      if not isinstance(so, ScoreOutput):
        raise TypeError("FusionLinearBlock parents must output ScoreOutput")
      xs.append(float(so.scaled))
      raws.append(float(so.raw))
    x = torch.tensor(xs, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
      y = self._lin(x).squeeze()
      scaled = float(torch.sigmoid(y))
    raw = float(sum(raws) / max(1, len(raws)))
    ctx.outputs[id(self)] = ScoreOutput(raw=raw, scaled=scaled)

  def get_state(self) -> dict[str, Any]:
    return {"lin": self._lin.state_dict()}

  def set_state(self, state: dict[str, Any]) -> None:
    sd = state.get("lin")
    if isinstance(sd, dict):
      self._lin.load_state_dict(sd)

