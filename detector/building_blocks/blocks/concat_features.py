"""LID-DS-style Concat for **feature dicts** (not for merging LAS/GAS scores — use fusion blocks)."""

from __future__ import annotations

from detector.building_blocks.core.base import BlockContext, BuildingBlock


class ConcatFeatures(BuildingBlock):
  """Merge parent outputs that are ``dict`` or ``(dict, meta)`` into one flat ``dict`` (last parent wins on key clash)."""

  def __init__(self, parents: list[BuildingBlock], *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    if len(parents) < 1:
      raise ValueError("ConcatFeatures needs at least one parent")
    self._parents = list(parents)

  def depends_on(self) -> list[BuildingBlock]:
    return list(self._parents)

  def forward(self, ctx: BlockContext) -> None:
    merged: dict[str, float] = {}
    meta = None
    for p in self._parents:
      val = ctx.get_parent_output(p)
      if isinstance(val, tuple) and val and isinstance(val[0], dict):
        merged.update(val[0])
        if len(val) > 1:
          meta = val[1]
      elif isinstance(val, dict):
        merged.update(val)
      else:
        raise TypeError("ConcatFeatures parents must emit dict or (dict, meta)")
    ctx.outputs[id(self)] = (merged, meta)

