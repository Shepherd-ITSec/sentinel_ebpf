from __future__ import annotations

from detector.building_blocks.core.base import BlockContext, BuildingBlock


class FeatureFieldExtractor(BuildingBlock):
  """Feature side: read one string field from ``EventEnvelope`` (e.g. ``syscall_name``)."""

  def __init__(self, field: str, *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._field = (field or "").strip()

  def depends_on(self) -> list[BuildingBlock]:
    return []

  def forward(self, ctx: BlockContext) -> None:
    evt = ctx.event
    raw = getattr(evt, self._field, "") if self._field else ""
    ctx.outputs[id(self)] = str(raw or "").strip().lower()

