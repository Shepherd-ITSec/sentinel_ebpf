"""Online IDS: run ``BuildingBlockManager`` per event; primary score from ``final_bb``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from detector.building_blocks.core.base import BlockContext, BuildingBlock
from detector.building_blocks.core.manager import BuildingBlockManager

if TYPE_CHECKING:
  import events_pb2


class OnlineIDS:
  """Holds a user-wired graph ending at ``final_bb``."""

  def __init__(self, final_bb: BuildingBlock, *, pipeline_id: str = "") -> None:
    self.pipeline_id = pipeline_id
    self._manager = BuildingBlockManager(final_bb)

  @property
  def manager(self) -> BuildingBlockManager:
    return self._manager

  @property
  def final_bb(self) -> BuildingBlock:
    return self._manager.final_bb

  def run_event(self, evt: "events_pb2.EventEnvelope") -> object:
    """Run the graph for one event and return the final block output."""
    ctx = BlockContext(evt)
    self._manager.run_event(ctx)
    return ctx.outputs[id(self._manager.final_bb)]

