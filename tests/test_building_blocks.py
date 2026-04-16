"""Tests for detector.building_blocks (DAG, fusion, scaling)."""

from __future__ import annotations

import math

import pytest

import events_pb2
from detector.building_blocks import ScoreOutput
from detector.building_blocks.blocks.fusion import FusionMeanBlock
from detector.building_blocks.blocks.scaling import ScalingBlock
from detector.building_blocks.core.base import BlockContext, BuildingBlock
from detector.building_blocks.core.manager import BuildingBlockManager, topological_order


class _ConstScore(BuildingBlock):
  def __init__(self, raw: float, scaled: float, *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._raw = raw
    self._scaled = scaled

  def depends_on(self):
    return []

  def forward(self, ctx: BlockContext) -> None:
    ctx.outputs[id(self)] = ScoreOutput(raw=self._raw, scaled=self._scaled)


class _SumScores(BuildingBlock):
  def __init__(self, p1: BuildingBlock, p2: BuildingBlock, *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._p1 = p1
    self._p2 = p2

  def depends_on(self):
    return [self._p1, self._p2]

  def forward(self, ctx: BlockContext) -> None:
    s1 = ctx.get_parent_output(self._p1)
    s2 = ctx.get_parent_output(self._p2)
    ctx.outputs[id(self)] = ScoreOutput(
      raw=float(s1.raw + s2.raw),
      scaled=float(s1.scaled + s2.scaled) / 2.0,
    )


class _CycleNode(BuildingBlock):
  def __init__(self, uid: str) -> None:
    super().__init__(block_uid=uid)
    self.peer: BuildingBlock | None = None

  def depends_on(self):
    return [self.peer] if self.peer is not None else []


def test_topological_order_dependencies_before_dependents() -> None:
  a = _ConstScore(0.1, 0.2, block_uid="a")
  b = _ConstScore(0.3, 0.4, block_uid="b")
  final_bb = _SumScores(a, b, block_uid="sum")
  order = topological_order(final_bb)
  pos = {id(x): i for i, x in enumerate(order)}
  assert pos[id(final_bb)] > pos[id(a)]
  assert pos[id(final_bb)] > pos[id(b)]


def test_cycle_raises() -> None:
  n1 = _CycleNode("n1")
  n2 = _CycleNode("n2")
  n1.peer = n2
  n2.peer = n1
  with pytest.raises(ValueError, match="cycle"):
    topological_order(n2)


def test_fusion_mean_two_parents_no_concat_dict() -> None:
  las = _ConstScore(1.0, 0.5, block_uid="las")
  gas = _ConstScore(3.0, 0.7, block_uid="gas")
  fus = FusionMeanBlock([las, gas], block_uid="fus")
  mgr = BuildingBlockManager(fus)
  evt = events_pb2.EventEnvelope(
    event_id="e1",
    syscall_name="openat",
    event_group="",
    ts_unix_nano=0,
    syscall_nr=2,
    comm="bash",
    pid="1",
    tid="2",
    uid="0",
    arg0="0",
    arg1="0",
  )
  ctx = BlockContext(evt)
  mgr.run_event(ctx)
  out = ctx.outputs[id(fus)]
  assert isinstance(out, ScoreOutput)
  assert out.raw == pytest.approx(2.0)
  assert out.scaled == pytest.approx(0.6)


def test_scaling_block_exp1m() -> None:
  parent = _ConstScore(2.0, 999.0, block_uid="p")
  sc = ScalingBlock(parent, "exp1m", block_uid="s")
  mgr = BuildingBlockManager(sc)
  evt = events_pb2.EventEnvelope(
    event_id="e1",
    syscall_name="read",
    event_group="",
    ts_unix_nano=0,
    syscall_nr=0,
    comm="x",
    pid="1",
    tid="1",
    uid="0",
    arg0="0",
    arg1="0",
  )
  ctx = BlockContext(evt)
  mgr.run_event(ctx)
  out = ctx.outputs[id(sc)]
  assert isinstance(out, ScoreOutput)
  assert out.raw == pytest.approx(2.0)
  assert out.scaled == pytest.approx(1.0 - math.exp(-2.0))
