"""Topological execution order: dependencies run before dependents."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from detector.building_blocks.core.base import BlockContext, BuildingBlock


def collect_reachable(final_bb: BuildingBlock) -> list[BuildingBlock]:
  """All blocks reachable from ``final_bb`` (including ``final_bb``)."""
  seen: set[int] = set()
  out: list[BuildingBlock] = []
  stack = [final_bb]
  while stack:
    b = stack.pop()
    bid = id(b)
    if bid in seen:
      continue
    seen.add(bid)
    out.append(b)
    for d in b.depends_on():
      stack.append(d)
  return out


def topological_order(final_bb: BuildingBlock) -> list[BuildingBlock]:
  """Dependencies first, then dependents (Kahn on edges dep -> block)."""
  nodes = collect_reachable(final_bb)
  node_set = {id(n) for n in nodes}
  children: dict[int, list[BuildingBlock]] = defaultdict(list)
  indegree: dict[int, int] = {}

  for b in nodes:
    indegree[id(b)] = len(b.depends_on())
    for d in b.depends_on():
      if id(d) not in node_set:
        raise ValueError("depends_on() references a block not reachable from final_bb graph")
      children[id(d)].append(b)

  q: deque[BuildingBlock] = deque([b for b in nodes if indegree[id(b)] == 0])
  ordered: list[BuildingBlock] = []
  while q:
    n = q.popleft()
    ordered.append(n)
    for c in children[id(n)]:
      indegree[id(c)] -= 1
      if indegree[id(c)] == 0:
        q.append(c)

  if len(ordered) != len(nodes):
    raise ValueError("BuildingBlock graph has a cycle or inconsistent depends_on()")
  return ordered


class BuildingBlockManager:
  """LID-DS-style manager: build order from ``final_bb``, run all reachable nodes per event."""

  def __init__(self, final_bb: BuildingBlock) -> None:
    self._final_bb = final_bb
    self._order = topological_order(final_bb)

  @property
  def final_bb(self) -> BuildingBlock:
    return self._final_bb

  @property
  def execution_order(self) -> list[BuildingBlock]:
    return list(self._order)

  def run_event(self, ctx: BlockContext) -> None:
    for bb in self._order:
      bb.forward(ctx)

  def all_blocks(self) -> Iterable[BuildingBlock]:
    return iter(self._order)

