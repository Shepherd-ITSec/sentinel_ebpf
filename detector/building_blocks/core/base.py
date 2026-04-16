"""LID-DS-style composable building blocks for online IDS.

Payload conventions (see module docstring on ctx):
- **Feature side:** ``dict[str, float]``, optional ``Meta``, tokens, embedding vectors.
- **Model / score side:** prefer :class:`ScoreOutput` (raw + scaled), scalars, or tensors —
  not ``dict`` as the default between score blocks. Dict-native detectors (z-score, freq1d)
  take a feature dict from a dedicated upstream block; their *output* is still
  :class:`ScoreOutput`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import events_pb2


@dataclass
class ScoreOutput:
  """Scalar anomaly scores (model side, not feature dict)."""

  raw: float
  scaled: float


@dataclass
class PrimaryScoreOutput:
  """Score selected for thresholding after calibration / suppression policy."""

  raw: float
  scaled: float
  primary: float
  score_mode: str
  suppressed: bool


@dataclass
class DecisionOutput:
  """Final score + threshold decision emitted by scoring blocks."""

  raw: float
  scaled: float
  primary: float
  score_mode: str
  suppressed: bool
  threshold: float
  anomaly: bool


class BlockContext:
  """Per-event execution context."""

  __slots__ = ("event", "outputs")

  def __init__(self, event: "events_pb2.EventEnvelope") -> None:
    self.event = event
    # BuildingBlock -> output value (any contract documented by producer/consumer)
    self.outputs: dict[int, Any] = {}

  def get_parent_output(self, parent: "BuildingBlock") -> Any:
    return self.outputs[id(parent)]


class BuildingBlock:
  """Base class: store ctor-wired deps, run once per graph evaluation per event.

  Subclasses implement :meth:`depends_on` and :meth:`forward`. Optional
  :meth:`get_state` / :meth:`set_state` for checkpointing (default empty).
  """

  def __init__(self, *, block_uid: str | None = None) -> None:
    self.block_uid: str = (block_uid or "").strip() or str(uuid.uuid4())

  def depends_on(self) -> list["BuildingBlock"]:
    return []

  def forward(self, ctx: BlockContext) -> None:
    raise NotImplementedError

  def get_state(self) -> dict[str, Any]:
    return {}

  def set_state(self, state: dict[str, Any]) -> None:
    return

