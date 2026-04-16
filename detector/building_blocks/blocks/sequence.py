from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from detector.building_blocks.core.base import BlockContext, BuildingBlock, ScoreOutput
from detector.building_blocks.primitives.sequence.sequence_context import (
  SequenceFeatureMeta,
  SequenceVectorContext,
  TokenClassTable,
)
from detector.building_blocks.primitives.sequence.sequence_mlp import OnlineSequenceNextTokenMLP

if TYPE_CHECKING:
  from detector.config import DetectorConfig

class SequenceNgramContextBlock(BuildingBlock):
  """Rolling n-gram over parent embedding vectors; outputs ``(dict[str,float], SequenceFeatureMeta)``."""

  def __init__(self, parents: list[BuildingBlock], cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    if len(parents) != 1:
      raise ValueError("SequenceNgramContextBlock expects exactly one parent (embedding vector)")
    self._parent = parents[0]
    self._classes = TokenClassTable()
    self._ctx = SequenceVectorContext(
      element_width=int(cfg.embedding_word2vec_dim),
      ngram_length=int(cfg.sequence_ngram_length),
      thread_aware=bool(cfg.sequence_thread_aware),
      feature_prefix="sequence_ctx",
    )

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    vec = np.asarray(ctx.get_parent_output(self._parent), dtype=np.float64)
    try:
      tid = int((ctx.event.tid or "0").strip())
    except ValueError:
      tid = 0
    token = (ctx.event.syscall_name or "").strip().lower()
    target_id = self._classes.get_label_idx(token)
    out, meta = self._ctx.observe_vector(
      stream_id=int(tid),
      vector=vec.tolist(),
      target_id=int(target_id),
      num_classes=int(self._classes.num_classes),
    )
    ctx.outputs[id(self)] = (out, meta)

  def get_state(self) -> dict[str, Any]:
    return {
      "classes": self._classes.to_serializable(),
      "ctx": self._ctx.get_state(),
    }

  def set_state(self, state: dict[str, Any]) -> None:
    pairs = state.get("classes")
    if isinstance(pairs, list):
      self._classes.load_from_pairs(list(pairs))
    cs = state.get("ctx")
    if isinstance(cs, dict):
      self._ctx.set_state(cs)


class SequenceNextTokenMLPBlock(BuildingBlock):
  """Sequence next-token MLP; parent must output ``(feature_dict, SequenceFeatureMeta)``."""

  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._det = OnlineSequenceNextTokenMLP(
      hidden_size=int(cfg.sequence_mlp_hidden_size),
      hidden_layers=int(cfg.sequence_mlp_hidden_layers),
      learning_rate=float(cfg.sequence_mlp_lr),
      model_device=str(cfg.model_device),
      seed=int(cfg.model_seed),
    )

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    payload = ctx.get_parent_output(self._parent)
    if not (isinstance(payload, tuple) and len(payload) == 2):
      raise TypeError("SequenceNextTokenMLPBlock parent must output (dict, Meta)")
    features, meta = payload[0], payload[1]
    raw, scaled = self._det.score_and_learn(features, meta=meta)
    ctx.outputs[id(self)] = ScoreOutput(raw=float(raw), scaled=float(scaled))

  def get_state(self) -> dict[str, Any]:
    return {"impl": self._det.get_state()}

  def set_state(self, state: dict[str, Any]) -> None:
    impl = state.get("impl")
    if isinstance(impl, dict):
      self._det.set_state(impl)

