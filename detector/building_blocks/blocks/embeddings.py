from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from detector.building_blocks.core.base import BlockContext, BuildingBlock
from detector.building_blocks.primitives.embeddings.word2vec import OnlineGensimWord2Vec

if TYPE_CHECKING:
  from detector.config import DetectorConfig


class Word2VecEmbeddingBlock(BuildingBlock):
  """Embedding block: parent must output syscall token string."""

  def __init__(self, parent: BuildingBlock, cfg: "DetectorConfig", *, block_uid: str | None = None) -> None:
    super().__init__(block_uid=block_uid)
    self._parent = parent
    self._w2v = OnlineGensimWord2Vec(
      vector_size=int(cfg.embedding_word2vec_dim),
      sentence_len=int(cfg.embedding_word2vec_sentence_len),
      seed=int(cfg.model_seed),
      w2v_window=int(cfg.embedding_word2vec_window),
      w2v_sg=int(cfg.embedding_word2vec_sg),
      update_every=int(cfg.embedding_word2vec_update_every),
      epochs=int(cfg.embedding_word2vec_epochs),
      warmup_events=int(cfg.warmup_events),
      post_warmup_lr_scale=float(cfg.embedding_word2vec_post_warmup_lr_scale),
    )

  def depends_on(self) -> list[BuildingBlock]:
    return [self._parent]

  def forward(self, ctx: BlockContext) -> None:
    token = str(ctx.get_parent_output(self._parent))
    try:
      tid = int((ctx.event.tid or "0").strip())
    except ValueError:
      tid = 0
    vec = self._w2v.observe_and_vector(int(tid), token)
    ctx.outputs[id(self)] = np.asarray(vec, dtype=np.float64)

  def get_state(self) -> dict[str, Any]:
    return {"w2v": self._w2v.get_state()}

  def set_state(self, state: dict[str, Any]) -> None:
    w = state.get("w2v")
    if isinstance(w, dict):
      self._w2v.set_state(w)
