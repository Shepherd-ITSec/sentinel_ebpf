"""
Per-``event_group`` Grimmer pipeline: thread-aware n-gram of syscall Word2Vec embeddings + online MLP.

Composition mirrors LID-DS ``ids_mlp_main.py``:
  ``SyscallName -> IntEmbedding -> W2VEmbedding -> Ngram(thread_aware) -> Select -> MLP``.

We use syscall **names** as Word2Vec tokens and as class labels (via ``SyscallClassTable``).

References:
  - ``third_party/LID-DS/algorithms/ids_mlp_main.py``
  - ``third_party/LID-DS/algorithms/features/impl/ngram.py``
  - ``third_party/LID-DS/algorithms/features/impl/w2v_embedding.py``
"""

from __future__ import annotations

import logging
from typing import Any

import events_pb2
from detector.config import DetectorConfig
from detector.grimmer.mlp_online import GrimmerOnlineMLP
from detector.grimmer.ngram_buffer import ThreadNgramBuffer
from detector.embeddings.online_word2vec import OnlineSyscallWord2Vec
from detector.grimmer.syscall_classes import SyscallClassTable

logger = logging.getLogger(__name__)


class GrimmerPipeline:
  """
  Stateful feature extractor + MLP for one ``event_group`` (or ``__default__``).

  Model-independent warmup: higher Word2Vec training intensity until ``warmup_events``;
  thereafter smaller effective learning rate via ``embedding_post_warmup_lr_scale``.
  """

  def __init__(self, cfg: DetectorConfig) -> None:
    self._cfg = cfg
    self._n_full = int(cfg.grimmer_ngram_length)
    if self._n_full < 2:
      raise ValueError("grimmer_ngram_length must be >= 2")
    self._n_context = self._n_full - 1
    self._emb_dim = int(cfg.embedding_word2vec_dim)
    self._warmup_events = int(getattr(cfg, "warmup_events", 0))
    self._update_every = max(1, int(cfg.embedding_word2vec_update_every))
    self._epochs = max(1, int(cfg.embedding_word2vec_epochs))
    self._lr_scale_post = float(cfg.embedding_word2vec_post_warmup_lr_scale)
    self._thread_aware = bool(cfg.grimmer_thread_aware)

    self._w2v = OnlineSyscallWord2Vec(
      vector_size=self._emb_dim,
      sentence_len=int(cfg.embedding_word2vec_sentence_len),
      seed=int(cfg.model_seed),
      w2v_window=int(cfg.embedding_word2vec_window),
      w2v_sg=int(cfg.embedding_word2vec_sg),
    )
    self._ngram = ThreadNgramBuffer(
      thread_aware=self._thread_aware,
      ngram_length=self._n_full,
      element_width=self._emb_dim,
    )
    self._classes = SyscallClassTable()
    input_dim = self._n_context * self._emb_dim
    self._mlp = GrimmerOnlineMLP(
      input_dim=input_dim,
      hidden_size=int(cfg.grimmer_mlp_hidden_size),
      hidden_layers=int(cfg.grimmer_mlp_hidden_layers),
      learning_rate=float(cfg.grimmer_mlp_lr),
      model_device=cfg.model_device,
      seed=int(cfg.model_seed),
    )
    self._event_index = 0

  @property
  def algorithm(self) -> str:
    return "grimmer_mlp"

  def _syscall_token(self, evt: events_pb2.EventEnvelope) -> str:
    return (evt.syscall_name or "").strip().lower() or "__empty__"

  def _tid(self, evt: events_pb2.EventEnvelope) -> int:
    try:
      return int((evt.tid or "0").strip() or "0")
    except ValueError:
      return 0

  def score_and_learn_event(self, evt: events_pb2.EventEnvelope) -> tuple[float, float]:
    """
    One streaming step: update Word2Vec, push embedding into thread n-gram buffer,
    then MLP score+learn when the buffer is full.

    Returns ``(raw, scaled)`` compatible with ``OnlineAnomalyDetector`` scaling for non-HST models.
    """
    self._event_index += 1
    name = self._syscall_token(evt)
    tid = self._tid(evt)

    self._w2v.observe(tid, name)

    in_warmup = self._event_index <= self._warmup_events
    if self._event_index % self._update_every == 0:
      pending = self._w2v.drain_pending_sentences()
      if pending:
        alpha_scale = 1.0 if in_warmup else self._lr_scale_post
        self._w2v.train_on_sentences(pending, epochs=self._epochs, alpha_scale=alpha_scale)

    emb = self._w2v.vector_for(name)
    y = self._classes.register(name)
    self._mlp.ensure_num_classes(self._classes.num_classes)

    full, ok = self._ngram.push(tid, emb.tolist())
    if not ok or full is None:
      return 0.0, 0.0

    x = self._ngram.context_prefix(full, self._n_context)
    return self._mlp.score_and_learn(x, y)

  @property
  def events_processed(self) -> int:
    return self._event_index

  def get_state(self) -> dict[str, Any]:
    return {
      "event_index": self._event_index,
      "classes": self._classes.to_serializable(),
      "mlp": self._mlp.get_state(),
    }

  def set_state(self, state: dict[str, Any]) -> None:
    self._event_index = int(state.get("event_index", 0))
    pairs = state.get("classes") or []
    self._classes.load_from_pairs(pairs)
    self._mlp.set_state(state.get("mlp") or {})
