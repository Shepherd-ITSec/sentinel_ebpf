from __future__ import annotations

from dataclasses import dataclass

from detector.embeddings.online_word2vec import OnlineTokenWord2Vec
from detector.sequence.class_table import TokenClassTable
from detector.sequence.ngram_buffer import StreamNgramBuffer


@dataclass(frozen=True)
class SequenceFeatureMeta:
  ready: bool
  target_id: int
  num_classes: int


class SequenceFeatureDict(dict[str, float]):
  """Feature dict plus sequence-model metadata kept out of generic feature space."""

  def __init__(self, values: dict[str, float], *, meta: SequenceFeatureMeta) -> None:
    super().__init__(values)
    self.sequence_meta = meta


class SequenceContextFeatureExtractor:
  """Stateful token-sequence featurizer that emits flat context features."""

  def __init__(
    self,
    *,
    vector_size: int,
    sentence_len: int,
    seed: int,
    w2v_window: int,
    w2v_sg: int,
    update_every: int,
    epochs: int,
    post_warmup_lr_scale: float,
    warmup_events: int,
    ngram_length: int,
    thread_aware: bool,
    feature_prefix: str,
  ) -> None:
    self._feature_prefix = str(feature_prefix).strip() or "sequence_ctx"
    self._n_full = int(ngram_length)
    if self._n_full < 2:
      raise ValueError("ngram_length must be >= 2")
    self._n_context = self._n_full - 1
    self._emb_dim = int(vector_size)
    self._update_every = max(1, int(update_every))
    self._epochs = max(1, int(epochs))
    self._lr_scale_post = float(post_warmup_lr_scale)
    self._warmup_events = int(warmup_events)
    self._event_index = 0
    self._classes = TokenClassTable()
    self._w2v = OnlineTokenWord2Vec(
      vector_size=self._emb_dim,
      sentence_len=int(sentence_len),
      seed=int(seed),
      w2v_window=int(w2v_window),
      w2v_sg=int(w2v_sg),
    )
    self._ngram = StreamNgramBuffer(
      thread_aware=bool(thread_aware),
      ngram_length=self._n_full,
      element_width=self._emb_dim,
    )
    self._context_feature_names = tuple(
      f"{self._feature_prefix}_{idx:03d}" for idx in range(self._n_context * self._emb_dim)
    )
    self._embedding_feature_names = tuple(f"syscall_w2v_{idx:03d}" for idx in range(self._emb_dim))

  @property
  def context_feature_names(self) -> tuple[str, ...]:
    return self._context_feature_names

  @property
  def embedding_feature_names(self) -> tuple[str, ...]:
    return self._embedding_feature_names

  def _zero_values(self) -> dict[str, float]:
    return {name: 0.0 for name in self._context_feature_names}

  def observe_token(self, *, stream_id: int, token: str) -> SequenceFeatureDict:
    """Observe one token and return a flattened n-gram context feature vector.

    This is used by sequence models (e.g. `sequence_mlp`) that predict the *next*
    token from the previous (n-1) token embeddings.

    Behavior:
    - Updates the online Word2Vec model with the new token for this stream.
    - Maintains an n-gram ring buffer per stream (thread-aware when configured).
    - Returns a fixed-size feature dict representing the context (previous (n-1)
      embeddings concatenated) once the buffer is full.

    Output:
    - Always returns a `SequenceFeatureDict` with the same keys (`context_feature_names`).
    - `sequence_meta.ready` is False until the n-gram buffer has enough history.
    - `sequence_meta.target_id` is the registered class id for `token`, which is
      typically used as the supervised target for next-token prediction.
    """
    self._event_index += 1
    target_id = self._classes.register(token)
    self._w2v.observe(int(stream_id), token)

    in_warmup = self._event_index <= self._warmup_events
    if self._event_index % self._update_every == 0:
      pending = self._w2v.drain_pending_sentences()
      if pending:
        alpha_scale = 1.0 if in_warmup else self._lr_scale_post
        self._w2v.train_on_sentences(pending, epochs=self._epochs, alpha_scale=alpha_scale)

    emb = self._w2v.vector_for(token)
    full, ok = self._ngram.push(int(stream_id), emb.tolist())
    if not ok or full is None:
      return SequenceFeatureDict(
        self._zero_values(),
        meta=SequenceFeatureMeta(
          ready=False,
          target_id=target_id,
          num_classes=self._classes.num_classes,
        ),
      )

    context = self._ngram.context_prefix(full, self._n_context)
    out = {
      name: float(value)
      for name, value in zip(self._context_feature_names, context, strict=True)
    }
    return SequenceFeatureDict(
      out,
      meta=SequenceFeatureMeta(
        ready=True,
        target_id=target_id,
        num_classes=self._classes.num_classes,
      ),
    )

  def observe_embedding(self, *, stream_id: int, token: str) -> dict[str, float]:
    """Observe one token and return its embedding features only (no n-gram context).

    This is used by models like `zscore` that operate on a per-event numeric vector
    and do not require next-token targets or n-gram buffering.

    Behavior:
    - Updates the online Word2Vec model with the new token for this stream.
    - Optionally trains on pending sentences every `update_every` events.
    - Returns the current token's embedding as `syscall_w2v_###` feature keys.
    """
    self._event_index += 1
    self._classes.register(token)
    self._w2v.observe(int(stream_id), token)

    in_warmup = self._event_index <= self._warmup_events
    if self._event_index % self._update_every == 0:
      pending = self._w2v.drain_pending_sentences()
      if pending:
        alpha_scale = 1.0 if in_warmup else self._lr_scale_post
        self._w2v.train_on_sentences(pending, epochs=self._epochs, alpha_scale=alpha_scale)

    emb = self._w2v.vector_for(token)
    return {
      name: float(value)
      for name, value in zip(self._embedding_feature_names, emb, strict=True)
    }

  def export_word2vec_matrix(self, *, limit: int | None = None) -> tuple[list[str], "list[list[float]]"]:
    """
    Export the current Word2Vec vocabulary and embedding matrix.

    This is intentionally a simple interchange format (tokens + nested lists) so callers
    can serialize it without depending on gensim internals.
    """
    tokens, mat = self._w2v.export_matrix(limit=limit)
    return tokens, mat.tolist()

  def get_state(self) -> dict:
    return {
      "feature_prefix": str(self._feature_prefix),
      "n_full": int(self._n_full),
      "emb_dim": int(self._emb_dim),
      "update_every": int(self._update_every),
      "epochs": int(self._epochs),
      "lr_scale_post": float(self._lr_scale_post),
      "warmup_events": int(self._warmup_events),
      "event_index": int(self._event_index),
      "classes": self._classes.to_serializable(),
      "w2v": self._w2v.get_state(),
      "ngram": self._ngram.get_state(),
    }

  def set_state(self, state: dict) -> None:
    # Keep constructor-derived shapes; only restore running state.
    self._event_index = int(state.get("event_index", self._event_index))
    self._classes.load_from_pairs(list(state.get("classes", []) or []))
    w2v_state = state.get("w2v", None)
    if isinstance(w2v_state, dict):
      self._w2v.set_state(w2v_state)
    ngram_state = state.get("ngram", None)
    if isinstance(ngram_state, dict):
      self._ngram.set_state(ngram_state)
