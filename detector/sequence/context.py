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

  @property
  def context_feature_names(self) -> tuple[str, ...]:
    return self._context_feature_names

  def _zero_values(self) -> dict[str, float]:
    return {name: 0.0 for name in self._context_feature_names}

  def observe_token(self, *, stream_id: int, token: str) -> SequenceFeatureDict:
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
