"""
Online Word2Vec over syscall-name tokens using gensim, adapted from LID-DS.

LID-DS collects fixed-length sentences (n-grams of syscall ints) without deduplication
optional via ``distinct=False``. We always keep duplicates (no dedup).

Reference:
  ``third_party/LID-DS/algorithms/features/impl/w2v_embedding.py`` — class ``W2VEmbedding``:
  inner ``Ngram`` for sentence shape, ``Word2Vec(..., min_count=1)``, ``fit()``.

We differ by calling ``build_vocab(update=True)`` and ``train()`` repeatedly for
continuous learning after an initial warmup phase.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


class OnlineTokenWord2Vec:
  """
  Per-actor sentence streams → periodic gensim Word2Vec updates.

  Each time a actor's token deque reaches ``sentence_len``, we enqueue that
  sentence (list of syscall name strings) for training.
  """

  def __init__(
    self,
    *,
    vector_size: int,
    sentence_len: int,
    seed: int,
    w2v_window: int,
    w2v_sg: int = 1,
    update_every: int = 1,
    epochs: int = 1,
    warmup_events: int = 0,
    post_warmup_lr_scale: float = 0.25,
    alpha: float = 0.025,
    min_alpha: float = 1e-4,
  ) -> None:
    self.vector_size = int(vector_size)
    self.sentence_len = int(sentence_len)
    self._seed = int(seed)
    self._w2v_window = int(w2v_window)
    self._w2v_sg = int(w2v_sg)
    self._update_every = max(1, int(update_every))
    self._epochs = max(1, int(epochs))
    self._warmup_events = int(warmup_events)
    self._post_warmup_lr_scale = float(post_warmup_lr_scale)
    self._event_index = 0
    self._alpha = float(alpha)
    self._min_alpha = float(min_alpha)
    self._model: Word2Vec | None = None
    self._sentences: dict[int, deque[str]] = {}
    self._pending_finished_sentences: list[list[str]] = []

  def append_word_to_sentence(self, sentence_id: int, word: str) -> None:
    """Append one word to the sentence; If sentence is full, enqueue it for training."""
    sentence_idx = int(sentence_id)
    token = (word or "").strip().lower() or "__empty__"
    if sentence_idx not in self._sentences:
      self._sentences[sentence_idx] = deque(maxlen=self.sentence_len)
    d = self._sentences[sentence_idx]
    d.append(token)
    if len(d) == self.sentence_len:
      self._pending_finished_sentences.append(list(d))

  def observe_and_vector(self, sentence_id: int, word: str) -> np.ndarray:
    """
    Observe one word and return its current embedding vector.

    This method encapsulates the "background learning" cadence:
    - collects per-actor fixed-length sentences
    - every `update_every` events, trains on all pending sentences (if any)
    - uses a warmup phase where LR is not scaled down, then applies `post_warmup_lr_scale`
    """
    self._event_index += 1
    self.append_word_to_sentence(sentence_id, word)

    in_warmup = self._event_index <= self._warmup_events
    if self._event_index % self._update_every == 0:
      pending = self.drain_pending_finished_sentences()
      if pending:
        alpha_scale = 1.0 if in_warmup else self._post_warmup_lr_scale
        self.train_on_finished_sentences(pending, epochs=self._epochs, alpha_scale=alpha_scale)

    return self.vector_for(word)

  def pending_count(self) -> int:
    return len(self._pending_finished_sentences)

  def drain_pending_finished_sentences(self) -> list[list[str]]:
    out = self._pending_finished_sentences
    self._pending_finished_sentences = []
    return out

  def train_on_finished_sentences(
    self,
    sentences: list[list[str]],
    *,
    epochs: int,
    alpha_scale: float = 1.0,
  ) -> None:
    if not sentences:
      return
    start_alpha = max(self._min_alpha, self._alpha * float(alpha_scale))
    end_alpha = max(self._min_alpha, self._min_alpha * float(alpha_scale))

    if self._model is None:
      self._model = Word2Vec(
        vector_size=self.vector_size,
        window=min(self._w2v_window, max(1, self.sentence_len - 1)),
        min_count=1,
        sg=self._w2v_sg, # 1 = skip-gram, 0 = continuous bag of words
        seed=self._seed,
        workers=1,
        alpha=start_alpha,
        min_alpha=end_alpha,
      )
      self._model.build_vocab(sentences)
      self._model.train(
        sentences,
        total_examples=len(sentences),
        epochs=int(max(1, epochs)),
        start_alpha=start_alpha,
        end_alpha=end_alpha,
      )
      logger.debug("Word2Vec initial build: %d sentences, vocab=%d", len(sentences), len(self._model.wv))
      return

    self._model.build_vocab(sentences, update=True)
    self._model.train(
      sentences,
      total_examples=len(sentences),
      epochs=int(max(1, epochs)),
      start_alpha=start_alpha,
      end_alpha=end_alpha,
    )

  def vector_for(self, token: str) -> np.ndarray:
    """Embedding for ``token``; zeros if unknown (before first successful train)."""
    tok = (token or "").strip().lower() or "__empty__"
    if self._model is None or tok not in self._model.wv:
      return np.zeros(self.vector_size, dtype=np.float32)
    return np.asarray(self._model.wv[tok], dtype=np.float32)

  def vocab_tokens(self) -> list[str]:
    """Return current vocabulary tokens (empty if untrained)."""
    if self._model is None:
      return []
    keys = self._model.wv.index_to_key
    # gensim typing can be loose here; normalize to plain strings
    return [str(k) for k in keys if k is not None]

  def export_matrix(self, *, limit: int | None = None) -> tuple[list[str], np.ndarray]:
    """
    Export (tokens, matrix) for visualization/analysis.

    tokens: list[str] length N
    matrix: float32 array shape (N, vector_size)
    """
    if self._model is None:
      return ([], np.zeros((0, self.vector_size), dtype=np.float32))
    keys = [str(k) for k in self._model.wv.index_to_key if k is not None]
    if limit is not None:
      keys = keys[: max(0, int(limit))]
    if not keys:
      return ([], np.zeros((0, self.vector_size), dtype=np.float32))
    mat = np.asarray([self._model.wv[k] for k in keys], dtype=np.float32)
    return (keys, mat)

  def get_state(self) -> dict:
    """Serialize full online state, including gensim model and pending buffers."""
    return {
      "vector_size": int(self.vector_size),
      "sentence_len": int(self.sentence_len),
      "seed": int(self._seed),
      "w2v_window": int(self._w2v_window),
      "w2v_sg": int(self._w2v_sg),
      "update_every": int(self._update_every),
      "epochs": int(self._epochs),
      "warmup_events": int(self._warmup_events),
      "post_warmup_lr_scale": float(self._post_warmup_lr_scale),
      "event_index": int(self._event_index),
      "alpha": float(self._alpha),
      "min_alpha": float(self._min_alpha),
      "model": self._model,
      "sentences": {int(t): list(d) for t, d in self._sentences.items()},
      "pending_finished_sentences": [list(s) for s in self._pending_finished_sentences],
    }

  def set_state(self, state: dict) -> None:
    self.vector_size = int(state.get("vector_size", self.vector_size))
    self.sentence_len = int(state.get("sentence_len", self.sentence_len))
    self._seed = int(state.get("seed", self._seed))
    self._w2v_window = int(state.get("w2v_window", self._w2v_window))
    self._w2v_sg = int(state.get("w2v_sg", self._w2v_sg))
    self._update_every = max(1, int(state.get("update_every", self._update_every)))
    self._epochs = max(1, int(state.get("epochs", self._epochs)))
    self._warmup_events = int(state.get("warmup_events", self._warmup_events))
    self._post_warmup_lr_scale = float(state.get("post_warmup_lr_scale", self._post_warmup_lr_scale))
    self._event_index = int(state.get("event_index", self._event_index))
    self._alpha = float(state.get("alpha", self._alpha))
    self._min_alpha = float(state.get("min_alpha", self._min_alpha))
    self._model = state.get("model", None)
    self._sentences = {}
    for sentence_id, words in (state.get("sentences", {}) or {}).items():  
      dq = deque(maxlen=self.sentence_len)
      for word in words or []:
        dq.append((str(word) or "").strip().lower() or "__empty__")
      self._sentences[int(sentence_id)] = dq
    self._pending_finished_sentences = [list(map(str, s)) for s in (state.get("pending_finished_sentences", []) or [])]