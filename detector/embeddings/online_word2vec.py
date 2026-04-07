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
  Per-thread sentence streams → periodic gensim Word2Vec updates.

  Each time a thread's token deque reaches ``sentence_len``, we enqueue that
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
    alpha: float = 0.025,
    min_alpha: float = 1e-4,
  ) -> None:
    self.vector_size = int(vector_size)
    self.sentence_len = int(sentence_len)
    self._seed = int(seed)
    self._w2v_window = int(w2v_window)
    self._w2v_sg = int(w2v_sg)
    self._alpha = float(alpha)
    self._min_alpha = float(min_alpha)
    self._model: Word2Vec | None = None
    self._token_streams: dict[int, deque[str]] = {}
    self._pending_sentences: list[list[str]] = []

  def observe(self, tid: int, token: str) -> None:
    """Append one token to the per-thread stream; maybe enqueue a training sentence."""
    t = int(tid)
    tok = (token or "").strip().lower() or "__empty__"
    if t not in self._token_streams:
      self._token_streams[t] = deque(maxlen=self.sentence_len)
    d = self._token_streams[t]
    d.append(tok)
    if len(d) == self.sentence_len:
      self._pending_sentences.append(list(d))

  def pending_count(self) -> int:
    return len(self._pending_sentences)

  def drain_pending_sentences(self) -> list[list[str]]:
    out = self._pending_sentences
    self._pending_sentences = []
    return out

  def train_on_sentences(
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
      "alpha": float(self._alpha),
      "min_alpha": float(self._min_alpha),
      "model": self._model,
      "token_streams": {int(t): list(d) for t, d in self._token_streams.items()},
      "pending_sentences": [list(s) for s in self._pending_sentences],
    }

  def set_state(self, state: dict) -> None:
    self.vector_size = int(state.get("vector_size", self.vector_size))
    self.sentence_len = int(state.get("sentence_len", self.sentence_len))
    self._seed = int(state.get("seed", self._seed))
    self._w2v_window = int(state.get("w2v_window", self._w2v_window))
    self._w2v_sg = int(state.get("w2v_sg", self._w2v_sg))
    self._alpha = float(state.get("alpha", self._alpha))
    self._min_alpha = float(state.get("min_alpha", self._min_alpha))
    self._model = state.get("model", None)
    self._token_streams = {}
    for t, toks in (state.get("token_streams", {}) or {}).items():
      dq = deque(maxlen=self.sentence_len)
      for tok in toks or []:
        dq.append((str(tok) or "").strip().lower() or "__empty__")
      self._token_streams[int(t)] = dq
    self._pending_sentences = [list(map(str, s)) for s in (state.get("pending_sentences", []) or [])]