from __future__ import annotations

from dataclasses import dataclass

from detector.building_blocks.primitives.sequence.ngram_buffer import StreamNgramBuffer


class TokenClassTable:
  """Map token string -> class index 0 .. N-1 (grows online)."""

  def __init__(self) -> None:
    self._name_to_id: dict[str, int] = {}

  def get_label_idx(self, name: str) -> int:
    key = (name or "").strip().lower() or "__empty__"
    if key not in self._name_to_id:
      self._name_to_id[key] = len(self._name_to_id)
    return self._name_to_id[key]

  @property
  def num_classes(self) -> int:
    return len(self._name_to_id)

  def to_serializable(self) -> list[tuple[str, int]]:
    return list(self._name_to_id.items())

  def load_from_pairs(self, pairs: list[tuple[str, int]]) -> None:
    self._name_to_id = {str(k): int(v) for k, v in pairs}


@dataclass(frozen=True)
class SequenceFeatureMeta:
  ready: bool
  target_id: int
  num_classes: int


class SequenceVectorContext:
  """Stateful rolling n-gram context over arbitrary fixed-width vectors."""

  def __init__(
    self,
    *,
    element_width: int,
    ngram_length: int,
    thread_aware: bool,
    feature_prefix: str,
  ) -> None:
    self._feature_prefix = str(feature_prefix).strip() or "sequence_ctx"
    self._n_max = int(ngram_length)
    if self._n_max < 2:
      raise ValueError("ngram_length must be >= 2")
    self._n_context = self._n_max - 1
    self._emb_dim = int(element_width)
    self._ngram = StreamNgramBuffer(
      actor_aware=bool(thread_aware),
      ngram_length=self._n_max,
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

  def observe_vector(
    self,
    *,
    stream_id: int,
    vector: list[float],
    target_id: int,
    num_classes: int,
  ) -> tuple[dict[str, float], SequenceFeatureMeta]:
    full, ok = self._ngram.push(int(stream_id), list(vector))
    if not ok or full is None:
      return self._zero_values(), SequenceFeatureMeta(
        ready=False, target_id=int(target_id), num_classes=int(num_classes)
      )

    context = StreamNgramBuffer.context_prefix(full, self._n_context)
    out: dict[str, float] = {
      name: float(value)
      for name, value in zip(self._context_feature_names, context, strict=True)
    }
    return out, SequenceFeatureMeta(ready=True, target_id=int(target_id), num_classes=int(num_classes))

  def get_state(self) -> dict:
    return {
      "feature_prefix": str(self._feature_prefix),
      "n_full": int(self._n_max),
      "emb_dim": int(self._emb_dim),
      "ngram": self._ngram.get_state(),
    }

  def set_state(self, state: dict) -> None:
    ngram_state = state.get("ngram", None)
    if isinstance(ngram_state, dict):
      self._ngram.set_state(ngram_state)

