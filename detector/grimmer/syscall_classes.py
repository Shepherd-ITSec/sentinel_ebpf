"""
Syscall name to dense class index for MLP softmax (online growing vocabulary).

Mirrors LID-DS ``IntEmbedding`` + ``OneHotEncoding`` idea: integer id per syscall,
with an extra residual unknown bucket in the original OHE — here we always
``register`` before labeling so the softmax size equals the number of distinct
syscalls seen so far (purely online).

References:
  - ``third_party/LID-DS/algorithms/features/impl/int_embedding.py``
  - ``third_party/LID-DS/algorithms/features/impl/one_hot_encoding.py``
"""

from __future__ import annotations


class SyscallClassTable:
  """Map syscall name string -> class index 0 .. N-1 (grows online)."""

  def __init__(self) -> None:
    self._name_to_id: dict[str, int] = {}

  def register(self, name: str) -> int:
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
