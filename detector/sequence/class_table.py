from __future__ import annotations


class TokenClassTable:
  """Map token string -> class index 0 .. N-1 (grows online)."""

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
