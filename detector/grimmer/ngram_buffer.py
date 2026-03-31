"""
Thread-aware rolling buffer for syscall embedding n-grams.

Ported from LID-DS ``Ngram`` (flattened dependency vectors into a per-thread deque).

Reference:
  ``third_party/LID-DS/algorithms/features/impl/ngram.py`` — class ``Ngram``,
  especially thread grouping via ``syscall.thread_id()`` and deque extension
  via ``_concat`` semantics.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable


def _concat(source_value, target_vector: list) -> None:
  """
  Port of ``Ngram._concat`` from LID-DS.

  Reference: ``third_party/LID-DS/algorithms/features/impl/ngram.py`` (``_concat``).
  """
  if isinstance(source_value, Iterable):
    if isinstance(source_value, str):
      target_vector.append(source_value)
    else:
      target_vector.extend(source_value)
  else:
    target_vector.append(source_value)


class ThreadNgramBuffer:
  """
  Maintains per-thread deques of flattened embedding components.

  Each incoming syscall contributes one embedding vector (length ``element_width``).
  When the deque reaches ``ngram_length * element_width`` elements, the window is complete.
  """

  def __init__(self, *, thread_aware: bool, ngram_length: int, element_width: int) -> None:
    self._thread_aware = thread_aware
    self._ngram_length = int(ngram_length)
    self._element_width = int(element_width)
    self._deque_length = self._ngram_length * self._element_width
    self._ngram_buffer: dict[int, deque[float]] = {}

  def clear(self) -> None:
    """Reset all thread buffers (cf. LID-DS ``Ngram.new_recording``)."""
    self._ngram_buffer = {}

  def push(self, tid: int, embedding: list[float]) -> tuple[list[float] | None, bool]:
    """
    Append one syscall embedding. Returns ``(full_window, is_complete)``.

    ``full_window`` is the flattened tuple/list of length ``ngram_length * element_width``
    when complete, else ``None``.
    """
    if len(embedding) != self._element_width:
      raise ValueError(
        f"embedding length {len(embedding)} != element_width {self._element_width}"
      )
    thread_id = int(tid) if self._thread_aware else 0
    if thread_id not in self._ngram_buffer:
      self._ngram_buffer[thread_id] = deque(maxlen=self._deque_length)

    buf = self._ngram_buffer[thread_id]
    chunk: list[float] = []
    _concat(embedding, chunk)
    buf.extend(chunk)

    if len(buf) == self._deque_length:
      return list(buf), True
    return None, False

  def context_prefix(self, full_window: list[float], context_ngrams: int) -> list[float]:
    """
    First ``context_ngrams * element_width`` values (LID-DS ``Select`` slice).

    Reference: ``third_party/LID-DS/algorithms/ids_mlp_main.py`` —
    ``Select(ngram, start=0, end=NGRAM_LENGTH * W2V_SIZE)`` with
    ``Ngram(..., ngram_length=NGRAM_LENGTH + 1)``.
    """
    take = int(context_ngrams) * self._element_width
    return full_window[:take]
