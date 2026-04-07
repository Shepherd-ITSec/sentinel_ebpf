from __future__ import annotations

from collections import defaultdict, deque


class StreamNgramBuffer:
  """Thread-aware rolling n-gram buffer over fixed-width float vectors."""

  def __init__(self, *, thread_aware: bool, ngram_length: int, element_width: int) -> None:
    self._thread_aware = bool(thread_aware)
    self._ngram_length = int(ngram_length)
    self._element_width = int(element_width)
    self._global = deque(maxlen=self._ngram_length)
    self._per_stream: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=self._ngram_length))

  def _buf(self, stream_id: int) -> deque[list[float]]:
    if not self._thread_aware:
      return self._global
    return self._per_stream[int(stream_id)]

  def push(self, stream_id: int, element: list[float]) -> tuple[list[float] | None, bool]:
    if len(element) != self._element_width:
      raise ValueError(f"Expected element width {self._element_width}, got {len(element)}")
    buf = self._buf(stream_id)
    buf.append(list(element))
    if len(buf) < self._ngram_length:
      return None, False
    out: list[float] = []
    for row in buf:
      out.extend(row)
    return out, True

  @staticmethod
  def context_prefix(full_ngram: list[float], n_context: int) -> list[float]:
    if n_context <= 0:
      return []
    width = len(full_ngram) // (n_context + 1)
    if width * (n_context + 1) != len(full_ngram):
      raise ValueError("Invalid n-gram width")
    return full_ngram[: n_context * width]

  def get_state(self) -> dict:
    def _dump_deque(dq: deque[list[float]]) -> list[list[float]]:
      return [list(row) for row in list(dq)]

    return {
      "thread_aware": bool(self._thread_aware),
      "ngram_length": int(self._ngram_length),
      "element_width": int(self._element_width),
      "global": _dump_deque(self._global),
      "per_stream": {int(k): _dump_deque(v) for k, v in self._per_stream.items()},
    }

  def set_state(self, state: dict) -> None:
    self._thread_aware = bool(state.get("thread_aware", self._thread_aware))
    self._ngram_length = int(state.get("ngram_length", self._ngram_length))
    self._element_width = int(state.get("element_width", self._element_width))
    self._global = deque(maxlen=self._ngram_length)
    for row in state.get("global", []) or []:
      self._global.append([float(x) for x in row])
    self._per_stream = defaultdict(lambda: deque(maxlen=self._ngram_length))
    per = state.get("per_stream", {}) or {}
    for k, rows in per.items():
      dq = self._per_stream[int(k)]
      for row in rows or []:
        dq.append([float(x) for x in row])
