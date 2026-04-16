from __future__ import annotations

import bisect
import math
from collections import deque


class OnlinePercentileCalibrator:
  """
  Online percentile calibration for anomaly scores.

  Maintains a fixed-size window of past log1p(raw_score) values and returns the
  percentile rank of the current value w.r.t. the past window (prequential).
  """

  def __init__(self, window_size: int = 2048, warmup: int = 128):
    self.window_size = int(max(32, window_size))
    self.warmup = int(max(0, warmup))
    self._queue: deque[float] = deque(maxlen=self.window_size)
    self._sorted: list[float] = []

  @staticmethod
  def _lograw(score_raw: float) -> float:
    return float(math.log1p(max(0.0, float(score_raw))))

  def percentile_prequential(self, score_raw: float) -> float:
    """
    Return percentile in [0,1] based on past window, then update window with this score.
    """
    x = self._lograw(score_raw)
    n = len(self._sorted)
    if n <= 0 or n < self.warmup:
      pct = 0.0
    else:
      k = bisect.bisect_right(self._sorted, x)
      pct = float(k) / float(n)

    if len(self._queue) >= self.window_size:
      old = self._queue.popleft()
      j = bisect.bisect_left(self._sorted, old)
      if 0 <= j < len(self._sorted):
        self._sorted.pop(j)

    self._queue.append(x)
    bisect.insort(self._sorted, x)
    return float(min(1.0, max(0.0, pct)))

  def get_state(self) -> dict[str, object]:
    return {
      "window_size": int(self.window_size),
      "warmup": int(self.warmup),
      "queue": list(self._queue),
      "sorted": list(self._sorted),
    }

  def set_state(self, state: dict[str, object]) -> None:
    self.window_size = int(state.get("window_size", self.window_size))
    self.warmup = int(state.get("warmup", self.warmup))
    self._queue = deque((float(x) for x in state.get("queue", []) or []), maxlen=self.window_size)
    self._sorted = [float(x) for x in state.get("sorted", []) or []]

