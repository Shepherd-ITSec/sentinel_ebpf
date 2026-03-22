#!/usr/bin/env python3
"""Compare offline global vs online prequential percentile flag counts (detector logic)."""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, List


class OnlinePercentileCalibrator:
    """Mirror of detector/model.py OnlinePercentileCalibrator."""

    def __init__(self, window_size: int = 2048, warmup: int = 128):
        self.window_size = int(max(32, window_size))
        self.warmup = int(max(0, warmup))
        self._queue: deque[float] = deque(maxlen=self.window_size)
        self._sorted: List[float] = []

    @staticmethod
    def _lograw(score_raw: float) -> float:
        return float(math.log1p(max(0.0, float(score_raw))))

    def percentile_prequential(self, score_raw: float) -> float:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_dump", type=Path, help="replay_dump.jsonl")
    ap.add_argument("--percentile", type=float, default=98.36, help="Global percentile cutoff (default 98.36)")
    ap.add_argument("--online-threshold", type=float, default=None, help="Flag if online pct >= this (default: percentile/100)")
    ap.add_argument("--window-size", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=128)
    args = ap.parse_args()

    pct_q = args.percentile / 100.0
    online_t = args.online_threshold if args.online_threshold is not None else pct_q

    xs: list[float] = []
    rows: list[tuple[str, float]] = []

    with args.replay_dump.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            if "score_raw" not in obj:
                continue
            raw = float(obj["score_raw"])
            g = str(obj.get("event_group") or "")
            x = math.log1p(max(0.0, raw))
            xs.append(x)
            rows.append((g, raw))

    import numpy as np

    arr = np.asarray(xs, dtype=np.float64)
    n = len(arr)
    # Offline: threshold = value at global percentile (same notion as "top (100-p)%")
    T_global = float(np.percentile(arr, args.percentile))
    offline_flagged = int(np.sum(arr >= T_global))

    # Online: one calibrator per event_group
    calibrators: Dict[str, OnlinePercentileCalibrator] = {}
    online_flagged = 0
    for g, raw in rows:
        cal = calibrators.get(g)
        if cal is None:
            cal = OnlinePercentileCalibrator(window_size=args.window_size, warmup=args.warmup)
            calibrators[g] = cal
        pct = cal.percentile_prequential(raw)
        if pct >= online_t:
            online_flagged += 1

    print(f"replay_dump: {args.replay_dump}")
    print(f"n_events (with score_raw): {n}")
    print(f"event_groups: {len(calibrators)}")
    print()
    print("--- Offline (oracle: all data visible) ---")
    print(f"  Metric: log1p(score_raw); threshold = np.percentile(..., {args.percentile}) = {T_global:.10g}")
    print(f"  Flagged (x >= T): {offline_flagged} ({100.0 * offline_flagged / n:.4f}%)")
    print()
    print("--- Online (prequential, detector logic) ---")
    print(f"  window_size={args.window_size}, warmup={args.warmup}")
    print(f"  Flag if online_pct >= {online_t:.6g}")
    print(f"  Flagged: {online_flagged} ({100.0 * online_flagged / n:.4f}%)")


if __name__ == "__main__":
    main()
