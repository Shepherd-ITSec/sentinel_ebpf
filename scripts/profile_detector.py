#!/usr/bin/env python3
"""Profile detector scoring. Run: uv run python scripts/profile_detector.py [n_events]"""
import cProfile
import os
import pstats
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DETECTOR_MODEL_ALGORITHM"] = "freq1d"
os.environ["DETECTOR_FREQ1D_BINS"] = "65536"
os.environ["DETECTOR_FREQ1D_MAX_CATEGORIES"] = "65536"

import events_pb2
from detector.config import load_config
from detector.server import DeterministicScorer


def make_event(i: int, event_group: str = "file") -> events_pb2.EventEnvelope:
    ts = 1700000000_000_000_000 + i * 1_000_000
    return events_pb2.EventEnvelope(
        event_id=f"evt-{i}",
        syscall_name="openat",
        event_group=event_group,
        hostname="node1",
        ts_unix_nano=ts,
        syscall_nr=i % 1000,
        comm="bash" if i % 3 == 0 else "nginx" if i % 3 == 1 else "python",
        pid=str(1000 + i % 100),
        tid=str(2000 + i % 100),
        uid="1000",
        arg0=str(i % 100),
        arg1="2",
        path=f"/etc/config/file_{i % 50}.conf",
        attributes={"flags": "O_RDONLY"},
    )


def run_profile(n_events: int = 500):
    cfg = load_config()
    scorer = DeterministicScorer(cfg)
    events = [make_event(i) for i in range(n_events)]

    def score_loop():
        for evt in events:
            scorer.score_event(evt)

    prof = cProfile.Profile()
    prof.enable()
    score_loop()
    prof.disable()

    s = StringIO()
    pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Profiling {n} events with Freq1D...")
    run_profile(n)
