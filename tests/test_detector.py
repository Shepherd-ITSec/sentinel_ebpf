"""Pipeline-only detector server tests."""

import time

import events_pb2
import pytest

from detector.config import DetectorConfig
from detector.server import RuleBasedDetector, _now_timestamp


def _evt(event_id: str, syscall_name: str = "openat") -> events_pb2.EventEnvelope:
  return events_pb2.EventEnvelope(
    event_id=event_id,
    syscall_name=syscall_name,
    event_group="process",
    ts_unix_nano=1234567890000000000,
    syscall_nr=2,
    comm="bash",
    pid="1",
    tid="2",
    uid="1000",
    arg0="-100",
    arg1="2",
    attributes={"fd_path": "/tmp/test"},
  )


def test_now_timestamp() -> None:
  before = time.time()
  ts = _now_timestamp()
  after = time.time()
  assert ts.seconds > 0
  assert 0 <= ts.nanos < 1_000_000_000
  ts_float = ts.seconds + (ts.nanos / 1_000_000_000)
  assert before <= ts_float <= after


def test_detector_score_event_delegates_to_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
  detector = RuleBasedDetector(DetectorConfig())
  seen = []
  expected = events_pb2.DetectionResponse(
    event_id="test-1",
    anomaly=True,
    reason="from fake scorer",
    score=0.75,
    score_raw=2.5,
  )

  def _fake_score_event(evt):
    seen.append(evt.event_id)
    return expected

  monkeypatch.setattr(detector.scorer, "score_event", _fake_score_event)
  resp = detector._score_event(_evt("test-1"))
  assert seen == ["test-1"]
  assert resp == expected
