"""Tests for detector/server.py."""
import json
import os

import pytest

import events_pb2
from detector.config import DetectorConfig
from detector.server import RuleBasedDetector, _init_event_dump, _now_timestamp


class TestDetector:
  """Test detector service."""

  def test_now_timestamp(self):
    ts = _now_timestamp()
    assert ts is not None
    assert ts.seconds > 0

  def test_detector_init(self):
    cfg = DetectorConfig(port=50051)
    detector = RuleBasedDetector(cfg)
    assert detector.cfg == cfg

  @pytest.mark.asyncio
  async def test_score_event_online(self):
    """Test online scoring behavior."""
    cfg = DetectorConfig(model_algorithm="halfspacetrees", hst_n_trees=5, hst_height=5, hst_window_size=10)
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      event_name="openat",
      event_type="",
      ts_unix_nano=1234567890000000000,
      data=["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-123"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_stream_events(self):
    """Test StreamEvents method with online scoring."""
    cfg = DetectorConfig(model_algorithm="halfspacetrees", hst_n_trees=5, hst_height=5, hst_window_size=10)
    detector = RuleBasedDetector(cfg)

    async def event_gen():
      for i in range(3):
        yield events_pb2.EventEnvelope(
          event_id=f"test-{i}",
          event_name="openat",
          event_type="",
          ts_unix_nano=1234567890000000000 + i,
          data=["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
        )

    responses = []
    async for resp in detector.StreamEvents(event_gen(), None):
      responses.append(resp)

    assert len(responses) == 3
    for i, resp in enumerate(responses):
      assert resp.event_id == f"test-{i}"
      assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_loda(self):
    """Test LODA scoring path."""
    cfg = DetectorConfig(
      model_algorithm="loda",
      loda_n_projections=5,
      loda_bins=8,
      loda_range=2.0,
      loda_ema_alpha=0.5,
      loda_hist_decay=1.0,
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-456",
      event_name="socket",
      event_type="",
      ts_unix_nano=1234567890000000000,
      data=["socket", "5", "bash", "1", "2", "1000", "1", "2", "", "2"],
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-456"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_memstream(self):
    """Test MemStream scoring path."""
    cfg = DetectorConfig(
      model_algorithm="memstream",
      mem_hidden_dim=8,
      mem_latent_dim=4,
      mem_memory_size=4,
      mem_lr=0.01,
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-789",
      event_name="execve",
      event_type="",
      ts_unix_nano=1234567890000000000,
      data=["execve", "4", "bash", "1", "2", "1000", "0", "0", "/bin/bash", ""],
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-789"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_report_anomaly(self):
    """Test ReportAnomaly method."""
    cfg = DetectorConfig()
    detector = RuleBasedDetector(cfg)

    report = events_pb2.AnomalyReport(
      event_id="test-123",
      reason="suspicious activity",
      score=0.85,
      labels={"severity": "high"},
    )

    resp = await detector.ReportAnomaly(report, None)
    assert resp is not None

  @pytest.mark.asyncio
  async def test_event_dump(self, tmp_path):
    """When EVENT_DUMP_PATH is set, metadata is written first, then every event is appended as JSONL."""
    dump_file = tmp_path / "events.jsonl"
    prev = os.environ.pop("EVENT_DUMP_PATH", None)
    try:
      os.environ["EVENT_DUMP_PATH"] = str(dump_file)
      cfg = DetectorConfig(model_algorithm="halfspacetrees", hst_n_trees=5, hst_height=5, hst_window_size=10)
      _init_event_dump(cfg)
      from collections import deque
      import detector.server as srv
      srv.RECENT_EVENTS = deque(maxlen=100)
      detector = RuleBasedDetector(cfg)

      async def one_event():
        yield events_pb2.EventEnvelope(
          event_id="dump-me",
          event_name="openat",
          event_type="",
          ts_unix_nano=1234567890000000000,
          data=["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
        )

      async for _ in detector.StreamEvents(one_event(), None):
        pass

      assert dump_file.exists()
      lines = dump_file.read_text().strip().split("\n")
      assert len(lines) == 2  # metadata + 1 event
      meta = json.loads(lines[0])
      assert meta.get("_meta") is True
      assert meta.get("config", {}).get("model_algorithm") == "halfspacetrees"
      assert "date" in meta
      assert "config" in meta
      row = json.loads(lines[1])
      assert row["event_id"] == "dump-me"
      assert row["event_name"] == "openat"
      assert "anomaly" in row
      assert "score" in row
      assert "container_id" in row
      assert "attributes" in row
    finally:
      if prev is not None:
        os.environ["EVENT_DUMP_PATH"] = prev
      else:
        os.environ.pop("EVENT_DUMP_PATH", None)
