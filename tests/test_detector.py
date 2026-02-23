"""Tests for detector/server.py."""
import pytest

import events_pb2
from detector.config import DetectorConfig
from detector.server import RuleBasedDetector, _now_timestamp


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
    assert detector.active_worker_count == 1

  def test_detector_forces_deterministic_single_worker(self):
    cfg = DetectorConfig(worker_count=8)
    detector = RuleBasedDetector(cfg)
    assert detector.active_worker_count == 1
    assert detector.configured_worker_count == 8

  @pytest.mark.asyncio
  async def test_score_event_online(self):
    """Test online scoring behavior."""
    cfg = DetectorConfig(model_algorithm="halfspacetrees", hst_n_trees=5, hst_height=5, hst_window_size=10)
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test", "2", "bash", "1", "2", "1000"],
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
          event_type="file_open",
          ts_unix_nano=1234567890000000000 + i,
          data=["/tmp/test", "2", "bash", "1", "2", "1000"],
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
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test", "2", "bash", "1", "2", "1000"],
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
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test", "2", "bash", "1", "2", "1000"],
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
