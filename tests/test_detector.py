"""Tests for detector/server.py."""
import json
import os
import time

import pytest
from google.protobuf.empty_pb2 import Empty

import events_pb2
import detector.server as server_mod
from detector.config import DetectorConfig
from detector.server import RuleBasedDetector, _init_event_dump, _now_timestamp


class TestDetector:
  """Test detector service."""

  def test_now_timestamp(self):
    before = time.time()
    ts = _now_timestamp()
    after = time.time()
    assert ts.seconds > 0
    assert 0 <= ts.nanos < 1_000_000_000
    ts_float = ts.seconds + (ts.nanos / 1_000_000_000)
    assert before <= ts_float <= after

  def test_detector_init(self):
    cfg = DetectorConfig(port=50051)
    detector = RuleBasedDetector(cfg)
    assert detector.cfg == cfg

  @pytest.mark.asyncio
  async def test_score_event_online(self):
    """Test online scoring behavior."""
    cfg = DetectorConfig(
      model_algorithm="halfspacetrees",
      hst_n_trees=5,
      hst_height=5,
      hst_window_size=10,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      syscall_name="openat",
      event_group="",
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

    resp = detector._score_event(evt)
    assert resp.event_id == "test-123"
    assert 0.0 <= resp.score <= 1.0
    assert hasattr(resp, "score_raw") and resp.score_raw >= 0

  @pytest.mark.asyncio
  async def test_stream_events(self):
    """Test StreamEvents method with online scoring."""
    cfg = DetectorConfig(
      model_algorithm="halfspacetrees",
      hst_n_trees=5,
      hst_height=5,
      hst_window_size=10,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    async def event_gen():
      for i in range(3):
        yield events_pb2.EventEnvelope(
          event_id=f"test-{i}",
          syscall_name="openat",
          event_group="",
          ts_unix_nano=1234567890000000000 + i,
          syscall_nr=2,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/tmp/test"},
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
      model_algorithm="loda_ema",
      loda_n_projections=5,
      loda_bins=8,
      loda_range=2.0,
      loda_ema_alpha=0.5,
      loda_hist_decay=1.0,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-456",
      syscall_name="socket",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=5,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="1",
      arg1="2",
      attributes={},
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-456"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_memstream(self):
    """Test MemStream scoring path."""
    cfg = DetectorConfig(
      model_algorithm="memstream",
      mem_memory_size=4,
      mem_lr=0.01,
      mem_beta=0.1,
      mem_k=3,
      mem_gamma=0.5,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-789",
      syscall_name="execve",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=4,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="0",
      arg1="0",
      attributes={"fd_path": "/bin/bash"},
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-789"
    assert 0.0 <= resp.score <= 1.0

  def test_score_event_uses_model_specific_feature_view(self, monkeypatch):
    captured = {}

    class FakeExtractor:
      def extract_feature_dict(self, evt, feature_view="default"):
        captured["feature_view"] = feature_view
        return {"f0": 0.0, "f1": 1.0}

    monkeypatch.setattr(server_mod, "build_feature_extractor", lambda cfg: FakeExtractor())
    cfg = DetectorConfig(model_algorithm="memstream", score_mode="scaled")
    detector = RuleBasedDetector(cfg)
    evt = events_pb2.EventEnvelope(
      event_id="view-test",
      syscall_name="execve",
      event_group="process",
      ts_unix_nano=1234567890000000000,
      syscall_nr=59,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="0",
      arg1="0",
      attributes={"fd_path": "/bin/bash"},
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "view-test"
    assert captured["feature_view"] == "memstream"

  @pytest.mark.parametrize("model_algorithm", ("freq1d",))
  def test_freq_models_use_frequency_feature_view(self, monkeypatch, model_algorithm):
    captured = {}

    class FakeExtractor:
      def extract_feature_dict(self, evt, feature_view="default"):
        captured["feature_view"] = feature_view
        return {"f0": 0.0, "f1": 1.0}

    monkeypatch.setattr(server_mod, "build_feature_extractor", lambda cfg: FakeExtractor())
    cfg = DetectorConfig(model_algorithm=model_algorithm, score_mode="scaled")
    detector = RuleBasedDetector(cfg)
    evt = events_pb2.EventEnvelope(
      event_id=f"{model_algorithm}-view-test",
      syscall_name="openat",
      event_group="file",
      ts_unix_nano=1234567890000000000,
      syscall_nr=257,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/tmp/test"},
    )

    resp = detector._score_event(evt)
    assert resp.event_id == f"{model_algorithm}-view-test"
    assert captured["feature_view"] == "frequency"

  @pytest.mark.asyncio
  async def test_score_event_zscore(self):
    """Test ZScore scoring path."""
    cfg = DetectorConfig(
      model_algorithm="zscore",
      zscore_min_count=5,
      zscore_std_floor=1e-3,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-zscore",
      syscall_name="execve",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=4,
      comm="bash",
      pid="1",
      tid="2",
      uid="1000",
      arg0="0",
      arg1="0",
      attributes={"fd_path": "/bin/bash"},
    )

    resp = detector._score_event(evt)
    assert resp.event_id == "test-zscore"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_freq1d(self):
    """Test Freq1D scoring path."""
    cfg = DetectorConfig(
      model_algorithm="freq1d",
      freq1d_bins=32,
      freq1d_alpha=1.0,
      freq1d_decay=1.0,
      freq1d_max_categories=128,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-freq1d",
      syscall_name="openat",
      event_group="",
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

    resp = detector._score_event(evt)
    assert resp.event_id == "test-freq1d"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_latentcluster(self):
    """Test latent-clustering scoring path."""
    cfg = DetectorConfig(
      model_algorithm="latentcluster",
      freq1d_bins=32,
      freq1d_alpha=1.0,
      freq1d_decay=1.0,
      freq1d_max_categories=128,
      latentcluster_max_clusters=4,
      latentcluster_spawn_threshold=4.0,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-latentcluster",
      syscall_name="openat",
      event_group="",
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

    resp = detector._score_event(evt)
    assert resp.event_id == "test-latentcluster"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_copulatree(self):
    """Test CopulaTree scoring path."""
    cfg = DetectorConfig(
      model_algorithm="copulatree",
      freq1d_bins=32,
      freq1d_alpha=1.0,
      freq1d_decay=1.0,
      freq1d_max_categories=128,
      copulatree_tree_update_interval=5,
      copulatree_importance_window=10,
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-copulatree",
      syscall_name="openat",
      event_group="",
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

    resp = detector._score_event(evt)
    assert resp.event_id == "test-copulatree"
    assert 0.0 <= resp.score <= 1.0

  @pytest.mark.asyncio
  async def test_score_event_knn(self):
    """Test KNN scoring path."""
    cfg = DetectorConfig(
      model_algorithm="knn",
      knn_k=3,
      knn_memory_size=64,
      knn_metric="euclidean",
      score_mode="scaled",
    )
    detector = RuleBasedDetector(cfg)

    evt = events_pb2.EventEnvelope(
      event_id="test-knn",
      syscall_name="openat",
      event_group="",
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

    resp = detector._score_event(evt)
    assert resp.event_id == "test-knn"
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
    assert isinstance(resp, Empty)

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
          syscall_name="openat",
          event_group="",
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
      assert row["syscall_name"] == "openat"
      assert "anomaly" in row
      assert "score" in row
      assert "score_raw" in row
      assert "container_id" in row
      assert "attributes" in row
    finally:
      if prev is not None:
        os.environ["EVENT_DUMP_PATH"] = prev
      else:
        os.environ.pop("EVENT_DUMP_PATH", None)
