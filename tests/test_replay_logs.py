"""Tests for scripts/replay_logs.py."""
import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.replay_logs import MAGIC, _detect_format, iter_events, iter_events_jsonl, open_stream


class TestReplayLogs:
  """Test replay_logs.py functions."""

  def test_open_stream_plain(self, temp_dir):
    test_file = temp_dir / "test.bin"
    test_file.write_bytes(b"test data")
    with open_stream(test_file) as f:
      assert f.read() == b"test data"

  def test_open_stream_gzip(self, temp_dir):
    import gzip
    test_file = temp_dir / "test.bin.gz"
    data = b"test data"
    with gzip.open(test_file, "wb") as f:
      f.write(data)
    with open_stream(test_file) as f:
      assert isinstance(f, gzip.GzipFile)
      assert f.read() == data

  def test_iter_events_single(self, temp_dir):
    log_file = temp_dir / "events.bin"
    payload = json.dumps({
      "event_id": "test-1",
      "event_name": "openat",
      "event_type": "",
      "ts_unix_nano": 1234567890000000000,
      "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
    }).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload
    log_file.write_bytes(record)

    events = list(iter_events(log_file))
    assert len(events) == 1
    assert events[0]["event_id"] == "test-1"
    assert events[0]["data"] == ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"]

  def test_iter_events_multiple(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    for i in range(3):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_name": "openat",
        "event_type": "",
        "ts_unix_nano": 1234567890000000000 + i * 1000000,
        "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    events = list(iter_events(log_file))
    assert len(events) == 3
    for i, evt in enumerate(events):
      assert evt["event_id"] == f"test-{i}"

  def test_iter_events_time_filter_start(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    base_ts = 1234567890000  # ms
    for i in range(5):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_name": "openat",
        "event_type": "",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,  # Convert to ns
        "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    # Filter: start from event 2
    events = list(iter_events(log_file, start_ms=base_ts + 2000))
    assert len(events) == 3  # Events 2, 3, 4

  def test_iter_events_time_filter_end(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    base_ts = 1234567890000  # ms
    for i in range(5):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_name": "openat",
        "event_type": "",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,
        "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    # Filter: end at event 2
    events = list(iter_events(log_file, end_ms=base_ts + 2500))
    assert len(events) == 3  # Events 0, 1, 2

  def test_iter_events_time_filter_range(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    base_ts = 1234567890000  # ms
    for i in range(5):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_name": "openat",
        "event_type": "",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,
        "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    # Filter: events 1-3
    events = list(iter_events(log_file, start_ms=base_ts + 1000, end_ms=base_ts + 3500))
    assert len(events) == 3  # Events 1, 2, 3

  def test_iter_events_max_events(self, temp_dir):
    """max_events stops after N events (for warmup replay)."""
    log_file = temp_dir / "events.bin"
    records = []
    for i in range(5):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_name": "openat",
        "event_type": "",
        "ts_unix_nano": 1234567890000000000 + i * 1000000,
        "data": ["openat", "2", "bash", "1", "2", "1000", "-100", "2", "/tmp/test", "2"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    events = list(iter_events(log_file, max_events=2))
    assert len(events) == 2
    assert events[0]["event_id"] == "test-0"
    assert events[1]["event_id"] == "test-1"

  def test_detect_format_evt1(self, temp_dir):
    log_file = temp_dir / "events.bin"
    log_file.write_bytes(MAGIC + b"\x00\x00\x00\x00")
    assert _detect_format(log_file) == "evt1"

  def test_detect_format_jsonl(self, temp_dir):
    jsonl_file = temp_dir / "events.jsonl"
    jsonl_file.write_text('{"event_id": "x", "data": []}\n')
    assert _detect_format(jsonl_file) == "jsonl"

  def test_iter_events_jsonl_lossless(self, temp_dir):
    jsonl_file = temp_dir / "detector-events.jsonl"
    jsonl_file.write_text(
      '{"event_id": "a", "event_name": "openat", "event_type": "file", "data": ["openat", "2", "bash", "1", "2", "1000", "0", "0", "/tmp/x", ""], '
      '"hostname": "h", "pod_name": "p", "namespace": "ns", "container_id": "cid", "attributes": {"return_value": "0"}, "ts_unix_nano": 1234567890000000000}\n'
    )
    events = list(iter_events_jsonl(jsonl_file))
    assert len(events) == 1
    obj = events[0]
    assert obj["event_id"] == "a"
    assert obj["pod_name"] == "p"
    assert obj["container_id"] == "cid"
    assert obj["attributes"] == {"return_value": "0"}

  def test_iter_events_jsonl_skips_metadata_line(self, temp_dir):
    """iter_events_jsonl skips _meta lines (no event_id) written by detector event dump."""
    jsonl_file = temp_dir / "events.jsonl"
    meta = json.dumps({"_meta": True, "date": "2026-03-09T12:00:00Z", "config": {}}) + "\n"
    evt = json.dumps({"event_id": "evt-1", "ts_unix_nano": 1234567890000000000, "data": ["x"]}) + "\n"
    jsonl_file.write_text(meta + evt)
    events = list(iter_events_jsonl(jsonl_file))
    assert len(events) == 1
    assert events[0]["event_id"] == "evt-1"

  def test_iter_events_jsonl_skip_second_half(self, temp_dir):
    """skip yields second half of events (for compare_replay_scores)."""
    jsonl_file = temp_dir / "events.jsonl"
    lines = []
    for i in range(6):
      lines.append(json.dumps({"event_id": f"evt-{i}", "ts_unix_nano": 1000000000 + i, "data": ["x"]}) + "\n")
    jsonl_file.write_text("".join(lines))
    events = list(iter_events_jsonl(jsonl_file, skip=3, max_events=3))
    assert len(events) == 3
    assert [e["event_id"] for e in events] == ["evt-3", "evt-4", "evt-5"]

  def test_iter_events_non_list_data_rejected_by_replay(self, temp_dir):
    """Replay should reject non-canonical non-list data vectors."""
    log_file = temp_dir / "events.bin"
    payload = json.dumps({
      "event_id": "test-1",
      "event_name": "openat",
      "event_type": "",
      "ts_unix_nano": 1234567890000000000,
      "data": {"filename": "/tmp/test", "bytes": "1024", "comm": "bash", "pid": "1", "tid": "2"},
    }).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload
    log_file.write_bytes(record)

    with patch("scripts.replay_logs.grpc.insecure_channel") as mock_channel:
      mock_stub = MagicMock()
      mock_channel.return_value = MagicMock()
      with patch("scripts.replay_logs.events_pb2_grpc.DetectorServiceStub", return_value=mock_stub):
        mock_stub.StreamEvents.side_effect = ValueError("invalid event record: 'data' must be an ordered list")
        with pytest.raises(ValueError, match="invalid event record"):
          from scripts.replay_logs import replay
          replay(str(log_file), "localhost:50051", "fast", None, None)
