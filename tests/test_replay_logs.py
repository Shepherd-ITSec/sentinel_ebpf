"""Tests for scripts/replay_logs.py."""
import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.replay_logs import MAGIC, iter_events, open_stream


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
      "event_type": "file_open",
      "ts_unix_nano": 1234567890000000000,
      "data": ["/tmp/test", "2", "bash", "1", "2", "1000"],
    }).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload
    log_file.write_bytes(record)

    events = list(iter_events(log_file))
    assert len(events) == 1
    assert events[0]["event_id"] == "test-1"
    assert events[0]["data"] == ["/tmp/test", "2", "bash", "1", "2", "1000"]

  def test_iter_events_multiple(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    for i in range(3):
      payload = json.dumps({
        "event_id": f"test-{i}",
        "event_type": "file_open",
        "ts_unix_nano": 1234567890000000000 + i * 1000000,
        "data": ["/tmp/test", "2", "bash", "1", "2", "1000"],
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
        "event_type": "file_open",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,  # Convert to ns
        "data": ["/tmp/test", "2", "bash", "1", "2", "1000"],
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
        "event_type": "file_open",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,
        "data": ["/tmp/test", "2", "bash", "1", "2", "1000"],
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
        "event_type": "file_open",
        "ts_unix_nano": (base_ts + i * 1000) * 1_000_000,
        "data": ["/tmp/test", "2", "bash", "1", "2", "1000"],
      }).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    # Filter: events 1-3
    events = list(iter_events(log_file, start_ms=base_ts + 1000, end_ms=base_ts + 3500))
    assert len(events) == 3  # Events 1, 2, 3

  def test_iter_events_backward_compat_map_format(self, temp_dir):
    """Test backward compatibility with old map format."""
    log_file = temp_dir / "events.bin"
    # Old format: data as map
    payload = json.dumps({
      "event_id": "test-1",
      "event_type": "file_open",
      "ts_unix_nano": 1234567890000000000,
      "data": {"filename": "/tmp/test", "bytes": "1024", "comm": "bash", "pid": "1", "tid": "2"},
    }).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload
    log_file.write_bytes(record)

    events = list(iter_events(log_file))
    assert len(events) == 1
    assert isinstance(events[0]["data"], dict)
    assert events[0]["data"]["filename"] == "/tmp/test"
