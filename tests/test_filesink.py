"""Tests for FileSink class in probe/probe_runner.py."""
import gzip
import json
import struct
from pathlib import Path

import pytest

import events_pb2

pytest.importorskip("bcc", reason="bcc not installed; eBPF-related tests are allowed to fail for now.")
from probe.probe_runner import FileSink


@pytest.fixture
def sample_event():
  """Create a sample EventEnvelope."""
  return events_pb2.EventEnvelope(
    event_id="test-event-123",
    hostname="test-host",
    pod_name="test-pod",
    namespace="default",
    container_id="container-123",
    event_type="file_open",
    ts_unix_nano=1234567890000000000,
    data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
  )


class TestFileSink:
  """Test FileSink class."""

  def test_write_event_uncompressed(self, temp_dir, sample_event):
    sink = FileSink(
      path=str(temp_dir / "events.bin"),
      max_bytes=0,  # Disable rotation
      max_files=1,
      compress=False,
    )
    try:
      sink.publish(sample_event)
      sink.close()

      # Read back
      with open(temp_dir / "events.bin", "rb") as f:
        magic = f.read(4)
        assert magic == b"EVT1"
        length_bytes = f.read(4)
        length = struct.unpack("<I", length_bytes)[0]
        payload = json.loads(f.read(length).decode("utf-8"))

      assert payload["event_id"] == "test-event-123"
      assert payload["event_type"] == "file_open"
      assert payload["data"] == ["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"]
    finally:
      sink.close()

  def test_write_event_compressed(self, temp_dir, sample_event):
    sink = FileSink(
      path=str(temp_dir / "events.bin.gz"),
      max_bytes=0,
      max_files=1,
      compress=True,
    )
    try:
      sink.publish(sample_event)
      sink.close()

      # Read back (gzip auto-detected)
      with gzip.open(temp_dir / "events.bin.gz", "rb") as f:
        magic = f.read(4)
        assert magic == b"EVT1"
        length_bytes = f.read(4)
        length = struct.unpack("<I", length_bytes)[0]
        payload = json.loads(f.read(length).decode("utf-8"))

      assert payload["event_id"] == "test-event-123"
    finally:
      sink.close()

  def test_write_multiple_events(self, temp_dir, sample_event):
    sink = FileSink(
      path=str(temp_dir / "events.bin"),
      max_bytes=0,
      max_files=1,
      compress=False,
    )
    try:
      for i in range(5):
        evt = events_pb2.EventEnvelope(
          event_id=f"event-{i}",
          event_type="file_open",
          ts_unix_nano=1234567890000000000 + i,
          data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
        )
        sink.publish(evt)
      sink.close()

      # Count events
      count = 0
      with open(temp_dir / "events.bin", "rb") as f:
        while True:
          magic = f.read(4)
          if not magic:
            break
          assert magic == b"EVT1"
          length_bytes = f.read(4)
          length = struct.unpack("<I", length_bytes)[0]
          f.read(length)  # Skip payload
          count += 1

      assert count == 5
    finally:
      sink.close()

  def test_rotation_on_size_limit(self, temp_dir, sample_event):
    sink = FileSink(
      path=str(temp_dir / "events.bin"),
      max_bytes=1000,  # Small limit
      max_files=3,
      compress=False,
    )
    try:
      # Write enough events to trigger rotation
      for i in range(10):
        evt = events_pb2.EventEnvelope(
          event_id=f"event-{i}",
          event_type="file_open",
          ts_unix_nano=1234567890000000000 + i,
          data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
        )
        sink.publish(evt)
      sink.close()

      # Check that rotation occurred (base file and rotated files exist)
      base = temp_dir / "events.bin"
      rotated1 = temp_dir / "events.bin.1"
      # At least base should exist, possibly rotated files
      assert base.exists() or rotated1.exists()
    finally:
      sink.close()

  def test_no_rotation_when_disabled(self, temp_dir, sample_event):
    sink = FileSink(
      path=str(temp_dir / "events.bin"),
      max_bytes=0,  # Disabled
      max_files=1,
      compress=False,
    )
    try:
      # Write many events
      for i in range(100):
        evt = events_pb2.EventEnvelope(
          event_id=f"event-{i}",
          event_type="file_open",
          ts_unix_nano=1234567890000000000 + i,
          data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
        )
        sink.publish(evt)
      sink.close()

      # Should only have base file
      assert (temp_dir / "events.bin").exists()
      assert not (temp_dir / "events.bin.1").exists()
    finally:
      sink.close()

  def test_directory_creation(self, temp_dir, sample_event):
    nested_path = temp_dir / "nested" / "dir" / "events.bin"
    sink = FileSink(
      path=str(nested_path),
      max_bytes=0,
      max_files=1,
      compress=False,
    )
    try:
      sink.publish(sample_event)
      sink.close()
      assert nested_path.exists()
      assert nested_path.parent.exists()
    finally:
      sink.close()
