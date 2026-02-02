"""Tests for scripts/decode_logs.py."""
import gzip
import json
import struct
from io import StringIO
from pathlib import Path

import pytest

from scripts.decode_logs import MAGIC, decode, open_stream


class TestDecodeLogs:
  """Test decode_logs.py functions."""

  def test_open_stream_plain(self, temp_dir):
    test_file = temp_dir / "test.bin"
    test_file.write_bytes(b"test data")
    with open_stream(test_file) as f:
      assert f.read() == b"test data"

  def test_open_stream_gzip(self, temp_dir):
    test_file = temp_dir / "test.bin.gz"
    data = b"test data"
    with gzip.open(test_file, "wb") as f:
      f.write(data)
    with open_stream(test_file) as f:
      assert isinstance(f, gzip.GzipFile)
      assert f.read() == data

  def test_decode_single_event(self, temp_dir):
    log_file = temp_dir / "events.bin"
    payload = json.dumps({"event_id": "test-1", "event_type": "file_open", "data": ["/tmp/test", "2", "bash", "1", "2", "1000"]}).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload
    log_file.write_bytes(record)

    out = StringIO()
    decode(log_file, out)
    result = json.loads(out.getvalue().strip())

    assert result["event_id"] == "test-1"
    assert result["event_type"] == "file_open"
    assert result["data"] == ["/tmp/test", "2", "bash", "1", "2", "1000"]

  def test_decode_multiple_events(self, temp_dir):
    log_file = temp_dir / "events.bin"
    records = []
    for i in range(3):
      payload = json.dumps({"event_id": f"test-{i}", "event_type": "file_open", "data": ["/tmp/test", "2", "bash", "1", "2", "1000"]}).encode("utf-8")
      records.append(MAGIC + struct.pack("<I", len(payload)) + payload)
    log_file.write_bytes(b"".join(records))

    out = StringIO()
    decode(log_file, out)
    lines = out.getvalue().strip().split("\n")

    assert len(lines) == 3
    for i, line in enumerate(lines):
      result = json.loads(line)
      assert result["event_id"] == f"test-{i}"

  def test_decode_gzipped_file(self, temp_dir):
    log_file = temp_dir / "events.bin.gz"
    payload = json.dumps({"event_id": "test-1", "event_type": "file_open", "data": ["/tmp/test", "2", "bash", "1", "2", "1000"]}).encode("utf-8")
    record = MAGIC + struct.pack("<I", len(payload)) + payload

    with gzip.open(log_file, "wb") as f:
      f.write(record)

    out = StringIO()
    decode(log_file, out)
    result = json.loads(out.getvalue().strip())

    assert result["event_id"] == "test-1"

  def test_decode_invalid_magic(self, temp_dir):
    log_file = temp_dir / "events.bin"
    log_file.write_bytes(b"INVALID" + struct.pack("<I", 10) + b"x" * 10)

    out = StringIO()
    with pytest.raises(ValueError, match="bad magic"):
      decode(log_file, out)

  def test_decode_truncated_file(self, temp_dir):
    log_file = temp_dir / "events.bin"
    # Write magic but incomplete length
    log_file.write_bytes(MAGIC + b"\x01")

    out = StringIO()
    decode(log_file, out)  # Should not raise, just stop
    assert out.getvalue() == ""

  def test_decode_truncated_payload(self, temp_dir):
    log_file = temp_dir / "events.bin"
    # Write magic and length but incomplete payload
    log_file.write_bytes(MAGIC + struct.pack("<I", 100) + b"x" * 50)

    out = StringIO()
    decode(log_file, out)  # Should not raise, just stop
    assert out.getvalue() == ""
