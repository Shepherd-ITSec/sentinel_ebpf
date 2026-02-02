"""Tests for EventEnvelope creation and serialization."""
import json

import pytest

import events_pb2


class TestEventEnvelope:
  """Test EventEnvelope proto message."""

  def test_create_minimal_event(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
    )
    assert evt.event_id == "test-123"
    assert evt.event_type == "file_open"
    assert evt.ts_unix_nano == 1234567890000000000
    assert len(evt.data) == 0

  def test_create_event_with_data_vector(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
    )
    assert len(evt.data) == 6
    assert evt.data[0] == "/tmp/test.txt"
    assert evt.data[1] == "2"
    assert evt.data[2] == "bash"
    assert evt.data[3] == "1234"
    assert evt.data[4] == "5678"
    assert evt.data[5] == "1000"

  def test_create_event_with_metadata(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      hostname="node-1",
      pod_name="test-pod",
      namespace="default",
      container_id="container-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/etc/hosts", "2", "cat", "5678", "5678", "0"],
    )
    assert evt.hostname == "node-1"
    assert evt.pod_name == "test-pod"
    assert evt.namespace == "default"
    assert evt.container_id == "container-123"
    assert evt.event_type == "file_open"

  def test_create_event_with_attributes(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test", "2", "bash", "1", "2", "1000"],
      attributes={"env": "prod", "team": "security"},
    )
    assert evt.attributes["env"] == "prod"
    assert evt.attributes["team"] == "security"

  def test_serialize_to_json(self):
    """Test that EventEnvelope can be serialized for file logging."""
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      hostname="node-1",
      pod_name="test-pod",
      namespace="default",
      container_id="container-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"],
    )

    # Simulate FileSink serialization
    payload = {
      "event_id": evt.event_id,
      "ts_unix_nano": evt.ts_unix_nano,
      "hostname": evt.hostname,
      "pod": evt.pod_name,
      "namespace": evt.namespace,
      "container_id": evt.container_id,
      "event_type": evt.event_type,
      "data": list(evt.data),
      "attributes": dict(evt.attributes),
    }

    json_str = json.dumps(payload, separators=(",", ":"))
    parsed = json.loads(json_str)

    assert parsed["event_id"] == "test-123"
    assert parsed["event_type"] == "file_open"
    assert parsed["data"] == ["/tmp/test.txt", "2", "bash", "1234", "5678", "1000"]

  def test_file_open_event_format(self):
    """Test file_open event format (ordered vector)."""
    evt = events_pb2.EventEnvelope(
      event_id="open-123",
      event_type="file_open",
      ts_unix_nano=1234567890000000000,
      data=["/etc/passwd", "2", "cat", "9999", "9999", "1000"],
    )

    # Verify order: [filename, flags, comm, pid, tid, uid]
    assert evt.data[0] == "/etc/passwd"  # filename
    assert evt.data[1] == "2"  # flags
    assert evt.data[2] == "cat"  # comm
    assert evt.data[3] == "9999"  # pid
    assert evt.data[4] == "9999"  # tid
    assert evt.data[5] == "1000"  # uid
