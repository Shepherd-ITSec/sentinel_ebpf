"""Tests for EventEnvelope creation and serialization."""
import json

import events_pb2
from event_envelope_codec import envelope_to_dict


class TestEventEnvelope:
  """Test EventEnvelope proto message."""

  def test_create_minimal_event(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      syscall_name="openat",
      event_group="",
      ts_unix_nano=1234567890000000000,
    )
    assert evt.event_id == "test-123"
    assert evt.syscall_name == "openat"
    assert evt.event_group == ""
    assert evt.ts_unix_nano == 1234567890000000000
    assert evt.syscall_nr == 0
    assert evt.comm == ""

  def test_create_event_with_syscall_fields(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      syscall_name="openat",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=2,
      comm="bash",
      pid="1234",
      tid="5678",
      uid="1000",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/tmp/test.txt"},
    )
    assert evt.syscall_nr == 2
    assert evt.comm == "bash"
    assert evt.pid == "1234"
    assert evt.tid == "5678"
    assert evt.uid == "1000"

  def test_create_event_with_metadata(self):
    evt = events_pb2.EventEnvelope(
      event_id="test-123",
      hostname="node-1",
      pod_name="test-pod",
      namespace="default",
      container_id="container-123",
      syscall_name="openat",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=2,
      comm="cat",
      pid="5678",
      tid="5678",
      uid="0",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/etc/hosts"},
    )
    assert evt.hostname == "node-1"
    assert evt.pod_name == "test-pod"
    assert evt.namespace == "default"
    assert evt.container_id == "container-123"
    assert evt.syscall_name == "openat"

  def test_create_event_with_attributes(self):
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
      attributes={"env": "prod", "team": "security", "fd_path": "/tmp/test"},
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
      syscall_name="openat",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=2,
      comm="bash",
      pid="1234",
      tid="5678",
      uid="1000",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/tmp/test.txt"},
    )

    payload = envelope_to_dict(evt)
    json_str = json.dumps(payload, separators=(",", ":"))
    parsed = json.loads(json_str)

    assert parsed["event_id"] == "test-123"
    assert parsed["syscall_name"] == "openat"
    assert parsed["event_group"] == ""
    assert parsed["syscall_nr"] == 2
    assert parsed["attributes"]["fd_path"] == "/tmp/test.txt"

  def test_generic_event_format(self):
    """Syscall fields match probe layout."""
    evt = events_pb2.EventEnvelope(
      event_id="open-123",
      syscall_name="openat",
      event_group="",
      ts_unix_nano=1234567890000000000,
      syscall_nr=2,
      comm="cat",
      pid="9999",
      tid="9999",
      uid="1000",
      arg0="-100",
      arg1="2",
      attributes={"fd_path": "/etc/passwd"},
    )

    assert evt.comm == "cat"
    assert evt.pid == "9999"
    assert evt.tid == "9999"
    assert evt.uid == "1000"
