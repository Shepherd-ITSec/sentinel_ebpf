"""Tests for detector/features.py."""
import numpy as np

import events_pb2
from detector.features import (
  extract_feature_dict,
)


def test_extract_feature_dict_returns_expected_shape():
  evt = events_pb2.EventEnvelope(
    event_id="feat-1",
    event_name="openat",
    event_type="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "cat", "123", "124", "1000", "-100", "2", "/etc/passwd", "2"],
  )
  values = extract_feature_dict(evt)
  # No event_type -> general features only (18)
  assert len(values) == 18
  vec = np.array(list(values.values()), dtype=np.float32)
  assert vec.shape == (18,)
  assert np.isfinite(vec).all()


def test_extract_feature_dict_handles_bad_and_negative_values():
  evt = events_pb2.EventEnvelope(
    event_id="feat-2",
    event_name="openat",
    event_type="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "bash", "bad-pid", "-1", "bad-uid", "-9", "nope", "/tmp/test", "x"],
  )
  values = extract_feature_dict(evt)
  assert all(np.isfinite(v) for v in values.values())
  # pid/tid/uid should be clamped to valid non-negative domain.
  assert values["pid_norm"] == 0.0
  assert values["tid_norm"] == 0.0
  assert values["uid_norm"] == 0.0


def test_extract_feature_dict_has_stable_features():
  evt = events_pb2.EventEnvelope(
    event_id="feat-3",
    event_name="openat",
    event_type="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "ls", "10", "10", "0", "-100", "64", "/bin/ls", "64"],
  )
  values = extract_feature_dict(evt)
  # Empty event_type -> general features only
  general_features = {
    "event_hash",
    "comm_hash",
    "path_hash",
    "pid_norm",
    "tid_norm",
    "uid_norm",
    "arg0_norm",
    "arg1_norm",
    "hour_norm",
    "minute_norm",
    "event_id_norm",
    "flags_hash",
    "path_depth_norm",
    "path_prefix_hash",
    "return_success",
    "return_errno_norm",
    "mount_ns_hash",
    "hostname_hash",
  }
  assert set(values.keys()) == general_features


def test_extract_feature_dict_path_and_return_attributes():
  # Path depth: /etc/passwd -> etc, passwd -> depth 2; /tmp -> depth 1
  evt = events_pb2.EventEnvelope(
    event_id="feat-path",
    event_name="openat",
    event_type="",
    hostname="node-1",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "-100", "0", "/etc/passwd", "O_RDONLY"],
    attributes={"return_value": "0", "mount_namespace": "4026531840"},
  )
  values = extract_feature_dict(evt)
  assert values["path_depth_norm"] == 2.0 / 20.0  # 2 components
  assert values["path_prefix_hash"] != 0.0  # "etc" hashed
  assert values["return_success"] == 1.0
  assert values["return_errno_norm"] >= 0.0
  assert values["hostname_hash"] != 0.0
  # Failed syscall
  evt_fail = events_pb2.EventEnvelope(
    event_id="feat-fail",
    event_name="connect",
    event_type="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "proc", "1", "1", "0", "3", "16", "", ""],
    attributes={"return_value": "-114", "mount_namespace": "4026531840"},
  )
  values_fail = extract_feature_dict(evt_fail)
  assert values_fail["return_success"] == 0.0
  assert values_fail["return_errno_norm"] > 0.0


def test_extract_feature_dict_adds_file_features_when_event_type_file():
  evt = events_pb2.EventEnvelope(
    event_id="feat-file",
    event_name="openat",
    event_type="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "0", "/etc/passwd", "O_RDONLY"],
  )
  values = extract_feature_dict(evt)
  assert "file_sensitive_path" in values
  assert values["file_sensitive_path"] == 1.0
  assert "file_tmp_path" in values
  assert values["file_tmp_path"] == 0.0
  assert "file_extension_hash" in values
  assert len(values) == 18 + 3


def test_extract_feature_dict_adds_network_features_when_event_type_network():
  evt = events_pb2.EventEnvelope(
    event_id="feat-net",
    event_name="connect",
    event_type="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
  )
  values = extract_feature_dict(evt)
  assert "net_addrlen_norm" in values
  assert "net_fd_norm" in values
  assert "net_socket_family_norm" in values
  assert "net_socket_type_hash" in values
  assert "net_dport_norm" in values
  assert "net_daddr_hash" in values
  assert "net_af_hash" in values
  assert "net_proc_rate_1" in values
  assert "net_host_rate_30" in values
  assert "net_daddr_rate_5" in values
  assert "net_proc_interarrival_std_1" in values
  assert "net_pair_interarrival_std_30" in values
  assert "net_host_dport_mean_5" in values
  assert "net_host_addrlen_std_1" in values
  assert "net_daddr_dport_mean_30" in values
  assert "net_proc_daddr_dport_std_120" in values
  # 7 static + 5*(4 rate + 4 interarrival mean+std + 4 proc dport/addrlen + 4 host + 4 daddr + 4 proc_daddr dport) = 7 + 5*20 = 107 network
  assert len(values) == 18 + 107


def test_extract_network_features_socket_family_type():
  """socket(): arg0=family, arg1=type (AF_INET=2, SOCK_STREAM=1)."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-socket",
    event_name="socket",
    event_type="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["socket", "41", "curl", "1", "1", "0", "2", "1", "", ""],
  )
  values = extract_feature_dict(evt)
  assert values["net_socket_family_norm"] == 2.0 / 10.0
  assert 0.0 <= values["net_socket_type_hash"] <= 1.0


def test_extract_network_features_sockaddr_from_attributes():
  """connect() with sin_port/sin_addr in attributes (or BETH data[7])."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-connect-attr",
    event_name="connect",
    event_type="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
    attributes={"sin_port": "443", "sin_addr": "192.168.1.1", "sa_family": "AF_INET"},
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 443.0 / 65535.0
  assert values["net_daddr_hash"] != 0.0
  assert values["net_af_hash"] != 0.0


def test_extract_network_features_sockaddr_from_beth_data():
  """BETH format: data[7] is stringified sockaddr dict."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-connect-beth",
    event_name="connect",
    event_type="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=[
      "connect", "42", "ssh", "1", "1", "0", "5", "{'sa_family': 'AF_INET', 'sin_port': '22', 'sin_addr': '10.0.0.2'}",
      "", "",
    ],
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 22.0 / 65535.0
  assert values["net_daddr_hash"] != 0.0


def test_extract_feature_dict_adds_process_features_when_event_type_process():
  evt = events_pb2.EventEnvelope(
    event_id="feat-proc",
    event_name="execve",
    event_type="process",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["execve", "59", "bash", "1", "1", "0", "0", "0", "/usr/bin/ls", ""],
  )
  values = extract_feature_dict(evt)
  assert "process_is_execve" in values
  assert values["process_is_execve"] == 1.0
  assert len(values) == 18 + 1


