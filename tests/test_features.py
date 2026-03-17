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
  # No event_type -> general features (20) + shared online stats (5 windows * 4)
  assert len(values) == 20 + 5 * 4
  vec = np.array(list(values.values()), dtype=np.float32)
  assert vec.shape == (20 + 5 * 4,)
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
  # Empty event_type -> general features + shared online stats (proc/host rate, proc interarrival)
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
    "weekday_norm",
    "week_of_month_norm",
    "event_id_norm",
    "flags_hash",
    "path_depth_norm",
    "path_prefix_hash",
    "return_success",
    "return_errno_norm",
    "mount_ns_hash",
    "hostname_hash",
  }
  for w in ("01", "1", "5", "30", "120"):
    general_features.add(f"proc_rate_{w}")
    general_features.add(f"host_rate_{w}")
    general_features.add(f"proc_interarrival_{w}")
    general_features.add(f"proc_interarrival_std_{w}")
  assert set(values.keys()) == general_features


def test_extract_feature_dict_weekday_and_week_of_month():
  """weekday_norm (0=Mon..6=Sun) and week_of_month_norm (1-4 quarters) from ts_unix_nano."""
  # 2023-11-14 22:13:20 UTC = Tuesday (weekday=1), day 14 = week 2 of month
  evt = events_pb2.EventEnvelope(
    event_id="feat-time",
    event_name="openat",
    event_type="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "cat", "1", "1", "0", "0", "0", "/tmp/x", ""],
  )
  values = extract_feature_dict(evt)
  assert "weekday_norm" in values
  assert "week_of_month_norm" in values
  assert 0.0 <= values["weekday_norm"] <= 1.0
  assert 0.0 <= values["week_of_month_norm"] <= 1.0
  # 2023-11-14 is Tuesday (weekday=1) -> 1/7
  assert abs(values["weekday_norm"] - 1.0 / 7.0) < 0.01
  # day 14 -> week 2 -> (2-1)/3 = 1/3
  assert abs(values["week_of_month_norm"] - 1.0 / 3.0) < 0.01


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
  assert "file_event_name_hash" in values
  assert "file_open_flags_hash" in values
  assert "file_arg0_norm" in values
  assert "file_arg1_norm" in values
  assert "proc_rate_1" in values
  assert "host_rate_30" in values
  assert "file_path_rate_5" in values
  assert "file_pair_interarrival_std_1" in values
  assert "file_proc_path_depth_mean_120" in values
  assert "file_host_path_depth_std_30" in values
  # 20 general + 5*4 shared online + 7 static file + 5*(2 rate + 4 interarrival + 6 path_depth) = 40+57 = 97
  assert len(values) == 97


def test_extract_file_features_open_flags_only_for_open_syscalls():
  """open/openat/openat2 use open_flags; unlink, chmod, chown etc. get file_open_flags_hash 0.0."""
  evt_openat = events_pb2.EventEnvelope(
    event_id="f1",
    event_name="openat",
    event_type="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "524288", "/etc/passwd", "O_RDONLY"],
    attributes={"open_flags": "O_RDONLY|O_CLOEXEC"},
  )
  values_open = extract_feature_dict(evt_openat)
  assert values_open["file_open_flags_hash"] != 0.0

  evt_unlink = events_pb2.EventEnvelope(
    event_id="f2",
    event_name="unlink",
    event_type="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["unlink", "87", "rm", "1", "1", "0", "0", "0", "/tmp/foo", ""],
  )
  values_unlink = extract_feature_dict(evt_unlink)
  assert values_unlink["file_open_flags_hash"] == 0.0
  assert values_unlink["file_tmp_path"] == 1.0

  evt_chmod = events_pb2.EventEnvelope(
    event_id="f3",
    event_name="chmod",
    event_type="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["chmod", "90", "chmod", "1", "1", "0", "493", "0", "/etc/shadow", ""],
  )
  values_chmod = extract_feature_dict(evt_chmod)
  assert values_chmod["file_open_flags_hash"] == 0.0
  assert values_chmod["file_event_name_hash"] != values_unlink["file_event_name_hash"]


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
  assert "proc_rate_1" in values
  assert "host_rate_30" in values
  assert "net_daddr_rate_5" in values
  assert "net_pair_interarrival_std_30" in values
  assert "net_host_dport_mean_5" in values
  assert "net_host_addrlen_std_1" in values
  assert "net_daddr_dport_mean_30" in values
  assert "net_proc_daddr_dport_std_120" in values
  # 20 general + 5*4 shared online + 7 static + 5*16 network-specific (no proc/host rate or proc interarrival) = 40+87 = 127
  assert len(values) == 127


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
  assert "process_is_fork" in values
  assert values["process_is_fork"] == 0.0  # execve, not fork
  assert len(values) == 20 + 5 * 4 + 2  # +2 for process_is_execve, process_is_fork


