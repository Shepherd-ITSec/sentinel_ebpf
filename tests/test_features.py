"""Tests for detector/features.py."""
import numpy as np

import events_pb2
from detector.features import (
  extract_feature_dict,
)


def _count_prefix(values: dict[str, float], prefix: str) -> int:
  return sum(1 for name in values if name.startswith(prefix))


def _count_path_depth_banks(values: dict[str, float]) -> int:
  return sum(_count_prefix(values, f"path_tok_d{d}_bucket_") for d in range(4))


def test_extract_feature_dict_returns_expected_shape():
  evt = events_pb2.EventEnvelope(
    event_id="feat-1",
    event_name="openat",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "cat", "123", "124", "1000", "-100", "2", "/etc/passwd", "2"],
  )
  values = extract_feature_dict(evt)
  event_feature_count = _count_prefix(values, "event_name_")
  assert _count_prefix(values, "comm_bucket_") == 64
  assert _count_prefix(values, "hostname_bucket_") == 32
  assert _count_prefix(values, "mount_ns_bucket_") == 32
  assert _count_path_depth_banks(values) == 256
  # Generic numeric features + event one-hot + fixed bucket banks + shared online stats.
  assert len(values) == 15 + event_feature_count + 64 + 32 + 32 + 256 + 5 * 4
  vec = np.array(list(values.values()), dtype=np.float32)
  assert vec.shape == (len(values),)
  assert np.isfinite(vec).all()


def test_extract_feature_dict_handles_bad_and_negative_values():
  evt = events_pb2.EventEnvelope(
    event_id="feat-2",
    event_name="openat",
    event_group="",
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
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "ls", "10", "10", "0", "-100", "64", "/bin/ls", "64"],
  )
  values = extract_feature_dict(evt)
  general_features = {
    "pid_norm",
    "tid_norm",
    "uid_norm",
    "arg0_norm",
    "arg1_norm",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "week_of_month_norm",
    "path_depth_norm",
    "return_success",
    "return_errno_norm",
    "event_name_openat",
  }
  for w in ("01", "1", "5", "30", "120"):
    general_features.add(f"proc_rate_{w}")
    general_features.add(f"host_rate_{w}")
    general_features.add(f"proc_interarrival_{w}")
    general_features.add(f"proc_interarrival_std_{w}")
  assert general_features.issubset(set(values.keys()))
  assert "event_hash" not in values
  assert "comm_hash" not in values
  assert "path_hash" not in values
  assert "event_id_norm" not in values
  assert "flags_hash" not in values
  assert "path_prefix_hash" not in values
  assert "mount_ns_hash" not in values
  assert "hostname_hash" not in values
  assert _count_prefix(values, "event_name_") >= 1
  assert _count_prefix(values, "comm_bucket_") == 64
  assert _count_prefix(values, "hostname_bucket_") == 32
  assert _count_prefix(values, "mount_ns_bucket_") == 32
  assert _count_path_depth_banks(values) == 256


def test_extract_feature_dict_weekday_and_week_of_month():
  """weekday_sin/cos and week_of_month_norm (1-4 quarters) from ts_unix_nano."""
  # 2023-11-14 22:13:20 UTC = Tuesday (weekday=1), day 14 = week 2 of month
  evt = events_pb2.EventEnvelope(
    event_id="feat-time",
    event_name="openat",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "cat", "1", "1", "0", "0", "0", "/tmp/x", ""],
  )
  values = extract_feature_dict(evt)
  assert "weekday_sin" in values
  assert "weekday_cos" in values
  assert "week_of_month_norm" in values
  assert 0.0 <= values["week_of_month_norm"] <= 1.0
  # 2023-11-14 is Tuesday (weekday=1): angle = 2π*(1/7)
  expect_angle = 2.0 * np.pi * (1.0 / 7.0)
  assert abs(values["weekday_sin"] - float(np.sin(expect_angle))) < 0.01
  assert abs(values["weekday_cos"] - float(np.cos(expect_angle))) < 0.01
  # day 14 -> week 2 -> (2-1)/3 = 1/3
  assert abs(values["week_of_month_norm"] - 1.0 / 3.0) < 0.01


def test_extract_feature_dict_path_and_return_attributes():
  # Path depth: /etc/passwd -> etc, passwd -> depth 2; /tmp -> depth 1
  evt = events_pb2.EventEnvelope(
    event_id="feat-path",
    event_name="openat",
    event_group="",
    hostname="node-1",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "-100", "0", "/etc/passwd", "O_RDONLY"],
    attributes={"return_value": "0", "mount_namespace": "4026531840"},
  )
  values = extract_feature_dict(evt)
  assert values["path_depth_norm"] == 2.0 / 20.0  # 2 components
  assert values["event_name_openat"] == 1.0
  assert sum(v for k, v in values.items() if k.startswith("hostname_bucket_")) == 1.0
  assert sum(v for k, v in values.items() if k.startswith("mount_ns_bucket_")) == 1.0
  assert sum(v for k, v in values.items() if k.startswith("path_tok_d")) >= 1.0
  assert values["return_success"] == 1.0
  assert values["return_errno_norm"] >= 0.0
  # Failed syscall
  evt_fail = events_pb2.EventEnvelope(
    event_id="feat-fail",
    event_name="connect",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "proc", "1", "1", "0", "3", "16", "", ""],
    attributes={"return_value": "-114", "mount_namespace": "4026531840"},
  )
  values_fail = extract_feature_dict(evt_fail)
  assert values_fail["return_success"] == 0.0
  assert values_fail["return_errno_norm"] > 0.0


def test_extract_feature_dict_adds_file_features_when_event_type_file():
  base_evt = events_pb2.EventEnvelope(
    event_id="feat-file-base",
    event_name="openat",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "0", "/etc/passwd", "O_RDONLY"],
  )
  evt = events_pb2.EventEnvelope(
    event_id="feat-file",
    event_name="openat",
    event_group="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "0", "/etc/passwd", "O_RDONLY"],
  )
  values = extract_feature_dict(evt)
  assert "file_sensitive_path" in values
  assert values["file_sensitive_path"] == 1.0
  assert "file_tmp_path" in values
  assert values["file_tmp_path"] == 0.0
  assert any(name.startswith("file_extension_bucket_") for name in values)
  assert values["file_event_name_openat"] == 1.0
  assert any(name.startswith("file_flags_bucket_") for name in values)
  assert "file_extension_hash" not in values
  assert "file_event_name_hash" not in values
  assert "file_flags_hash" not in values
  assert "file_arg0_norm" in values
  assert "file_arg1_norm" in values
  assert "proc_rate_1" in values
  assert "host_rate_30" in values
  assert "file_path_rate_5" in values
  assert "file_pair_interarrival_std_1" in values
  assert "file_proc_path_depth_mean_120" in values
  assert "file_host_path_depth_std_30" in values
  assert len(values) > len(extract_feature_dict(base_evt))


def test_extract_file_features_flags_are_schema_stable():
  """Encoded file flags stay schema-stable across file syscalls."""
  evt_openat = events_pb2.EventEnvelope(
    event_id="f1",
    event_name="openat",
    event_group="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "524288", "/etc/passwd", "O_RDONLY"],
    attributes={"open_flags": "O_RDONLY|O_CLOEXEC"},
  )
  values_open = extract_feature_dict(evt_openat)
  assert any(name.startswith("file_flags_bucket_") for name in values_open)
  assert sum(value for name, value in values_open.items() if name.startswith("file_flags_bucket_")) >= 1.0

  evt_unlink = events_pb2.EventEnvelope(
    event_id="f2",
    event_name="unlink",
    event_group="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["unlink", "87", "rm", "1", "1", "0", "0", "0", "/tmp/foo", ""],
  )
  values_unlink = extract_feature_dict(evt_unlink)
  assert any(name.startswith("file_flags_bucket_") for name in values_unlink)
  assert sum(value for name, value in values_unlink.items() if name.startswith("file_flags_bucket_")) >= 1.0
  assert values_unlink["file_tmp_path"] == 1.0

  evt_chmod = events_pb2.EventEnvelope(
    event_id="f3",
    event_name="chmod",
    event_group="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["chmod", "90", "chmod", "1", "1", "0", "493", "0", "/etc/shadow", ""],
  )
  values_chmod = extract_feature_dict(evt_chmod)
  assert any(name.startswith("file_flags_bucket_") for name in values_chmod)
  assert values_chmod["file_event_name_chmod"] == 1.0
  assert values_unlink["file_event_name_unlink"] == 1.0


def test_extract_feature_dict_adds_network_features_when_event_type_network():
  base_evt = events_pb2.EventEnvelope(
    event_id="feat-net-base",
    event_name="connect",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
  )
  evt = events_pb2.EventEnvelope(
    event_id="feat-net",
    event_name="connect",
    event_group="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
  )
  values = extract_feature_dict(evt)
  assert "net_addrlen_norm" in values
  assert "net_fd_norm" in values
  assert "net_socket_family_norm" in values
  assert any(name.startswith("net_socket_type_bucket_") for name in values)
  assert "net_dport_norm" in values
  assert any(name.startswith("net_daddr_bucket_") for name in values)
  assert any(name.startswith("net_af_") for name in values)
  assert "net_socket_type_hash" not in values
  assert "net_daddr_hash" not in values
  assert "net_af_hash" not in values
  assert "proc_rate_1" in values
  assert "host_rate_30" in values
  assert "net_daddr_rate_5" in values
  assert "net_pair_interarrival_std_30" in values
  assert "net_host_dport_mean_5" in values
  assert "net_host_addrlen_std_1" in values
  assert "net_daddr_dport_mean_30" in values
  assert "net_proc_daddr_dport_std_120" in values
  assert len(values) > len(extract_feature_dict(base_evt))


def test_extract_network_features_socket_family_type():
  """socket(): arg0=family, arg1=type (AF_INET=2, SOCK_STREAM=1)."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-socket",
    event_name="socket",
    event_group="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["socket", "41", "curl", "1", "1", "0", "2", "1", "", ""],
  )
  values = extract_feature_dict(evt)
  assert values["net_socket_family_norm"] == 2.0 / 10.0
  assert sum(value for name, value in values.items() if name.startswith("net_socket_type_bucket_")) == 1.0


def test_extract_network_features_sockaddr_from_attributes():
  """connect() with sin_port/sin_addr in attributes (or BETH data[7])."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-connect-attr",
    event_name="connect",
    event_group="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
    attributes={"sin_port": "443", "sin_addr": "192.168.1.1", "sa_family": "AF_INET"},
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 443.0 / 65535.0
  assert sum(value for name, value in values.items() if name.startswith("net_daddr_bucket_")) == 1.0
  assert values["net_af_af_inet"] == 1.0


def test_extract_network_features_sockaddr_from_beth_data():
  """BETH format: data[7] is stringified sockaddr dict."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-connect-beth",
    event_name="connect",
    event_group="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=[
      "connect", "42", "ssh", "1", "1", "0", "5", "{'sa_family': 'AF_INET', 'sin_port': '22', 'sin_addr': '10.0.0.2'}",
      "", "",
    ],
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 22.0 / 65535.0
  assert sum(value for name, value in values.items() if name.startswith("net_daddr_bucket_")) == 1.0


def test_extract_feature_dict_hash_encoding_returns_legacy_schema():
  """encoding='hash' produces legacy scalar hash features (event_hash, comm_hash, etc.)."""
  evt = events_pb2.EventEnvelope(
    event_id="feat-hash",
    event_name="openat",
    event_group="",
    hostname="node-1",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "0", "/etc/passwd", "O_RDONLY"],
    attributes={"return_value": "0", "mount_namespace": "4026531840"},
  )
  values = extract_feature_dict(evt, encoding="hash")
  assert "event_hash" in values
  assert "comm_hash" in values
  assert "path_hash" in values
  assert "event_id_norm" in values
  assert "flags_hash" in values
  assert "path_prefix_hash" in values
  assert "mount_ns_hash" in values
  assert "hostname_hash" in values
  assert "event_name_openat" not in values
  assert "comm_bucket_000" not in values
  assert "path_tok_d0_bucket_000" not in values
  assert len(values) == 23 + 5 * 4  # 23 generic + shared online stats


def test_extract_feature_dict_hash_encoding_keeps_legacy_type_specific_hashes():
  file_evt = events_pb2.EventEnvelope(
    event_id="feat-file-hash",
    event_name="openat",
    event_group="file",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "257", "cat", "1", "1", "0", "0", "0", "/etc/passwd", "O_RDONLY"],
    attributes={"open_flags": "O_RDONLY|O_CLOEXEC"},
  )
  file_values = extract_feature_dict(file_evt, encoding="hash")
  assert "file_extension_hash" in file_values
  assert "file_event_name_hash" in file_values
  assert "file_flags_hash" in file_values
  assert "file_event_name_openat" not in file_values
  assert not any(name.startswith("file_extension_bucket_") for name in file_values)

  net_evt = events_pb2.EventEnvelope(
    event_id="feat-net-hash",
    event_name="connect",
    event_group="network",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["connect", "42", "curl", "1", "1", "0", "3", "16", "", ""],
    attributes={"sin_port": "443", "sin_addr": "192.168.1.1", "sa_family": "AF_INET"},
  )
  net_values = extract_feature_dict(net_evt, encoding="hash")
  assert "net_socket_type_hash" in net_values
  assert "net_daddr_hash" in net_values
  assert "net_af_hash" in net_values
  assert not any(name.startswith("net_socket_type_bucket_") for name in net_values)
  assert "net_af_af_inet" not in net_values


def test_extract_feature_dict_adds_process_features_when_event_type_process():
  base_evt = events_pb2.EventEnvelope(
    event_id="feat-proc-base",
    event_name="execve",
    event_group="",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["execve", "59", "bash", "1", "1", "0", "0", "0", "/usr/bin/ls", ""],
  )
  evt = events_pb2.EventEnvelope(
    event_id="feat-proc",
    event_name="execve",
    event_group="process",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["execve", "59", "bash", "1", "1", "0", "0", "0", "/usr/bin/ls", ""],
  )
  values = extract_feature_dict(evt)
  assert "process_is_execve" in values
  assert values["process_is_execve"] == 1.0
  assert "process_is_fork" in values
  assert values["process_is_fork"] == 0.0  # execve, not fork
  assert len(values) == len(extract_feature_dict(base_evt)) + 2


