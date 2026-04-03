"""Tests for detector/features.py."""
import numpy as np

import events_pb2
from detector.features import (
  extract_feature_dict,
)


def _evt(
  *,
  event_id="e",
  syscall_name="openat",
  event_group="",
  ts_unix_nano=1_700_000_000_000_000_000,
  syscall_nr=2,
  comm="cat",
  pid="1",
  tid="1",
  uid="0",
  arg0="0",
  arg1="0",
  path="",
  hostname="",
  attributes=None,
):
  e = events_pb2.EventEnvelope(
    event_id=event_id,
    syscall_name=syscall_name,
    event_group=event_group,
    ts_unix_nano=ts_unix_nano,
    syscall_nr=syscall_nr,
    comm=comm,
    pid=pid,
    tid=tid,
    uid=uid,
    arg0=arg0,
    arg1=arg1,
    hostname=hostname,
  )
  if path:
    e.attributes["fd_path"] = path
  if attributes:
    for k, v in attributes.items():
      e.attributes[k] = v
  return e


def _count_prefix(values: dict[str, float], prefix: str) -> int:
  return sum(1 for name in values if name.startswith(prefix))


def _count_path_depth_banks(values: dict[str, float]) -> int:
  return sum(_count_prefix(values, f"path_tok_d{d}_bucket_") for d in range(4))


def test_extract_feature_dict_returns_expected_shape():
  evt = _evt(
    event_id="feat-1",
    comm="cat",
    pid="123",
    tid="124",
    uid="1000",
    arg0="-100",
    arg1="2",
    path="/etc/passwd",
  )
  values = extract_feature_dict(evt)
  assert _count_prefix(values, "event_name_") == 0
  assert _count_prefix(values, "comm_bucket_") == 64
  assert _count_prefix(values, "hostname_bucket_") == 32
  assert _count_path_depth_banks(values) == 256
  assert len(values) == 11 + 64 + 32 + 256 + 5 * 4
  vec = np.array(list(values.values()), dtype=np.float32)
  assert vec.shape == (len(values),)
  assert np.isfinite(vec).all()


def test_extract_feature_dict_handles_bad_and_negative_values():
  evt = _evt(
    event_id="feat-2",
    comm="bash",
    pid="bad-pid",
    tid="-1",
    uid="bad-uid",
    arg0="-9",
    arg1="nope",
    path="/tmp/test",
  )
  values = extract_feature_dict(evt)
  assert all(np.isfinite(v) for v in values.values())
  assert values["pid_norm"] == 0.0
  assert values["tid_norm"] == 0.0
  assert values["uid_norm"] == 0.0


def test_extract_feature_dict_has_stable_features():
  evt = _evt(
    event_id="feat-3",
    comm="ls",
    pid="10",
    tid="10",
    uid="0",
    arg0="-100",
    arg1="64",
    path="/bin/ls",
  )
  values = extract_feature_dict(evt)
  general_features = {
    "pid_norm",
    "tid_norm",
    "uid_norm",
    "arg0_norm",
    "arg1_norm",
    "day_cycle_sin",
    "day_cycle_cos",
    "week_of_month_norm",
    "path_depth_norm",
    "return_success",
    "return_errno_norm",
  }
  for w in ("01", "1", "5", "30", "120"):
    general_features.add(f"proc_rate_{w}")
    general_features.add(f"host_rate_{w}")
    general_features.add(f"proc_interarrival_{w}")
    general_features.add(f"proc_interarrival_std_{w}")
  assert general_features.issubset(set(values.keys()))
  assert "comm_hash" not in values
  assert "path_hash" not in values
  assert "event_id_norm" not in values
  assert "flags_hash" not in values
  assert "path_prefix_hash" not in values
  assert "hostname_hash" not in values
  assert _count_prefix(values, "event_name_") == 0
  assert _count_prefix(values, "comm_bucket_") == 64
  assert _count_prefix(values, "hostname_bucket_") == 32
  assert _count_path_depth_banks(values) == 256


def test_extract_feature_dict_day_cycle_and_week_of_month():
  evt = _evt(
    event_id="feat-time",
    path="/tmp/x",
  )
  values = extract_feature_dict(evt)
  assert "day_cycle_sin" in values
  assert "day_cycle_cos" in values
  assert "week_of_month_norm" in values
  assert 0.0 <= values["week_of_month_norm"] <= 1.0
  ts_ns = 1_700_000_000_000_000_000
  expect_fraction = float(ts_ns % 86_400_000_000_000) / 86_400_000_000_000.0
  expect_angle = 2.0 * np.pi * expect_fraction
  assert abs(values["day_cycle_sin"] - float(np.sin(expect_angle))) < 0.01
  assert abs(values["day_cycle_cos"] - float(np.cos(expect_angle))) < 0.01
  assert abs(values["week_of_month_norm"] - 1.0 / 3.0) < 0.01


def test_extract_feature_dict_path_and_return_attributes():
  evt = _evt(
    event_id="feat-path",
    event_group="file",
    syscall_nr=257,
    hostname="node-1",
    arg0="-100",
    path="/etc/passwd",
    attributes={"return_value": "0"},
  )
  values = extract_feature_dict(evt)
  assert values["path_depth_norm"] == 2.0 / 20.0
  assert values["group_syscall_openat"] == 1.0
  assert sum(v for k, v in values.items() if k.startswith("hostname_bucket_")) == 1.0
  assert sum(v for k, v in values.items() if k.startswith("path_tok_d")) >= 1.0
  assert values["return_success"] == 1.0
  assert values["return_errno_norm"] >= 0.0
  evt_fail = _evt(
    event_id="feat-fail",
    syscall_name="connect",
    event_group="network",
    syscall_nr=42,
    comm="proc",
    arg0="3",
    arg1="16",
    attributes={"return_value": "-114"},
  )
  values_fail = extract_feature_dict(evt_fail)
  assert values_fail["return_success"] == 0.0
  assert values_fail["return_errno_norm"] > 0.0


def test_extract_feature_dict_adds_file_features_when_event_type_file():
  base_evt = _evt(event_id="feat-file-base", syscall_nr=257, path="/etc/passwd")
  evt = _evt(
    event_id="feat-file",
    event_group="file",
    syscall_nr=257,
    path="/etc/passwd",
  )
  values = extract_feature_dict(evt)
  assert "group_sensitive_path" in values
  assert values["group_sensitive_path"] == 1.0
  assert "group_tmp_path" in values
  assert values["group_tmp_path"] == 0.0
  assert any(name.startswith("group_ext_bucket_") for name in values)
  assert values["group_syscall_openat"] == 1.0
  assert any(name.startswith("group_flags_bucket_") for name in values)
  assert "group_ext_hash" not in values
  assert "group_syscall_hash" not in values
  assert "group_flags_hash" not in values
  assert "arg0_norm" in values
  assert "arg1_norm" in values
  assert "proc_rate_1" in values
  assert "host_rate_30" in values
  assert "group_path_rate_5" in values
  assert "group_pair_interarrival_std_1" in values
  assert "group_proc_path_depth_mean_120" in values
  assert "group_host_path_depth_std_30" in values
  assert len(values) > len(extract_feature_dict(base_evt))


def test_extract_file_features_flags_are_schema_stable():
  evt_openat = _evt(
    event_id="f1",
    event_group="file",
    syscall_nr=257,
    arg1="524288",
    path="/etc/passwd",
    attributes={"flags": "O_RDONLY|O_CLOEXEC"},
  )
  values_open = extract_feature_dict(evt_openat)
  assert any(name.startswith("group_flags_bucket_") for name in values_open)
  assert sum(value for name, value in values_open.items() if name.startswith("group_flags_bucket_")) >= 1.0

  evt_unlink = _evt(
    event_id="f2",
    syscall_name="unlink",
    event_group="file",
    syscall_nr=87,
    comm="rm",
    path="/tmp/foo",
  )
  values_unlink = extract_feature_dict(evt_unlink)
  assert any(name.startswith("group_flags_bucket_") for name in values_unlink)
  assert sum(value for name, value in values_unlink.items() if name.startswith("group_flags_bucket_")) >= 1.0
  assert values_unlink["group_tmp_path"] == 1.0

  evt_chmod = _evt(
    event_id="f3",
    syscall_name="chmod",
    event_group="file",
    syscall_nr=90,
    comm="chmod",
    arg0="493",
    path="/etc/shadow",
  )
  values_chmod = extract_feature_dict(evt_chmod)
  assert any(name.startswith("group_flags_bucket_") for name in values_chmod)
  assert "group_syscall_chmod" not in values_chmod
  assert values_unlink["group_syscall_unlink"] == 1.0


def test_extract_feature_dict_adds_network_features_when_event_type_network():
  base_evt = _evt(
    event_id="feat-net-base",
    syscall_name="connect",
    syscall_nr=42,
    comm="curl",
    arg0="3",
    arg1="16",
  )
  evt = _evt(
    event_id="feat-net",
    syscall_name="connect",
    event_group="network",
    syscall_nr=42,
    comm="curl",
    arg0="3",
    arg1="16",
  )
  values = extract_feature_dict(evt)
  assert values["group_syscall_connect"] == 1.0
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
  evt = _evt(
    event_id="feat-socket",
    syscall_name="socket",
    event_group="network",
    syscall_nr=41,
    comm="curl",
    arg0="2",
    arg1="1",
  )
  values = extract_feature_dict(evt)
  assert values["group_syscall_socket"] == 1.0
  assert values["net_socket_family_norm"] == 2.0 / 10.0
  assert sum(value for name, value in values.items() if name.startswith("net_socket_type_bucket_")) == 1.0


def test_extract_network_features_sockaddr_from_attributes():
  evt = _evt(
    event_id="feat-connect-attr",
    syscall_name="connect",
    event_group="network",
    syscall_nr=42,
    comm="curl",
    arg0="3",
    arg1="16",
    attributes={
      "fd_sock_remote_port": "443",
      "fd_sock_remote_addr": "192.168.1.1",
      "fd_sock_family": "AF_INET",
    },
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 443.0 / 65535.0
  assert sum(value for name, value in values.items() if name.startswith("net_daddr_bucket_")) == 1.0
  assert values["net_af_af_inet"] == 1.0


def test_extract_network_features_sockaddr_from_attributes_connect():
  """Synthetic connect rows carry remote endpoint in ``fd_sock_*`` attributes."""
  evt = _evt(
    event_id="feat-connect-arg1",
    syscall_name="connect",
    event_group="network",
    syscall_nr=42,
    comm="ssh",
    arg0="5",
    arg1="16",
    attributes={
      "fd_sock_family": "AF_INET",
      "fd_sock_remote_port": "22",
      "fd_sock_remote_addr": "10.0.0.2",
    },
  )
  values = extract_feature_dict(evt)
  assert values["net_dport_norm"] == 22.0 / 65535.0
  assert sum(value for name, value in values.items() if name.startswith("net_daddr_bucket_")) == 1.0


def test_extract_feature_dict_frequency_view_returns_expected_schema():
  evt = _evt(
    event_id="feat-frequency",
    syscall_nr=257,
    hostname="node-1",
    arg1="O_RDONLY",
    path="/etc/passwd",
    attributes={"return_value": "0"},
  )
  values = extract_feature_dict(evt, feature_view="frequency")
  assert "event_name_hash" not in values
  assert "comm_hash" in values
  assert "path_hash" in values
  assert "event_id_norm" in values
  assert "flags_hash" not in values
  assert "path_prefix_hash" in values
  assert "hostname_hash" in values
  assert "comm_bucket_000" not in values
  assert "path_tok_d0_bucket_000" not in values
  assert "proc_rate_1" not in values
  assert "day_cycle_sin" not in values
  assert "day_fraction_norm" in values
  assert len(values) == 15


def test_extract_feature_dict_zscore_view_uses_syscall_embedding_and_day_cycle():
  evt = _evt(
    event_id="feat-zscore",
    syscall_name="execve",
    tid="7",
    ts_unix_nano=1_700_000_000_000_000_000,
  )
  values = extract_feature_dict(evt, feature_view="zscore")
  assert "day_cycle_sin" in values
  assert "day_cycle_cos" in values
  assert "week_of_month_norm" not in values
  assert "pid_norm" not in values
  assert "sequence_ctx_000" not in values
  assert len([name for name in values if name.startswith("syscall_w2v_")]) == 5


def test_extract_feature_dict_frequency_view_keeps_type_specific_hashes():
  file_evt = _evt(
    event_id="feat-file-frequency",
    event_group="file",
    syscall_nr=257,
    path="/etc/passwd",
    attributes={"flags": "O_RDONLY|O_CLOEXEC"},
  )
  file_values = extract_feature_dict(file_evt, feature_view="frequency")
  assert "event_id_norm" in file_values
  assert "group_sensitive_path" not in file_values
  assert "group_tmp_path" not in file_values
  assert "group_ext_hash" in file_values
  assert "group_flags_hash" in file_values
  assert "group_syscall_openat" not in file_values
  assert "group_path_rate_5" not in file_values
  assert not any(name.startswith("group_ext_bucket_") for name in file_values)

  net_evt = _evt(
    event_id="feat-net-frequency",
    syscall_name="connect",
    event_group="network",
    syscall_nr=42,
    comm="curl",
    arg0="3",
    arg1="16",
    attributes={
      "fd_sock_remote_port": "443",
      "fd_sock_remote_addr": "192.168.1.1",
      "fd_sock_family": "AF_INET",
    },
  )
  net_values = extract_feature_dict(net_evt, feature_view="frequency")
  assert "event_id_norm" in net_values
  assert "group_syscall_connect" not in net_values
  assert "net_socket_type_hash" in net_values
  assert "net_daddr_hash" in net_values
  assert "net_af_hash" in net_values
  assert "net_pair_rate_5" not in net_values
  assert not any(name.startswith("net_socket_type_bucket_") for name in net_values)
  assert "net_af_af_inet" not in net_values


def test_extract_feature_dict_adds_process_features_when_event_type_process():
  base_evt = _evt(
    event_id="feat-proc-base",
    syscall_name="execve",
    syscall_nr=59,
    comm="bash",
    path="/usr/bin/ls",
  )
  evt = _evt(
    event_id="feat-proc",
    syscall_name="execve",
    event_group="process",
    syscall_nr=59,
    comm="bash",
    path="/usr/bin/ls",
  )
  values = extract_feature_dict(evt)
  assert values.get("group_syscall_execve") == 1.0
  assert values.get("group_syscall_fork") == 0.0
  assert len(values) > len(extract_feature_dict(base_evt))


def test_memstream_feature_view_drops_sparse_context_banks():
  evt = _evt(
    event_id="feat-mem-view",
    event_group="file",
    syscall_nr=257,
    hostname="node-1",
    arg0="-100",
    path="/etc/passwd",
    attributes={"return_value": "0", "flags": "O_RDONLY|O_CLOEXEC"},
  )
  values = extract_feature_dict(evt, feature_view="memstream")
  assert "pid_norm" in values
  assert "group_sensitive_path" in values
  assert "group_path_rate_5" in values
  assert "group_syscall_openat" not in values
  assert _count_prefix(values, "comm_bucket_") == 0
  assert _count_prefix(values, "hostname_bucket_") == 0
  assert _count_path_depth_banks(values) == 0
  assert _count_prefix(values, "group_ext_bucket_") == 0
  assert _count_prefix(values, "group_flags_bucket_") == 0


def test_loda_feature_view_keeps_small_network_identity_and_drops_open_world_banks():
  evt = _evt(
    event_id="feat-loda-view",
    syscall_name="socket",
    event_group="network",
    syscall_nr=41,
    hostname="node-1",
    comm="curl",
    arg0="2",
    arg1="1",
  )
  values = extract_feature_dict(evt, feature_view="loda")
  assert "net_fd_norm" in values
  assert _count_prefix(values, "group_syscall_") == 0
  assert "net_af_af_inet" in values
  assert _count_prefix(values, "net_socket_type_bucket_") == 16
  assert _count_prefix(values, "net_daddr_bucket_") == 0
  assert _count_prefix(values, "comm_bucket_") == 0
  assert _count_path_depth_banks(values) == 0


def test_frequency_feature_view_matches_expected_file_schema():
  evt = _evt(
    event_id="feat-frequency-view",
    event_group="file",
    syscall_nr=257,
    hostname="node-1",
    path="/etc/passwd",
    attributes={"return_value": "0", "flags": "O_RDONLY|O_CLOEXEC"},
  )
  values = extract_feature_dict(evt, feature_view="frequency")
  assert "event_id_norm" in values
  assert "flags_hash" not in values
  assert "comm_hash" in values
  assert "path_hash" in values
  assert "group_sensitive_path" not in values
  assert "group_syscall_hash" not in values
  assert "group_flags_hash" in values
  assert "proc_rate_1" not in values
  assert "event_name_openat" not in values
  assert _count_prefix(values, "comm_bucket_") == 0
  assert _count_prefix(values, "group_flags_bucket_") == 0


def test_custom_event_group_uses_group_syscalls_from_rules(temp_dir, monkeypatch):
  """Arbitrary group name; syscall vocabulary comes from groups.<name>.syscalls in rules.yaml."""
  from detector import features as feat_mod

  rules = temp_dir / "rules.yaml"
  rules.write_text(
    """groups:
  fs_activity:
    syscalls: [openat]
    features:
      sensitive_paths: [/secret]
      tmp_paths: [/tmp]
rules:
  - name: r
    group: fs_activity
"""
  )
  monkeypatch.setenv("DETECTOR_RULES_PATH", str(rules))
  feat_mod._detector_rules_path.cache_clear()
  feat_mod._rules_config.cache_clear()
  try:
    evt = _evt(
      event_id="custom-g",
      event_group="fs_activity",
      syscall_name="openat",
      syscall_nr=257,
      path="/secret/data",
    )
    values = extract_feature_dict(evt)
    assert values.get("group_sensitive_path") == 1.0
    assert values.get("group_syscall_openat") == 1.0
  finally:
    feat_mod._detector_rules_path.cache_clear()
    feat_mod._rules_config.cache_clear()
