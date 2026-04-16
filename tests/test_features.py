"""Tests for feature primitives."""
import numpy as np

import events_pb2
from detector.building_blocks.primitives.features import (
  extract_feature_dict,
  feature_view_for_algorithm,
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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert _count_prefix(values, "event_name_") == 0
  assert _count_prefix(values, "comm_bucket_") == 0
  assert len(values) > 0
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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert all(np.isfinite(v) for v in values.values())
  assert values["pid_norm"] == 0.0
  freq, _ = extract_feature_dict(evt, feature_view="frequency")
  assert 0.0 <= freq["pid_hash"] < 1.0
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
  values, _ = extract_feature_dict(evt, feature_view="full")
  general_features = {
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
    "path_prefix_hash",
  }
  assert general_features.issubset(set(values.keys()))
  assert "flags_hash" not in values
  assert _count_prefix(values, "event_name_") == 0
  assert _count_prefix(values, "comm_bucket_") == 0


def test_extract_feature_dict_day_cycle_and_week_of_month():
  evt = _evt(
    event_id="feat-time",
    path="/tmp/x",
  )
  values, _ = extract_feature_dict(evt, feature_view="full")
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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert values["path_depth_norm"] == 2.0 / 20.0
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
  values_fail, _ = extract_feature_dict(evt_fail, feature_view="full")
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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert "file_sensitive_path" in values
  assert values["file_sensitive_path"] == 1.0
  assert "file_tmp_path" in values
  assert values["file_tmp_path"] == 0.0
  assert "file_flags_hash" in values
  assert "arg0_norm" in values
  assert "arg1_norm" in values
  assert "group_path_rate_5" not in values
  assert len(values) > len(extract_feature_dict(base_evt, feature_view="full")[0])


def test_extract_file_features_flags_are_schema_stable():
  evt_openat = _evt(
    event_id="f1",
    event_group="file",
    syscall_nr=257,
    arg1="524288",
    path="/etc/passwd",
    attributes={"flags": "O_RDONLY|O_CLOEXEC"},
  )
  values_open, _ = extract_feature_dict(evt_openat, feature_view="full")
  assert "file_flags_hash" in values_open

  evt_unlink = _evt(
    event_id="f2",
    syscall_name="unlink",
    event_group="file",
    syscall_nr=87,
    comm="rm",
    path="/tmp/foo",
  )
  values_unlink, _ = extract_feature_dict(evt_unlink, feature_view="full")
  assert "file_flags_hash" in values_unlink
  assert values_unlink["file_tmp_path"] == 1.0

  evt_chmod = _evt(
    event_id="f3",
    syscall_name="chmod",
    event_group="file",
    syscall_nr=90,
    comm="chmod",
    arg0="493",
    path="/etc/shadow",
  )
  values_chmod, _ = extract_feature_dict(evt_chmod, feature_view="full")
  assert "file_flags_hash" in values_chmod
  # no per-syscall one-hot features in current extractor


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert "net_socket_family_norm" in values
  assert "net_dport_norm" in values
  assert "net_pair_rate_5" not in values
  assert len(values) > len(extract_feature_dict(base_evt, feature_view="full")[0])


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert values["net_socket_family_norm"] == 2.0 / 10.0


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert values["net_dport_norm"] == 443.0 / 65535.0


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert values["net_dport_norm"] == 22.0 / 65535.0


def test_extract_feature_dict_frequency_view_returns_expected_schema():
  evt = _evt(
    event_id="feat-frequency",
    syscall_nr=257,
    hostname="node-1",
    arg1="O_RDONLY",
    path="/etc/passwd",
    attributes={"return_value": "0"},
  )
  values, _ = extract_feature_dict(evt, feature_view="frequency")
  assert "path_hash" in values
  assert "flags_hash" not in values
  assert "path_prefix_hash" in values
  assert "hostname_hash" in values
  assert "comm_bucket_000" not in values
  assert "path_tok_d0_bucket_000" not in values
  assert "proc_rate_1" not in values
  assert "day_cycle_sin" not in values
  assert "day_fraction_norm" in values
  assert len(values) >= 6


def test_extract_feature_dict_zscore_algorithm_uses_frequency_view_schema():
  evt = _evt(
    event_id="feat-zscore",
    event_group="file",
    syscall_nr=257,
    syscall_name="openat",
    hostname="node-1",
    path="/etc/passwd",
    attributes={"return_value": "0", "flags": "O_RDONLY"},
    ts_unix_nano=1_700_000_000_000_000_000,
  )
  # zscore should use the frequency view (hashed categoricals, day_fraction_norm time feature).
  values, _ = extract_feature_dict(evt, feature_view=feature_view_for_algorithm("zscore"))
  assert "day_fraction_norm" in values
  assert "day_cycle_sin" not in values
  assert "pid_norm" not in values
  assert "flags_hash" not in values
  assert "path_hash" in values


def test_extract_feature_dict_frequency_view_keeps_type_specific_hashes():
  file_evt = _evt(
    event_id="feat-file-frequency",
    event_group="file",
    syscall_nr=257,
    path="/etc/passwd",
    attributes={"flags": "O_RDONLY|O_CLOEXEC"},
  )
  file_values, _ = extract_feature_dict(file_evt, feature_view="frequency")
  assert "file_sensitive_path" not in file_values
  assert "file_tmp_path" not in file_values
  assert "file_flags_hash" not in file_values
  assert "group_path_rate_5" not in file_values
  assert "path_hash" in file_values

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
  net_values, _ = extract_feature_dict(net_evt, feature_view="frequency")
  assert "net_socket_type_hash" in net_values
  assert "net_pair_rate_5" not in net_values
  assert "hostname_hash" in net_values


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert "pid_norm" in values
  assert len(values) == len(extract_feature_dict(base_evt, feature_view="full")[0])


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
  values, _ = extract_feature_dict(evt, feature_view="memstream")
  assert "pid_hash" not in values
  assert "file_sensitive_path" in values
  assert "group_path_rate_5" not in values
  assert "group_syscall_openat" not in values
  assert _count_prefix(values, "comm_bucket_") == 0
  assert _count_prefix(values, "hostname_bucket_") == 0
  assert _count_prefix(values, "file_ext_bucket_") == 0
  assert _count_prefix(values, "file_flags_bucket_") == 0
  assert "comm_hash" not in values


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
  values, _ = extract_feature_dict(evt, feature_view="full")
  assert "net_socket_family_norm" in values
  assert _count_prefix(values, "group_syscall_") == 0
  assert _count_prefix(values, "net_socket_type_bucket_") == 0
  assert _count_prefix(values, "net_daddr_bucket_") == 0
  assert _count_prefix(values, "comm_bucket_") == 0


def test_frequency_feature_view_matches_expected_file_schema():
  evt = _evt(
    event_id="feat-frequency-view",
    event_group="file",
    syscall_nr=257,
    hostname="node-1",
    path="/etc/passwd",
    attributes={"return_value": "0", "flags": "O_RDONLY|O_CLOEXEC"},
  )
  values, _ = extract_feature_dict(evt, feature_view="frequency")
  assert "flags_hash" not in values
  assert "path_hash" in values
  assert "file_sensitive_path" not in values
  assert "group_syscall_hash" not in values
  assert "file_flags_hash" not in values
  assert "proc_rate_1" not in values
  assert "event_name_openat" not in values
  assert _count_prefix(values, "comm_bucket_") == 0
  assert _count_prefix(values, "file_flags_bucket_") == 0


def test_custom_event_group_uses_group_syscalls_from_rules(temp_dir, monkeypatch):
  """Arbitrary group name; syscall vocabulary comes from groups.<name>.syscalls in rules.yaml."""
  from detector.building_blocks.primitives import features as feat_mod

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
    # Custom groups currently don't have a dedicated feature layer; only file/network/process do.
    values, _ = extract_feature_dict(evt, feature_view="default")
    # Default view still includes the always-on syscall number + time signal.
    assert set(values.keys()) == {"syscall_nr_norm", "day_cycle_sin", "day_cycle_cos"}
  finally:
    feat_mod._detector_rules_path.cache_clear()
    feat_mod._rules_config.cache_clear()
