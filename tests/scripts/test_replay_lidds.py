"""Tests for scripts/replay_lidds.py."""

import json
from pathlib import Path

from scripts.replay_lidds import convert_lidds_to_jsonl, convert_syscall_to_envelope, infer_event_group


class _Direction:
  def __init__(self, name: str):
    self.name = name


class _FakeSyscall:
  def __init__(
    self,
    *,
    name: str,
    params: dict[str, str],
    ts_ns: int = 1_700_000_000_000_000_000,
    uid: int = 1000,
    pid: int = 1234,
    tid: int = 1234,
    comm: str = "nginx",
    direction: str = "CLOSE",
  ):
    self._name = name
    self._params = params
    self._ts_ns = ts_ns
    self._uid = uid
    self._pid = pid
    self._tid = tid
    self._comm = comm
    self._direction = _Direction(direction)

  def timestamp_unix_in_ns(self) -> int:
    return self._ts_ns

  def user_id(self) -> int:
    return self._uid

  def process_id(self) -> int:
    return self._pid

  def process_name(self) -> str:
    return self._comm

  def thread_id(self) -> int:
    return self._tid

  def name(self) -> str:
    return self._name

  def direction(self):
    return self._direction

  def params(self) -> dict[str, str]:
    return dict(self._params)


def test_infer_event_group_prefers_rule_map():
  assert infer_event_group("openat", {"openat": "network"}) == "network"


def test_convert_syscall_to_envelope_enriched_network():
  s = _FakeSyscall(
    name="connect",
    params={
      "fd": "3",
      "addrlen": "16",
      "sin_port": "443",
      "sin_addr": "10.1.2.3",
      "sa_family": "AF_INET",
      "res": "-111",
      "flags": "0",
    },
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(
    s,
    event_id="lidds-test-1",
    hostname="h1",
    syscall_to_group={"connect": "network"},
  )
  assert env["event_id"] == "lidds-test-1"
  assert env["syscall_name"] == "connect"
  assert env["event_group"] == "network"
  assert env["syscall_nr"] == 42
  assert env["arg0"] == "3"
  assert env["arg1"] == "16"
  assert env["attributes"]["return_value"] == "-111"
  assert env["attributes"]["fd_sock_remote_port"] == "443"
  assert env["attributes"]["fd_sock_remote_addr"] == "10.1.2.3"
  assert env["attributes"]["fd_sock_family"] == "AF_INET"
  assert env["attributes"]["fd_resource_kind"] == "tcp"


def test_path_from_fd_params_prefers_in_fd_before_out_fd():
  s = _FakeSyscall(
    name="read",
    params={
      "in_fd": "1(<f>/first)",
      "out_fd": "2(<f>/second)",
    },
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={})
  assert env["attributes"]["fd_path"] == "/first"
  assert env["attributes"]["fd_resource_kind"] == "file"


def test_path_skips_non_file_fd_then_uses_next_param():
  s = _FakeSyscall(
    name="read",
    params={"fd": "36(<4t>172.17.0.1:80->172.17.0.2:443)", "in_fd": "1(<f>/tmp/x)"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={})
  assert env["attributes"]["fd_path"] == "/tmp/x"
  assert env["attributes"]["fd_sock_local_addr"] == "172.17.0.1"
  assert env["attributes"]["fd_sock_local_port"] == "80"
  assert env["attributes"]["fd_sock_remote_addr"] == "172.17.0.2"
  assert env["attributes"]["fd_sock_remote_port"] == "443"
  assert env["attributes"]["fd_resource_kind"] == "tcp"
  assert env["attributes"]["fd_sock_family"] == "AF_INET"
  assert env["arg0"] == "36"


def test_sendto_strace_4u_dns_tuple():
  s = _FakeSyscall(
    name="sendto",
    params={"arg0": "36(<4u>127.0.0.1:32842->127.0.0.11:53)"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={"sendto": "network"})
  assert env["arg0"] == "36"
  assert env["attributes"]["fd_sock_local_addr"] == "127.0.0.1"
  assert env["attributes"]["fd_sock_local_port"] == "32842"
  assert env["attributes"]["fd_sock_remote_addr"] == "127.0.0.11"
  assert env["attributes"]["fd_sock_remote_port"] == "53"
  assert env["attributes"]["fd_sock_family"] == "AF_INET"
  assert env["attributes"]["fd_resource_kind"] == "udp"


def test_strace_6u_ipv6_udp_tuple():
  s = _FakeSyscall(
    name="sendto",
    params={"fd": "3(<6u>[::1]:32842->[2001:db8::11]:53)"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={"sendto": "network"})
  assert env["attributes"]["fd_sock_local_addr"] == "::1"
  assert env["attributes"]["fd_sock_remote_addr"] == "2001:db8::11"
  assert env["attributes"]["fd_sock_remote_port"] == "53"
  assert env["attributes"]["fd_sock_family"] == "AF_INET6"
  assert env["attributes"]["fd_resource_kind"] == "udp"


def test_strace_unknown_proto_letter_tuple_sets_unknown_kind():
  s = _FakeSyscall(
    name="fcntl",
    params={"fd": "1(<4x>10.0.0.1:1->10.0.0.2:2)", "cmd": "0"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={"fcntl": "file"})
  assert env["attributes"]["fd_sock_remote_port"] == "2"
  assert env["attributes"]["fd_resource_kind"] == "unknown"


def test_mmap_strips_decorators_but_keeps_flags_attribute():
  s = _FakeSyscall(
    name="mmap",
    params={"arg0": "-1(EPERM)", "arg1": "74(MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={"mmap": "file"})
  assert env["arg0"] == "-1"
  assert env["arg1"] == "74"
  assert env["attributes"]["flags"] == "74(MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)"


def test_path_skips_socket_decorated_fd():
  s = _FakeSyscall(
    name="fcntl",
    params={"fd": "36(<4t>172.17.0.1:80->172.17.0.2:443)", "cmd": "5"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={"fcntl": "file"})
  assert "fd_path" not in env["attributes"]
  assert env["attributes"]["fd_sock_remote_port"] == "443"
  assert env["attributes"]["fd_resource_kind"] == "tcp"
  assert env["attributes"]["fd_sock_family"] == "AF_INET"
  assert env["arg0"] == "36"
  assert env["arg1"] == "5"


def test_convert_syscall_to_envelope_fcntl_path_from_fd_decoration():
  s = _FakeSyscall(
    name="fcntl",
    params={"fd": "9(<f>/etc/passwd)", "cmd": "3", "res": "0"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(
    s,
    event_id="evt-fcntl",
    hostname="h1",
    syscall_to_group={"fcntl": "file"},
  )
  assert env["attributes"]["fd_path"] == "/etc/passwd"
  assert env["arg0"] == "9"
  assert env["arg1"] == "3"
  assert env["syscall_name"] == "fcntl"
  assert "path" not in env["attributes"]


def test_fstat_numeric_arg0_and_path_in_attributes():
  s = _FakeSyscall(
    name="fstat",
    params={"fd": "36(<f>/etc/hosts)", "res": "0"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(
    s,
    event_id="evt-fstat",
    hostname="h1",
    syscall_to_group={"fstat": "file"},
  )
  assert env["arg0"] == "36"
  assert env["attributes"]["fd_path"] == "/etc/hosts"
  assert env["syscall_nr"] == 5


def test_convert_syscall_to_envelope_path_decode_openat():
  # "/tmp/demo" base64
  s = _FakeSyscall(
    name="openat",
    params={"dirfd": "-100", "pathname": "L3RtcC9kZW1v", "flags": "577", "res": "5"},
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(
    s,
    event_id="evt-2",
    hostname="h1",
    syscall_to_group={"openat": "file"},
  )
  assert env["event_group"] == "file"
  assert env["attributes"]["fd_path"] == "/tmp/demo"
  assert "path" not in env["attributes"]
  assert env["arg0"] == "-100"
  assert env["arg1"] == "577"
  assert isinstance(env["attributes"], dict)
  assert env["attributes"]["return_value"] == "5"


def test_convert_lidds_to_jsonl_writes_replay_compatible_rows(monkeypatch, temp_dir):
  repo = Path(__file__).resolve().parent.parent.parent
  rules_path = repo / "scripts" / "replay_lidds_rules.yaml"
  out = temp_dir / "lidds.jsonl"

  class _FakeRec:
    def __init__(self, name: str, path: str, meta: dict, syscall: _FakeSyscall) -> None:
      self.name = name
      self.path = path
      self._meta = meta
      self._syscall = syscall

    def metadata(self) -> dict:
      return self._meta

    def syscalls(self):
      yield self._syscall

  def _fake_recordings(_scenario_path, _split, _recording_type, _lidds_root=None):
    meta_benign = {"exploit": False, "time": {"exploit": []}}
    meta_attack = {
      "exploit": True,
      "time": {"exploit": [{"absolute": 1000.0, "name": "attack", "source": "TCPDUMP"}]},
    }
    return [
      _FakeRec(
        "rec-1",
        "/tmp/test/normal/rec-1.zip",
        meta_benign,
        _FakeSyscall(
          name="openat",
          params={"dirfd": "-100", "pathname": "L2V0Yy9wYXNzd2Q=", "flags": "0", "res": "3"},
          direction="CLOSE",
          ts_ns=500_000_000_000,
        ),
      ),
      _FakeRec(
        "rec-2",
        "/tmp/test/attack/rec-2.zip",
        meta_attack,
        _FakeSyscall(
          name="connect",
          params={"fd": "7", "addrlen": "16", "dest_port": "443", "dest_ip": "1.2.3.4", "res": "0"},
          direction="CLOSE",
          ts_ns=1_005_000_000_000,
        ),
      ),
    ]

  monkeypatch.setattr("scripts.replay_lidds._lidds_recordings", _fake_recordings)
  n = convert_lidds_to_jsonl(
    scenario_path=Path("/tmp/unused"),
    split="test",
    out_jsonl=out,
    rules_path=rules_path,
    recording_type=None,
    hostname="lidds-host",
    event_id_prefix="lidds",
  )
  assert n == 2

  rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
  assert len(rows) == 2
  for i, row in enumerate(rows):
    assert row["event_id"] == f"lidds-test-{i}"
    assert isinstance(row.get("attributes"), dict)
    assert row["hostname"] == "lidds-host"
    assert "ts_unix_nano" in row
    assert "malicious" in row
  assert rows[0]["lidds_recording_name"] == "rec-1"
  assert rows[0]["lidds_recording_path"] == "/tmp/test/normal/rec-1.zip"
  assert rows[0]["malicious"] is False
  assert rows[1]["attributes"]["fd_sock_remote_port"] == "443"
  assert rows[1]["attributes"]["fd_sock_remote_addr"] == "1.2.3.4"
  assert rows[1]["malicious"] is True


def test_malicious_false_before_exploit_timestamp():
  meta = {"exploit": True, "time": {"exploit": [{"absolute": 1000.0}]}}
  s = _FakeSyscall(name="read", params={"fd": "0"}, ts_ns=999_000_000_000, direction="CLOSE")
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={}, lidds_metadata=meta)
  assert env["malicious"] is False


def test_malicious_true_at_or_after_exploit_timestamp():
  meta = {"exploit": True, "time": {"exploit": [{"absolute": 1000.0}]}}
  s = _FakeSyscall(name="read", params={"fd": "0"}, ts_ns=1_000_000_000_000, direction="CLOSE")
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={}, lidds_metadata=meta)
  assert env["malicious"] is True


def test_malicious_true_when_exploit_flag_but_no_absolute_times():
  meta = {"exploit": True, "time": {"exploit": []}}
  s = _FakeSyscall(name="read", params={"fd": "0"}, ts_ns=100, direction="CLOSE")
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={}, lidds_metadata=meta)
  assert env["malicious"] is True


def test_malicious_omitted_when_no_metadata():
  s = _FakeSyscall(name="read", params={"fd": "0"}, direction="CLOSE")
  env = convert_syscall_to_envelope(s, event_id="e", hostname="h", syscall_to_group={})
  assert "malicious" not in env


def test_lidds_sock_roles_top_level_from_container_ips():
  meta = {
    "exploit": False,
    "time": {"exploit": []},
    "container": [
      {"ip": "10.0.0.1", "name": "a", "role": "attacker"},
      {"ip": "10.0.0.2", "name": "v", "role": "victim"},
    ],
  }
  s = _FakeSyscall(
    name="connect",
    params={
      "fd": "3",
      "addrlen": "16",
      "dest_ip": "10.0.0.2",
      "dest_port": "443",
      "sin_local_addr": "10.0.0.1",
      "sin_local_port": "12345",
      "res": "0",
    },
    direction="CLOSE",
  )
  env = convert_syscall_to_envelope(
    s,
    event_id="e",
    hostname="h",
    syscall_to_group={"connect": "network"},
    lidds_metadata=meta,
  )
  assert env["lidds_sock_local_role"] == "attacker"
  assert env["lidds_sock_remote_role"] == "victim"
  assert "lidds_sock_local_role" not in env["attributes"]
