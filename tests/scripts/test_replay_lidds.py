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
  assert env["event_name"] == "connect"
  assert env["event_group"] == "network"
  assert env["syscall_nr"] == 42
  assert env["arg0"] == "3"
  assert env["arg1"] == "16"
  assert env["attributes"]["return_value"] == "-111"
  assert env["attributes"]["sin_port"] == "443"
  assert env["attributes"]["sin_addr"] == "10.1.2.3"
  assert env["attributes"]["sa_family"] == "AF_INET"


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
  assert env["path"] == "/tmp/demo"
  assert env["arg0"] == "-100"
  assert env["arg1"] == "577"
  assert isinstance(env["attributes"], dict)
  assert env["attributes"]["return_value"] == "5"


def test_convert_lidds_to_jsonl_writes_replay_compatible_rows(monkeypatch, temp_dir):
  rules_path = Path("/home/felix/sentinel_ebpf/charts/sentinel-ebpf/rules.yaml")
  out = temp_dir / "lidds.jsonl"

  def _fake_iter(_scenario_path, _split, _recording_type, _lidds_root=None):
    yield _FakeSyscall(
      name="openat",
      params={"dirfd": "-100", "pathname": "L2V0Yy9wYXNzd2Q=", "flags": "0", "res": "3"},
      direction="CLOSE",
    )
    yield _FakeSyscall(
      name="connect",
      params={"fd": "7", "addrlen": "16", "dest_port": "443", "dest_ip": "1.2.3.4", "res": "0"},
      direction="CLOSE",
    )

  monkeypatch.setattr("scripts.replay_lidds._iter_lidds_syscalls", _fake_iter)
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
