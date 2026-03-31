#!/usr/bin/env python3
"""Convert LID-DS 2021 recordings to detector JSONL and optionally replay.

Output matches the live probe / EventEnvelope layout: resolved fd metadata uses
``attributes["fd_path"]``, ``attributes["fd_sock_*"]``, ``attributes["fd_resource_kind"]``,
and optional ``attributes["fd_sock_family"]`` (no ``attributes.path``, ``sin_*``, or ``dest_*``).

Strace-style fd parameters like ``36(<f>/etc/hosts)`` are split: the numeric fd is
``arg0`` / ``arg1`` (matching the probe); the file path goes to ``fd_path``.
Strace inet socket decorations ``<4t>``, ``<4u>``, ``<6t>``, ``<6u>``, and other
``<digits><letter>`` forms with ``local:port->remote:port`` populate ``fd_sock_*``;
``t`` → ``fd_resource_kind`` ``tcp``, ``u`` → ``udp``, other letters → ``unknown``.
``<6…>`` sets ``fd_sock_family`` to ``AF_INET6`` when not already present; ``<4…>`` to ``AF_INET``.
``<p>`` implies ``fd_resource_kind`` ``pipe``.

When LID-DS encodes a file inside ``fd=…(<f>/path)`` (or ``in_fd`` / ``out_fd``),
order is ``fd``, ``in_fd``, ``out_fd``, then ``arg0``; first ``<f>`` path wins.
Socket tuples do not set ``fd_path``.

LID-DS-only extensions (top-level, **after** the core envelope; omitted by
``envelope_from_dict`` / gRPC replay so the detector never sees them):

- ``lidds_recording_name``, ``lidds_recording_path``, ``lidds_recording_line_id``
- ``malicious`` — from ``exploit`` + ``time.exploit[*].absolute`` vs ``ts_unix_nano``
- ``lidds_sock_local_role``, ``lidds_sock_remote_role`` — when ``fd_sock_*_addr`` matches
  a ``container[].ip`` role (``attacker``, ``victim``, ``normal``, …)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))
_DEFAULT_LIDDS_ROOT = _ROOT / "third_party" / "LID-DS"
# Default rules for syscall -> event_group during conversion only (not the chart / probe rules file).
_DEFAULT_REPLAY_LIDD_RULES_PATH = Path(__file__).resolve().parent / "replay_lidds_rules.yaml"

from probe.events import EVENT_NAME_TO_ID, UNKNOWN_EVENT_ID
from rules_config import load_rules_config
from scripts.replay_logs import replay


log = logging.getLogger(__name__)

_NETWORK_EVENTS = frozenset({"socket", "connect", "bind", "listen", "accept", "accept4"})
_FILE_EVENTS = frozenset({"open", "openat", "openat2", "unlink", "unlinkat", "rename", "renameat", "chmod", "chown", "read", "write"})
_PROCESS_EVENTS = frozenset({"execve", "fork", "clone"})

_PATH_PARAM_KEYS = (
  "pathname",
  "filename",
  "path",
  "name",
  "target",
  "source",
  "oldname",
  "newname",
)

_LIDDS_FD_FILE_TAG = "<f>"
_LIDDS_FD_PIPE_TAG = "<p>"
# Strace prints AF_INET TCP/UDP as <4t>/<4u>; AF_INET6 as <6t>/<6u>; same host:port->host:port shape.
_STRACE_INET_SOCK_TAG_RE = re.compile(r"<(\d+)([a-zA-Z])>")
# One path, same as probe; first param with a <f>… filepath wins (see module docstring).
_LIDDS_FD_PARAM_KEYS = ("fd", "in_fd", "out_fd", "arg0")
_RETURN_PARAM_KEYS = ("res", "ret", "retval", "return", "return_value")
_ARG0_KEYS = ("arg0", "fd", "dirfd", "domain", "flags", "oldfd")
_ARG1_KEYS = ("arg1", "cmd", "flags", "type", "addrlen", "newfd")


def _safe_str(value: Any, default: str = "") -> str:
  if value is None:
    return default
  return str(value)


def _wait_for_detector(target: str, timeout_s: float) -> None:
  """Wait for gRPC health to become SERVING."""
  try:
    import grpc  # local import to avoid hard dependency for conversion-only usage
    from grpc_health.v1 import health_pb2, health_pb2_grpc
  except Exception as exc:  # pragma: no cover
    raise RuntimeError("grpc + grpcio-health-checking are required to manage a detector process") from exc

  deadline = time.time() + timeout_s
  last_error = ""
  while time.time() < deadline:
    try:
      channel = grpc.insecure_channel(target)
      stub = health_pb2_grpc.HealthStub(channel)
      resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=2.0)  # pyright: ignore[reportAttributeAccessIssue]
      if resp.status == health_pb2.HealthCheckResponse.SERVING:
        return
      last_error = f"health status={resp.status}"
    except Exception as exc:
      last_error = str(exc)
    time.sleep(0.5)
  raise RuntimeError(f"detector at {target} did not become ready: {last_error}")


def _start_detector(
  *,
  port: int,
  env_overrides: dict[str, str] | None,
  quiet: bool,
) -> subprocess.Popen:
  env = os.environ.copy()
  env["DETECTOR_PORT"] = str(port)
  env["DETECTOR_EVENTS_PORT"] = str(port + 1)  # avoid HTTP port collision when using custom gRPC port
  if quiet:
    env["DETECTOR_QUIET"] = "1"
  if env_overrides:
    env.update(env_overrides)
  cmd = [sys.executable, "-m", "detector.server"]
  return subprocess.Popen(
    cmd,
    env=env,
    stdout=subprocess.DEVNULL if quiet else None,
    stderr=subprocess.DEVNULL if quiet else None,
  )


def _stop_detector(proc: subprocess.Popen) -> None:
  if proc.poll() is not None:
    return
  proc.terminate()
  try:
    proc.wait(timeout=10)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait(timeout=5)


def _maybe_b64_decode(value: str) -> str:
  raw = (value or "").strip()
  if not raw:
    return ""
  if len(raw) % 4 != 0:
    return raw
  try:
    decoded = base64.b64decode(raw, validate=True)
  except Exception:
    return raw
  try:
    text = decoded.decode("utf-8")
  except UnicodeDecodeError:
    return raw
  # Keep raw token if decode result looks like mostly non-printable bytes.
  if text and sum(ch.isprintable() for ch in text) / float(len(text)) > 0.8:
    return text
  return raw


def _strip_lidds_strace_fd_suffix(value: str) -> str:
  """Strip strace class suffix from fd tokens: ``36(<f>/p)`` → ``36``.

  Values like ``5(F_SETFL)`` are left unchanged (parenthesis is not followed by ``<``).
  """
  raw = (value or "").strip()
  if not raw:
    return raw
  j = 0
  if raw[0] == "-":
    j = 1
  start = j
  while j < len(raw) and raw[j].isdigit():
    j += 1
  if j > start and j + 1 < len(raw) and raw[j] == "(" and raw[j + 1] == "<":
    return raw[:j]
  return raw


def _strip_lidds_strace_numeric_suffix(value: str) -> str:
  """Strip any parenthesized suffix from a numeric token.

  Examples:
    - ``-1(EPERM)`` → ``-1``
    - ``74(MAP_PRIVATE|MAP_ANONYMOUS)`` → ``74``
    - ``36(<4u>127.0.0.1:1->127.0.0.2:2)`` → ``36``
  """
  raw = (value or "").strip()
  if not raw:
    return raw
  j = 0
  if raw[0] == "-":
    j = 1
  start = j
  while j < len(raw) and raw[j].isdigit():
    j += 1
  if j > start and j < len(raw) and raw[j] == "(":
    return raw[:j]
  return raw


def _syscall_group_map(rules_path: Path) -> dict[str, str]:
  cfg = load_rules_config(rules_path)
  out: dict[str, str] = {}
  for rule in cfg.rules:
    if not rule.enabled:
      continue
    gcfg = cfg.groups[rule.group]
    for name in gcfg.syscalls:
      n = (name or "").strip().lower()
      if n and n not in out:
        out[n] = rule.group
  return out


def infer_event_group(event_name: str, syscall_to_group: dict[str, str]) -> str:
  name = (event_name or "").strip().lower()
  if not name:
    return ""
  if name in syscall_to_group:
    return syscall_to_group[name]
  if name in _NETWORK_EVENTS:
    return "network"
  if name in _FILE_EVENTS:
    return "file"
  if name in _PROCESS_EVENTS:
    return "process"
  return ""


def _pick_param(params: dict[str, str], *keys: str) -> str:
  for key in keys:
    if key in params:
      return _safe_str(params[key])
  return ""


def _event_path(params: dict[str, str]) -> str:
  for key in _PATH_PARAM_KEYS:
    if key in params:
      return _maybe_b64_decode(_safe_str(params[key]))
  return ""


def _path_from_lidds_fd_param(value: str) -> str:
  """Extract `/path` from strace-style `9(<f>/etc/passwd)` when present."""
  raw = (value or "").strip()
  i = raw.find(_LIDDS_FD_FILE_TAG)
  if i < 0:
    return ""
  start = i + len(_LIDDS_FD_FILE_TAG)
  end = raw.rfind(")")
  if end > start:
    return raw[start:end]
  return raw[start:]


def _path_from_lidds_fd_params(params: dict[str, str]) -> str:
  for key in _LIDDS_FD_PARAM_KEYS:
    if key not in params:
      continue
    extracted = _path_from_lidds_fd_param(_safe_str(params[key]))
    if extracted:
      return extracted
  return ""


def _resolved_fd_path(params: dict[str, str]) -> str:
  """VFS path for ``attributes['fd_path']``: explicit path params, else first <f> fd decoration."""
  p = _event_path(params)
  if p:
    return p
  return _path_from_lidds_fd_params(params)


def _host_port_pair(blob: str) -> tuple[str, str]:
  """Parse ``host:port`` for IPv4 (last ``:`` separates port)."""
  s = blob.strip()
  if not s or ":" not in s:
    return "", ""
  host, _, port = s.rpartition(":")
  return host.strip(), port.strip()


def _host_port_pair_inet(blob: str) -> tuple[str, str]:
  """Parse one side of a strace socket tuple (IPv4 or ``[ipv6]:port``)."""
  s = blob.strip()
  if not s:
    return "", ""
  if s.startswith("["):
    close = s.find("]")
    if close <= 1:
      return "", ""
    host = s[1:close].strip()
    rest = s[close + 1 :].strip()
    if rest.startswith(":"):
      return host, rest[1:].strip()
    return host, ""
  return _host_port_pair(s)


def _strace_socket_proto_kind(proto_letter: str) -> str:
  """Map strace socket class letter to ``fd_resource_kind`` (or ``unknown``)."""
  p = (proto_letter or "").lower()
  if p == "t":
    return "tcp"
  if p == "u":
    return "udp"
  return "unknown"


def _parse_lidds_strace_socket_tuple(value: str) -> tuple[dict[str, str], str]:
  """Parse first ``…(<4t>…->…)`` / ``<4u>`` / ``<6u>`` / … decoration into ``fd_sock_*`` and kind.

  Returns ``(attrs, kind)`` where ``kind`` is ``tcp``, ``udp``, ``unknown``, or ``""`` if none matched.
  """
  raw = (value or "").strip()
  search = 0
  while search < len(raw):
    i = raw.find("(<", search)
    if i < 0:
      return {}, ""
    m = _STRACE_INET_SOCK_TAG_RE.match(raw, i + 1)
    if not m:
      search = i + 1
      continue
    family_digit = m.group(1)
    proto_letter = m.group(2)
    tag_end = m.end()
    end_paren = raw.rfind(")")
    blob = raw[tag_end:end_paren].strip() if end_paren > tag_end else raw[tag_end:].strip()
    if "->" not in blob:
      search = i + 1
      continue
    left, _, right = blob.partition("->")
    la, lp = _host_port_pair_inet(left)
    ra, rp = _host_port_pair_inet(right)
    if not (la or lp or ra or rp):
      search = i + 1
      continue
    out: dict[str, str] = {}
    if la:
      out["fd_sock_local_addr"] = la
    if lp:
      out["fd_sock_local_port"] = lp
    if ra:
      out["fd_sock_remote_addr"] = ra
    if rp:
      out["fd_sock_remote_port"] = rp
    if family_digit == "6":
      out["fd_sock_family"] = "AF_INET6"
    elif family_digit == "4":
      out["fd_sock_family"] = "AF_INET"
    kind = _strace_socket_proto_kind(proto_letter)
    return out, kind
  return {}, ""


def _params_contain_pipe_decoration(params: dict[str, str]) -> bool:
  for key in _LIDDS_FD_PARAM_KEYS:
    v = params.get(key)
    if v and _LIDDS_FD_PIPE_TAG in _safe_str(v):
      return True
  return False


def _merge_strace_socket_attrs_from_params(params: dict[str, str], attrs: dict[str, str]) -> None:
  """Fill missing ``fd_sock_*`` / ``fd_resource_kind`` from strace inet socket decorations."""
  for key in _LIDDS_FD_PARAM_KEYS:
    v = params.get(key)
    if not v:
      continue
    sock, kind = _parse_lidds_strace_socket_tuple(_safe_str(v))
    for k, val in sock.items():
      if val and not attrs.get(k):
        attrs[k] = val
    if kind and not attrs.get("fd_resource_kind"):
      attrs["fd_resource_kind"] = kind


def _finalize_fd_resource_kind(
  event_name: str,
  attrs: dict[str, str],
  params: dict[str, str],
) -> None:
  if attrs.get("fd_resource_kind"):
    return
  has_sock = any(
    (attrs.get(k) or "").strip()
    for k in (
      "fd_sock_local_addr",
      "fd_sock_local_port",
      "fd_sock_remote_addr",
      "fd_sock_remote_port",
    )
  )
  if has_sock:
    # LID-DS param-based connect info without strace decoration; assume TCP-style stream socket.
    attrs["fd_resource_kind"] = "tcp"
    return
  if (attrs.get("fd_path") or "").strip():
    attrs["fd_resource_kind"] = "file"
    return
  if _params_contain_pipe_decoration(params):
    attrs["fd_resource_kind"] = "pipe"
    return
  name = (event_name or "").strip().lower()
  if name in _NETWORK_EVENTS:
    attrs["fd_resource_kind"] = "unknown"
    return
  attrs["fd_resource_kind"] = "unknown"


def _extract_return_value(params: dict[str, str], direction_char: str) -> str:
  if direction_char != "<":
    return ""
  for key in _RETURN_PARAM_KEYS:
    if key in params:
      return _safe_str(params[key], "0")
  return ""


def _extract_arg0(event_name: str, params: dict[str, str]) -> str:
  if "arg0" in params:
    return _safe_str(params["arg0"])
  name = (event_name or "").strip().lower()
  if name in {"socket"}:
    return _pick_param(params, "domain", "family", "fd", "dirfd")
  if name in {"open"}:
    return _pick_param(params, "flags", "fd", "dirfd")
  if name in {"openat", "openat2"}:
    return _pick_param(params, "dirfd", "fd")
  return _pick_param(params, *_ARG0_KEYS)


def _extract_arg1(event_name: str, params: dict[str, str]) -> str:
  if "arg1" in params:
    return _safe_str(params["arg1"])
  name = (event_name or "").strip().lower()
  if name in {"connect", "bind", "accept", "accept4", "listen"}:
    return _pick_param(params, "addrlen", "backlog", "type")
  if name == "socket":
    return _pick_param(params, "type", "flags")
  if name in {"openat", "openat2"}:
    return _pick_param(params, "flags", "mode")
  if name == "open":
    return _pick_param(params, "mode")
  return _pick_param(params, *_ARG1_KEYS)


def _extract_attrs(event_name: str, params: dict[str, str], direction_char: str) -> dict[str, str]:
  attrs: dict[str, str] = {}
  ret = _extract_return_value(params, direction_char)
  if ret:
    attrs["return_value"] = ret

  flags = _pick_param(params, "flags")
  if flags:
    attrs["flags"] = flags
  else:
    # Some LID-DS exports carry human-readable flag decorations in arg fields.
    # Keep them as attributes.flags while args stay numeric-stripped.
    name = (event_name or "").strip().lower()
    if name == "mmap":
      f = _pick_param(params, "arg1")
      if f and "(" in f:
        attrs["flags"] = f

  remote_port = _pick_param(params, "sin_port", "dest_port")
  if remote_port:
    attrs["fd_sock_remote_port"] = remote_port
  remote_addr = _pick_param(params, "sin_addr", "dest_ip")
  if remote_addr:
    attrs["fd_sock_remote_addr"] = remote_addr
  local_port = _pick_param(params, "sin_local_port", "src_port")
  if local_port:
    attrs["fd_sock_local_port"] = local_port
  local_addr = _pick_param(params, "sin_local_addr", "src_ip", "src_addr")
  if local_addr:
    attrs["fd_sock_local_addr"] = local_addr
  fam = _pick_param(params, "sa_family", "family")
  if fam:
    attrs["fd_sock_family"] = fam
  return attrs


def _lidds_event_malicious_from_metadata(ts_unix_nano: int, meta: dict[str, Any] | None) -> bool | None:
  """Per-event label from LID-DS ``<recording>.json`` (see ``Recording2021.metadata()``).

  - ``exploit: false`` → benign for every syscall.
  - ``exploit: true`` with ``time.exploit[*].absolute`` (epoch seconds, float) → ``malicious`` iff
    ``ts_unix_nano >= min(absolute) * 1e9``.
  - ``exploit: true`` but no usable absolute exploit times → treat whole recording as malicious (attack-only).
  - Missing/invalid metadata → ``None`` (omit ``malicious`` on the envelope).
  """
  if not meta or not isinstance(meta, dict):
    return None
  if not meta.get("exploit"):
    return False
  time_block = meta.get("time")
  if not isinstance(time_block, dict):
    return True
  raw_exploits = time_block.get("exploit")
  if not isinstance(raw_exploits, list):
    raw_exploits = []
  starts_ns: list[int] = []
  for entry in raw_exploits:
    if not isinstance(entry, dict):
      continue
    abs_t = entry.get("absolute")
    if abs_t is None:
      continue
    try:
      starts_ns.append(int(round(float(abs_t) * 1_000_000_000)))
    except (TypeError, ValueError):
      continue
  if not starts_ns:
    return True
  return ts_unix_nano >= min(starts_ns)


def _normalize_ip_for_lidds_role_match(addr: str) -> str:
  """Normalize address for lookup against LID-DS ``container[].ip`` strings."""
  raw = (addr or "").strip()
  if not raw:
    return ""
  if raw.startswith("[") and "]" in raw:
    raw = raw[1 : raw.index("]")].strip()
  return raw.lower()


def _lidds_container_ip_to_role(meta: dict[str, Any] | None) -> dict[str, str]:
  """Map normalized IP → lowercase ``container[].role``."""
  if not meta or not isinstance(meta, dict):
    return {}
  out: dict[str, str] = {}
  for c in meta.get("container") or []:
    if not isinstance(c, dict):
      continue
    ip = _normalize_ip_for_lidds_role_match(str(c.get("ip") or ""))
    role = str(c.get("role") or "").strip().lower()
    if ip and role:
      out[ip] = role
  return out


def _lidds_sock_endpoint_roles(attrs: dict[str, str], meta: dict[str, Any] | None) -> tuple[str, str]:
  """Return ``(local_role, remote_role)`` from ``fd_sock_*_addr`` vs metadata containers."""
  m = _lidds_container_ip_to_role(meta)
  if not m:
    return "", ""
  loc = _normalize_ip_for_lidds_role_match(str(attrs.get("fd_sock_local_addr") or ""))
  rem = _normalize_ip_for_lidds_role_match(str(attrs.get("fd_sock_remote_addr") or ""))
  return (m.get(loc, "") if loc else "", m.get(rem, "") if rem else "")


def convert_syscall_to_envelope(
  syscall: Any,
  *,
  event_id: str,
  hostname: str,
  syscall_to_group: dict[str, str],
  recording_name: str = "",
  recording_path: str = "",
  lidds_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
  event_name = _safe_str(syscall.name()).strip().lower()
  params_raw = syscall.params() or {}
  params = {_safe_str(k): _safe_str(v) for k, v in params_raw.items()}
  direction_char = ">" if _safe_str(syscall.direction()) == "Direction.OPEN" else "<"
  if hasattr(syscall.direction(), "name"):
    direction_char = ">" if syscall.direction().name == "OPEN" else "<"

  path = _resolved_fd_path(params)
  arg0_raw = _extract_arg0(event_name, params)
  arg1_raw = _extract_arg1(event_name, params)
  env: dict[str, Any] = {
    "event_id": event_id,
    "ts_unix_nano": int(syscall.timestamp_unix_in_ns()),
    "hostname": hostname,
    "syscall_name": event_name,
    "event_group": infer_event_group(event_name, syscall_to_group),
    "syscall_nr": int(EVENT_NAME_TO_ID.get(event_name, UNKNOWN_EVENT_ID)),
    "comm": _safe_str(syscall.process_name()),
    "pid": _safe_str(syscall.process_id(), "0"),
    "tid": _safe_str(syscall.thread_id(), "0"),
    "uid": _safe_str(syscall.user_id(), "0"),
    "arg0": _strip_lidds_strace_numeric_suffix(_strip_lidds_strace_fd_suffix(arg0_raw)),
    "arg1": _strip_lidds_strace_numeric_suffix(_strip_lidds_strace_fd_suffix(arg1_raw)),
    "attributes": _extract_attrs(event_name, params, direction_char),
  }
  if not isinstance(env["attributes"], dict):
    env["attributes"] = {}
  attrs = env["attributes"]
  _merge_strace_socket_attrs_from_params(params, attrs)
  if path:
    attrs["fd_path"] = str(path)
  _finalize_fd_resource_kind(event_name, attrs, params)
  mal = _lidds_event_malicious_from_metadata(int(env["ts_unix_nano"]), lidds_metadata)
  loc_role, rem_role = _lidds_sock_endpoint_roles(attrs, lidds_metadata)

  if recording_name:
    env["lidds_recording_name"] = str(recording_name)
  if recording_path:
    env["lidds_recording_path"] = str(recording_path)
  if getattr(syscall, "line_id", None) is not None:
    env["lidds_recording_line_id"] = str(getattr(syscall, "line_id"))
  if mal is not None:
    env["malicious"] = bool(mal)
  if loc_role:
    env["lidds_sock_local_role"] = loc_role
  if rem_role:
    env["lidds_sock_remote_role"] = rem_role
  return env


def _lidds_recordings(
  scenario_path: Path,
  split: str,
  recording_type: str | None,
  lidds_root: Path | None = None,
) -> list[Any]:
  if lidds_root is not None:
    root_str = str(lidds_root.resolve())
    if root_str not in sys.path:
      sys.path.insert(0, root_str)
  try:
    from dataloader.dataloader_factory import dataloader_factory  # pyright: ignore[reportMissingImports]
    from dataloader.data_loader_2021 import RecordingType  # pyright: ignore[reportMissingImports]
  except ImportError as exc:
    raise RuntimeError(
      "LID-DS dataloader not available. Install/import LID-DS first "
      "(https://github.com/LID-DS/LID-DS)."
    ) from exc

  split_name = split.strip().lower()
  loader = dataloader_factory(str(scenario_path))
  func = {
    "training": loader.training_data,
    "validation": loader.validation_data,
    "test": loader.test_data,
  }.get(split_name)
  if func is None:
    raise ValueError("split must be one of: training, validation, test")

  type_obj = None
  if recording_type:
    normalized = recording_type.strip().upper()
    try:
      type_obj = RecordingType[normalized]
    except KeyError as exc:
      valid = ", ".join(sorted(m.name for m in RecordingType))
      raise ValueError(f"invalid recording type {recording_type!r}; choose from: {valid}") from exc

  return list(func(recording_type=type_obj))


def _iter_lidds_syscalls_from_recordings(
  recordings: list[Any],
) -> Iterable[tuple[str, str, dict[str, Any] | None, Any]]:
  for rec in recordings:
    rec_name = getattr(rec, "name", "")
    rec_path = getattr(rec, "path", "")
    meta: dict[str, Any] | None = None
    md = getattr(rec, "metadata", None)
    if callable(md):
      try:
        raw_meta = md()
        meta = raw_meta if isinstance(raw_meta, dict) else None
      except Exception as exc:
        log.warning("LID-DS metadata() failed for %s: %s", rec_path, exc)
        meta = None
    for syscall in rec.syscalls():
      yield rec_name, rec_path, meta, syscall


def _iter_lidds_syscalls(
  scenario_path: Path,
  split: str,
  recording_type: str | None,
  lidds_root: Path | None = None,
) -> Iterable[tuple[str, str, dict[str, Any] | None, Any]]:
  yield from _iter_lidds_syscalls_from_recordings(
    _lidds_recordings(scenario_path, split, recording_type, lidds_root),
  )


def convert_lidds_to_jsonl(
  *,
  scenario_path: Path,
  split: str,
  out_jsonl: Path,
  rules_path: Path,
  recording_type: str | None = None,
  hostname: str = "lidds",
  event_id_prefix: str = "lidds",
  max_events: int | None = None,
  lidds_root: Path | None = None,
) -> int:
  syscall_to_group = _syscall_group_map(rules_path)
  out_jsonl.parent.mkdir(parents=True, exist_ok=True)

  recordings = _lidds_recordings(scenario_path, split, recording_type, lidds_root)
  n = 0
  syscall_iter = _iter_lidds_syscalls_from_recordings(recordings)
  if tqdm is not None:
    syscall_iter = tqdm(
      syscall_iter,
      desc=f"Convert LID-DS ({split})",
      unit=" evt",
      file=sys.stderr,
    )
  with out_jsonl.open("w", encoding="utf-8") as f:
    for recording_name, recording_path, meta, syscall in syscall_iter:
      if max_events is not None and n >= max_events:
        break
      event_id = f"{event_id_prefix}-{split}-{n}"
      env = convert_syscall_to_envelope(
        syscall,
        event_id=event_id,
        hostname=hostname,
        syscall_to_group=syscall_to_group,
        recording_name=recording_name,
        recording_path=recording_path,
        lidds_metadata=meta,
      )
      f.write(json.dumps(env, separators=(",", ":")) + "\n")
      n += 1
  return n


def _build_arg_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(
    description="Convert LID-DS 2021 recordings to detector JSONL and optionally replay via scripts/replay_logs.py.",
  )
  ap.add_argument("--scenario-path", required=True, help="Path to LID-DS scenario directory.")
  ap.add_argument("--split", choices=["training", "validation", "test"], default="test")
  ap.add_argument(
    "--recording-type",
    default=None,
    help="Optional LID-DS RecordingType filter (e.g. NORMAL, NORMAL_AND_ATTACK, ATTACK, IDLE).",
  )
  ap.add_argument("--out-jsonl", required=True, help="Output detector-compatible JSONL path.")
  ap.add_argument(
    "--rules-path",
    default=str(_DEFAULT_REPLAY_LIDD_RULES_PATH),
    help="Rules YAML for syscall->event_group mapping during conversion (default: scripts/replay_lidds_rules.yaml).",
  )
  ap.add_argument("--hostname", default="lidds")
  ap.add_argument("--event-id-prefix", default="lidds")
  ap.add_argument(
    "--lidds-root",
    default=os.environ.get("LID_DS_ROOT", "").strip() or str(_DEFAULT_LIDDS_ROOT),
    help="Path to local LID-DS repository root containing the `dataloader` package. "
    "Defaults to third_party/LID-DS in this repo; can be overridden via LID_DS_ROOT.",
  )
  ap.add_argument("--max-convert-events", type=int, default=None, help="Stop conversion after N events.")

  ap.add_argument("--target", default="localhost:50051", help="Detector gRPC target for replay.")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast")
  ap.add_argument("--start-ms", type=int, default=None)
  ap.add_argument("--end-ms", type=int, default=None)
  ap.add_argument("--max-events", type=int, default=None, help="Max events to replay (after conversion).")
  ap.add_argument("--skip", type=int, default=None)
  ap.add_argument("--convert-only", action="store_true", help="Only convert to JSONL; do not replay.")

  ap.add_argument(
    "--start-detector",
    action="store_true",
    help="Start a local detector subprocess for replay. Overrides --target to localhost:<--detector-port>.",
  )
  ap.add_argument("--detector-port", type=int, default=50051, help="Detector gRPC port when using --start-detector.")
  ap.add_argument(
    "--detector-algorithm",
    default="grimmer_mlp",
    help="Value for DETECTOR_MODEL_ALGORITHM when using --start-detector (default: grimmer_mlp).",
  )
  ap.add_argument(
    "--event-dump-path",
    default=None,
    help="When using --start-detector, set EVENT_DUMP_PATH so detector writes JSONL entries with scores/anomaly flags.",
  )
  ap.add_argument(
    "--detector-startup-timeout",
    type=float,
    default=30.0,
    help="Seconds to wait for detector readiness when using --start-detector.",
  )
  ap.add_argument(
    "--save-checkpoint",
    default=None,
    help="After replay, write a detector checkpoint (.pkl) by training offline on the converted JSONL. Intended for --detector-algorithm=grimmer_mlp.",
  )
  return ap


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  args = _build_arg_parser().parse_args()

  scenario_path = Path(args.scenario_path)
  out_jsonl = Path(args.out_jsonl)
  rules_path = Path(args.rules_path)
  converted = convert_lidds_to_jsonl(
    scenario_path=scenario_path,
    split=args.split,
    out_jsonl=out_jsonl,
    rules_path=rules_path,
    recording_type=args.recording_type,
    hostname=args.hostname,
    event_id_prefix=args.event_id_prefix,
    max_events=args.max_convert_events,
    lidds_root=Path(args.lidds_root) if args.lidds_root else None,
  )
  log.info("Converted %d LID-DS events to %s", converted, out_jsonl)
  if args.convert_only:
    return

  detector_proc: subprocess.Popen | None = None
  target = args.target
  if args.start_detector:
    target = f"localhost:{int(args.detector_port)}"
    env_overrides: dict[str, str] = {"DETECTOR_MODEL_ALGORITHM": str(args.detector_algorithm)}
    if args.event_dump_path:
      env_overrides["EVENT_DUMP_PATH"] = str(Path(args.event_dump_path))
    log.info("Starting detector for replay (target=%s, overrides=%s)", target, env_overrides)
    detector_proc = _start_detector(port=int(args.detector_port), env_overrides=env_overrides, quiet=True)
    try:
      _wait_for_detector(target, timeout_s=float(args.detector_startup_timeout))
    except Exception:
      _stop_detector(detector_proc)
      raise

  try:
    replay(
      out_jsonl,
      target,
      args.pace,
      args.start_ms,
      args.end_ms,
      total=converted,
      label="Replay LID-DS",
      max_events=args.max_events,
      skip=args.skip,
    )
  finally:
    if detector_proc is not None:
      _stop_detector(detector_proc)

  if args.save_checkpoint:
    ckpt_path = Path(args.save_checkpoint)
    log.info("Saving checkpoint to %s (offline train from converted JSONL)...", ckpt_path)
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
      [
        sys.executable,
        "-m",
        "scripts.train_detector_checkpoint",
        str(out_jsonl),
        "--algorithm",
        str(args.detector_algorithm),
        "--out",
        str(ckpt_path),
      ],
      cwd=repo_root,
      check=True,
    )


if __name__ == "__main__":
  main()
