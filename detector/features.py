"""Feature extraction for anomaly detection.

EventEnvelope is universal: event_name = syscall name (openat, connect, socket, …);
event_type = rule-defined category (network, file, process, or empty). We always add
general features; depending on event_type we add type-specific features. Final
feature vectors may have different sizes (different features) per event.
"""
import ast
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable

import numpy as np

import events_pb2


@dataclass
class _DecayedMoments:
  last_ts: float = 0.0
  has_last: bool = False
  weight: float = 0.0
  sum_v: float = 0.0
  sum_sq: float = 0.0


class _OnlineFeatureStats:
  """
  Generic online statistics layer with exponential time decay.

  This is a generalized utility: callers define a logical stream key (group + key),
  metric name, and value. The class tracks decayed moments for multiple windows.
  """

  def __init__(self, windows: Iterable[tuple[str, float]], max_states: int = 120_000):
    self._windows = tuple(windows)
    self._states: "OrderedDict[tuple[str, str, str, str], _DecayedMoments]" = OrderedDict()
    self._last_seen: "OrderedDict[tuple[str, str], float]" = OrderedDict()
    self._max_states = max_states

  def _evict_if_needed(self, store: OrderedDict, key) -> None:
    if key in store:
      store.move_to_end(key)
      return
    if len(store) >= self._max_states:
      store.popitem(last=False)

  def _get_state(self, group: str, metric: str, window_name: str, stream_key: str) -> _DecayedMoments:
    key = (group, metric, window_name, stream_key)
    self._evict_if_needed(self._states, key)
    if key not in self._states:
      self._states[key] = _DecayedMoments()
    return self._states[key]

  def _remember_last_seen(self, group: str, stream_key: str, ts_s: float) -> float:
    key = (group, stream_key)
    prev = self._last_seen.get(key, ts_s)
    self._evict_if_needed(self._last_seen, key)
    self._last_seen[key] = ts_s
    return prev

  def update_value(self, group: str, metric: str, stream_key: str, ts_s: float, value: float) -> Dict[str, tuple[float, float, float]]:
    """
    Update decayed moments and return per-window (mean, std, weight) for this stream.
    """
    out: Dict[str, tuple[float, float, float]] = {}
    v = float(value)
    for window_name, tau_s in self._windows:
      s = self._get_state(group, metric, window_name, stream_key)
      if s.has_last:
        dt = max(0.0, ts_s - s.last_ts)
        decay = float(np.exp(-dt / max(tau_s, 1e-6)))
        s.weight *= decay
        s.sum_v *= decay
        s.sum_sq *= decay
      s.last_ts = ts_s
      s.has_last = True
      s.weight += 1.0
      s.sum_v += v
      s.sum_sq += v * v
      if s.weight > 1e-9:
        mean = s.sum_v / s.weight
        var = max(0.0, (s.sum_sq / s.weight) - (mean * mean))
      else:
        mean = 0.0
        var = 0.0
      out[window_name] = (mean, float(np.sqrt(var)), s.weight)
    return out

  def update_interarrival(self, group: str, metric: str, stream_key: str, ts_s: float, dt_scale_s: float = 60.0) -> Dict[str, tuple[float, float, float]]:
    """
    Track inter-arrival time in a normalized/log-scaled form.
    """
    prev = self._remember_last_seen(group, stream_key, ts_s)
    dt = max(0.0, ts_s - prev)
    dt_norm = min(float(np.log1p(dt) / np.log1p(max(dt_scale_s, 1.0))), 1.0)
    return self.update_value(group, metric, stream_key, ts_s, dt_norm)


# Five decay windows to align with Kitsune-style time scales (short to long).
_ONLINE_WINDOWS = (("01", 0.1), ("1", 1.0), ("5", 5.0), ("30", 30.0), ("120", 120.0))
_ONLINE_STATS = _OnlineFeatureStats(_ONLINE_WINDOWS)


def _safe_int(raw: str, default: int = 0) -> int:
  try:
    return int(raw)
  except (ValueError, OverflowError, TypeError):
    return default


@lru_cache(maxsize=8192)
def _hash01(value: str) -> float:
  if not value:
    return 0.0
  digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
  return (int(digest, 16) % 10000) / 10000.0


def _path_components(path: str) -> list:
  """Return non-empty path components (e.g. /etc/passwd -> ['etc','passwd'])."""
  if not path:
    return []
  return [p for p in path.strip().strip("/").split("/") if p]


def _parse_sockaddr_from_evt(evt: events_pb2.EventEnvelope) -> Dict[str, str]:
  """
  Try to get destination port, address, and family from event (BETH or attributes).
  Returns dict with keys sin_port, sin_addr, sa_family (values may be empty).
  """
  out: Dict[str, str] = {"sin_port": "", "sin_addr": "", "sa_family": ""}
  attrs = dict(evt.attributes or {})
  out["sin_port"] = (attrs.get("sin_port") or attrs.get("dest_port") or "").strip()
  out["sin_addr"] = (attrs.get("sin_addr") or attrs.get("dest_ip") or "").strip()
  out["sa_family"] = (attrs.get("sa_family") or "").strip()

  # BETH sometimes puts sockaddr as string in data[7] (arg1)
  if not out["sin_port"] and not out["sin_addr"] and not out["sa_family"]:
    raw = (evt.data[7] if len(evt.data) > 7 else "").strip()
    if raw and raw.startswith("{"):
      try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
          out["sin_port"] = str(obj.get("sin_port", "") or "")
          out["sin_addr"] = str(obj.get("sin_addr", "") or "")
          out["sa_family"] = str(obj.get("sa_family", "") or "")
      except (ValueError, SyntaxError):
        pass
  return out


def _extract_feature_values(evt: events_pb2.EventEnvelope) -> tuple[float, ...]:
  """
  Compute normalized feature values from an EventEnvelope.

  Original 10 features (event/comm/path hashes, pid/tid/uid, arg0/arg1, hour/minute)
  plus extended features from attributes and path semantics for better anomaly signal.
  """
  # Canonical generic vector order:
  # [event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]
  event_name = (evt.event_name or "") or (evt.data[0] if len(evt.data) > 0 else "")
  event_id_str = evt.data[1] if len(evt.data) > 1 else "0"
  comm = evt.data[2] if len(evt.data) > 2 else ""
  pid_str = evt.data[3] if len(evt.data) > 3 else "0"
  tid_str = evt.data[4] if len(evt.data) > 4 else "0"
  uid_str = evt.data[5] if len(evt.data) > 5 else "0"
  arg0_str = evt.data[6] if len(evt.data) > 6 else "0"
  arg1_str = evt.data[7] if len(evt.data) > 7 else "0"
  path = (evt.data[8] if len(evt.data) > 8 else "").strip()
  flags = evt.data[9] if len(evt.data) > 9 else ""

  attrs = dict(evt.attributes or {})

  # --- Original features ---
  event_hash_normalized = _hash01(event_name)
  comm_hash_normalized = _hash01(comm)
  path_hash_normalized = _hash01(path)

  pid_val = max(0, _safe_int(pid_str, default=0))
  pid_normalized = min(pid_val / 4194304.0, 1.0)
  tid_val = max(0, _safe_int(tid_str, default=0))
  tid_normalized = min(tid_val / 4194304.0, 1.0)
  uid_val = max(0, _safe_int(uid_str, default=0))
  uid_normalized = min(uid_val / 4294967295.0, 1.0)

  arg0_val = abs(_safe_int(arg0_str, default=0))
  arg1_val = abs(_safe_int(arg1_str, default=0))
  arg0_norm = min(np.log1p(arg0_val) / 20.0, 1.0)
  arg1_norm = min(np.log1p(arg1_val) / 20.0, 1.0)

  ts_ns = evt.ts_unix_nano
  ts_s = ts_ns // 1_000_000_000
  hour_of_day = (ts_s // 3600) % 24
  hour_normalized = hour_of_day / 24.0
  minute_of_hour = (ts_s // 60) % 60
  minute_normalized = minute_of_hour / 60.0

  # --- Extended features (syscall-argument and context) ---
  # Numeric syscall ID (e.g. 42=connect, 257=openat) – helps distinguish call types
  event_id_val = max(0, _safe_int(event_id_str, default=0))
  event_id_norm = min(event_id_val / 500.0, 1.0)

  # Flags (e.g. O_RDONLY vs O_WRONLY) – open/openat semantics
  flags_hash = _hash01(flags)

  # Path structure: depth and top-level directory (sensitive paths: /etc, /tmp, /proc, etc.)
  components = _path_components(path)
  path_depth = len(components)
  path_depth_norm = min(path_depth / 20.0, 1.0)
  path_prefix = components[0] if components else ""
  path_prefix_hash = _hash01(path_prefix)

  # Return value from attributes: success vs errno (failed syscalls are often security-relevant)
  return_val = _safe_int(attrs.get("return_value", "0"), default=0)
  return_success = 1.0 if return_val >= 0 else 0.0
  return_errno_norm = min(np.log1p(abs(return_val)) / 12.0, 1.0)

  # Mount namespace (container/isolation context)
  mount_ns = attrs.get("mount_namespace", "") or ""
  mount_ns_hash = _hash01(mount_ns)

  # Hostname (multi-node / multi-host)
  hostname_hash = _hash01(evt.hostname or "")

  return (
    event_hash_normalized,
    comm_hash_normalized,
    path_hash_normalized,
    pid_normalized,
    tid_normalized,
    uid_normalized,
    arg0_norm,
    arg1_norm,
    hour_normalized,
    minute_normalized,
    event_id_norm,
    flags_hash,
    path_depth_norm,
    path_prefix_hash,
    return_success,
    return_errno_norm,
    mount_ns_hash,
    hostname_hash,
  )


def _extract_file_features(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """
  Type-specific features for event_type == 'file'.

  Covers all file syscalls that trigger this extractor: open, openat, openat2,
  unlink, unlinkat, rename, renameat, chmod, chown. They share the same event layout
  (data[8]=path, data[6]/[7]=arg0/arg1, attributes open_flags for open*). One unified
  feature set: path semantics, syscall identity, open flags (when applicable), and
  Kitsune-style online stats (rate, interarrival, path-depth streams) per process,
  host, path, and (comm, path) pair.
  """
  path = (evt.data[8] if len(evt.data) > 8 else "").strip()
  path_lower = path.lower()
  comm = (evt.data[2] if len(evt.data) > 2 else "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  event_name = (evt.event_name or "") or (evt.data[0] if len(evt.data) > 0 else "")
  arg0_str = evt.data[6] if len(evt.data) > 6 else "0"
  arg1_str = evt.data[7] if len(evt.data) > 7 else "0"
  arg0_val = abs(_safe_int(arg0_str, default=0))
  arg1_val = abs(_safe_int(arg1_str, default=0))
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0
  attrs = dict(evt.attributes or {})

  # Path semantics (all file syscalls have at least one path)
  sensitive_prefixes = ("/etc", "/root", "/bin", "/usr/bin", "/sbin", "/usr/sbin")
  file_sensitive_path = 1.0 if any(path_lower.startswith(p) for p in sensitive_prefixes) else 0.0
  file_tmp_path = 1.0 if path_lower.startswith("/tmp") or path_lower.startswith("/var/tmp") else 0.0
  components = _path_components(path)
  ext = ""
  if components:
    last = components[-1]
    if "." in last:
      ext = "." + last.split(".")[-1]
  file_extension_hash = _hash01(ext)
  path_depth_norm = min(len(components) / 20.0, 1.0)

  # Syscall identity (open vs unlink vs rename vs chmod vs chown etc.)
  file_event_name_hash = _hash01(event_name)
  # open_flags only meaningful for open/openat/openat2; others get 0.0 so feature always present
  open_flags_str = (attrs.get("open_flags") or (evt.data[9] if len(evt.data) > 9 else "")) or ""
  file_open_flags_hash = _hash01(open_flags_str) if event_name in ("open", "openat", "openat2") else 0.0

  # Normalized args (dfd/flags for openat, mode for chmod, user/group for chown, etc.)
  file_arg0_norm = min(np.log1p(arg0_val) / 20.0, 1.0)
  file_arg1_norm = min(np.log1p(arg1_val) / 20.0, 1.0)

  # Online statistics: only type-specific streams (path, pair). proc/host rate and
  # proc interarrival are in general features.
  path_key = path or "unknown"
  pair_key = f"{comm}|{path_key}"
  path_rate = _ONLINE_STATS.update_value("file", "rate", path_key, ts_s, 1.0)
  pair_rate = _ONLINE_STATS.update_value("file", "rate", pair_key, ts_s, 1.0)
  pair_dt = _ONLINE_STATS.update_interarrival("file", "interarrival", pair_key, ts_s, dt_scale_s=30.0)
  proc_path_depth = _ONLINE_STATS.update_value("file", "path_depth", comm, ts_s, path_depth_norm)
  host_path_depth = _ONLINE_STATS.update_value("file", "path_depth", host, ts_s, path_depth_norm)
  pair_path_depth = _ONLINE_STATS.update_value("file", "path_depth", pair_key, ts_s, path_depth_norm)

  def _rate01(weight: float) -> float:
    return float(1.0 - np.exp(-max(0.0, weight)))

  out: Dict[str, float] = {
    "file_sensitive_path": file_sensitive_path,
    "file_tmp_path": file_tmp_path,
    "file_extension_hash": file_extension_hash,
    "file_event_name_hash": file_event_name_hash,
    "file_open_flags_hash": file_open_flags_hash,
    "file_arg0_norm": file_arg0_norm,
    "file_arg1_norm": file_arg1_norm,
  }
  for w in _ONLINE_WINDOWS:
    wn = w[0]
    out[f"file_path_rate_{wn}"] = _rate01(path_rate[wn][2])
    out[f"file_pair_rate_{wn}"] = _rate01(pair_rate[wn][2])
    out[f"file_pair_interarrival_{wn}"] = pair_dt[wn][0]
    out[f"file_pair_interarrival_std_{wn}"] = pair_dt[wn][1]
    out[f"file_proc_path_depth_mean_{wn}"] = proc_path_depth[wn][0]
    out[f"file_proc_path_depth_std_{wn}"] = proc_path_depth[wn][1]
    out[f"file_host_path_depth_mean_{wn}"] = host_path_depth[wn][0]
    out[f"file_host_path_depth_std_{wn}"] = host_path_depth[wn][1]
    out[f"file_pair_path_depth_mean_{wn}"] = pair_path_depth[wn][0]
    out[f"file_pair_path_depth_std_{wn}"] = pair_path_depth[wn][1]
  return out


def _extract_network_features(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """Type-specific features for event_type == 'network'."""
  arg0_str = evt.data[6] if len(evt.data) > 6 else "0"
  arg1_str = evt.data[7] if len(evt.data) > 7 else "0"
  arg0_val = abs(_safe_int(arg0_str, default=0))
  arg1_val = abs(_safe_int(arg1_str, default=0))
  event_name = (evt.event_name or "") or (evt.data[0] if len(evt.data) > 0 else "")
  comm = (evt.data[2] if len(evt.data) > 2 else "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0

  # connect(): arg0=fd, arg1=addrlen. socket(): arg0=family, arg1=type (same data slots).
  net_fd_norm = min(np.log1p(arg0_val) / 20.0, 1.0)
  net_addrlen_norm = min(np.log1p(arg1_val) / 10.0, 1.0)

  sockaddr = _parse_sockaddr_from_evt(evt)

  # socket() arg0 = family (AF_INET=2, AF_INET6=10); connect() from parsed sockaddr/attributes.
  family_val = 0
  if event_name == "socket":
    family_val = arg0_val
  else:
    af = (sockaddr.get("sa_family") or "").upper()
    if "INET6" in af or af == "10":
      family_val = 10
    elif "INET" in af or af == "2":
      family_val = 2
  net_socket_family_norm = min(family_val / 10.0, 1.0)

  # socket() arg1 = type: SOCK_STREAM=1, SOCK_DGRAM=2.
  type_val = arg1_val if event_name == "socket" else 0
  net_socket_type_hash = _hash01(str(type_val))

  # Destination port from parsed sockaddr (BETH data[7] or attributes sin_port/dest_port).
  port_str = sockaddr.get("sin_port") or ""
  dport = _safe_int(port_str, default=0) if port_str else 0
  net_dport_norm = min(dport / 65535.0, 1.0) if dport else 0.0

  # Destination IP and address family hashes (host/socket-pair identity).
  daddr = sockaddr.get("sin_addr") or ""
  net_daddr_hash = _hash01(daddr)
  net_af_hash = _hash01(sockaddr.get("sa_family") or ("AF_INET" if family_val == 2 else "AF_INET6" if family_val == 10 else ""))

  # --- Online statistics: only type-specific streams (pair, daddr). proc/host rate
  # and proc interarrival are in general features.
  pair_key = f"{comm}|{daddr}|{dport}"
  daddr_key = daddr or "unknown"
  pair_rate = _ONLINE_STATS.update_value("network", "rate", pair_key, ts_s, 1.0)
  daddr_rate = _ONLINE_STATS.update_value("network", "rate", daddr_key, ts_s, 1.0)
  pair_dt = _ONLINE_STATS.update_interarrival("network", "interarrival", pair_key, ts_s, dt_scale_s=30.0)
  proc_dport = _ONLINE_STATS.update_value("network", "dport", comm, ts_s, net_dport_norm)
  proc_addrlen = _ONLINE_STATS.update_value("network", "addrlen", comm, ts_s, net_addrlen_norm)
  host_dport = _ONLINE_STATS.update_value("network", "dport", host, ts_s, net_dport_norm)
  host_addrlen = _ONLINE_STATS.update_value("network", "addrlen", host, ts_s, net_addrlen_norm)
  daddr_dport = _ONLINE_STATS.update_value("network", "dport", daddr_key, ts_s, net_dport_norm)
  proc_daddr_key = f"{comm}|{daddr}"
  proc_daddr_dport = _ONLINE_STATS.update_value("network", "dport", proc_daddr_key, ts_s, net_dport_norm)

  def _rate01(weight: float) -> float:
    return float(1.0 - np.exp(-max(0.0, weight)))

  out: Dict[str, float] = {
    "net_addrlen_norm": net_addrlen_norm,
    "net_fd_norm": net_fd_norm,
    "net_socket_family_norm": net_socket_family_norm,
    "net_socket_type_hash": net_socket_type_hash,
    "net_dport_norm": net_dport_norm,
    "net_daddr_hash": net_daddr_hash,
    "net_af_hash": net_af_hash,
  }
  for w in _ONLINE_WINDOWS:
    wn = w[0]
    out[f"net_pair_rate_{wn}"] = _rate01(pair_rate[wn][2])
    out[f"net_daddr_rate_{wn}"] = _rate01(daddr_rate[wn][2])
    out[f"net_pair_interarrival_{wn}"] = pair_dt[wn][0]
    out[f"net_pair_interarrival_std_{wn}"] = pair_dt[wn][1]
    out[f"net_proc_dport_mean_{wn}"] = proc_dport[wn][0]
    out[f"net_proc_dport_std_{wn}"] = proc_dport[wn][1]
    out[f"net_proc_addrlen_mean_{wn}"] = proc_addrlen[wn][0]
    out[f"net_proc_addrlen_std_{wn}"] = proc_addrlen[wn][1]
    out[f"net_host_dport_mean_{wn}"] = host_dport[wn][0]
    out[f"net_host_dport_std_{wn}"] = host_dport[wn][1]
    out[f"net_host_addrlen_mean_{wn}"] = host_addrlen[wn][0]
    out[f"net_host_addrlen_std_{wn}"] = host_addrlen[wn][1]
    out[f"net_daddr_dport_mean_{wn}"] = daddr_dport[wn][0]
    out[f"net_daddr_dport_std_{wn}"] = daddr_dport[wn][1]
    out[f"net_proc_daddr_dport_mean_{wn}"] = proc_daddr_dport[wn][0]
    out[f"net_proc_daddr_dport_std_{wn}"] = proc_daddr_dport[wn][1]
  return out


def _extract_process_features(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """Type-specific features for event_type == 'process' (exec/spawn)."""
  event_name = (evt.event_name or "") or (evt.data[0] if len(evt.data) > 0 else "")
  process_is_execve = 1.0 if event_name == "execve" else 0.0
  process_is_fork = 1.0 if event_name == "fork" else 0.0
  return {
    "process_is_execve": process_is_execve,
    "process_is_fork": process_is_fork,
  }


def _extract_general_online_stats(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """
  Online stats that share the same composition across event types: proc rate,
  host rate, proc interarrival. Used for all events so type-specific extractors
  do not duplicate these fields.
  """
  comm = (evt.data[2] if len(evt.data) > 2 else "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0
  proc_rate = _ONLINE_STATS.update_value("general", "rate", comm, ts_s, 1.0)
  host_rate = _ONLINE_STATS.update_value("general", "rate", host, ts_s, 1.0)
  proc_dt = _ONLINE_STATS.update_interarrival("general", "interarrival", comm, ts_s, dt_scale_s=30.0)

  def _rate01(weight: float) -> float:
    return float(1.0 - np.exp(-max(0.0, weight)))

  out: Dict[str, float] = {}
  for w in _ONLINE_WINDOWS:
    wn = w[0]
    out[f"proc_rate_{wn}"] = _rate01(proc_rate[wn][2])
    out[f"host_rate_{wn}"] = _rate01(host_rate[wn][2])
    out[f"proc_interarrival_{wn}"] = proc_dt[wn][0]
    out[f"proc_interarrival_std_{wn}"] = proc_dt[wn][1]
  return out


def extract_feature_dict(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """
  Extract features as a dict for streaming models.

  Always includes general features (including shared online stats: proc/host rate,
  proc interarrival). If evt.event_type is set (e.g. 'file', 'network', 'process'),
  appends type-specific features. Features and thus vector size can differ per event.
  """
  values = _extract_feature_values(evt)
  out: Dict[str, float] = {
    "event_hash": values[0],
    "comm_hash": values[1],
    "path_hash": values[2],
    "pid_norm": values[3],
    "tid_norm": values[4],
    "uid_norm": values[5],
    "arg0_norm": values[6],
    "arg1_norm": values[7],
    "hour_norm": values[8],
    "minute_norm": values[9],
    "event_id_norm": values[10],
    "flags_hash": values[11],
    "path_depth_norm": values[12],
    "path_prefix_hash": values[13],
    "return_success": values[14],
    "return_errno_norm": values[15],
    "mount_ns_hash": values[16],
    "hostname_hash": values[17],
  }
  out.update(_extract_general_online_stats(evt))
  event_type = (evt.event_type or "").strip().lower()
  if event_type == "file":
    out.update(_extract_file_features(evt))
  elif event_type == "network":
    out.update(_extract_network_features(evt))
  elif event_type == "process":
    out.update(_extract_process_features(evt))
  return out
