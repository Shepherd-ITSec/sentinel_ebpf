"""Feature extraction for anomaly detection.

EventEnvelope carries syscall fields by name (syscall_nr, comm, pid, tid, uid, arg0, arg1, path);
event_name is the syscall name; event_group is the rule-defined category (network, file, process, or empty).
We always add general features; depending on event_group we add type-specific features.
"""
import ast
import hashlib
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

import events_pb2
from rules_config import RulesConfig, load_rules_config


@dataclass
class _DecayedMoments:
  last_ts: float = 0.0
  has_last: bool = False
  weight: float = 0.0
  sum_v: float = 0.0
  sum_sq: float = 0.0


@dataclass(frozen=True)
class _FeatureViewSpec:
  include_general_context_buckets: bool = True
  include_general_path_tokens: bool = True
  include_general_online_stats: bool = True
  include_file_sensitive_tmp: bool = True
  include_file_event_name: bool = True
  include_file_extension_bucket: bool = True
  include_file_flags_bucket: bool = True
  include_file_online_stats: bool = True
  include_network_event_name: bool = True
  include_network_socket_type_bucket: bool = True
  include_network_daddr_bucket: bool = True
  include_network_af_onehot: bool = True
  include_network_online_stats: bool = True
  use_hash_for_categoricals: bool = False
  # False for frequency: use hour_norm / minute_norm / weekday_norm instead of sin/cos pairs.
  cyclic_time_features: bool = True


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

_REPO_RULES_PATH = Path(__file__).resolve().parents[1] / "charts" / "sentinel-ebpf" / "rules.yaml"
_COMM_BUCKETS = 64
_HOSTNAME_BUCKETS = 32
# Path encoding (encoded mode): depth-positioned token banks.
# Keep the total budget stable: 4 depths × 64 buckets = 256.
_PATH_DEPTH_BUCKETS = 4
_PATH_TOKEN_BUCKETS_PER_DEPTH = 64
_FILE_EXTENSION_BUCKETS = 32
_FILE_FLAGS_BUCKETS = 64
_NET_SOCKET_TYPE_BUCKETS = 16
_NET_DADDR_BUCKETS = 64
_DEFAULT_FILE_EVENT_NAMES = ("open", "openat", "openat2", "unlink", "unlinkat", "rename", "renameat", "chmod", "chown", "read", "write")
_DEFAULT_NET_EVENT_NAMES = ("socket", "connect", "bind", "listen", "accept", "accept4")
_NET_AF_VALUES = ("af_inet", "af_inet6", "af_unix", "af_netlink", "af_other")
_DEFAULT_SENSITIVE_PATH_PREFIXES = ("/etc", "/root", "/bin", "/usr/bin", "/sbin", "/usr/sbin")
_DEFAULT_TMP_PATH_PREFIXES = ("/tmp", "/var/tmp")

# Linux ID limits for normalization (PID_MAX_LIMIT 2^22, uid_t 2^32-1)
_PID_MAX = 4_194_304
_UID_MAX = 4_294_967_295
# Normalization scales
_ARG_LOG_SCALE = 20.0
_PATH_DEPTH_CAP = 20.0
_RETURN_ERRNO_SCALE = 12.0
_EVENT_ID_MAX = 500.0
_PORT_MAX = 65535.0
_ADDRLEN_SCALE = 10.0
_SOCKET_FAMILY_MAX = 10.0
_RATE_CAP = 50.0

_FULL_FEATURE_VIEW = _FeatureViewSpec()
_LODA_FEATURE_VIEW = _FeatureViewSpec(
  include_general_context_buckets=False,
  include_general_path_tokens=False,
  include_file_event_name=False,
  include_file_extension_bucket=False,
  include_file_flags_bucket=False,
  include_network_event_name=False,
  include_network_daddr_bucket=False,
)
_MEMSTREAM_FEATURE_VIEW = _FeatureViewSpec(
  include_general_context_buckets=False,
  include_general_path_tokens=False,
  include_file_event_name=False,
  include_file_extension_bucket=False,
  include_file_flags_bucket=False,
  include_network_event_name=False,
  include_network_socket_type_bucket=False,
  include_network_daddr_bucket=False,
)
# Frequency-model view (freq1d, indep_marginal, gausscop, copulatree, latentcluster): scalar hashes + event_id_norm,
# no global flags_hash, no file sensitive/tmp binaries, no online rate/interarrival streams.
_FREQUENCY_FEATURE_VIEW = _FeatureViewSpec(
  use_hash_for_categoricals=True,
  cyclic_time_features=False,
  include_file_sensitive_tmp=False,
  include_general_online_stats=False,
  include_file_online_stats=False,
  include_network_online_stats=False,
)


def _safe_int(raw: str, default: int = 0) -> int:
  try:
    return int(raw)
  except (ValueError, OverflowError, TypeError):
    return default


def _norm_pid(val: int) -> float:
  return min(max(0, val) / _PID_MAX, 1.0)


def _norm_uid(val: int) -> float:
  return min(max(0, val) / _UID_MAX, 1.0)


def _norm_arg(val: int, scale: float = _ARG_LOG_SCALE) -> float:
  return min(np.log1p(abs(val)) / scale, 1.0)


def _norm_path_depth(components: list) -> float:
  return min(len(components) / _PATH_DEPTH_CAP, 1.0)


def _norm_return_errno(return_val: int) -> tuple[float, float]:
  success = 1.0 if return_val >= 0 else 0.0
  errno_norm = min(np.log1p(abs(return_val)) / _RETURN_ERRNO_SCALE, 1.0)
  return success, errno_norm


def _rate01(weight: float) -> float:
  w = max(0.0, float(weight))
  return float(min(np.log1p(w) / np.log1p(_RATE_CAP), 1.0))


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


def _sanitize_feature_name(value: str) -> str:
  cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
  return cleaned or "unknown"


@lru_cache(maxsize=1)
def _detector_rules_path() -> Path:
  env_path = (os.environ.get("DETECTOR_RULES_PATH") or "").strip()
  if env_path:
    return Path(env_path)
  mounted_path = Path("/etc/sentinel-ebpf/rules.yaml")
  if mounted_path.exists():
    return mounted_path
  return _REPO_RULES_PATH


@lru_cache(maxsize=1)
def _rules_config() -> RulesConfig | None:
  try:
    return load_rules_config(_detector_rules_path())
  except (OSError, ValueError):
    return None


def _group_syscalls(event_group: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
  cfg = _rules_config()
  if cfg is None:
    return tuple(sorted(set(fallback)))
  target = (event_group or "").strip().lower()
  names = {syscall for rule in cfg.rules if rule.group == target for syscall in rule.syscalls}
  if not names:
    return tuple(sorted(set(fallback)))
  return tuple(sorted(names))


def _group_feature_values(event_group: str, feature_name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
  cfg = _rules_config()
  if cfg is None:
    return tuple(sorted(set(fallback)))
  group = cfg.groups.get((event_group or "").strip().lower())
  if group is None:
    return tuple(sorted(set(fallback)))
  values = group.features.get((feature_name or "").strip().lower(), ())
  if not values:
    return tuple(sorted(set(fallback)))
  return tuple(sorted(set(values)))


def feature_view_for_algorithm(algorithm: str | None) -> str:
  """Map algorithm to feature view."""
  algo = (algorithm or "").strip().lower()
  if algo in ("freq1d", "indep_marginal", "gausscop", "copulatree", "latentcluster"):
    return "frequency"
  if algo in ("loda", "loda_ema"):
    return "loda"
  if algo == "memstream":
    return "memstream"
  return "default"


def _feature_view_spec(feature_view: str | None) -> _FeatureViewSpec:
  normalized = (feature_view or "default").strip().lower()
  if normalized == "frequency":
    return _FREQUENCY_FEATURE_VIEW
  if normalized in ("default", "full", "encoded"):
    return _FULL_FEATURE_VIEW
  if normalized == "loda":
    return _LODA_FEATURE_VIEW
  if normalized == "memstream":
    return _MEMSTREAM_FEATURE_VIEW
  raise ValueError(f"Unknown feature_view={feature_view!r}; expected one of: default, frequency, loda, memstream")


@lru_cache(maxsize=16)
def _bucket_feature_names(prefix: str, bucket_count: int) -> tuple[str, ...]:
  return tuple(f"{prefix}_{idx:03d}" for idx in range(bucket_count))


def _bucket_index(value: str, bucket_count: int) -> int:
  digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
  return int(digest, 16) % bucket_count


def _bucketize_value(prefix: str, value: str, bucket_count: int) -> Dict[str, float]:
  out = {name: 0.0 for name in _bucket_feature_names(prefix, bucket_count)}
  normalized = (value or "").strip().lower() or "__empty__"
  out[f"{prefix}_{_bucket_index(normalized, bucket_count):03d}"] = 1.0
  return out


def _normalize_path_token(token: str) -> str:
  token = (token or "").strip().lower()
  if not token:
    return ""
  if token.isdigit():
    return "<num>"
  if re.fullmatch(r"[0-9a-f]{8,}", token):
    return "<hex>"
  if re.fullmatch(r"[0-9a-f-]{32,}", token):
    return "<hex>"
  if len(token) >= 12 and any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
    return "<id>"
  if len(token) > 32:
    token = token[:32]
  return token


def _tokenize_path(path: str) -> tuple[str, ...]:
  tokens: list[str] = []
  for component in _path_components(path.lower()):
    for part in re.split(r"[._-]+", component):
      normalized = _normalize_path_token(part)
      if normalized:
        tokens.append(normalized)
  if not tokens:
    return ("<empty>",)
  return tuple(tokens)


def _tokenize_path_by_depth(path: str) -> list[tuple[str, ...]]:
  """
  Tokenize path into per-depth token tuples.

  Depth is based on '/' components (directories / basename). Each component is further split
  on [._-] and normalized (numbers/hex/id collapse) to reduce churn.
  """
  comps = _path_components(path.lower())
  out: list[tuple[str, ...]] = []
  for comp in comps:
    toks: list[str] = []
    for part in re.split(r"[._-]+", comp):
      normalized = _normalize_path_token(part)
      if normalized:
        toks.append(normalized)
    out.append(tuple(toks) if toks else ("<empty>",))
  if not out:
    out = [("<empty>",)]
  return out


def _bucketize_tokens(prefix: str, tokens: tuple[str, ...], bucket_count: int) -> Dict[str, float]:
  out = {name: 0.0 for name in _bucket_feature_names(prefix, bucket_count)}
  for token in set(tokens):
    out[f"{prefix}_{_bucket_index(token, bucket_count):03d}"] = 1.0
  return out


def _bucketize_path_tokens_by_depth(path: str) -> Dict[str, float]:
  """
  Depth-positioned path token buckets.

  Emits fixed banks:
  - path_tok_d0_bucket_000..063
  - path_tok_d1_bucket_000..063
  - path_tok_d2_bucket_000..063
  - path_tok_d3_bucket_000..063
  """
  tokens_by_depth = _tokenize_path_by_depth(path)
  out: Dict[str, float] = {}
  for d in range(_PATH_DEPTH_BUCKETS):
    if d < len(tokens_by_depth):
      toks = tokens_by_depth[d]
    else:
      toks = ("<empty>",)
    out.update(_bucketize_tokens(f"path_tok_d{d}_bucket", toks, _PATH_TOKEN_BUCKETS_PER_DEPTH))
  return out


def _onehot_features(prefix: str, selected: str, known_values: tuple[str, ...]) -> Dict[str, float]:
  selected_norm = (selected or "").strip().lower()
  out: Dict[str, float] = {}
  for value in known_values:
    out[f"{prefix}_{_sanitize_feature_name(value)}"] = 1.0 if selected_norm == value else 0.0
  return out


def _normalize_flag_token(token: str) -> str:
  token = (token or "").strip().lower()
  if not token:
    return ""
  if token.isdigit():
    return "<num>"
  return _sanitize_feature_name(token)


def _tokenize_flags(flags: str, *, empty_token: str = "<none>") -> tuple[str, ...]:
  parts = [_normalize_flag_token(part) for part in re.split(r"[|,\s]+", flags or "")]
  tokens = tuple(part for part in parts if part)
  return tokens or (empty_token,)


def _normalize_af_label(raw: str, family_val: int) -> str:
  af = (raw or "").strip().upper()
  if "INET6" in af or af == "10" or family_val == 10:
    return "af_inet6"
  if "INET" in af or af == "2" or family_val == 2:
    return "af_inet"
  if "UNIX" in af or "LOCAL" in af:
    return "af_unix"
  if "NETLINK" in af or family_val == 16:
    return "af_netlink"
  return "af_other"


def _parse_sockaddr_from_evt(evt: Any) -> Dict[str, str]:
  """
  Try to get destination port, address, and family from event (attributes or arg1).
  Returns dict with keys sin_port, sin_addr, sa_family (values may be empty).
  """
  out: Dict[str, str] = {"sin_port": "", "sin_addr": "", "sa_family": ""}
  attrs = dict(evt.attributes or {})
  out["sin_port"] = (attrs.get("sin_port") or attrs.get("dest_port") or "").strip()
  out["sin_addr"] = (attrs.get("sin_addr") or attrs.get("dest_ip") or "").strip()
  out["sa_family"] = (attrs.get("sa_family") or "").strip()

  # Fallback: arg1 may contain stringified sockaddr dict (synthetic / legacy)
  if not out["sin_port"] and not out["sin_addr"] and not out["sa_family"]:
    raw = (getattr(evt, "arg1", None) or "").strip()
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


def _syscall_flags_numeric_string(evt: Any) -> str:
  """
  Raw open/socket flag bits as a string, for file flag buckets/hash and fallbacks. The probe stores the same
  value in data_t.flags and in arg0 (open) or arg1 (openat, openat2, socket); we only read args.
  """
  name = (evt.event_name or "").strip().lower()
  if name == "open":
    return (evt.arg0 or "").strip()
  if name in ("openat", "openat2"):
    return (evt.arg1 or "").strip()
  if name == "socket":
    return (evt.arg1 or "").strip()
  return ""


def _extract_generic_features(evt: Any, view: _FeatureViewSpec) -> Dict[str, float]:
  """Generic features shared across all event types."""
  event_id_str = str(int(evt.syscall_nr))
  comm = evt.comm or ""
  pid_str = evt.pid or "0"
  tid_str = evt.tid or "0"
  uid_str = evt.uid or "0"
  arg0_str = evt.arg0 or "0"
  arg1_str = evt.arg1 or "0"
  path = (evt.path or "").strip()
  attrs = dict(evt.attributes or {})

  pid_val = _safe_int(pid_str, default=0)
  tid_val = _safe_int(tid_str, default=0)
  uid_val = _safe_int(uid_str, default=0)
  arg0_val = _safe_int(arg0_str, default=0)
  arg1_val = _safe_int(arg1_str, default=0)
  event_id_val = max(0, _safe_int(event_id_str, default=0))

  ts_ns = evt.ts_unix_nano
  ts_s = ts_ns // 1_000_000_000
  hour_of_day = (ts_s // 3600) % 24
  minute_of_hour = (ts_s // 60) % 60
  hour_angle = 2.0 * float(np.pi) * (float(hour_of_day) / 24.0)
  minute_angle = 2.0 * float(np.pi) * (float(minute_of_hour) / 60.0)
  dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
  weekday = dt.weekday()
  weekday_angle = 2.0 * float(np.pi) * (float(weekday) / 7.0)
  day_of_month = dt.day
  week_of_month = min(4, (day_of_month - 1) // 7 + 1)

  components = _path_components(path)
  path_prefix = components[0] if components else ""
  return_val = _safe_int(attrs.get("return_value", "0"), default=0)
  return_success, return_errno_norm = _norm_return_errno(return_val)

  out: Dict[str, float] = {
    "pid_norm": _norm_pid(pid_val),
    "tid_norm": _norm_pid(tid_val),
    "uid_norm": _norm_uid(uid_val),
    "arg0_norm": _norm_arg(arg0_val),
    "arg1_norm": _norm_arg(arg1_val),
    "path_depth_norm": _norm_path_depth(components),
    "return_success": return_success,
    "return_errno_norm": return_errno_norm,
  }
  if view.cyclic_time_features:
    out.update(
      {
        "hour_sin": float(np.sin(hour_angle)),
        "hour_cos": float(np.cos(hour_angle)),
        "minute_sin": float(np.sin(minute_angle)),
        "minute_cos": float(np.cos(minute_angle)),
        "weekday_sin": float(np.sin(weekday_angle)),
        "weekday_cos": float(np.cos(weekday_angle)),
      }
    )
  else:
    out.update(
      {
        "hour_norm": float(hour_of_day) / 23.0,
        "minute_norm": float(minute_of_hour) / 59.0,
        "weekday_norm": float(weekday) / 6.0,
      }
    )
  out["week_of_month_norm"] = (week_of_month - 1) / 3.0

  if view.use_hash_for_categoricals:
    out["event_id_norm"] = min(event_id_val / _EVENT_ID_MAX, 1.0)
    if view.include_general_context_buckets:
      out["comm_hash"] = _hash01(comm)
      out["hostname_hash"] = _hash01(evt.hostname or "")
    if view.include_general_path_tokens:
      out["path_hash"] = _hash01(path)
      out["path_prefix_hash"] = _hash01(path_prefix)
  else:
    if view.include_general_context_buckets:
      out.update(_bucketize_value("comm_bucket", comm, _COMM_BUCKETS))
      out.update(_bucketize_value("hostname_bucket", evt.hostname or "", _HOSTNAME_BUCKETS))
    if view.include_general_path_tokens:
      out.update(_bucketize_path_tokens_by_depth(path))
  return out


def _extract_file_features(evt: Any, view: _FeatureViewSpec = _FULL_FEATURE_VIEW) -> Dict[str, float]:
  """
  Type-specific features for event_group == 'file'.

  Covers all file syscalls that trigger this extractor: open, openat, openat2,
  unlink, unlinkat, rename, renameat, chmod, chown. They share the same event layout
  (path, arg0/arg1, attributes flags for open*). One unified
  feature set: path semantics, syscall identity, open flags (when applicable), and
  Kitsune-style online stats (rate, interarrival, path-depth streams) per process,
  host, path, and (comm, path) pair.
  """
  path = (evt.path or "").strip()
  path_lower = path.lower()
  comm = (evt.comm or "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  event_name = (evt.event_name or "").strip().lower()
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0
  attrs = dict(evt.attributes or {})

  sensitive_prefixes = _group_feature_values("file", "sensitive_paths", _DEFAULT_SENSITIVE_PATH_PREFIXES)
  tmp_prefixes = _group_feature_values("file", "tmp_paths", _DEFAULT_TMP_PATH_PREFIXES)
  file_sensitive_path = 1.0 if any(path_lower.startswith(p) for p in sensitive_prefixes) else 0.0
  file_tmp_path = 1.0 if any(path_lower.startswith(p) for p in tmp_prefixes) else 0.0
  components = _path_components(path) if (view.include_file_extension_bucket or view.include_file_online_stats) else []
  ext = ""
  if components:
    last = components[-1]
    if "." in last:
      ext = "." + last.split(".")[-1]
  flags_str = (attrs.get("flags") or _syscall_flags_numeric_string(evt)) or ""

  out: Dict[str, float] = {}
  if view.include_file_sensitive_tmp:
    out["file_sensitive_path"] = file_sensitive_path
    out["file_tmp_path"] = file_tmp_path
  if view.use_hash_for_categoricals:
    if view.include_file_extension_bucket:
      out["file_extension_hash"] = _hash01(ext)
    if view.include_file_flags_bucket:
      out["file_flags_hash"] = _hash01(flags_str)
  else:
    if view.include_file_event_name:
      out.update(_onehot_features("file_event_name", event_name, _group_syscalls("file", _DEFAULT_FILE_EVENT_NAMES)))
    if view.include_file_extension_bucket:
      out.update(_bucketize_value("file_extension_bucket", ext, _FILE_EXTENSION_BUCKETS))
    if view.include_file_flags_bucket:
      flag_tokens = _tokenize_flags(flags_str)
      out.update(_bucketize_tokens("file_flags_bucket", flag_tokens, _FILE_FLAGS_BUCKETS))
  if view.include_file_online_stats:
    path_depth_norm = _norm_path_depth(components)
    path_key = path or "unknown"
    pair_key = f"{comm}|{path_key}"
    path_rate = _ONLINE_STATS.update_value("file", "rate", path_key, ts_s, 1.0)
    pair_rate = _ONLINE_STATS.update_value("file", "rate", pair_key, ts_s, 1.0)
    pair_dt = _ONLINE_STATS.update_interarrival("file", "interarrival", pair_key, ts_s, dt_scale_s=30.0)
    proc_path_depth = _ONLINE_STATS.update_value("file", "path_depth", comm, ts_s, path_depth_norm)
    host_path_depth = _ONLINE_STATS.update_value("file", "path_depth", host, ts_s, path_depth_norm)
    pair_path_depth = _ONLINE_STATS.update_value("file", "path_depth", pair_key, ts_s, path_depth_norm)
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


def _extract_network_features(evt: Any, view: _FeatureViewSpec = _FULL_FEATURE_VIEW) -> Dict[str, float]:
  """Type-specific features for event_group == 'network'."""
  arg0_val = _safe_int(evt.arg0 or "0", default=0)
  arg1_val = _safe_int(evt.arg1 or "0", default=0)
  event_name = evt.event_name or ""
  comm = (evt.comm or "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0

  net_fd_norm = _norm_arg(arg0_val)
  net_addrlen_norm = min(np.log1p(abs(arg1_val)) / _ADDRLEN_SCALE, 1.0)

  sockaddr = _parse_sockaddr_from_evt(evt)
  family_val = 0
  if event_name == "socket":
    family_val = arg0_val
  else:
    af = (sockaddr.get("sa_family") or "").upper()
    if "INET6" in af or af == "10":
      family_val = 10
    elif "INET" in af or af == "2":
      family_val = 2
  net_socket_family_norm = min(family_val / _SOCKET_FAMILY_MAX, 1.0)

  type_val = arg1_val if event_name == "socket" else 0
  port_str = sockaddr.get("sin_port") or ""
  dport = _safe_int(port_str, default=0) if port_str else 0
  net_dport_norm = min(dport / _PORT_MAX, 1.0) if dport else 0.0

  daddr = sockaddr.get("sin_addr") or ""
  af_label = _normalize_af_label(sockaddr.get("sa_family") or "", family_val)
  event_name_key = (evt.event_name or "").strip().lower()

  out: Dict[str, float] = {
    "net_addrlen_norm": net_addrlen_norm,
    "net_fd_norm": net_fd_norm,
    "net_socket_family_norm": net_socket_family_norm,
    "net_dport_norm": net_dport_norm,
  }
  if view.use_hash_for_categoricals:
    if view.include_network_socket_type_bucket:
      out["net_socket_type_hash"] = _hash01(str(type_val))
    if view.include_network_daddr_bucket:
      out["net_daddr_hash"] = _hash01(daddr)
    if view.include_network_af_onehot:
      af_str = sockaddr.get("sa_family") or ("AF_INET" if family_val == 2 else "AF_INET6" if family_val == 10 else "")
      out["net_af_hash"] = _hash01(af_str)
  else:
    if view.include_network_event_name:
      out.update(
        _onehot_features("net_event_name", event_name_key, _group_syscalls("network", _DEFAULT_NET_EVENT_NAMES))
      )
    if view.include_network_socket_type_bucket:
      out.update(_bucketize_value("net_socket_type_bucket", str(type_val), _NET_SOCKET_TYPE_BUCKETS))
    if view.include_network_daddr_bucket:
      out.update(_bucketize_value("net_daddr_bucket", daddr, _NET_DADDR_BUCKETS))
    if view.include_network_af_onehot:
      out.update(_onehot_features("net_af", af_label, _NET_AF_VALUES))
  if view.include_network_online_stats:
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


def _extract_process_features(evt: Any) -> Dict[str, float]:
  """Type-specific features for event_group == 'process' (exec/spawn)."""
  event_name = evt.event_name or ""
  process_is_execve = 1.0 if event_name == "execve" else 0.0
  process_is_fork = 1.0 if event_name == "fork" else 0.0
  return {
    "process_is_execve": process_is_execve,
    "process_is_fork": process_is_fork,
  }


def _extract_general_online_stats(evt: Any) -> Dict[str, float]:
  """
  Online stats that share the same composition across event types: proc rate,
  host rate, proc interarrival. Used for all events so type-specific extractors
  do not duplicate these fields.
  """
  comm = (evt.comm or "") or "unknown"
  host = (evt.hostname or "") or "unknown"
  ts_s = float(evt.ts_unix_nano) / 1_000_000_000.0
  proc_rate = _ONLINE_STATS.update_value("general", "rate", comm, ts_s, 1.0)
  host_rate = _ONLINE_STATS.update_value("general", "rate", host, ts_s, 1.0)
  proc_dt = _ONLINE_STATS.update_interarrival("general", "interarrival", comm, ts_s, dt_scale_s=30.0)

  out: Dict[str, float] = {}
  for w in _ONLINE_WINDOWS:
    wn = w[0]
    out[f"proc_rate_{wn}"] = _rate01(proc_rate[wn][2])
    out[f"host_rate_{wn}"] = _rate01(host_rate[wn][2])
    out[f"proc_interarrival_{wn}"] = proc_dt[wn][0]
    out[f"proc_interarrival_std_{wn}"] = proc_dt[wn][1]
  return out


def extract_feature_dict(evt: Any, feature_view: str = "default") -> Dict[str, float]:
  """
  Extract features as a dict for streaming models.

  feature_view:
    - default: full encoded schema (one-hot, bucket banks, path tokens)
    - frequency: scalar-hash schema for frequency-family models (`event_id_norm`, comm/path/hostname hashes,
      type-specific *_hash); omits `flags_hash`, `file_sensitive_path`/`file_tmp_path`, and all online stats
    - loda: bounded numerics + compact identity/boolean blocks
    - memstream: dense-ish bounded numerics + smallest identity/boolean blocks

  Always includes general features. If evt.event_group is set (e.g. 'file', 'network',
  'process'), appends type-specific features. Features and thus vector size can differ
  per event group and per selected feature view.
  """
  view = _feature_view_spec(feature_view or "default")
  out = _extract_generic_features(evt, view=view)
  if view.include_general_online_stats:
    out.update(_extract_general_online_stats(evt))
  event_group = (evt.event_group or "").strip().lower()
  if event_group == "file":
    out.update(_extract_file_features(evt, view=view))
  elif event_group == "network":
    out.update(_extract_network_features(evt, view=view))
  elif event_group == "process":
    out.update(_extract_process_features(evt))
  return out
