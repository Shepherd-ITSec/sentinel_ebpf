"""Feature extraction for anomaly detection.

EventEnvelope carries syscall fields by name (syscall_nr, comm, pid, tid, uid, arg0, arg1, path);
syscall_name is the Linux syscall name; event_group is the rule-defined category (any declared group name, or empty).
"""
import hashlib
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

from detector.config import load_config
from detector.embeddings.online_word2vec import OnlineTokenWord2Vec
from detector.meta import Meta
from detector.sequence.context import SequenceFeatureMeta, SequenceVectorContext, TokenClassTable
from rules_config import RulesConfig, load_rules_config


@dataclass(frozen=True)
class _FeatureViewSpec:
  include_sequence: bool = False
  include_hashes: bool = False # Only for frequency based views
  include_general_process_context: bool = True
  include_general_time_context: bool = True
  include_general_path_context: bool = True
  include_general_return_context: bool = True

  include_file_sensitive_tmp: bool = True
  include_file_flags: bool = True

  include_network_ports: bool = True
  include_network_socket_type: bool = True

  # `day_cycle` emits a full-day sin/cos pair. `day_fraction` emits a scalar in [0, 1].
  time_feature_mode: str = "day_cycle"


_REPO_RULES_PATH = Path(__file__).resolve().parents[1] / "charts" / "sentinel-ebpf" / "rules.yaml"
_DEFAULT_SENSITIVE_PATH_PREFIXES = ("/etc", "/root", "/bin", "/usr/bin", "/sbin", "/usr/sbin")
_DEFAULT_TMP_PATH_PREFIXES = ("/tmp", "/var/tmp")

# Normalization scales
_PID_MAX = 4_194_304
_UID_MAX = 4_294_967_295
_ARG_LOG_SCALE = 20.0
_PATH_DEPTH_CAP = 20.0
_RETURN_ERRNO_SCALE = 12.0
_SYSCALL_NR_MAX = 500.0
_PORT_MAX = 65535.0
_ADDRLEN_SCALE = 10.0
_SOCKET_FAMILY_MAX = 10.0
_SEQUENCE_CONTEXT_PREFIX = "sequence_ctx"
_DAY_NS = 86_400_000_000_000

# Feature views
_FULL_FEATURE_VIEW = _FeatureViewSpec()
_DEFAULT_FEATURE_VIEW = _FeatureViewSpec(
  include_general_process_context=False,
  include_general_time_context=False,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_flags=False,
  include_network_ports=False,
)
_FREQUENCY_FEATURE_VIEW = _FeatureViewSpec(
  include_hashes=True,
  include_general_process_context=False,
  include_general_time_context=False,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_sensitive_tmp=False,
  include_file_flags=False,
  include_network_ports=False,
  include_network_socket_type=False,
  time_feature_mode="day_fraction",
)
_SEQUENCE_FEATURE_VIEW = _FeatureViewSpec(
  # Sequence models: add sequence context + (optional) time features.
  include_sequence=True,
  include_hashes=False,
  include_general_process_context=False,
  include_general_time_context=True,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_sensitive_tmp=False,
  include_file_flags=False,
  include_network_ports=False,
  include_network_socket_type=False,
  time_feature_mode="day_cycle",
)


class FeatureExtractor:
  """
  Stateful feature extractor that keeps sequence context state and wraps the extraction logic.
  """

  def __init__(
    self,
    *,
    sequence_w2v: OnlineTokenWord2Vec,
    sequence_classes: TokenClassTable,
    sequence_context: SequenceVectorContext,
  ) -> None:
    self._sequence_w2v = sequence_w2v
    self._sequence_classes = sequence_classes
    self._sequence_context = sequence_context

  @classmethod
  def from_config(cls, cfg: Any) -> "FeatureExtractor":
    emb_dim = int(getattr(cfg, "embedding_word2vec_dim"))
    return cls(
      sequence_w2v=OnlineTokenWord2Vec(
        vector_size=emb_dim,
        sentence_len=int(getattr(cfg, "embedding_word2vec_sentence_len")),
        seed=int(getattr(cfg, "model_seed")),
        w2v_window=int(getattr(cfg, "embedding_word2vec_window")),
        w2v_sg=int(getattr(cfg, "embedding_word2vec_sg")),
        update_every=int(getattr(cfg, "embedding_word2vec_update_every")),
        epochs=int(getattr(cfg, "embedding_word2vec_epochs")),
        warmup_events=int(getattr(cfg, "warmup_events")),
        post_warmup_lr_scale=float(getattr(cfg, "embedding_word2vec_post_warmup_lr_scale")),
      ),
      sequence_classes=TokenClassTable(),
      sequence_context=SequenceVectorContext(
        element_width=emb_dim,
        ngram_length=int(getattr(cfg, "sequence_ngram_length")),
        thread_aware=bool(getattr(cfg, "sequence_thread_aware")),
        feature_prefix=_SEQUENCE_CONTEXT_PREFIX,
      ),
    )

  def _extract_sequence_features(self, evt: Any) -> tuple[Dict[str, float], SequenceFeatureMeta]:
    try:
      actor_id = int((getattr(evt, "tid", "0")).strip())
    except ValueError:
      actor_id = 0
    token = (getattr(evt, "syscall_name", "")).strip().lower()
    target_id = self._sequence_classes.get_label_idx(token)
    w2v_emb = self._sequence_w2v.observe_and_vector(int(actor_id), token)
    out, meta = self._sequence_context.observe_vector(
      stream_id=int(actor_id),
      vector=w2v_emb.tolist(),
      target_id=int(target_id),
      num_classes=int(self._sequence_classes.num_classes),
    )
    return out, meta

  def extract_feature_dict(self, evt: Any, feature_view: str = "default") -> tuple[Dict[str, float], Meta | None]:
    """Return (feature dict, meta). ``meta`` is set only for the sequence feature view."""
    meta: Meta | None = None
    view = _feature_view_spec(feature_view or "default")
    out = _extract_generic_features(evt, view)
    if view.include_sequence:
      seq, meta = self._extract_sequence_features(evt)
      out.update(seq)
    event_group = (evt.event_group or "").strip().lower() or "__empty__"
    out.update(_extract_group_features(evt, view, event_group))
    return out, meta

  def get_state(self) -> dict:
    return {
      "sequence_context": {
        "classes": self._sequence_classes.to_serializable(),
        "w2v": self._sequence_w2v.get_state(),
        "context": self._sequence_context.get_state(),
      },
    }

  def set_state(self, state: dict) -> None:
    seq_state = state.get("sequence_context", None)
    if isinstance(seq_state, dict):
      self._sequence_classes.load_from_pairs(list(seq_state.get("classes", []) or []))
      w2v_state = seq_state.get("w2v", None)
      if isinstance(w2v_state, dict):
        self._sequence_w2v.set_state(w2v_state)
      ctx_state = seq_state.get("context", None)
      if isinstance(ctx_state, dict):
        self._sequence_context.set_state(ctx_state)




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
  gcfg = cfg.groups.get(target)
  if gcfg is None or not gcfg.syscalls:
    return tuple(sorted(set(fallback)))
  return tuple(sorted(gcfg.syscalls))


def _get_group_feature_value(event_group: str, feature_name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
  """
  Get the feature values for a given event group and feature name from the rules config.

  Args:
    event_group: The event group to get the feature values for.
    feature_name: The feature name to get the feature values for.
    fallback: The fallback values to use if the event group or feature name is not found.

  Returns:
    A tuple of feature values.
  """
  cfg = _rules_config()
  fallback_values = tuple(sorted(set(fallback)))
  if cfg is None: # no rules config, use fallback values
    return fallback_values
  group = cfg.groups.get((event_group or "").strip().lower())
  if group is None: # no group found, use fallback values
    return fallback_values
  values = group.features.get((feature_name or "").strip().lower(), ())
  if not values: # no values found, use fallback values
    return fallback_values
  return tuple(sorted(set(values))) # return the real values as a tuple


def feature_view_for_algorithm(algorithm: str | None) -> str:
  """Map algorithm to feature view."""
  algo = (algorithm or "").strip().lower()
  if algo in ("sequence_mlp", "sequence_transformer"):
    return "sequence"
  if algo in ("freq1d", "copulatree", "latentcluster", "zscore"):
    return "frequency"
  return "default"


def _feature_view_spec(feature_view: str | None) -> _FeatureViewSpec:
  normalized = (feature_view or "default").strip().lower()
  if normalized == "sequence":
    return _SEQUENCE_FEATURE_VIEW
  if normalized == "frequency":
    return _FREQUENCY_FEATURE_VIEW
  if normalized in ("full"):
    return _FULL_FEATURE_VIEW
  else: # default
    return _DEFAULT_FEATURE_VIEW


def _extract_time_features(ts_ns: int, *, mode: str) -> Dict[str, float]:
  """
  Extract time features based on the mode.

  Args:
    ts_ns: The timestamp in nanoseconds.
    mode: The time feature mode (day_cycle or day_fraction).

  Returns:
    A dictionary of time features (day_cycle_sin, day_cycle_cos, day_fraction_norm).
  """
  day_fraction = (float(int(ts_ns) % _DAY_NS) / float(_DAY_NS)) if _DAY_NS > 0 else 0.0
  if mode == "day_cycle":
    day_angle = 2.0 * float(np.pi) * day_fraction
    return {
      "day_cycle_sin": float(np.sin(day_angle)),
      "day_cycle_cos": float(np.cos(day_angle)),
    }
  if mode == "day_fraction":
    return {"day_fraction_norm": day_fraction}
  raise ValueError(f"Unknown time_feature_mode={mode!r}")


def _onehot_features(prefix: str, selected: str, known_values: tuple[str, ...]) -> Dict[str, float]:
  """
  Extract one-hot encoding for features based on the selected value.

  Args:
    prefix: The prefix for the feature name.
    selected: The selected value.
    known_values: The known values to use for the one-hot encoding.

  Returns:
    A dictionary of one-hot features.
  """
  selected_norm = (selected or "").strip().lower()
  out: Dict[str, float] = {}
  for value in known_values:
    out[f"{prefix}_{_sanitize_feature_name(value)}"] = 1.0 if selected_norm == value else 0.0
  return out


def _parse_sockaddr_from_evt(evt: Any) -> Dict[str, str]:
  """
  Socket endpoint fields for network features, from ``attributes.fd_sock_*``.
  Returns dict with keys sin_port, sin_addr, sa_family (internal names; values from remote endpoint).
  """
  out: Dict[str, str] = {"sin_port": "", "sin_addr": "", "sa_family": ""}
  attrs = dict(evt.attributes or {})
  out["sin_port"] = (attrs.get("fd_sock_remote_port") or "").strip()
  out["sin_addr"] = (attrs.get("fd_sock_remote_addr") or "").strip()
  out["sa_family"] = (attrs.get("fd_sock_family") or "").strip()
  return out


def _extract_generic_features(evt: Any, view: _FeatureViewSpec) -> Dict[str, float]:
  """
  General layer: features shared across every event 
  (syscall_nr_norm, time)
  """
  out: Dict[str, float] = {}
  pid_val = _safe_int(evt.pid or "0", default=0)
  attrs = dict(evt.attributes or {})
  path = str(attrs.get("fd_path", "") or "").strip()
  
  # process context
  if view.include_general_process_context:
    tid_val = _safe_int(evt.tid or "0", default=0)
    uid_val = _safe_int(evt.uid or "0", default=0)
    arg0_val = _safe_int(evt.arg0 or "0", default=0)
    arg1_val = _safe_int(evt.arg1 or "0", default=0)
    out.update({
      "pid_norm": _norm_pid(pid_val),
      "tid_norm": _norm_pid(tid_val),
      "uid_norm": _norm_uid(uid_val),
      "arg0_norm": _norm_arg(arg0_val),
      "arg1_norm": _norm_arg(arg1_val),
    }) 
  # time context
  ts_ns = evt.ts_unix_nano
  if view.include_general_time_context:
    ts_s = ts_ns // 1_000_000_000
    dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
    day_of_month = dt.day
    week_of_month = min(4, (day_of_month - 1) // 7 + 1)
    out["week_of_month_norm"] = (week_of_month - 1) / 3.0
  # path context
  if view.include_general_path_context:
    components = _path_components(path)
    path_prefix = components[0] if components else ""
    out.update({
      "path_depth_norm": _norm_path_depth(components),
      "path_prefix_hash": _hash01(path_prefix),
    })
  # return context
  if view.include_general_return_context:
    rv = _safe_int(attrs.get("return_value", "0"), default=0)
    return_success, return_errno_norm = _norm_return_errno(rv)
    out.update({
      "return_success": return_success,
      "return_errno_norm": return_errno_norm,
    })  
  # hashes
  if view.include_hashes:
    out["hostname_hash"] = _hash01(str(evt.hostname or ""))
    out["pid_hash"] = _hash01(str(evt.pid or "0"))
    out["path_hash"] = _hash01(path)
    components = _path_components(path)
    path_prefix = components[0] if components else ""
    out["path_prefix_hash"] = _hash01(path_prefix)
  # Always include the normalized syscall number + some kind of time.
  syscall_nr_val = max(0, _safe_int(str(int(evt.syscall_nr)), default=0))
  out["syscall_nr_norm"] = min(syscall_nr_val / _SYSCALL_NR_MAX, 1.0)
  out.update(_extract_time_features(ts_ns, mode=view.time_feature_mode))
  
  return out


def _extract_file_group_features(evt: Any, view: _FeatureViewSpec, event_group: str) -> Dict[str, float]:
  """
  File layer: sensitive/tmp path flags and extension/flags (``file_*``).
  Only for ``event_group == "file"``.

  Currently, does not add any features without include_file_sensitive_tmp or include_file_flags.
  """
  attrs = dict(evt.attributes or {})
  path = str(attrs.get("fd_path", "") or "").strip()
  g = (event_group or "").strip().lower()

  out: Dict[str, float] = {}
  if view.include_file_sensitive_tmp:
    path_lower = path.lower()
    sensitive_prefixes = _get_group_feature_value(g, "sensitive_paths", _DEFAULT_SENSITIVE_PATH_PREFIXES)
    tmp_prefixes = _get_group_feature_value(g, "tmp_paths", _DEFAULT_TMP_PATH_PREFIXES)
    file_sensitive_path = 1.0 if any(path_lower.startswith(p) for p in sensitive_prefixes) else 0.0
    file_tmp_path = 1.0 if any(path_lower.startswith(p) for p in tmp_prefixes) else 0.0
    out["file_sensitive_path"] = file_sensitive_path
    out["file_tmp_path"] = file_tmp_path
  if view.include_file_flags:
    flags_str = attrs.get("flags")  or ""
    out["file_flags_hash"] = _hash01(flags_str)
  return out


def _extract_proc_group_features(evt: Any, view: _FeatureViewSpec, event_group: str) -> Dict[str, float]:
  """
  Process layer: path semantics for process syscalls (``proc_*``).
  Only for ``event_group == "process"``.

  Currently, does not add any features.
  """
  out: Dict[str, float] = {}
  return out


def _extract_network_group_features(evt: Any, view: _FeatureViewSpec) -> Dict[str, float]:
  """
  Network layer: sockaddr-derived ``net_*`` features. Only for ``event_group == \"network\"``.
  Only for ``event_group == "network"``.

  Currently, does not add any features without include_network_ports or include_network_socket_type or include_hashs.
  """
  out: Dict[str, float] = {}
  arg0_val = _safe_int(evt.arg0 or "0", default=0)
  arg1_val = _safe_int(evt.arg1 or "0", default=0)
  syscall_display = evt.syscall_name or ""

  sockaddr = _parse_sockaddr_from_evt(evt)
  if view.include_network_socket_type:
    family_val = 0
    if syscall_display == "socket":
      family_val = arg0_val
    else:
      af = (sockaddr.get("sa_family") or "").upper()
      if "INET6" in af or af == "10":
        family_val = 10
      elif "INET" in af or af == "2":
        family_val = 2
    net_socket_family_norm = min(family_val / _SOCKET_FAMILY_MAX, 1.0)
    out["net_socket_family_norm"] = net_socket_family_norm

  type_val = arg1_val if syscall_display == "socket" else 0
  port_str = sockaddr.get("sin_port") or ""
  dport = _safe_int(port_str, default=0) if port_str else 0
  if view.include_network_ports:
    net_dport_norm = min(dport / _PORT_MAX, 1.0) if dport else 0.0
    out["net_dport_norm"] = net_dport_norm

  daddr = sockaddr.get("sin_addr") or ""
  if view.include_hashes:
    out["net_socket_type_hash"] = _hash01(str(type_val))
    out["net_daddr_hash"] = _hash01(daddr)
    af_str = sockaddr.get("sa_family") or ("AF_INET" if family_val == 2 else "AF_INET6" if family_val == 10 else "")
    out["net_af_hash"] = _hash01(af_str)
  return out


def _extract_group_features(
  evt: Any,
  view: _FeatureViewSpec,
  event_group: str,
) -> Dict[str, float]:
  group = (event_group or "").strip().lower()

  out: Dict[str, float] = {}
  if group == "network": 
    # network
    out.update(_extract_network_group_features(evt, view))
  elif group == "process": 
    # process
    out.update(_extract_proc_group_features(evt, view, event_group))
  elif group == "file": 
    # file
    out.update(_extract_file_group_features(evt, view, event_group))
  # custom groups do not have a specific layer, so we don't extract any features for them
  return out


def build_feature_extractor(cfg: Any | None = None) -> FeatureExtractor:
  if cfg is None:
    cfg = load_config()
  return FeatureExtractor.from_config(cfg)


@lru_cache(maxsize=1)
def _default_feature_extractor() -> FeatureExtractor:
  return build_feature_extractor()


def extract_feature_dict(evt: Any, feature_view: str = "default") -> tuple[Dict[str, float], Meta | None]:
  """Return ``(features, meta)`` for a single event. ``meta`` is ``None`` except for the sequence view."""
  return _default_feature_extractor().extract_feature_dict(evt, feature_view=feature_view)
