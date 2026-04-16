from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from detector.building_blocks.primitives.features.views import _FeatureViewSpec

_PID_MAX = 4_194_304
_UID_MAX = 4_294_967_295
_ARG_LOG_SCALE = 20.0
_PATH_DEPTH_CAP = 20.0
_RETURN_ERRNO_SCALE = 12.0
_SYSCALL_NR_MAX = 500.0
_PORT_MAX = 65535.0
_SOCKET_FAMILY_MAX = 10.0
_DAY_NS = 86_400_000_000_000


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


def _hash01(value: str) -> float:
  if not value:
    return 0.0
  digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
  return (int(digest, 16) % 10000) / 10000.0


def _path_components(path: str) -> list:
  if not path:
    return []
  return [p for p in path.strip().strip("/").split("/") if p]


def _sanitize_feature_name(value: str) -> str:
  cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
  return cleaned or "unknown"


def _extract_time_features(ts_ns: int, *, mode: str) -> Dict[str, float]:
  day_fraction = (float(int(ts_ns) % _DAY_NS) / float(_DAY_NS)) if _DAY_NS > 0 else 0.0
  if mode == "day_cycle":
    day_angle = 2.0 * float(np.pi) * day_fraction
    return {"day_cycle_sin": float(np.sin(day_angle)), "day_cycle_cos": float(np.cos(day_angle))}
  if mode == "day_fraction":
    return {"day_fraction_norm": day_fraction}
  raise ValueError(f"Unknown time_feature_mode={mode!r}")


def _parse_sockaddr_from_evt(evt: Any) -> Dict[str, str]:
  out: Dict[str, str] = {"sin_port": "", "sin_addr": "", "sa_family": ""}
  attrs = dict(evt.attributes or {})
  out["sin_port"] = (attrs.get("fd_sock_remote_port") or "").strip()
  out["sin_addr"] = (attrs.get("fd_sock_remote_addr") or "").strip()
  out["sa_family"] = (attrs.get("fd_sock_family") or "").strip()
  return out


def _extract_generic_features(evt: Any, view: _FeatureViewSpec) -> Dict[str, float]:
  out: Dict[str, float] = {}
  pid_val = _safe_int(evt.pid or "0", default=0)
  attrs = dict(evt.attributes or {})
  path = str(attrs.get("fd_path", "") or "").strip()
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
  ts_ns = evt.ts_unix_nano
  if view.include_general_time_context:
    ts_s = ts_ns // 1_000_000_000
    dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
    day_of_month = dt.day
    week_of_month = min(4, (day_of_month - 1) // 7 + 1)
    out["week_of_month_norm"] = (week_of_month - 1) / 3.0
  if view.include_general_path_context:
    components = _path_components(path)
    path_prefix = components[0] if components else ""
    out.update({"path_depth_norm": _norm_path_depth(components), "path_prefix_hash": _hash01(path_prefix)})
  if view.include_general_return_context:
    rv = _safe_int(attrs.get("return_value", "0"), default=0)
    return_success, return_errno_norm = _norm_return_errno(rv)
    out.update({"return_success": return_success, "return_errno_norm": return_errno_norm})
  if view.include_hashes:
    out["hostname_hash"] = _hash01(str(evt.hostname or ""))
    out["pid_hash"] = _hash01(str(evt.pid or "0"))
    out["path_hash"] = _hash01(path)
    components = _path_components(path)
    path_prefix = components[0] if components else ""
    out["path_prefix_hash"] = _hash01(path_prefix)
  syscall_nr_val = max(0, _safe_int(str(int(evt.syscall_nr)), default=0))
  out["syscall_nr_norm"] = min(syscall_nr_val / _SYSCALL_NR_MAX, 1.0)
  out.update(_extract_time_features(ts_ns, mode=view.time_feature_mode))
  return out
