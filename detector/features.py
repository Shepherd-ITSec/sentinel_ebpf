"""Feature extraction for anomaly detection."""
import hashlib
from functools import lru_cache
from typing import Dict, List

import numpy as np

import events_pb2


def _safe_int(raw: str, default: int = 0) -> int:
  try:
    return int(raw)
  except (ValueError, OverflowError):
    return default


@lru_cache(maxsize=8192)
def _hash01(value: str) -> float:
  digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
  return (int(digest, 16) % 10000) / 10000.0


def _extract_feature_values(evt: events_pb2.EventEnvelope) -> tuple[float, ...]:
  """
  Compute normalized feature values from an EventEnvelope.

  Features:
  - event_hash: Hash of event name (normalized to 0-1)
  - comm_hash: Hash of command name (normalized to 0-1)
  - path_hash: Hash of path argument (normalized to 0-1)
  - pid_normalized: Process ID (normalized)
  - tid_normalized: Thread ID (normalized)
  - uid_normalized: User ID (normalized)
  - arg0_norm: Event-specific numeric arg0 (log scale)
  - arg1_norm: Event-specific numeric arg1 (log scale)
  - hour_of_day: Hour component of timestamp
  - minute_of_hour: Minute component of timestamp

  Returns:
    tuple of 10 normalized feature values
  """
  # Canonical generic vector order:
  # [event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]
  event_name = evt.data[0] if len(evt.data) > 0 else (evt.event_type or "")
  comm = evt.data[2] if len(evt.data) > 2 else ""
  pid_str = evt.data[3] if len(evt.data) > 3 else "0"
  tid_str = evt.data[4] if len(evt.data) > 4 else "0"
  uid_str = evt.data[5] if len(evt.data) > 5 else "0"
  arg0_str = evt.data[6] if len(evt.data) > 6 else "0"
  arg1_str = evt.data[7] if len(evt.data) > 7 else "0"
  path = evt.data[8] if len(evt.data) > 8 else ""

  # Feature 1: Event name hash
  event_hash_normalized = _hash01(event_name)

  # Feature 2: Comm hash
  comm_hash_normalized = _hash01(comm)

  # Feature 3: Path hash
  path_hash_normalized = _hash01(path)

  # Feature 4: PID normalized (assuming max PID ~ 2^22)
  pid_val = max(0, _safe_int(pid_str, default=0))
  pid_normalized = min(pid_val / 4194304.0, 1.0)

  # Feature 5: TID normalized
  tid_val = max(0, _safe_int(tid_str, default=0))
  tid_normalized = min(tid_val / 4194304.0, 1.0)

  # Feature 6: UID normalized (assuming max UID ~ 2^32)
  uid_val = max(0, _safe_int(uid_str, default=0))
  uid_normalized = min(uid_val / 4294967295.0, 1.0)

  # Feature 7-8: generic numeric args (signed values mapped to positive magnitudes)
  arg0_val = abs(_safe_int(arg0_str, default=0))
  arg1_val = abs(_safe_int(arg1_str, default=0))
  arg0_norm = min(np.log1p(arg0_val) / 20.0, 1.0)
  arg1_norm = min(np.log1p(arg1_val) / 20.0, 1.0)

  # Feature 9: Hour of day (from timestamp)
  ts_ns = evt.ts_unix_nano
  ts_s = ts_ns // 1_000_000_000
  hour_of_day = (ts_s // 3600) % 24
  hour_normalized = hour_of_day / 24.0

  # Feature 10: Minute of hour
  minute_of_hour = (ts_s // 60) % 60
  minute_normalized = minute_of_hour / 60.0

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
  )


def extract_features(evt: events_pb2.EventEnvelope) -> np.ndarray:
  """Extract numerical features as a numpy vector for batch use."""
  return np.array(_extract_feature_values(evt), dtype=np.float32)


def extract_feature_dict(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """Extract features as a dict for streaming models (River)."""
  values = _extract_feature_values(evt)
  return {
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
  }


def extract_batch_features(events: List[events_pb2.EventEnvelope]) -> np.ndarray:
  """Extract features for a batch of events."""
  return np.array([extract_features(evt) for evt in events], dtype=np.float32)
