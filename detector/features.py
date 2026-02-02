"""Feature extraction for anomaly detection."""
import hashlib
from typing import Dict, List, Optional

import numpy as np

import events_pb2


def extract_features(evt: events_pb2.EventEnvelope) -> np.ndarray:
  """
  Extract numerical features from an EventEnvelope for streaming models.
  
  Features:
  - bytes_or_flags: File operation size or open flags (log scale)
  - path_hash: Hash of file path (normalized to 0-1)
  - comm_hash: Hash of command name (normalized to 0-1)
  - pid_normalized: Process ID (normalized)
  - tid_normalized: Thread ID (normalized)
  - uid_normalized: User ID (normalized)
  - is_sensitive_path: Binary indicator for sensitive paths
  - path_depth: Depth of file path
  - hour_of_day: Hour component of timestamp
  - minute_of_hour: Minute component of timestamp
  
  Returns:
    numpy array of shape (9,) with normalized features
  """
  features = []
  
  # Extract data fields (ordered vector):
  # - file_open: [filename, flags, comm, pid, tid, uid]
  # - file_open: [filename, flags, comm, pid, tid, uid]
  if evt.event_type == "file_open":
    filename = evt.data[0] if len(evt.data) > 0 else ""
    bytes_str = evt.data[1] if len(evt.data) > 1 else "0"
    comm = evt.data[2] if len(evt.data) > 2 else ""
    pid_str = evt.data[3] if len(evt.data) > 3 else "0"
    tid_str = evt.data[4] if len(evt.data) > 4 else "0"
    uid_str = evt.data[5] if len(evt.data) > 5 else "0"
  else:
    filename = evt.data[0] if len(evt.data) > 0 else ""
    bytes_str = evt.data[1] if len(evt.data) > 1 else "0"
    comm = evt.data[2] if len(evt.data) > 2 else ""
    pid_str = evt.data[3] if len(evt.data) > 3 else "0"
    tid_str = evt.data[4] if len(evt.data) > 4 else "0"
    uid_str = "0"
  
  # Feature 1: Bytes or open flags (log scale, normalized)
  try:
    bytes_val = int(bytes_str)
    bytes_log = np.log1p(bytes_val)  # log(1+x) to handle 0
    bytes_normalized = min(bytes_log / 20.0, 1.0)  # Cap at log(1e8) ~ 20
  except (ValueError, OverflowError):
    bytes_normalized = 0.0
  
  # Feature 2: Path hash (normalized to 0-1)
  path_hash = int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)
  path_hash_normalized = (path_hash % 10000) / 10000.0
  
  # Feature 3: Comm hash (normalized to 0-1)
  comm_hash = int(hashlib.md5(comm.encode()).hexdigest()[:8], 16)
  comm_hash_normalized = (comm_hash % 10000) / 10000.0
  
  # Feature 4: PID normalized (assuming max PID ~ 2^22)
  try:
    pid_val = int(pid_str)
    pid_normalized = min(pid_val / 4194304.0, 1.0)
  except (ValueError, OverflowError):
    pid_normalized = 0.0
  
  # Feature 5: TID normalized
  try:
    tid_val = int(tid_str)
    tid_normalized = min(tid_val / 4194304.0, 1.0)
  except (ValueError, OverflowError):
    tid_normalized = 0.0
  
  # Feature 6: UID normalized (assuming max UID ~ 2^32)
  try:
    uid_val = int(uid_str)
    uid_normalized = min(uid_val / 4294967295.0, 1.0)
  except (ValueError, OverflowError):
    uid_normalized = 0.0

  # Feature 6: Is sensitive path (binary)
  sensitive_paths = ["/etc", "/bin", "/usr", "/sbin", "/boot", "/root", "/var/log"]
  is_sensitive = 1.0 if any(filename.startswith(p) for p in sensitive_paths) else 0.0
  
  # Feature 7: Path depth
  path_depth = filename.count("/")
  path_depth_normalized = min(path_depth / 20.0, 1.0)
  
  # Feature 8: Hour of day (from timestamp)
  ts_ns = evt.ts_unix_nano
  ts_s = ts_ns // 1_000_000_000
  hour_of_day = (ts_s // 3600) % 24
  hour_normalized = hour_of_day / 24.0
  
  # Feature 9: Minute of hour
  minute_of_hour = (ts_s // 60) % 60
  minute_normalized = minute_of_hour / 60.0
  
  return np.array([
    bytes_normalized,
    path_hash_normalized,
    comm_hash_normalized,
    pid_normalized,
    tid_normalized,
    uid_normalized,
    is_sensitive,
    path_depth_normalized,
    hour_normalized,
    minute_normalized,
  ], dtype=np.float32)


def extract_feature_dict(evt: events_pb2.EventEnvelope) -> Dict[str, float]:
  """Extract features as a dict for streaming models (River)."""
  values = extract_features(evt)
  return {
    "bytes_norm": float(values[0]),
    "path_hash": float(values[1]),
    "comm_hash": float(values[2]),
    "pid_norm": float(values[3]),
    "tid_norm": float(values[4]),
    "uid_norm": float(values[5]),
    "is_sensitive": float(values[6]),
    "path_depth": float(values[7]),
    "hour_norm": float(values[8]),
    "minute_norm": float(values[9]),
  }


def extract_batch_features(events: List[events_pb2.EventEnvelope]) -> np.ndarray:
  """Extract features for a batch of events."""
  return np.array([extract_features(evt) for evt in events], dtype=np.float32)
