"""Tests for detector/features.py."""
import numpy as np

import events_pb2
from detector.features import extract_batch_features, extract_feature_dict, extract_features


def test_extract_features_returns_expected_shape():
  evt = events_pb2.EventEnvelope(
    event_id="feat-1",
    event_type="openat",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "cat", "123", "124", "1000", "-100", "2", "/etc/passwd", "2"],
  )
  vec = extract_features(evt)
  assert vec.shape == (10,)
  assert vec.dtype == np.float32


def test_extract_features_handles_bad_and_negative_values():
  evt = events_pb2.EventEnvelope(
    event_id="feat-2",
    event_type="openat",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "bash", "bad-pid", "-1", "bad-uid", "-9", "nope", "/tmp/test", "x"],
  )
  vec = extract_features(evt)
  assert np.isfinite(vec).all()
  # pid/tid/uid should be clamped to valid non-negative domain.
  assert float(vec[3]) == 0.0
  assert float(vec[4]) == 0.0
  assert float(vec[5]) == 0.0


def test_extract_feature_dict_has_stable_keys():
  evt = events_pb2.EventEnvelope(
    event_id="feat-3",
    event_type="openat",
    ts_unix_nano=1_700_000_000_000_000_000,
    data=["openat", "2", "ls", "10", "10", "0", "-100", "64", "/bin/ls", "64"],
  )
  values = extract_feature_dict(evt)
  assert set(values.keys()) == {
    "event_hash",
    "comm_hash",
    "path_hash",
    "pid_norm",
    "tid_norm",
    "uid_norm",
    "arg0_norm",
    "arg1_norm",
    "hour_norm",
    "minute_norm",
  }


def test_extract_batch_features_stacks_vectors():
  events = [
    events_pb2.EventEnvelope(
      event_id=f"feat-batch-{i}",
      event_type="openat",
      ts_unix_nano=1_700_000_000_000_000_000 + i,
      data=["openat", "2", "bash", "1", "1", "1000", "-100", "2", f"/tmp/{i}", "2"],
    )
    for i in range(3)
  ]
  batch = extract_batch_features(events)
  assert batch.shape == (3, 10)
