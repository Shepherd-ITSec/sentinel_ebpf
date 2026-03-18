"""Focused tests for Freq1D frequency baseline detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def test_freq1d_raw_score_finite_and_non_negative():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=32,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=128,
    freq1d_aggregation="mean",
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9, "comm_hash": 0.1234, "return_success": 1.0}
  for _ in range(10):
    raw, _ = det.score_and_learn(features)
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_freq1d_shifted_distribution_scores_higher():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=64,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=512,
    freq1d_aggregation="mean",
    seed=2,
  )
  rng = np.random.default_rng(2)

  # Learn a tight numeric distribution around 0.1 and a small categorical set.
  for _ in range(400):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    det.score_and_learn(features)

  normal = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    normal.append(det.score_only(features)[0])

  shifted = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "comm_hash": 0.9999,  # rare/unseen category
      "return_success": 1.0,
    }
    shifted.append(det.score_only(features)[0])

  assert float(np.mean(shifted)) > float(np.mean(normal))


def test_freq1d_checkpoint_save_load_preserves_scores():
  rng = np.random.default_rng(42)
  events = []
  for i in range(101):
    events.append({
      "x_norm": float(np.clip(rng.normal(0.2, 0.05), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.6, 0.05), 0.0, 1.0)),
      "comm_hash": float((i % 10) / 100.0),
      "return_success": float(i % 2),
    })

  full = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, freq1d_aggregation="mean", seed=7)
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, freq1d_aggregation="mean", seed=7)
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, freq1d_aggregation="mean", seed=7)
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])[0]
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)


def test_freq1d_categorical_cap_does_not_explode():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=16,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=8,
    freq1d_aggregation="mean",
    seed=9,
  )
  for i in range(200):
    features = {
      "comm_hash": float((i % 10000) / 10000.0),
      "x_norm": float((i % 16) / 15.0),
      "return_success": float(i % 2),
    }
    det.score_and_learn(features)

  impl = det.impl
  assert impl.algorithm == "freq1d"
  assert impl._cat_counts is not None
  # Only comm_hash and return_success are categorical -> ensure per-feature cap holds.
  for d in impl._cat_counts:
    assert len(d) <= impl.max_categories


def test_freq1d_treats_bucket_and_event_onehot_features_as_categorical():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=16,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=8,
    freq1d_aggregation="mean",
    seed=10,
  )
  features = {
    "event_name_openat": 1.0,
    "event_name_connect": 0.0,
    "comm_bucket_000": 1.0,
    "comm_bucket_001": 0.0,
    "hostname_bucket_000": 1.0,
    "mount_ns_bucket_000": 1.0,
    "path_tok_d0_bucket_000": 1.0,
    "return_success": 1.0,
    "x_norm": 0.25,
  }
  det.score_and_learn(features)

  impl = det.impl
  assert impl.algorithm == "freq1d"
  assert impl._feature_names is not None
  categorical_names = {
    name
    for name, kind in zip(impl._feature_names, impl._kind)
    if kind == "cat"
  }
  assert "event_name_openat" in categorical_names
  assert "comm_bucket_000" in categorical_names
  assert "hostname_bucket_000" in categorical_names
  assert "mount_ns_bucket_000" in categorical_names
  assert "path_tok_d0_bucket_000" in categorical_names
  assert "x_norm" not in categorical_names


def test_freq1d_treats_type_specific_encoded_banks_as_categorical():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=16,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=8,
    freq1d_aggregation="mean",
    seed=11,
  )
  features = {
    "file_event_name_openat": 1.0,
    "file_extension_bucket_000": 1.0,
    "file_flags_bucket_000": 1.0,
    "net_socket_type_bucket_000": 1.0,
    "net_daddr_bucket_000": 1.0,
    "net_af_af_inet": 1.0,
    "net_fd_norm": 0.25,
  }
  det.score_and_learn(features)

  impl = det.impl
  assert impl.algorithm == "freq1d"
  assert impl._feature_names is not None
  categorical_names = {
    name
    for name, kind in zip(impl._feature_names, impl._kind)
    if kind == "cat"
  }
  assert "file_event_name_openat" in categorical_names
  assert "file_extension_bucket_000" in categorical_names
  assert "file_flags_bucket_000" in categorical_names
  assert "net_socket_type_bucket_000" in categorical_names
  assert "net_daddr_bucket_000" in categorical_names
  assert "net_af_af_inet" in categorical_names
  assert "net_fd_norm" not in categorical_names


def test_freq1d_aggregation_modes_ordering():
  """sum >= mean; topk_mean(k=1) ~= max; soft_topk_mean is between mean and max (typical)."""
  rng = np.random.default_rng(123)
  # Warm up a detector to avoid cold-start effects dominating.
  base_events = []
  for _ in range(200):
    base_events.append({
      "x_norm": float(np.clip(rng.normal(0.2, 0.05), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.4, 0.05), 0.0, 1.0)),
      "z_norm": float(np.clip(rng.normal(0.6, 0.05), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    })

  feat = {
    "x_norm": 0.95,
    "y_norm": 0.95,
    "z_norm": 0.95,
    "comm_hash": 0.9999,
    "return_success": 1.0,
  }

  def run(aggregation: str, topk: int = 8, temp: float = 0.25) -> float:
    det = OnlineAnomalyDetector(
      algorithm="freq1d",
      freq1d_bins=64,
      freq1d_max_categories=256,
      freq1d_alpha=1.0,
      freq1d_decay=1.0,
      freq1d_aggregation=aggregation,
      freq1d_topk=topk,
      freq1d_soft_topk_temperature=temp,
      seed=1,
    )
    for e in base_events:
      det.score_and_learn(e)
    return float(det.score_only(feat)[0])

  s_sum = run("sum")
  s_mean = run("mean")
  s_top1 = run("topk_mean", topk=1)
  s_soft = run("soft_topk_mean", topk=8, temp=0.25)

  assert s_sum >= s_mean
  assert s_top1 >= s_mean
  assert s_soft >= s_mean
  assert s_soft <= s_top1 + 1e-9
