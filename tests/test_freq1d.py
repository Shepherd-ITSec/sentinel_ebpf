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
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9, "comm_hash": 0.1234, "return_success": 1.0}
  for _ in range(10):
    raw = det.score_and_learn_raw(features)
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_freq1d_shifted_distribution_scores_higher():
  det = OnlineAnomalyDetector(
    algorithm="freq1d",
    freq1d_bins=64,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=512,
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
    det.score_and_learn_raw(features)

  normal = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    normal.append(det.score_only_raw(features))

  shifted = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "comm_hash": 0.9999,  # rare/unseen category
      "return_success": 1.0,
    }
    shifted.append(det.score_only_raw(features))

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

  full = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, seed=7)
  for i in range(100):
    full.score_and_learn_raw(events[i])
  full_score = full.score_only_raw(events[100])

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, seed=7)
    for i in range(100):
      ckpt_det.score_and_learn_raw(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(algorithm="freq1d", freq1d_bins=32, freq1d_max_categories=128, seed=7)
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn_raw(events[i])
    loaded_score = loaded.score_only_raw(events[100])
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
    seed=9,
  )
  for i in range(200):
    features = {
      "comm_hash": float((i % 10000) / 10000.0),
      "x_norm": float((i % 16) / 15.0),
      "return_success": float(i % 2),
    }
    det.score_and_learn_raw(features)

  impl = det.impl
  assert impl.algorithm == "freq1d"
  assert impl._cat_counts is not None
  # Only comm_hash and return_success are categorical -> ensure per-feature cap holds.
  for d in impl._cat_counts:
    assert len(d) <= impl.max_categories
