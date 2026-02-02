"""Optional model quality sanity checks (skipped by default)."""
import os

import numpy as np
import pytest

from detector.model import OnlineAnomalyDetector


RUN_OPTIONAL = os.environ.get("RUN_OPTIONAL_MODEL_TESTS") == "1"


@pytest.mark.skipif(not RUN_OPTIONAL, reason="set RUN_OPTIONAL_MODEL_TESTS=1 to run")
def test_loda_detects_anomaly_shift():
  rng = np.random.default_rng(0)

  def make_features(vec):
    return {f"f{i}": float(v) for i, v in enumerate(vec)}

  detector = OnlineAnomalyDetector(
    algorithm="loda",
    hst_n_trees=5,
    hst_height=5,
    hst_window_size=32,
    loda_n_projections=10,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    mem_hidden_dim=16,
    mem_latent_dim=4,
    mem_memory_size=32,
    mem_lr=0.01,
    seed=42,
  )

  for _ in range(200):
    x = rng.normal(0.0, 1.0, size=9)
    detector.score_and_learn(make_features(x))

  normal_scores = []
  for _ in range(20):
    x = rng.normal(0.0, 1.0, size=9)
    normal_scores.append(detector.score_and_learn(make_features(x)))

  anomaly_scores = []
  for _ in range(20):
    x = rng.normal(12.0, 1.0, size=9)
    anomaly_scores.append(detector.score_and_learn(make_features(x)))

  assert np.mean(anomaly_scores) > np.mean(normal_scores)


@pytest.mark.skipif(not RUN_OPTIONAL, reason="set RUN_OPTIONAL_MODEL_TESTS=1 to run")
def test_memstream_detects_anomaly_shift():
  rng = np.random.default_rng(1)

  def make_features(vec):
    return {f"f{i}": float(v) for i, v in enumerate(vec)}

  detector = OnlineAnomalyDetector(
    algorithm="memstream",
    hst_n_trees=5,
    hst_height=5,
    hst_window_size=32,
    loda_n_projections=10,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    mem_hidden_dim=16,
    mem_latent_dim=4,
    mem_memory_size=64,
    mem_lr=0.005,
    seed=7,
  )

  for _ in range(200):
    x = rng.normal(0.0, 1.0, size=9)
    detector.score_and_learn(make_features(x))

  normal_scores = []
  for _ in range(20):
    x = rng.normal(0.0, 1.0, size=9)
    normal_scores.append(detector.score_and_learn(make_features(x)))

  anomaly_scores = []
  for _ in range(20):
    x = rng.normal(6.0, 1.0, size=9)
    anomaly_scores.append(detector.score_and_learn(make_features(x)))

  assert np.mean(anomaly_scores) > np.mean(normal_scores) + 0.05
