"""Focused tests for LODA-style online detector."""

import numpy as np

from detector.model import OnlineAnomalyDetector


def _make_features(vec):
  return {f"f{i}": float(v) for i, v in enumerate(vec)}


def test_loda_scores_in_unit_interval():
  detector = OnlineAnomalyDetector(
    algorithm="loda",
    hst_n_trees=5,
    hst_height=5,
    hst_window_size=32,
    loda_n_projections=16,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    mem_hidden_dim=16,
    mem_latent_dim=4,
    mem_memory_size=32,
    mem_lr=0.005,
    seed=3,
  )
  rng = np.random.default_rng(3)
  for _ in range(30):
    score = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    assert 0.0 <= score <= 1.0


def test_loda_anomaly_shift_scores_higher():
  detector = OnlineAnomalyDetector(
    algorithm="loda",
    hst_n_trees=5,
    hst_height=5,
    hst_window_size=32,
    loda_n_projections=20,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.03,
    loda_hist_decay=1.0,
    mem_hidden_dim=16,
    mem_latent_dim=4,
    mem_memory_size=32,
    mem_lr=0.005,
    seed=5,
  )
  rng = np.random.default_rng(5)
  for _ in range(200):
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))

  normal_scores = [
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    for _ in range(25)
  ]
  anomaly_scores = [
    detector.score_and_learn(_make_features(rng.normal(8.0, 1.0, size=9)))
    for _ in range(25)
  ]
  assert np.mean(anomaly_scores) > np.mean(normal_scores)
