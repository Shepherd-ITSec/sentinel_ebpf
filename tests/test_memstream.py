"""Focused tests for MemStream-style online detector."""

import numpy as np
import torch

from detector.model import OnlineAnomalyDetector


def _make_features(vec):
  return {f"f{i}": float(v) for i, v in enumerate(vec)}


def test_memstream_scores_in_unit_interval():
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
    mem_memory_size=32,
    mem_lr=0.005,
    model_device="auto",
    seed=7,
  )
  rng = np.random.default_rng(7)
  for _ in range(20):
    score = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    assert 0.0 <= score <= 1.0


def test_memstream_gates_memory_updates_for_extreme_anomalies():
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
    mem_memory_size=16,
    mem_lr=0.005,
    model_device="auto",
    seed=11,
  )
  rng = np.random.default_rng(11)

  for _ in range(80):
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))

  impl = detector.impl
  assert impl.algorithm == "memstream"
  before_idx = impl._mem_index

  for _ in range(20):
    detector.score_and_learn(_make_features(rng.normal(30.0, 1.0, size=9)))

  after_idx = impl._mem_index
  # Most extreme outliers should be blocked from memory updates.
  assert (after_idx - before_idx) % impl.memory_size <= 4


def test_memstream_auto_selects_expected_device():
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
    mem_memory_size=16,
    mem_lr=0.005,
    model_device="auto",
    seed=13,
  )
  detector.score_and_learn(_make_features(np.zeros(9)))
  impl = detector.impl
  assert impl.algorithm == "memstream"
  expected = "cuda" if torch.cuda.is_available() else "cpu"
  assert impl._device.type == expected
