"""Tests for standalone indep_marginal (product of 1D marginals) detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.features import feature_view_for_algorithm
from detector.model import OnlineAnomalyDetector


def test_indep_marginal_feature_view_is_frequency():
  assert feature_view_for_algorithm("indep_marginal") == "frequency"


def test_indep_marginal_raw_bounded_and_cold_start_product():
  det = OnlineAnomalyDetector(
    algorithm="indep_marginal",
    freq1d_bins=32,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=128,
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9}
  raw, scaled = det.score_and_learn(features)
  # Geometric mean of two cold-start marginals: (1/bins)^(1/2) per dim -> 1 - 1/bins
  expected = 1.0 - 1.0 / 32.0
  assert np.isfinite(raw)
  assert 0.0 <= raw <= 1.0
  assert abs(raw - expected) < 1e-9
  assert scaled == raw


def test_indep_marginal_repeated_same_event_lowers_score_vs_cold_start():
  det = OnlineAnomalyDetector(
    algorithm="indep_marginal",
    freq1d_bins=64,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=256,
    seed=2,
  )
  f = {"x_norm": 0.5, "y_norm": 0.5}
  cold, _ = det.score_and_learn(f)
  for _ in range(499):
    det.score_and_learn(f)
  raw, _ = det.score_only(f)
  assert cold > raw
  assert raw < 0.5


def test_indep_marginal_checkpoint_save_load_preserves_scores():
  rng = np.random.default_rng(42)
  events = []
  for i in range(101):
    events.append({
      "x_norm": float(np.clip(rng.normal(0.2, 0.05), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.6, 0.05), 0.0, 1.0)),
      "comm_hash": float((i % 10) / 100.0),
      "return_success": float(i % 2),
    })

  full = OnlineAnomalyDetector(
    algorithm="indep_marginal",
    freq1d_bins=32,
    freq1d_max_categories=128,
    seed=7,
  )
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(
      algorithm="indep_marginal",
      freq1d_bins=32,
      freq1d_max_categories=128,
      seed=7,
    )
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(
      algorithm="indep_marginal",
      freq1d_bins=32,
      freq1d_max_categories=128,
      seed=7,
    )
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])[0]
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)
