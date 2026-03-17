"""Focused tests for GaussCop Gaussian copula detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def test_gausscop_raw_score_finite_and_non_negative():
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_alpha=1.0,
    gausscop_decay=1.0,
    gausscop_max_categories=128,
    gausscop_reg=0.01,
    gausscop_u_clamp=1e-6,
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9, "comm_hash": 0.1234, "return_success": 1.0}
  for _ in range(10):
    raw, _ = det.score_and_learn(features)
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_gausscop_cold_start_returns_zero():
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    seed=2,
  )
  features = {"x_norm": 0.5, "y_norm": 0.5, "comm_hash": 0.1234}
  raw0, _ = det.score_and_learn(features)
  raw1, _ = det.score_and_learn(features)
  assert raw0 == 0.0
  assert raw1 == 0.0


def test_gausscop_checkpoint_save_load_preserves_scores():
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
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    seed=7,
  )
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(
      algorithm="gausscop",
      gausscop_bins=32,
      gausscop_max_categories=128,
      seed=7,
    )
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(
      algorithm="gausscop",
      gausscop_bins=32,
      gausscop_max_categories=128,
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


def test_gausscop_anomalous_scores_finite_after_warmup():
  """After warmup on normal data, both normal and anomalous samples produce finite scores."""
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=64,
    gausscop_alpha=1.0,
    gausscop_decay=1.0,
    gausscop_max_categories=512,
    seed=2,
  )
  rng = np.random.default_rng(2)

  for _ in range(400):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    det.score_and_learn(features)

  normal_scores = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    normal_scores.append(det.score_only(features)[0])

  shifted_scores = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "comm_hash": 0.9999,
      "return_success": 1.0,
    }
    shifted_scores.append(det.score_only(features)[0])

  assert all(np.isfinite(s) for s in normal_scores)
  assert all(np.isfinite(s) for s in shifted_scores)
  assert all(s >= 0.0 for s in normal_scores)
  assert all(s >= 0.0 for s in shifted_scores)


def test_gausscop_scaled_score_gradual():
  """Scaled scores use raw/(1+raw) for gradual spread (avoid 0/1 bimodal)."""
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    seed=3,
  )
  rng = np.random.default_rng(3)
  for _ in range(100):
    det.score_and_learn({
      "x_norm": float(np.clip(rng.normal(0.3, 0.1), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0)),
      "comm_hash": float(rng.integers(0, 100) / 100.0),
      "return_success": float(rng.choice([0.0, 1.0])),
    })

  raw_scores = []
  scaled_scores = []
  for _ in range(50):
    features = {
      "x_norm": float(np.clip(rng.normal(0.3, 0.1), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0)),
      "comm_hash": float(rng.integers(0, 100) / 100.0),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    raw, scaled = det.score_only(features)
    raw_scores.append(raw)
    scaled_scores.append(scaled)

  # With scale=10, raw in [0,5] maps to scaled in [0, ~0.39]; raw in [5,15] to [0.39, 0.78]
  # Scaled should have variance (not all 0 or all 1)
  assert min(scaled_scores) < max(scaled_scores), "scaled scores should vary, not be bimodal"
  assert all(0 <= s <= 1 for s in scaled_scores)


def test_gausscop_max_features_uses_subset():
  """With max_features=2, importance_window=10, after 15 events copula uses 2 features."""
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    gausscop_max_features=2,
    gausscop_importance_window=10,
    seed=4,
  )
  rng = np.random.default_rng(4)
  for i in range(15):
    det.score_and_learn({
      "a_norm": float(np.clip(rng.normal(0.2, 0.1), 0.0, 1.0)),
      "b_norm": float(np.clip(rng.normal(0.6, 0.1), 0.0, 1.0)),
      "c_norm": float(np.clip(rng.normal(0.4, 0.1), 0.0, 1.0)),
      "comm_hash": float(rng.integers(0, 10) / 10.0),
      "return_success": float(rng.choice([0.0, 1.0])),
    })
  impl = det.impl
  assert impl._z_outer_sum is not None
  assert impl._z_outer_sum.shape[0] == 2
  assert impl._selected_indices is not None
  assert len(impl._selected_indices) == 2


def test_gausscop_max_features_zero_uses_all():
  """With max_features=0 (use all), full z is used."""
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    gausscop_max_features=0,
    gausscop_importance_window=500,
    seed=5,
  )
  rng = np.random.default_rng(5)
  for i in range(100):
    det.score_and_learn({
      "a_norm": float(np.clip(rng.normal(0.2, 0.1), 0.0, 1.0)),
      "b_norm": float(np.clip(rng.normal(0.6, 0.1), 0.0, 1.0)),
      "comm_hash": float(rng.integers(0, 10) / 10.0),
      "return_success": float(rng.choice([0.0, 1.0])),
    })
  impl = det.impl
  assert impl._selected_indices is None
  assert impl._z_outer_sum is not None
  assert impl._z_outer_sum.shape[0] == 4  # a_norm, b_norm, comm_hash, return_success


def test_gausscop_max_features_ge_d_uses_all():
  """With max_features >= d (e.g. 100), full z is used."""
  det = OnlineAnomalyDetector(
    algorithm="gausscop",
    gausscop_bins=32,
    gausscop_max_categories=128,
    gausscop_max_features=100,
    gausscop_importance_window=500,
    seed=6,
  )
  rng = np.random.default_rng(6)
  for i in range(100):
    det.score_and_learn({
      "a_norm": float(np.clip(rng.normal(0.2, 0.1), 0.0, 1.0)),
      "b_norm": float(np.clip(rng.normal(0.6, 0.1), 0.0, 1.0)),
      "comm_hash": float(rng.integers(0, 10) / 10.0),
      "return_success": float(rng.choice([0.0, 1.0])),
    })
  impl = det.impl
  assert impl._selected_indices is None
  assert impl._z_outer_sum is not None
  assert impl._z_outer_sum.shape[0] == 4
