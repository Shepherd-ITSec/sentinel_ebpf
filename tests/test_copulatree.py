"""Focused tests for the streaming CopulaTree detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def test_copulatree_raw_score_finite_and_non_negative():
  det = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=32,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=128,
    copulatree_tree_update_interval=5,
    seed=1,
  )
  rng = np.random.default_rng(1)
  for _ in range(20):
    raw, _ = det.score_and_learn({
      "x_norm": float(np.clip(rng.normal(0.25, 0.05), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.75, 0.05), 0.0, 1.0)),
    })
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_copulatree_cold_start_returns_zero():
  det = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=32,
    freq1d_max_categories=128,
    copulatree_tree_update_interval=3,
    seed=2,
  )
  features = {"x_norm": 0.5, "y_norm": 0.5}
  raw0, _ = det.score_and_learn(features)
  raw1, _ = det.score_and_learn(features)
  assert raw0 == 0.0
  assert raw1 == 0.0


def test_copulatree_checkpoint_save_load_preserves_scores():
  rng = np.random.default_rng(42)
  events = []
  for _ in range(81):
    x = float(np.clip(rng.normal(0.35, 0.04), 0.0, 1.0))
    y = float(np.clip(x + rng.normal(0.0, 0.015), 0.0, 1.0))
    z = float(np.clip(1.0 - x + rng.normal(0.0, 0.02), 0.0, 1.0))
    events.append({"x_norm": x, "y_norm": y, "z_norm": z})

  full = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=32,
    freq1d_max_categories=128,
    copulatree_tree_update_interval=5,
    seed=7,
  )
  for i in range(80):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[80])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(
      algorithm="copulatree",
      freq1d_bins=32,
      freq1d_max_categories=128,
      copulatree_tree_update_interval=5,
      seed=7,
    )
    for i in range(80):
      ckpt_det.score_and_learn(events[i])
      if i == 39:
        ckpt_det.save_checkpoint(ckpt, 40)

    loaded = OnlineAnomalyDetector(
      algorithm="copulatree",
      freq1d_bins=32,
      freq1d_max_categories=128,
      copulatree_tree_update_interval=5,
      seed=7,
    )
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 40
    for i in range(40, 80):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[80])[0]
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)


def test_copulatree_builds_tree_after_warmup():
  det = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=32,
    freq1d_max_categories=128,
    copulatree_tree_update_interval=4,
    seed=3,
  )
  rng = np.random.default_rng(3)
  for _ in range(25):
    x = float(np.clip(rng.normal(0.2, 0.04), 0.0, 1.0))
    det.score_and_learn({
      "x_norm": x,
      "y_norm": float(np.clip(x + rng.normal(0.0, 0.02), 0.0, 1.0)),
      "z_norm": float(np.clip(1.0 - x + rng.normal(0.0, 0.02), 0.0, 1.0)),
    })
  impl = det.impl
  assert impl._pair_outer_ema is not None
  assert impl._pair_outer_ema.shape == (3, 3)
  assert len(impl._tree_edges) == 2


def test_copulatree_max_features_uses_subset():
  det = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=32,
    freq1d_max_categories=128,
    copulatree_max_features=2,
    copulatree_importance_window=10,
    copulatree_tree_update_interval=3,
    seed=4,
  )
  rng = np.random.default_rng(4)
  for _ in range(20):
    base = float(np.clip(rng.normal(0.4, 0.06), 0.0, 1.0))
    det.score_and_learn({
      "a_norm": base,
      "b_norm": float(np.clip(base + rng.normal(0.0, 0.015), 0.0, 1.0)),
      "c_norm": float(np.clip(rng.normal(0.8, 0.05), 0.0, 1.0)),
      "d_norm": float(np.clip(rng.normal(0.1, 0.05), 0.0, 1.0)),
    })
  impl = det.impl
  assert impl._selected_indices is not None
  assert len(impl._selected_indices) == 2
  assert impl._pair_outer_ema is not None
  assert impl._pair_outer_ema.shape == (2, 2)
  assert len(impl._tree_edges) == 1


def test_copulatree_dependency_break_scores_higher_than_baseline():
  det = OnlineAnomalyDetector(
    algorithm="copulatree",
    freq1d_bins=64,
    freq1d_max_categories=128,
    copulatree_tree_update_interval=5,
    seed=5,
  )
  rng = np.random.default_rng(5)
  for _ in range(300):
    x = float(np.clip(rng.normal(0.3, 0.05), 0.0, 1.0))
    det.score_and_learn({
      "x_norm": x,
      "y_norm": float(np.clip(x + rng.normal(0.0, 0.01), 0.0, 1.0)),
    })

  normal_scores = []
  broken_scores = []
  for _ in range(50):
    x = float(np.clip(rng.normal(0.3, 0.05), 0.0, 1.0))
    normal_scores.append(det.score_only({
      "x_norm": x,
      "y_norm": float(np.clip(x + rng.normal(0.0, 0.01), 0.0, 1.0)),
    })[0])
    broken_scores.append(det.score_only({
      "x_norm": x,
      "y_norm": float(np.clip(1.0 - x + rng.normal(0.0, 0.01), 0.0, 1.0)),
    })[0])

  assert np.mean(broken_scores) > np.mean(normal_scores)
