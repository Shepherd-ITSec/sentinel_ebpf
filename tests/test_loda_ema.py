"""Focused tests for LODA-EMA online detector (custom implementation)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from detector.model import OnlineAnomalyDetector


def _make_features(vec):
  return {f"f{i}": float(v) for i, v in enumerate(vec)}


def test_loda_ema_scores_in_unit_interval():
  detector = OnlineAnomalyDetector(
    algorithm="loda_ema",
    loda_n_projections=16,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    seed=3,
  )
  rng = np.random.default_rng(3)
  for _ in range(30):
    _, score = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    assert 0.0 <= score <= 1.0


def test_loda_ema_anomaly_shift_scores_higher():
  detector = OnlineAnomalyDetector(
    algorithm="loda_ema",
    loda_n_projections=20,
    loda_bins=32,
    loda_range=3.0,
    loda_ema_alpha=0.03,
    loda_hist_decay=1.0,
    seed=5,
  )
  rng = np.random.default_rng(5)
  for _ in range(200):
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))

  normal_scores = [
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))[1]
    for _ in range(25)
  ]
  anomaly_scores = [
    detector.score_and_learn(_make_features(rng.normal(8.0, 1.0, size=9)))[1]
    for _ in range(25)
  ]
  assert np.mean(anomaly_scores) > np.mean(normal_scores)


def test_loda_ema_effective_projections_scale_with_input_dim():
  small = OnlineAnomalyDetector(
    algorithm="loda_ema",
    loda_n_projections=4,
    loda_bins=16,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    seed=19,
  )
  small.score_and_learn(_make_features(np.zeros(9)))
  small_proj = small.impl._effective_n_projections

  large = OnlineAnomalyDetector(
    algorithm="loda_ema",
    loda_n_projections=4,
    loda_bins=16,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    seed=19,
  )
  large.score_and_learn(_make_features(np.zeros(144)))
  large_proj = large.impl._effective_n_projections

  assert small_proj is not None
  assert large_proj is not None
  assert large_proj > small_proj


def test_loda_ema_checkpoint_save_load_preserves_scores():
  """Save checkpoint after N events, load, continue; scores match full replay."""
  rng = np.random.default_rng(42)
  events = [{f"f{i}": float(v) for i, v in enumerate(rng.normal(0.0, 1.0, size=12))} for _ in range(101)]

  # Full replay: learn 0..99, score event 100
  full = OnlineAnomalyDetector(algorithm="loda_ema", seed=7)
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[1]

  # Checkpoint at 50, load, replay 50..99, score event 100
  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(algorithm="loda_ema", seed=7)
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(algorithm="loda_ema", seed=7)
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])[1]
    assert abs(full_score - loaded_score) < 1e-6
  finally:
    ckpt.unlink(missing_ok=True)


def test_loda_ema_get_last_debug():
  """get_last_debug returns per-projection excess signals after score_and_learn."""
  det = OnlineAnomalyDetector(algorithm="loda_ema", seed=11)
  det.score_and_learn(_make_features(np.zeros(9)))
  debug_before = det.get_last_debug()
  assert isinstance(debug_before, dict)
  assert "score_raw" in debug_before
  assert "mean_projection_excess" in debug_before
  assert "max_projection_excess" in debug_before
  assert "per_projection_excess" in debug_before
  assert abs(debug_before["score_raw"] - debug_before["mean_projection_excess"]) < 1e-9
  assert debug_before["max_projection_excess"] >= debug_before["mean_projection_excess"]
  assert len(debug_before["per_projection_excess"]) == det.impl._effective_n_projections


def test_loda_ema_attribution_space_lograw_matches_log1p_raw_score():
  det = OnlineAnomalyDetector(algorithm="loda_ema", seed=11)
  det.score_and_learn(_make_features(np.zeros(9)))
  features = _make_features(np.linspace(0.0, 1.0, num=9))
  score_raw = det.score_only(features)[0]
  score_lograw, _ = det.compute_feature_attribution(features, epsilon=0.01, score_mode="lograw")
  assert abs(score_lograw - float(np.log1p(max(0.0, score_raw)))) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in test runtime")
def test_loda_ema_can_run_on_cuda_device():
  detector = OnlineAnomalyDetector(
    algorithm="loda_ema",
    loda_n_projections=8,
    loda_bins=16,
    loda_range=3.0,
    loda_ema_alpha=0.05,
    loda_hist_decay=1.0,
    model_device="cuda",
    seed=17,
  )
  detector.score_and_learn(_make_features(np.zeros(9)))
  assert detector.impl.algorithm == "loda_ema"
  assert detector.impl._device.type == "cuda"
