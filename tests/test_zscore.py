"""Focused tests for online ZScore baseline detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def _make_features(vec):
  return {f"f{i}": float(v) for i, v in enumerate(vec)}


def test_zscore_scores_in_unit_interval():
  detector = OnlineAnomalyDetector(
    algorithm="zscore",
    zscore_min_count=10,
    zscore_std_floor=1e-3,
    seed=3,
  )
  rng = np.random.default_rng(3)
  for _ in range(40):
    score = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    assert 0.0 <= score <= 1.0


def test_zscore_anomaly_shift_scores_higher():
  detector = OnlineAnomalyDetector(
    algorithm="zscore",
    zscore_min_count=20,
    zscore_std_floor=1e-3,
    seed=5,
  )
  rng = np.random.default_rng(5)
  for _ in range(200):
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))

  normal_scores = [
    detector.score_and_learn_raw(_make_features(rng.normal(0.0, 1.0, size=9)))
    for _ in range(25)
  ]
  anomaly_scores = [
    detector.score_and_learn_raw(_make_features(rng.normal(8.0, 1.0, size=9)))
    for _ in range(25)
  ]
  assert np.mean(anomaly_scores) > np.mean(normal_scores)


def test_zscore_checkpoint_save_load_preserves_scores():
  """Save checkpoint after N events, load, continue; scores match full replay."""
  rng = np.random.default_rng(42)
  events = [{f"f{i}": float(v) for i, v in enumerate(rng.normal(0.0, 1.0, size=12))} for _ in range(101)]

  full = OnlineAnomalyDetector(algorithm="zscore", zscore_min_count=5, seed=7)
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(algorithm="zscore", zscore_min_count=5, seed=7)
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(algorithm="zscore", zscore_min_count=5, seed=7)
    idx = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)


def test_zscore_near_constant_features_stable():
  detector = OnlineAnomalyDetector(
    algorithm="zscore",
    zscore_min_count=5,
    zscore_std_floor=1e-3,
    seed=11,
  )
  base = np.full(9, 0.25, dtype=np.float64)
  for _ in range(20):
    score = detector.score_and_learn(_make_features(base))
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0

  score_raw = detector.score_only_raw(_make_features(base))
  assert np.isfinite(score_raw)
  assert score_raw >= 0.0
