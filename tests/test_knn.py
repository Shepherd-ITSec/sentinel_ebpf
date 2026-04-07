"""Focused tests for KNN baseline detector."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def test_knn_raw_score_finite_and_non_negative():
  det = OnlineAnomalyDetector(
    algorithm="knn",
    knn_k=3,
    knn_memory_size=64,
    knn_metric="euclidean",
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9, "comm_hash": 0.1234, "return_success": 1.0}
  for _ in range(10):
    raw, _ = det.score_and_learn(features)
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_knn_shifted_distribution_scores_higher():
  det = OnlineAnomalyDetector(
    algorithm="knn",
    knn_k=5,
    knn_memory_size=512,
    knn_metric="euclidean",
    seed=2,
  )
  rng = np.random.default_rng(2)

  for _ in range(500):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    det.score_and_learn(features)

  normal = []
  for _ in range(100):
    features = {
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    }
    normal.append(det.score_only(features)[0])

  shifted = []
  for _ in range(100):
    features = {
      "x_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "comm_hash": 0.9999,
      "return_success": 1.0,
    }
    shifted.append(det.score_only(features)[0])

  assert float(np.mean(shifted)) > float(np.mean(normal))


def test_knn_checkpoint_save_load_preserves_scores():
  rng = np.random.default_rng(42)
  events = []
  for i in range(101):
    events.append({
      "x_norm": float(np.clip(rng.normal(0.2, 0.05), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.6, 0.05), 0.0, 1.0)),
      "comm_hash": float((i % 10) / 100.0),
      "return_success": float(i % 2),
    })

  full = OnlineAnomalyDetector(algorithm="knn", knn_k=3, knn_memory_size=64, seed=7)
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(algorithm="knn", knn_k=3, knn_memory_size=64, seed=7)
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(algorithm="knn", knn_k=3, knn_memory_size=64, seed=7)
    idx, _ = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])[0]
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)
