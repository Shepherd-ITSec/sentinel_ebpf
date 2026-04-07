"""Focused tests for latent-clustering detector on top of Freq1D marginals."""

import tempfile
from pathlib import Path

import numpy as np

from detector.model import OnlineAnomalyDetector


def test_latentcluster_raw_score_finite_and_non_negative():
  det = OnlineAnomalyDetector(
    algorithm="latentcluster",
    freq1d_bins=32,
    freq1d_max_categories=128,
    latentcluster_max_clusters=4,
    latentcluster_reg=0.25,
    latentcluster_spawn_threshold=4.0,
    seed=1,
  )
  features = {"a_norm": 0.1, "b_norm": 0.9, "comm_hash": 0.1234, "return_success": 1.0}
  for _ in range(10):
    raw, _ = det.score_and_learn(features)
    assert np.isfinite(raw)
    assert raw >= 0.0


def test_latentcluster_shifted_distribution_scores_higher():
  det = OnlineAnomalyDetector(
    algorithm="latentcluster",
    freq1d_bins=64,
    freq1d_alpha=1.0,
    freq1d_decay=1.0,
    freq1d_max_categories=256,
    latentcluster_max_clusters=8,
    latentcluster_reg=0.25,
    latentcluster_spawn_threshold=4.0,
    seed=2,
  )
  rng = np.random.default_rng(2)

  for _ in range(300):
    det.score_and_learn({
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    })

  normal = []
  for _ in range(40):
    normal.append(det.score_only({
      "x_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.02), 0.0, 1.0)),
      "comm_hash": float(rng.choice([0.1000, 0.2000, 0.3000])),
      "return_success": float(rng.choice([0.0, 1.0])),
    })[0])

  shifted = []
  for _ in range(40):
    shifted.append(det.score_only({
      "x_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.02), 0.0, 1.0)),
      "comm_hash": 0.9999,
      "return_success": 1.0,
    })[0])

  assert float(np.mean(shifted)) > float(np.mean(normal))


def test_latentcluster_spawns_multiple_clusters_for_separated_modes():
  det = OnlineAnomalyDetector(
    algorithm="latentcluster",
    freq1d_bins=64,
    freq1d_max_categories=128,
    latentcluster_max_clusters=4,
    latentcluster_reg=0.25,
    latentcluster_spawn_threshold=3.0,
    seed=3,
  )
  rng = np.random.default_rng(3)

  for _ in range(120):
    det.score_and_learn({
      "x_norm": float(np.clip(rng.normal(0.1, 0.01), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.1, 0.01), 0.0, 1.0)),
      "comm_hash": 0.1000,
      "return_success": 1.0,
    })
  for _ in range(120):
    det.score_and_learn({
      "x_norm": float(np.clip(rng.normal(0.9, 0.01), 0.0, 1.0)),
      "y_norm": float(np.clip(rng.normal(0.9, 0.01), 0.0, 1.0)),
      "comm_hash": 0.9000,
      "return_success": 1.0,
    })

  impl = det.impl
  assert impl.algorithm == "latentcluster"
  assert impl._active_clusters >= 2


def test_latentcluster_checkpoint_save_load_preserves_scores():
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
    algorithm="latentcluster",
    freq1d_bins=32,
    freq1d_max_categories=128,
    latentcluster_max_clusters=6,
    seed=7,
  )
  for i in range(100):
    full.score_and_learn(events[i])
  full_score = full.score_only(events[100])[0]

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt = Path(f.name)
  try:
    ckpt_det = OnlineAnomalyDetector(
      algorithm="latentcluster",
      freq1d_bins=32,
      freq1d_max_categories=128,
      latentcluster_max_clusters=6,
      seed=7,
    )
    for i in range(100):
      ckpt_det.score_and_learn(events[i])
      if i == 49:
        ckpt_det.save_checkpoint(ckpt, 50)

    loaded = OnlineAnomalyDetector(
      algorithm="latentcluster",
      freq1d_bins=32,
      freq1d_max_categories=128,
      latentcluster_max_clusters=6,
      seed=7,
    )
    idx, _ = loaded.load_checkpoint(ckpt)
    assert idx == 50
    for i in range(50, 100):
      loaded.score_and_learn(events[i])
    loaded_score = loaded.score_only(events[100])[0]
    assert abs(full_score - loaded_score) < 1e-12
  finally:
    ckpt.unlink(missing_ok=True)
