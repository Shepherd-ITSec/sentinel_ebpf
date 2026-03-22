"""Focused tests for MemStream-style online detector."""

import numpy as np
import torch

from detector.model import OnlineAnomalyDetector


def _make_features(vec):
  return {f"f{i}": float(v) for i, v in enumerate(vec)}


def test_memstream_scores_in_unit_interval():
  detector = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=32,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
    seed=7,
  )
  rng = np.random.default_rng(7)
  for _ in range(20):
    _, score = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
    assert 0.0 <= score <= 1.0


def test_memstream_gates_memory_updates_for_extreme_anomalies():
  detector = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=16,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
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
    mem_memory_size=16,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
    seed=13,
  )
  detector.score_and_learn(_make_features(np.zeros(9)))
  impl = detector.impl
  assert impl.algorithm == "memstream"
  expected = "cuda" if torch.cuda.is_available() else "cpu"
  assert impl._device.type == expected


def test_memstream_effective_dims_scale_with_input_dim():
  """Paper: latent = 2×input_dim. Encoder is Linear(in, 2*in) + Tanh."""
  small = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=16,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
    seed=23,
  )
  small.score_and_learn(_make_features(np.zeros(9)))
  small_impl = small.impl
  assert small_impl._model is not None
  small_latent = small_impl._model.encoder[0].out_features
  assert small_latent == 18  # 2 * 9

  large = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=16,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
    seed=23,
  )
  large.score_and_learn(_make_features(np.zeros(196)))
  large_impl = large.impl
  assert large_impl._model is not None
  large_latent = large_impl._model.encoder[0].out_features
  assert large_latent == 392  # 2 * 196

  assert large_latent > small_latent


def test_memstream_exposes_debug_signals_and_counters():
  detector = OnlineAnomalyDetector(
    algorithm="memstream",
    mem_memory_size=8,
    mem_lr=0.005,
    mem_beta=0.1,
    mem_k=3,
    mem_gamma=0.5,
    seed=31,
  )
  rng = np.random.default_rng(31)

  for _ in range(24):
    detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))

  debug = detector.get_last_debug()
  assert debug["mode"] == "score_and_learn"
  assert debug["score_raw"] >= 0.0
  assert debug["beta"] >= 0.0
  assert debug["mem_filled_after"] <= 8
  assert 0.0 <= debug["memory_fill_fraction"] <= 1.0
  assert debug["accepted_updates_total"] >= 1
  assert debug["rejected_updates_total"] >= 0
  assert debug["overwrite_updates_total"] >= 0

  impl = detector.impl
  assert impl.algorithm == "memstream"
  state = impl.get_state()
  assert state["accepted_updates"] == debug["accepted_updates_total"]
  assert state["rejected_updates"] == debug["rejected_updates_total"]
  assert state["overwrite_updates"] == debug["overwrite_updates_total"]


def test_memstream_transformed_input_modes_smoke():
  modes = ("freq1d_u", "freq1d_z", "freq1d_surprisal")
  rng = np.random.default_rng(41)

  for mode in modes:
    detector = OnlineAnomalyDetector(
      algorithm="memstream",
      mem_memory_size=16,
      mem_lr=0.005,
      mem_beta=0.1,
      mem_k=3,
      mem_gamma=0.5,
      mem_input_mode=mode,
      freq1d_bins=32,
      freq1d_alpha=1.0,
      freq1d_decay=0.99,
      freq1d_max_categories=128,
      seed=41,
    )
    for _ in range(12):
      raw, scaled = detector.score_and_learn(_make_features(rng.normal(0.0, 1.0, size=9)))
      assert raw >= 0.0
      assert 0.0 <= scaled <= 1.0

    debug = detector.get_last_debug()
    assert debug["input_mode"] == mode
    assert debug["mem_filled_after"] <= 16
    assert debug["score_raw"] >= 0.0
