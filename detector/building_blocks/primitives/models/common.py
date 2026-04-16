from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _auto_loda_projections(input_dim: int) -> int:
  # Keep enough random views as dimensionality grows, but cap runtime.
  return max(8, min(256, int(np.ceil(2.0 * np.sqrt(max(1, input_dim))))))


def _auto_kitnet_max_size_ae(input_dim: int) -> int:
  # Kitsune sub-autoencoders should not exceed input size.
  return max(2, min(32, int(np.ceil(np.sqrt(max(1, input_dim))))))


def _fenwick_prefix_sum(tree: Any, i: int) -> float:
  """Prefix sum [0..i] inclusive. fenwick uses exclusive stop, so prefix_sum(i+1)."""
  return float(tree.prefix_sum(i + 1))


class _BothScoresMixin:
  """Mixin for impls that use 1-exp(-max(0,raw)) squash. Used when impl is instantiated directly (e.g. tests)."""

  def score_only(self, features: Dict[str, float], *, meta: Any | None = None) -> tuple[float, float]:
    raw = self.score_only_raw(features, meta=meta)
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return (float(raw), scaled)

  def score_and_learn(self, features: Dict[str, float], *, meta: Any | None = None) -> tuple[float, float]:
    raw = self.score_and_learn_raw(features, meta=meta)
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return (float(raw), scaled)


def _resolve_torch_device(preference: str) -> torch.device:
  pref = preference.strip().lower()
  if pref not in ("auto", "cpu", "cuda"):
    raise ValueError("Invalid model_device value: %s. Choose from: auto, cpu, cuda" % preference)

  if pref == "cpu":
    return torch.device("cpu")

  cuda_available = torch.cuda.is_available()
  if pref == "cuda":
    if not cuda_available:
      raise RuntimeError("DETECTOR_MODEL_DEVICE=cuda requested but CUDA is not available in this torch runtime")
    return torch.device("cuda")

  if cuda_available:
    return torch.device("cuda")
  return torch.device("cpu")
