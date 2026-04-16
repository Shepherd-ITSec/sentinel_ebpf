from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from detector.building_blocks.primitives.models.common import _BothScoresMixin, _auto_loda_projections, _resolve_torch_device


class OnlineLODAEMA(_BothScoresMixin):
  """
  LODA-style anomaly detector with EMA-based adaptive normalization. (Self-implementation)
  Uses sparse random projections + online histograms; streaming from first event.
  EMA for mean/var normalization; optional hist_decay for forgetting.
  """

  def __init__(
    self,
    n_projections: int,
    bins: int,
    proj_range: float,
    ema_alpha: float,
    hist_decay: float,
    model_device: str,
    seed: int,
  ):
    self.algorithm = "loda_ema"
    self.n_projections = n_projections
    self.bins = bins
    self.proj_range = proj_range
    self.ema_alpha = ema_alpha
    self.hist_decay = hist_decay
    self.seed = seed
    self._device = _resolve_torch_device(model_device)
    self._feature_names: Optional[List[str]] = None
    self._effective_n_projections: Optional[int] = None
    self._weights: Optional[torch.Tensor] = None
    self._mean: Optional[torch.Tensor] = None
    self._var: Optional[torch.Tensor] = None
    self._counts: Optional[torch.Tensor] = None
    self._last_debug: Dict[str, Any] = {}
    logging.info(
      "Initialized %s (projections=%d, bins=%d, range=%.2f, ema_alpha=%.3f, hist_decay=%.3f, device=%s)",
      self.algorithm,
      n_projections,
      bins,
      proj_range,
      ema_alpha,
      hist_decay,
      self._device.type,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_names = sorted(features.keys())
    dim = len(self._feature_names)
    effective_n_projections = max(self.n_projections, _auto_loda_projections(dim))
    self._effective_n_projections = effective_n_projections
    rng = np.random.default_rng(self.seed)
    nnz = max(1, int(np.sqrt(dim)))
    weights = np.zeros((effective_n_projections, dim), dtype=np.float32)
    for i in range(effective_n_projections):
      idx = rng.choice(dim, size=nnz, replace=False)
      w = rng.normal(0.0, 1.0, size=nnz).astype(np.float32)
      norm = float(np.linalg.norm(w))
      if norm > 1e-12:
        w = w / norm
      weights[i, idx] = w
    self._weights = torch.from_numpy(weights).to(self._device)
    self._mean = torch.zeros(effective_n_projections, dtype=torch.float32, device=self._device)
    self._var = torch.ones(effective_n_projections, dtype=torch.float32, device=self._device)
    self._counts = torch.zeros((effective_n_projections, self.bins), dtype=torch.float32, device=self._device)
    logging.info("LODA input_dim=%d effective_projections=%d", dim, effective_n_projections)

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("LODA feature names not initialized")
    vec = np.array([float(features[k]) for k in self._feature_names], dtype=np.float32)
    return torch.from_numpy(vec).to(self._device)

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    if self._weights is None:
      self._init_from_features(features)
    x = self._vectorize(features)
    weights = self._weights
    mean = self._mean
    var = self._var
    counts = self._counts
    if weights is None or mean is None or var is None or counts is None:
      raise RuntimeError("LODA not initialized")

    effective_counts = counts * self.hist_decay if self.hist_decay < 1.0 else counts
    projections = weights @ x
    std = torch.sqrt(var + 1e-6)
    normalized = (projections - mean) / std
    normalized = torch.clamp(normalized, -self.proj_range, self.proj_range)
    bin_idx = ((normalized + self.proj_range) / (2.0 * self.proj_range) * self.bins).to(torch.long)
    bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)

    row_idx = torch.arange(weights.shape[0], device=self._device)
    smoothing = 1.0
    totals = effective_counts.sum(dim=1) + smoothing * self.bins
    probs = (effective_counts[row_idx, bin_idx] + smoothing) / totals
    surprisal = -torch.log(probs)
    baseline = math.log(float(self.bins))
    excess = torch.clamp(surprisal - baseline, min=0.0)
    score = float(torch.mean(excess).item())
    excess_np = excess.cpu().numpy()
    self._last_debug = {
      "score_raw": score,
      "mean_projection_excess": score,
      "max_projection_excess": float(np.max(excess_np)),
      "per_projection_excess": [float(x) for x in excess_np],
    }
    return score

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    if self._weights is None:
      self._init_from_features(features)
    x = self._vectorize(features)
    weights = self._weights
    mean = self._mean
    var = self._var
    counts = self._counts
    if weights is None or mean is None or var is None or counts is None:
      raise RuntimeError("LODA not initialized")

    if self.hist_decay < 1.0:
      counts *= self.hist_decay

    projections = weights @ x
    std = torch.sqrt(var + 1e-6)
    normalized = (projections - mean) / std
    normalized = torch.clamp(normalized, -self.proj_range, self.proj_range)
    bin_idx = ((normalized + self.proj_range) / (2.0 * self.proj_range) * self.bins).to(torch.long)
    bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)

    row_idx = torch.arange(weights.shape[0], device=self._device)
    smoothing = 1.0
    totals = counts.sum(dim=1) + smoothing * self.bins
    probs = (counts[row_idx, bin_idx] + smoothing) / totals
    surprisal = -torch.log(probs)
    baseline = math.log(float(self.bins))
    excess = torch.clamp(surprisal - baseline, min=0.0)
    score = float(torch.mean(excess).item())

    counts[row_idx, bin_idx] += 1.0
    alpha = self.ema_alpha
    delta = projections - mean
    mean[:] = mean + alpha * delta
    var[:] = (1.0 - alpha) * var + alpha * (delta * delta)
    excess_np = excess.cpu().numpy()
    self._last_debug = {
      "score_raw": score,
      "mean_projection_excess": score,
      "max_projection_excess": float(np.max(excess_np)),
      "per_projection_excess": [float(x) for x in excess_np],
    }
    return score

  def get_last_debug(self) -> Dict[str, Any]:
    return dict(self._last_debug)

  def get_state(self) -> Dict[str, Any]:
    if self._weights is None:
      raise RuntimeError("LODA not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "effective_n_projections": self._effective_n_projections,
      "weights": self._weights.cpu().numpy(),
      "mean": self._mean.cpu().numpy(),
      "var": self._var.cpu().numpy(),
      "counts": self._counts.cpu().numpy(),
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self._feature_names = list(state["feature_names"])
    self._effective_n_projections = int(state["effective_n_projections"])
    self._weights = torch.from_numpy(np.asarray(state["weights"], dtype=np.float32)).to(self._device)
    self._mean = torch.from_numpy(np.asarray(state["mean"], dtype=np.float32)).to(self._device)
    self._var = torch.from_numpy(np.asarray(state["var"], dtype=np.float32)).to(self._device)
    self._counts = torch.from_numpy(np.asarray(state["counts"], dtype=np.float32)).to(self._device)
