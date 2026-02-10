"""Online anomaly detection models."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from river import anomaly


class OnlineHalfSpaceTrees:
  """Online anomaly detector using River Half-Space Trees."""

  def __init__(self, n_trees: int, height: int, window_size: int, seed: int):
    self.algorithm = "halfspacetrees"
    self.model: Any = anomaly.HalfSpaceTrees(
      n_trees=n_trees,
      height=height,
      window_size=window_size,
      seed=seed,
    )
    logging.info(
      "Initialized %s (n_trees=%d, height=%d, window_size=%d)",
      self.algorithm,
      n_trees,
      height,
      window_size,
    )

  def score_and_learn(self, features: Dict[str, float]) -> float:
    score = self.model.score_one(features)
    self.model.learn_one(features)  # in-place; learn_one returns None
    return float(score)


class OnlineLODA:
  """
  Online LODA-style anomaly detector for streaming vectors.
  Uses random projections + online histograms to estimate 1D densities.
  """

  def __init__(
    self,
    n_projections: int,
    bins: int,
    proj_range: float,
    ema_alpha: float,
    hist_decay: float,
    seed: int,
  ):
    self.algorithm = "loda"
    self.n_projections = n_projections
    self.bins = bins
    self.proj_range = proj_range
    self.ema_alpha = ema_alpha
    self.hist_decay = hist_decay
    self.seed = seed
    self._feature_keys: Optional[List[str]] = None
    self._weights: Optional[np.ndarray] = None
    self._mean: Optional[np.ndarray] = None
    self._var: Optional[np.ndarray] = None
    self._counts: Optional[np.ndarray] = None
    logging.info(
      "Initialized %s (projections=%d, bins=%d, range=%.2f, ema_alpha=%.3f, hist_decay=%.3f)",
      self.algorithm,
      n_projections,
      bins,
      proj_range,
      ema_alpha,
      hist_decay,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_keys = sorted(features.keys())
    dim = len(self._feature_keys)
    rng = np.random.default_rng(self.seed)
    self._weights = rng.normal(0.0, 1.0, size=(self.n_projections, dim)).astype(np.float32)
    self._mean = np.zeros(self.n_projections, dtype=np.float32)
    self._var = np.ones(self.n_projections, dtype=np.float32)
    self._counts = np.zeros((self.n_projections, self.bins), dtype=np.float32)

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_keys is None:
      self._init_from_features(features)
    if self._feature_keys is None:
      raise RuntimeError("LODA feature keys not initialized")
    return np.array([float(features[k]) for k in self._feature_keys], dtype=np.float32)

  def score_and_learn(self, features: Dict[str, float]) -> float:
    if self._weights is None:
      self._init_from_features(features)
    x = self._vectorize(features)
    weights = self._weights
    mean = self._mean
    var = self._var
    counts = self._counts
    if weights is None or mean is None or var is None or counts is None:
      raise RuntimeError("LODA not initialized")

    projections = weights @ x
    alpha = self.ema_alpha
    mean[:] = (1.0 - alpha) * mean + alpha * projections
    var[:] = (1.0 - alpha) * var + alpha * (projections - mean) ** 2

    std = np.sqrt(var + 1e-6)
    normalized = (projections - mean) / std
    normalized = np.clip(normalized, -self.proj_range, self.proj_range)
    bin_idx = ((normalized + self.proj_range) / (2.0 * self.proj_range) * self.bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, self.bins - 1)

    if self.hist_decay < 1.0:
      counts *= self.hist_decay
    counts[np.arange(self.n_projections), bin_idx] += 1.0

    smoothing = 1.0
    totals = counts.sum(axis=1) + smoothing * self.bins
    probs = (counts[np.arange(self.n_projections), bin_idx] + smoothing) / totals
    score = float(np.mean(-np.log(probs)))
    return 1.0 - float(np.exp(-score))


class _AutoEncoder(torch.nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
    super().__init__()
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, latent_dim),
    )
    self.decoder = torch.nn.Sequential(
      torch.nn.Linear(latent_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, input_dim),
    )

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = self.encoder(x)
    recon = self.decoder(z)
    return z, recon


class OnlineMemStream:
  """
  Online MemStream-style detector using an autoencoder + memory buffer.
  """

  def __init__(
    self,
    hidden_dim: int,
    latent_dim: int,
    memory_size: int,
    lr: float,
    seed: int,
  ):
    self.algorithm = "memstream"
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.memory_size = memory_size
    self.lr = lr
    self.seed = seed
    self._feature_keys: Optional[List[str]] = None
    self._model: Optional[_AutoEncoder] = None
    self._optimizer: Optional[torch.optim.Optimizer] = None
    self._memory: Optional[torch.Tensor] = None
    self._mem_index = 0
    self._mem_filled = 0
    logging.info(
      "Initialized %s (hidden=%d, latent=%d, memory=%d, lr=%.5f)",
      self.algorithm,
      hidden_dim,
      latent_dim,
      memory_size,
      lr,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_keys = sorted(features.keys())
    input_dim = len(self._feature_keys)
    torch.manual_seed(self.seed)
    self._model = _AutoEncoder(input_dim=input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    self._memory = torch.zeros(self.memory_size, self.latent_dim, dtype=torch.float32)

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_keys is None:
      self._init_from_features(features)
    if self._feature_keys is None:
      raise RuntimeError("MemStream feature keys not initialized")
    vec = np.array([float(features[k]) for k in self._feature_keys], dtype=np.float32)
    return torch.from_numpy(vec).unsqueeze(0)

  def _nearest_memory(self, z: torch.Tensor) -> torch.Tensor:
    if self._memory is None or self._mem_filled == 0:
      return z
    memory = self._memory[: self._mem_filled]
    dists = torch.sum((memory - z) ** 2, dim=1)
    idx = torch.argmin(dists)
    return memory[idx].unsqueeze(0)

  def _update_memory(self, z: torch.Tensor) -> None:
    if self._memory is None:
      return
    if self._mem_filled < self.memory_size:
      self._memory[self._mem_filled] = z
      self._mem_filled += 1
      return
    self._memory[self._mem_index] = z
    self._mem_index = (self._mem_index + 1) % self.memory_size

  def score_and_learn(self, features: Dict[str, float]) -> float:
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")

    x = self._vectorize(features)
    self._model.train()
    with torch.no_grad():
      z, _ = self._model(x)
      mem_z = self._nearest_memory(z)
      recon = self._model.decoder(mem_z)
      recon_error = torch.mean((x - recon) ** 2).item()
    score = 1.0 - float(np.exp(-recon_error))

    self._optimizer.zero_grad()
    z_train, recon_train = self._model(x)
    loss = torch.mean((x - recon_train) ** 2)
    loss.backward()
    self._optimizer.step()

    self._update_memory(z_train.detach().squeeze(0))
    return score


class OnlineAnomalyDetector:
  """Factory wrapper for online anomaly detectors."""

  def __init__(
    self,
    algorithm: str,
    hst_n_trees: int,
    hst_height: int,
    hst_window_size: int,
    loda_n_projections: int,
    loda_bins: int,
    loda_range: float,
    loda_ema_alpha: float,
    loda_hist_decay: float,
    mem_hidden_dim: int,
    mem_latent_dim: int,
    mem_memory_size: int,
    mem_lr: float,
    seed: int,
  ):
    algo = algorithm.lower()
    if algo == "halfspacetrees":
      self.impl = OnlineHalfSpaceTrees(
        n_trees=hst_n_trees,
        height=hst_height,
        window_size=hst_window_size,
        seed=seed,
      )
    elif algo == "loda":
      self.impl = OnlineLODA(
        n_projections=loda_n_projections,
        bins=loda_bins,
        proj_range=loda_range,
        ema_alpha=loda_ema_alpha,
        hist_decay=loda_hist_decay,
        seed=seed,
      )
    elif algo == "memstream":
      self.impl = OnlineMemStream(
        hidden_dim=mem_hidden_dim,
        latent_dim=mem_latent_dim,
        memory_size=mem_memory_size,
        lr=mem_lr,
        seed=seed,
      )
    else:
      raise ValueError("Unknown algorithm: %s. Choose from: halfspacetrees, loda, memstream" % algorithm)
    self.algorithm = self.impl.algorithm

  def score_and_learn(self, features: Dict[str, float]) -> float:
    return self.impl.score_and_learn(features)
