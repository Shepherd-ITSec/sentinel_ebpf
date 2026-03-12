"""Online anomaly detection models."""
import contextlib
import io
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
try:
  from river import anomaly as River_Anomaly
except ImportError:
  River_Anomaly = None
try:
  from pysad.models import KitNet as Pysad_KitNet
except ImportError:
  Pysad_KitNet = None
try:
  from sklearn.neighbors import NearestNeighbors
except ImportError:
  NearestNeighbors = None


def _auto_loda_projections(input_dim: int) -> int:
  # Keep enough random views as dimensionality grows, but cap runtime.
  return max(8, min(256, int(np.ceil(2.0 * np.sqrt(max(1, input_dim))))))


def _auto_mem_hidden_dim(input_dim: int) -> int:
  # Scale hidden size sub-linearly with input dimension.
  d = float(max(1, input_dim))
  return max(16, min(256, int(np.ceil(2.0 * np.sqrt(d) * np.log2(d + 1.0)))))


def _auto_mem_latent_dim(hidden_dim: int) -> int:
  return max(4, min(64, hidden_dim // 4))


def _auto_kitnet_max_size_ae(input_dim: int) -> int:
  # Kitsune sub-autoencoders should not exceed input size.
  return max(2, min(32, int(np.ceil(np.sqrt(max(1, input_dim))))))


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

  # auto
  if cuda_available:
    return torch.device("cuda")
  return torch.device("cpu")


class OnlineHalfSpaceTrees:
  """Online anomaly detector using River Half-Space Trees. (Library-based wrapper)"""

  def __init__(self, n_trees: int, height: int, window_size: int, model_device: str, seed: int):
    self.algorithm = "halfspacetrees"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but River HalfSpaceTrees is CPU-only; using CPU",
        self.algorithm,
      )
    self.model: Any = River_Anomaly.HalfSpaceTrees(
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

  def score_only(self, features: Dict[str, float]) -> float:
    """Return anomaly score without updating model state (read-only)."""
    return self.score_only_raw(features)

  def score_only_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score."""
    return float(self.model.score_one(features))

  def score_and_learn(self, features: Dict[str, float]) -> float:
    return self.score_and_learn_raw(features)

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    score = self.model.score_one(features)
    self.model.learn_one(features)  # in-place; learn_one returns None
    return float(score)

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    return {"model": self.model}

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self.model = state["model"]


class OnlineLODA:
  """
  Online LODA-style anomaly detector for streaming vectors. (Self-implementation)
  Uses sparse random projections + online histograms to estimate 1D densities.

  Reference design:
  - Paper: LODA (MLJ 2016): https://doi.org/10.1007/s10994-015-5521-0
  - Practical references:
    - PyOD LODA: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/loda.html
    - PySAD LODA: https://pysad.readthedocs.io/en/latest/_modules/pysad/models/loda.html
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
    self.algorithm = "loda"
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
    # LODA-style sparse random projections: about sqrt(d) non-zero entries.
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

  def score_only(self, features: Dict[str, float]) -> float:
    """Return anomaly score without updating model state (read-only)."""
    score_raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, score_raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score."""
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
    score = float(torch.mean(-torch.log(probs)).item())
    return score

  def score_and_learn(self, features: Dict[str, float]) -> float:
    score_raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, score_raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
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
    score = float(torch.mean(-torch.log(probs)).item())

    # Learn after scoring so the current event doesn't lower its own anomaly score.
    counts[row_idx, bin_idx] += 1.0
    alpha = self.ema_alpha
    delta = projections - mean
    mean[:] = mean + alpha * delta
    var[:] = (1.0 - alpha) * var + alpha * (delta * delta)
    return score

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
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
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self._effective_n_projections = int(state["effective_n_projections"])
    self._weights = torch.from_numpy(np.asarray(state["weights"], dtype=np.float32)).to(self._device)
    self._mean = torch.from_numpy(np.asarray(state["mean"], dtype=np.float32)).to(self._device)
    self._var = torch.from_numpy(np.asarray(state["var"], dtype=np.float32)).to(self._device)
    self._counts = torch.from_numpy(np.asarray(state["counts"], dtype=np.float32)).to(self._device)


class OnlineKitNet:
  """Online anomaly detector using PySAD KitNet. (Library-based wrapper)"""

  def __init__(
    self,
    max_size_ae: int,
    grace_feature_mapping: int,
    grace_anomaly_detector: int,
    learning_rate: float,
    hidden_ratio: float,
    model_device: str,
    seed: int,
  ):
    self.algorithm = "kitnet"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but PySAD KitNet is CPU-only; using CPU",
        self.algorithm,
      )
    self.seed = seed
    self.max_size_ae = max_size_ae
    self.grace_feature_mapping = grace_feature_mapping
    self.grace_anomaly_detector = grace_anomaly_detector
    self.learning_rate = learning_rate
    self.hidden_ratio = hidden_ratio
    self._feature_names: Optional[List[str]] = None
    self._effective_max_size_ae: Optional[int] = None
    self._model = None
    logging.info(
      "Initialized %s (max_size_ae=%d, grace_fm=%d, grace_ad=%d, lr=%.4f, hidden_ratio=%.3f)",
      self.algorithm,
      max_size_ae,
      grace_feature_mapping,
      grace_anomaly_detector,
      learning_rate,
      hidden_ratio,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    if Pysad_KitNet is None:
      raise RuntimeError("PySAD KitNet not available. Install detector deps: `uv sync --extra detector`.")
    self._feature_names = sorted(features.keys())
    dim = len(self._feature_names)
    # Effective max_size_ae:
    # - If config (self.max_size_ae) is non-zero, use it directly (clamped to dim).
    # - If config is 0, fall back to auto mode based on input dim.
    if self.max_size_ae:
      effective_max_size_ae = min(dim, int(self.max_size_ae))
    else:
      effective_max_size_ae = min(dim, _auto_kitnet_max_size_ae(dim))
    self._effective_max_size_ae = effective_max_size_ae
    self._model = Pysad_KitNet(
      max_size_ae=effective_max_size_ae,
      grace_feature_mapping=self.grace_feature_mapping,
      grace_anomaly_detector=self.grace_anomaly_detector,
      learning_rate=self.learning_rate,
      hidden_ratio=self.hidden_ratio,
    )
    logging.info("KitNet input_dim=%d effective_max_size_ae=%d", dim, effective_max_size_ae)

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("KitNet feature names not initialized")
    return np.array([float(features[k]) for k in self._feature_names], dtype=np.float64)

  def score_only(self, features: Dict[str, float]) -> float:
    """Return anomaly score without updating model state (read-only)."""
    raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score."""
    x = self._vectorize(features)
    if self._model is None:
      raise RuntimeError("KitNet not initialized")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      try:
        raw = float(self._model.score_partial(x))
      except AttributeError as exc:
        if "model" not in str(exc):
          raise
        logging.debug("KitNet cold-start before internal model init; using raw score 0.0")
        raw = 0.0
    if not np.isfinite(raw):
      logging.warning("KitNet produced non-finite score (%s); falling back to 0.0", raw)
      raw = 0.0
    return max(0.0, float(raw))

  def score_and_learn(self, features: Dict[str, float]) -> float:
    raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    x = self._vectorize(features)
    if self._model is None:
      raise RuntimeError("KitNet not initialized")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      # Prequential order: score first, then learn from this instance.
      # Some PySAD KitNet versions can raise on cold-start before fit_partial
      # creates the internal `model` object. Treat that first score as neutral.
      try:
        raw = float(self._model.score_partial(x))
      except AttributeError as exc:
        if "model" not in str(exc):
          raise
        logging.debug("KitNet cold-start before internal model init; using raw score 0.0")
        raw = 0.0
      self._model.fit_partial(x)
    if not np.isfinite(raw):
      logging.warning("KitNet produced non-finite score (%s); falling back to 0.0", raw)
      raw = 0.0
    return max(0.0, float(raw))

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    if self._model is None:
      raise RuntimeError("KitNet not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "effective_max_size_ae": self._effective_max_size_ae,
      "model": self._model,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self._effective_max_size_ae = int(state["effective_max_size_ae"])
    self._model = state["model"]



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
  Online MemStream-style detector using an autoencoder + latent memory. (Self-implementation)

  Reference design:
  - Paper: MemStream (WWW'22): https://arxiv.org/abs/2106.03837
  - Official code: https://github.com/Stream-AD/MemStream

  Adaptation for this project:
  - No supervised warmup subset is available, so memory is filled online.
  - Memory updates are threshold-gated to reduce poisoning from high-anomaly points.
  """

  def __init__(
    self,
    hidden_dim: int,
    latent_dim: int,
    memory_size: int,
    lr: float,
    model_device: str,
    seed: int,
  ):
    self.algorithm = "memstream"
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.memory_size = memory_size
    self.lr = lr
    self.seed = seed
    self.model_device = model_device
    self._device = _resolve_torch_device(model_device)
    self._feature_names: Optional[List[str]] = None
    self._model: Optional[_AutoEncoder] = None
    self._optimizer: Optional[torch.optim.Optimizer] = None
    self._memory_latent: Optional[torch.Tensor] = None
    self._memory_input: Optional[torch.Tensor] = None
    self._norm_mean: Optional[torch.Tensor] = None
    self._norm_std: Optional[torch.Tensor] = None
    self._mem_index = 0
    self._mem_filled = 0
    self._score_ema: Optional[float] = None
    self._score_var_ema: float = 0.0
    self._ema_alpha = 0.05
    self._beta_floor = 0.05
    self._beta_sigma = 2.5
    self._warmup_accept = max(8, min(32, self.memory_size // 4))
    self._noise_std = 1e-3
    logging.info(
      "Initialized %s (hidden=%d, latent=%d, memory=%d, lr=%.5f, device=%s)",
      self.algorithm,
      hidden_dim,
      latent_dim,
      memory_size,
      lr,
      self._device.type,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_names = sorted(features.keys())
    input_dim = len(self._feature_names)
    auto_hidden = _auto_mem_hidden_dim(input_dim)
    auto_latent = _auto_mem_latent_dim(auto_hidden)
    effective_hidden = max(self.hidden_dim, auto_hidden)
    effective_latent = max(self.latent_dim, auto_latent)
    # Keep latent bottleneck strictly below hidden size.
    effective_latent = min(effective_latent, max(4, effective_hidden - 1))
    self.hidden_dim = effective_hidden
    self.latent_dim = effective_latent
    torch.manual_seed(self.seed)
    if self._device.type == "cuda":
      torch.cuda.manual_seed_all(self.seed)
    self._model = _AutoEncoder(input_dim=input_dim, hidden_dim=effective_hidden, latent_dim=effective_latent).to(self._device)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    self._memory_latent = torch.zeros(self.memory_size, effective_latent, dtype=torch.float32, device=self._device)
    self._memory_input = torch.zeros(self.memory_size, input_dim, dtype=torch.float32, device=self._device)
    self._norm_mean = torch.zeros(input_dim, dtype=torch.float32, device=self._device)
    self._norm_std = torch.ones(input_dim, dtype=torch.float32, device=self._device)
    logging.info(
      "MemStream input_dim=%d effective_hidden=%d effective_latent=%d",
      input_dim,
      effective_hidden,
      effective_latent,
    )

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("MemStream feature names not initialized")
    vec = np.array([float(features[k]) for k in self._feature_names], dtype=np.float32)
    return torch.from_numpy(vec).to(self._device)

  def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    if self._norm_mean is None or self._norm_std is None:
      return x
    safe_std = torch.where(self._norm_std > 1e-6, self._norm_std, torch.ones_like(self._norm_std))
    return (x - self._norm_mean) / safe_std

  def _refresh_norm_from_memory(self) -> None:
    if self._memory_input is None or self._mem_filled == 0:
      return
    mem = self._memory_input[: self._mem_filled]
    self._norm_mean = mem.mean(dim=0)
    std = mem.std(dim=0, unbiased=False)
    self._norm_std = torch.where(std > 1e-6, std, torch.ones_like(std))

  def _memory_distance(self, z: torch.Tensor, k: int = 3) -> float:
    if self._memory_latent is None or self._mem_filled == 0:
      return 0.0
    memory = self._memory_latent[: self._mem_filled]
    dists = torch.norm(memory - z.unsqueeze(0), p=1, dim=1)
    k_eff = max(1, min(k, int(self._mem_filled)))
    topk = torch.topk(dists, k=k_eff, largest=False).values
    return float(topk.mean().item())

  def _adaptive_beta(self) -> float:
    if self._score_ema is None:
      return self._beta_floor
    sigma = float(np.sqrt(max(0.0, self._score_var_ema)))
    return max(self._beta_floor, self._score_ema + self._beta_sigma * sigma)

  def _update_score_stats(self, score_raw: float) -> None:
    if self._score_ema is None:
      self._score_ema = score_raw
      self._score_var_ema = 0.0
      return
    alpha = self._ema_alpha
    prev = self._score_ema
    self._score_ema = (1.0 - alpha) * self._score_ema + alpha * score_raw
    delta = score_raw - prev
    self._score_var_ema = max(0.0, (1.0 - alpha) * self._score_var_ema + alpha * (delta * delta))

  def _should_update_memory(self, score_raw: float) -> bool:
    if self._mem_filled < self._warmup_accept:
      return True
    return score_raw <= self._adaptive_beta()

  def _write_memory(self, x: torch.Tensor, z: torch.Tensor) -> None:
    if self._memory_latent is None or self._memory_input is None:
      return
    if self._mem_filled < self.memory_size:
      idx = self._mem_filled
      self._mem_filled += 1
    else:
      idx = self._mem_index
      self._mem_index = (self._mem_index + 1) % self.memory_size
    self._memory_latent[idx] = z.detach()
    self._memory_input[idx] = x.detach()
    self._refresh_norm_from_memory()

  def score_only(self, features: Dict[str, float]) -> float:
    """Return anomaly score without updating model state (read-only)."""
    score_raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, score_raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score."""
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")

    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.eval()
    with torch.no_grad():
      z_eval, recon_eval = self._model(x_norm)
      z_eval = z_eval.squeeze(0)
      recon_error = float(torch.mean((x_norm - recon_eval) ** 2).item())
      memory_error = self._memory_distance(z_eval, k=3)
      score_raw = (0.8 * memory_error) + (0.2 * recon_error)
    return max(0.0, float(score_raw))

  def score_and_learn(self, features: Dict[str, float]) -> float:
    score_raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, score_raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")

    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.train()

    with torch.no_grad():
      z_eval, recon_eval = self._model(x_norm)
      z_eval = z_eval.squeeze(0)
      recon_error = float(torch.mean((x_norm - recon_eval) ** 2).item())
      memory_error = self._memory_distance(z_eval, k=3)
      # MemStream-style memory distance is primary; recon error stabilizes early phase.
      score_raw = (0.8 * memory_error) + (0.2 * recon_error)

    self._update_score_stats(score_raw)
    update_allowed = self._should_update_memory(score_raw)

    if update_allowed:
      noisy = x_norm + (self._noise_std * torch.randn_like(x_norm))
      self._optimizer.zero_grad()
      z_train, recon_train = self._model(noisy)
      loss = torch.mean((x_norm - recon_train) ** 2)
      loss.backward()
      self._optimizer.step()
      self._write_memory(x, z_train.squeeze(0))
    return max(0.0, float(score_raw))

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    if self._model is None:
      raise RuntimeError("MemStream not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()},
      "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
      "memory_latent": self._memory_latent.cpu().numpy() if self._memory_latent is not None else None,
      "memory_input": self._memory_input.cpu().numpy() if self._memory_input is not None else None,
      "norm_mean": self._norm_mean.cpu().numpy() if self._norm_mean is not None else None,
      "norm_std": self._norm_std.cpu().numpy() if self._norm_std is not None else None,
      "mem_index": self._mem_index,
      "mem_filled": self._mem_filled,
      "score_ema": self._score_ema,
      "score_var_ema": self._score_var_ema,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    # Init model from feature names so architecture exists
    dummy = {k: 0.0 for k in self._feature_names}
    self._init_from_features(dummy)
    self._model.load_state_dict(
      {k: v.to(self._device) for k, v in state["model_state"].items()}
    )
    if state["optimizer_state"] and self._optimizer:
      self._optimizer.load_state_dict(state["optimizer_state"])
    if state["memory_latent"] is not None:
      self._memory_latent = torch.from_numpy(state["memory_latent"]).to(self._device)
    if state["memory_input"] is not None:
      self._memory_input = torch.from_numpy(state["memory_input"]).to(self._device)
    if state["norm_mean"] is not None:
      self._norm_mean = torch.from_numpy(state["norm_mean"]).to(self._device)
    if state["norm_std"] is not None:
      self._norm_std = torch.from_numpy(state["norm_std"]).to(self._device)
    self._mem_index = int(state["mem_index"])
    self._mem_filled = int(state["mem_filled"])
    self._score_ema = state.get("score_ema")
    self._score_var_ema = float(state.get("score_var_ema", 0.0))


class OnlineZScore:
  """Simple online per-feature Z-score baseline. (Self-implementation)"""

  def __init__(self, min_count: int, std_floor: float, model_device: str, seed: int):
    del seed  # deterministic, no RNG state required
    self.algorithm = "zscore"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but zscore is CPU-only; using CPU",
        self.algorithm,
      )
    self.min_count = int(max(1, min_count))
    self.std_floor = float(max(1e-12, std_floor))
    self._feature_names: Optional[List[str]] = None
    self._count = 0
    self._mean: Optional[np.ndarray] = None
    self._m2: Optional[np.ndarray] = None
    logging.info(
      "Initialized %s (min_count=%d, std_floor=%.6f)",
      self.algorithm,
      self.min_count,
      self.std_floor,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_names = sorted(features.keys())
    dim = len(self._feature_names)
    self._count = 0
    self._mean = np.zeros(dim, dtype=np.float64)
    self._m2 = np.zeros(dim, dtype=np.float64)
    logging.info("ZScore input_dim=%d", dim)

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("ZScore feature names not initialized")
    return np.array([float(features[k]) for k in self._feature_names], dtype=np.float64)

  def _raw_from_vector(self, x: np.ndarray) -> float:
    mean = self._mean
    m2 = self._m2
    if mean is None or m2 is None:
      raise RuntimeError("ZScore not initialized")
    if self._count < self.min_count:
      return 0.0
    denom = float(max(1, self._count - 1))
    variance = np.maximum(m2 / denom, self.std_floor * self.std_floor)
    std = np.sqrt(variance)
    z_abs = np.abs((x - mean) / std)
    return float(np.mean(z_abs))

  def _learn_vector(self, x: np.ndarray) -> None:
    mean = self._mean
    m2 = self._m2
    if mean is None or m2 is None:
      raise RuntimeError("ZScore not initialized")
    self._count += 1
    delta = x - mean
    mean += delta / float(self._count)
    delta2 = x - mean
    m2 += delta * delta2

  def score_only(self, features: Dict[str, float]) -> float:
    raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    x = self._vectorize(features)
    return max(0.0, self._raw_from_vector(x))

  def score_and_learn(self, features: Dict[str, float]) -> float:
    raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    x = self._vectorize(features)
    raw = self._raw_from_vector(x)
    self._learn_vector(x)
    return max(0.0, raw)

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    if self._feature_names is None or self._mean is None or self._m2 is None:
      raise RuntimeError("ZScore not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "min_count": self.min_count,
      "std_floor": self.std_floor,
      "count": self._count,
      "mean": self._mean,
      "m2": self._m2,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self.min_count = int(state.get("min_count", self.min_count))
    self.std_floor = float(state.get("std_floor", self.std_floor))
    self._count = int(state["count"])
    self._mean = np.asarray(state["mean"], dtype=np.float64)
    self._m2 = np.asarray(state["m2"], dtype=np.float64)


class OnlineKNN:
  """Online KNN anomaly detector with sliding memory. (Library-based wrapper)"""

  def __init__(self, k: int, memory_size: int, metric: str, model_device: str, seed: int):
    del seed  # deterministic, no RNG state required
    self.algorithm = "knn"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but KNN is CPU-only; using CPU",
        self.algorithm,
      )
    self.k = int(max(1, k))
    self.memory_size = int(max(8, memory_size))
    self.metric = str(metric or "euclidean")
    self._feature_names: Optional[List[str]] = None
    self._memory: Optional[np.ndarray] = None
    self._mem_index = 0
    self._mem_filled = 0
    logging.info(
      "Initialized %s (k=%d, memory_size=%d, metric=%s)",
      self.algorithm,
      self.k,
      self.memory_size,
      self.metric,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    if NearestNeighbors is None:
      raise RuntimeError("scikit-learn not available. Install detector deps: `uv sync --extra detector`.")
    self._feature_names = sorted(features.keys())
    dim = len(self._feature_names)
    self._memory = np.zeros((self.memory_size, dim), dtype=np.float64)
    self._mem_index = 0
    self._mem_filled = 0
    logging.info("KNN input_dim=%d", dim)

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("KNN feature names not initialized")
    return np.array([float(features[k]) for k in self._feature_names], dtype=np.float64)

  def _raw_from_vector(self, x: np.ndarray) -> float:
    if NearestNeighbors is None:
      raise RuntimeError("scikit-learn not available. Install detector deps: `uv sync --extra detector`.")
    if self._memory is None:
      raise RuntimeError("KNN not initialized")
    if self._mem_filled <= 0:
      return 0.0
    memory = self._memory[: self._mem_filled]
    n_neighbors = min(self.k, self._mem_filled)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
    nn.fit(memory)
    dists, _ = nn.kneighbors(x.reshape(1, -1), n_neighbors=n_neighbors, return_distance=True)
    raw = float(np.mean(dists))
    if not np.isfinite(raw):
      logging.warning("KNN produced non-finite score (%s); falling back to 0.0", raw)
      return 0.0
    return max(0.0, raw)

  def _learn_vector(self, x: np.ndarray) -> None:
    if self._memory is None:
      raise RuntimeError("KNN not initialized")
    if self._mem_filled < self.memory_size:
      idx = self._mem_filled
      self._mem_filled += 1
    else:
      idx = self._mem_index
      self._mem_index = (self._mem_index + 1) % self.memory_size
    self._memory[idx] = x

  def score_only(self, features: Dict[str, float]) -> float:
    raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    x = self._vectorize(features)
    return self._raw_from_vector(x)

  def score_and_learn(self, features: Dict[str, float]) -> float:
    raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    x = self._vectorize(features)
    raw = self._raw_from_vector(x)
    self._learn_vector(x)
    return max(0.0, raw)

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    if self._feature_names is None or self._memory is None:
      raise RuntimeError("KNN not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "k": self.k,
      "memory_size": self.memory_size,
      "metric": self.metric,
      "memory": self._memory,
      "mem_index": self._mem_index,
      "mem_filled": self._mem_filled,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self.k = int(state.get("k", self.k))
    self.memory_size = int(state.get("memory_size", self.memory_size))
    self.metric = str(state.get("metric", self.metric))
    self._memory = np.asarray(state["memory"], dtype=np.float64)
    self._mem_index = int(state["mem_index"])
    self._mem_filled = int(state["mem_filled"])


class OnlineFreq1D:
  """
  Simple frequency / rarity baseline using independent 1D marginals. (Self-implementation)

  - Numeric features: fixed-bin histogram over [0, 1] (clamped).
  - Categorical features: capped per-feature count tables for *_hash and binary flags.

  Score (raw) = mean over features of excess surprisal:
    max(0, -log p(x) - (-log p_uniform))
  so cold-start produces score ~0 (uniform baseline).
  """

  _BINARY_FEATURES = frozenset({"return_success", "file_sensitive_path", "file_tmp_path", "process_is_execve", "process_is_fork"})

  def __init__(
    self,
    bins: int,
    alpha: float,
    decay: float,
    max_categories: int,
    model_device: str,
    seed: int,
  ):
    del seed  # deterministic, no RNG state required
    self.algorithm = "freq1d"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but freq1d is CPU-only; using CPU",
        self.algorithm,
      )

    self.bins = int(max(2, bins))
    self.alpha = float(alpha)
    if not (self.alpha > 0.0):
      raise ValueError("freq1d alpha must be > 0")
    self.decay = float(decay)
    if not (0.0 < self.decay <= 1.0):
      raise ValueError("freq1d decay must be in (0, 1]")
    self.max_categories = int(max(4, max_categories))

    self._feature_names: Optional[List[str]] = None
    self._kind: Optional[List[str]] = None  # "num" or "cat"
    self._cat_mode: Optional[List[Optional[str]]] = None  # "hash" | "binary" | None

    self._num_slot: Optional[List[int]] = None
    self._num_counts: List[np.ndarray] = []
    self._num_scale: List[float] = []

    self._cat_slot: Optional[List[int]] = None
    self._cat_counts: List[Dict[int, float]] = []
    self._cat_other: List[float] = []
    self._cat_scale: List[float] = []

    logging.info(
      "Initialized %s (bins=%d, alpha=%.3f, decay=%.4f, max_categories=%d)",
      self.algorithm,
      self.bins,
      self.alpha,
      self.decay,
      self.max_categories,
    )

  def _is_categorical(self, name: str) -> tuple[bool, Optional[str]]:
    if name.endswith("_hash"):
      return True, "hash"
    if name in self._BINARY_FEATURES:
      return True, "binary"
    return False, None

  def _init_from_features(self, features: Dict[str, float]) -> None:
    self._feature_names = sorted(features.keys())
    kinds: List[str] = []
    cat_modes: List[Optional[str]] = []
    num_slot: List[int] = [-1] * len(self._feature_names)
    cat_slot: List[int] = [-1] * len(self._feature_names)

    for i, name in enumerate(self._feature_names):
      is_cat, mode = self._is_categorical(name)
      if is_cat:
        kinds.append("cat")
        cat_modes.append(mode)
        cat_slot[i] = len(self._cat_counts)
        self._cat_counts.append({})
        self._cat_other.append(0.0)
        self._cat_scale.append(1.0)
      else:
        kinds.append("num")
        cat_modes.append(None)
        num_slot[i] = len(self._num_counts)
        self._num_counts.append(np.zeros(self.bins, dtype=np.float64))
        self._num_scale.append(1.0)

    self._kind = kinds
    self._cat_mode = cat_modes
    self._num_slot = num_slot
    self._cat_slot = cat_slot
    logging.info(
      "Freq1D initialized with %d features (%d numeric, %d categorical)",
      len(self._feature_names),
      len(self._num_counts),
      len(self._cat_counts),
    )

  def _maybe_rescale_numeric(self, slot: int) -> None:
    scale = self._num_scale[slot]
    if scale >= 1e-6:
      return
    self._num_counts[slot] *= scale
    self._num_scale[slot] = 1.0

  def _maybe_rescale_categorical(self, slot: int) -> None:
    scale = self._cat_scale[slot]
    if scale >= 1e-6:
      return
    if self._cat_counts[slot]:
      for k in list(self._cat_counts[slot].keys()):
        self._cat_counts[slot][k] *= scale
    self._cat_other[slot] *= scale
    self._cat_scale[slot] = 1.0

  def _bin_numeric(self, x: float) -> int:
    v = float(x)
    if v <= 0.0:
      return 0
    if v >= 1.0:
      return self.bins - 1
    idx = int(v * self.bins)
    if idx < 0:
      return 0
    if idx >= self.bins:
      return self.bins - 1
    return idx

  def _cat_key(self, mode: str, x: float) -> int:
    v = float(x)
    if mode == "binary":
      return 1 if v >= 0.5 else 0
    # hash: _hash01 uses 4 decimal digits (0..9999)/10000. Treat as categorical ID.
    if v <= 0.0:
      return 0
    if v >= 1.0:
      return 9999
    k = int(v * 10000.0 + 1e-6)
    if k < 0:
      return 0
    if k > 9999:
      return 9999
    return k

  def _score_numeric_slot(self, slot: int, x: float) -> float:
    counts = self._num_counts[slot]
    total = float(counts.sum())
    idx = self._bin_numeric(x)
    c = float(counts[idx])
    p = (c + self.alpha) / (total + self.alpha * self.bins) if total >= 0.0 else 1.0 / float(self.bins)
    p = max(1e-300, float(p))
    baseline = math.log(float(self.bins))
    return max(0.0, -math.log(p) - baseline)

  def _learn_numeric_slot(self, slot: int, x: float) -> None:
    self._num_scale[slot] *= self.decay
    self._maybe_rescale_numeric(slot)
    scale = self._num_scale[slot]
    idx = self._bin_numeric(x)
    self._num_counts[slot][idx] += 1.0 / max(scale, 1e-12)

  def _score_cat_slot(self, slot: int, mode: str, x: float) -> float:
    counts = self._cat_counts[slot]
    other = float(self._cat_other[slot])
    total = other + float(sum(counts.values()))
    k = self._cat_key(mode, x)
    c = float(counts.get(k, 0.0))
    # K = tracked categories + OTHER bucket
    alphabet = max(2, len(counts) + 1)
    p = (c + self.alpha) / (total + self.alpha * alphabet) if total >= 0.0 else 1.0 / float(alphabet)
    p = max(1e-300, float(p))
    baseline = math.log(float(alphabet))
    return max(0.0, -math.log(p) - baseline)

  def _learn_cat_slot(self, slot: int, mode: str, x: float) -> None:
    self._cat_scale[slot] *= self.decay
    self._maybe_rescale_categorical(slot)
    scale = self._cat_scale[slot]
    k = self._cat_key(mode, x)
    counts = self._cat_counts[slot]
    if k in counts:
      counts[k] = float(counts[k]) + (1.0 / max(scale, 1e-12))
      return
    if len(counts) < self.max_categories:
      counts[k] = 1.0 / max(scale, 1e-12)
      return
    self._cat_other[slot] = float(self._cat_other[slot]) + (1.0 / max(scale, 1e-12))

  def _ensure_init(self, features: Dict[str, float]) -> None:
    if self._feature_names is None:
      self._init_from_features(features)

  def score_only(self, features: Dict[str, float]) -> float:
    raw = self.score_only_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_only_raw(self, features: Dict[str, float]) -> float:
    self._ensure_init(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    total = 0.0
    n = 0
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        if slot >= 0:
          total += self._score_numeric_slot(slot, x)
          n += 1
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        if slot >= 0:
          total += self._score_cat_slot(slot, mode, x)
          n += 1
    return max(0.0, total / float(max(1, n)))

  def score_and_learn(self, features: Dict[str, float]) -> float:
    raw = self.score_and_learn_raw(features)
    return 1.0 - float(np.exp(-max(0.0, raw)))

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    self._ensure_init(features)
    raw = self.score_only_raw(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        if slot >= 0:
          self._learn_numeric_slot(slot, x)
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        if slot >= 0:
          self._learn_cat_slot(slot, mode, x)
    return max(0.0, float(raw))

  def get_state(self) -> Dict[str, Any]:
    if self._feature_names is None or self._kind is None or self._cat_mode is None or self._num_slot is None or self._cat_slot is None:
      raise RuntimeError("Freq1D not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "bins": self.bins,
      "alpha": self.alpha,
      "decay": self.decay,
      "max_categories": self.max_categories,
      "kind": list(self._kind),
      "cat_mode": list(self._cat_mode),
      "num_counts": [c for c in self._num_counts],
      "num_scale": list(self._num_scale),
      "cat_counts": [dict(d) for d in self._cat_counts],
      "cat_other": list(self._cat_other),
      "cat_scale": list(self._cat_scale),
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self.bins = int(state.get("bins", self.bins))
    self.alpha = float(state.get("alpha", self.alpha))
    self.decay = float(state.get("decay", self.decay))
    self.max_categories = int(state.get("max_categories", self.max_categories))
    self._feature_names = list(state["feature_names"])
    self._kind = list(state["kind"])
    self._cat_mode = list(state["cat_mode"])

    # Rebuild slot mappings and storage
    self._num_counts = [np.asarray(c, dtype=np.float64) for c in state.get("num_counts", [])]
    self._num_scale = [float(s) for s in state.get("num_scale", [])]
    self._cat_counts = [dict(d) for d in state.get("cat_counts", [])]
    self._cat_other = [float(v) for v in state.get("cat_other", [])]
    self._cat_scale = [float(s) for s in state.get("cat_scale", [])]

    num_slot: List[int] = [-1] * len(self._feature_names)
    cat_slot: List[int] = [-1] * len(self._feature_names)
    n_num = 0
    n_cat = 0
    for i, k in enumerate(self._kind):
      if k == "num":
        num_slot[i] = n_num
        n_num += 1
      else:
        cat_slot[i] = n_cat
        n_cat += 1
    self._num_slot = num_slot
    self._cat_slot = cat_slot


class OnlineAnomalyDetector:
  """Factory wrapper for online anomaly detectors. (Self-implementation)"""

  def __init__(
    self,
    algorithm: str,
    *,
    hst_n_trees: int = 25,
    hst_height: int = 15,
    hst_window_size: int = 250,
    loda_n_projections: int = 20,
    loda_bins: int = 64,
    loda_range: float = 3.0,
    loda_ema_alpha: float = 0.01,
    loda_hist_decay: float = 1.0,
    kitnet_max_size_ae: int = 10,
    kitnet_grace_feature_mapping: int = 10000,
    kitnet_grace_anomaly_detector: int = 50000,
    kitnet_learning_rate: float = 0.1,
    kitnet_hidden_ratio: float = 0.75,
    mem_hidden_dim: int = 32,
    mem_latent_dim: int = 8,
    mem_memory_size: int = 128,
    mem_lr: float = 0.001,
    zscore_min_count: int = 20,
    zscore_std_floor: float = 1e-3,
    knn_k: int = 5,
    knn_memory_size: int = 1024,
    knn_metric: str = "euclidean",
    freq1d_bins: int = 64,
    freq1d_alpha: float = 1.0,
    freq1d_decay: float = 1.0,
    freq1d_max_categories: int = 2048,
    model_device: str = "auto",
    seed: int = 42,
  ):
    algo = algorithm.lower()
    if algo == "halfspacetrees":
      self.impl = OnlineHalfSpaceTrees(
        n_trees=hst_n_trees,
        height=hst_height,
        window_size=hst_window_size,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "loda":
      self.impl = OnlineLODA(
        n_projections=loda_n_projections,
        bins=loda_bins,
        proj_range=loda_range,
        ema_alpha=loda_ema_alpha,
        hist_decay=loda_hist_decay,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "kitnet":
      self.impl = OnlineKitNet(
        max_size_ae=kitnet_max_size_ae,
        grace_feature_mapping=kitnet_grace_feature_mapping,
        grace_anomaly_detector=kitnet_grace_anomaly_detector,
        learning_rate=kitnet_learning_rate,
        hidden_ratio=kitnet_hidden_ratio,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "memstream":
      self.impl = OnlineMemStream(
        hidden_dim=mem_hidden_dim,
        latent_dim=mem_latent_dim,
        memory_size=mem_memory_size,
        lr=mem_lr,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "zscore":
      self.impl = OnlineZScore(
        min_count=zscore_min_count,
        std_floor=zscore_std_floor,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "knn":
      self.impl = OnlineKNN(
        k=knn_k,
        memory_size=knn_memory_size,
        metric=knn_metric,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "freq1d":
      self.impl = OnlineFreq1D(
        bins=freq1d_bins,
        alpha=freq1d_alpha,
        decay=freq1d_decay,
        max_categories=freq1d_max_categories,
        model_device=model_device,
        seed=seed,
      )
    else:
      raise ValueError("Unknown algorithm: %s. Choose from: halfspacetrees, loda, kitnet, memstream, zscore, knn, freq1d" % algorithm)
    self.algorithm = self.impl.algorithm

  def score_only(self, features: Dict[str, float]) -> float:
    """Return anomaly score without updating model state (read-only)."""
    return self.impl.score_only(features)

  def score_only_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score."""
    return self.impl.score_only_raw(features)

  def score_and_learn(self, features: Dict[str, float]) -> float:
    return self.impl.score_and_learn(features)

  def score_and_learn_raw(self, features: Dict[str, float]) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    return self.impl.score_and_learn_raw(features)

  def save_checkpoint(self, path: Path, checkpoint_index: int) -> None:
    """Save detector state after learning events 0..checkpoint_index-1."""
    state = {
      "algorithm": self.algorithm,
      "checkpoint_index": checkpoint_index,
      "impl_state": self.impl.get_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
      pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Saved checkpoint at index %d to %s", checkpoint_index, path)

  def load_checkpoint(self, path: Path) -> int:
    """Load detector state. Returns checkpoint_index (events 0..index-1 were learned)."""
    path = Path(path)
    with path.open("rb") as f:
      state = pickle.load(f)
    if state["algorithm"] != self.algorithm:
      raise ValueError(
        "Checkpoint algorithm %r does not match detector %r"
        % (state["algorithm"], self.algorithm)
      )
    self.impl.set_state(state["impl_state"])
    idx = int(state["checkpoint_index"])
    logging.info("Loaded checkpoint from %s (index %d)", path, idx)
    return idx

  def compute_feature_attribution(
    self,
    features: Dict[str, float],
    epsilon: float = 0.01,
    *,
    score_mode: str = "raw",
  ) -> Tuple[float, Dict[str, float]]:
    """Model-agnostic perturbation-based attribution. Returns (score, {name: attribution})."""
    if score_mode not in ("raw", "lograw", "scaled"):
      raise ValueError("score_mode must be 'raw', 'lograw', or 'scaled'")
    if score_mode == "raw":
      score_fn = self.score_only_raw
    elif score_mode == "scaled":
      score_fn = self.score_only
    else:
      # log(1 + raw) behaves like percent change for large raw values:
      # log(1+raw+Δ) - log(1+raw) ≈ Δ/(1+raw)
      score_fn = lambda f: math.log1p(max(0.0, float(self.score_only_raw(f))))
    return compute_feature_attribution(self, features, epsilon, score_fn=score_fn)


def compute_feature_attribution(
  detector: OnlineAnomalyDetector,
  features: Dict[str, float],
  epsilon: float = 0.01,
  binary_threshold: float = 0.01,
  *,
  score_fn=None,
) -> Tuple[float, Dict[str, float]]:
  """
  Model-agnostic perturbation-based feature attribution.
  Returns (score, {feature_name: attribution}).
  Positive attribution = feature pushes score up (more anomalous).

  Binary features (value in [0, binary_threshold] or [1-binary_threshold, 1]) use flip:
  attribution = (score(feature=1) - score(feature=0)) * value.
  Avoids invalid interpolation and threshold effects from ±epsilon.

  Continuous features use finite difference:
  attribution = (s_plus - s_minus) / (2*epsilon) * value.
  """
  if score_fn is None:
    score_fn = detector.score_only
  score = float(score_fn(features))
  names = sorted(features.keys())
  attribution: Dict[str, float] = {}
  for name in names:
    val = float(features[name])
    is_binary = val <= binary_threshold or val >= (1.0 - binary_threshold)
    if is_binary:
      features_0 = dict(features)
      features_0[name] = 0.0
      features_1 = dict(features)
      features_1[name] = 1.0
      s_0 = float(score_fn(features_0))
      s_1 = float(score_fn(features_1))
      attribution[name] = float((s_1 - s_0) * val)
    else:
      val_plus = max(0.0, min(1.0, val + epsilon))
      val_minus = max(0.0, min(1.0, val - epsilon))
      features_plus = dict(features)
      features_plus[name] = val_plus
      features_minus = dict(features)
      features_minus[name] = val_minus
      s_plus = float(score_fn(features_plus))
      s_minus = float(score_fn(features_minus))
      grad_approx = (s_plus - s_minus) / (2.0 * epsilon) if epsilon > 0 else 0.0
      attribution[name] = float(grad_approx * val)
  return (score, attribution)
