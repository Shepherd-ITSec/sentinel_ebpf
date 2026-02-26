"""Online anomaly detection models."""
import contextlib
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from river import anomaly
try:
  from pysad.models import KitNet as PySADKitNet
except ImportError:
  PySADKitNet = None


class OnlineHalfSpaceTrees:
  """Online anomaly detector using River Half-Space Trees."""

  def __init__(self, n_trees: int, height: int, window_size: int, model_device: str, seed: int):
    self.algorithm = "halfspacetrees"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but River HalfSpaceTrees is CPU-only; using CPU",
        self.algorithm,
      )
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
    self._feature_keys: Optional[List[str]] = None
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
    self._feature_keys = sorted(features.keys())
    dim = len(self._feature_keys)
    rng = np.random.default_rng(self.seed)
    # LODA-style sparse random projections: about sqrt(d) non-zero entries.
    nnz = max(1, int(np.sqrt(dim)))
    weights = np.zeros((self.n_projections, dim), dtype=np.float32)
    for i in range(self.n_projections):
      idx = rng.choice(dim, size=nnz, replace=False)
      w = rng.normal(0.0, 1.0, size=nnz).astype(np.float32)
      norm = float(np.linalg.norm(w))
      if norm > 1e-12:
        w = w / norm
      weights[i, idx] = w
    self._weights = torch.from_numpy(weights).to(self._device)
    self._mean = torch.zeros(self.n_projections, dtype=torch.float32, device=self._device)
    self._var = torch.ones(self.n_projections, dtype=torch.float32, device=self._device)
    self._counts = torch.zeros((self.n_projections, self.bins), dtype=torch.float32, device=self._device)

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_keys is None:
      self._init_from_features(features)
    if self._feature_keys is None:
      raise RuntimeError("LODA feature keys not initialized")
    vec = np.array([float(features[k]) for k in self._feature_keys], dtype=np.float32)
    return torch.from_numpy(vec).to(self._device)

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

    if self.hist_decay < 1.0:
      counts *= self.hist_decay

    projections = weights @ x

    std = torch.sqrt(var + 1e-6)
    normalized = (projections - mean) / std
    normalized = torch.clamp(normalized, -self.proj_range, self.proj_range)
    bin_idx = ((normalized + self.proj_range) / (2.0 * self.proj_range) * self.bins).to(torch.long)
    bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)

    row_idx = torch.arange(self.n_projections, device=self._device)
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
    return 1.0 - float(np.exp(-score))


class OnlineKitNet:
  """Online anomaly detector using PySAD KitNet."""

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
    self._feature_keys: Optional[List[str]] = None
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
    if PySADKitNet is None:
      raise RuntimeError("PySAD KitNet not available. Install detector deps: `uv sync --extra detector`.")
    self._feature_keys = sorted(features.keys())
    self._model = PySADKitNet(
      max_size_ae=self.max_size_ae,
      grace_feature_mapping=self.grace_feature_mapping,
      grace_anomaly_detector=self.grace_anomaly_detector,
      learning_rate=self.learning_rate,
      hidden_ratio=self.hidden_ratio,
    )

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_keys is None:
      self._init_from_features(features)
    if self._feature_keys is None:
      raise RuntimeError("KitNet feature keys not initialized")
    return np.array([float(features[k]) for k in self._feature_keys], dtype=np.float64)

  def score_and_learn(self, features: Dict[str, float]) -> float:
    x = self._vectorize(features)
    if self._model is None:
      raise RuntimeError("KitNet not initialized")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      if hasattr(self._model, "fit_score_partial"):
        raw = float(self._model.fit_score_partial(x))
      else:
        raw = float(self._model.score_partial(x))
        self._model.fit_partial(x)
    if not np.isfinite(raw):
      raw = 0.0
    return 1.0 - float(np.exp(-max(0.0, raw)))


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


class OnlineMemStream:
  """
  Online MemStream-style detector using an autoencoder + latent memory.

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
    self._feature_keys: Optional[List[str]] = None
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
    self._feature_keys = sorted(features.keys())
    input_dim = len(self._feature_keys)
    torch.manual_seed(self.seed)
    if self._device.type == "cuda":
      torch.cuda.manual_seed_all(self.seed)
    self._model = _AutoEncoder(input_dim=input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim).to(self._device)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    self._memory_latent = torch.zeros(self.memory_size, self.latent_dim, dtype=torch.float32, device=self._device)
    self._memory_input = torch.zeros(self.memory_size, input_dim, dtype=torch.float32, device=self._device)
    self._norm_mean = torch.zeros(input_dim, dtype=torch.float32, device=self._device)
    self._norm_std = torch.ones(input_dim, dtype=torch.float32, device=self._device)

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_keys is None:
      self._init_from_features(features)
    if self._feature_keys is None:
      raise RuntimeError("MemStream feature keys not initialized")
    vec = np.array([float(features[k]) for k in self._feature_keys], dtype=np.float32)
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

  def score_and_learn(self, features: Dict[str, float]) -> float:
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

    score = 1.0 - float(np.exp(-max(0.0, score_raw)))
    return score


class OnlineAnomalyDetector:
  """Factory wrapper for online anomaly detectors."""

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
    else:
      raise ValueError("Unknown algorithm: %s. Choose from: halfspacetrees, loda, kitnet, memstream" % algorithm)
    self.algorithm = self.impl.algorithm

  def score_and_learn(self, features: Dict[str, float]) -> float:
    return self.impl.score_and_learn(features)
