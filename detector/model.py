"""Online anomaly detection models."""
import contextlib
import bisect
import io
import logging
import math
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from fenwick import FenwickTree
import torch
from scipy.stats import norm as scipy_norm
try:
  from river import anomaly as River_Anomaly
except ImportError:
  River_Anomaly = None
try:
  from pysad.models import KitNet as Pysad_KitNet
  from pysad.models import LODA as Pysad_LODA
except ImportError:
  Pysad_KitNet = None
  Pysad_LODA = None
try:
  from sklearn.neighbors import NearestNeighbors
except ImportError:
  NearestNeighbors = None

import events_pb2
from detector.meta import Meta
from detector.sequence.mlp import OnlineSequenceMLP
from detector.sequence.context import SequenceFeatureMeta


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

  def score_only(self, features: Dict[str, float], *, meta: Meta | None = None) -> tuple[float, float]:
    raw = self.score_only_raw(features, meta=meta)
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return (float(raw), scaled)

  def score_and_learn(self, features: Dict[str, float], *, meta: Meta | None = None) -> tuple[float, float]:
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

  def score_only_raw(self, features: Dict[str, float], *, meta: Meta | None = None) -> float:
    """Return the underlying (unsquashed) score."""
    return float(self.model.score_one(features))

  def score_only(self, features: Dict[str, float]) -> tuple[float, float]:
    raw = self.score_only_raw(features)
    return (float(raw), float(raw))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Meta | None = None) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    score = self.model.score_one(features)
    self.model.learn_one(features)  # in-place; learn_one returns None
    return float(score)

  def score_and_learn(self, features: Dict[str, float]) -> tuple[float, float]:
    raw = self.score_and_learn_raw(features)
    return (float(raw), float(raw))

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    return {"model": self.model}

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self.model = state["model"]


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

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
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
    surprisal = -torch.log(probs)
    baseline = math.log(float(self.bins))
    excess = torch.clamp(surprisal - baseline, min=0.0)
    score = float(torch.mean(excess).item())

    # Learn after scoring so the current event doesn't lower its own anomaly score.
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


class OnlinePysadLODA(_BothScoresMixin):
  """Online anomaly detector using PySAD LODA. (Library-based wrapper, paper-faithful)"""

  def __init__(
    self,
    num_bins: int,
    num_random_cuts: int,
    model_device: str,
    seed: int,
  ):
    self.algorithm = "loda"
    self.num_bins = num_bins
    self.num_random_cuts = num_random_cuts
    self.seed = seed
    self._feature_names: Optional[List[str]] = None
    self._model = None
    logging.info(
      "Initialized %s (PySAD, bins=%d, cuts=%d)",
      self.algorithm,
      num_bins,
      num_random_cuts,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    if Pysad_LODA is None:
      raise RuntimeError("PySAD LODA not available. Install detector deps: `uv sync --extra detector`.")
    self._feature_names = sorted(features.keys())
    self._model = Pysad_LODA(num_bins=self.num_bins, num_random_cuts=self.num_random_cuts)
    logging.info("PySAD LODA input_dim=%d", len(self._feature_names))

  def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("PySAD LODA feature names not initialized")
    return np.array([float(features[k]) for k in self._feature_names], dtype=np.float64)

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    """Return the underlying (unsquashed) score."""
    x = self._vectorize(features)
    if self._model is None:
      raise RuntimeError("PySAD LODA not initialized")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      try:
        raw_val = self._model.score_partial(x)
        raw = float(np.asarray(raw_val).flat[0]) if raw_val is not None else 0.0
      except AttributeError as exc:
        exc_str = str(exc).lower()
        if "projections" not in exc_str and "model" not in exc_str and "histogram" not in exc_str:
          raise
        logging.debug("PySAD LODA cold-start before fit; using raw score 0.0")
        raw = 0.0
    if not np.isfinite(raw):
      logging.warning("PySAD LODA produced non-finite score (%s); falling back to 0.0", raw)
      raw = 0.0
    return max(0.0, float(raw))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    x = self._vectorize(features)
    if self._model is None:
      raise RuntimeError("PySAD LODA not initialized")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
      try:
        raw_val = self._model.score_partial(x)
        raw = float(np.asarray(raw_val).flat[0]) if raw_val is not None else 0.0
        need_fit = True
      except AttributeError as exc:
        exc_str = str(exc).lower()
        if "projections" not in exc_str and "model" not in exc_str and "histogram" not in exc_str:
          raise
        logging.debug("PySAD LODA cold-start before fit; fitting first, using raw score 0.0")
        self._model.fit_partial(x)
        raw = 0.0
        need_fit = False
      if need_fit:
        self._model.fit_partial(x)
    if not np.isfinite(raw):
      logging.warning("PySAD LODA produced non-finite score (%s); falling back to 0.0", raw)
      raw = 0.0
    return max(0.0, float(raw))

  def get_state(self) -> Dict[str, Any]:
    """Return picklable state for checkpointing."""
    if self._model is None:
      raise RuntimeError("PySAD LODA not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "model": self._model,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self._model = state["model"]


class OnlineKitNet(_BothScoresMixin):
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

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
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

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
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


class _MemStreamAutoEncoder(torch.nn.Module):
  """Paper architecture: Linear(in, 2*in) + Tanh, Linear(2*in, in). Latent = 2×input_dim."""

  def __init__(self, input_dim: int):
    super().__init__()
    latent_dim = 2 * input_dim
    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(input_dim, latent_dim),
      torch.nn.Tanh(),
    )
    self.decoder = torch.nn.Linear(latent_dim, input_dim)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = self.encoder(x)
    recon = self.decoder(z)
    return z, recon


class OnlineMemStream(_BothScoresMixin):
  """
  MemStream-style detector: paper-aligned autoencoder + latent memory.

  Paper: MemStream (WWW'22) https://arxiv.org/abs/2106.03837
  Official code: https://github.com/Stream-AD/MemStream

  Architecture: encoder Linear(in, 2*in) + Tanh, decoder Linear(2*in, in).
  Score: K-NN discounted L1 distance to memory. Memory: FIFO when score ≤ β.
  When no warmup path is set, memory is filled online (adaptive).
  """

  def __init__(
    self,
    memory_size: int,
    lr: float,
    beta: float,
    k: int,
    gamma: float,
    input_mode: str,
    freq1d_bins: int,
    freq1d_alpha: float,
    freq1d_decay: float,
    freq1d_max_categories: int,
    model_device: str,
    seed: int,
    warmup_accept: int = 512,
  ):
    self.algorithm = "memstream"
    self.memory_size = memory_size
    self.lr = lr
    self.beta = beta
    self.k = k
    self.gamma = gamma
    self.seed = seed
    self.input_mode = str(input_mode).strip().lower()
    if self.input_mode not in ("raw", "freq1d_u", "freq1d_z", "freq1d_surprisal", "freq1d_z_surprisal"):
      raise ValueError(
        "memstream input_mode must be one of: raw, freq1d_u, freq1d_z, freq1d_surprisal, freq1d_z_surprisal"
      )
    self._frontend_u_clamp = 1e-6
    self._frontend_z_clip = 8.0
    self._frontend_surprisal_clip = 12.0
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
    self._warmup_accept = warmup_accept
    self._noise_std = 1e-3
    self._accepted_updates = 0
    self._rejected_updates = 0
    self._overwrite_updates = 0
    self._last_debug: Dict[str, Any] = {}
    self._frontend_marginals: Optional[OnlineFreq1D] = None
    if self.input_mode != "raw":
      self._frontend_marginals = OnlineFreq1D(
        bins=freq1d_bins,
        alpha=freq1d_alpha,
        decay=freq1d_decay,
        max_categories=freq1d_max_categories,
        aggregation="mean",
        topk=1,
        soft_topk_temperature=1.0,
        model_device="cpu",
        seed=0,
      )
    self._exp: Optional[torch.Tensor] = None  # gamma^i for i in range(k), set on device when model inits
    logging.info(
      "Initialized %s (memory=%d, lr=%.5f, beta=%.4f, k=%d, gamma=%.4f, input_mode=%s, device=%s)",
      self.algorithm,
      memory_size,
      lr,
      beta,
      k,
      gamma,
      self.input_mode,
      self._device.type,
    )

  def _frontend_transform(self, features: Dict[str, float]) -> Dict[str, float]:
    if self.input_mode == "raw":
      return dict(features)
    if self._frontend_marginals is None:
      raise RuntimeError("MemStream frontend marginals not initialized")
    u = self._frontend_marginals.get_cdf_vector(features)
    names = self._frontend_marginals._feature_names
    if names is None:
      raise RuntimeError("MemStream frontend feature names not initialized")
    if self.input_mode == "freq1d_u":
      return {f"u::{name}": float(u[i]) for i, name in enumerate(names)}
    u_clipped = np.clip(u, self._frontend_u_clamp, 1.0 - self._frontend_u_clamp)
    z = np.clip(scipy_norm.ppf(u_clipped), -self._frontend_z_clip, self._frontend_z_clip)
    if self.input_mode == "freq1d_z":
      return {f"z::{name}": float(z[i]) for i, name in enumerate(names)}
    excess = np.clip(self._frontend_marginals.get_excess_vector(features), 0.0, self._frontend_surprisal_clip)
    if self.input_mode == "freq1d_surprisal":
      return {f"s::{name}": float(excess[i]) for i, name in enumerate(names)}
    transformed: Dict[str, float] = {}
    for i, name in enumerate(names):
      transformed[f"z::{name}"] = float(z[i])
      transformed[f"s::{name}"] = float(excess[i])
    return transformed

  def _init_model_for_feature_names(self, feature_names: List[str]) -> None:
    self._feature_names = list(feature_names)
    input_dim = len(self._feature_names)
    latent_dim = 2 * input_dim  # Paper: latent = 2×input_dim
    torch.manual_seed(self.seed)
    if self._device.type == "cuda":
      torch.cuda.manual_seed_all(self.seed)
    self._model = _MemStreamAutoEncoder(input_dim=input_dim).to(self._device)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    self._memory_latent = torch.zeros(self.memory_size, latent_dim, dtype=torch.float32, device=self._device)
    self._memory_input = torch.zeros(self.memory_size, input_dim, dtype=torch.float32, device=self._device)
    self._norm_mean = torch.zeros(input_dim, dtype=torch.float32, device=self._device)
    self._norm_std = torch.ones(input_dim, dtype=torch.float32, device=self._device)
    self._exp = torch.tensor([self.gamma ** i for i in range(self.k)], dtype=torch.float32, device=self._device)
    logging.info(
      "MemStream input_dim=%d latent_dim=%d (paper: 2×input)",
      input_dim,
      latent_dim,
    )

  def _init_from_features(self, features: Dict[str, float]) -> None:
    transformed = self._frontend_transform(features)
    self._init_model_for_feature_names(sorted(transformed.keys()))

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("MemStream feature names not initialized")
    transformed = self._frontend_transform(features)
    vec = np.array([float(transformed[k]) for k in self._feature_names], dtype=np.float32)
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

  def _memory_distance(self, z: torch.Tensor) -> float:
    """K-NN discounted L1 distance (paper). (topk * gamma^i).sum() / exp.sum(). Raw output, no scaling."""
    if self._memory_latent is None or self._mem_filled == 0 or self._exp is None:
      return 0.0
    memory = self._memory_latent[: self._mem_filled]
    dists = torch.norm(memory - z.unsqueeze(0), p=1, dim=1)
    k_eff = max(1, min(self.k, int(self._mem_filled)))
    topk_vals = torch.topk(dists, k=k_eff, largest=False).values
    exp = self._exp[:k_eff]
    return float((topk_vals * exp).sum().item() / exp.sum().item())

  def _should_update_memory(self, score_raw: float) -> bool:
    """Paper: update memory when anomaly score (raw K-NN distance) ≤ β."""
    if self._mem_filled < self._warmup_accept:
      return True
    return score_raw <= self.beta

  def _write_memory(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, Any]:
    if self._memory_latent is None or self._memory_input is None:
      return {
        "memory_slot": None,
        "overwrite": False,
        "mem_filled_after": self._mem_filled,
        "mem_index_after": self._mem_index,
      }
    overwrite = self._mem_filled >= self.memory_size
    if self._mem_filled < self.memory_size:
      idx = self._mem_filled
      self._mem_filled += 1
    else:
      idx = self._mem_index
      self._mem_index = (self._mem_index + 1) % self.memory_size
    self._memory_latent[idx] = z.detach()
    self._memory_input[idx] = x.detach()
    self._refresh_norm_from_memory()
    return {
      "memory_slot": int(idx),
      "overwrite": overwrite,
      "mem_filled_after": int(self._mem_filled),
      "mem_index_after": int(self._mem_index),
    }

  def get_last_debug(self) -> Dict[str, Any]:
    return dict(self._last_debug)

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    """Return the underlying (unsquashed) score. Paper: K-NN discounted L1 to memory."""
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")

    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.eval()
    with torch.no_grad():
      z_eval, _ = self._model(x_norm)
      z_eval = z_eval.squeeze(0)
      score_raw = self._memory_distance(z_eval)
    score_raw = max(0.0, float(score_raw))
    self._last_debug = {
      "mode": "score_only",
      "score_raw": score_raw,
      "beta": float(self.beta),
      "update_allowed": None,
      "update_reason": "score_only",
      "memory_error": score_raw,
      "recon_error": 0.0,
      "mem_filled_before": int(self._mem_filled),
      "mem_filled_after": int(self._mem_filled),
      "memory_fill_fraction": float(self._mem_filled / max(1, self.memory_size)),
      "memory_slot": None,
      "memory_overwrite": False,
      "memory_size": int(self.memory_size),
      "warmup_accept": int(self._warmup_accept),
      "input_mode": self.input_mode,
    }
    return score_raw

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    """Return the underlying (unsquashed) score, learning from this instance."""
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")

    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.train()

    with torch.no_grad():
      z_eval, _ = self._model(x_norm)
      z_eval = z_eval.squeeze(0)
      score_raw = self._memory_distance(z_eval)

    mem_filled_before = int(self._mem_filled)
    mem_index_before = int(self._mem_index)
    update_allowed = self._should_update_memory(score_raw)
    update_reason = "warmup" if mem_filled_before < self._warmup_accept else (
      "score_below_beta" if update_allowed else "score_above_beta"
    )
    train_loss: Optional[float] = None
    write_info: Dict[str, Any] = {
      "memory_slot": None,
      "overwrite": False,
      "mem_filled_after": mem_filled_before,
      "mem_index_after": mem_index_before,
    }

    recon_error: float
    if update_allowed:
      noisy = x_norm + (self._noise_std * torch.randn_like(x_norm))
      self._optimizer.zero_grad()
      z_train, recon_train = self._model(noisy)
      loss = torch.mean((x_norm - recon_train) ** 2)
      loss.backward()
      self._optimizer.step()
      train_loss = float(loss.item())
      recon_error = train_loss
      write_info = self._write_memory(x, z_train.squeeze(0))
      self._accepted_updates += 1
      if bool(write_info["overwrite"]):
        self._overwrite_updates += 1
    else:
      self._rejected_updates += 1
      self._model.eval()
      with torch.no_grad():
        _, recon_eval = self._model(x_norm)
        recon_error = float(torch.mean((x_norm - recon_eval) ** 2).item())

    score_raw = max(0.0, float(score_raw))
    self._last_debug = {
      "mode": "score_and_learn",
      "score_raw": score_raw,
      "beta": float(self.beta),
      "update_allowed": bool(update_allowed),
      "update_reason": update_reason,
      "train_loss": train_loss,
      "mem_filled_before": mem_filled_before,
      "mem_filled_after": int(write_info["mem_filled_after"]),
      "mem_index_before": mem_index_before,
      "mem_index_after": int(write_info["mem_index_after"]),
      "memory_fill_fraction": float(int(write_info["mem_filled_after"]) / max(1, self.memory_size)),
      "memory_slot": write_info["memory_slot"],
      "memory_overwrite": bool(write_info["overwrite"]),
      "memory_size": int(self.memory_size),
      "warmup_accept": int(self._warmup_accept),
      "input_mode": self.input_mode,
      "accepted_updates_total": int(self._accepted_updates),
      "rejected_updates_total": int(self._rejected_updates),
      "overwrite_updates_total": int(self._overwrite_updates),
      "memory_error": score_raw,
      "recon_error": recon_error,
    }
    return score_raw

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
      "input_mode": self.input_mode,
      "frontend_marginals_state": self._frontend_marginals.get_state() if self._frontend_marginals is not None and self.input_mode != "raw" else None,
      "mem_index": self._mem_index,
      "mem_filled": self._mem_filled,
      "accepted_updates": self._accepted_updates,
      "rejected_updates": self._rejected_updates,
      "overwrite_updates": self._overwrite_updates,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self.input_mode = str(state.get("input_mode", self.input_mode))
    frontend_state = state.get("frontend_marginals_state")
    if frontend_state is not None:
      if self._frontend_marginals is None:
        raise RuntimeError("MemStream frontend state present but frontend is disabled")
      self._frontend_marginals.set_state(frontend_state)
    self._feature_names = list(state["feature_names"])
    self._init_model_for_feature_names(self._feature_names)
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
    self._accepted_updates = int(state.get("accepted_updates", 0))
    self._rejected_updates = int(state.get("rejected_updates", 0))
    self._overwrite_updates = int(state.get("overwrite_updates", 0))


class OnlineZScore(_BothScoresMixin):
  """Simple online per-feature Z-score baseline. (Self-implementation)"""

  def __init__(self, min_count: int, std_floor: float, topk: int, model_device: str, seed: int):
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
    self.topk = int(max(1, topk))
    self._feature_names: Optional[List[str]] = None
    self._count = 0
    self._mean: Optional[np.ndarray] = None
    self._m2: Optional[np.ndarray] = None
    logging.info(
      "Initialized %s (min_count=%d, std_floor=%.6f, topk=%d)",
      self.algorithm,
      self.min_count,
      self.std_floor,
      self.topk,
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
    k = min(len(z_abs), self.topk)
    if k <= 0:
      return 0.0
    topk = np.partition(z_abs, -k)[-k:]
    return float(np.mean(topk))

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

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    x = self._vectorize(features)
    return max(0.0, self._raw_from_vector(x))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
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
      "topk": self.topk,
      "count": self._count,
      "mean": self._mean,
      "m2": self._m2,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint state."""
    self._feature_names = list(state["feature_names"])
    self.min_count = int(state.get("min_count", self.min_count))
    self.std_floor = float(state.get("std_floor", self.std_floor))
    self.topk = int(state.get("topk", self.topk))
    self._count = int(state["count"])
    self._mean = np.asarray(state["mean"], dtype=np.float64)
    self._m2 = np.asarray(state["m2"], dtype=np.float64)


class OnlineKNN(_BothScoresMixin):
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

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    x = self._vectorize(features)
    return self._raw_from_vector(x)

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
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


class OnlineFreq1D(_BothScoresMixin):
  """
  Simple frequency / rarity baseline using independent 1D marginals. (Self-implementation)

  - Numeric features: fixed-bin histogram over [0, 1] (clamped).
  - Categorical features: capped per-feature count tables for *_hash and binary flags.

  Score (raw) = configurable aggregation over features of excess surprisal:
    max(0, -log p(x) - (-log p_uniform))
  so cold-start produces score ~0 (uniform baseline).
  """

  _BINARY_FEATURES = frozenset({
    "return_success",
    "file_sensitive_path",
    "file_tmp_path",
    "proc_sensitive_path",
    "proc_tmp_path",
  })
  _HASH_KEY_SPACE = 10000  # hash mode keys are 0..9999

  def __init__(
    self,
    bins: int,
    alpha: float,
    decay: float,
    max_categories: int,
    aggregation: str,
    topk: int,
    soft_topk_temperature: float,
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
    self.aggregation = str(aggregation).strip().lower()
    if self.aggregation not in ("sum", "mean", "topk_mean", "soft_topk_mean"):
      raise ValueError("freq1d aggregation must be one of: sum, mean, topk_mean, soft_topk_mean")
    self.topk = int(topk)
    if self.topk <= 0:
      raise ValueError("freq1d topk must be > 0")
    self.soft_topk_temperature = float(soft_topk_temperature)
    if not (self.soft_topk_temperature > 0.0):
      raise ValueError("freq1d soft_topk_temperature must be > 0")

    self._feature_names: Optional[List[str]] = None
    self._kind: Optional[List[str]] = None  # "num" or "cat"
    self._cat_mode: Optional[List[Optional[str]]] = None  # "hash" | "binary" | None

    self._num_slot: Optional[List[int]] = None
    self._num_counts: List[np.ndarray] = []
    self._num_fenwick: List[FenwickTree] = []
    self._num_scale: List[float] = []

    self._cat_slot: Optional[List[int]] = None
    self._cat_counts: List[Any] = []  # dict for binary/hash-small, np.ndarray for hash-large
    self._cat_hash_fenwick: List[Optional[FenwickTree]] = []  # only for hash with max_categories >= 10000
    self._cat_other: List[float] = []
    self._cat_scale: List[float] = []

    logging.info(
      "Initialized %s (bins=%d, alpha=%.3f, decay=%.4f, max_categories=%d, aggregation=%s, topk=%d, soft_temp=%.4f)",
      self.algorithm,
      self.bins,
      self.alpha,
      self.decay,
      self.max_categories,
      self.aggregation,
      self.topk,
      self.soft_topk_temperature,
    )

  def _aggregate(self, scores: List[float]) -> float:
    if not scores:
      return 0.0
    if self.aggregation == "sum":
      return float(sum(scores))
    if self.aggregation == "mean":
      return float(sum(scores) / float(len(scores)))
    if self.aggregation == "topk_mean":
      k = min(len(scores), self.topk)
      if k <= 0:
        return 0.0
      arr = np.asarray(scores, dtype=np.float64)
      # partial select top-k without full sort
      topk = np.partition(arr, -k)[-k:]
      return float(np.mean(topk))
    # soft_topk_mean
    arr = np.asarray(scores, dtype=np.float64)
    k = min(len(arr), self.topk)
    if k <= 0:
      return 0.0
    # Focus on top-k for stability/perf, then softmax within that set.
    topk = np.partition(arr, -k)[-k:]
    temp = float(self.soft_topk_temperature)
    m = float(np.max(topk))
    w = np.exp((topk - m) / temp)
    denom = float(np.sum(w))
    if denom <= 0.0 or not np.isfinite(denom):
      return float(np.mean(topk))
    return float(np.sum(w * topk) / denom)

  _BINARY_PREFIXES = (
    "group_syscall_",
  )

  def _is_categorical(self, name: str) -> tuple[bool, Optional[str]]:
    if name.endswith("_hash"):
      return True, "hash"
    if name in self._BINARY_FEATURES or name.startswith(self._BINARY_PREFIXES):
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
        if mode == "hash" and self.max_categories >= self._HASH_KEY_SPACE:
          arr = np.zeros(self._HASH_KEY_SPACE, dtype=np.float64)
          self._cat_counts.append(arr)
          self._cat_hash_fenwick.append(FenwickTree(self._HASH_KEY_SPACE))
        else:
          self._cat_counts.append({})
          self._cat_hash_fenwick.append(None)
        self._cat_other.append(0.0)
        self._cat_scale.append(1.0)
      else:
        kinds.append("num")
        cat_modes.append(None)
        num_slot[i] = len(self._num_counts)
        self._num_counts.append(np.zeros(self.bins, dtype=np.float64))
        self._num_fenwick.append(FenwickTree(self.bins))
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
    self._num_fenwick[slot].init(list(self._num_counts[slot]))
    self._num_scale[slot] = 1.0

  def _maybe_rescale_categorical(self, slot: int) -> None:
    scale = self._cat_scale[slot]
    if scale >= 1e-6:
      return
    counts = self._cat_counts[slot]
    if isinstance(counts, np.ndarray):
      counts *= scale
      self._cat_hash_fenwick[slot].init(list(counts))
    elif counts:
      for k in list(counts.keys()):
        counts[k] *= scale
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
    fenwick = self._num_fenwick[slot]
    total = _fenwick_prefix_sum(fenwick, self.bins - 1)
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
    delta = 1.0 / max(scale, 1e-12)
    self._num_counts[slot][idx] += delta
    self._num_fenwick[slot].add(idx, delta)

  def _score_cat_slot(self, slot: int, mode: str, x: float) -> float:
    counts = self._cat_counts[slot]
    k = self._cat_key(mode, x)
    if isinstance(counts, np.ndarray):
      fenwick = self._cat_hash_fenwick[slot]
      total = _fenwick_prefix_sum(fenwick, self._HASH_KEY_SPACE - 1)
      c = float(counts[k])
      alphabet = max(2, int(np.count_nonzero(counts)) + 1)
    else:
      other = float(self._cat_other[slot])
      total = other + float(sum(counts.values()))
      c = float(counts.get(k, 0.0))
      alphabet = max(2, len(counts) + 1)
    p = (c + self.alpha) / (total + self.alpha * alphabet) if total >= 0.0 else 1.0 / float(alphabet)
    p = max(1e-300, float(p))
    baseline = math.log(float(alphabet))
    return max(0.0, -math.log(p) - baseline)

  def _cdf_numeric_slot(self, slot: int, x: float) -> float:
    """CDF F(x) = P(X <= x) from histogram, clamped to (0, 1)."""
    fenwick = self._num_fenwick[slot]
    total = _fenwick_prefix_sum(fenwick, self.bins - 1)
    idx = self._bin_numeric(x)
    cum = _fenwick_prefix_sum(fenwick, idx)
    cdf = (cum + self.alpha) / (total + self.alpha * self.bins) if total >= 0.0 else (idx + 1) / float(self.bins)
    return float(np.clip(cdf, 1e-12, 1.0 - 1e-12))

  def _cdf_cat_slot(self, slot: int, mode: str, x: float) -> float:
    """CDF F(x) = P(X <= k) from count table for keys <= bin(x), clamped to (0, 1)."""
    counts = self._cat_counts[slot]
    k = self._cat_key(mode, x)
    if isinstance(counts, np.ndarray):
      fenwick = self._cat_hash_fenwick[slot]
      total = _fenwick_prefix_sum(fenwick, self._HASH_KEY_SPACE - 1)
      cum = _fenwick_prefix_sum(fenwick, k)
      alphabet = max(2, int(np.count_nonzero(counts)) + 1)
    else:
      other = float(self._cat_other[slot])
      total = other + float(sum(counts.values()))
      cum = float(sum(c for key, c in counts.items() if key <= k))
      alphabet = max(2, len(counts) + 1)
    cdf = (cum + self.alpha) / (total + self.alpha * alphabet) if total >= 0.0 else 0.5
    return float(np.clip(cdf, 1e-12, 1.0 - 1e-12))

  def get_cdf_vector(self, features: Dict[str, float]) -> np.ndarray:
    """Return u_i = F_i(x_i) for each feature in _feature_names order."""
    self._ensure_init(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    out = np.zeros(len(self._feature_names), dtype=np.float64)
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        if slot >= 0:
          out[i] = self._cdf_numeric_slot(slot, x)
        else:
          out[i] = 0.5
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        if slot >= 0:
          out[i] = self._cdf_cat_slot(slot, mode, x)
        else:
          out[i] = 0.5
    return out

  def get_excess_vector(self, features: Dict[str, float]) -> np.ndarray:
    """Return per-feature excess surprisal in _feature_names order."""
    self._ensure_init(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    out = np.zeros(len(self._feature_names), dtype=np.float64)
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        if slot >= 0:
          out[i] = self._score_numeric_slot(slot, x)
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        if slot >= 0:
          out[i] = self._score_cat_slot(slot, mode, x)
    return out

  def _learn_cat_slot(self, slot: int, mode: str, x: float) -> None:
    self._cat_scale[slot] *= self.decay
    self._maybe_rescale_categorical(slot)
    scale = self._cat_scale[slot]
    k = self._cat_key(mode, x)
    delta = 1.0 / max(scale, 1e-12)
    counts = self._cat_counts[slot]
    if isinstance(counts, np.ndarray):
      counts[k] += delta
      self._cat_hash_fenwick[slot].add(k, delta)
    else:
      if k in counts:
        counts[k] = float(counts[k]) + delta
        return
      if len(counts) < self.max_categories:
        counts[k] = delta
        return
      self._cat_other[slot] = float(self._cat_other[slot]) + delta

  def _ensure_init(self, features: Dict[str, float]) -> None:
    if self._feature_names is None:
      self._init_from_features(features)

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    self._ensure_init(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    scores: List[float] = []
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        if slot >= 0:
          scores.append(self._score_numeric_slot(slot, x))
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        if slot >= 0:
          scores.append(self._score_cat_slot(slot, mode, x))
    return max(0.0, float(self._aggregate(scores)))

  def learn_only(self, features: Dict[str, float]) -> None:
    """Update marginals from features without computing score. Use when caller only needs CDF/learn."""
    self._ensure_init(features)
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

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    self._ensure_init(features)
    raw = self.score_only_raw(features)
    self.learn_only(features)
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
      "aggregation": self.aggregation,
      "topk": self.topk,
      "soft_topk_temperature": self.soft_topk_temperature,
      "kind": list(self._kind),
      "cat_mode": list(self._cat_mode),
      "num_counts": [c for c in self._num_counts],
      "num_scale": list(self._num_scale),
      "cat_counts": [
        d.tolist() if isinstance(d, np.ndarray) else dict(d) for d in self._cat_counts
      ],
      "cat_other": list(self._cat_other),
      "cat_scale": list(self._cat_scale),
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self.bins = int(state.get("bins", self.bins))
    self.alpha = float(state.get("alpha", self.alpha))
    self.decay = float(state.get("decay", self.decay))
    self.max_categories = int(state.get("max_categories", self.max_categories))
    self.aggregation = str(state.get("aggregation", self.aggregation)).strip().lower()
    self.topk = int(state.get("topk", self.topk))
    self.soft_topk_temperature = float(state.get("soft_topk_temperature", self.soft_topk_temperature))
    self._feature_names = list(state["feature_names"])
    self._kind = list(state["kind"])
    self._cat_mode = list(state["cat_mode"])

    # Rebuild slot mappings and storage
    self._num_counts = [np.asarray(c, dtype=np.float64) for c in state.get("num_counts", [])]
    self._num_fenwick = [FenwickTree(len(c)) for c in self._num_counts]
    for i, c in enumerate(self._num_counts):
      self._num_fenwick[i].init(list(c))
    self._num_scale = [float(s) for s in state.get("num_scale", [])]

    raw_cat = state.get("cat_counts", [])
    self._cat_counts = []
    self._cat_hash_fenwick = []
    for i, d in enumerate(raw_cat):
      if isinstance(d, list) and len(d) == self._HASH_KEY_SPACE:
        arr = np.asarray(d, dtype=np.float64)
        self._cat_counts.append(arr)
        fw = FenwickTree(self._HASH_KEY_SPACE)
        fw.init(list(arr))
        self._cat_hash_fenwick.append(fw)
      else:
        self._cat_counts.append(dict(d))
        self._cat_hash_fenwick.append(None)
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


class OnlineCopulaTree(_BothScoresMixin):
  """
  Streaming copula-tree detector on top of Freq1D marginals.

  Inspired by:
  - Gábor Horváth, Edith Kovács, Roland Molontay, Szabolcs Nováczki,
    "Copula-based anomaly scoring and localization for large-scale, high-dimensional continuous data"
    (arXiv:1912.02166) https://arxiv.org/abs/1912.02166

  The implementation follows the paper's sparse-tree idea, but keeps the repo's
  online contract by:
  - learning marginals with OnlineFreq1D
  - tracking pairwise dependence with EMA outer products in Gaussianized space
  - rebuilding a maximum-spanning tree periodically
  - scoring with Gaussian pair-copula edge surprisal aggregated over the tree
  """

  def __init__(
    self,
    bins: int,
    alpha: float,
    decay: float,
    max_categories: int,
    u_clamp: float,
    reg: float,
    max_features: int,
    importance_window: int,
    tree_update_interval: int,
    edge_score_aggregation: str,
    edge_score_topk: int,
    model_device: str,
    seed: int,
  ):
    del seed
    self.algorithm = "copulatree"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but copulatree is CPU-only; using CPU",
        self.algorithm,
      )
    self.u_clamp = float(u_clamp)
    if not (0.0 < self.u_clamp < 0.5):
      raise ValueError("copulatree u_clamp must be in (0, 0.5)")
    self.reg = float(reg)
    if not (self.reg > 0.0):
      raise ValueError("copulatree reg must be > 0")
    self._max_features = int(max_features)
    if self._max_features < 0:
      raise ValueError("copulatree max_features must be >= 0")
    self._importance_window = int(importance_window)
    if self._importance_window <= 0:
      raise ValueError("copulatree importance_window must be > 0")
    self._tree_update_interval = int(tree_update_interval)
    if self._tree_update_interval <= 0:
      raise ValueError("copulatree tree_update_interval must be > 0")
    self.edge_score_aggregation = str(edge_score_aggregation).strip().lower()
    if self.edge_score_aggregation not in ("sum", "mean", "topk_mean"):
      raise ValueError("copulatree edge_score_aggregation must be one of: sum, mean, topk_mean")
    self.edge_score_topk = int(edge_score_topk)
    if self.edge_score_topk <= 0:
      raise ValueError("copulatree edge_score_topk must be > 0")
    self._importance_ema_alpha = 1.0 - 1.0 / max(self._importance_window, 1)
    self._pair_ema_alpha = 1.0 - 1.0 / max(self._tree_update_interval, 1)

    self._marginals = OnlineFreq1D(
      bins=bins,
      alpha=alpha,
      decay=decay,
      max_categories=max_categories,
      aggregation="mean",
      topk=1,
      soft_topk_temperature=1.0,
      model_device="cpu",
      seed=0,
    )
    self._n = 0
    self._pair_outer_ema: Optional[np.ndarray] = None
    self._importance: Optional[np.ndarray] = None
    self._selected_indices: Optional[np.ndarray] = None
    self._events_since_selection = 0
    self._tree_edges: List[Tuple[int, int]] = []
    self._events_since_tree = 0
    logging.info(
      "Initialized %s (u_clamp=%.2e, reg=%.4f, max_features=%d, importance_window=%d, tree_update_interval=%d, aggregation=%s, topk=%d)",
      self.algorithm,
      self.u_clamp,
      self.reg,
      self._max_features,
      self._importance_window,
      self._tree_update_interval,
      self.edge_score_aggregation,
      self.edge_score_topk,
    )

  def _to_z(self, features: Dict[str, float]) -> np.ndarray:
    u = self._marginals.get_cdf_vector(features)
    u = np.clip(u, self.u_clamp, 1.0 - self.u_clamp)
    return scipy_norm.ppf(u).astype(np.float64)

  def _aggregate_edge_scores(self, scores: List[float]) -> float:
    if not scores:
      return 0.0
    if self.edge_score_aggregation == "sum":
      return float(sum(scores))
    if self.edge_score_aggregation == "mean":
      return float(sum(scores) / float(len(scores)))
    k = min(len(scores), self.edge_score_topk)
    if k <= 0:
      return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    topk = np.partition(arr, -k)[-k:]
    return float(np.mean(topk))

  def _ensure_pair_dim(self, d: int) -> None:
    if self._pair_outer_ema is not None and self._pair_outer_ema.shape[0] == d:
      return
    self._pair_outer_ema = np.zeros((d, d), dtype=np.float64)
    self._tree_edges = []
    self._events_since_tree = 0

  def _reset_dependence_state(self) -> None:
    self._pair_outer_ema = None
    self._tree_edges = []
    self._events_since_tree = 0
    self._n = 0

  def _corr_matrix(self) -> Optional[np.ndarray]:
    if self._pair_outer_ema is None:
      return None
    diag = np.maximum(np.diag(self._pair_outer_ema), self.reg)
    scale = np.sqrt(np.outer(diag, diag))
    corr = self._pair_outer_ema / np.maximum(scale, 1e-12)
    corr = np.clip(corr, -0.995, 0.995)
    np.fill_diagonal(corr, 1.0)
    return corr

  def _max_spanning_tree(self, weights: np.ndarray) -> List[Tuple[int, int]]:
    d = int(weights.shape[0])
    if d <= 1:
      return []
    in_tree = np.zeros(d, dtype=bool)
    parent = np.zeros(d, dtype=np.int64)
    best = np.full(d, -np.inf, dtype=np.float64)
    in_tree[0] = True
    best[:] = weights[0]
    parent[:] = 0
    edges: List[Tuple[int, int]] = []
    for _ in range(d - 1):
      remaining = np.where(~in_tree)[0]
      if remaining.size == 0:
        break
      idx = int(remaining[np.argmax(best[remaining])])
      if not np.isfinite(best[idx]):
        idx = int(remaining[0])
      edges.append((int(parent[idx]), idx))
      in_tree[idx] = True
      for j in np.where(~in_tree)[0]:
        w = float(weights[idx, j])
        if w > best[j]:
          best[j] = w
          parent[j] = idx
    return edges

  def _refresh_tree(self) -> None:
    corr = self._corr_matrix()
    if corr is None:
      self._tree_edges = []
      return
    weights = np.abs(corr)
    np.fill_diagonal(weights, -np.inf)
    self._tree_edges = self._max_spanning_tree(weights)

  def _pair_score(self, zi: float, zj: float, rho: float) -> float:
    rho = float(np.clip(rho, -0.995, 0.995))
    denom = max(1e-6, 1.0 - rho * rho)
    quad = ((rho * rho) * (zi * zi + zj * zj) - 2.0 * rho * zi * zj) / denom
    neg_log_c = 0.5 * math.log(denom) + 0.5 * quad
    return max(0.0, float(neg_log_c))

  def _score_selected_z(self, z: np.ndarray) -> float:
    if self._n < 2 or len(z) <= 1 or not self._tree_edges:
      return 0.0
    corr = self._corr_matrix()
    if corr is None:
      return 0.0
    scores: List[float] = []
    for i, j in self._tree_edges:
      if i >= len(z) or j >= len(z):
        continue
      scores.append(self._pair_score(float(z[i]), float(z[j]), float(corr[i, j])))
    return max(0.0, float(self._aggregate_edge_scores(scores)))

  def _maybe_select_and_get_z(self, z: np.ndarray, do_learn: bool) -> np.ndarray:
    d = len(z)
    past_warmup = self._selected_indices is not None or self._n >= self._importance_window
    use_selection = (
      self._max_features > 0
      and self._max_features < d
      and past_warmup
    )

    if self._importance is None or len(self._importance) != d:
      self._importance = np.zeros(d, dtype=np.float64)

    if do_learn:
      self._importance = (
        self._importance_ema_alpha * self._importance
        + (1.0 - self._importance_ema_alpha) * np.abs(z)
      )
      self._events_since_selection += 1

    if use_selection:
      if self._events_since_selection >= self._importance_window and do_learn:
        self._events_since_selection = 0
        new_indices = np.argsort(-self._importance)[: self._max_features]
        if self._selected_indices is None or not np.array_equal(self._selected_indices, new_indices):
          self._selected_indices = new_indices
          self._reset_dependence_state()
      if self._selected_indices is not None:
        return z[self._selected_indices]
    else:
      if self._selected_indices is not None:
        self._selected_indices = None
        self._reset_dependence_state()

    return z

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    z = self._to_z(features)
    z_tree = self._maybe_select_and_get_z(z, do_learn=False)
    return self._score_selected_z(z_tree)

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    z = self._to_z(features)
    z_tree = self._maybe_select_and_get_z(z, do_learn=True)
    self._ensure_pair_dim(len(z_tree))

    raw = self._score_selected_z(z_tree)

    self._marginals.learn_only(features)
    outer = np.outer(z_tree, z_tree)
    if self._pair_outer_ema is None:
      self._pair_outer_ema = outer.astype(np.float64)
    else:
      self._pair_outer_ema = (
        self._pair_ema_alpha * self._pair_outer_ema
        + (1.0 - self._pair_ema_alpha) * outer
      )
    self._n += 1
    self._events_since_tree += 1
    if not self._tree_edges or self._events_since_tree >= self._tree_update_interval:
      self._refresh_tree()
      self._events_since_tree = 0
    return max(0.0, float(raw))

  def get_state(self) -> Dict[str, Any]:
    return {
      "marginals_state": self._marginals.get_state(),
      "_n": self._n,
      "u_clamp": self.u_clamp,
      "reg": self.reg,
      "max_features": self._max_features,
      "importance_window": self._importance_window,
      "tree_update_interval": self._tree_update_interval,
      "edge_score_aggregation": self.edge_score_aggregation,
      "edge_score_topk": self.edge_score_topk,
      "_pair_outer_ema": self._pair_outer_ema.copy() if self._pair_outer_ema is not None else None,
      "_importance": self._importance.copy() if self._importance is not None else None,
      "_selected_indices": self._selected_indices.copy() if self._selected_indices is not None else None,
      "_events_since_selection": self._events_since_selection,
      "_tree_edges": [tuple(map(int, edge)) for edge in self._tree_edges],
      "_events_since_tree": self._events_since_tree,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self._marginals.set_state(state["marginals_state"])
    self._n = int(state.get("_n", 0))
    self.u_clamp = float(state.get("u_clamp", self.u_clamp))
    self.reg = float(state.get("reg", self.reg))
    self._max_features = int(state.get("max_features", self._max_features))
    self._importance_window = int(state.get("importance_window", self._importance_window))
    self._tree_update_interval = int(state.get("tree_update_interval", self._tree_update_interval))
    self.edge_score_aggregation = str(state.get("edge_score_aggregation", self.edge_score_aggregation))
    self.edge_score_topk = int(state.get("edge_score_topk", self.edge_score_topk))
    self._importance_ema_alpha = 1.0 - 1.0 / max(self._importance_window, 1)
    self._pair_ema_alpha = 1.0 - 1.0 / max(self._tree_update_interval, 1)
    pair_outer = state.get("_pair_outer_ema")
    self._pair_outer_ema = np.asarray(pair_outer, dtype=np.float64) if pair_outer is not None else None
    importance = state.get("_importance")
    self._importance = np.asarray(importance, dtype=np.float64) if importance is not None else None
    selected = state.get("_selected_indices")
    self._selected_indices = np.asarray(selected, dtype=np.int64) if selected is not None else None
    self._events_since_selection = int(state.get("_events_since_selection", 0))
    self._tree_edges = [tuple(map(int, edge)) for edge in state.get("_tree_edges", [])]
    self._events_since_tree = int(state.get("_events_since_tree", 0))


class OnlineLatentCluster(_BothScoresMixin):
  """
  Online latent-clustering detector on top of Freq1D-normalized marginals.

  Pipeline:
  - Reuse OnlineFreq1D to learn per-feature marginals online.
  - Convert event features to CDF coordinates u_i and Gaussianized z_i.
  - Maintain a small bank of latent cluster centers with diagonal variances.
  - Score by the best standardized distance to an existing cluster.
  - Update clusters only for likely-normal points; spawn a new cluster when an
    event is consistently far from all existing clusters and capacity remains.
  """

  def __init__(
    self,
    bins: int,
    alpha: float,
    decay: float,
    max_categories: int,
    max_clusters: int,
    u_clamp: float,
    reg: float,
    update_alpha: float,
    spawn_threshold: float,
    model_device: str,
    seed: int,
  ):
    del seed
    self.algorithm = "latentcluster"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning(
        "algorithm=%s requested on CUDA, but latentcluster is CPU-only; using CPU",
        self.algorithm,
      )
    self.max_clusters = int(max(1, max_clusters))
    self.u_clamp = float(u_clamp)
    if not (0.0 < self.u_clamp < 0.5):
      raise ValueError("latentcluster u_clamp must be in (0, 0.5)")
    self.reg = float(reg)
    if not (self.reg > 0.0):
      raise ValueError("latentcluster reg must be > 0")
    self.update_alpha = float(update_alpha)
    if not (0.0 < self.update_alpha <= 1.0):
      raise ValueError("latentcluster update_alpha must be in (0, 1]")
    self.spawn_threshold = float(spawn_threshold)
    if not (self.spawn_threshold > 0.0):
      raise ValueError("latentcluster spawn_threshold must be > 0")

    self._marginals = OnlineFreq1D(
      bins=bins,
      alpha=alpha,
      decay=decay,
      max_categories=max_categories,
      aggregation="mean",
      topk=1,
      soft_topk_temperature=1.0,
      model_device="cpu",
      seed=0,
    )
    self._centers: Optional[np.ndarray] = None
    self._vars: Optional[np.ndarray] = None
    self._weights: Optional[np.ndarray] = None
    self._active_clusters = 0
    self._score_ema: Optional[float] = None
    self._score_var_ema: float = 0.0
    self._ema_alpha = 0.05
    self._beta_floor = 1.5
    self._beta_sigma = 2.5
    self._warmup_accept = max(16, 4 * self.max_clusters)
    self._accepted_updates = 0
    logging.info(
      "Initialized %s (max_clusters=%d, u_clamp=%.2e, reg=%.4f, update_alpha=%.3f, spawn_threshold=%.3f)",
      self.algorithm,
      self.max_clusters,
      self.u_clamp,
      self.reg,
      self.update_alpha,
      self.spawn_threshold,
    )

  def _to_z(self, features: Dict[str, float]) -> np.ndarray:
    u = self._marginals.get_cdf_vector(features)
    u = np.clip(u, self.u_clamp, 1.0 - self.u_clamp)
    return scipy_norm.ppf(u).astype(np.float64)

  def _ensure_cluster_arrays(self, d: int) -> None:
    if self._centers is not None and self._centers.shape[1] == d:
      return
    self._centers = np.zeros((self.max_clusters, d), dtype=np.float64)
    self._vars = np.ones((self.max_clusters, d), dtype=np.float64)
    self._weights = np.zeros(self.max_clusters, dtype=np.float64)
    self._active_clusters = 0

  def _cluster_scores(self, z: np.ndarray) -> np.ndarray:
    if self._centers is None or self._vars is None or self._active_clusters <= 0:
      return np.zeros(0, dtype=np.float64)
    centers = self._centers[: self._active_clusters]
    vars_ = np.maximum(self._vars[: self._active_clusters], self.reg)
    diff = centers - z[np.newaxis, :]
    return np.sum((diff * diff) / vars_, axis=1)

  def _best_cluster(self, z: np.ndarray) -> tuple[Optional[int], float]:
    scores = self._cluster_scores(z)
    if scores.size == 0:
      return None, 0.0
    idx = int(np.argmin(scores))
    return idx, float(scores[idx])

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

  def _should_update(self, score_raw: float) -> bool:
    if self._accepted_updates < self._warmup_accept:
      return True
    return score_raw <= self._adaptive_beta()

  def _spawn_cluster(self, z: np.ndarray) -> None:
    if self._centers is None or self._vars is None or self._weights is None:
      self._ensure_cluster_arrays(len(z))
    if self._centers is None or self._vars is None or self._weights is None:
      raise RuntimeError("latentcluster arrays not initialized")
    if self._active_clusters < self.max_clusters:
      idx = self._active_clusters
      self._active_clusters += 1
    else:
      idx = int(np.argmin(self._weights))
    self._centers[idx] = z
    self._vars[idx] = np.ones_like(z, dtype=np.float64)
    self._weights[idx] = 1.0

  def _update_cluster(self, idx: int, z: np.ndarray) -> None:
    if self._centers is None or self._vars is None or self._weights is None:
      raise RuntimeError("latentcluster arrays not initialized")
    weight = float(self._weights[idx])
    eta = max(self.update_alpha, 1.0 / max(1.0, weight + 1.0))
    center_old = self._centers[idx].copy()
    center_new = (1.0 - eta) * center_old + eta * z
    resid = z - center_new
    var_new = (1.0 - eta) * self._vars[idx] + eta * (resid * resid)
    self._centers[idx] = center_new
    self._vars[idx] = np.maximum(self.reg, var_new)
    self._weights[idx] = weight + 1.0

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    z = self._to_z(features)
    self._ensure_cluster_arrays(len(z))
    _, best = self._best_cluster(z)
    return max(0.0, float(best))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    z = self._to_z(features)
    self._ensure_cluster_arrays(len(z))
    best_idx, best = self._best_cluster(z)
    raw = max(0.0, float(best))
    self._update_score_stats(raw)
    should_spawn = best_idx is None or (
      raw > self.spawn_threshold and self._active_clusters < self.max_clusters
    )

    if should_spawn:
      self._spawn_cluster(z)
      self._marginals.learn_only(features)
      self._accepted_updates += 1
      return raw

    if self._should_update(raw):
      self._update_cluster(best_idx, z)
      self._marginals.learn_only(features)
      self._accepted_updates += 1
    return raw

  def get_state(self) -> Dict[str, Any]:
    return {
      "marginals_state": self._marginals.get_state(),
      "max_clusters": self.max_clusters,
      "u_clamp": self.u_clamp,
      "reg": self.reg,
      "update_alpha": self.update_alpha,
      "spawn_threshold": self.spawn_threshold,
      "centers": self._centers.copy() if self._centers is not None else None,
      "vars": self._vars.copy() if self._vars is not None else None,
      "weights": self._weights.copy() if self._weights is not None else None,
      "active_clusters": self._active_clusters,
      "score_ema": self._score_ema,
      "score_var_ema": self._score_var_ema,
      "accepted_updates": self._accepted_updates,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self._marginals.set_state(state["marginals_state"])
    self.max_clusters = int(state.get("max_clusters", self.max_clusters))
    self.u_clamp = float(state.get("u_clamp", self.u_clamp))
    self.reg = float(state.get("reg", self.reg))
    self.update_alpha = float(state.get("update_alpha", self.update_alpha))
    self.spawn_threshold = float(state.get("spawn_threshold", self.spawn_threshold))
    centers = state.get("centers")
    vars_ = state.get("vars")
    weights = state.get("weights")
    self._centers = np.asarray(centers, dtype=np.float64) if centers is not None else None
    self._vars = np.asarray(vars_, dtype=np.float64) if vars_ is not None else None
    self._weights = np.asarray(weights, dtype=np.float64) if weights is not None else None
    self._active_clusters = int(state.get("active_clusters", 0))
    self._score_ema = state.get("score_ema")
    self._score_var_ema = float(state.get("score_var_ema", 0.0))
    self._accepted_updates = int(state.get("accepted_updates", 0))


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
    mem_memory_size: int = 512,
    mem_lr: float = 0.01,
    mem_beta: float = 0.1,
    mem_k: int = 3,
    mem_gamma: float = 0.0,
    mem_input_mode: str = "raw",
    mem_warmup_accept: int = 512,
    zscore_min_count: int = 20,
    zscore_std_floor: float = 1e-3,
    zscore_topk: int = 8,
    knn_k: int = 5,
    knn_memory_size: int = 1024,
    knn_metric: str = "euclidean",
    freq1d_bins: int = 64,
    freq1d_alpha: float = 1.0,
    freq1d_decay: float = 1.0,
    freq1d_max_categories: int = 2048,
    freq1d_aggregation: str = "mean",
    freq1d_topk: int = 8,
    freq1d_soft_topk_temperature: float = 0.25,
    copulatree_u_clamp: float = 1e-6,
    copulatree_reg: float = 0.05,
    copulatree_max_features: int = 30,
    copulatree_importance_window: int = 500,
    copulatree_tree_update_interval: int = 100,
    copulatree_edge_score_aggregation: str = "mean",
    copulatree_edge_score_topk: int = 8,
    latentcluster_max_clusters: int = 8,
    latentcluster_u_clamp: float = 1e-6,
    latentcluster_reg: float = 0.25,
    latentcluster_update_alpha: float = 0.05,
    latentcluster_spawn_threshold: float = 6.0,
    model_device: str = "auto",
    seed: int = 42,
    # Word2Vec embedding config (general feature extractor component)
    embedding_word2vec_dim: int = 5,
    embedding_word2vec_sentence_len: int = 7,
    embedding_word2vec_window: int = 5,
    embedding_word2vec_sg: int = 1,
    embedding_word2vec_update_every: int = 25,
    embedding_word2vec_epochs: int = 1,
    embedding_word2vec_post_warmup_lr_scale: float = 0.1,
    # Sequence MLP config (used only by algorithm=sequence_mlp)
    sequence_mlp_hidden_size: int = 150,
    sequence_mlp_hidden_layers: int = 4,
    sequence_mlp_lr: float = 0.003,
    warmup_events: int = 0,
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
      raise RuntimeError(
        "algorithm=loda (PySAD LODA) does not work: PySAD fit_partial overwrites histograms "
        "instead of accumulating, producing scores ~0. Use algorithm=loda_ema instead."
      )
    elif algo == "loda_ema":
      self.impl = OnlineLODAEMA(
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
        memory_size=mem_memory_size,
        lr=mem_lr,
        beta=mem_beta,
        k=mem_k,
        gamma=mem_gamma,
        input_mode=mem_input_mode,
        freq1d_bins=freq1d_bins,
        freq1d_alpha=freq1d_alpha,
        freq1d_decay=freq1d_decay,
        freq1d_max_categories=freq1d_max_categories,
        model_device=model_device,
        seed=seed,
        warmup_accept=mem_warmup_accept,
      )
    elif algo == "zscore":
      self.impl = OnlineZScore(
        min_count=zscore_min_count,
        std_floor=zscore_std_floor,
        topk=zscore_topk,
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
        aggregation=freq1d_aggregation,
        topk=freq1d_topk,
        soft_topk_temperature=freq1d_soft_topk_temperature,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "copulatree":
      self.impl = OnlineCopulaTree(
        bins=freq1d_bins,
        alpha=freq1d_alpha,
        decay=freq1d_decay,
        max_categories=freq1d_max_categories,
        u_clamp=copulatree_u_clamp,
        reg=copulatree_reg,
        max_features=copulatree_max_features,
        importance_window=copulatree_importance_window,
        tree_update_interval=copulatree_tree_update_interval,
        edge_score_aggregation=copulatree_edge_score_aggregation,
        edge_score_topk=copulatree_edge_score_topk,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "latentcluster":
      self.impl = OnlineLatentCluster(
        bins=freq1d_bins,
        alpha=freq1d_alpha,
        decay=freq1d_decay,
        max_categories=freq1d_max_categories,
        max_clusters=latentcluster_max_clusters,
        u_clamp=latentcluster_u_clamp,
        reg=latentcluster_reg,
        update_alpha=latentcluster_update_alpha,
        spawn_threshold=latentcluster_spawn_threshold,
        model_device=model_device,
        seed=seed,
      )
    elif algo == "sequence_mlp":
      self.impl = OnlineSequenceMLP(
        hidden_size=sequence_mlp_hidden_size,
        hidden_layers=sequence_mlp_hidden_layers,
        learning_rate=sequence_mlp_lr,
        model_device=model_device,
        seed=seed,
      )
    else:
      raise ValueError(
        "Unknown algorithm: %s. Choose from: halfspacetrees, loda_ema, kitnet, memstream, zscore, knn, "
        "freq1d, copulatree, latentcluster, sequence_mlp"
        % algorithm
      )
    self.algorithm = self.impl.algorithm

  def score_and_learn_event(
    self,
    evt: "events_pb2.EventEnvelope",
    *,
    feature_fn: Callable[["events_pb2.EventEnvelope"], Any],
  ) -> tuple[float, float]:
    """Unified server entrypoint: extract features, then pass the feature dict to the model."""
    res = feature_fn(evt)
    meta: Meta | None = None
    features: Dict[str, float]
    if (
      isinstance(res, tuple)
      and len(res) == 2
      and isinstance(res[0], dict)
      and (res[1] is None or isinstance(res[1], Meta))
    ):
      features = cast(Dict[str, float], res[0])
      meta = cast(Meta | None, res[1])
    elif isinstance(res, dict):
      features = cast(Dict[str, float], res)
    else:
      raise TypeError("feature_fn must return Dict[str,float] or (Dict[str,float], Meta|None)")
    return self.score_and_learn(features, meta=meta)

  def score_only(self, features: Dict[str, float], *, meta: Meta | None = None) -> tuple[float, float]:
    """Return (raw, scaled) without updating model state (read-only)."""
    raw = self.impl.score_only_raw(features, meta=meta)
    if self.algorithm == "halfspacetrees":
      scaled = float(raw)
    elif self.algorithm in ("copulatree", "latentcluster"):
      scaled = float(max(0.0, raw) / (1.0 + max(0.0, raw)))
    else:
      scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return (float(raw), scaled)

  def score_and_learn(
    self, features: Dict[str, float], *, meta: Meta | None = None
  ) -> tuple[float, float]:
    """Learn once and return (raw, scaled). Scaled is 0-1 for most models; raw==scaled for HalfSpaceTrees."""
    raw = self.impl.score_and_learn_raw(features, meta=meta)
    if self.algorithm == "halfspacetrees":
      scaled = float(raw)
    elif self.algorithm in ("copulatree", "latentcluster"):
      scaled = float(max(0.0, raw) / (1.0 + max(0.0, raw)))
    else:
      scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return (float(raw), scaled)

  def get_last_debug(self) -> Dict[str, Any]:
    getter = getattr(self.impl, "get_last_debug", None)
    if getter is None:
      return {}
    debug = getter()
    return debug if isinstance(debug, dict) else {}

  def save_checkpoint(self, path: Path, checkpoint_index: int, *, feature_state: dict | None = None) -> None:
    """Save detector state after learning events 0..checkpoint_index-1."""
    state = {
      "algorithm": self.algorithm,
      "checkpoint_index": checkpoint_index,
      "impl_state": self.impl.get_state(),
      "feature_state": feature_state,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
      pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Saved checkpoint at index %d to %s", checkpoint_index, path)

  def load_checkpoint(self, path: Path) -> tuple[int, dict | None]:
    """Load detector state. Returns (checkpoint_index, feature_state)."""
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
    feature_state = state.get("feature_state", None)
    return idx, (feature_state if isinstance(feature_state, dict) else None)

  def compute_feature_attribution(
    self,
    features: Dict[str, float],
    epsilon: float = 0.01,
    *,
    score_mode: str = "raw",
    meta: Meta | None = None,
  ) -> Tuple[float, Dict[str, float]]:
    """Model-agnostic perturbation-based attribution. Returns (score, {name: attribution})."""
    if score_mode not in ("raw", "lograw", "scaled"):
      raise ValueError("score_mode must be 'raw', 'lograw', or 'scaled'")
    if score_mode == "raw":
      score_fn = lambda f: self.score_only(f, meta=meta)[0]
    elif score_mode == "scaled":
      score_fn = lambda f: self.score_only(f, meta=meta)[1]
    else:
      score_fn = lambda f: math.log1p(max(0.0, float(self.score_only(f, meta=meta)[0])))
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
    score_fn = lambda f: detector.score_only(f)[1]
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


class OnlinePercentileCalibrator:
  """
  Online percentile calibration for anomaly scores.

  Maintains a fixed-size window of past log1p(raw_score) values and returns the
  percentile rank of the current value w.r.t. the past window (prequential).
  """

  def __init__(self, window_size: int = 2048, warmup: int = 128):
    self.window_size = int(max(32, window_size))
    self.warmup = int(max(0, warmup))
    self._queue: deque[float] = deque(maxlen=self.window_size)
    self._sorted: List[float] = []

  @staticmethod
  def _lograw(score_raw: float) -> float:
    return float(math.log1p(max(0.0, float(score_raw))))

  def percentile_prequential(self, score_raw: float) -> float:
    """
    Return percentile in [0,1] based on past window, then update window with this score.
    """
    x = self._lograw(score_raw)
    n = len(self._sorted)
    if n <= 0 or n < self.warmup:
      pct = 0.0
    else:
      k = bisect.bisect_right(self._sorted, x)
      pct = float(k) / float(n)

    if len(self._queue) >= self.window_size:
      old = self._queue.popleft()
      j = bisect.bisect_left(self._sorted, old)
      if 0 <= j < len(self._sorted):
        self._sorted.pop(j)

    self._queue.append(x)
    bisect.insort(self._sorted, x)
    return float(min(1.0, max(0.0, pct)))
