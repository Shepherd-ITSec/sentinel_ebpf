from __future__ import annotations

import contextlib
import io
import logging
from typing import Any, Dict, List, Optional

import numpy as np
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

from detector.building_blocks.primitives.models.common import _BothScoresMixin, _auto_kitnet_max_size_ae, _resolve_torch_device


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

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    return float(self.model.score_one(features))

  def score_only(self, features: Dict[str, float]) -> tuple[float, float]:
    raw = self.score_only_raw(features)
    return (float(raw), float(raw))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    score = self.model.score_one(features)
    self.model.learn_one(features)
    return float(score)

  def score_and_learn(self, features: Dict[str, float]) -> tuple[float, float]:
    raw = self.score_and_learn_raw(features)
    return (float(raw), float(raw))

  def get_state(self) -> Dict[str, Any]:
    return {"model": self.model}

  def set_state(self, state: Dict[str, Any]) -> None:
    self.model = state["model"]


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
      self._model.fit_partial(x)
    if not np.isfinite(raw):
      logging.warning("KitNet produced non-finite score (%s); falling back to 0.0", raw)
      raw = 0.0
    return max(0.0, float(raw))

  def get_state(self) -> Dict[str, Any]:
    if self._model is None:
      raise RuntimeError("KitNet not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "effective_max_size_ae": self._effective_max_size_ae,
      "model": self._model,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self._feature_names = list(state["feature_names"])
    self._effective_max_size_ae = int(state["effective_max_size_ae"])
    self._model = state["model"]


class OnlineKNN(_BothScoresMixin):
  """Online KNN anomaly detector with sliding memory. (Library-based wrapper)"""

  def __init__(self, k: int, memory_size: int, metric: str, model_device: str, seed: int):
    del seed
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
    x = self._vectorize(features)
    raw = self._raw_from_vector(x)
    self._learn_vector(x)
    return max(0.0, raw)

  def get_state(self) -> Dict[str, Any]:
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
    self._feature_names = list(state["feature_names"])
    self.k = int(state.get("k", self.k))
    self.memory_size = int(state.get("memory_size", self.memory_size))
    self.metric = str(state.get("metric", self.metric))
    self._memory = np.asarray(state["memory"], dtype=np.float64)
    self._mem_index = int(state["mem_index"])
    self._mem_filled = int(state["mem_filled"])
