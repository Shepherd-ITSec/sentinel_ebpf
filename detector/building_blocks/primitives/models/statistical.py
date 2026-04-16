from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fenwick import FenwickTree
from scipy.stats import norm as scipy_norm

from detector.building_blocks.primitives.models.common import _BothScoresMixin, _fenwick_prefix_sum, _resolve_torch_device


class OnlineZScore(_BothScoresMixin):
  """Simple online per-feature Z-score baseline. (Self-implementation)"""

  def __init__(self, min_count: int, std_floor: float, topk: int, model_device: str, seed: int):
    del seed
    self.algorithm = "zscore"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning("algorithm=%s requested on CUDA, but zscore is CPU-only; using CPU", self.algorithm)
    self.min_count = int(max(1, min_count))
    self.std_floor = float(max(1e-12, std_floor))
    self.topk = int(max(1, topk))
    self._feature_names: Optional[List[str]] = None
    self._count = 0
    self._mean: Optional[np.ndarray] = None
    self._m2: Optional[np.ndarray] = None
    logging.info("Initialized %s (min_count=%d, std_floor=%.6f, topk=%d)", self.algorithm, self.min_count, self.std_floor, self.topk)

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
    x = self._vectorize(features)
    raw = self._raw_from_vector(x)
    self._learn_vector(x)
    return max(0.0, raw)

  def get_state(self) -> Dict[str, Any]:
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
    self._feature_names = list(state["feature_names"])
    self.min_count = int(state.get("min_count", self.min_count))
    self.std_floor = float(state.get("std_floor", self.std_floor))
    self.topk = int(state.get("topk", self.topk))
    self._count = int(state["count"])
    self._mean = np.asarray(state["mean"], dtype=np.float64)
    self._m2 = np.asarray(state["m2"], dtype=np.float64)


class OnlineFreq1D(_BothScoresMixin):
  _BINARY_FEATURES = frozenset({"return_success", "file_sensitive_path", "file_tmp_path", "proc_sensitive_path", "proc_tmp_path"})
  _HASH_KEY_SPACE = 10000
  _BINARY_PREFIXES = ("group_syscall_",)

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
    del seed
    self.algorithm = "freq1d"
    requested_device = _resolve_torch_device(model_device)
    self.device = "cpu"
    if requested_device.type == "cuda":
      logging.warning("algorithm=%s requested on CUDA, but freq1d is CPU-only; using CPU", self.algorithm)
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
    self._kind: Optional[List[str]] = None
    self._cat_mode: Optional[List[Optional[str]]] = None
    self._num_slot: Optional[List[int]] = None
    self._num_counts: List[np.ndarray] = []
    self._num_fenwick: List[FenwickTree] = []
    self._num_scale: List[float] = []
    self._cat_slot: Optional[List[int]] = None
    self._cat_counts: List[Any] = []
    self._cat_hash_fenwick: List[Optional[FenwickTree]] = []
    self._cat_other: List[float] = []
    self._cat_scale: List[float] = []

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
      topk = np.partition(arr, -k)[-k:]
      return float(np.mean(topk))
    arr = np.asarray(scores, dtype=np.float64)
    k = min(len(arr), self.topk)
    if k <= 0:
      return 0.0
    topk = np.partition(arr, -k)[-k:]
    temp = float(self.soft_topk_temperature)
    m = float(np.max(topk))
    w = np.exp((topk - m) / temp)
    denom = float(np.sum(w))
    if denom <= 0.0 or not np.isfinite(denom):
      return float(np.mean(topk))
    return float(np.sum(w * topk) / denom)

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
    self._num_counts = []
    self._num_fenwick = []
    self._num_scale = []
    self._cat_counts = []
    self._cat_hash_fenwick = []
    self._cat_other = []
    self._cat_scale = []
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
    return min(max(0, idx), self.bins - 1)

  def _cat_key(self, mode: str, x: float) -> int:
    v = float(x)
    if mode == "binary":
      return 1 if v >= 0.5 else 0
    if v <= 0.0:
      return 0
    if v >= 1.0:
      return 9999
    return min(max(0, int(v * 10000.0 + 1e-6)), 9999)

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
    fenwick = self._num_fenwick[slot]
    total = _fenwick_prefix_sum(fenwick, self.bins - 1)
    idx = self._bin_numeric(x)
    cum = _fenwick_prefix_sum(fenwick, idx)
    cdf = (cum + self.alpha) / (total + self.alpha * self.bins) if total >= 0.0 else (idx + 1) / float(self.bins)
    return float(np.clip(cdf, 1e-12, 1.0 - 1e-12))

  def _cdf_cat_slot(self, slot: int, mode: str, x: float) -> float:
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
    self._ensure_init(features)
    if self._feature_names is None or self._kind is None or self._num_slot is None or self._cat_slot is None or self._cat_mode is None:
      raise RuntimeError("Freq1D not initialized")
    out = np.zeros(len(self._feature_names), dtype=np.float64)
    for i, name in enumerate(self._feature_names):
      x = float(features.get(name, 0.0))
      if self._kind[i] == "num":
        slot = self._num_slot[i]
        out[i] = self._cdf_numeric_slot(slot, x) if slot >= 0 else 0.5
      else:
        slot = self._cat_slot[i]
        mode = self._cat_mode[i] or "hash"
        out[i] = self._cdf_cat_slot(slot, mode, x) if slot >= 0 else 0.5
    return out

  def get_excess_vector(self, features: Dict[str, float]) -> np.ndarray:
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
      "cat_counts": [d.tolist() if isinstance(d, np.ndarray) else dict(d) for d in self._cat_counts],
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
    self._num_counts = [np.asarray(c, dtype=np.float64) for c in state.get("num_counts", [])]
    self._num_fenwick = [FenwickTree(len(c)) for c in self._num_counts]
    for i, c in enumerate(self._num_counts):
      self._num_fenwick[i].init(list(c))
    self._num_scale = [float(s) for s in state.get("num_scale", [])]
    raw_cat = state.get("cat_counts", [])
    self._cat_counts = []
    self._cat_hash_fenwick = []
    for d in raw_cat:
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
      logging.warning("algorithm=%s requested on CUDA, but copulatree is CPU-only; using CPU", self.algorithm)
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
    self._marginals = OnlineFreq1D(bins=bins, alpha=alpha, decay=decay, max_categories=max_categories, aggregation="mean", topk=1, soft_topk_temperature=1.0, model_device="cpu", seed=0)
    self._n = 0
    self._pair_outer_ema: Optional[np.ndarray] = None
    self._importance: Optional[np.ndarray] = None
    self._selected_indices: Optional[np.ndarray] = None
    self._events_since_selection = 0
    self._tree_edges: List[Tuple[int, int]] = []
    self._events_since_tree = 0

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
    return float(np.mean(np.partition(arr, -k)[-k:]))

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
    scores = [self._pair_score(float(z[i]), float(z[j]), float(corr[i, j])) for i, j in self._tree_edges if i < len(z) and j < len(z)]
    return max(0.0, float(self._aggregate_edge_scores(scores)))

  def _maybe_select_and_get_z(self, z: np.ndarray, do_learn: bool) -> np.ndarray:
    d = len(z)
    past_warmup = self._selected_indices is not None or self._n >= self._importance_window
    use_selection = self._max_features > 0 and self._max_features < d and past_warmup
    if self._importance is None or len(self._importance) != d:
      self._importance = np.zeros(d, dtype=np.float64)
    if do_learn:
      self._importance = self._importance_ema_alpha * self._importance + (1.0 - self._importance_ema_alpha) * np.abs(z)
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
    return self._score_selected_z(self._maybe_select_and_get_z(z, do_learn=False))

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
      self._pair_outer_ema = self._pair_ema_alpha * self._pair_outer_ema + (1.0 - self._pair_ema_alpha) * outer
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
      logging.warning("algorithm=%s requested on CUDA, but latentcluster is CPU-only; using CPU", self.algorithm)
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
    self._marginals = OnlineFreq1D(bins=bins, alpha=alpha, decay=decay, max_categories=max_categories, aggregation="mean", topk=1, soft_topk_temperature=1.0, model_device="cpu", seed=0)
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
    should_spawn = best_idx is None or (raw > self.spawn_threshold and self._active_clusters < self.max_clusters)
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
