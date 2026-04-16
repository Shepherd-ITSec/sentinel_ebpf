from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from detector.building_blocks.core.cfg import model_impl_kwargs
from detector.building_blocks.primitives.models import (
  OnlineCopulaTree,
  OnlineFreq1D,
  OnlineHalfSpaceTrees,
  OnlineKNN,
  OnlineKitNet,
  OnlineLODAEMA,
  OnlineLatentCluster,
  OnlineMemStream,
  OnlineZScore,
)

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def scaled_score_for_algorithm(algorithm: str, raw: float) -> float:
  algo = (algorithm or "").strip().lower()
  if algo == "halfspacetrees":
    return float(raw)
  if algo in ("copulatree", "latentcluster"):
    pos = max(0.0, float(raw))
    return float(pos / (1.0 + pos))
  return float(1.0 - math.exp(-max(0.0, float(raw))))


def new_model_impl(algorithm: str, cfg: "DetectorConfig") -> Any:
  kw = model_impl_kwargs(cfg)
  algo = (algorithm or "").strip().lower()
  if algo == "halfspacetrees":
    return OnlineHalfSpaceTrees(
      n_trees=int(kw["hst_n_trees"]),
      height=int(kw["hst_height"]),
      window_size=int(kw["hst_window_size"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "loda_ema":
    return OnlineLODAEMA(
      n_projections=int(kw["loda_n_projections"]),
      bins=int(kw["loda_bins"]),
      proj_range=float(kw["loda_range"]),
      ema_alpha=float(kw["loda_ema_alpha"]),
      hist_decay=float(kw["loda_hist_decay"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "kitnet":
    return OnlineKitNet(
      max_size_ae=int(kw["kitnet_max_size_ae"]),
      grace_feature_mapping=int(kw["kitnet_grace_feature_mapping"]),
      grace_anomaly_detector=int(kw["kitnet_grace_anomaly_detector"]),
      learning_rate=float(kw["kitnet_learning_rate"]),
      hidden_ratio=float(kw["kitnet_hidden_ratio"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "memstream":
    return OnlineMemStream(
      memory_size=int(kw["mem_memory_size"]),
      lr=float(kw["mem_lr"]),
      beta=float(kw["mem_beta"]),
      k=int(kw["mem_k"]),
      gamma=float(kw["mem_gamma"]),
      input_mode=str(kw["mem_input_mode"]),
      freq1d_bins=int(kw["freq1d_bins"]),
      freq1d_alpha=float(kw["freq1d_alpha"]),
      freq1d_decay=float(kw["freq1d_decay"]),
      freq1d_max_categories=int(kw["freq1d_max_categories"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
      warmup_accept=int(kw["mem_warmup_accept"]),
    )
  if algo == "zscore":
    return OnlineZScore(
      min_count=int(kw["zscore_min_count"]),
      std_floor=float(kw["zscore_std_floor"]),
      topk=int(kw["zscore_topk"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "knn":
    return OnlineKNN(
      k=int(kw["knn_k"]),
      memory_size=int(kw["knn_memory_size"]),
      metric=str(kw["knn_metric"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "freq1d":
    return OnlineFreq1D(
      bins=int(kw["freq1d_bins"]),
      alpha=float(kw["freq1d_alpha"]),
      decay=float(kw["freq1d_decay"]),
      max_categories=int(kw["freq1d_max_categories"]),
      aggregation=str(kw["freq1d_aggregation"]),
      topk=int(kw["freq1d_topk"]),
      soft_topk_temperature=float(kw["freq1d_soft_topk_temperature"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "copulatree":
    return OnlineCopulaTree(
      bins=int(kw["freq1d_bins"]),
      alpha=float(kw["freq1d_alpha"]),
      decay=float(kw["freq1d_decay"]),
      max_categories=int(kw["freq1d_max_categories"]),
      u_clamp=float(kw["copulatree_u_clamp"]),
      reg=float(kw["copulatree_reg"]),
      max_features=int(kw["copulatree_max_features"]),
      importance_window=int(kw["copulatree_importance_window"]),
      tree_update_interval=int(kw["copulatree_tree_update_interval"]),
      edge_score_aggregation=str(kw["copulatree_edge_score_aggregation"]),
      edge_score_topk=int(kw["copulatree_edge_score_topk"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  if algo == "latentcluster":
    return OnlineLatentCluster(
      bins=int(kw["freq1d_bins"]),
      alpha=float(kw["freq1d_alpha"]),
      decay=float(kw["freq1d_decay"]),
      max_categories=int(kw["freq1d_max_categories"]),
      max_clusters=int(kw["latentcluster_max_clusters"]),
      u_clamp=float(kw["latentcluster_u_clamp"]),
      reg=float(kw["latentcluster_reg"]),
      update_alpha=float(kw["latentcluster_update_alpha"]),
      spawn_threshold=float(kw["latentcluster_spawn_threshold"]),
      model_device=str(kw["model_device"]),
      seed=int(kw["seed"]),
    )
  raise ValueError(f"Unknown model algorithm: {algorithm!r}")

