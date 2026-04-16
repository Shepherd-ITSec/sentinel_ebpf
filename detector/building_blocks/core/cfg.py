"""Map :class:`detector.config.DetectorConfig` to model implementation kwargs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def model_impl_kwargs(cfg: "DetectorConfig") -> dict[str, Any]:
  """Kwargs shared by stock model building blocks."""
  return {
    "hst_n_trees": cfg.hst_n_trees,
    "hst_height": cfg.hst_height,
    "hst_window_size": cfg.hst_window_size,
    "loda_n_projections": cfg.loda_n_projections,
    "loda_bins": cfg.loda_bins,
    "loda_range": cfg.loda_range,
    "loda_ema_alpha": cfg.loda_ema_alpha,
    "loda_hist_decay": cfg.loda_hist_decay,
    "kitnet_max_size_ae": cfg.kitnet_max_size_ae,
    "kitnet_grace_feature_mapping": cfg.kitnet_grace_feature_mapping,
    "kitnet_grace_anomaly_detector": cfg.kitnet_grace_anomaly_detector,
    "kitnet_learning_rate": cfg.kitnet_learning_rate,
    "kitnet_hidden_ratio": cfg.kitnet_hidden_ratio,
    "mem_memory_size": cfg.mem_memory_size,
    "mem_lr": cfg.mem_lr,
    "mem_beta": cfg.mem_beta,
    "mem_k": cfg.mem_k,
    "mem_gamma": cfg.mem_gamma,
    "mem_input_mode": cfg.mem_input_mode,
    "mem_warmup_accept": cfg.mem_warmup_accept,
    "zscore_min_count": cfg.zscore_min_count,
    "zscore_std_floor": cfg.zscore_std_floor,
    "zscore_topk": cfg.zscore_topk,
    "knn_k": cfg.knn_k,
    "knn_memory_size": cfg.knn_memory_size,
    "knn_metric": cfg.knn_metric,
    "freq1d_bins": cfg.freq1d_bins,
    "freq1d_alpha": cfg.freq1d_alpha,
    "freq1d_decay": cfg.freq1d_decay,
    "freq1d_max_categories": cfg.freq1d_max_categories,
    "freq1d_aggregation": cfg.freq1d_aggregation,
    "freq1d_topk": cfg.freq1d_topk,
    "freq1d_soft_topk_temperature": cfg.freq1d_soft_topk_temperature,
    "copulatree_u_clamp": cfg.copulatree_u_clamp,
    "copulatree_reg": cfg.copulatree_reg,
    "copulatree_max_features": cfg.copulatree_max_features,
    "copulatree_importance_window": cfg.copulatree_importance_window,
    "copulatree_tree_update_interval": cfg.copulatree_tree_update_interval,
    "copulatree_edge_score_aggregation": cfg.copulatree_edge_score_aggregation,
    "copulatree_edge_score_topk": cfg.copulatree_edge_score_topk,
    "latentcluster_max_clusters": cfg.latentcluster_max_clusters,
    "latentcluster_u_clamp": cfg.latentcluster_u_clamp,
    "latentcluster_reg": cfg.latentcluster_reg,
    "latentcluster_update_alpha": cfg.latentcluster_update_alpha,
    "latentcluster_spawn_threshold": cfg.latentcluster_spawn_threshold,
    "model_device": cfg.model_device,
    "seed": cfg.model_seed,
    "warmup_events": cfg.warmup_events,
    "embedding_word2vec_dim": cfg.embedding_word2vec_dim,
    "embedding_word2vec_sentence_len": cfg.embedding_word2vec_sentence_len,
    "embedding_word2vec_window": cfg.embedding_word2vec_window,
    "embedding_word2vec_sg": cfg.embedding_word2vec_sg,
    "embedding_word2vec_update_every": cfg.embedding_word2vec_update_every,
    "embedding_word2vec_epochs": cfg.embedding_word2vec_epochs,
    "embedding_word2vec_post_warmup_lr_scale": cfg.embedding_word2vec_post_warmup_lr_scale,
    "sequence_mlp_hidden_size": cfg.sequence_mlp_hidden_size,
    "sequence_mlp_hidden_layers": cfg.sequence_mlp_hidden_layers,
    "sequence_mlp_lr": cfg.sequence_mlp_lr,
  }

