#!/usr/bin/env python3
"""Train an online detector on EVT1/JSONL and save a checkpoint.

This is an offline equivalent of "start detector, replay events": it loads a log file,
feeds each event through the same `OnlineAnomalyDetector.score_and_learn_event()` path,
then writes a pickle checkpoint via `OnlineAnomalyDetector.save_checkpoint()`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict

import events_pb2
from detector.config import load_config
from detector.model import OnlineAnomalyDetector
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl

log = logging.getLogger(Path(__file__).stem)


def _dict_to_event_envelope(obj: dict) -> events_pb2.EventEnvelope:
  from event_envelope_codec import envelope_from_dict

  return envelope_from_dict(obj)


def _make_detector(algo: str) -> tuple[OnlineAnomalyDetector, Callable[[events_pb2.EventEnvelope], Dict[str, float]]]:
  cfg = load_config()
  detector = OnlineAnomalyDetector(
    algorithm=algo,
    hst_n_trees=cfg.hst_n_trees,
    hst_height=cfg.hst_height,
    hst_window_size=cfg.hst_window_size,
    loda_n_projections=cfg.loda_n_projections,
    loda_bins=cfg.loda_bins,
    loda_range=cfg.loda_range,
    loda_ema_alpha=cfg.loda_ema_alpha,
    loda_hist_decay=cfg.loda_hist_decay,
    kitnet_max_size_ae=cfg.kitnet_max_size_ae,
    kitnet_grace_feature_mapping=cfg.kitnet_grace_feature_mapping,
    kitnet_grace_anomaly_detector=cfg.kitnet_grace_anomaly_detector,
    kitnet_learning_rate=cfg.kitnet_learning_rate,
    kitnet_hidden_ratio=cfg.kitnet_hidden_ratio,
    mem_memory_size=cfg.mem_memory_size,
    mem_beta=cfg.mem_beta,
    mem_k=cfg.mem_k,
    mem_gamma=cfg.mem_gamma,
    mem_lr=cfg.mem_lr,
    mem_input_mode=cfg.mem_input_mode,
    mem_warmup_accept=cfg.mem_warmup_accept,
    zscore_min_count=cfg.zscore_min_count,
    zscore_std_floor=cfg.zscore_std_floor,
    knn_k=cfg.knn_k,
    knn_memory_size=cfg.knn_memory_size,
    knn_metric=cfg.knn_metric,
    freq1d_bins=cfg.freq1d_bins,
    freq1d_alpha=cfg.freq1d_alpha,
    freq1d_decay=cfg.freq1d_decay,
    freq1d_max_categories=cfg.freq1d_max_categories,
    freq1d_aggregation=cfg.freq1d_aggregation,
    freq1d_topk=cfg.freq1d_topk,
    freq1d_soft_topk_temperature=cfg.freq1d_soft_topk_temperature,
    copulatree_u_clamp=cfg.copulatree_u_clamp,
    copulatree_reg=cfg.copulatree_reg,
    copulatree_max_features=cfg.copulatree_max_features,
    copulatree_importance_window=cfg.copulatree_importance_window,
    copulatree_tree_update_interval=cfg.copulatree_tree_update_interval,
    copulatree_edge_score_aggregation=cfg.copulatree_edge_score_aggregation,
    copulatree_edge_score_topk=cfg.copulatree_edge_score_topk,
    latentcluster_max_clusters=cfg.latentcluster_max_clusters,
    latentcluster_u_clamp=cfg.latentcluster_u_clamp,
    latentcluster_reg=cfg.latentcluster_reg,
    latentcluster_update_alpha=cfg.latentcluster_update_alpha,
    latentcluster_spawn_threshold=cfg.latentcluster_spawn_threshold,
    model_device=cfg.model_device,
    seed=cfg.model_seed,
    embedding_word2vec_dim=cfg.embedding_word2vec_dim,
    embedding_word2vec_sentence_len=cfg.embedding_word2vec_sentence_len,
    embedding_word2vec_window=cfg.embedding_word2vec_window,
    embedding_word2vec_sg=cfg.embedding_word2vec_sg,
    embedding_word2vec_update_every=cfg.embedding_word2vec_update_every,
    embedding_word2vec_epochs=cfg.embedding_word2vec_epochs,
    embedding_word2vec_post_warmup_lr_scale=cfg.embedding_word2vec_post_warmup_lr_scale,
    grimmer_ngram_length=cfg.grimmer_ngram_length,
    grimmer_thread_aware=cfg.grimmer_thread_aware,
    grimmer_mlp_hidden_size=cfg.grimmer_mlp_hidden_size,
    grimmer_mlp_hidden_layers=cfg.grimmer_mlp_hidden_layers,
    grimmer_mlp_lr=cfg.grimmer_mlp_lr,
    warmup_events=cfg.warmup_events,
  )

  def feature_fn(_evt: events_pb2.EventEnvelope) -> Dict[str, float]:
    # For `grimmer_mlp`, the detector consumes the full event and never calls this.
    # For other algorithms, use feature extraction via `detector/server.py` or `feature_attribution.py` instead.
    raise RuntimeError("feature extraction is not implemented in this trainer; use algorithm=grimmer_mlp")

  return detector, feature_fn


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  ap = argparse.ArgumentParser(description="Train detector offline on a log and save a checkpoint (intended for grimmer_mlp).")
  ap.add_argument("logfile", type=Path, help="Path to detector-events.jsonl or EVT1 events.bin (supports .gz).")
  ap.add_argument("--algorithm", default="grimmer_mlp", help="Detector algorithm to train (default: grimmer_mlp).")
  ap.add_argument("--out", type=Path, required=True, help="Output checkpoint path (.pkl).")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (for quick smoke tests).")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events.")
  args = ap.parse_args()

  path = Path(args.logfile)
  if not path.exists():
    log.error("File not found: %s", path)
    raise SystemExit(1)

  detector, feature_fn = _make_detector(args.algorithm)
  fmt = _detect_format(path)
  event_iter = iter_events_jsonl(path, max_events=args.max_events, skip=args.skip) if fmt == "jsonl" else iter_events(path, max_events=args.max_events, skip=args.skip)

  n = 0
  for obj in event_iter:
    evt = _dict_to_event_envelope(obj)
    detector.score_and_learn_event(evt, feature_fn=feature_fn)
    n += 1
    if (n % 10000) == 0:
      log.info("Trained on %d events...", n)

  if n == 0:
    log.error("No events processed; nothing to save.")
    raise SystemExit(2)

  detector.save_checkpoint(args.out, n)
  log.info("Saved checkpoint: %s (events learned: %d)", args.out, n)


if __name__ == "__main__":
  main()

