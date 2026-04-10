"""Checkpoint roundtrip preserves sequence feature state (Word2Vec, class table, n-grams)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import events_pb2
from detector.config import DetectorConfig
from detector.features import build_feature_extractor
from detector.model import OnlineAnomalyDetector


def test_sequence_mlp_checkpoint_restores_word2vec_and_scores_match() -> None:
  cfg = DetectorConfig(
    model_algorithm="sequence_mlp",
    sequence_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_window=2,
    embedding_word2vec_update_every=1,
    embedding_word2vec_epochs=1,
    warmup_events=0,
    sequence_mlp_hidden_size=16,
    sequence_mlp_hidden_layers=1,
    sequence_mlp_lr=0.05,
    model_device="cpu",
    model_seed=0,
  )

  pattern = ("open", "read", "write", "close", "mmap", "munmap", "open", "read")
  events = [
    events_pb2.EventEnvelope(event_id=str(i), syscall_name=name, tid="7")
    for i, name in enumerate(pattern)
  ]

  def make_pair():
    extractor = build_feature_extractor(cfg)
    detector = OnlineAnomalyDetector(
      algorithm="sequence_mlp",
      model_device=cfg.model_device,
      seed=cfg.model_seed,
      warmup_events=cfg.warmup_events,
      embedding_word2vec_dim=cfg.embedding_word2vec_dim,
      embedding_word2vec_sentence_len=cfg.embedding_word2vec_sentence_len,
      embedding_word2vec_window=cfg.embedding_word2vec_window,
      embedding_word2vec_sg=cfg.embedding_word2vec_sg,
      embedding_word2vec_update_every=cfg.embedding_word2vec_update_every,
      embedding_word2vec_epochs=cfg.embedding_word2vec_epochs,
      embedding_word2vec_post_warmup_lr_scale=cfg.embedding_word2vec_post_warmup_lr_scale,
      sequence_mlp_hidden_size=cfg.sequence_mlp_hidden_size,
      sequence_mlp_hidden_layers=cfg.sequence_mlp_hidden_layers,
      sequence_mlp_lr=cfg.sequence_mlp_lr,
    )

    def feature_fn(evt: events_pb2.EventEnvelope):
      return extractor.extract_feature_dict(evt, feature_view="sequence")

    return detector, extractor, feature_fn

  # Full replay baseline.
  full_det, full_ex, full_fn = make_pair()
  for evt in events[:-1]:
    full_det.score_and_learn_event(evt, feature_fn=full_fn)
  last_feats, last_meta = full_fn(events[-1])
  full_raw, full_scaled = full_det.score_only(last_feats, meta=last_meta)

  # Checkpoint at mid-point; load and continue should match exactly.
  ckpt_det, ckpt_ex, ckpt_fn = make_pair()
  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
    ckpt_path = Path(f.name)
  try:
    for evt in events[:4]:
      ckpt_det.score_and_learn_event(evt, feature_fn=ckpt_fn)
    ckpt_det.save_checkpoint(ckpt_path, 4, feature_state=ckpt_ex.get_state())

    loaded_det, loaded_ex, loaded_fn = make_pair()
    idx, feature_state = loaded_det.load_checkpoint(ckpt_path)
    assert idx == 4
    assert feature_state is not None
    loaded_ex.set_state(feature_state)

    for evt in events[4:-1]:
      loaded_det.score_and_learn_event(evt, feature_fn=loaded_fn)
    ld_feats, ld_meta = loaded_fn(events[-1])
    loaded_raw, loaded_scaled = loaded_det.score_only(ld_feats, meta=ld_meta)

    assert np.isfinite(full_raw) and np.isfinite(loaded_raw)
    assert np.isfinite(full_scaled) and np.isfinite(loaded_scaled)
    assert abs(float(full_raw) - float(loaded_raw)) < 1e-12
    assert abs(float(full_scaled) - float(loaded_scaled)) < 1e-12
  finally:
    ckpt_path.unlink(missing_ok=True)

