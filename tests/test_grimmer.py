"""Tests for the generic sequence feature view and sequence MLP."""

from __future__ import annotations

import numpy as np

import events_pb2
from detector.config import DetectorConfig
from detector.embeddings.online_word2vec import OnlineTokenWord2Vec
from detector.features import build_feature_extractor, feature_view_for_algorithm
from detector.model import OnlineAnomalyDetector
from detector.sequence.class_table import TokenClassTable
from detector.sequence.mlp import OnlineSequenceMLP
from detector.sequence.ngram_buffer import StreamNgramBuffer


def test_feature_view_maps_sequence_algorithms() -> None:
  assert feature_view_for_algorithm("sequence_mlp") == "sequence"
  assert feature_view_for_algorithm("sequence_transformer") == "sequence"


def test_extract_feature_dict_emits_sequence_context_with_metadata() -> None:
  cfg = DetectorConfig(
    sequence_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_update_every=1,
    warmup_events=0,
    model_seed=0,
  )
  extractor = build_feature_extractor(cfg)
  names: set[str] | None = None
  ready_seen = False
  for i, syscall_name in enumerate(("open", "read", "write", "close", "open", "read")):
    evt = events_pb2.EventEnvelope(event_id=str(i), syscall_name=syscall_name, tid="7")
    values = extractor.extract_feature_dict(evt, feature_view="sequence")
    current_names = set(values)
    if names is None:
      names = current_names
    assert current_names == names
    assert "sequence_ctx_ready" not in values
    meta = getattr(values, "sequence_meta", None)
    assert meta is not None
    if meta.ready:
      ready_seen = True
  assert ready_seen
  assert names is not None
  assert len([name for name in names if name.startswith("sequence_ctx_")]) == 18


def test_thread_ngram_buffers_are_per_tid() -> None:
  buf = StreamNgramBuffer(thread_aware=True, ngram_length=3, element_width=2)
  _, ok1 = buf.push(1, [1.0, 0.0])
  assert not ok1
  _, ok2 = buf.push(2, [0.0, 1.0])
  assert not ok2
  _, ok3 = buf.push(1, [0.5, 0.5])
  assert not ok3
  full, ok4 = buf.push(1, [0.0, 0.0])
  assert ok4 and full is not None
  assert len(full) == 6
  ctx = buf.context_prefix(full, n_context=2)
  assert len(ctx) == 4


def test_token_class_table_grows() -> None:
  table = TokenClassTable()
  assert table.register("read") == 0
  assert table.register("write") == 1
  assert table.register("read") == 0
  assert table.num_classes == 2


def test_online_word2vec_trains_and_looks_up() -> None:
  w2v = OnlineTokenWord2Vec(vector_size=4, sentence_len=3, seed=0, w2v_window=2)
  for _ in range(3):
    w2v.observe(0, "a")
    w2v.observe(0, "b")
    w2v.observe(0, "c")
  sentences = w2v.drain_pending_sentences()
  w2v.train_on_sentences(sentences, epochs=5)
  vector = w2v.vector_for("a")
  assert vector.shape == (4,)
  assert np.linalg.norm(vector) > 1e-6


def test_sequence_mlp_grows_output_layer() -> None:
  mlp = OnlineSequenceMLP(
    hidden_size=8,
    hidden_layers=1,
    learning_rate=0.05,
    model_device="cpu",
    seed=0,
  )
  features_a = {f"sequence_ctx_{i:03d}": 0.0 for i in range(6)}
  features_a = type("FeatureDict", (dict,), {})(features_a)
  features_a.sequence_meta = type("Meta", (), {"ready": True, "target_id": 0, "num_classes": 1})()
  raw_a, _ = mlp.score_and_learn(features_a)
  assert 0.0 <= raw_a <= 1.0

  features_b = {f"sequence_ctx_{i:03d}": 0.0 for i in range(6)}
  features_b = type("FeatureDict", (dict,), {})(features_b)
  features_b.sequence_meta = type("Meta", (), {"ready": True, "target_id": 1, "num_classes": 2})()
  raw_b, _ = mlp.score_and_learn(features_b)
  assert 0.0 <= raw_b <= 1.0


def test_sequence_mlp_learns_repeating_pattern() -> None:
  cfg = DetectorConfig(
    model_algorithm="sequence_mlp",
    sequence_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_window=2,
    warmup_events=500,
    embedding_word2vec_update_every=1,
    embedding_word2vec_epochs=1,
    sequence_mlp_hidden_size=32,
    sequence_mlp_hidden_layers=1,
    sequence_mlp_lr=0.05,
    model_device="cpu",
    model_seed=0,
  )
  extractor = build_feature_extractor(cfg)
  detector = OnlineAnomalyDetector(
    algorithm=cfg.model_algorithm,
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
  pattern = ("open", "read", "write", "close")
  early: list[float] = []
  late: list[float] = []
  for i in range(200):
    syscall_name = pattern[i % len(pattern)]
    evt = events_pb2.EventEnvelope(event_id=str(i), syscall_name=syscall_name, tid="99")
    features = extractor.extract_feature_dict(evt, feature_view="sequence")
    raw, _ = detector.score_and_learn(features)
    assert np.isfinite(raw)
    if 20 <= i < 40:
      early.append(raw)
    elif i >= 160:
      late.append(raw)
  assert early and late
  assert float(np.mean(late)) <= float(np.mean(early)) + 0.35


def test_sequence_mlp_constructed_via_factory() -> None:
  cfg = DetectorConfig(
    model_algorithm="sequence_mlp",
    sequence_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_update_every=1,
    warmup_events=0,
    model_device="cpu",
    model_seed=0,
  )
  detector = OnlineAnomalyDetector(
    algorithm=cfg.model_algorithm,
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
  extractor = build_feature_extractor(cfg)
  evt = events_pb2.EventEnvelope(event_id="1", syscall_name="read", tid="1")
  raw, scaled = detector.score_and_learn_event(
    evt,
    feature_fn=lambda event: extractor.extract_feature_dict(event, feature_view="sequence"),
  )
  assert np.isfinite(raw)
  assert 0.0 <= scaled <= 1.0


def test_generic_model_can_consume_sequence_features() -> None:
  cfg = DetectorConfig(
    model_algorithm="zscore",
    sequence_ngram_length=4,
    embedding_word2vec_dim=4,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_update_every=1,
    warmup_events=0,
    zscore_min_count=1,
    model_seed=0,
  )
  extractor = build_feature_extractor(cfg)
  detector = OnlineAnomalyDetector(
    algorithm="zscore",
    zscore_min_count=cfg.zscore_min_count,
    zscore_std_floor=cfg.zscore_std_floor,
  )
  for i, syscall_name in enumerate(("open", "read", "write", "close", "open", "read", "write", "close")):
    evt = events_pb2.EventEnvelope(event_id=str(i), syscall_name=syscall_name, tid="11")
    features = extractor.extract_feature_dict(evt, feature_view="sequence")
    raw, scaled = detector.score_and_learn(features)
    assert np.isfinite(raw)
    assert 0.0 <= scaled <= 1.0
