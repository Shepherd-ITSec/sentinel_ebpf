"""Tests for Grimmer / LID-DS-ported syscall n-gram + Word2Vec + MLP pipeline."""

from __future__ import annotations

import numpy as np
import pytest

import events_pb2
from detector.config import DetectorConfig
from detector.features import extract_feature_dict, feature_view_for_algorithm
from detector.grimmer.mlp_online import GrimmerOnlineMLP
from detector.grimmer.ngram_buffer import ThreadNgramBuffer
from detector.embeddings.online_word2vec import OnlineSyscallWord2Vec
from detector.grimmer.pipeline import GrimmerPipeline
from detector.grimmer.syscall_classes import SyscallClassTable
from detector.model import OnlineAnomalyDetector


def test_feature_view_maps_grimmer_mlp() -> None:
  assert feature_view_for_algorithm("grimmer_mlp") == "grimmer_mlp"


def test_extract_feature_dict_rejects_grimmer_view() -> None:
  evt = events_pb2.EventEnvelope(event_id="1", syscall_name="read", tid="1")
  with pytest.raises(ValueError, match="stateful"):
    extract_feature_dict(evt, feature_view="grimmer_mlp")


def test_thread_ngram_buffers_are_per_tid() -> None:
  """Ported semantics: thread-aware mode keys buffers by tid (LID-DS Ngram)."""
  buf = ThreadNgramBuffer(thread_aware=True, ngram_length=3, element_width=2)
  _, ok1 = buf.push(1, [1.0, 0.0])
  assert not ok1
  _, ok2 = buf.push(2, [0.0, 1.0])
  assert not ok2
  _, ok3 = buf.push(1, [0.5, 0.5])
  assert not ok3
  full, ok4 = buf.push(1, [0.0, 0.0])
  assert ok4 and full is not None
  assert len(full) == 6
  ctx = buf.context_prefix(full, context_ngrams=2)
  assert len(ctx) == 4


def test_syscall_class_table_grows() -> None:
  t = SyscallClassTable()
  assert t.register("read") == 0
  assert t.register("write") == 1
  assert t.register("read") == 0
  assert t.num_classes == 2


def test_online_word2vec_trains_and_looks_up() -> None:
  w = OnlineSyscallWord2Vec(vector_size=4, sentence_len=3, seed=0, w2v_window=2)
  for _ in range(3):
    w.observe(0, "a")
    w.observe(0, "b")
    w.observe(0, "c")
  sents = w.drain_pending_sentences()
  w.train_on_sentences(sents, epochs=5)
  v = w.vector_for("a")
  assert v.shape == (4,)
  assert np.linalg.norm(v) > 1e-6


def test_grimmer_mlp_grows_output_layer() -> None:
  mlp = GrimmerOnlineMLP(
    input_dim=6,
    hidden_size=8,
    hidden_layers=1,
    learning_rate=0.05,
    model_device="cpu",
    seed=0,
  )
  mlp.ensure_num_classes(1)
  x = np.zeros(6, dtype=np.float32)
  r1, _ = mlp.score_and_learn(x, 0)
  assert 0.0 <= r1 <= 1.0
  mlp.ensure_num_classes(2)
  r2, _ = mlp.score_and_learn(x, 1)
  assert 0.0 <= r2 <= 1.0


def test_grimmer_pipeline_learns_repeating_pattern() -> None:
  cfg = DetectorConfig(
    model_algorithm="grimmer_mlp",
    grimmer_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_window=2,
    warmup_events=500,
    embedding_word2vec_update_every=1,
    embedding_word2vec_epochs=1,
    grimmer_mlp_hidden_size=32,
    grimmer_mlp_hidden_layers=1,
    grimmer_mlp_lr=0.05,
    model_device="cpu",
    model_seed=0,
  )
  pipe = GrimmerPipeline(cfg)
  pattern = ("open", "read", "write", "close")
  early: list[float] = []
  late: list[float] = []
  for i in range(200):
    s = pattern[i % len(pattern)]
    evt = events_pb2.EventEnvelope(event_id=str(i), syscall_name=s, tid="99")
    raw, _ = pipe.score_and_learn_event(evt)
    assert np.isfinite(raw)
    if 20 <= i < 40:
      early.append(raw)
    elif i >= 160:
      late.append(raw)
  assert early and late
  # After many repetitions, mean surprisal should not increase vs early training window.
  assert float(np.mean(late)) <= float(np.mean(early)) + 0.35


def test_grimmer_suppress_warmup_flag() -> None:
  cfg = DetectorConfig(
    warmup_events=0,
  )
  pipe = GrimmerPipeline(cfg)
  evt = events_pb2.EventEnvelope(event_id="1", syscall_name="read", tid="1")
  pipe.score_and_learn_event(evt)
  assert pipe.events_processed >= 1


def test_grimmer_mlp_constructed_via_factory() -> None:
  cfg = DetectorConfig(
    model_algorithm="grimmer_mlp",
    grimmer_ngram_length=4,
    embedding_word2vec_dim=6,
    embedding_word2vec_sentence_len=3,
    embedding_word2vec_update_every=1,
    warmup_events=0,
    model_device="cpu",
    model_seed=0,
  )
  det = OnlineAnomalyDetector(
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
    grimmer_ngram_length=cfg.grimmer_ngram_length,
    grimmer_thread_aware=cfg.grimmer_thread_aware,
    grimmer_mlp_hidden_size=cfg.grimmer_mlp_hidden_size,
    grimmer_mlp_hidden_layers=cfg.grimmer_mlp_hidden_layers,
    grimmer_mlp_lr=cfg.grimmer_mlp_lr,
  )
  evt = events_pb2.EventEnvelope(event_id="1", syscall_name="read", tid="1")
  raw, scaled = det.score_and_learn_event(evt, feature_fn=lambda e: {})
  assert np.isfinite(raw)
  assert 0.0 <= scaled <= 1.0
