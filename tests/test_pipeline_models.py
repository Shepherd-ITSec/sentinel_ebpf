from __future__ import annotations

import math

import events_pb2
import pytest

from detector.building_blocks import (
  OnlineIDS,
  load_pipeline_checkpoint,
  save_pipeline_checkpoint,
)
from detector.building_blocks.core.base import DecisionOutput
from detector.config import DetectorConfig, detector_config_to_dict
from detector.pipelines import build_final_bb


def _event(i: int, syscall_name: str = "open") -> events_pb2.EventEnvelope:
  return events_pb2.EventEnvelope(
    event_id=str(i),
    event_group="process",
    pid="10",
    tid="10",
    uid="1000",
    comm="pytest",
    syscall_name=syscall_name,
    syscall_nr=2,
  )


@pytest.mark.parametrize("algorithm", ["zscore", "kitnet", "sequence_mlp"])
def test_single_model_pipeline_constructs_representative_algorithms(algorithm: str) -> None:
  cfg = DetectorConfig(
    pipeline_id="single_model",
    model_algorithm=algorithm,
    warmup_events=0,
    embedding_word2vec_update_every=1,
    embedding_word2vec_sentence_len=3,
    sequence_ngram_length=3,
    model_seed=7,
  )
  ids = OnlineIDS(build_final_bb(cfg), pipeline_id=cfg.pipeline_id)
  out = ids.run_event(_event(1, "open"))
  assert isinstance(out, DecisionOutput)
  assert math.isfinite(out.raw)
  assert 0.0 <= out.scaled <= 1.0


def test_pipeline_checkpoint_round_trip_preserves_next_score(tmp_path) -> None:
  cfg = DetectorConfig(
    pipeline_id="single_model",
    model_algorithm="zscore",
    warmup_events=0,
    model_seed=13,
  )
  ids = OnlineIDS(build_final_bb(cfg), pipeline_id=cfg.pipeline_id)
  for i in range(20):
    ids.run_event(_event(i, "read" if i % 2 else "open"))

  checkpoint = tmp_path / "pipeline.pkl"
  save_pipeline_checkpoint(
    checkpoint,
    pipeline_id=cfg.pipeline_id,
    manager=ids.manager,
    checkpoint_index=20,
    extra={"detector_config": detector_config_to_dict(cfg)},
  )
  expected = ids.run_event(_event(21, "write"))

  loaded = OnlineIDS(build_final_bb(cfg), pipeline_id=cfg.pipeline_id)
  _, extra = load_pipeline_checkpoint(checkpoint, loaded.manager)
  restored = loaded.run_event(_event(21, "write"))
  assert extra["detector_config"]["pipeline_id"] == cfg.pipeline_id
  assert extra["detector_config"]["model_algorithm"] == cfg.model_algorithm
  assert isinstance(expected, DecisionOutput)
  assert isinstance(restored, DecisionOutput)
  assert restored.raw == pytest.approx(expected.raw)
  assert restored.scaled == pytest.approx(expected.scaled)
  assert restored.primary == pytest.approx(expected.primary)
  assert restored.anomaly == expected.anomaly


def test_las_gas_fusion_pipeline_runs() -> None:
  cfg = DetectorConfig(
    pipeline_id="las_gas_fusion",
    warmup_events=0,
    embedding_word2vec_update_every=1,
    embedding_word2vec_sentence_len=3,
    sequence_ngram_length=3,
    model_seed=1,
  )
  ids = OnlineIDS(build_final_bb(cfg), pipeline_id=cfg.pipeline_id)
  scores = []
  for i, syscall_name in enumerate(("open", "read", "write", "close", "open", "read", "write")):
    out = ids.run_event(_event(i, syscall_name))
    assert isinstance(out, DecisionOutput)
    scores.append((out.raw, out.scaled))
  assert any(raw > 0.0 for raw, _ in scores)
  assert all(0.0 <= scaled <= 1.0 for _, scaled in scores)
