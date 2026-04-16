"""Tests for scripts/score_from_checkpoint.py."""

import json
import logging
import pickle
from types import SimpleNamespace

import pytest

from detector.building_blocks.core.base import DecisionOutput
from detector.config import DetectorConfig, detector_config_to_dict
from scripts.performance_measurement import PerformanceEvent, PerformanceMeasurement
import scripts.score_from_checkpoint as score_from_checkpoint
from scripts.score_from_checkpoint import _default_summary_path, _iter_event_dicts


def _write_jsonl(path, rows):
  path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_iter_event_dicts_max_recordings_stops_on_new_recording(temp_dir):
  log_file = temp_dir / "events.jsonl"
  _write_jsonl(
    log_file,
    [
      {"event_id": "evt-1", "ts_unix_nano": 1, "syscall_name": "openat", "lidds_recording_name": "rec-1"},
      {"event_id": "evt-2", "ts_unix_nano": 2, "syscall_name": "read", "lidds_recording_name": "rec-1"},
      {"event_id": "evt-3", "ts_unix_nano": 3, "syscall_name": "write", "lidds_recording_name": "rec-2"},
      {"event_id": "evt-4", "ts_unix_nano": 4, "syscall_name": "close", "lidds_recording_name": "rec-2"},
      {"event_id": "evt-5", "ts_unix_nano": 5, "syscall_name": "execve", "lidds_recording_name": "rec-3"},
    ],
  )

  events = list(_iter_event_dicts(log_file, max_recordings=2))

  assert [event["event_id"] for event in events] == ["evt-1", "evt-2", "evt-3", "evt-4"]


def test_iter_event_dicts_max_recordings_combines_with_max_events(temp_dir):
  log_file = temp_dir / "events.jsonl"
  _write_jsonl(
    log_file,
    [
      {"event_id": "evt-1", "ts_unix_nano": 1, "syscall_name": "openat", "lidds_recording_name": "rec-1"},
      {"event_id": "evt-2", "ts_unix_nano": 2, "syscall_name": "read", "lidds_recording_name": "rec-1"},
      {"event_id": "evt-3", "ts_unix_nano": 3, "syscall_name": "write", "lidds_recording_name": "rec-2"},
      {"event_id": "evt-4", "ts_unix_nano": 4, "syscall_name": "close", "lidds_recording_name": "rec-2"},
    ],
  )

  events = list(_iter_event_dicts(log_file, max_events=3, max_recordings=2))

  assert [event["event_id"] for event in events] == ["evt-1", "evt-2", "evt-3"]


def test_iter_event_dicts_max_recordings_requires_recording_name(temp_dir):
  log_file = temp_dir / "events.jsonl"
  _write_jsonl(
    log_file,
    [
      {"event_id": "evt-1", "ts_unix_nano": 1, "syscall_name": "openat"},
    ],
  )

  with pytest.raises(ValueError, match="lidds_recording_name"):
    list(_iter_event_dicts(log_file, max_recordings=1))


def test_performance_measurement_reports_lidds_style_metrics():
  perf = PerformanceMeasurement()
  perf.set_threshold(0.7)
  for event in [
    PerformanceEvent(False, False, "attack-hit", 1),
    PerformanceEvent(True, True, "attack-hit", 2_000_000_000),
    PerformanceEvent(True, True, "attack-hit", 3_000_000_000),
    PerformanceEvent(False, False, "attack-miss", 4),
    PerformanceEvent(False, True, "attack-miss", 5_000_000_000),
    PerformanceEvent(False, False, "benign-clean", 6),
    PerformanceEvent(True, False, "benign-noisy", 7),
    PerformanceEvent(False, False, "benign-noisy", 8),
  ]:
    perf.add_event(event)

  summary = perf.get_results()

  assert summary["threshold"] == pytest.approx(0.7)
  assert summary["true_positives"] == 2
  assert summary["false_positives"] == 1
  assert summary["true_negatives"] == 4
  assert summary["false_negatives"] == 1
  assert summary["correct_alarm_count"] == 1
  assert summary["false_alarm_count"] == 1
  assert summary["exploit_count"] == 2
  assert summary["detection_rate"] == pytest.approx(0.5)
  assert summary["precision_syscall"] == pytest.approx(2 / 3)
  assert summary["recall_syscall"] == pytest.approx(2 / 3)
  assert summary["f1_syscall"] == pytest.approx(2 / 3)
  assert summary["consecutive_false_positives_normal"] == 1
  assert summary["consecutive_false_positives_exploits"] == 0
  assert summary["precision_with_cfa"] == pytest.approx(0.5)
  assert summary["precision_with_syscalls"] == pytest.approx(0.5)
  assert summary["f1_cfa"] == pytest.approx(0.5)
  assert summary["percent_events_flagged"] == pytest.approx(37.5)
  assert summary["percent_events_gt_anomalous"] == pytest.approx(37.5)
  assert summary["mean_detection_delay_s"] == pytest.approx(0.0)
  assert summary["metric_levels"]["precision_with_syscalls"] == "mixed"


def test_default_summary_path_uses_sidecar_json_name(tmp_path):
  out_path = tmp_path / "scores.jsonl"

  summary_path = _default_summary_path(out_path)

  assert summary_path == tmp_path / "scores.summary.json"


def test_main_writes_anomaly_and_reports_confusion(tmp_path, monkeypatch, caplog):
  log_file = tmp_path / "events.jsonl"
  checkpoint = tmp_path / "checkpoint.pkl"
  out_file = tmp_path / "scores.jsonl"
  summary_file = tmp_path / "scores.summary.json"
  checkpoint.write_bytes(
    pickle.dumps(
      {
        "format": "building_blocks_v1",
        "pipeline_id": "single_model",
        "checkpoint_index": 3,
        "blocks": {},
        "extra": {
          "detector_config": detector_config_to_dict(
            DetectorConfig(
              pipeline_id="single_model",
              model_algorithm="zscore",
              threshold=0.7,
              score_mode="scaled",
              suppress_anomalies_during_warmup=False,
              warmup_events=0,
            )
          )
        },
      }
    )
  )
  _write_jsonl(
    log_file,
    [
      {
        "event_id": "evt-1",
        "event_group": "g",
        "syscall_name": "openat",
        "ts_unix_nano": 1,
        "malicious": False,
        "lidds_recording_name": "benign-rec",
      },
      {
        "event_id": "evt-2",
        "event_group": "g",
        "syscall_name": "read",
        "ts_unix_nano": 2,
        "malicious": False,
        "lidds_recording_name": "attack-rec",
      },
      {
        "event_id": "evt-3",
        "event_group": "g",
        "syscall_name": "write",
        "ts_unix_nano": 3,
        "malicious": True,
        "lidds_recording_name": "attack-rec",
      },
    ],
  )

  class FakeIDS:
    def __init__(self):
      self._scores = iter([(0.2, 0.4), (0.3, 0.2), (2.0, 0.9)])
      self.manager = object()

    def run_event(self, evt):
      raw, scaled = next(self._scores)
      return DecisionOutput(
        raw=float(raw),
        scaled=float(scaled),
        primary=float(scaled),
        score_mode="scaled",
        suppressed=False,
        threshold=0.7,
        anomaly=float(scaled) >= 0.7,
      )

  seen_cfg = {}

  def _build_final_bb(cfg):
    seen_cfg["pipeline_id"] = cfg.pipeline_id
    seen_cfg["threshold"] = cfg.threshold
    seen_cfg["score_mode"] = cfg.score_mode
    return object()

  monkeypatch.setattr(score_from_checkpoint, "build_final_bb", _build_final_bb)
  monkeypatch.setattr(score_from_checkpoint, "OnlineIDS", lambda final_bb, pipeline_id: FakeIDS())
  monkeypatch.setattr(score_from_checkpoint, "load_pipeline_checkpoint", lambda path, manager: None)
  monkeypatch.setattr(
    score_from_checkpoint,
    "_dict_to_event_envelope",
    lambda obj: SimpleNamespace(
      event_id=obj["event_id"],
      event_group=obj.get("event_group", ""),
      syscall_name=obj["syscall_name"],
    ),
  )
  monkeypatch.setattr(
    "sys.argv",
    [
      "score_from_checkpoint.py",
      str(log_file),
      "--checkpoint",
      str(checkpoint),
      "--out",
      str(out_file),
    ],
  )

  caplog.set_level(logging.INFO)
  score_from_checkpoint.main()

  rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
  summary = json.loads(summary_file.read_text(encoding="utf-8"))
  assert seen_cfg == {"pipeline_id": "single_model", "threshold": 0.7, "score_mode": "scaled"}
  assert [row["anomaly"] for row in rows] == [False, False, True]
  assert [row["expected_anomaly"] for row in rows] == [False, False, True]
  assert [row["score_primary"] for row in rows] == [pytest.approx(0.4), pytest.approx(0.2), pytest.approx(0.9)]
  assert all(row["score_mode"] == "scaled" for row in rows)
  assert all(row["suppress_primary"] is False for row in rows)
  assert all(row["threshold"] == pytest.approx(0.7) for row in rows)
  assert summary["out"] == str(out_file)
  assert summary["summary_out"] == str(summary_file)
  assert summary["score_mode"] == "scaled"
  assert summary["events_scored"] == 3
  assert summary["predicted_anomalies"] == 1
  assert summary["labeled_events"] == 3
  assert summary["performance"]["true_positives"] == 1
  assert summary["performance"]["true_negatives"] == 2
  assert summary["performance"]["correct_alarm_count"] == 1
  assert summary["performance"]["detection_rate"] == pytest.approx(1.0)
  assert summary["performance"]["precision_syscall"] == pytest.approx(1.0)
  assert any("LID-DS-style performance evaluation:" in record.message for record in caplog.records)


def test_main_uses_checkpoint_config_not_live_environment(tmp_path, monkeypatch):
  log_file = tmp_path / "events.jsonl"
  checkpoint = tmp_path / "checkpoint.pkl"
  out_file = tmp_path / "scores.jsonl"
  _write_jsonl(
    log_file,
    [
      {
        "event_id": "evt-1",
        "event_group": "g",
        "syscall_name": "openat",
        "ts_unix_nano": 1,
        "malicious": False,
      },
    ],
  )
  checkpoint.write_bytes(
    pickle.dumps(
      {
        "format": "building_blocks_v1",
        "pipeline_id": "sequence_mlp",
        "checkpoint_index": 1,
        "blocks": {},
        "extra": {
          "detector_config": detector_config_to_dict(
            DetectorConfig(
              pipeline_id="sequence_mlp",
              threshold=0.25,
              score_mode="percentile",
              warmup_events=3,
            )
          )
        },
      }
    )
  )

  seen_cfg = {}

  def _build_final_bb(cfg):
    seen_cfg["pipeline_id"] = cfg.pipeline_id
    seen_cfg["threshold"] = cfg.threshold
    seen_cfg["score_mode"] = cfg.score_mode
    seen_cfg["warmup_events"] = cfg.warmup_events
    return object()

  class FakeIDS:
    def __init__(self):
      self.manager = object()

    def run_event(self, evt):
      return DecisionOutput(
        raw=1.0,
        scaled=0.5,
        primary=0.5,
        score_mode="percentile",
        suppressed=False,
        threshold=0.25,
        anomaly=True,
      )

  monkeypatch.setattr(score_from_checkpoint, "build_final_bb", _build_final_bb)
  monkeypatch.setattr(score_from_checkpoint, "OnlineIDS", lambda final_bb, pipeline_id: FakeIDS())
  monkeypatch.setattr(score_from_checkpoint, "load_pipeline_checkpoint", lambda path, manager: None)
  monkeypatch.setattr(
    score_from_checkpoint,
    "_dict_to_event_envelope",
    lambda obj: SimpleNamespace(
      event_id=obj["event_id"],
      event_group=obj.get("event_group", ""),
      syscall_name=obj["syscall_name"],
    ),
  )
  monkeypatch.setattr(
    "sys.argv",
    [
      "score_from_checkpoint.py",
      str(log_file),
      "--checkpoint",
      str(checkpoint),
      "--out",
      str(out_file),
    ],
  )

  score_from_checkpoint.main()

  assert seen_cfg == {
    "pipeline_id": "sequence_mlp",
    "threshold": 0.25,
    "score_mode": "percentile",
    "warmup_events": 3,
  }
