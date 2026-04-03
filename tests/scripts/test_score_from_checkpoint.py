"""Tests for scripts/score_from_checkpoint.py."""

import json
import logging
from types import SimpleNamespace

import pytest

import scripts.score_from_checkpoint as score_from_checkpoint
from scripts.score_from_checkpoint import (
  _confusion_summary,
  _default_summary_path,
  _iter_event_dicts,
  _recording_detection_summary,
)


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


def test_confusion_summary_counts_binary_outcomes():
  summary = _confusion_summary(
    [
      (True, True),
      (True, False),
      (False, False),
      (False, True),
    ]
  )

  assert summary["tp"] == 1
  assert summary["fp"] == 1
  assert summary["tn"] == 1
  assert summary["fn"] == 1
  assert summary["samples"] == 4
  assert summary["flagged_samples"] == 2
  assert summary["precision"] == pytest.approx(0.5)
  assert summary["recall"] == pytest.approx(0.5)
  assert summary["f1"] == pytest.approx(0.5)
  assert summary["accuracy"] == pytest.approx(0.5)
  assert summary["flagged_rate"] == pytest.approx(0.5)


def test_recording_detection_summary_counts_attack_recordings_once():
  summary = _recording_detection_summary(
    [
      ("attack-hit", False, False),
      ("attack-hit", True, True),
      ("attack-hit", True, True),
      ("attack-miss", False, False),
      ("attack-miss", False, True),
      ("benign-clean", False, False),
      ("benign-noisy", True, False),
      ("benign-noisy", False, False),
    ]
  )

  assert summary["recordings"] == 4
  assert summary["attack_recordings"] == 2
  assert summary["benign_recordings"] == 2
  assert summary["detected_attack_recordings"] == 1
  assert summary["missed_attack_recordings"] == 1
  assert summary["benign_recordings_with_alarm"] == 1
  assert summary["detection_rate"] == pytest.approx(0.5)
  assert summary["benign_recording_alarm_rate"] == pytest.approx(0.5)


def test_default_summary_path_uses_sidecar_json_name(tmp_path):
  out_path = tmp_path / "scores.jsonl"

  summary_path = _default_summary_path(out_path)

  assert summary_path == tmp_path / "scores.summary.json"


def test_main_writes_anomaly_and_reports_confusion(tmp_path, monkeypatch, caplog):
  log_file = tmp_path / "events.jsonl"
  checkpoint = tmp_path / "checkpoint.pkl"
  out_file = tmp_path / "scores.jsonl"
  summary_file = tmp_path / "scores.summary.json"
  checkpoint.write_text("checkpoint", encoding="utf-8")
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

  class FakeDetector:
    def __init__(self):
      self._scores = iter([(0.2, 0.4), (0.3, 0.2), (2.0, 0.9)])

    def load_checkpoint(self, path):
      assert path == checkpoint

    def score_and_learn_event(self, evt, feature_fn):
      feature_fn(evt)
      return next(self._scores)

  monkeypatch.setattr(score_from_checkpoint, "_make_detector", lambda algorithm: (FakeDetector(), lambda evt: {}))
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
    score_from_checkpoint,
    "load_config",
    lambda: SimpleNamespace(
      threshold=0.7,
      score_mode="scaled",
      suppress_anomalies_during_warmup=False,
      warmup_events=0,
      percentile_window_size=2048,
      percentile_warmup=128,
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
  assert summary["event_metrics"]["tp"] == 1
  assert summary["event_metrics"]["tn"] == 2
  assert summary["recording_metrics"]["detected_attack_recordings"] == 1
  assert summary["recording_metrics"]["detection_rate"] == pytest.approx(1.0)
  assert any('"detected_attack_recordings": 1' in record.message for record in caplog.records)
  assert any('"detection_rate": 1.0' in record.message for record in caplog.records)
  assert any('"benign_recordings_with_alarm": 0' in record.message for record in caplog.records)
  assert any('"flagged_samples": 1' in record.message for record in caplog.records)
  assert any('"flagged_rate": 0.3333333333333333' in record.message for record in caplog.records)
  assert any('"tp": 1' in record.message for record in caplog.records)
  assert any('"tn": 2' in record.message for record in caplog.records)
