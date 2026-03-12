"""Tests for scripts/dataset_stats.py."""

import json
from pathlib import Path

import pytest

from scripts.dataset_stats import (
  _count_duplicates,
  _record_hash,
  _record_without,
  _score_stats,
  load_records,
)


class TestDatasetStats:
  """Test dataset_stats helpers and report shape."""

  def test_load_records_empty_file(self, temp_dir):
    p = temp_dir / "empty.jsonl"
    p.write_text("")
    records, scores = load_records(p)
    assert records == []
    assert scores == []

  def test_load_records_skips_lines_without_score(self, temp_dir):
    p = temp_dir / "mixed.jsonl"
    p.write_text('{"event_id": "1"}\n{"event_id": "2", "score": 0.5}\n')
    records, scores = load_records(p)
    assert len(records) == 1
    assert scores == [0.5]

  def test_load_records_with_anomaly(self, temp_dir):
    p = temp_dir / "with_anomaly.jsonl"
    lines = [
      '{"event_id": "a", "score": 0.3, "anomaly": false}\n',
      '{"event_id": "b", "score": 0.9, "anomaly": true}\n',
    ]
    p.write_text("".join(lines))
    records, scores = load_records(p)
    assert len(records) == 2
    assert records[0]["anomaly"] is False
    assert records[1]["anomaly"] is True
    assert scores == [0.3, 0.9]

  def test_score_stats_empty(self):
    assert _score_stats([]) == {}

  def test_score_stats_single(self):
    out = _score_stats([1.0])
    assert out["count"] == 1
    assert out["min"] == out["max"] == out["mean"] == 1.0
    assert out["p50"] == out["p95"] == out["p99"] == 1.0

  def test_score_stats_many(self):
    scores = [0.1 * i for i in range(11)]  # 0, 0.1, ..., 1.0
    out = _score_stats(scores)
    assert out["count"] == 11
    assert out["min"] == 0.0
    assert out["max"] == 1.0
    assert out["mean"] == pytest.approx(0.5)
    assert out["p50"] == pytest.approx(0.5)
    # p99 with n=11 uses index int(0.99*10)=9 -> 0.9
    assert out["p99"] == pytest.approx(0.9)

  def test_record_hash_deterministic(self):
    r = {"a": 1, "b": 2}
    assert _record_hash(r) == _record_hash(r)
    assert _record_hash(r) == _record_hash({"b": 2, "a": 1})

  def test_record_without(self):
    r = {"a": 1, "b": 2, "c": 3}
    assert _record_without(r, ("b",)) == {"a": 1, "c": 3}
    assert _record_without(r, ("a", "c")) == {"b": 2}

  def test_count_duplicates_exact_only(self):
    records = [
      {"x": 1, "score": 0.5},
      {"x": 1, "score": 0.5},
      {"x": 2, "score": 0.5},
    ]
    exact, one, two, three = _count_duplicates(records, ["x", "score"], max_diffs=1)
    assert exact == 1  # one duplicate pair
    assert one is two is three is None

  def test_count_duplicates_zero_diffs(self):
    records = [{"x": 1, "score": 0.5}]
    exact, one, two, three = _count_duplicates(records, ["x", "score"], max_diffs=0)
    assert exact is one is two is three is None

  def test_report_json_shape_and_truly_abnormal(self, temp_dir):
    # 60 anomalies, 40 non-anomalies; all same pattern (only event_id/score differ) -> pattern_count=100 for each anomaly -> common
    lines = []
    for i in range(100):
      anomaly = i < 60
      lines.append(json.dumps({"event_id": str(i), "score": 0.9 if anomaly else 0.2, "anomaly": anomaly}) + "\n")
    p = temp_dir / "replay.jsonl"
    p.write_text("".join(lines))
    out_json = temp_dir / "report.json"

    import scripts.dataset_stats as ds
    orig_argv = __import__("sys").argv
    try:
      __import__("sys").argv = ["dataset_stats.py", str(p), "--out", str(out_json), "--diffs", "0"]
      ds.main()
    finally:
      __import__("sys").argv = orig_argv

    report = json.loads(out_json.read_text())
    assert report["total_events"] == 100
    assert report["anomalies"] == 60
    assert report["fraction_anomalies"] == 0.6
    assert "anomaly_pattern_frequency" in report
    assert report["anomaly_pattern_frequency"]["count"] == 60
    assert report["anomaly_pattern_frequency"]["median"] == 100  # same pattern for all
    assert "truly_abnormal_interpretation" in report
    assert "score_all" in report
    assert "score_anomaly" in report
    assert "score_non_anomaly" in report
    assert report["score_anomaly"]["count"] == 60
    assert report["score_non_anomaly"]["count"] == 40

  def test_report_flagged_events_common_pattern(self, temp_dir):
    # 5 anomalies out of 1000, but same pattern as rest -> pattern_count=1000 for each anomaly -> mostly_common
    lines = []
    for i in range(1000):
      anomaly = i < 5
      lines.append(json.dumps({"event_id": str(i), "score": 0.95 if anomaly else 0.3, "anomaly": anomaly}) + "\n")
    p = temp_dir / "rare.jsonl"
    p.write_text("".join(lines))
    out_json = temp_dir / "report.json"

    import scripts.dataset_stats as ds
    orig_argv = __import__("sys").argv
    try:
      __import__("sys").argv = ["dataset_stats.py", str(p), "--out", str(out_json), "--diffs", "0"]
      ds.main()
    finally:
      __import__("sys").argv = orig_argv

    report = json.loads(out_json.read_text())
    assert report["fraction_anomalies"] == pytest.approx(0.005)
    assert report["truly_abnormal_interpretation"] == "mostly_common"
    assert report["anomaly_pattern_frequency"]["fraction_pattern_common_gt100"] == 1.0

  def test_report_flagged_events_truly_rare(self, temp_dir):
    # Flagged events each have a unique pattern (different event_name) -> pattern_count=1 -> mostly_rare
    lines = []
    for i in range(100):
      anomaly = i < 10
      rec = {"event_id": str(i), "event_name": f"evt_{i}", "score": 0.95 if anomaly else 0.3, "anomaly": anomaly}
      lines.append(json.dumps(rec) + "\n")
    p = temp_dir / "rare_patterns.jsonl"
    p.write_text("".join(lines))
    out_json = temp_dir / "report.json"

    import scripts.dataset_stats as ds
    orig_argv = __import__("sys").argv
    try:
      __import__("sys").argv = ["dataset_stats.py", str(p), "--out", str(out_json), "--diffs", "0"]
      ds.main()
    finally:
      __import__("sys").argv = orig_argv

    report = json.loads(out_json.read_text())
    assert report["anomalies"] == 10
    assert report["truly_abnormal_interpretation"] == "mostly_rare"
    assert report["anomaly_pattern_frequency"]["fraction_pattern_appears_once"] == 1.0
    assert report["anomaly_pattern_frequency"]["median"] == 1
