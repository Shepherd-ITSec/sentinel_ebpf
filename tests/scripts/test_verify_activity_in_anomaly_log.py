"""Tests for verify_activity_in_anomaly_log.py."""

import json

from scripts.verify_activity_in_anomaly_log import (
  _extract_path,
  _load_all_events,
  _load_anomaly_entries,
  _path_matches,
  _which_pattern,
  verify,
  verify_with_scores,
)


def _evt(*, event_id: str = "x", **kwargs) -> str:
  base = {
    "event_id": event_id,
    "syscall_name": "openat",
    "syscall_nr": 257,
    "comm": "cat",
    "pid": "1",
    "tid": "2",
    "uid": "0",
    "arg0": "0",
    "arg1": "0",
    "attributes": {"fd_path": "/etc/passwd"},
  }
  if "path" in kwargs:
    base["attributes"]["fd_path"] = kwargs.pop("path")
  if "fd_path" in kwargs:
    base["attributes"]["fd_path"] = kwargs.pop("fd_path")
  base.update(kwargs)
  return json.dumps(base, separators=(",", ":"))


def test_script_extract_path():
  """Path is the path field."""
  entry = json.loads(_evt(path="/etc/passwd"))
  assert _extract_path(entry) == "/etc/passwd"

  assert _extract_path({}) == ""
  assert _extract_path({"attributes": {"fd_path": ""}}) == ""


def test_script_path_matches():
  assert _path_matches("/etc/passwd", "/etc/passwd")
  assert _path_matches("/tmp/sentinel-test-1.txt", "/tmp/sentinel-test-*.txt")
  assert _path_matches("/proc/12345/status", "/proc/*/status")
  assert not _path_matches("/etc/other", "/etc/passwd")
  assert not _path_matches("", "/etc/passwd")


def test_script_load_anomaly_entries_anomaly_log(tmp_path):
  """Load from anomaly log JSONL."""
  log = tmp_path / "anomalies.jsonl"
  log.write_text(
    _evt(event_id="a", path="/etc/passwd") + "\n" + _evt(event_id="b", path="/etc/shadow") + "\n"
  )
  entries = _load_anomaly_entries(log, None)
  assert len(entries) == 2
  assert _extract_path(entries[0]) == "/etc/passwd"
  assert _extract_path(entries[1]) == "/etc/shadow"


def test_script_load_anomaly_entries_event_dump(tmp_path):
  """Load from event dump, filter anomaly=True."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    '{"_meta":true}\n'
    + _evt(event_id="a", anomaly=False, path="/tmp/x")
    + "\n"
    + _evt(event_id="b", anomaly=True, path="/etc/passwd")
    + "\n"
  )
  entries = _load_anomaly_entries(None, dump)
  assert len(entries) == 1
  assert _extract_path(entries[0]) == "/etc/passwd"


def test_script_load_anomaly_log_preferred_over_event_dump(tmp_path):
  """When both exist and anomaly log has content, use anomaly log."""
  log = tmp_path / "anomalies.jsonl"
  log.write_text(_evt(event_id="x", comm="x", path="/etc/passwd") + "\n")
  dump = tmp_path / "events.jsonl"
  dump.write_text(_evt(event_id="y", comm="y", anomaly=True, path="/etc/shadow") + "\n")
  entries = _load_anomaly_entries(log, dump)
  assert len(entries) == 1
  assert _extract_path(entries[0]) == "/etc/passwd"


def test_script_load_fallback_to_event_dump_when_anomaly_log_empty(tmp_path):
  """When anomaly log exists but empty, use event dump."""
  log = tmp_path / "anomalies.jsonl"
  log.write_text("")
  dump = tmp_path / "events.jsonl"
  dump.write_text(_evt(event_id="b", comm="b", anomaly=True, path="/etc/shadow") + "\n")
  entries = _load_anomaly_entries(log, dump)
  assert len(entries) == 1
  assert _extract_path(entries[0]) == "/etc/shadow"


def test_script_verify_matched_unmatched(tmp_path):
  """verify returns matched and unmatched paths."""
  log = tmp_path / "anomalies.jsonl"
  log.write_text(_evt(event_id="a", path="/etc/passwd") + "\n")
  matched, unmatched, entries = verify(
    expected_paths=["/etc/passwd", "/etc/shadow", "/tmp/sentinel-test-*.txt"],
    anomaly_log=log,
  )
  assert matched == {"/etc/passwd"}
  assert unmatched == {"/etc/shadow", "/tmp/sentinel-test-*.txt"}
  assert len(entries) == 1


def test_script_verify_glob_matching(tmp_path):
  """Glob patterns match correctly."""
  log = tmp_path / "anomalies.jsonl"
  log.write_text(
    _evt(event_id="a", path="/tmp/sentinel-test-3.txt")
    + "\n"
    + _evt(event_id="b", path="/proc/42/status")
    + "\n"
  )
  matched, unmatched, _ = verify(
    expected_paths=["/tmp/sentinel-test-*.txt", "/proc/*/status", "/etc/passwd"],
    anomaly_log=log,
  )
  assert "/tmp/sentinel-test-*.txt" in matched
  assert "/proc/*/status" in matched
  assert "/etc/passwd" in unmatched


def test_script_which_pattern():
  assert _which_pattern("/etc/passwd", ["/etc/passwd", "/etc/shadow"]) == "/etc/passwd"
  assert _which_pattern("/tmp/sentinel-test-1.txt", ["/tmp/sentinel-test-*.txt"]) == "/tmp/sentinel-test-*.txt"
  assert _which_pattern("/other/path", ["/etc/passwd"]) is None


def test_script_load_all_events(tmp_path):
  """Load all events from event dump (not just anomalies)."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    '{"_meta":true}\n'
    + _evt(event_id="a", anomaly=False, score=0.1, path="/tmp/x")
    + "\n"
    + _evt(event_id="b", anomaly=True, score=0.9, path="/etc/passwd")
    + "\n"
  )
  entries = _load_all_events(dump)
  assert len(entries) == 2
  assert entries[0]["event_id"] == "a" and entries[0]["score"] == 0.1 and not entries[0]["anomaly"]
  assert entries[1]["event_id"] == "b" and entries[1]["score"] == 0.9 and entries[1]["anomaly"]


def test_script_load_all_events_time_filter(tmp_path):
  """Time filter excludes events outside [after, after+within]."""
  base_ns = 1_700_000_000_000_000_000
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    _evt(event_id="a", ts_unix_nano=base_ns, path="/etc/passwd")
    + "\n"
    + _evt(event_id="b", ts_unix_nano=base_ns + 60_000_000_000, path="/etc/passwd")
    + "\n"
    + _evt(event_id="c", ts_unix_nano=base_ns + 200_000_000_000, path="/etc/passwd")
    + "\n"
  )
  within_s = 120
  entries = _load_all_events(dump, after_ts_ns=base_ns, within_ns=int(within_s * 1e9))
  assert len(entries) == 2
  assert entries[0]["event_id"] == "a"
  assert entries[1]["event_id"] == "b"


def test_script_verify_with_scores_benign_flagged(tmp_path):
  """Benign path incorrectly flagged = false positive."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    _evt(event_id="a", anomaly=True, score=0.85, path="/tmp/sentinel-test-1.txt")
    + "\n"
    + _evt(event_id="b", anomaly=True, score=0.92, path="/etc/passwd")
    + "\n"
  )
  result = verify_with_scores(
    benign_patterns=["/tmp/sentinel-test-*.txt"],
    sensitive_patterns=["/etc/passwd"],
    event_dump=dump,
    anomaly_log=None,
  )
  assert result["benign_flagged"] == 1
  assert result["benign_ok"] == 0
  assert result["sensitive_flagged"] == 1
  assert result["sensitive_missed"] == 0
  assert len(result["events"]) == 2
  benign_evt = next(e for e in result["events"] if e["benign"])
  assert benign_evt["score"] == 0.85
  assert benign_evt["anomaly"] is True


def test_script_verify_with_scores_sequence_filter(tmp_path):
  """Sequence filter excludes events in wrong order (head before cat)."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    _evt(event_id="a", ts_unix_nano=1000, comm="head", path="/etc/passwd")
    + "\n"
    + _evt(event_id="b", ts_unix_nano=2000, comm="cat", path="/etc/passwd")
    + "\n"
    + _evt(event_id="c", ts_unix_nano=3000, comm="ls", path="/etc/passwd")
    + "\n"
  )
  result = verify_with_scores(
    benign_patterns=[],
    sensitive_patterns=["/etc/passwd"],
    event_dump=dump,
    anomaly_log=None,
    sequence_filter=True,
  )
  assert len(result["events"]) == 2
  comms = [e["comm"] for e in result["events"]]
  assert comms == ["cat", "ls"]


def test_script_verify_with_scores_benign_ok(tmp_path):
  """Benign path not flagged = correct."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(
    _evt(event_id="a", anomaly=False, score=0.12, path="/tmp/sentinel-test-1.txt")
    + "\n"
    + _evt(event_id="b", anomaly=True, score=0.88, path="/etc/passwd")
    + "\n"
  )
  result = verify_with_scores(
    benign_patterns=["/tmp/sentinel-test-*.txt"],
    sensitive_patterns=["/etc/passwd"],
    event_dump=dump,
    anomaly_log=None,
  )
  assert result["benign_flagged"] == 0
  assert result["benign_ok"] == 1
  assert result["sensitive_flagged"] == 1
  benign_evt = next(e for e in result["events"] if e["benign"])
  assert benign_evt["score"] == 0.12
  assert benign_evt["anomaly"] is False


def test_script_verify_with_scores_not_found_patterns(tmp_path):
  """Expected paths with no events are reported as not_found_patterns."""
  dump = tmp_path / "events.jsonl"
  dump.write_text(_evt(event_id="a", anomaly=True, score=0.9, path="/etc/passwd") + "\n")
  result = verify_with_scores(
    benign_patterns=["/tmp/sentinel-test-*.txt", "/proc/version"],
    sensitive_patterns=["/etc/passwd", "/etc/shadow"],
    event_dump=dump,
    anomaly_log=None,
  )
  assert result["sensitive_flagged"] == 1
  assert "/etc/passwd" not in result["not_found_patterns"]
  assert "/etc/shadow" in result["not_found_patterns"]
  assert "/tmp/sentinel-test-*.txt" in result["not_found_patterns"]
  assert "/proc/version" in result["not_found_patterns"]
