"""Tests for scripts/model_diagnostic.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _make_synthetic_events(n: int, base_ts_ns: int = 1700000000000000000) -> list[dict]:
  """Generate synthetic JSONL events that produce valid features for memstream/loda_ema."""
  events = []
  for i in range(n):
    ts = base_ts_ns + i * 1_000_000
    is_shifted = i >= int(n * 0.8)
    event_name = "openat" if i % 3 != 0 else "read"
    syscall_id = 257 if event_name == "openat" else 0
    comm = "bash" if i % 5 == 0 else "sshd"
    uid = "1000"
    arg0 = "-100"
    arg1 = "524288"
    path = f"/tmp/file_{i % 100}.txt"
    flags = "524288"
    return_value = "3"
    flags = "O_RDONLY"

    if is_shifted:
      comm = "python" if i % 2 == 0 else "curl"
      uid = "0"
      arg0 = str(4096 + (i % 17) * 256)
      arg1 = str(2048 + (i % 11) * 128)
      path = f"/root/.ssh/id_{i % 9}" if i % 2 == 0 else f"/etc/shadow.d/secret_{i % 13}"
      flags = "2101248"
      return_value = "-13" if i % 4 == 0 else "7"
      flags = "O_CREAT|O_WRONLY|O_TRUNC"
    if event_name == "openat":
      arg1 = flags
    events.append({
      "event_id": f"test-{i:06d}",
      "event_name": event_name,
      "event_group": "file",
      "event_type": "file",
      "ts_unix_nano": ts,
      "syscall_nr": syscall_id,
      "comm": comm,
      "pid": str(1000 + i),
      "tid": str(1000 + i),
      "uid": uid,
      "arg0": arg0,
      "arg1": arg1,
      "path": path,
      "hostname": "test-host",
      "pod_name": "test-pod",
      "namespace": "default",
      "container_id": "",
      "attributes": {"flags": flags, "return_value": return_value},
    })
  return events


@pytest.mark.slow
def test_model_diagnostic_produces_meaningful_results(temp_dir):
  """Run model_diagnostic on a few thousand synthetic events; assert outputs are meaningful."""
  events = _make_synthetic_events(3500)
  events_path = temp_dir / "synthetic_events.jsonl"
  with events_path.open("w", encoding="utf-8") as f:
    for evt in events:
      f.write(json.dumps(evt) + "\n")

  out_dir = temp_dir / "diagnostic_out"
  repo_root = Path(__file__).resolve().parent.parent.parent
  script = repo_root / "scripts" / "model_diagnostic.py"

  for algorithm in ("memstream", "loda_ema"):
    proc = subprocess.run(
      [
        sys.executable,
        str(script),
        "--algorithm",
        algorithm,
        "--limit",
        "3000",
        "--out-dir",
        str(out_dir / algorithm),
        "--window-size",
        "500",
        str(events_path),
      ],
      cwd=str(repo_root),
      capture_output=True,
      text=True,
      timeout=120,
    )
    assert proc.returncode == 0, f"{algorithm}: stderr={proc.stderr!r}"

    alg_out = out_dir / algorithm
    assert alg_out.exists(), f"{algorithm}: out dir missing"

    # Score summary
    summary_path = alg_out / "score_summary.json"
    assert summary_path.exists(), f"{algorithm}: score_summary.json missing"
    summary = json.loads(summary_path.read_text())
    assert summary["algorithm"] == algorithm
    assert summary["events_processed"] == 3000
    assert summary["analysis_start_event"] >= 0
    assert 1 <= summary["events_analyzed"] <= 3000
    assert "raw" in summary and "scaled" in summary
    raw = summary["raw"]
    assert raw["min"] is not None and raw["max"] is not None
    assert 0 <= raw["min"] <= raw["max"]
    scaled = summary["scaled"]
    assert 0 <= scaled["min"] <= 1.0 and 0 <= scaled["max"] <= 1.0
    post_warmup = summary["post_warmup"]
    assert post_warmup["scaled"]["std"] is not None
    assert len(summary["anomaly_rates_by_window"]["windows"]) == 6

    # Model diagnostics
    diag_path = alg_out / "model_diagnostics.json"
    assert diag_path.exists(), f"{algorithm}: model_diagnostics.json missing"
    diag = json.loads(diag_path.read_text())
    assert diag["algorithm"] == algorithm
    assert diag["events_processed"] == 3000
    assert diag["analysis_start_event"] == summary["analysis_start_event"]
    assert diag["events_analyzed"] == summary["events_analyzed"]
    assert diag["score_vs_model_signal_corr"] is not None

    sample_events_path = alg_out / "sample_events.json"
    assert sample_events_path.exists(), f"{algorithm}: sample_events.json missing"
    sample_events = json.loads(sample_events_path.read_text())
    assert sample_events
    for sample in sample_events.values():
      assert sample["event_index"] >= summary["analysis_start_event"]

    # Timeseries (one line per event)
    ts_path = alg_out / "score_timeseries.jsonl"
    assert ts_path.exists(), f"{algorithm}: score_timeseries.jsonl missing"
    lines = ts_path.read_text().strip().split("\n")
    assert len(lines) == 3000, f"{algorithm}: expected 3000 timeseries lines, got {len(lines)}"

    first_record = json.loads(lines[0])
    last_record = json.loads(lines[-1])
    assert "score_raw" in first_record and "score_scaled" in first_record
    assert 0 <= first_record["score_scaled"] <= 1.0
    assert 0 <= last_record["score_scaled"] <= 1.0

    if algorithm == "memstream":
      assert "memory_error" in first_record
      assert "update_allowed" in first_record
      assert "mem_filled_after" in first_record
      assert 0.0 <= diag["accepted_update_rate"] <= 1.0
      assert 0.0 <= diag["rejected_update_rate"] <= 1.0
      assert "memory_error" in diag["last_debug"]
    else:
      assert "max_projection_excess" in first_record
      assert "mean_projection_excess" in first_record
      assert last_record["max_projection_excess"] >= last_record["mean_projection_excess"]
      assert abs(last_record["model_signal"] - last_record["max_projection_excess"]) < 1e-6
      assert "max_projection_excess" in diag["last_debug"]

    # Report
    report_path = alg_out / "diagnostic_report.md"
    assert report_path.exists(), f"{algorithm}: diagnostic_report.md missing"
    report = report_path.read_text()
    assert "Interpretation" in report or "score" in report.lower()
    assert "Analysis start event" in report
