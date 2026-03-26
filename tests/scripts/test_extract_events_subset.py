"""Tests for scripts/extract_events_subset.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_extract_renames_event_type_to_event_group_and_strips_unused_fields(temp_dir):
    """Output has event_group, no event_type, only replay fields."""
    inp = temp_dir / "in.jsonl"
    out = temp_dir / "out.jsonl"
    events = [
        {
            "event_id": "e1",
            "event_name": "openat",
            "event_type": "file",
            "syscall_nr": 257,
            "comm": "bash",
            "pid": "1",
            "tid": "2",
            "uid": "1000",
            "arg0": "0",
            "arg1": "0",
            "path": "/tmp/x",
            "hostname": "h1",
            "pod_name": "p1",
            "namespace": "ns",
            "container_id": "c1",
            "ts_unix_nano": 1700000000000000000,
            "attributes": {"a": "b"},
            "timestamp": "ignored",
            "anomaly": False,
            "score": 0.5,
        },
    ]
    with inp.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    repo = Path(__file__).resolve().parent.parent.parent
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "extract_events_subset.py"), str(inp), "-o", str(out), "-n", "10"],
        check=True,
    )

    with out.open() as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "event_group" in obj
    assert obj["event_group"] == "file"
    assert "event_type" not in obj
    assert set(obj.keys()) == {
        "event_id",
        "hostname",
        "pod_name",
        "namespace",
        "container_id",
        "ts_unix_nano",
        "event_name",
        "event_group",
        "syscall_nr",
        "comm",
        "pid",
        "tid",
        "uid",
        "arg0",
        "arg1",
        "path",
        "attributes",
    }


def test_extract_skips_metadata_lines(temp_dir):
    """Lines without event_id are skipped."""
    inp = temp_dir / "in.jsonl"
    out = temp_dir / "out.jsonl"
    with inp.open("w") as f:
        f.write(json.dumps({"_meta": True, "config": {}}) + "\n")
        f.write(
            json.dumps(
                {
                    "event_id": "e1",
                    "event_name": "openat",
                    "event_type": "file",
                    "syscall_nr": 0,
                    "ts_unix_nano": 0,
                }
            )
            + "\n"
        )

    repo = Path(__file__).resolve().parent.parent.parent
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "extract_events_subset.py"), str(inp), "-o", str(out), "-n", "10"],
        check=True,
    )

    with out.open() as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["event_id"] == "e1"


def test_extract_respects_line_limit(temp_dir):
    """Only first N event lines are written."""
    inp = temp_dir / "in.jsonl"
    out = temp_dir / "out.jsonl"
    with inp.open("w") as f:
        for i in range(10):
            f.write(json.dumps({"event_id": f"e{i}", "event_type": "file", "syscall_nr": 0, "ts_unix_nano": i}) + "\n")

    repo = Path(__file__).resolve().parent.parent.parent
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "extract_events_subset.py"), str(inp), "-o", str(out), "-n", "3"],
        check=True,
    )

    with out.open() as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 3
