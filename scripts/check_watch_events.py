#!/usr/bin/env python3
"""Debug script: search for matching events in live detector or historical dataset.

Current events (live detector):
  DETECTOR_EVENTS_URL=http://localhost:50052/recent_events python scripts/check_watch_events.py [filter]
  # With port-forward: kubectl port-forward svc/sentinel-ebpf-detector 50052:50052

Historical dataset (event dump file):
  python scripts/check_watch_events.py events_09_03_26.jsonl [filter]
  tail -n 50000 dataset.jsonl | python scripts/check_watch_events.py - [filter]
"""
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

URL = os.environ.get("DETECTOR_EVENTS_URL", "")
LIMIT = 500


def load_from_file(path: Path, tail_lines: int | None = None) -> list:
    entries = []
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines.append(line)
            if tail_lines and len(lines) > tail_lines:
                lines.pop(0)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            if "event_id" in obj or "syscall_nr" in obj:
                entries.append(obj)
        except json.JSONDecodeError:
            pass
    return entries


def get_comm_path(e: dict) -> tuple[str, str]:
    comm = str(e.get("comm") or "")
    attrs = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
    path = str(attrs.get("fd_path") or e.get("fd_path") or "")
    return comm, path


def matches_filter(e: dict, filter_parts: list[str]) -> bool:
    comm, path = get_comm_path(e)
    event_name = (e.get("syscall_name") or "").lower()
    comm_l = comm.lower()
    path_l = path.lower()
    for part in filter_parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip().lower(), v.strip().lower()
            if k == "comm" and v not in comm_l:
                return False
            if k == "fd_path" and v not in path_l:
                return False
            if k in ("event", "event_name", "syscall", "syscall_name") and v not in event_name:
                return False
        elif part.lower() not in f"{event_name} {comm_l} {path_l}":
            return False
    return True


def main():
    filter_parts = []
    log_path = None
    tail_lines = 50000  # For large files, only scan last N lines
    for arg in sys.argv[1:]:
        if arg.startswith("--tail="):
            tail_lines = int(arg.split("=", 1)[1])
        elif arg == "-":
            log_path = Path("/dev/stdin")
        elif "=" in arg:
            filter_parts.append(arg)
        elif Path(arg).exists():
            log_path = Path(arg)
        else:
            filter_parts.append(arg)
    if not filter_parts:
        filter_parts = ["fd_path=/etc"]

    entries = []
    if log_path:
        if str(log_path) == "-":
            entries = load_from_file(Path("/dev/stdin"), tail_lines=tail_lines)
        elif log_path.exists():
            entries = load_from_file(log_path, tail_lines=tail_lines)
        print(f"Loaded {len(entries)} events from {log_path}")
    elif URL:
        fetch_url = f"{URL}?limit={LIMIT}" if "?" not in URL else f"{URL}&limit={LIMIT}"
        try:
            req = Request(fetch_url, headers={"User-Agent": "sentinel-check"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            entries = data.get("entries", [])
            print(f"Fetched {len(entries)} events from detector")
        except Exception as e:
            print(f"Fetch failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage:", file=sys.stderr)
        print("  Live detector: DETECTOR_EVENTS_URL=http://host:50052/recent_events python check_watch_events.py [filter]", file=sys.stderr)
        print("  Dataset file:  python check_watch_events.py <events.jsonl> [filter]", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("No events.")
        return

    matches = [e for e in entries if matches_filter(e, filter_parts)]
    print(f"\nMatches for {filter_parts!r}: {len(matches)}")
    for m in matches[-5:]:
        comm, path = get_comm_path(m)
        print(f"  {m.get('syscall_name')} comm={comm!r} path={path!r}")


if __name__ == "__main__":
    main()
