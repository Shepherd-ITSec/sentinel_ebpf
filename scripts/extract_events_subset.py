#!/usr/bin/env python3
"""Extract first N event lines from a JSONL file for model_diagnostic replay.

Renames event_type -> event_group and keeps only fields used by replay_logs /
model_diagnostic._dict_to_event_envelope.
"""

import argparse
import json
import sys

def _to_replay_event(obj: dict, from_key: str, to_key: str) -> dict:
    """Build minimal event dict for replay, with from_key renamed to to_key."""
    event_group = obj.get(to_key) or obj.get(from_key) or ""
    pod_name = obj.get("pod_name") or obj.get("pod") or ""
    data = obj.get("data", [])
    if not isinstance(data, list):
        data = []
    attributes = obj.get("attributes") or {}
    if not isinstance(attributes, dict):
        attributes = {}

    return {
        "event_id": obj.get("event_id", ""),
        "hostname": obj.get("hostname", ""),
        "pod_name": pod_name,
        "namespace": obj.get("namespace", ""),
        "container_id": obj.get("container_id", ""),
        "ts_unix_nano": int(obj.get("ts_unix_nano", 0)),
        "event_name": obj.get("event_name", "") or (data[0] if data else ""),
        "event_group": event_group,
        "data": data,
        "attributes": attributes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract first N event lines for model_diagnostic replay; rename event_type -> event_group"
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="events_17_03_26.jsonl",
        help="Input JSONL file (default: events_17_03_26.jsonl)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="events_17_03_26_1M.jsonl",
        help="Output JSONL file (default: events_17_03_26_1M.jsonl)",
    )
    ap.add_argument(
        "-n",
        "--lines",
        type=int,
        default=1_000_000,
        help="Number of event lines to extract (default: 1_000_000)",
    )
    ap.add_argument(
        "--from-key",
        default="event_type",
        help="Source key to rename to event_group (default: event_type)",
    )
    args = ap.parse_args()

    written = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            if written >= args.lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"json decode error: {e}", file=sys.stderr)
                continue
            if "event_id" not in obj:
                continue
            out = _to_replay_event(obj, args.from_key, "event_group")
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if written % 100_000 == 0:
                print(f"Written {written:,} events...", file=sys.stderr)

    print(f"Extracted {written:,} events to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
