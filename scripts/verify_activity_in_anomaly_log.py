#!/usr/bin/env python3
"""Verify which generated activity paths appear in the detector's anomaly log.

Reads event dump (all events with scores) or anomaly log, extracts paths from
the canonical data vector [event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags],
reports which paths were flagged as anomalies, outputs scores per event,
and checks whether benign activity was incorrectly flagged (false positives).
"""

import argparse
import csv
import fnmatch
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Canonical data vector: [event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]

# Default: benign paths (expected NOT flagged); sensitive paths (expected flagged)
DEFAULT_BENIGN_PATTERNS = [
    "/tmp/sentinel-test-*.txt",
    "/proc/*/status",
    "/proc/version",
]
DEFAULT_SENSITIVE_PATTERNS = [
    "/etc/passwd",
    "/etc/shadow",
    "/etc/group",
    "/etc/sudoers",
    "/etc/hosts",
    "/etc/ssh/sshd_config",
    "/root/.ssh/id_rsa",
    "/etc/ssl/private",
    "/etc/sentinel-test-write.txt",
]

# Expected (comm,) sequence per path pattern from generate-activity.sh.
# Events must appear in this order to be considered "ours".
EXPECTED_SEQUENCES: Dict[str, List[Tuple[str, ...]]] = {
    "/etc/passwd": [("cat",), ("ls",), ("head",), ("cat",), ("head",)],
    "/etc/shadow": [("cat",), ("ls",), ("head",)],
    "/etc/group": [("cat",), ("ls",), ("head",)],
    "/etc/sudoers": [("cat",), ("ls",), ("head",)],
    "/etc/hosts": [("cat",), ("ls",), ("head",)],
    "/etc/ssh/sshd_config": [("cat",), ("ls",), ("head",)],
    "/root/.ssh/id_rsa": [("cat",), ("ls",), ("head",)],
    "/etc/ssl/private": [("cat",), ("ls",), ("head",)],
    "/etc/sentinel-test-write.txt": [],  # echo/rm - skip for now
    "/tmp/sentinel-test-*.txt": [("cat",)],
    "/proc/*/status": [("head",)],
    "/proc/version": [("cat",)],  # repeated NORMAL_OPS times; we allow multiple
}


def _get_expected_sequence(pattern: str) -> List[Tuple[str, ...]]:
    """Return expected (comm,) sequence for pattern."""
    return EXPECTED_SEQUENCES.get(pattern, [("cat",), ("ls",), ("head",)])


def _filter_by_sequence(events: List[dict], normal_ops: int = 10) -> List[dict]:
    """Keep only events that match expected (comm,) sequence per path, in timestamp order."""
    by_path: Dict[str, List[dict]] = defaultdict(list)
    for e in events:
        by_path[e["path"]].append(e)

    out: List[dict] = []
    for path, path_events in by_path.items():
        path_events.sort(key=lambda x: x.get("ts_unix_nano", 0))
        pattern = path_events[0]["pattern"] if path_events else ""
        seq = _get_expected_sequence(pattern)
        if not seq:
            continue
        idx = 0
        repeat_last = pattern == "/proc/version" and seq == [("cat",)]
        taken = 0
        for evt in path_events:
            comm = (evt.get("comm", "") or "").strip().lower()
            if idx < len(seq):
                expected_comm = seq[idx][0].lower()
                if comm == expected_comm:
                    out.append(evt)
                    idx += 1
                    taken += 1
            elif repeat_last and comm == seq[-1][0].lower():
                if taken >= normal_ops:
                    break
                out.append(evt)
                taken += 1
    return sorted(out, key=lambda x: (x.get("ts_unix_nano", 0), x["path"]))


def _extract_path(entry: dict) -> str:
    """Extract file path from event entry."""
    return str(entry.get("path") or "").strip()


def _extract_comm(entry: dict) -> str:
    """Extract process name (comm) from event entry."""
    return str(entry.get("comm") or "").strip()


def _path_matches(log_path: str, expected: str) -> bool:
    """Check if log_path matches expected (exact or fnmatch glob)."""
    if not log_path:
        return False
    if log_path == expected:
        return True
    if fnmatch.fnmatch(log_path, expected):
        return True
    if expected.endswith("*") or "?" in expected:
        return fnmatch.fnmatch(log_path, expected)
    return False


def _which_pattern(path: str, patterns: List[str]) -> Optional[str]:
    """Return the first matching pattern, or None."""
    for p in patterns:
        if _path_matches(path, p):
            return p
    return None


def _load_all_events(
    event_dump: Optional[Path],
    after_ts_ns: Optional[int] = None,
    within_ns: Optional[int] = None,
) -> List[dict]:
    """Load all events from event dump (has score and anomaly per event).
    If after_ts_ns/within_ns set, filter to events in [after_ts_ns, after_ts_ns + within_ns].
    """
    entries: List[dict] = []
    if not event_dump or not event_dump.exists():
        return entries
    with event_dump.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("_meta"):
                continue
            if "event_id" not in obj:
                continue
            if after_ts_ns is not None and within_ns is not None:
                ts = obj.get("ts_unix_nano", 0)
                if ts < after_ts_ns or ts > after_ts_ns + within_ns:
                    continue
            entries.append(obj)
    return entries


def _load_anomaly_entries(
    anomaly_log: Optional[Path],
    event_dump: Optional[Path],
) -> List[dict]:
    """Load anomaly entries from anomaly log, or from event dump (anomaly=True)."""
    entries: List[dict] = []

    if anomaly_log and anomaly_log.exists():
        with anomaly_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                if "event_id" not in obj:
                    continue
                entries.append(obj)
        if entries:
            return entries

    if event_dump and event_dump.exists():
        with event_dump.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                if not obj.get("anomaly"):
                    continue
                entries.append(obj)
    return entries


def verify(
    expected_paths: List[str],
    anomaly_log: Optional[Path] = None,
    event_dump: Optional[Path] = None,
) -> Tuple[Set[str], Set[str], List[dict]]:
    """
    Returns (matched_paths, unmatched_paths, anomaly_entries).
    matched_paths: expected paths that appeared in anomaly log
    unmatched_paths: expected paths not found in anomaly log
    anomaly_entries: raw entries from the log
    """
    entries = _load_anomaly_entries(anomaly_log, event_dump)
    log_paths: Set[str] = set()
    for e in entries:
        p = _extract_path(e)
        if p:
            log_paths.add(p)

    matched: Set[str] = set()
    unmatched: Set[str] = set()
    for exp in expected_paths:
        found = any(_path_matches(lp, exp) for lp in log_paths)
        if found:
            matched.add(exp)
        else:
            unmatched.add(exp)
    return matched, unmatched, entries


def verify_with_scores(
    benign_patterns: List[str],
    sensitive_patterns: List[str],
    event_dump: Optional[Path],
    anomaly_log: Optional[Path],
    after_ts_ns: Optional[int] = None,
    within_ns: Optional[int] = None,
    comm_filter: Optional[Set[str]] = None,
    sequence_filter: bool = True,
    normal_ops: int = 10,
) -> Dict:
    """
    Load all events, match to expected paths, report scores and benign-flagged status.
    Returns dict with: events, benign_flagged, benign_ok, sensitive_flagged, sensitive_missed.
    """
    all_events = _load_all_events(event_dump, after_ts_ns=after_ts_ns, within_ns=within_ns)
    anomaly_entries = _load_anomaly_entries(anomaly_log, event_dump)
    anomaly_event_ids = {e.get("event_id") for e in anomaly_entries}

    matched_events: List[dict] = []
    for evt in all_events:
        path = _extract_path(evt)
        if not path:
            continue
        benign_pat = _which_pattern(path, benign_patterns)
        sens_pat = _which_pattern(path, sensitive_patterns)
        pattern = benign_pat or sens_pat
        if not pattern:
            continue
        score = evt.get("score", 0.0)
        anomaly = evt.get("anomaly", False)
        if not anomaly and evt.get("event_id") in anomaly_event_ids:
            anomaly = True
        event_name = evt.get("event_name", "") or ""
        comm = _extract_comm(evt)
        if comm_filter is not None and comm not in comm_filter:
            continue
        matched_events.append({
            "path": path,
            "pattern": pattern,
            "benign": benign_pat is not None,
            "score": score,
            "anomaly": anomaly,
            "event_id": evt.get("event_id", ""),
            "event_name": event_name or "",
            "comm": comm,
            "ts_unix_nano": evt.get("ts_unix_nano", 0),
        })

    if sequence_filter:
        matched_events = _filter_by_sequence(matched_events, normal_ops=normal_ops)

    benign_flagged = sum(1 for e in matched_events if e["benign"] and e["anomaly"])
    benign_ok = sum(1 for e in matched_events if e["benign"] and not e["anomaly"])
    sensitive_flagged = sum(1 for e in matched_events if not e["benign"] and e["anomaly"])
    sensitive_missed = sum(1 for e in matched_events if not e["benign"] and not e["anomaly"])

    # Paths we found events for
    found_paths: Set[str] = {e["pattern"] for e in matched_events}
    not_found_patterns: List[str] = [p for p in benign_patterns + sensitive_patterns if p not in found_paths]

    return {
        "events": matched_events,
        "benign_flagged": benign_flagged,
        "benign_ok": benign_ok,
        "sensitive_flagged": sensitive_flagged,
        "sensitive_missed": sensitive_missed,
        "not_found_patterns": not_found_patterns,
        "benign_patterns": benign_patterns,
        "sensitive_patterns": sensitive_patterns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify which generated activity paths appear in detector anomaly log",
    )
    ap.add_argument(
        "--anomaly-log",
        type=Path,
        default=None,
        help="Path to anomaly log JSONL (only anomalies). Checked first.",
    )
    ap.add_argument(
        "--event-dump",
        type=Path,
        default=None,
        help="Path to event dump JSONL. Used if anomaly log missing; filters for anomaly=True.",
    )
    ap.add_argument(
        "--expected",
        nargs="+",
        default=None,
        help="Expected paths (fnmatch globs ok). Default: built-in list from generate-activity.sh",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest file (one path per line). Overrides --expected.",
    )
    ap.add_argument(
        "--benign",
        nargs="+",
        default=None,
        help="Paths expected to be benign (not flagged). Default: /tmp/sentinel-test-*.txt, /proc/*/status, /proc/version",
    )
    ap.add_argument(
        "--sensitive",
        nargs="+",
        default=None,
        help="Paths expected to be sensitive (flagged). Default: /etc/*, /root/*",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print every event; default is compact per-path summary.",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write full per-event output to CSV file (path, comm, event_name, score, kind, anomaly).",
    )
    ap.add_argument(
        "--after",
        type=float,
        default=None,
        help="Only consider events after this Unix timestamp (seconds). Use with --within to filter for generator events.",
    )
    ap.add_argument(
        "--within",
        type=float,
        default=120.0,
        help="Time window in seconds after --after (default 120). Events outside [after, after+within] are excluded.",
    )
    ap.add_argument(
        "--comm",
        type=str,
        default=None,
        help="Comma-separated list of process names (e.g. cat,head,ls). Only include events from these comms.",
    )
    ap.add_argument(
        "--no-sequence",
        action="store_true",
        help="Disable sequence filter; include all matching events (not just those in expected order).",
    )
    ap.add_argument(
        "--normal-ops",
        type=int,
        default=10,
        help="NORMAL_OPS from generator (for /proc/version repeat count). Default 10.",
    )
    args = ap.parse_args()

    if args.benign is not None:
        benign_patterns = args.benign
    else:
        benign_patterns = DEFAULT_BENIGN_PATTERNS

    if args.sensitive is not None:
        sensitive_patterns = args.sensitive
    else:
        sensitive_patterns = DEFAULT_SENSITIVE_PATTERNS

    if args.manifest and args.manifest.exists():
        all_paths = [p.strip() for p in args.manifest.read_text().splitlines() if p.strip()]
        # Heuristic: /tmp, /proc = benign; rest = sensitive
        benign_patterns = [p for p in all_paths if "/tmp" in p or "/proc" in p]
        sensitive_patterns = [p for p in all_paths if p not in benign_patterns]
    elif args.expected:
        all_paths = args.expected
        benign_patterns = [p for p in all_paths if "/tmp" in p or "/proc" in p]
        sensitive_patterns = [p for p in all_paths if p not in benign_patterns]

    expected = benign_patterns + sensitive_patterns

    if not args.anomaly_log and not args.event_dump:
        print("No anomaly log or event dump provided. Use --anomaly-log or --event-dump.", file=sys.stderr)
        sys.exit(1)

    after_ts_ns = int(args.after * 1_000_000_000) if args.after is not None else None
    within_ns = int(args.within * 1_000_000_000) if args.within is not None else None
    comm_filter: Optional[Set[str]] = None
    if args.comm:
        comm_filter = {c.strip().lower() for c in args.comm.split(",") if c.strip()}

    # Prefer event dump for full scores (all events); fall back to anomaly-only
    if args.event_dump and args.event_dump.exists():
        result = verify_with_scores(
            benign_patterns=benign_patterns,
            sensitive_patterns=sensitive_patterns,
            event_dump=args.event_dump,
            anomaly_log=args.anomaly_log,
            after_ts_ns=after_ts_ns,
            within_ns=within_ns,
            comm_filter=comm_filter,
            sequence_filter=not args.no_sequence,
            normal_ops=args.normal_ops,
        )
        events = result["events"]
        print("=== Activity verification (with scores) ===")
        print("")
        if after_ts_ns is not None:
            print(f"(Filtered to events in time window: {args.after:.1f}s + {args.within}s)")
        if comm_filter:
            print(f"(Filtered to comm: {', '.join(sorted(comm_filter))})")
        if not args.no_sequence:
            print("(Filtered to expected sequence: cat→ls→head per path)")
        if after_ts_ns is not None or comm_filter or not args.no_sequence:
            print("")
        print("Summary:")
        print(f"  Benign:    ✓ {result['benign_ok']} ok, ✗ {result['benign_flagged']} false positive(s)")
        print(f"  Sensitive: ✓ {result['sensitive_flagged']} detected, ✗ {result['sensitive_missed']} missed")
        not_found = result.get("not_found_patterns", [])
        if not_found:
            print("")
            print("  Paths with no events found (likely filtered on noisy path or outside time window):")
            for p in sorted(not_found):
                kind = "benign  " if p in result.get("benign_patterns", []) else "sensitive"
                print(f"    ✗ {p}  [{kind}]")
        print("")
        # Compact per-path: path, ops, comms, count, score range, flagged count
        by_path: dict = defaultdict(lambda: {"scores": [], "flagged": 0, "benign": False, "ops": set(), "comms": set()})
        for e in events:
            k = e["path"]
            by_path[k]["scores"].append(e["score"])
            if e["anomaly"]:
                by_path[k]["flagged"] += 1
            by_path[k]["benign"] = e["benign"]
            if e.get("event_name"):
                by_path[k]["ops"].add(e["event_name"])
            if e.get("comm"):
                by_path[k]["comms"].add(e["comm"])
        print("Per-path (compact):")
        print("-" * 72)
        for path in sorted(by_path.keys()):
            info = by_path[path]
            scores = info["scores"]
            n = len(scores)
            lo, hi = min(scores), max(scores)
            avg = sum(scores) / n
            kind = "benign  " if info["benign"] else "sensitive"
            flag_n = info["flagged"]
            flag_str = f", {flag_n} flagged" if flag_n > 0 else ""
            ops_str = ",".join(sorted(info["ops"])) if info["ops"] else "-"
            comms_str = ",".join(sorted(info["comms"])) if info["comms"] else "-"
            print(f"  {path:<35} {ops_str:<10} {comms_str:<12} n={n} score={avg:.3f} [{lo:.3f}-{hi:.3f}]{flag_str}  [{kind}]")
        print("-" * 72)
        if args.verbose and events:
            print("")
            print("Per-event (verbose):")
            for e in sorted(events, key=lambda x: (x["path"], x["event_id"])):
                kind = "benign   " if e["benign"] else "sensitive"
                flag = "FLAGGED" if e["anomaly"] else "ok"
                op = e.get("event_name", "")
                comm = e.get("comm", "")
                print(f"  {e['path']:<45} {op:<10} {comm:<8} score={e['score']:.4f}  [{kind}] {flag}")
        if args.log_file:
            with args.log_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "comm", "event_name", "score", "kind", "anomaly", "status"])
                for e in sorted(events, key=lambda x: (x["path"], x["event_id"])):
                    kind = "benign" if e["benign"] else "sensitive"
                    anomaly_val = "true" if e["anomaly"] else "false"
                    writer.writerow([
                        e["path"],
                        e.get("comm", ""),
                        e.get("event_name", ""),
                        f"{e['score']:.4f}",
                        kind,
                        anomaly_val,
                        "found",
                    ])
                for p in result.get("not_found_patterns", []):
                    kind = "benign" if p in result.get("benign_patterns", []) else "sensitive"
                    writer.writerow([p, "(not found)", "", "", kind, "false", "not_found"])
            print("")
            print(f"  Full log: {args.log_file}")
        if result["benign_flagged"] > 0:
            print("")
            print("  ⚠ Benign activity was incorrectly flagged as anomaly (false positives)")
        if result["sensitive_missed"] > 0:
            print("")
            print("  ⚠ Some sensitive paths were not flagged (missed detections)")
        sensitive_not_found = [p for p in result.get("not_found_patterns", []) if p in result.get("sensitive_patterns", [])]
        if sensitive_not_found:
            print("")
            print("  ⚠ Sensitive paths not found in event log (we accessed these - probe/rules may filter them):")
            for p in sensitive_not_found:
                print(f"    {p}")
        if not events and expected:
            print("")
            print("  No matching events found. Probe may not have captured these paths (check rules).")
        sensitive_not_found = [p for p in result.get("not_found_patterns", []) if p in result.get("sensitive_patterns", [])]
        exit_ok = (
            not sensitive_not_found
            and (result["sensitive_flagged"] > 0 or (not sensitive_patterns and result["benign_flagged"] == 0))
        )
        sys.exit(0 if exit_ok else 1)
    else:
        # Fallback: anomaly log only (no per-event scores for non-flagged)
        matched, unmatched, entries = verify(
            expected_paths=expected,
            anomaly_log=args.anomaly_log,
            event_dump=args.event_dump,
        )
        print("=== Activity verification ===")
        print("(Event dump not available; scores only for anomaly log entries. Use --event-dump for full scores.)")
        print("")
        print(f"Anomaly entries in log: {len(entries)}")
        print(f"Expected paths checked: {len(expected)}")
        print(f"Matched (in anomaly log): {len(matched)}")
        for e in entries:
            path = _extract_path(e)
            score = e.get("score", 0.0)
            print(f"  ✓ {path}  score={score:.4f}")
        if args.log_file and entries:
            with args.log_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "comm", "event_name", "score", "kind", "anomaly"])
                for e in entries:
                    path = _extract_path(e)
                    score = e.get("score", 0.0)
                    op = e.get("event_name", "") or ""
                    comm = _extract_comm(e)
                    writer.writerow([path, comm, op, f"{score:.4f}", "sensitive", "true"])
            print("")
            print(f"  Full log: {args.log_file}")
        if unmatched:
            print(f"Unmatched (expected but not in anomaly log): {len(unmatched)}")
            for p in sorted(unmatched):
                print(f"  ✗ {p}")
        sensitive = set(sensitive_patterns)
        if sensitive and not matched:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
