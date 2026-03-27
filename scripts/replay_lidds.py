#!/usr/bin/env python3
"""Convert LID-DS 2021 recordings to detector JSONL and optionally replay."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
  tqdm = None

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))
_DEFAULT_LIDDS_ROOT = _ROOT / "third_party" / "LID-DS"

from probe.events import EVENT_NAME_TO_ID, UNKNOWN_EVENT_ID
from rules_config import load_rules_config
from scripts.replay_logs import replay


log = logging.getLogger(__name__)

_NETWORK_EVENTS = frozenset({"socket", "connect", "bind", "listen", "accept", "accept4"})
_FILE_EVENTS = frozenset({"open", "openat", "openat2", "unlink", "unlinkat", "rename", "renameat", "chmod", "chown", "read", "write"})
_PROCESS_EVENTS = frozenset({"execve", "fork", "clone"})

_PATH_PARAM_KEYS = (
  "pathname",
  "filename",
  "path",
  "target",
  "source",
  "oldname",
  "newname",
)
_RETURN_PARAM_KEYS = ("res", "ret", "retval", "return", "return_value")
_ARG0_KEYS = ("arg0", "fd", "dirfd", "domain", "flags", "oldfd")
_ARG1_KEYS = ("arg1", "flags", "type", "addrlen", "newfd")


def _safe_str(value: Any, default: str = "") -> str:
  if value is None:
    return default
  return str(value)


def _maybe_b64_decode(value: str) -> str:
  raw = (value or "").strip()
  if not raw:
    return ""
  if len(raw) % 4 != 0:
    return raw
  try:
    decoded = base64.b64decode(raw, validate=True)
  except Exception:
    return raw
  try:
    text = decoded.decode("utf-8")
  except UnicodeDecodeError:
    return raw
  # Keep raw token if decode result looks like mostly non-printable bytes.
  if text and sum(ch.isprintable() for ch in text) / float(len(text)) > 0.8:
    return text
  return raw


def _syscall_group_map(rules_path: Path) -> dict[str, str]:
  cfg = load_rules_config(rules_path)
  out: dict[str, str] = {}
  for rule in cfg.rules:
    if not rule.enabled:
      continue
    for name in rule.syscalls:
      n = (name or "").strip().lower()
      if n and n not in out:
        out[n] = rule.group
  return out


def infer_event_group(event_name: str, syscall_to_group: dict[str, str]) -> str:
  name = (event_name or "").strip().lower()
  if not name:
    return ""
  if name in syscall_to_group:
    return syscall_to_group[name]
  if name in _NETWORK_EVENTS:
    return "network"
  if name in _FILE_EVENTS:
    return "file"
  if name in _PROCESS_EVENTS:
    return "process"
  return ""


def _pick_param(params: dict[str, str], *keys: str) -> str:
  for key in keys:
    if key in params:
      return _safe_str(params[key])
  return ""


def _event_path(params: dict[str, str]) -> str:
  for key in _PATH_PARAM_KEYS:
    if key in params:
      return _maybe_b64_decode(_safe_str(params[key]))
  return ""


def _extract_return_value(params: dict[str, str], direction_char: str) -> str:
  if direction_char != "<":
    return ""
  for key in _RETURN_PARAM_KEYS:
    if key in params:
      return _safe_str(params[key], "0")
  return ""


def _extract_arg0(event_name: str, params: dict[str, str]) -> str:
  if "arg0" in params:
    return _safe_str(params["arg0"])
  name = (event_name or "").strip().lower()
  if name in {"socket"}:
    return _pick_param(params, "domain", "family", "fd", "dirfd")
  if name in {"open"}:
    return _pick_param(params, "flags", "fd", "dirfd")
  if name in {"openat", "openat2"}:
    return _pick_param(params, "dirfd", "fd")
  return _pick_param(params, *_ARG0_KEYS)


def _extract_arg1(event_name: str, params: dict[str, str]) -> str:
  if "arg1" in params:
    return _safe_str(params["arg1"])
  name = (event_name or "").strip().lower()
  if name in {"connect", "bind", "accept", "accept4", "listen"}:
    return _pick_param(params, "addrlen", "backlog", "type")
  if name == "socket":
    return _pick_param(params, "type", "flags")
  if name in {"openat", "openat2"}:
    return _pick_param(params, "flags", "mode")
  if name == "open":
    return _pick_param(params, "mode")
  return _pick_param(params, *_ARG1_KEYS)


def _extract_attrs(event_name: str, params: dict[str, str], direction_char: str) -> dict[str, str]:
  attrs: dict[str, str] = {}
  ret = _extract_return_value(params, direction_char)
  if ret:
    attrs["return_value"] = ret

  flags = _pick_param(params, "flags")
  if flags:
    attrs["flags"] = flags

  for src_key, dst_key in (
    ("sin_port", "sin_port"),
    ("dest_port", "dest_port"),
    ("sin_addr", "sin_addr"),
    ("dest_ip", "dest_ip"),
    ("sa_family", "sa_family"),
    ("family", "sa_family"),
  ):
    value = _pick_param(params, src_key)
    if value:
      attrs[dst_key] = value
  return attrs


def convert_syscall_to_envelope(
  syscall: Any,
  *,
  event_id: str,
  hostname: str,
  syscall_to_group: dict[str, str],
  recording_name: str = "",
  recording_path: str = "",
) -> dict[str, Any]:
  event_name = _safe_str(syscall.name()).strip().lower()
  params_raw = syscall.params() or {}
  params = {_safe_str(k): _safe_str(v) for k, v in params_raw.items()}
  direction_char = ">" if _safe_str(syscall.direction()) == "Direction.OPEN" else "<"
  if hasattr(syscall.direction(), "name"):
    direction_char = ">" if syscall.direction().name == "OPEN" else "<"

  env: dict[str, Any] = {
    "event_id": event_id,
    "ts_unix_nano": int(syscall.timestamp_unix_in_ns()),
    "hostname": hostname,
    "syscall_name": event_name,
    "event_group": infer_event_group(event_name, syscall_to_group),
    "syscall_nr": int(EVENT_NAME_TO_ID.get(event_name, UNKNOWN_EVENT_ID)),
    "comm": _safe_str(syscall.process_name()),
    "pid": _safe_str(syscall.process_id(), "0"),
    "tid": _safe_str(syscall.thread_id(), "0"),
    "uid": _safe_str(syscall.user_id(), "0"),
    "arg0": _extract_arg0(event_name, params),
    "arg1": _extract_arg1(event_name, params),
    "path": _event_path(params),
    "attributes": _extract_attrs(event_name, params, direction_char),
  }
  if not isinstance(env["attributes"], dict):
    env["attributes"] = {}
  if recording_name:
    env["attributes"]["recording_name"] = str(recording_name)
  if recording_path:
    env["attributes"]["recording_path"] = str(recording_path)
  if getattr(syscall, "line_id", None) is not None:
    env["attributes"]["recording_line_id"] = str(getattr(syscall, "line_id"))
  return env


def _iter_lidds_syscalls(
  scenario_path: Path,
  split: str,
  recording_type: str | None,
  lidds_root: Path | None = None,
) -> Iterable[tuple[str, str, Any]]:
  if lidds_root is not None:
    root_str = str(lidds_root.resolve())
    if root_str not in sys.path:
      # Ensure `from dataloader.dataloader_factory import dataloader_factory` works
      # when LID-DS is checked out outside this repository.
      sys.path.insert(0, root_str)
  try:
    from dataloader.dataloader_factory import dataloader_factory  # pyright: ignore[reportMissingImports]
    from dataloader.data_loader_2021 import RecordingType  # pyright: ignore[reportMissingImports]
  except ImportError as exc:
    raise RuntimeError(
      "LID-DS dataloader not available. Install/import LID-DS first "
      "(https://github.com/LID-DS/LID-DS)."
    ) from exc

  split_name = split.strip().lower()
  loader = dataloader_factory(str(scenario_path))
  func = {
    "training": loader.training_data,
    "validation": loader.validation_data,
    "test": loader.test_data,
  }.get(split_name)
  if func is None:
    raise ValueError("split must be one of: training, validation, test")

  type_obj = None
  if recording_type:
    normalized = recording_type.strip().upper()
    try:
      type_obj = RecordingType[normalized]
    except KeyError as exc:
      valid = ", ".join(sorted(m.name for m in RecordingType))
      raise ValueError(f"invalid recording type {recording_type!r}; choose from: {valid}") from exc

  recordings = func(recording_type=type_obj)
  for rec in recordings:
    for syscall in rec.syscalls():
      yield getattr(rec, "name", ""), getattr(rec, "path", ""), syscall


def convert_lidds_to_jsonl(
  *,
  scenario_path: Path,
  split: str,
  out_jsonl: Path,
  rules_path: Path,
  recording_type: str | None = None,
  hostname: str = "lidds",
  event_id_prefix: str = "lidds",
  max_events: int | None = None,
  lidds_root: Path | None = None,
) -> int:
  syscall_to_group = _syscall_group_map(rules_path)
  out_jsonl.parent.mkdir(parents=True, exist_ok=True)

  n = 0
  syscall_iter = _iter_lidds_syscalls(scenario_path, split, recording_type, lidds_root)
  if tqdm is not None:
    syscall_iter = tqdm(syscall_iter, desc=f"Convert LID-DS ({split})", unit=" evt", file=sys.stderr)
  with out_jsonl.open("w", encoding="utf-8") as f:
    for recording_name, recording_path, syscall in syscall_iter:
      if max_events is not None and n >= max_events:
        break
      event_id = f"{event_id_prefix}-{split}-{n}"
      env = convert_syscall_to_envelope(
        syscall,
        event_id=event_id,
        hostname=hostname,
        syscall_to_group=syscall_to_group,
        recording_name=recording_name,
        recording_path=recording_path,
      )
      f.write(json.dumps(env, separators=(",", ":")) + "\n")
      n += 1
  return n


def _build_arg_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(
    description="Convert LID-DS 2021 recordings to detector JSONL and optionally replay via scripts/replay_logs.py.",
  )
  ap.add_argument("--scenario-path", required=True, help="Path to LID-DS scenario directory.")
  ap.add_argument("--split", choices=["training", "validation", "test"], default="test")
  ap.add_argument(
    "--recording-type",
    default=None,
    help="Optional LID-DS RecordingType filter (e.g. NORMAL, NORMAL_AND_ATTACK, ATTACK, IDLE).",
  )
  ap.add_argument("--out-jsonl", required=True, help="Output detector-compatible JSONL path.")
  ap.add_argument(
    "--rules-path",
    default=str(_ROOT / "charts" / "sentinel-ebpf" / "rules.yaml"),
    help="Path to rules.yaml used for syscall->event_group mapping.",
  )
  ap.add_argument("--hostname", default="lidds")
  ap.add_argument("--event-id-prefix", default="lidds")
  ap.add_argument(
    "--lidds-root",
    default=os.environ.get("LID_DS_ROOT", "").strip() or str(_DEFAULT_LIDDS_ROOT),
    help="Path to local LID-DS repository root containing the `dataloader` package. "
    "Defaults to third_party/LID-DS in this repo; can be overridden via LID_DS_ROOT.",
  )
  ap.add_argument("--max-convert-events", type=int, default=None, help="Stop conversion after N events.")

  ap.add_argument("--target", default="localhost:50051", help="Detector gRPC target for replay.")
  ap.add_argument("--pace", choices=["fast", "realtime"], default="fast")
  ap.add_argument("--start-ms", type=int, default=None)
  ap.add_argument("--end-ms", type=int, default=None)
  ap.add_argument("--max-events", type=int, default=None, help="Max events to replay (after conversion).")
  ap.add_argument("--skip", type=int, default=None)
  ap.add_argument("--convert-only", action="store_true", help="Only convert to JSONL; do not replay.")
  return ap


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  args = _build_arg_parser().parse_args()

  scenario_path = Path(args.scenario_path)
  out_jsonl = Path(args.out_jsonl)
  rules_path = Path(args.rules_path)
  converted = convert_lidds_to_jsonl(
    scenario_path=scenario_path,
    split=args.split,
    out_jsonl=out_jsonl,
    rules_path=rules_path,
    recording_type=args.recording_type,
    hostname=args.hostname,
    event_id_prefix=args.event_id_prefix,
    max_events=args.max_convert_events,
    lidds_root=Path(args.lidds_root) if args.lidds_root else None,
  )
  log.info("Converted %d LID-DS events to %s", converted, out_jsonl)
  if args.convert_only:
    return
  replay(
    out_jsonl,
    args.target,
    args.pace,
    args.start_ms,
    args.end_ms,
    total=converted,
    label="Replay LID-DS",
    max_events=args.max_events,
    skip=args.skip,
  )


if __name__ == "__main__":
  main()
