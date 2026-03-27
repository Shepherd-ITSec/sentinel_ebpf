"""Helpers for normalizing legacy JSONL event records (compact ``data`` array → envelope fields)."""

from __future__ import annotations

from typing import Any


def expand_legacy_data_field(obj: dict[str, Any]) -> dict[str, Any]:
  """
  Legacy dumps used a top-level ``data`` list:
  [syscall_name, syscall_nr, comm, pid, tid, uid, arg0, arg1, path, optional_10th].

  Merge into the flat fields ``envelope_from_dict`` expects and remove ``data``.
  If ``data`` is missing or too short, return ``obj`` unchanged.
  """
  raw = obj.get("data")
  if not isinstance(raw, list) or len(raw) < 9:
    return obj
  cells = ["" if x is None else str(x) for x in raw]
  name, nr_s, comm, pid, tid, uid, arg0, arg1, path = cells[0:9]
  out = {k: v for k, v in obj.items() if k != "data"}
  try:
    syscall_nr = int(nr_s, 0)
  except (TypeError, ValueError):
    syscall_nr = 0
  if syscall_nr < 0:
    syscall_nr = 0
  if syscall_nr > 0xFFFFFFFF:
    syscall_nr = 0xFFFFFFFF
  out["syscall_name"] = name
  out["syscall_nr"] = syscall_nr
  out["comm"] = comm
  out["pid"] = pid
  out["tid"] = tid
  out["uid"] = uid
  out["arg0"] = arg0
  out["arg1"] = arg1
  out["path"] = path
  return out
