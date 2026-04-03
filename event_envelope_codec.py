"""JSON/EVT1 dict layout for EventEnvelope (shared by probe sink, replay, diagnostics)."""
from __future__ import annotations

from typing import Any

import events_pb2


def envelope_to_dict(evt: events_pb2.EventEnvelope) -> dict[str, Any]:
  return {
    "event_id": evt.event_id,
    "ts_unix_nano": evt.ts_unix_nano,
    "hostname": evt.hostname,
    "pod": evt.pod_name,
    "namespace": evt.namespace,
    "container_id": evt.container_id,
    "syscall_name": evt.syscall_name,
    "event_group": evt.event_group,
    "syscall_nr": evt.syscall_nr,
    "comm": evt.comm,
    "pid": evt.pid,
    "tid": evt.tid,
    "uid": evt.uid,
    "arg0": evt.arg0,
    "arg1": evt.arg1,
    "attributes": dict(evt.attributes),
  }


def envelope_from_dict(obj: dict) -> events_pb2.EventEnvelope:
  pod_name = obj.get("pod_name", obj.get("pod", ""))
  syscall_name = obj.get("syscall_name", obj.get("event_name", ""))
  raw_nr = obj.get("syscall_nr", 0)
  try:
    syscall_nr = int(raw_nr)
  except (TypeError, ValueError):
    syscall_nr = 0
  if syscall_nr < 0:
    syscall_nr = 0
  if syscall_nr > 0xFFFFFFFF:
    syscall_nr = 0xFFFFFFFF
  return events_pb2.EventEnvelope(
    event_id=str(obj.get("event_id", "")),
    hostname=str(obj.get("hostname", "")),
    pod_name=str(pod_name),
    namespace=str(obj.get("namespace", "")),
    container_id=str(obj.get("container_id", "")),
    ts_unix_nano=int(obj.get("ts_unix_nano", 0)),
    syscall_name=str(syscall_name),
    event_group=str(obj.get("event_group", "")),
    syscall_nr=syscall_nr,
    comm=str(obj.get("comm", "")),
    pid=str(obj.get("pid", "")),
    tid=str(obj.get("tid", "")),
    uid=str(obj.get("uid", "")),
    arg0=str(obj.get("arg0", "")),
    arg1=str(obj.get("arg1", "")),
    attributes=dict(obj.get("attributes", {}) or {}),
  )
