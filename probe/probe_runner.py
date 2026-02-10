import ctypes
import gzip
import json
import logging
import os
import queue
import signal
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import grpc
from bcc import BPF

import events_pb2
import events_pb2_grpc
from probe.config import AppConfig, load_config
from probe.rules import RuleEngine

# BPF-side rule limits. Keep in sync with BPF program definitions.
# Verifier limit ~8192 jumps; unrolled loops (MAX_RULES * (MAX_PREFIX_LEN + COMM_LEN)) must stay under.
MAX_RULES = 24
MAX_PREFIX_LEN = 24
COMM_LEN = 16


class BpfRule(ctypes.Structure):
  _fields_ = [
    ("enabled", ctypes.c_uint32),
    ("event_type", ctypes.c_uint32),  # 0 read, 1 write
    ("prefix_len", ctypes.c_uint32),
    ("comm_len", ctypes.c_uint32),
    ("pid", ctypes.c_uint32),
    ("tid", ctypes.c_uint32),
    ("uid", ctypes.c_uint32),
    ("prefix", ctypes.c_char * MAX_PREFIX_LEN),
    ("comm", ctypes.c_char * COMM_LEN),
  ]


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/dcache.h>
#include <linux/path.h>
#include <linux/fs.h>

#define MAX_RULES 24
#define MAX_PREFIX_LEN 24
#define COMM_LEN 16
#define RINGBUF_PAGES __RINGBUF_PAGES__

struct rule_t {
    u32 enabled;
    u32 event_type; // 0 open
    u32 prefix_len;
    u32 comm_len;
    u32 pid;
    u32 tid;
    u32 uid;
    char prefix[MAX_PREFIX_LEN];
    char comm[COMM_LEN];
};

struct data_t {
    u64 ts;
    u32 pid;
    u32 tid;
    u32 uid;
    u32 flags;
    char comm[COMM_LEN];
    char filename[256];
};

BPF_ARRAY(rules, struct rule_t, MAX_RULES);
BPF_ARRAY(rule_count, u32, 1);
BPF_RINGBUF_OUTPUT(events, RINGBUF_PAGES);

/* No early return/break in loop so clang can unroll (fixed trip count COMM_LEN). */
static __inline int comm_matches(struct data_t *data, const struct rule_t *rule) {
    int match = 1;
    if (rule->comm_len == 0) {
        return 1;
    }
#pragma unroll
    for (int i = 0; i < COMM_LEN; i++) {
        if (i < rule->comm_len) {
            if (data->comm[i] != rule->comm[i] || data->comm[i] == 0) {
                match = 0;
            }
        }
    }
    return match;
}

/* No early return/break in loop so clang can unroll (fixed trip count MAX_PREFIX_LEN). */
static __inline int prefix_matches(struct data_t *data, const struct rule_t *rule) {
    int match = 1;
    if (rule->prefix_len == 0) {
        return 1;
    }
#pragma unroll
    for (int i = 0; i < MAX_PREFIX_LEN; i++) {
        if (i < rule->prefix_len) {
            if (data->filename[i] != rule->prefix[i] || data->filename[i] == 0) {
                match = 0;
            }
        }
    }
    return match;
}

/* No break/continue/return in loop so clang can unroll (fixed trip count MAX_RULES). */
static __inline int rule_allows(struct data_t *data) {
    u32 idx = 0;
    u32 *count = rule_count.lookup(&idx);
    if (!count || *count == 0) {
        return 1;
    }
    int allowed = 0;
#pragma unroll
    for (int i = 0; i < MAX_RULES; i++) {
        if (i < *count) {
            u32 key = i;
            struct rule_t *rule = rules.lookup(&key);
            if (rule && rule->enabled && rule->event_type == 0 &&
                (rule->pid == 0 || rule->pid == data->pid) &&
                (rule->tid == 0 || rule->tid == data->tid) &&
                (rule->uid == 0 || rule->uid == data->uid) &&
                comm_matches(data, rule) && prefix_matches(data, rule)) {
                allowed = 1;
            }
        }
    }
    return allowed;
}

int trace_open(struct pt_regs *ctx, const char __user *filename, int flags) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.uid = bpf_get_current_uid_gid();
    data.flags = flags;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    if (filename) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), filename);
    }

    if (!rule_allows(&data)) {
        return 0;
    }

    events.ringbuf_output(&data, sizeof(data), 0);
    return 0;
}

int trace_openat(struct pt_regs *ctx, int dfd, const char __user *filename, int flags) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.uid = bpf_get_current_uid_gid();
    data.flags = flags;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    if (filename) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), filename);
    }

    if (!rule_allows(&data)) {
        return 0;
    }

    events.ringbuf_output(&data, sizeof(data), 0);
    return 0;
}
"""


def _build_bpf_program(ring_buffer_pages: int) -> str:
  pages = max(1, int(ring_buffer_pages))
  return BPF_PROGRAM.replace("__RINGBUF_PAGES__", str(pages))


def _hostname_metadata():
  return {
    "hostname": socket.gethostname(),
    "cluster": os.environ.get("CLUSTER_NAME", ""),
    "node": os.environ.get("NODE_NAME", ""),
    "pod": os.environ.get("POD_NAME", ""),
    "namespace": os.environ.get("POD_NAMESPACE", ""),
    "container_id": os.environ.get("CONTAINER_ID", ""),
  }


def _event_attributes(meta: dict) -> dict:
  attrs = {}
  cluster = meta.get("cluster", "")
  node = meta.get("node", "")
  if cluster:
    attrs["cluster"] = cluster
  if node:
    attrs["node"] = node
  return attrs


def _decode_open_flags(flags: int) -> str:
  flag_names = []
  if flags & os.O_RDONLY:
    flag_names.append("O_RDONLY")
  if flags & os.O_WRONLY:
    flag_names.append("O_WRONLY")
  if flags & os.O_RDWR:
    flag_names.append("O_RDWR")
  for name in [
    "O_CREAT",
    "O_TRUNC",
    "O_APPEND",
    "O_CLOEXEC",
    "O_EXCL",
    "O_DIRECTORY",
    "O_NOFOLLOW",
    "O_SYNC",
    "O_DSYNC",
  ]:
    if hasattr(os, name) and (flags & getattr(os, name)):
      flag_names.append(name)
  return "|".join(flag_names) if flag_names else "0"


class GrpcStreamer:
  def __init__(self, cfg: AppConfig):
    self.cfg = cfg
    self.queue: "queue.Queue[events_pb2.EventEnvelope]" = queue.Queue(maxsize=cfg.stream.queue_length)
    self._stop = threading.Event()
    self._thread: Optional[threading.Thread] = None

  def publish(self, evt: events_pb2.EventEnvelope):
    try:
      self.queue.put_nowait(evt)
    except queue.Full:
      logging.warning("dropping event: queue full")

  def _request_iter(self):
    while not self._stop.is_set():
      try:
        item = self.queue.get(timeout=1)
      except queue.Empty:
        continue
      if item is None:
        break
      yield item

  def _run_stream(self):
    backoff = 1
    while not self._stop.is_set():
      try:
        with grpc.insecure_channel(self.cfg.stream.endpoint) as channel:
          stub = events_pb2_grpc.DetectorServiceStub(channel)
          for resp in stub.StreamEvents(self._request_iter()):
            if resp.anomaly:
              logging.warning("anomaly: %s score=%.3f reason=%s", resp.event_id, resp.score, resp.reason)
            else:
              logging.debug("response: %s", resp.event_id)
        backoff = 1
      except grpc.RpcError as rpc_err:
        logging.error("grpc stream error: %s", rpc_err)
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

  def start(self):
    if self.cfg.stream.mode != "grpc":
      return
    self._thread = threading.Thread(target=self._run_stream, daemon=True)
    self._thread.start()

  def stop(self):
    self._stop.set()
    try:
      self.queue.put_nowait(None)
    except queue.Full:
      pass
    if self._thread:
      self._thread.join(timeout=5)


class FileSink:
  MAGIC = b"EVT1"

  def __init__(self, path: str, max_bytes: int, max_files: int, compress: bool):
    self.base_path = Path(path)
    self.max_bytes = max_bytes
    self.max_files = max_files
    self.compress = compress
    self.lock = threading.Lock()
    self.base_path.parent.mkdir(parents=True, exist_ok=True)
    self._fh = None
    self._open()

  def _open(self):
    mode = "ab"
    if self.compress:
      self._fh = gzip.open(self.base_path, mode)
    else:
      self._fh = open(self.base_path, mode)

  def _size(self) -> int:
    try:
      return self.base_path.stat().st_size
    except FileNotFoundError:
      return 0

  def _rotate(self):
    if self.max_files <= 1:
      return
    if self._fh:
      self._fh.close()
    for i in range(self.max_files - 1, 0, -1):
      src = self.base_path.with_suffix(self.base_path.suffix + ("" if i == 1 else f".{i-1}"))
      dst = self.base_path.with_suffix(self.base_path.suffix + f".{i}")
      if src.exists():
        src.replace(dst)
    # base becomes .1
    if self.base_path.exists():
      self.base_path.replace(self.base_path.with_suffix(self.base_path.suffix + ".1"))
    self._open()

  def publish(self, env: events_pb2.EventEnvelope):
    # Serialize data as ordered array (more efficient than map)
    payload = {
      "event_id": env.event_id,
      "ts_unix_nano": env.ts_unix_nano,
      "hostname": env.hostname,
      "pod": env.pod_name,
      "namespace": env.namespace,
      "container_id": env.container_id,
      "event_type": env.event_type,
      "data": list(env.data),  # Ordered vector
      "attributes": dict(env.attributes),
    }
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    record = self.MAGIC + struct.pack("<I", len(blob)) + blob
    with self.lock:
      if self.max_bytes and self._size() + len(record) > self.max_bytes:
        self._rotate()
      self._fh.write(record)
      self._fh.flush()

  def close(self):
    with self.lock:
      if self._fh:
        self._fh.close()
        self._fh = None


class HealthHandler(BaseHTTPRequestHandler):
  def do_GET(self):  # noqa: N802
    if self.path.startswith("/metrics"):
      self.send_response(200)
      self.send_header("Content-Type", "text/plain; version=0.0.4")
      self.end_headers()
      self.wfile.write(b"# HELP sentinel_ebpf_dummy Always 1\n# TYPE sentinel_ebpf_dummy gauge\nsentinel_ebpf_dummy 1\n")
      return
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b"ok")

  def log_message(self, format, *args):  # noqa: A003
    return


def start_health_server(port: int) -> HTTPServer:
  server = HTTPServer(("0.0.0.0", port), HealthHandler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server


class ProbeRunner:
  def __init__(self, cfg: AppConfig):
    self.cfg = cfg
    self.bpf = BPF(text=_build_bpf_program(cfg.stream.ring_buffer_pages))
    sys_prefix = self.bpf.get_syscall_prefix().decode()
    self.bpf.attach_kprobe(event=f"{sys_prefix}open", fn_name="trace_open")
    self.bpf.attach_kprobe(event=f"{sys_prefix}openat", fn_name="trace_openat")
    self.streamer = GrpcStreamer(cfg)
    self.filesink: Optional[FileSink] = None
    self._stop = threading.Event()
    self._metadata = _hostname_metadata()
    self.rule_engine = RuleEngine(cfg.rules_file)
    self._load_rules_into_bpf()
    if self.cfg.stream.mode == "file":
      self.filesink = FileSink(
        path=self.cfg.stream.file_path,
        max_bytes=self.cfg.stream.rotate_max_bytes,
        max_files=self.cfg.stream.rotate_max_files,
        compress=self.cfg.stream.compress,
      )

  def _compile_rules(self):
    compiled = []
    for rule in self.rule_engine.rules:
      if not rule.enabled:
        continue
      if rule.event != "file_open":
        continue
      event_type = 0
      prefixes = rule.match.path_prefixes or [""]
      comms = rule.match.comms or [""]
      pids = rule.match.pids or [0]
      tids = rule.match.tids or [0]
      uids = rule.match.uids or [0]
      for prefix in prefixes:
        for comm in comms:
          for pid in pids:
            for tid in tids:
              for uid in uids:
                compiled.append(
                  {
                    "event_type": event_type,
                    "prefix": prefix,
                    "comm": comm,
                    "pid": pid,
                    "tid": tid,
                    "uid": uid,
                  }
                )
    return compiled

  def _load_rules_into_bpf(self):
    compiled = self._compile_rules()
    count = min(len(compiled), MAX_RULES)
    rules_map = self.bpf["rules"]
    count_map = self.bpf["rule_count"]
    count_map[ctypes.c_int(0)] = ctypes.c_uint(count)
    for idx in range(count):
      item = compiled[idx]
      prefix_bytes = item["prefix"].encode("utf-8")[:MAX_PREFIX_LEN]
      comm_bytes = item["comm"].encode("utf-8")[:COMM_LEN]
      entry = BpfRule(
        enabled=1,
        event_type=item["event_type"],
        prefix_len=len(prefix_bytes),
        comm_len=len(comm_bytes),
        pid=item["pid"],
        tid=item["tid"],
        uid=item["uid"],
        prefix=prefix_bytes,
        comm=comm_bytes,
      )
      rules_map[ctypes.c_int(idx)] = entry

  def _handle_event(self, cpu, data, size):  # noqa: ARG002
    event = self.bpf["events"].event(data)
    filename = event.filename.decode(errors="ignore").rstrip("\x00")
    comm = event.comm.decode(errors="ignore").rstrip("\x00")
    event_type = "file_open"

    # Ordered vector: [filename, flags, comm, pid, tid, uid] for file_open
    attributes = _event_attributes(self._metadata)
    attributes["open_flags"] = _decode_open_flags(int(event.flags))
    env = events_pb2.EventEnvelope(
      event_id=str(uuid.uuid4()),
      hostname=self._metadata["hostname"],
      pod_name=self._metadata["pod"],
      namespace=self._metadata["namespace"],
      container_id=self._metadata["container_id"],
      ts_unix_nano=event.ts,
      event_type=event_type,
      data=[filename, str(event.flags), comm, str(event.pid), str(event.tid), str(event.uid)],
      attributes=attributes,
    )

    if self.cfg.stream.mode == "stdout":
      payload = {
        "event_type": event_type,
        "event_id": env.event_id,
        "ts_unix_nano": env.ts_unix_nano,
        "hostname": env.hostname,
        "pod": env.pod_name,
        "namespace": env.namespace,
        "attributes": dict(env.attributes),
        "data": list(env.data),  # Ordered vector
      }
      print(json.dumps(payload))
    elif self.cfg.stream.mode == "file" and self.filesink:
      self.filesink.publish(env)
    else:
      self.streamer.publish(env)

  def run(self):
    self.streamer.start()
    self.bpf["events"].open_ring_buffer(self._handle_event)
    while not self._stop.is_set():
      try:
        self.bpf.ring_buffer_poll(timeout=1000)
      except KeyboardInterrupt:
        self.stop()
      except Exception as exc:  # pylint: disable=broad-except
        logging.exception("polling error: %s", exc)

  def stop(self):
    self._stop.set()
    self.streamer.stop()
    if self.filesink:
      self.filesink.close()


def configure_logging(level: str):
  lvl = getattr(logging, level.upper(), logging.INFO)
  logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")


def main():
  cfg = load_config()
  configure_logging(cfg.log_level)
  health_server = start_health_server(cfg.health_port)
  runner = ProbeRunner(cfg)

  def handle_signal(signum, frame):  # noqa: ANN001
    logging.info("received signal %s", signum)
    runner.stop()
    health_server.shutdown()

  signal.signal(signal.SIGTERM, handle_signal)
  signal.signal(signal.SIGINT, handle_signal)

  runner.run()


if __name__ == "__main__":
  main()
