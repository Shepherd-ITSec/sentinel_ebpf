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


def _load_bpf_program() -> str:
  """Load BPF program from separate file."""
  bpffile = Path(__file__).parent / "probe.bpf.c"
  if not bpffile.exists():
    raise FileNotFoundError(f"BPF program file not found: {bpffile}")
  return bpffile.read_text(encoding="utf-8")


def _build_bpf_program(ring_buffer_pages: int) -> str:
  """Load and build BPF program with ring buffer pages substitution."""
  pages = max(1, int(ring_buffer_pages))
  program = _load_bpf_program()
  return program.replace("__RINGBUF_PAGES__", str(pages))


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
      # Log first few drops, then throttle logging to avoid spam
      if not hasattr(self, '_drop_count'):
        self._drop_count = 0
      self._drop_count += 1
      if self._drop_count <= 10 or self._drop_count % 1000 == 0:
        logging.warning("dropping event: queue full (total dropped: %d)", self._drop_count)

  def _request_iter(self):
    # Use blocking get with timeout to avoid busy-waiting
    # This allows the generator to yield items as they become available
    while not self._stop.is_set():
      try:
        # Use get() instead of get_nowait() to block until item available or timeout
        item = self.queue.get(timeout=0.1)
        if item is None:
          break
        yield item
        # Mark task as done to allow queue size to decrease
        self.queue.task_done()
      except queue.Empty:
        # Yield control briefly when queue is empty
        continue

  def _run_stream(self):
    backoff = 1
    events_sent = 0
    last_queue_size_log = 0
    while not self._stop.is_set():
      try:
        queue_size = self.queue.qsize()
        if queue_size > 0 and (events_sent == 0 or queue_size != last_queue_size_log):
          logging.info("Connecting to detector at %s (queue size: %d)", self.cfg.stream.endpoint, queue_size)
          last_queue_size_log = queue_size
        with grpc.insecure_channel(self.cfg.stream.endpoint) as channel:
          stub = events_pb2_grpc.DetectorServiceStub(channel)
          if events_sent == 0:
            logging.info("gRPC stream connected, starting to send events")
          for resp in stub.StreamEvents(self._request_iter()):
            events_sent += 1
            if resp.anomaly:
              logging.warning("anomaly: %s score=%.3f reason=%s", resp.event_id, resp.score, resp.reason)
            else:
              logging.debug("response: %s", resp.event_id)
            # Log progress periodically
            if events_sent % 10000 == 0:
              logging.info("Sent %d events to detector (queue size: %d)", events_sent, self.queue.qsize())
          logging.warning("gRPC stream ended (detector closed connection) after %d events", events_sent)
          backoff = 1
          events_sent = 0
      except grpc.RpcError as rpc_err:
        logging.error("grpc stream error: %s (events sent before error: %d, queue size: %d)", 
                     rpc_err, events_sent, self.queue.qsize())
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)
        events_sent = 0
      except Exception as exc:
        logging.error("Unexpected error in gRPC stream: %s (queue size: %d)", exc, self.queue.qsize(), exc_info=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)
        events_sent = 0

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
    self.streamer = GrpcStreamer(cfg)
    self.filesink: Optional[FileSink] = None
    self._stop = threading.Event()
    self._metadata = _hostname_metadata()
    self.rule_engine = RuleEngine(cfg.rules_file)
    self._load_rules_into_bpf()
    # Get boot time to convert boot-relative timestamps to Unix epoch nanoseconds
    # bpf_ktime_get_ns() returns nanoseconds since boot, we need Unix epoch
    # Read /proc/uptime to get seconds since boot, then calculate boot time
    try:
      with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.read().split()[0])
      current_time_ns = int(time.time() * 1e9)
      self._boot_time_ns = current_time_ns - int(uptime_seconds * 1e9)
    except Exception:
      # Fallback: assume boot time is current time (will be slightly off)
      logging.warning("Could not read /proc/uptime, using current time as boot time")
      self._boot_time_ns = 0
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

    # Convert boot-relative timestamp to Unix epoch nanoseconds
    # bpf_ktime_get_ns() returns nanoseconds since boot, not since epoch
    ts_unix_nano = event.ts + self._boot_time_ns

    # Ordered vector: [filename, flags, comm, pid, tid, uid] for file_open
    attributes = _event_attributes(self._metadata)
    attributes["open_flags"] = _decode_open_flags(int(event.flags))
    env = events_pb2.EventEnvelope(
      event_id=str(uuid.uuid4()),
      hostname=self._metadata["hostname"],
      pod_name=self._metadata["pod"],
      namespace=self._metadata["namespace"],
      container_id=self._metadata["container_id"],
      ts_unix_nano=ts_unix_nano,
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
