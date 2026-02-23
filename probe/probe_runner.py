import asyncio
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
from collections import deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import grpc
import grpc.aio
from bcc import BPF

import events_pb2
import events_pb2_grpc
from probe.config import AppConfig, load_config
from probe.open_flags import decode_open_flags
from probe.rules import RuleEngine

# Fast UUID generation using counter + timestamp (much faster than uuid.uuid4())
_event_counter = 0
_event_counter_lock = threading.Lock()

def _fast_event_id() -> str:
    """Generate a fast unique event ID using counter + timestamp."""
    global _event_counter
    with _event_counter_lock:
        _event_counter = (_event_counter + 1) % (2**32)
        counter = _event_counter
    # Use timestamp (microseconds) + counter for uniqueness
    timestamp_us = int(time.time() * 1e6)
    return f"{timestamp_us:016x}-{counter:08x}"

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


class GrpcStreamer:
  def __init__(self, cfg: AppConfig):
    self.cfg = cfg
    # Use deque with lock for better performance than queue.Queue
    self._queue: deque = deque(maxlen=cfg.stream.queue_length)
    self._lock = threading.Lock()
    self._stop = threading.Event()
    self._thread: Optional[threading.Thread] = None
    self._drop_count = 0
    self._events_sent = 0  # Track total events sent
    self._events_sent_lock = threading.Lock()  # Lock for events_sent counter
    self._batch_size = max(1, cfg.stream.batch_size)

  def publish(self, evt: events_pb2.EventEnvelope):
    # Ultra-fast publish - minimize lock hold time
    # Check length first to detect drops (deque with maxlen will drop oldest, not raise error)
    with self._lock:
      queue_len = len(self._queue)
      if queue_len < self._queue.maxlen:
        self._queue.append(evt)
        return
      # Queue full - drop event (don't append, as deque would drop oldest)
      self._drop_count += 1
      if self._drop_count <= 10 or self._drop_count % 1000 == 0:
        logging.warning("dropping event: queue full (total dropped: %d)", self._drop_count)

  async def _request_iter(self):
    # Async generator for streaming events efficiently
    # This async generator is called by async gRPC to get events to send
    # We yield events immediately when available, with minimal blocking
    consecutive_empty = 0
    while not self._stop.is_set():
      # Collect events quickly with minimal lock time
      batch = []
      with self._lock:
        # Grab as many as we can up to batch size
        # Use list comprehension for better performance
        batch_size = min(self._batch_size, len(self._queue))
        if batch_size > 0:
          batch = [self._queue.popleft() for _ in range(batch_size)]
      
      if batch:
        consecutive_empty = 0
        # Yield all events in batch immediately - no await needed here
        for item in batch:
          if item is None:
            return
          yield item
        # After yielding batch, immediately check for more without sleep
        continue
      else:
        # No events available - yield control to event loop
        # Use exponential backoff for empty checks to reduce CPU usage
        consecutive_empty += 1
        if consecutive_empty < 10:
          # First 10 empty checks - no sleep, just yield control
          await asyncio.sleep(0)  # Yield to event loop without delay
        elif consecutive_empty < 100:
          # Next 90 checks - very short delay
          await asyncio.sleep(0.000001)  # 1 microsecond
        elif consecutive_empty < 1000:
          # Next 900 checks - slightly longer
          await asyncio.sleep(0.00001)  # 10 microseconds
        else:
          # After many empty checks - longer delay
          await asyncio.sleep(0.0001)  # 100 microseconds

  async def _run_stream(self):
    backoff = 1
    events_sent_ref = [0]  # Use list for mutable reference (per-stream counter)
    last_queue_size_log = 0
    # Reuse channel for better performance
    channel = None
    while not self._stop.is_set():
      try:
        with self._lock:
          queue_size = len(self._queue)
        if queue_size > 0 and (events_sent_ref[0] == 0 or queue_size != last_queue_size_log):
          logging.info("Connecting to detector at %s (queue size: %d)", self.cfg.stream.endpoint, queue_size)
          last_queue_size_log = queue_size
        
        # Create or reuse async channel
        if channel is None:
          # Use keepalive and optimize for high throughput
          options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            # Increase message size limits for better throughput
            ('grpc.max_send_message_length', 4 * 1024 * 1024),  # 4MB
            ('grpc.max_receive_message_length', 4 * 1024 * 1024),  # 4MB
          ]
          channel = grpc.aio.insecure_channel(self.cfg.stream.endpoint, options=options)
        
        stub = events_pb2_grpc.DetectorServiceStub(channel)
        if events_sent_ref[0] == 0:
          logging.info("Async gRPC stream connected, starting to send events")
        
        # Stream events asynchronously - responses don't block sending
        # This is the key advantage of async: we can send and receive concurrently
        response_count = 0
        initial_events_sent = 0
        with self._events_sent_lock:
          initial_events_sent = self._events_sent
        
        async for resp in stub.StreamEvents(self._request_iter()):
          response_count += 1
          events_sent_ref[0] = response_count
          # Update total events sent counter (cumulative across all streams)
          # Update every 100 events to reduce lock contention
          if response_count % 100 == 0:
            with self._events_sent_lock:
              self._events_sent = initial_events_sent + response_count
          # Only log anomalies for performance - skip normal responses
          if resp.anomaly:
            logging.warning("anomaly: %s score=%.3f reason=%s", resp.event_id, resp.score, resp.reason)
          # Log progress periodically (less frequent for performance)
          if response_count % 50000 == 0:
            with self._lock:
              queue_size = len(self._queue)
            logging.info("Processed %d responses from detector (queue size: %d)", response_count, queue_size)
        
        # Update final count for this stream
        with self._events_sent_lock:
          self._events_sent = initial_events_sent + response_count
        
        logging.warning("gRPC stream ended (detector closed connection) after %d events", events_sent_ref[0])
        await channel.close()
        channel = None
        backoff = 1
        events_sent_ref[0] = 0  # Reset per-stream counter
        
      except grpc.RpcError as rpc_err:
        if channel:
          await channel.close()
          channel = None
        with self._lock:
          queue_size = len(self._queue)
        # Update final count for this stream before resetting
        if response_count > 0:
          with self._events_sent_lock:
            self._events_sent = initial_events_sent + response_count
        
        logging.error("grpc stream error: %s (events processed: %d, queue size: %d)", 
                     rpc_err, events_sent_ref[0], queue_size)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30)
        events_sent_ref[0] = 0
      except Exception as exc:
        if channel:
          await channel.close()
          channel = None
        with self._lock:
          queue_size = len(self._queue)
        # Update final count for this stream before resetting
        if response_count > 0:
          with self._events_sent_lock:
            self._events_sent = initial_events_sent + response_count
        
        logging.error("Unexpected error in gRPC stream: %s (queue size: %d)", exc, queue_size, exc_info=True)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30)
        events_sent_ref[0] = 0

  def start(self):
    if self.cfg.stream.mode != "grpc":
      return
    
    def run_async_loop():
      # Create a new event loop for this thread
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(self._run_stream())
      finally:
        loop.close()
    
    self._thread = threading.Thread(target=run_async_loop, daemon=True)
    self._thread.start()

  def stop(self):
    self._stop.set()
    with self._lock:
      self._queue.append(None)
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
      
      # Get metrics from probe runner if available
      metrics_lines = []
      
      # Try to get probe runner instance (set by main)
      probe_runner = getattr(self.server, "_probe_runner", None)
      if probe_runner and hasattr(probe_runner, "streamer"):
        streamer = probe_runner.streamer
        with streamer._lock:
          queue_size = len(streamer._queue)
          drop_count = streamer._drop_count
        
        with streamer._events_sent_lock:
          events_sent = streamer._events_sent
        
        rules_compiled = getattr(probe_runner, "_compiled_rule_count", 0)
        rules_loaded = getattr(probe_runner, "_loaded_rule_count", 0)
        metrics_lines.extend([
          "# HELP sentinel_ebpf_probe_events_sent_total Total number of events sent to detector",
          "# TYPE sentinel_ebpf_probe_events_sent_total counter",
          f"sentinel_ebpf_probe_events_sent_total {events_sent}",
          "",
          "# HELP sentinel_ebpf_probe_queue_size Current number of events in queue",
          "# TYPE sentinel_ebpf_probe_queue_size gauge",
          f"sentinel_ebpf_probe_queue_size {queue_size}",
          "",
          "# HELP sentinel_ebpf_probe_events_dropped_total Total number of events dropped (queue full)",
          "# TYPE sentinel_ebpf_probe_events_dropped_total counter",
          f"sentinel_ebpf_probe_events_dropped_total {drop_count}",
          "",
          "# HELP sentinel_ebpf_probe_rules_compiled Total number of compiled userspace rules",
          "# TYPE sentinel_ebpf_probe_rules_compiled gauge",
          f"sentinel_ebpf_probe_rules_compiled {rules_compiled}",
          "",
          "# HELP sentinel_ebpf_probe_rules_loaded Total number of rules loaded into BPF map",
          "# TYPE sentinel_ebpf_probe_rules_loaded gauge",
          f"sentinel_ebpf_probe_rules_loaded {rules_loaded}",
          "",
          "# HELP sentinel_ebpf_probe_rules_truncated_total Total number of compiled rules dropped due to MAX_RULES cap",
          "# TYPE sentinel_ebpf_probe_rules_truncated_total gauge",
          f"sentinel_ebpf_probe_rules_truncated_total {max(0, rules_compiled - rules_loaded)}",
          "",
        ])
      else:
        # Fallback: at least return something if probe runner not available
        metrics_lines.extend([
          "# HELP sentinel_ebpf_probe_events_sent_total Total number of events sent to detector",
          "# TYPE sentinel_ebpf_probe_events_sent_total counter",
          "sentinel_ebpf_probe_events_sent_total 0",
          "",
          "# HELP sentinel_ebpf_probe_queue_size Current number of events in queue",
          "# TYPE sentinel_ebpf_probe_queue_size gauge",
          "sentinel_ebpf_probe_queue_size 0",
          "",
          "# HELP sentinel_ebpf_probe_events_dropped_total Total number of events dropped (queue full)",
          "# TYPE sentinel_ebpf_probe_events_dropped_total counter",
          "sentinel_ebpf_probe_events_dropped_total 0",
          "",
          "# HELP sentinel_ebpf_probe_rules_compiled Total number of compiled userspace rules",
          "# TYPE sentinel_ebpf_probe_rules_compiled gauge",
          "sentinel_ebpf_probe_rules_compiled 0",
          "",
          "# HELP sentinel_ebpf_probe_rules_loaded Total number of rules loaded into BPF map",
          "# TYPE sentinel_ebpf_probe_rules_loaded gauge",
          "sentinel_ebpf_probe_rules_loaded 0",
          "",
          "# HELP sentinel_ebpf_probe_rules_truncated_total Total number of compiled rules dropped due to MAX_RULES cap",
          "# TYPE sentinel_ebpf_probe_rules_truncated_total gauge",
          "sentinel_ebpf_probe_rules_truncated_total 0",
          "",
        ])
      
      self.wfile.write("\n".join(metrics_lines).encode("utf-8"))
      return
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b"ok")

  def log_message(self, format, *args):  # noqa: A003
    return


def start_health_server(port: int, probe_runner=None) -> HTTPServer:
  server = HTTPServer(("0.0.0.0", port), HealthHandler)
  # Store probe runner reference so metrics handler can access it
  server._probe_runner = probe_runner
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
    self._compiled_rule_count = 0
    self._loaded_rule_count = 0
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
    
    # Pre-compute static attributes to avoid repeated dict creation
    self._static_attrs = _event_attributes(self._metadata)
    
    # Pre-compute string conversions for common values
    self._event_type = "file_open"
    
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
    self._compiled_rule_count = len(compiled)
    count = min(len(compiled), MAX_RULES)
    self._loaded_rule_count = count
    dropped = len(compiled) - count
    if dropped > 0:
      logging.warning(
        "Compiled %d rules but only %d loaded into BPF (MAX_RULES=%d). %d rule(s) dropped.",
        len(compiled),
        count,
        MAX_RULES,
        dropped,
      )
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
    
    # Optimize string decoding - faster null-terminated string extraction
    filename_bytes = event.filename
    null_idx = filename_bytes.find(b'\x00')
    filename = filename_bytes[:null_idx].decode(errors="ignore") if null_idx >= 0 else filename_bytes.decode(errors="ignore")

    # Exclude paths configured in rules (e.g. /proc) before building the event
    if self.rule_engine.path_excluded(filename):
      return

    comm_bytes = event.comm
    null_idx = comm_bytes.find(b'\x00')
    comm = comm_bytes[:null_idx].decode(errors="ignore") if null_idx >= 0 else comm_bytes.decode(errors="ignore")

    # Convert boot-relative timestamp to Unix epoch nanoseconds
    ts_unix_nano = event.ts + self._boot_time_ns

    # Reuse pre-computed attributes dict and update only open_flags
    # Use dict() constructor instead of copy() for slightly better performance
    attributes = dict(self._static_attrs)
    attributes["open_flags"] = decode_open_flags(int(event.flags))
    
    # Use fast event ID generation (counter + timestamp instead of uuid.uuid4())
    # Pre-convert integers to strings once
    flags_str = str(event.flags)
    pid_str = str(event.pid)
    tid_str = str(event.tid)
    uid_str = str(event.uid)
    
    env = events_pb2.EventEnvelope(
      event_id=_fast_event_id(),
      hostname=self._metadata["hostname"],
      pod_name=self._metadata["pod"],
      namespace=self._metadata["namespace"],
      container_id=self._metadata["container_id"],
      ts_unix_nano=ts_unix_nano,
      event_type=self._event_type,
      # Pre-converted strings to avoid repeated conversions
      data=[filename, flags_str, comm, pid_str, tid_str, uid_str],
      attributes=attributes,
    )

    if self.cfg.stream.mode == "stdout":
      payload = {
        "event_type": self._event_type,
        "event_id": env.event_id,
        "ts_unix_nano": env.ts_unix_nano,
        "hostname": env.hostname,
        "pod": env.pod_name,
        "namespace": env.namespace,
        "attributes": dict(env.attributes),
        "data": list(env.data),
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
        # Reduce timeout to 1ms for maximum responsiveness
        # This allows very fast consumption of events from the ring buffer
        # Lower timeout = more frequent polling = faster event processing
        # At high event rates, we want to poll as frequently as possible
        self.bpf.ring_buffer_poll(timeout=1)
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
  runner = ProbeRunner(cfg)
  health_server = start_health_server(cfg.health_port, probe_runner=runner)

  def handle_signal(signum, frame):  # noqa: ANN001
    logging.info("received signal %s", signum)
    runner.stop()
    health_server.shutdown()

  signal.signal(signal.SIGTERM, handle_signal)
  signal.signal(signal.SIGINT, handle_signal)

  runner.run()


if __name__ == "__main__":
  main()
