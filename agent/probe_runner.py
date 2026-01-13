import json
import logging
import os
import queue
import signal
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import grpc
from bcc import BPF

import events_pb2
import events_pb2_grpc
from agent.config import AppConfig, load_config
from agent.rules import RuleEngine


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>

struct data_t {
    u64 ts;
    u32 pid;
    u32 tid;
    u32 op; // 0 read, 1 write
    u64 bytes;
    char comm[16];
    char filename[256];
};

BPF_PERF_OUTPUT(events);

int trace_write(struct pt_regs *ctx, struct file *file, const char __user *buf, size_t count, loff_t *pos) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.op = 1;
    data.bytes = count;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    if (file) {
        struct dentry *de = file->f_path.dentry;
        if (de) {
            bpf_probe_read_kernel_str(&data.filename, sizeof(data.filename), de->d_name.name);
        }
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int trace_read(struct pt_regs *ctx, struct file *file, char __user *buf, size_t count, loff_t *pos) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.op = 0;
    data.bytes = count;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    if (file) {
        struct dentry *de = file->f_path.dentry;
        if (de) {
            bpf_probe_read_kernel_str(&data.filename, sizeof(data.filename), de->d_name.name);
        }
    }

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""


def _hostname_metadata():
  return {
    "hostname": socket.gethostname(),
    "node": os.environ.get("NODE_NAME", ""),
    "pod": os.environ.get("POD_NAME", ""),
    "namespace": os.environ.get("POD_NAMESPACE", ""),
    "container_id": os.environ.get("CONTAINER_ID", ""),
  }


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
    self.bpf = BPF(text=BPF_PROGRAM)
    self.bpf.attach_kprobe(event="vfs_write", fn_name="trace_write")
    self.bpf.attach_kprobe(event="vfs_read", fn_name="trace_read")
    self.streamer = GrpcStreamer(cfg)
    self._stop = threading.Event()
    self._metadata = _hostname_metadata()
    self.rule_engine = RuleEngine(cfg.rules_file)

  def _handle_event(self, cpu, data, size):  # noqa: ARG002
    event = self.bpf["events"].event(data)
    filename = event.filename.decode(errors="ignore").rstrip("\x00")
    comm = event.comm.decode(errors="ignore").rstrip("\x00")
    event_type = "file_read" if event.op == 0 else "file_write"

    if not self.rule_engine.allow(event_type, filename, comm):
      return

    env = events_pb2.EventEnvelope(
      event_id=str(uuid.uuid4()),
      hostname=self._metadata["hostname"],
      pod_name=self._metadata["pod"],
      namespace=self._metadata["namespace"],
      container_id=self._metadata["container_id"],
      ts_unix_nano=event.ts,
      event_type=event_type,
      data={
        "filename": filename,
        "bytes": str(event.bytes),
        "comm": comm,
        "pid": str(event.pid),
        "tid": str(event.tid),
      },
    )

    if self.cfg.stream.mode == "stdout":
      payload = {
        "event_type": event_type,
        "event_id": env.event_id,
        "ts_unix_nano": env.ts_unix_nano,
        "hostname": env.hostname,
        "pod": env.pod_name,
        "namespace": env.namespace,
        "data": env.data,
      }
      print(json.dumps(payload))
    else:
      self.streamer.publish(env)

  def run(self):
    self.streamer.start()
    self.bpf["events"].open_perf_buffer(self._handle_event)
    while not self._stop.is_set():
      try:
        self.bpf.perf_buffer_poll(timeout=1000)
      except KeyboardInterrupt:
        self.stop()
      except Exception as exc:  # pylint: disable=broad-except
        logging.exception("polling error: %s", exc)

  def stop(self):
    self._stop.set()
    self.streamer.stop()


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
