import gzip
import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

MAGIC = b"EVT1"


def _http_get(url: str, timeout: float = 2.0) -> str:
  if not url:
    return ""
  try:
    req = Request(url, headers={"User-Agent": "sentinel-ebpf-ui"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
      return resp.read().decode("utf-8", errors="replace")
  except Exception as exc:  # noqa: BLE001
    return f"error: {exc}"


def _grpc_health(addr: str, timeout: float = 2.0) -> str:
  if not addr:
    return ""
  try:
    with grpc.insecure_channel(addr) as channel:
      stub = health_pb2_grpc.HealthStub(channel)
      resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=timeout)
      return health_pb2.HealthCheckResponse.ServingStatus.Name(resp.status)
  except Exception as exc:  # noqa: BLE001
    return f"error: {exc}"


def _open_stream(path: Path):
  with path.open("rb") as f:
    head = f.read(2)
  if head == b"\x1f\x8b":
    return gzip.open(path, "rb")
  return path.open("rb")


def _tail_evt1(path: Path, limit: int):
  entries = deque(maxlen=limit)
  with _open_stream(path) as f:
    while True:
      magic = f.read(4)
      if not magic:
        break
      if magic != MAGIC:
        break
      raw_len = f.read(4)
      if len(raw_len) < 4:
        break
      length = int.from_bytes(raw_len, "little")
      payload = f.read(length)
      if len(payload) < length:
        break
      try:
        entries.append(json.loads(payload.decode("utf-8")))
      except json.JSONDecodeError:
        continue
  return list(entries)


def _tail_ndjson(path: Path, limit: int):
  entries = deque(maxlen=limit)
  with _open_stream(path) as f:
    for raw in f:
      try:
        entries.append(json.loads(raw.decode("utf-8")))
      except json.JSONDecodeError:
        continue
  return list(entries)


def _fetch_recent_events_from_detector(events_url: str, limit: int):
  """Fetch recent events from detector's HTTP API (gRPC mode log tail)."""
  if not events_url:
    return None
  try:
    url = f"{events_url}?limit={limit}" if "?" not in events_url else f"{events_url}&limit={limit}"
    raw = _http_get(url, timeout=3.0)
    if raw.startswith("error:"):
      return None
    data = json.loads(raw)
    return data.get("entries", [])
  except Exception:  # noqa: BLE001
    return None


def _read_logs(path_str: str, limit: int):
  if not path_str:
    return {"message": "LOG_PATH not set", "entries": []}
  path = Path(path_str)
  if not path.exists():
    # gRPC mode: try detector recent-events API
    events_url = os.environ.get("DETECTOR_EVENTS_URL", "")
    entries = _fetch_recent_events_from_detector(events_url, limit) if events_url else None
    if entries is not None:
      return {"message": "Live tail from detector (gRPC stream).", "entries": entries}
    return {
      "message": "No log file at this path. When the probe streams to the detector (gRPC), it does not write to a file; use file mode and a shared volume to see logs here.",
      "entries": [],
    }
  try:
    with _open_stream(path) as f:
      magic = f.read(4)
    if magic == MAGIC:
      return {"entries": _tail_evt1(path, limit)}
    return {"entries": _tail_ndjson(path, limit)}
  except Exception as exc:  # noqa: BLE001
    return {"error": str(exc), "entries": []}


def _load_html_template() -> str:
  path = Path(__file__).resolve().parent / "index.html"
  return path.read_text(encoding="utf-8")


_HTML_TEMPLATE = _load_html_template()


class Handler(BaseHTTPRequestHandler):
  def _json(self, obj, status=200):
    data = json.dumps(obj).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def do_GET(self):  # noqa: N802
    if self.path == "/" or self.path.startswith("/index"):
      poll = os.environ.get("UI_POLL_SECONDS", "10")
      limit = os.environ.get("UI_LOG_LIMIT", "50")
      html = _HTML_TEMPLATE.replace("%POLL_SECONDS%", poll).replace("%LOG_LIMIT%", limit)
      data = html.encode("utf-8")
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)
      return

    if self.path.startswith("/api/status"):
      probe_health = os.environ.get("PROBE_HEALTH_URL", "")
      detector_addr = os.environ.get("DETECTOR_GRPC_ADDR", "")
      status = {
        "ts": int(time.time()),
        "probe_health": _http_get(probe_health),
        "detector_health": _grpc_health(detector_addr),
      }
      self._json(status)
      return

    if self.path.startswith("/api/metrics"):
      probe_metrics = os.environ.get("PROBE_METRICS_URL", "")
      detector_metrics = os.environ.get("DETECTOR_METRICS_URL", "")
      metrics = {
        "ts": int(time.time()),
        "probe_metrics": _http_get(probe_metrics) if probe_metrics else "(not configured)",
        "detector_metrics": _http_get(detector_metrics) if detector_metrics else "(not configured)",
      }
      self._json(metrics)
      return

    if self.path.startswith("/api/logs"):
      query = parse_qs(urlparse(self.path).query)
      limit = int(query.get("limit", [os.environ.get("UI_LOG_LIMIT", "50")])[0])
      limit = max(1, min(limit, 500))
      log_path = os.environ.get("LOG_PATH", "")
      self._json(_read_logs(log_path, limit))
      return

    self.send_response(404)
    self.end_headers()

  def log_message(self, format, *args):  # noqa: A003
    return


def main():
  port = int(os.environ.get("UI_PORT", "8080"))
  server = HTTPServer(("0.0.0.0", port), Handler)
  server.serve_forever()


if __name__ == "__main__":
  main()
