import gzip
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

logging.basicConfig(level=logging.WARNING)

# Track the number of events we've seen in the buffer (to detect new events)
# Since the buffer rotates, we track the total count from the detector
_last_total_events = 0
_last_poll_lock = threading.Lock()

# Track per-poll counts for time-series chart (rolling window of last N polls)
_poll_history = deque(maxlen=20)  # Keep last 20 poll intervals
_poll_history_lock = threading.Lock()

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
      detector_metrics_url = os.environ.get("DETECTOR_METRICS_URL", "")
      
      # Get detector worker count from metrics
      worker_count = None
      if detector_metrics_url:
        try:
          metrics_text = _http_get(detector_metrics_url, timeout=2.0)
          for line in metrics_text.split("\n"):
            if line.startswith("sentinel_ebpf_detector_worker_count"):
              parts = line.split()
              if len(parts) >= 2:
                worker_count = int(float(parts[1]))
                break
        except Exception:
          pass
      
      status = {
        "ts": int(time.time()),
        "probe_health": _http_get(probe_health),
        "detector_health": _grpc_health(detector_addr),
        "detector_worker_count": worker_count,
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

    if self.path.startswith("/api/anomalies"):
      query = parse_qs(urlparse(self.path).query)
      limit = int(query.get("limit", ["1000"])[0])
      limit = max(1, min(limit, 5000))
      detector_events_url = os.environ.get("DETECTOR_EVENTS_URL", "")
      if detector_events_url:
        # Extract base URL (remove /recent_events if present)
        base_url = detector_events_url.replace("/recent_events", "")
        anomalies_url = f"{base_url}/anomalies?limit={limit}"
        try:
          raw = _http_get(anomalies_url, timeout=3.0)
          if not raw.startswith("error:"):
            data = json.loads(raw)
            self._json(data)
            return
        except Exception:
          pass
      self._json({"entries": [], "total": 0})
      return

    if self.path.startswith("/api/calls_chart"):
      # Get recent events and count by event_type for current poll interval
      # Use total events count from metrics to track new events since last poll
      global _last_total_events
      detector_events_url = os.environ.get("DETECTOR_EVENTS_URL", "")
      detector_metrics_url = os.environ.get("DETECTOR_METRICS_URL", "")
      counts = {}
      
      current_time = time.time()
      
      if detector_events_url:
        try:
          # Get current total events processed from metrics
          current_total_events = 0
          if detector_metrics_url:
            try:
              metrics_text = _http_get(detector_metrics_url, timeout=2.0)
              for line in metrics_text.split("\n"):
                if line.startswith("sentinel_ebpf_detector_events_total"):
                  parts = line.split()
                  if len(parts) >= 2:
                    current_total_events = int(float(parts[1]))
                    break
            except Exception:
              pass
          
          # DETECTOR_EVENTS_URL should already point to /recent_events endpoint
          # Request up to 10000 events (matching detector buffer size)
          if "?" in detector_events_url:
            events_url = f"{detector_events_url}&limit=10000"
          else:
            events_url = f"{detector_events_url}?limit=10000"
          raw = _http_get(events_url, timeout=3.0)
          if raw and not raw.startswith("error:"):
            try:
              data = json.loads(raw)
              entries = data.get("entries", [])
              
              # Get the last total events count
              with _last_poll_lock:
                last_total_events = _last_total_events
                is_first_poll = (last_total_events == 0)
              
              if is_first_poll:
                # First poll: initialize counter and show recent events immediately
                # Use a reasonable window (last 30 seconds) to show activity right away
                # This provides immediate feedback while avoiding huge spikes of very old events
                recent_window_seconds = 30
                recent_threshold_ns = int(current_time * 1e9) - (recent_window_seconds * 1_000_000_000)
                
                for entry in entries:
                  event_ts_ns = entry.get("ts_unix_nano", 0)
                  # Count events from the last 30 seconds
                  if event_ts_ns > recent_threshold_ns:
                    event_type = entry.get("event_type", "unknown")
                    counts[event_type] = counts.get(event_type, 0) + 1
                
                # Always initialize the counter on first poll, even if no recent events
                # This ensures subsequent polls can track new events immediately
                with _last_poll_lock:
                  _last_total_events = current_total_events if current_total_events > 0 else 0
              else:
                # Subsequent polls: count events that arrived since last poll
                # Calculate how many new events were processed
                new_events_count = current_total_events - last_total_events
                
                if new_events_count > 0:
                  # Count the newest events in the buffer (up to new_events_count)
                  # Since entries are returned oldest-first (first N entries), 
                  # the newest events are at the end
                  new_entries = entries[-new_events_count:] if new_events_count <= len(entries) else entries
                  
                  for entry in new_entries:
                    event_type = entry.get("event_type", "unknown")
                    counts[event_type] = counts.get(event_type, 0) + 1
                
                # Update the last total events count
                with _last_poll_lock:
                  _last_total_events = current_total_events
              
              # Store this poll's data in history
              poll_time = current_time
              with _poll_history_lock:
                _poll_history.append({
                  "time": poll_time,
                  "counts": counts.copy() if counts else {}
                })
            except json.JSONDecodeError as e:
              logging.error(f"Failed to parse JSON from detector: {e}, raw={raw[:200]}")
            except Exception as e:
              logging.error(f"Error processing chart data: {e}", exc_info=True)
        except Exception as e:
          logging.debug(f"Error fetching chart data: {e}")
      # Return time-series data: last N polls with counts per event type
      with _poll_history_lock:
        history = list(_poll_history)
      
      # Format data for chart: time buckets with counts per event type
      time_buckets = []
      for poll_data in history:
        poll_time_iso = datetime.fromtimestamp(poll_data["time"], tz=timezone.utc).strftime("%H:%M:%S")
        time_buckets.append({
          "time": poll_time_iso,
          "timestamp": poll_data["time"],
          "counts": poll_data["counts"]
        })
      
      self._json({"time_buckets": time_buckets})
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
