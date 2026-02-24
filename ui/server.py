import gzip
import hashlib
import json
import logging
import os
import re
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
_capacity_lock = threading.Lock()
_capacity_prev_events_total = None
_capacity_prev_ts = None
_capacity_prev_drops_total = None


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


def _hash01(value: str) -> float:
  digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
  return (int(digest, 16) % 10000) / 10000.0


def _parse_prometheus_metrics(text: str) -> dict:
  metrics = {}
  if not text or text.startswith("error:"):
    return metrics
  for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$", line)
    if not m:
      continue
    name, value = m.groups()
    try:
      metrics[name] = float(value)
    except ValueError:
      continue
  return metrics


def _to_int(value, default=0) -> int:
  try:
    return int(float(value))
  except (ValueError, TypeError):
    return default


def _compute_capacity_summary(probe_metrics: dict, detector_metrics: dict) -> dict:
  global _capacity_prev_events_total, _capacity_prev_ts, _capacity_prev_drops_total

  has_host_metrics = "sentinel_ebpf_probe_host_cpu_usage_percent" in probe_metrics
  cpu_pct = float(probe_metrics.get("sentinel_ebpf_probe_host_cpu_usage_percent", 0.0))
  mem_pct = float(probe_metrics.get("sentinel_ebpf_probe_host_memory_usage_percent", 0.0))
  load1 = float(probe_metrics.get("sentinel_ebpf_probe_host_load1", 0.0))
  cpu_count = max(1.0, float(probe_metrics.get("sentinel_ebpf_probe_host_cpu_count", 1.0)))
  queue_size = _to_int(probe_metrics.get("sentinel_ebpf_probe_queue_size", 0))
  raw_queue_capacity = _to_int(probe_metrics.get("sentinel_ebpf_probe_queue_capacity", 0), 0)
  queue_capacity = raw_queue_capacity if raw_queue_capacity > 0 else 0
  drops_total = _to_int(probe_metrics.get("sentinel_ebpf_probe_events_dropped_total", 0))
  queue_ratio = min(1.0, queue_size / queue_capacity) if queue_capacity > 0 else None

  now = time.time()
  events_total = detector_metrics.get("sentinel_ebpf_detector_events_total")
  detector_eps = 0.0
  drops_delta = 0
  with _capacity_lock:
    if events_total is not None and _capacity_prev_events_total is not None and _capacity_prev_ts is not None:
      dt = now - _capacity_prev_ts
      de = float(events_total) - float(_capacity_prev_events_total)
      if dt > 0 and de >= 0:
        detector_eps = de / dt
    if _capacity_prev_drops_total is not None:
      drops_delta = max(0, drops_total - int(_capacity_prev_drops_total))
    _capacity_prev_events_total = events_total
    _capacity_prev_ts = now
    _capacity_prev_drops_total = drops_total

  load_per_cpu = load1 / cpu_count
  queue_high = queue_ratio is not None and queue_ratio >= 0.9
  queue_mid = queue_ratio is not None and queue_ratio >= 0.6
  if cpu_pct >= 90 or mem_pct >= 92 or queue_high or drops_delta > 0:
    status = "Saturated"
    action = "increase_detector_replicas_and_reduce_pressure"
    hint = "Node is under pressure. Add detector replicas and reduce ingest/rule scope before increasing load."
    safe_to_scale = False
  elif cpu_pct >= 75 or mem_pct >= 82 or queue_mid or load_per_cpu >= 0.9:
    status = "Near Limit"
    action = "careful_increment"
    hint = "Close to capacity. Increase detector replicas by +1 only and monitor queue/drops closely."
    safe_to_scale = True
  else:
    status = "OK"
    action = "can_scale_gradually"
    hint = "Capacity looks healthy. You can increase detector replicas gradually while monitoring queue and drops."
    safe_to_scale = True

  return {
    "status": status,
    "safe_to_scale": safe_to_scale,
    "recommended_action": action,
    "replica_hint": hint,
    "host": {
      "cpu_usage_percent": round(cpu_pct, 2),
      "memory_usage_percent": round(mem_pct, 2),
      "load1": round(load1, 3),
      "cpu_count": int(cpu_count),
      "load_per_cpu": round(load_per_cpu, 3),
      "has_host_metrics": has_host_metrics,
    },
    "pipeline": {
      "queue_size": queue_size,
      "queue_capacity": queue_capacity,
      "queue_fill_ratio": round(queue_ratio, 3) if queue_ratio is not None else None,
      "drops_total": drops_total,
      "drops_delta": drops_delta,
      "detector_events_per_sec": round(detector_eps, 2),
    },
  }


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
      
      # Get detector algorithm from metrics
      detector_algorithm = None
      if detector_metrics_url:
        try:
          metrics_text = _http_get(detector_metrics_url, timeout=2.0)
          for line in metrics_text.split("\n"):
            # Parse sentinel_ebpf_detector_info{algorithm="memstream"} 1
            if "sentinel_ebpf_detector_info" in line and "algorithm=" in line:
              match = re.search(r'algorithm="([^"]+)"', line)
              if match:
                detector_algorithm = match.group(1)
        except Exception:
          pass
      
      status = {
        "ts": int(time.time()),
        "probe_health": _http_get(probe_health),
        "detector_health": _grpc_health(detector_addr),
        "detector_algorithm": detector_algorithm,
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

    if self.path.startswith("/api/score_map"):
      query = parse_qs(urlparse(self.path).query)
      limit = int(query.get("limit", ["500"])[0])
      limit = max(10, min(limit, 2000))
      log_path = os.environ.get("LOG_PATH", "")
      logs = _read_logs(log_path, limit)
      entries = logs.get("entries", []) if isinstance(logs, dict) else []
      points = []
      for entry in entries:
        data = entry.get("data", []) if isinstance(entry, dict) else []
        path = ""
        comm = ""
        pid = ""
        tid = ""
        uid = ""
        if isinstance(data, list):
          # Canonical vector: [event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]
          path = data[8] if len(data) > 8 else (data[0] if len(data) > 0 else "")
          comm = data[2] if len(data) > 2 else ""
          pid = data[3] if len(data) > 3 else ""
          tid = data[4] if len(data) > 4 else ""
          uid = data[5] if len(data) > 5 else ""
        elif isinstance(entry, dict):
          # Fallback for non-canonical legacy JSON payloads.
          path = str(entry.get("path", ""))
          comm = str(entry.get("comm", ""))
          pid = str(entry.get("pid", ""))
          tid = str(entry.get("tid", ""))
          uid = str(entry.get("uid", ""))
        score = float(entry.get("score", 0.0) or 0.0)
        points.append({
          "x": _hash01(path),
          "y": _hash01(comm),
          "score": max(0.0, min(1.0, score)),
          "anomaly": bool(entry.get("anomaly", False)),
          "reason": entry.get("reason", ""),
          "event_type": entry.get("event_type", ""),
          "hostname": entry.get("hostname", ""),
          "path": path,
          "comm": comm,
          "pid": pid,
          "tid": tid,
          "uid": uid,
          "ts_unix_nano": entry.get("ts_unix_nano", 0),
        })
      self._json({
        "points": points,
        "total": len(points),
        "message": logs.get("message", "") if isinstance(logs, dict) else "",
      })
      return

    if self.path.startswith("/api/capacity"):
      probe_metrics_url = os.environ.get("PROBE_METRICS_URL", "")
      detector_metrics_url = os.environ.get("DETECTOR_METRICS_URL", "")
      probe_text = _http_get(probe_metrics_url) if probe_metrics_url else ""
      detector_text = _http_get(detector_metrics_url) if detector_metrics_url else ""
      probe_metrics = _parse_prometheus_metrics(probe_text)
      detector_metrics = _parse_prometheus_metrics(detector_text)
      summary = _compute_capacity_summary(probe_metrics, detector_metrics)
      summary["ts"] = int(time.time())
      self._json(summary)
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
      # Count event types per poll interval, handling detector restarts robustly.
      global _last_total_events
      detector_events_url = os.environ.get("DETECTOR_EVENTS_URL", "")
      detector_metrics_url = os.environ.get("DETECTOR_METRICS_URL", "")
      counts = {}
      current_time = time.time()

      if detector_events_url:
        try:
          current_total_events = None
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
              current_total_events = None

          # DETECTOR_EVENTS_URL should already point to /recent_events endpoint.
          # Request up to 10000 events (matching detector buffer size).
          if "?" in detector_events_url:
            events_url = f"{detector_events_url}&limit=10000"
          else:
            events_url = f"{detector_events_url}?limit=10000"
          raw = _http_get(events_url, timeout=3.0)
          if raw and not raw.startswith("error:"):
            try:
              data = json.loads(raw)
              entries = data.get("entries", [])

              with _last_poll_lock:
                last_total_events = _last_total_events

              # Fall back to a short time window when metric counters are unavailable
              # or after detector restarts/counter resets.
              def count_recent_window() -> dict:
                window_counts = {}
                recent_window_seconds = 30
                threshold_ns = int(current_time * 1e9) - (recent_window_seconds * 1_000_000_000)
                for entry in entries:
                  event_ts_ns = entry.get("ts_unix_nano", 0)
                  if event_ts_ns > threshold_ns:
                    event_type = entry.get("event_type", "unknown")
                    window_counts[event_type] = window_counts.get(event_type, 0) + 1
                return window_counts

              if current_total_events is None:
                counts = count_recent_window()
              elif last_total_events == 0:
                counts = count_recent_window()
              elif current_total_events < last_total_events:
                # Detector restarted or metrics counter reset.
                counts = count_recent_window()
              else:
                new_events_count = current_total_events - last_total_events
                if new_events_count > 0:
                  # Entries are oldest-first, so the newest events are at the end.
                  new_entries = entries[-new_events_count:] if new_events_count <= len(entries) else entries
                  for entry in new_entries:
                    event_type = entry.get("event_type", "unknown")
                    counts[event_type] = counts.get(event_type, 0) + 1

              # Update baseline counter after processing this poll.
              if current_total_events is not None:
                with _last_poll_lock:
                  _last_total_events = current_total_events

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
