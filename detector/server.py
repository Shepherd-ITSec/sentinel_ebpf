import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

# Import Optional for type hints
from urllib.parse import parse_qs, urlparse

import grpc
from google.protobuf import empty_pb2, timestamp_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

import events_pb2
import events_pb2_grpc
from detector.config import DetectorConfig, load_config
from detector.features import extract_feature_dict
from detector.model import OnlineAnomalyDetector
from scripts.replay_logs import replay  # type: ignore

# Ring buffer of recent events for UI log tail in gRPC mode (GET /recent_events).
# Size is set during initialization from config
RECENT_EVENTS: Optional[deque] = None
_events_lock = threading.Lock()

# Anomaly list for UI (all anomalies, not just recent)
ANOMALIES: deque = deque(maxlen=1000)  # Keep last 1000 anomalies
_anomalies_lock = threading.Lock()

# Anomaly log file
_anomaly_log_path: Optional[Path] = None
_anomaly_log_lock = threading.Lock()

# Metrics counters (thread-safe)
_metrics_lock = threading.Lock()
_metrics = {
  "events_total": 0,
  "anomalies_total": 0,
  "errors_total": 0,
  "last_event_timestamp": 0.0,
}


def _init_anomaly_log():
  """Initialize anomaly log file if configured."""
  global _anomaly_log_path
  log_path = os.environ.get("ANOMALY_LOG_PATH", "")
  if log_path:
    _anomaly_log_path = Path(log_path)
    _anomaly_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Anomaly logging enabled: %s", _anomaly_log_path)


def _log_anomaly(evt, resp) -> None:
  """Log anomaly to file if configured."""
  if not _anomaly_log_path or not resp.anomaly:
    return
  try:
    entry = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "event_id": evt.event_id,
      "event_type": evt.event_type or "",
      "data": list(evt.data) if evt.data else [],
      "hostname": evt.hostname or "",
      "pod_name": evt.pod_name or "",
      "namespace": evt.namespace or "",
      "ts_unix_nano": evt.ts_unix_nano,
      "score": round(resp.score, 4),
      "reason": resp.reason or "",
    }
    with _anomaly_log_lock:
      with _anomaly_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
  except Exception as e:
    logging.error(f"Failed to log anomaly to file: {e}", exc_info=True)


def _recent_events_append(evt, resp) -> None:
  entry = {
    "event_id": evt.event_id,
    "event_type": evt.event_type or "",
    "data": list(evt.data) if evt.data else [],
    "hostname": evt.hostname or "",
    "ts_unix_nano": evt.ts_unix_nano,
    "anomaly": resp.anomaly,
    "score": round(resp.score, 4),
    "reason": resp.reason or "",
  }
  with _events_lock:
    if RECENT_EVENTS is not None:
      RECENT_EVENTS.append(entry)
  
  # Add to anomalies list if it's an anomaly
  if resp.anomaly:
    anomaly_entry = {
      "event_id": evt.event_id,
      "event_type": evt.event_type or "",
      "data": list(evt.data) if evt.data else [],
      "hostname": evt.hostname or "",
      "pod_name": evt.pod_name or "",
      "namespace": evt.namespace or "",
      "ts_unix_nano": evt.ts_unix_nano,
      "score": round(resp.score, 4),
      "reason": resp.reason or "",
    }
    with _anomalies_lock:
      ANOMALIES.append(anomaly_entry)
    
    # Log to file
    _log_anomaly(evt, resp)


def _now_timestamp() -> timestamp_pb2.Timestamp:
  ts = timestamp_pb2.Timestamp()
  ts.FromDatetime(datetime.now(timezone.utc))
  return ts


class DeterministicScorer:
  """Single-model scorer that preserves deterministic learning semantics."""

  def __init__(self, cfg: DetectorConfig):
    self.cfg = cfg
    self.detector = OnlineAnomalyDetector(
      algorithm=cfg.model_algorithm,
      hst_n_trees=cfg.hst_n_trees,
      hst_height=cfg.hst_height,
      hst_window_size=cfg.hst_window_size,
      loda_n_projections=cfg.loda_n_projections,
      loda_bins=cfg.loda_bins,
      loda_range=cfg.loda_range,
      loda_ema_alpha=cfg.loda_ema_alpha,
      loda_hist_decay=cfg.loda_hist_decay,
      mem_hidden_dim=cfg.mem_hidden_dim,
      mem_latent_dim=cfg.mem_latent_dim,
      mem_memory_size=cfg.mem_memory_size,
      mem_lr=cfg.mem_lr,
      seed=cfg.model_seed,
    )
    self._lock = threading.Lock()

  def score_event(self, evt: events_pb2.EventEnvelope) -> events_pb2.DetectionResponse:
    anomaly = False
    reason = ""
    score = 0.0

    try:
      with self._lock:
        features = extract_feature_dict(evt)
        score = self.detector.score_and_learn(features)
        anomaly = score >= self.cfg.threshold

      if anomaly:
        reason = f"{self.detector.algorithm} anomaly score {score:.3f} exceeds threshold {self.cfg.threshold}"
    except Exception as e:
      logging.error("Error scoring event %s: %s", evt.event_id, e, exc_info=True)
      anomaly = False
      score = 0.0
      reason = f"Scoring error: {str(e)}"

    return events_pb2.DetectionResponse(  # type: ignore[attr-defined]
      event_id=evt.event_id,
      anomaly=anomaly,
      reason=reason,
      score=min(score, 1.0),
      ts=_now_timestamp(),
    )


class RuleBasedDetector(events_pb2_grpc.DetectorServiceServicer):
  def __init__(self, cfg: DetectorConfig):
    self.cfg = cfg
    self.scorer = DeterministicScorer(cfg)
    self.active_worker_count = 1
    self.configured_worker_count = max(1, cfg.worker_count)
    if self.configured_worker_count > 1:
      logging.warning(
        "DETECTOR_WORKER_COUNT=%d is configured but deterministic mode uses a single scoring model.",
        self.configured_worker_count,
      )
    logging.info("Initialized detector in deterministic single-model mode")
  
  def _score_event(self, evt: events_pb2.EventEnvelope) -> events_pb2.DetectionResponse:
    return self.scorer.score_event(evt)

  async def StreamEvents(self, request_iterator, context):  # noqa: N802
    async for evt in request_iterator:
      resp = self._score_event(evt)
      _recent_events_append(evt, resp)

      with _metrics_lock:
        _metrics["events_total"] += 1
        _metrics["last_event_timestamp"] = time.time()
        if resp.anomaly:
          _metrics["anomalies_total"] += 1
        if resp.reason and "error" in resp.reason.lower():
          _metrics["errors_total"] += 1

      if resp.anomaly:
        logging.warning("anomaly id=%s reason=%s score=%.3f", resp.event_id, resp.reason, resp.score)
      else:
        logging.debug("event ok id=%s", resp.event_id)

      yield resp

  async def ReportAnomaly(self, request, context):  # noqa: N802
    logging.warning("reported anomaly id=%s reason=%s score=%.3f labels=%s", request.event_id, request.reason, request.score, dict(request.labels))
    return empty_pb2.Empty()


async def serve():
  global RECENT_EVENTS
  cfg = load_config()
  
  # Initialize recent events buffer with configured size
  RECENT_EVENTS = deque(maxlen=cfg.recent_events_buffer_size)
  logging.info("Initialized recent events buffer with size %d", cfg.recent_events_buffer_size)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  _init_anomaly_log()
  server = grpc.aio.server()
  events_pb2_grpc.add_DetectorServiceServicer_to_server(RuleBasedDetector(cfg), server)

  health_svc = health.HealthServicer()
  health_pb2_grpc.add_HealthServicer_to_server(health_svc, server)
  health_svc.set("", health_pb2.HealthCheckResponse.SERVING)

  listen_addr = f"[::]:{cfg.port}"
  server.add_insecure_port(listen_addr)
  await server.start()
  logging.info("detector listening on %s", listen_addr)

  events_http_server = None
  if getattr(cfg, "events_http_port", 0) and cfg.events_http_port > 0:
    class DetectorHTTPHandler(BaseHTTPRequestHandler):
      def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/recent_events":
          limit = 50
          try:
            qs = parse_qs(parsed.query)
            if "limit" in qs:
              limit = max(1, min(10000, int(qs["limit"][0])))  # Increased max limit to match buffer size
          except (ValueError, IndexError):
            pass
          with _events_lock:
            if RECENT_EVENTS is None:
              entries = []
            else:
              entries = list(RECENT_EVENTS)[-limit:]
          body = json.dumps({"entries": entries}).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
        elif parsed.path == "/metrics":
          # Prometheus-format metrics
          with _metrics_lock:
            m = _metrics.copy()
          lines = [
            "# HELP sentinel_ebpf_detector_events_total Total number of events processed",
            "# TYPE sentinel_ebpf_detector_events_total counter",
            f"sentinel_ebpf_detector_events_total {m['events_total']}",
            "",
            "# HELP sentinel_ebpf_detector_anomalies_total Total number of anomalies detected",
            "# TYPE sentinel_ebpf_detector_anomalies_total counter",
            f"sentinel_ebpf_detector_anomalies_total {m['anomalies_total']}",
            "",
            "# HELP sentinel_ebpf_detector_errors_total Total number of scoring errors",
            "# TYPE sentinel_ebpf_detector_errors_total counter",
            f"sentinel_ebpf_detector_errors_total {m['errors_total']}",
            "",
            "# HELP sentinel_ebpf_detector_last_event_timestamp_seconds Unix timestamp of last processed event",
            "# TYPE sentinel_ebpf_detector_last_event_timestamp_seconds gauge",
            f"sentinel_ebpf_detector_last_event_timestamp_seconds {m['last_event_timestamp']:.3f}",
            "",
            "# HELP sentinel_ebpf_detector_recent_events_count Current number of events in recent buffer",
            "# TYPE sentinel_ebpf_detector_recent_events_count gauge",
            f"sentinel_ebpf_detector_recent_events_count {len(RECENT_EVENTS) if RECENT_EVENTS is not None else 0}",
            "",
            "# HELP sentinel_ebpf_detector_worker_count Number of parallel workers",
            "# TYPE sentinel_ebpf_detector_worker_count gauge",
            "sentinel_ebpf_detector_worker_count 1",
            "",
            "# HELP sentinel_ebpf_detector_worker_configured_count Configured worker count (for compatibility)",
            "# TYPE sentinel_ebpf_detector_worker_configured_count gauge",
            f"sentinel_ebpf_detector_worker_configured_count {cfg.worker_count}",
            "",
            "# HELP sentinel_ebpf_detector_info Detector metadata (algorithm name)",
            "# TYPE sentinel_ebpf_detector_info gauge",
            f'sentinel_ebpf_detector_info{{algorithm="{cfg.model_algorithm}"}} 1',
          ]
          body = "\n".join(lines).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
        elif parsed.path == "/anomalies":
          # Return list of all anomalies
          limit = 1000
          try:
            qs = parse_qs(parsed.query)
            if "limit" in qs:
              limit = max(1, min(5000, int(qs["limit"][0])))
          except (ValueError, IndexError):
            pass
          with _anomalies_lock:
            entries = list(ANOMALIES)[-limit:]
          body = json.dumps({"entries": entries, "total": len(ANOMALIES)}).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
        else:
          self.send_response(404)
          self.end_headers()
      def log_message(self, format, *args):  # noqa: A003
        return
    events_http_server = HTTPServer(("0.0.0.0", cfg.events_http_port), DetectorHTTPHandler)
    thread = threading.Thread(target=events_http_server.serve_forever, daemon=True)
    thread.start()
    logging.info("detector HTTP API on port %s (/recent_events, /metrics)", cfg.events_http_port)

  try:
    await server.wait_for_termination()
  except KeyboardInterrupt:
    logging.info("shutting down detector")
    if events_http_server:
      events_http_server.shutdown()
    await server.stop(grace=None)


def main():
  import argparse

  parser = argparse.ArgumentParser(description="Detector service with optional replay test mode")
  parser.add_argument("--replay-log", help="Path to EVT1 log to replay to the detector")
  parser.add_argument("--replay-pace", default="fast", choices=["fast", "realtime"], help="Replay pacing")
  parser.add_argument("--replay-start-ms", type=int, default=None, help="Start timestamp ms")
  parser.add_argument("--replay-end-ms", type=int, default=None, help="End timestamp ms")
  args = parser.parse_args()

  if args.replay_log:
    # Start server in background loop and replay into it.
    async def run_with_replay():
      cfg = load_config()
      logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
      server = grpc.aio.server()
      events_pb2_grpc.add_DetectorServiceServicer_to_server(RuleBasedDetector(cfg), server)
      health_svc = health.HealthServicer()
      health_pb2_grpc.add_HealthServicer_to_server(health_svc, server)
      health_svc.set("", health_pb2.HealthCheckResponse.SERVING)
      listen_addr = f"[::]:{cfg.port}"
      server.add_insecure_port(listen_addr)
      await server.start()
      logging.info("detector listening on %s", listen_addr)
      # Replay from log into this server.
      replay(args.replay_log, f"localhost:{cfg.port}", args.replay_pace, args.replay_start_ms, args.replay_end_ms)
      await server.wait_for_termination()

    asyncio.run(run_with_replay())
  else:
    asyncio.run(serve())


if __name__ == "__main__":
  main()
