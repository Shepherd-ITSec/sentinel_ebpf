import asyncio
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
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
RECENT_EVENTS: deque = deque(maxlen=500)
_events_lock = threading.Lock()

# Metrics counters (thread-safe)
_metrics_lock = threading.Lock()
_metrics = {
  "events_total": 0,
  "anomalies_total": 0,
  "errors_total": 0,
  "last_event_timestamp": 0.0,
}


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
    RECENT_EVENTS.append(entry)


def _now_timestamp() -> timestamp_pb2.Timestamp:
  ts = timestamp_pb2.Timestamp()
  ts.FromDatetime(datetime.now(timezone.utc))
  return ts


class RuleBasedDetector(events_pb2_grpc.DetectorServiceServicer):
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

  def _score_event(self, evt):
    """
    Score an event using River and learn online on every event.
    """
    anomaly = False
    reason = ""
    score = 0.0
    
    # Extract features
    try:
      features = extract_feature_dict(evt)
      score = self.detector.score_and_learn(features)
      anomaly = score >= self.cfg.threshold

      if anomaly:
        reason = f"{self.detector.algorithm} anomaly score {score:.3f} exceeds threshold {self.cfg.threshold}"
    except Exception as e:
      logging.error(f"Error scoring event {evt.event_id}: {e}", exc_info=True)
      # On error, mark as normal
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

  async def StreamEvents(self, request_iterator, context):  # noqa: N802
    async for evt in request_iterator:
      resp = self._score_event(evt)
      _recent_events_append(evt, resp)
      # Update metrics
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
              limit = max(1, min(500, int(qs["limit"][0])))
          except (ValueError, IndexError):
            pass
          with _events_lock:
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
            f"sentinel_ebpf_detector_recent_events_count {len(RECENT_EVENTS)}",
          ]
          body = "\n".join(lines).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
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
