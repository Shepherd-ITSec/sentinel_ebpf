import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict
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

# Event dump file (all incoming events)
_event_dump_path: Optional[Path] = None
_event_dump_lock = threading.Lock()

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
    # Pre-create the file so eval runs always have an explicit artifact,
    # even when zero anomalies are emitted.
    _anomaly_log_path.touch(exist_ok=True)
    logging.info("Anomaly logging enabled: %s", _anomaly_log_path)


def _init_event_dump(cfg: Optional[DetectorConfig] = None):
  """Initialize event dump file if configured (all incoming events). Writes metadata header when file is empty."""
  global _event_dump_path
  dump_path = os.environ.get("EVENT_DUMP_PATH", "")
  if dump_path:
    _event_dump_path = Path(dump_path)
    _event_dump_path.parent.mkdir(parents=True, exist_ok=True)
    _event_dump_path.touch(exist_ok=True)
    # Write metadata header when file is empty (new run)
    if _event_dump_path.stat().st_size == 0 and cfg is not None:
      meta = {
        "_meta": True,
        "date": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
      }
      with _event_dump_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    logging.info("Event dump enabled: %s", _event_dump_path)


def _log_anomaly(evt, resp) -> None:
  """Log anomaly to file if configured."""
  if not _anomaly_log_path or not resp.anomaly:
    return
  try:
    entry = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "event_id": evt.event_id,
      "event_name": evt.event_name or "",
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


def _dump_event(evt, resp) -> None:
  """Append one JSONL line for the event (and detection result) to the event dump file if configured."""
  if not _event_dump_path:
    return
  try:
    entry = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "event_id": evt.event_id,
      "event_name": evt.event_name or "",
      "event_type": evt.event_type or "",
      "data": list(evt.data) if evt.data else [],
      "hostname": evt.hostname or "",
      "pod_name": evt.pod_name or "",
      "namespace": evt.namespace or "",
      "container_id": evt.container_id or "",
      "attributes": dict(evt.attributes or {}),
      "ts_unix_nano": evt.ts_unix_nano,
      "anomaly": resp.anomaly,
      "score": round(resp.score, 4),
      "reason": resp.reason or "",
    }
    with _event_dump_lock:
      with _event_dump_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
  except Exception as e:
    logging.error("Failed to dump event to file: %s", e, exc_info=True)


def _recent_events_append(evt, resp) -> None:
  entry = {
    "event_id": evt.event_id,
    "event_name": evt.event_name or "",
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
      "event_name": evt.event_name or "",
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

  # Dump all events to file if configured
  _dump_event(evt, resp)


def _now_timestamp() -> timestamp_pb2.Timestamp:
  ts = timestamp_pb2.Timestamp()
  ts.FromDatetime(datetime.now(timezone.utc))
  return ts


class DeterministicScorer:
  """Per-event_type scorer that preserves deterministic learning semantics.

  We maintain one OnlineAnomalyDetector instance per event_type (including the
  empty/default type). Each model sees a consistent feature vector shape, since
  extract_feature_dict() always returns the same feature set for a given type.
  """

  def __init__(self, cfg: DetectorConfig):
    self.cfg = cfg
    self._models: dict[str, OnlineAnomalyDetector] = {}
    self._lock = threading.Lock()

  def _key_for_type(self, event_type: str) -> str:
    t = (event_type or "").strip().lower()
    return t or "__default__"

  def _get_model(self, event_type: str) -> OnlineAnomalyDetector:
    key = self._key_for_type(event_type)
    model = self._models.get(key)
    if model is None:
      model = OnlineAnomalyDetector(
        algorithm=self.cfg.model_algorithm,
        hst_n_trees=self.cfg.hst_n_trees,
        hst_height=self.cfg.hst_height,
        hst_window_size=self.cfg.hst_window_size,
        loda_n_projections=self.cfg.loda_n_projections,
        loda_bins=self.cfg.loda_bins,
        loda_range=self.cfg.loda_range,
        loda_ema_alpha=self.cfg.loda_ema_alpha,
        loda_hist_decay=self.cfg.loda_hist_decay,
        kitnet_max_size_ae=self.cfg.kitnet_max_size_ae,
        kitnet_grace_feature_mapping=self.cfg.kitnet_grace_feature_mapping,
        kitnet_grace_anomaly_detector=self.cfg.kitnet_grace_anomaly_detector,
        kitnet_learning_rate=self.cfg.kitnet_learning_rate,
        kitnet_hidden_ratio=self.cfg.kitnet_hidden_ratio,
        mem_hidden_dim=self.cfg.mem_hidden_dim,
        mem_latent_dim=self.cfg.mem_latent_dim,
        mem_memory_size=self.cfg.mem_memory_size,
        mem_lr=self.cfg.mem_lr,
        zscore_min_count=self.cfg.zscore_min_count,
        zscore_std_floor=self.cfg.zscore_std_floor,
        knn_k=self.cfg.knn_k,
        knn_memory_size=self.cfg.knn_memory_size,
        knn_metric=self.cfg.knn_metric,
        freq1d_bins=self.cfg.freq1d_bins,
        freq1d_alpha=self.cfg.freq1d_alpha,
        freq1d_decay=self.cfg.freq1d_decay,
        freq1d_max_categories=self.cfg.freq1d_max_categories,
        model_device=self.cfg.model_device,
        seed=self.cfg.model_seed,
      )
      self._models[key] = model
      logging.info("Initialized model for event_type=%r (algorithm=%s)", key, model.algorithm)
    return model

  def score_event(self, evt: events_pb2.EventEnvelope) -> events_pb2.DetectionResponse:
    anomaly = False
    reason = ""
    score = 0.0

    try:
      with self._lock:
        features = extract_feature_dict(evt)
        model = self._get_model(evt.event_type or "")
        if getattr(self.cfg, "score_mode", "raw") == "scaled":
          score = model.score_and_learn(features)
        else:
          score = model.score_and_learn_raw(features)
        anomaly = score >= self.cfg.threshold

      if anomaly:
        reason = f"{model.algorithm} anomaly score {score:.3f} exceeds threshold {self.cfg.threshold}"
    except Exception as e:
      logging.error("Error scoring event %s: %s", evt.event_id, e, exc_info=True)
      anomaly = False
      score = 0.0
      reason = f"Scoring error: {str(e)}"

    resp_score = float(score)
    if getattr(self.cfg, "score_mode", "raw") == "scaled":
      resp_score = min(resp_score, 1.0)

    return events_pb2.DetectionResponse(  # type: ignore[attr-defined]
      event_id=evt.event_id,
      anomaly=anomaly,
      reason=reason,
      score=resp_score,
      ts=_now_timestamp(),
    )


class RuleBasedDetector(events_pb2_grpc.DetectorServiceServicer):
  def __init__(self, cfg: DetectorConfig):
    self.cfg = cfg
    self.scorer = DeterministicScorer(cfg)
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
        n = _metrics["events_total"]
        anomalies = _metrics["anomalies_total"]

      if not os.environ.get("DETECTOR_QUIET"):
        if (n % 10000) == 0 and n > 0:
          logging.info("processed %d events (%d anomalies so far)", n, anomalies)
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
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  logging.info("loading config...")
  cfg = load_config()
  logging.info("config loaded: algorithm=%s threshold=%.2f gRPC_port=%d HTTP_port=%d", cfg.model_algorithm, cfg.threshold, cfg.port, cfg.events_http_port)

  logging.info("initializing anomaly model (%s)...", cfg.model_algorithm)
  RECENT_EVENTS = deque(maxlen=cfg.recent_events_buffer_size)
  _init_anomaly_log()
  _init_event_dump(cfg)
  servicer = RuleBasedDetector(cfg)
  logging.info("model initialized (recent_events buffer size=%d)", cfg.recent_events_buffer_size)

  logging.info("starting gRPC server...")
  server = grpc.aio.server()
  events_pb2_grpc.add_DetectorServiceServicer_to_server(servicer, server)
  health_svc = health.HealthServicer()
  health_pb2_grpc.add_HealthServicer_to_server(health_svc, server)
  health_svc.set("", health_pb2.HealthCheckResponse.SERVING)

  listen_addr = f"[::]:{cfg.port}"
  logging.info("binding gRPC port %s...", listen_addr)
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
            "# HELP sentinel_ebpf_detector_info Detector metadata (algorithm name)",
            "# TYPE sentinel_ebpf_detector_info gauge",
            f'sentinel_ebpf_detector_info{{algorithm="{cfg.model_algorithm}"}} 1',
            "",
            "# HELP sentinel_ebpf_detector_threshold Anomaly score threshold (events with score >= this are flagged)",
            "# TYPE sentinel_ebpf_detector_threshold gauge",
            f"sentinel_ebpf_detector_threshold {cfg.threshold}",
          ]
          body = "\n".join(lines).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
          self.send_header("Content-Length", str(len(body)))
          self.end_headers()
          self.wfile.write(body)
        elif parsed.path == "/events_dump":
          # Search or tail the event dump file (when EVENT_DUMP_PATH is set).
          # ?search=cat+shells - server-side search, returns only matching events (scans last N lines).
          # ?limit=N (no search) - returns last N events.
          limit = 50000
          search = ""
          try:
            qs = parse_qs(parsed.query)
            if "limit" in qs:
              limit = max(1, min(100000, int(qs["limit"][0])))
            if "search" in qs and qs["search"]:
              search = (qs["search"][0] or "").strip()
          except (ValueError, IndexError):
            pass

          def _event_matches_search(obj: dict, query: str) -> bool:
            """Match event against search query (same logic as UI eventMatchesFilter)."""
            if not query:
              return True
            data = obj.get("data") or []
            comm = (data[2] if len(data) > 2 else "") or obj.get("attributes", {}).get("comm", "")
            path = (data[8] if len(data) > 8 else "") or obj.get("attributes", {}).get("path", "") or obj.get("attributes", {}).get("filename", "")
            event_name = (obj.get("event_name") or "").lower()
            path_s = (path or "").lower()
            comm_s = (comm or "").lower()
            hostname = (obj.get("hostname") or "").lower()
            event_type = (obj.get("event_type") or "").lower()
            data_joined = " ".join(str(x) for x in data).lower()
            haystack = f"{event_name} {comm_s} {path_s} {hostname} {event_type} {data_joined}"
            parts = query.strip().split()
            for part in parts:
              if not part:
                continue
              m = re.match(r"^(\w+)=(.+)$", part)
              if m:
                field, value = m.group(1).lower(), m.group(2).lower()
                if field == "comm" and value not in comm_s:
                  return False
                if field in ("path", "file") and value not in path_s:
                  return False
                if field in ("event", "event_name", "syscall") and value not in event_name:
                  return False
                if field == "hostname" and value not in hostname:
                  return False
                if field == "type" and value not in event_type:
                  return False
              elif part.lower() not in haystack:
                return False
            return True

          entries = []
          if _event_dump_path and _event_dump_path.exists():
            try:
              with _event_dump_path.open("r", encoding="utf-8") as f:
                lines = deque(f, maxlen=limit)
              for line in lines:
                line = line.strip()
                if not line:
                  continue
                try:
                  obj = json.loads(line)
                  if obj.get("_meta"):
                    continue
                  if "event_id" not in obj:
                    continue
                  if search and not _event_matches_search(obj, search):
                    continue
                  entries.append(obj)
                except json.JSONDecodeError:
                  continue
            except Exception as e:
              logging.warning("events_dump read failed: %s", e)
          body = json.dumps({"entries": entries}).encode("utf-8")
          self.send_response(200)
          self.send_header("Content-Type", "application/json")
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

  logging.info("ready; accepting events.")
  try:
    await server.wait_for_termination()
  except KeyboardInterrupt:
    logging.info("shutting down detector")
    if events_http_server:
      events_http_server.shutdown()
    await server.stop(grace=None)


def main():
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  logging.info("detector process starting (imports and setup may take a moment)...")
  asyncio.run(serve())


if __name__ == "__main__":
  main()
