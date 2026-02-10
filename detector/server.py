import asyncio
import logging
import time
from datetime import datetime, timezone

import grpc
from google.protobuf import empty_pb2, timestamp_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

import events_pb2
import events_pb2_grpc
from detector.config import DetectorConfig, load_config
from detector.features import extract_feature_dict
from detector.model import OnlineAnomalyDetector
from scripts.replay_logs import replay  # type: ignore


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
  try:
    await server.wait_for_termination()
  except KeyboardInterrupt:
    logging.info("shutting down detector")
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
