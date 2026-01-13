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


def _now_timestamp() -> timestamp_pb2.Timestamp:
  ts = timestamp_pb2.Timestamp()
  ts.FromDatetime(datetime.now(timezone.utc))
  return ts


class RuleBasedDetector(events_pb2_grpc.DetectorServiceServicer):
  def __init__(self, cfg: DetectorConfig):
    self.cfg = cfg

  def _score_event(self, evt: events_pb2.EventEnvelope) -> events_pb2.DetectionResponse:
    # Pass-through baseline: detector logs only. Real anomaly models should replace this.
    anomaly = False
    reason = ""
    score = 0.0
    return events_pb2.DetectionResponse(
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
  logging.info("detector listening on %s", listen_addr)
  await server.start()
  try:
    await server.wait_for_termination()
  except KeyboardInterrupt:
    logging.info("shutting down detector")
    await server.stop(grace=None)


def main():
  asyncio.run(serve())


if __name__ == "__main__":
  main()
