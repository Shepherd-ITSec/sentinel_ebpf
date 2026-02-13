import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ProbeConfig:
  enabled: bool = True
  path_prefixes: List[str] = field(default_factory=list)


@dataclass
class StreamConfig:
  mode: str = "grpc"  # grpc|stdout|file
  endpoint: str = "localhost:50051"
  batch_size: int = 512  # Increased batch size for better gRPC throughput (was 128)
  queue_length: int = 50000  # Increased default queue size (was 10000)
  ring_buffer_pages: int = 256  # Increased ring buffer pages for higher throughput (was 128)
  tls_enabled: bool = False
  ca_secret: str = ""
  file_path: str = "/var/log/sentinel-ebpf/events.bin"
  rotate_max_bytes: int = 50 * 1024 * 1024
  rotate_max_files: int = 3
  compress: bool = False


@dataclass
class AppConfig:
  log_level: str = "info"
  health_port: int = 9101
  metrics_enabled: bool = True
  metrics_port: int = 9102
  rules_file: str = "/etc/sentinel-ebpf/rules.yaml"
  probes: ProbeConfig = field(default_factory=ProbeConfig)
  stream: StreamConfig = field(default_factory=StreamConfig)


def load_config() -> AppConfig:
  path = os.environ.get("PROBE_CONFIG", "/etc/sentinel-ebpf/probe-config.yaml")
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

  probes_cfg = data.get("probes", {}).get("fileWrites", {})
  stream_cfg = data.get("stream", {})
  grpc_cfg = data.get("grpc", {})
  metrics_cfg = data.get("metrics", {})
  health_cfg = data.get("health", {})

  stream = StreamConfig(
    mode=stream_cfg.get("mode", "grpc"),
    endpoint=grpc_cfg.get("endpoint", stream_cfg.get("endpoint", "localhost:50051")),
    batch_size=int(stream_cfg.get("batchSize", 512)),  # Increased default from 64
    queue_length=int(stream_cfg.get("queueLength", 50000)),  # Increased default from 1024
    ring_buffer_pages=int(stream_cfg.get("ringBufferPages", 256)),  # Increased default from 64
    tls_enabled=grpc_cfg.get("tlsEnabled", False),
    ca_secret=grpc_cfg.get("caSecret", ""),
    file_path=stream_cfg.get("file", {}).get("path", "/var/log/sentinel-ebpf/events.bin"),
    rotate_max_bytes=int(stream_cfg.get("file", {}).get("rotateMaxBytes", 50 * 1024 * 1024)),
    rotate_max_files=int(stream_cfg.get("file", {}).get("rotateMaxFiles", 3)),
    compress=bool(stream_cfg.get("file", {}).get("compress", False)),
  )

  probes = ProbeConfig(
    enabled=bool(probes_cfg.get("enabled", True)),
    path_prefixes=list(probes_cfg.get("pathPrefixes", [])),
  )

  return AppConfig(
    log_level=str(data.get("logLevel", "info")),
    health_port=int(health_cfg.get("port", 9101)),
    metrics_enabled=bool(metrics_cfg.get("enabled", True)),
    metrics_port=int(metrics_cfg.get("port", 9102)),
    rules_file=str(data.get("rulesFile", "/etc/sentinel-ebpf/rules.yaml")),
    probes=probes,
    stream=stream,
  )
