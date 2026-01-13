import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectorConfig:
  port: int = 50051
  high_write_bytes: int = 10 * 1024 * 1024
  sensitive_prefixes: List[str] = field(default_factory=lambda: ["/etc", "/bin", "/usr", "/sbin", "/boot"])
  allowed_comms: List[str] = field(default_factory=list)


def load_config() -> DetectorConfig:
  port = int(os.environ.get("DETECTOR_PORT", "50051"))
  high_write_bytes = int(os.environ.get("DETECTOR_HIGH_WRITE_BYTES", str(10 * 1024 * 1024)))
  sensitive = os.environ.get("DETECTOR_SENSITIVE_PREFIXES", "/etc,/bin,/usr,/sbin,/boot")
  allowed = os.environ.get("DETECTOR_ALLOWED_COMMS", "")
  return DetectorConfig(
    port=port,
    high_write_bytes=high_write_bytes,
    sensitive_prefixes=[p for p in sensitive.split(",") if p],
    allowed_comms=[c for c in allowed.split(",") if c],
  )
