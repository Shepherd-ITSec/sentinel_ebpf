import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectorConfig:
  port: int = 50051
  events_http_port: int = 50052  # 0 to disable; serves GET /recent_events for UI log tail in gRPC mode
  high_write_bytes: int = 10 * 1024 * 1024
  sensitive_prefixes: List[str] = field(default_factory=lambda: ["/etc", "/bin", "/usr", "/sbin", "/boot"])
  allowed_comms: List[str] = field(default_factory=list)
  # River model configuration
  model_algorithm: str = "halfspacetrees"  # halfspacetrees | loda | memstream
  threshold: float = 0.5  # Anomaly score threshold (0-1)
  hst_n_trees: int = 25
  hst_height: int = 15
  hst_window_size: int = 250
  loda_n_projections: int = 20
  loda_bins: int = 64
  loda_range: float = 3.0
  loda_ema_alpha: float = 0.01
  loda_hist_decay: float = 1.0
  mem_hidden_dim: int = 32
  mem_latent_dim: int = 8
  mem_memory_size: int = 128
  mem_lr: float = 0.001
  model_seed: int = 42


def load_config() -> DetectorConfig:
  port = int(os.environ.get("DETECTOR_PORT", "50051"))
  events_http_port = int(os.environ.get("DETECTOR_EVENTS_PORT", "50052"))
  high_write_bytes = int(os.environ.get("DETECTOR_HIGH_WRITE_BYTES", str(10 * 1024 * 1024)))
  sensitive = os.environ.get("DETECTOR_SENSITIVE_PREFIXES", "/etc,/bin,/usr,/sbin,/boot")
  allowed = os.environ.get("DETECTOR_ALLOWED_COMMS", "")
  
  model_algorithm = os.environ.get("DETECTOR_MODEL_ALGORITHM", "halfspacetrees")
  threshold = float(os.environ.get("DETECTOR_THRESHOLD", "0.5"))
  hst_n_trees = int(os.environ.get("DETECTOR_HST_N_TREES", "25"))
  hst_height = int(os.environ.get("DETECTOR_HST_HEIGHT", "15"))
  hst_window_size = int(os.environ.get("DETECTOR_HST_WINDOW_SIZE", "250"))
  loda_n_projections = int(os.environ.get("DETECTOR_LODA_PROJECTIONS", "20"))
  loda_bins = int(os.environ.get("DETECTOR_LODA_BINS", "64"))
  loda_range = float(os.environ.get("DETECTOR_LODA_RANGE", "3.0"))
  loda_ema_alpha = float(os.environ.get("DETECTOR_LODA_EMA_ALPHA", "0.01"))
  loda_hist_decay = float(os.environ.get("DETECTOR_LODA_HIST_DECAY", "1.0"))
  mem_hidden_dim = int(os.environ.get("DETECTOR_MEMSTREAM_HIDDEN_DIM", "32"))
  mem_latent_dim = int(os.environ.get("DETECTOR_MEMSTREAM_LATENT_DIM", "8"))
  mem_memory_size = int(os.environ.get("DETECTOR_MEMSTREAM_MEMORY_SIZE", "128"))
  mem_lr = float(os.environ.get("DETECTOR_MEMSTREAM_LR", "0.001"))
  model_seed = int(os.environ.get("DETECTOR_MODEL_SEED", "42"))
  
  return DetectorConfig(
    port=port,
    events_http_port=events_http_port,
    high_write_bytes=high_write_bytes,
    sensitive_prefixes=[p for p in sensitive.split(",") if p],
    allowed_comms=[c for c in allowed.split(",") if c],
    model_algorithm=model_algorithm,
    threshold=threshold,
    hst_n_trees=hst_n_trees,
    hst_height=hst_height,
    hst_window_size=hst_window_size,
    loda_n_projections=loda_n_projections,
    loda_bins=loda_bins,
    loda_range=loda_range,
    loda_ema_alpha=loda_ema_alpha,
    loda_hist_decay=loda_hist_decay,
    mem_hidden_dim=mem_hidden_dim,
    mem_latent_dim=mem_latent_dim,
    mem_memory_size=mem_memory_size,
    mem_lr=mem_lr,
    model_seed=model_seed,
  )
