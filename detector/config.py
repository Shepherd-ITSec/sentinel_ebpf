import os
from dataclasses import dataclass


@dataclass
class DetectorConfig:
  port: int = 50051
  events_http_port: int = 50052  # 0 to disable; serves GET /recent_events for UI log tail in gRPC mode
  recent_events_buffer_size: int = 10000  # Size of recent events ring buffer for UI (default: 10000)
  # River model configuration
  model_algorithm: str = "memstream"  # halfspacetrees | loda | memstream
  threshold: float = 0.9  # Anomaly score threshold (0-1). Lower (e.g. 0.3) to flag more events when scores are mostly low.
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
  recent_events_buffer_size_str = os.environ.get("DETECTOR_RECENT_EVENTS_BUFFER_SIZE", "10000")
  recent_events_buffer_size = int(recent_events_buffer_size_str) if recent_events_buffer_size_str else 10000
  events_http_port = int(os.environ.get("DETECTOR_EVENTS_PORT", "50052"))
  
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
    recent_events_buffer_size=recent_events_buffer_size,
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
