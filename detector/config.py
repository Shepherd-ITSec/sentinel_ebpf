import os
from dataclasses import dataclass


@dataclass
class DetectorConfig:
  port: int = 50051
  events_http_port: int = 50052  # 0 to disable; serves GET /recent_events for UI log tail in gRPC mode
  recent_events_buffer_size: int = 10000  # Size of recent events ring buffer for UI (default: 10000)
  # River model configuration
  model_algorithm: str = "halfspacetrees"  # halfspacetrees | loda | memstream
  threshold: float = 0.7  # Anomaly score threshold (0-1). Lower (e.g. 0.3) to flag more events when scores are mostly low.
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
  defaults = DetectorConfig()
  port = int(os.environ.get("DETECTOR_PORT", str(defaults.port)))
  recent_events_buffer_size_str = os.environ.get("DETECTOR_RECENT_EVENTS_BUFFER_SIZE", str(defaults.recent_events_buffer_size))
  recent_events_buffer_size = int(recent_events_buffer_size_str) if recent_events_buffer_size_str else defaults.recent_events_buffer_size
  events_http_port = int(os.environ.get("DETECTOR_EVENTS_PORT", str(defaults.events_http_port)))

  model_algorithm = os.environ.get("DETECTOR_MODEL_ALGORITHM", defaults.model_algorithm)
  threshold = float(os.environ.get("DETECTOR_THRESHOLD", str(defaults.threshold)))
  hst_n_trees = int(os.environ.get("DETECTOR_HST_N_TREES", str(defaults.hst_n_trees)))
  hst_height = int(os.environ.get("DETECTOR_HST_HEIGHT", str(defaults.hst_height)))
  hst_window_size = int(os.environ.get("DETECTOR_HST_WINDOW_SIZE", str(defaults.hst_window_size)))
  loda_n_projections = int(os.environ.get("DETECTOR_LODA_PROJECTIONS", str(defaults.loda_n_projections)))
  loda_bins = int(os.environ.get("DETECTOR_LODA_BINS", str(defaults.loda_bins)))
  loda_range = float(os.environ.get("DETECTOR_LODA_RANGE", str(defaults.loda_range)))
  loda_ema_alpha = float(os.environ.get("DETECTOR_LODA_EMA_ALPHA", str(defaults.loda_ema_alpha)))
  loda_hist_decay = float(os.environ.get("DETECTOR_LODA_HIST_DECAY", str(defaults.loda_hist_decay)))
  mem_hidden_dim = int(os.environ.get("DETECTOR_MEMSTREAM_HIDDEN_DIM", str(defaults.mem_hidden_dim)))
  mem_latent_dim = int(os.environ.get("DETECTOR_MEMSTREAM_LATENT_DIM", str(defaults.mem_latent_dim)))
  mem_memory_size = int(os.environ.get("DETECTOR_MEMSTREAM_MEMORY_SIZE", str(defaults.mem_memory_size)))
  mem_lr = float(os.environ.get("DETECTOR_MEMSTREAM_LR", str(defaults.mem_lr)))
  model_seed = int(os.environ.get("DETECTOR_MODEL_SEED", str(defaults.model_seed)))

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
