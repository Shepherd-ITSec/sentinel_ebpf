import os
from dataclasses import dataclass


@dataclass
class DetectorConfig:
  port: int = 50051
  events_http_port: int = 50052  # 0 to disable; serves GET /recent_events for UI log tail in gRPC mode
  recent_events_buffer_size: int = 10000  # Size of recent events ring buffer for UI (default: 10000)
  # River model configuration
  model_algorithm: str = "freq1d"  # halfspacetrees | loda_ema | kitnet | memstream | zscore | knn | freq1d | copulatree | latentcluster | sequence_mlp
  threshold: float = 0.9  # Anomaly score threshold (0-1). Lower (e.g. 0.3) to flag more events when scores are mostly low.
  hst_n_trees: int = 25
  hst_height: int = 15
  hst_window_size: int = 250
  loda_n_projections: int = 20
  loda_bins: int = 64
  loda_range: float = 3.0
  loda_ema_alpha: float = 0.01
  loda_hist_decay: float = 1.0
  kitnet_max_size_ae: int = 10
  kitnet_grace_feature_mapping: int = 10000
  kitnet_grace_anomaly_detector: int = 50000
  kitnet_learning_rate: float = 0.1
  kitnet_hidden_ratio: float = 0.75
  mem_memory_size: int = 512
  mem_lr: float = 0.01
  mem_beta: float = 0.1
  mem_k: int = 3
  mem_gamma: float = 0.0  # 0 = 1-NN distance only (paper default)
  mem_input_mode: str = "raw"  # raw | freq1d_u | freq1d_z | freq1d_surprisal | freq1d_z_surprisal
  mem_warmup_accept: int = 512  # grace period: first N events always accepted (no normal-only warmup)
  mem_warmup_path: str = ""  # optional path to JSONL/EVT1 for normal-only warmup (empty = online only)
  zscore_min_count: int = 20
  zscore_std_floor: float = 1e-3
  zscore_topk: int = 8  # mean of the largest K per-feature |z| scores
  knn_k: int = 5
  knn_memory_size: int = 1024
  knn_metric: str = "euclidean"
  freq1d_bins: int = 65536
  freq1d_alpha: float = 1.0
  freq1d_decay: float = 1.0
  freq1d_max_categories: int = 65536
  # Freq1D score aggregation (how per-feature surprisal is combined)
  # - sum: sum of per-feature excess surprisal
  # - mean: mean of per-feature excess surprisal
  # - topk_mean: mean of top-K per-feature excess surprisal
  # - soft_topk_mean: softmax-weighted mean (temperature-controlled)
  freq1d_aggregation: str = "mean"  # sum | mean | topk_mean | soft_topk_mean
  freq1d_topk: int = 8  # used by topk_mean and soft_topk_mean
  freq1d_soft_topk_temperature: float = 0.25  # used by soft_topk_mean; smaller => closer to max
  copulatree_u_clamp: float = 1e-6
  copulatree_reg: float = 0.05
  copulatree_max_features: int = 30  # max features for copula tree (0 = use all)
  copulatree_importance_window: int = 500  # events before first selection; also reselection interval
  copulatree_tree_update_interval: int = 100  # events between maximum-spanning-tree rebuilds
  copulatree_edge_score_aggregation: str = "mean"  # sum | mean | topk_mean
  copulatree_edge_score_topk: int = 8  # used by topk_mean
  latentcluster_max_clusters: int = 8
  latentcluster_u_clamp: float = 1e-6
  latentcluster_reg: float = 0.25
  latentcluster_update_alpha: float = 0.05
  latentcluster_spawn_threshold: float = 6.0
  # score_mode:
  # - raw: threshold on raw model score (unbounded, model-dependent)
  # - scaled: threshold on bounded [0,1] score (most models: 1-exp(-raw))
  # - percentile: threshold on online percentile of (log1p(raw)) per event_group
  score_mode: str = "percentile"  # raw | scaled | percentile
  # percentile calibration window (per event_group): number of past scores kept for percentile estimate
  percentile_window_size: int = 2048
  # warmup samples before percentile thresholding becomes active (before that percentile score is 0)
  percentile_warmup: int = 128
  # auto: use CUDA when available, else CPU for torch-backed models (LODA, MemStream)
  # cpu/cuda: force explicit device choice
  model_device: str = "auto"
  model_seed: int = 42
  # Model-independent warmup gate (per event_group): suppress anomaly decisions and UI score for first N events.
  warmup_events: int = 10000
  suppress_anomalies_during_warmup: bool = True
  # Word2Vec syscall embedding (general feature extractor component).
  embedding_word2vec_dim: int = 5
  embedding_word2vec_sentence_len: int = 7
  embedding_word2vec_window: int = 5
  embedding_word2vec_sg: int = 1  # 1=skip-gram, 0=CBOW
  embedding_word2vec_update_every: int = 250
  embedding_word2vec_epochs: int = 1
  embedding_word2vec_post_warmup_lr_scale: float = 0.1

  # Syscall-sequence next-token prediction (MLP), ported/adapted from LID-DS.
  sequence_ngram_length: int = 8  # full window: N syscalls in buffer; MLP input uses first N-1 embeddings.
  sequence_thread_aware: bool = True
  sequence_mlp_hidden_size: int = 150
  sequence_mlp_hidden_layers: int = 4
  sequence_mlp_lr: float = 0.003


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
  kitnet_max_size_ae = int(os.environ.get("DETECTOR_KITNET_MAX_SIZE_AE", str(defaults.kitnet_max_size_ae)))
  kitnet_grace_feature_mapping = int(
    os.environ.get("DETECTOR_KITNET_GRACE_FEATURE_MAPPING", str(defaults.kitnet_grace_feature_mapping))
  )
  kitnet_grace_anomaly_detector = int(
    os.environ.get("DETECTOR_KITNET_GRACE_ANOMALY_DETECTOR", str(defaults.kitnet_grace_anomaly_detector))
  )
  kitnet_learning_rate = float(os.environ.get("DETECTOR_KITNET_LEARNING_RATE", str(defaults.kitnet_learning_rate)))
  kitnet_hidden_ratio = float(os.environ.get("DETECTOR_KITNET_HIDDEN_RATIO", str(defaults.kitnet_hidden_ratio)))
  mem_memory_size = int(os.environ.get("DETECTOR_MEMSTREAM_MEMORY_SIZE", str(defaults.mem_memory_size)))
  mem_lr = float(os.environ.get("DETECTOR_MEMSTREAM_LR", str(defaults.mem_lr)))
  mem_beta = float(os.environ.get("DETECTOR_MEMSTREAM_BETA", str(defaults.mem_beta)))
  mem_k = int(os.environ.get("DETECTOR_MEMSTREAM_K", str(defaults.mem_k)))
  mem_gamma = float(os.environ.get("DETECTOR_MEMSTREAM_GAMMA", str(defaults.mem_gamma)))
  mem_input_mode = os.environ.get("DETECTOR_MEMSTREAM_INPUT_MODE", defaults.mem_input_mode).strip().lower()
  mem_warmup_accept = int(os.environ.get("DETECTOR_MEMSTREAM_WARMUP_ACCEPT", str(defaults.mem_warmup_accept)))
  mem_warmup_path = (os.environ.get("DETECTOR_MEMSTREAM_WARMUP_PATH", defaults.mem_warmup_path) or "").strip()
  if mem_input_mode not in ("raw", "freq1d_u", "freq1d_z", "freq1d_surprisal", "freq1d_z_surprisal"):
    raise ValueError(
      "Invalid DETECTOR_MEMSTREAM_INPUT_MODE=%r; must be one of: raw, freq1d_u, freq1d_z, freq1d_surprisal, freq1d_z_surprisal"
      % mem_input_mode
    )
  zscore_min_count = int(os.environ.get("DETECTOR_ZSCORE_MIN_COUNT", str(defaults.zscore_min_count)))
  zscore_std_floor = float(os.environ.get("DETECTOR_ZSCORE_STD_FLOOR", str(defaults.zscore_std_floor)))
  zscore_topk = int(os.environ.get("DETECTOR_ZSCORE_TOPK", str(defaults.zscore_topk)))
  if zscore_topk <= 0:
    raise ValueError(f"Invalid DETECTOR_ZSCORE_TOPK={zscore_topk!r}; must be > 0")
  knn_k = int(os.environ.get("DETECTOR_KNN_K", str(defaults.knn_k)))
  knn_memory_size = int(os.environ.get("DETECTOR_KNN_MEMORY_SIZE", str(defaults.knn_memory_size)))
  knn_metric = os.environ.get("DETECTOR_KNN_METRIC", defaults.knn_metric)
  freq1d_bins = int(os.environ.get("DETECTOR_FREQ1D_BINS", str(defaults.freq1d_bins)))
  freq1d_alpha = float(os.environ.get("DETECTOR_FREQ1D_ALPHA", str(defaults.freq1d_alpha)))
  freq1d_decay = float(os.environ.get("DETECTOR_FREQ1D_DECAY", str(defaults.freq1d_decay)))
  freq1d_max_categories = int(os.environ.get("DETECTOR_FREQ1D_MAX_CATEGORIES", str(defaults.freq1d_max_categories)))
  freq1d_aggregation = os.environ.get("DETECTOR_FREQ1D_AGGREGATION", defaults.freq1d_aggregation).strip().lower()
  if freq1d_aggregation not in ("sum", "mean", "topk_mean", "soft_topk_mean"):
    raise ValueError(
      f"Invalid DETECTOR_FREQ1D_AGGREGATION={freq1d_aggregation!r}; must be one of: sum, mean, topk_mean, soft_topk_mean"
    )
  freq1d_topk = int(os.environ.get("DETECTOR_FREQ1D_TOPK", str(defaults.freq1d_topk)))
  if freq1d_topk <= 0:
    raise ValueError(f"Invalid DETECTOR_FREQ1D_TOPK={freq1d_topk!r}; must be > 0")
  freq1d_soft_topk_temperature = float(
    os.environ.get("DETECTOR_FREQ1D_SOFT_TOPK_TEMPERATURE", str(defaults.freq1d_soft_topk_temperature))
  )
  if not (freq1d_soft_topk_temperature > 0.0):
    raise ValueError(
      f"Invalid DETECTOR_FREQ1D_SOFT_TOPK_TEMPERATURE={freq1d_soft_topk_temperature!r}; must be > 0"
    )
  copulatree_u_clamp = float(os.environ.get("DETECTOR_COPULATREE_U_CLAMP", str(defaults.copulatree_u_clamp)))
  if not (0.0 < copulatree_u_clamp < 0.5):
    raise ValueError(f"Invalid DETECTOR_COPULATREE_U_CLAMP={copulatree_u_clamp!r}; must be in (0, 0.5)")
  copulatree_reg = float(os.environ.get("DETECTOR_COPULATREE_REG", str(defaults.copulatree_reg)))
  if not (copulatree_reg > 0.0):
    raise ValueError(f"Invalid DETECTOR_COPULATREE_REG={copulatree_reg!r}; must be > 0")
  copulatree_max_features = int(
    os.environ.get("DETECTOR_COPULATREE_MAX_FEATURES", str(defaults.copulatree_max_features))
  )
  if copulatree_max_features < 0:
    raise ValueError(f"Invalid DETECTOR_COPULATREE_MAX_FEATURES={copulatree_max_features!r}; must be >= 0")
  copulatree_importance_window = int(
    os.environ.get("DETECTOR_COPULATREE_IMPORTANCE_WINDOW", str(defaults.copulatree_importance_window))
  )
  if copulatree_importance_window <= 0:
    raise ValueError(
      f"Invalid DETECTOR_COPULATREE_IMPORTANCE_WINDOW={copulatree_importance_window!r}; must be > 0"
    )
  copulatree_tree_update_interval = int(
    os.environ.get("DETECTOR_COPULATREE_TREE_UPDATE_INTERVAL", str(defaults.copulatree_tree_update_interval))
  )
  if copulatree_tree_update_interval <= 0:
    raise ValueError(
      f"Invalid DETECTOR_COPULATREE_TREE_UPDATE_INTERVAL={copulatree_tree_update_interval!r}; must be > 0"
    )
  copulatree_edge_score_aggregation = os.environ.get(
    "DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION",
    defaults.copulatree_edge_score_aggregation,
  ).strip().lower()
  if copulatree_edge_score_aggregation not in ("sum", "mean", "topk_mean"):
    raise ValueError(
      "Invalid DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION=%r; must be one of: sum, mean, topk_mean"
      % copulatree_edge_score_aggregation
    )
  copulatree_edge_score_topk = int(
    os.environ.get("DETECTOR_COPULATREE_EDGE_SCORE_TOPK", str(defaults.copulatree_edge_score_topk))
  )
  if copulatree_edge_score_topk <= 0:
    raise ValueError(f"Invalid DETECTOR_COPULATREE_EDGE_SCORE_TOPK={copulatree_edge_score_topk!r}; must be > 0")
  latentcluster_max_clusters = int(
    os.environ.get("DETECTOR_LATENTCLUSTER_MAX_CLUSTERS", str(defaults.latentcluster_max_clusters))
  )
  if latentcluster_max_clusters <= 0:
    raise ValueError(f"Invalid DETECTOR_LATENTCLUSTER_MAX_CLUSTERS={latentcluster_max_clusters!r}; must be > 0")
  latentcluster_u_clamp = float(
    os.environ.get("DETECTOR_LATENTCLUSTER_U_CLAMP", str(defaults.latentcluster_u_clamp))
  )
  if not (0.0 < latentcluster_u_clamp < 0.5):
    raise ValueError(f"Invalid DETECTOR_LATENTCLUSTER_U_CLAMP={latentcluster_u_clamp!r}; must be in (0, 0.5)")
  latentcluster_reg = float(os.environ.get("DETECTOR_LATENTCLUSTER_REG", str(defaults.latentcluster_reg)))
  if not (latentcluster_reg > 0.0):
    raise ValueError(f"Invalid DETECTOR_LATENTCLUSTER_REG={latentcluster_reg!r}; must be > 0")
  latentcluster_update_alpha = float(
    os.environ.get("DETECTOR_LATENTCLUSTER_UPDATE_ALPHA", str(defaults.latentcluster_update_alpha))
  )
  if not (0.0 < latentcluster_update_alpha <= 1.0):
    raise ValueError(
      f"Invalid DETECTOR_LATENTCLUSTER_UPDATE_ALPHA={latentcluster_update_alpha!r}; must be in (0, 1]"
    )
  latentcluster_spawn_threshold = float(
    os.environ.get("DETECTOR_LATENTCLUSTER_SPAWN_THRESHOLD", str(defaults.latentcluster_spawn_threshold))
  )
  if not (latentcluster_spawn_threshold > 0.0):
    raise ValueError(
      f"Invalid DETECTOR_LATENTCLUSTER_SPAWN_THRESHOLD={latentcluster_spawn_threshold!r}; must be > 0"
    )
  score_mode = os.environ.get("DETECTOR_SCORE_MODE", defaults.score_mode).strip().lower()
  if score_mode not in ("raw", "scaled", "percentile"):
    raise ValueError(f"Invalid DETECTOR_SCORE_MODE={score_mode!r}; must be 'raw', 'scaled', or 'percentile'")
  percentile_window_size = int(os.environ.get("DETECTOR_PERCENTILE_WINDOW_SIZE", str(defaults.percentile_window_size)))
  if percentile_window_size < 32:
    raise ValueError(
      f"Invalid DETECTOR_PERCENTILE_WINDOW_SIZE={percentile_window_size!r}; must be >= 32"
    )
  percentile_warmup = int(os.environ.get("DETECTOR_PERCENTILE_WARMUP", str(defaults.percentile_warmup)))
  if percentile_warmup < 0:
    raise ValueError(
      f"Invalid DETECTOR_PERCENTILE_WARMUP={percentile_warmup!r}; must be >= 0"
    )
  model_device = os.environ.get("DETECTOR_MODEL_DEVICE", defaults.model_device).strip().lower()
  model_seed = int(os.environ.get("DETECTOR_MODEL_SEED", str(defaults.model_seed)))
  warmup_events = int(os.environ.get("DETECTOR_WARMUP_EVENTS", str(defaults.warmup_events)))
  if warmup_events < 0:
    raise ValueError(f"Invalid DETECTOR_WARMUP_EVENTS={warmup_events!r}; must be >= 0")
  _swg = os.environ.get(
    "DETECTOR_SUPPRESS_ANOMALIES_DURING_WARMUP",
    str(defaults.suppress_anomalies_during_warmup),
  ).strip().lower()
  suppress_anomalies_during_warmup = _swg in ("1", "true", "yes")

  sequence_ngram_length = int(os.environ.get("DETECTOR_SEQUENCE_NGRAM_LENGTH", str(defaults.sequence_ngram_length)))
  if sequence_ngram_length < 2:
    raise ValueError(f"Invalid DETECTOR_SEQUENCE_NGRAM_LENGTH={sequence_ngram_length!r}; must be >= 2")
  _ta = os.environ.get("DETECTOR_SEQUENCE_THREAD_AWARE", str(defaults.sequence_thread_aware)).strip().lower()
  sequence_thread_aware = _ta in ("1", "true", "yes")
  embedding_word2vec_dim = int(
    os.environ.get("DETECTOR_EMBEDDING_WORD2VEC_DIM", str(defaults.embedding_word2vec_dim))
  )
  if embedding_word2vec_dim < 1:
    raise ValueError(f"Invalid DETECTOR_EMBEDDING_WORD2VEC_DIM={embedding_word2vec_dim!r}; must be >= 1")
  embedding_word2vec_sentence_len = int(
    os.environ.get(
      "DETECTOR_EMBEDDING_WORD2VEC_SENTENCE_LEN",
      str(defaults.embedding_word2vec_sentence_len),
    )
  )
  if embedding_word2vec_sentence_len < 2:
    raise ValueError(
      f"Invalid DETECTOR_EMBEDDING_WORD2VEC_SENTENCE_LEN={embedding_word2vec_sentence_len!r}; must be >= 2"
    )
  embedding_word2vec_window = int(
    os.environ.get("DETECTOR_EMBEDDING_WORD2VEC_WINDOW", str(defaults.embedding_word2vec_window))
  )
  embedding_word2vec_sg = int(
    os.environ.get("DETECTOR_EMBEDDING_WORD2VEC_SG", str(defaults.embedding_word2vec_sg))
  )
  if embedding_word2vec_sg not in (0, 1):
    raise ValueError(f"Invalid DETECTOR_EMBEDDING_WORD2VEC_SG={embedding_word2vec_sg!r}; must be 0 or 1")
  embedding_word2vec_update_every = int(
    os.environ.get(
      "DETECTOR_EMBEDDING_WORD2VEC_UPDATE_EVERY",
      str(defaults.embedding_word2vec_update_every),
    )
  )
  if embedding_word2vec_update_every < 1:
    raise ValueError(
      f"Invalid DETECTOR_EMBEDDING_WORD2VEC_UPDATE_EVERY={embedding_word2vec_update_every!r}; must be >= 1"
    )
  embedding_word2vec_epochs = int(
    os.environ.get("DETECTOR_EMBEDDING_WORD2VEC_EPOCHS", str(defaults.embedding_word2vec_epochs))
  )
  embedding_word2vec_post_warmup_lr_scale = float(
    os.environ.get(
      "DETECTOR_EMBEDDING_WORD2VEC_POST_WARMUP_LR_SCALE",
      str(defaults.embedding_word2vec_post_warmup_lr_scale),
    )
  )
  sequence_mlp_hidden_size = int(os.environ.get("DETECTOR_SEQUENCE_MLP_HIDDEN_SIZE", str(defaults.sequence_mlp_hidden_size)))
  sequence_mlp_hidden_layers = int(
    os.environ.get("DETECTOR_SEQUENCE_MLP_HIDDEN_LAYERS", str(defaults.sequence_mlp_hidden_layers))
  )
  sequence_mlp_lr = float(os.environ.get("DETECTOR_SEQUENCE_MLP_LR", str(defaults.sequence_mlp_lr)))

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
    kitnet_max_size_ae=kitnet_max_size_ae,
    kitnet_grace_feature_mapping=kitnet_grace_feature_mapping,
    kitnet_grace_anomaly_detector=kitnet_grace_anomaly_detector,
    kitnet_learning_rate=kitnet_learning_rate,
    kitnet_hidden_ratio=kitnet_hidden_ratio,
    mem_memory_size=mem_memory_size,
    mem_lr=mem_lr,
    mem_beta=mem_beta,
    mem_k=mem_k,
    mem_gamma=mem_gamma,
    mem_input_mode=mem_input_mode,
    mem_warmup_accept=mem_warmup_accept,
    mem_warmup_path=mem_warmup_path,
    zscore_min_count=zscore_min_count,
    zscore_std_floor=zscore_std_floor,
    zscore_topk=zscore_topk,
    knn_k=knn_k,
    knn_memory_size=knn_memory_size,
    knn_metric=knn_metric,
    freq1d_bins=freq1d_bins,
    freq1d_alpha=freq1d_alpha,
    freq1d_decay=freq1d_decay,
    freq1d_max_categories=freq1d_max_categories,
    freq1d_aggregation=freq1d_aggregation,
    freq1d_topk=freq1d_topk,
    freq1d_soft_topk_temperature=freq1d_soft_topk_temperature,
    copulatree_u_clamp=copulatree_u_clamp,
    copulatree_reg=copulatree_reg,
    copulatree_max_features=copulatree_max_features,
    copulatree_importance_window=copulatree_importance_window,
    copulatree_tree_update_interval=copulatree_tree_update_interval,
    copulatree_edge_score_aggregation=copulatree_edge_score_aggregation,
    copulatree_edge_score_topk=copulatree_edge_score_topk,
    latentcluster_max_clusters=latentcluster_max_clusters,
    latentcluster_u_clamp=latentcluster_u_clamp,
    latentcluster_reg=latentcluster_reg,
    latentcluster_update_alpha=latentcluster_update_alpha,
    latentcluster_spawn_threshold=latentcluster_spawn_threshold,
    score_mode=score_mode,
    percentile_window_size=percentile_window_size,
    percentile_warmup=percentile_warmup,
    model_device=model_device,
    model_seed=model_seed,
    warmup_events=warmup_events,
    suppress_anomalies_during_warmup=suppress_anomalies_during_warmup,
    embedding_word2vec_dim=embedding_word2vec_dim,
    embedding_word2vec_sentence_len=embedding_word2vec_sentence_len,
    embedding_word2vec_window=embedding_word2vec_window,
    embedding_word2vec_sg=embedding_word2vec_sg,
    embedding_word2vec_update_every=embedding_word2vec_update_every,
    embedding_word2vec_epochs=embedding_word2vec_epochs,
    embedding_word2vec_post_warmup_lr_scale=embedding_word2vec_post_warmup_lr_scale,
    sequence_ngram_length=sequence_ngram_length,
    sequence_thread_aware=sequence_thread_aware,
    sequence_mlp_hidden_size=sequence_mlp_hidden_size,
    sequence_mlp_hidden_layers=sequence_mlp_hidden_layers,
    sequence_mlp_lr=sequence_mlp_lr,
  )
