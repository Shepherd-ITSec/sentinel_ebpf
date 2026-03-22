"""Tests for probe/config.py and detector/config.py."""
import os

from probe.config import load_config as load_probe_config
from detector.config import load_config as load_detector_config


class TestProbeConfig:
  """Test probe configuration loading."""

  def test_load_default_config(self, temp_dir):
    config_file = temp_dir / "probe-config.yaml"
    config_file.write_text("""logLevel: info
health:
  port: 9101
metrics:
  enabled: true
  port: 9102
rulesFile: /etc/sentinel-ebpf/rules.yaml
stream:
  mode: grpc
grpc:
  endpoint: localhost:50051
""")
    os.environ["PROBE_CONFIG"] = str(config_file)
    try:
      cfg = load_probe_config()
      assert cfg.log_level == "info"
      assert cfg.health_port == 9101
      assert cfg.metrics_enabled is True
      assert cfg.metrics_port == 9102
      assert cfg.rules_file == "/etc/sentinel-ebpf/rules.yaml"
      assert cfg.stream.mode == "grpc"
      assert cfg.stream.endpoint == "localhost:50051"
    finally:
      del os.environ["PROBE_CONFIG"]

  def test_load_file_mode_config(self, temp_dir):
    config_file = temp_dir / "probe-config.yaml"
    config_file.write_text("""stream:
  mode: file
  file:
    path: /var/log/test/events.bin
    rotateMaxBytes: 10485760
    rotateMaxFiles: 3
    compress: true
""")
    os.environ["PROBE_CONFIG"] = str(config_file)
    try:
      cfg = load_probe_config()
      assert cfg.stream.mode == "file"
      assert cfg.stream.file_path == "/var/log/test/events.bin"
      assert cfg.stream.rotate_max_bytes == 10485760
      assert cfg.stream.rotate_max_files == 3
      assert cfg.stream.compress is True
    finally:
      del os.environ["PROBE_CONFIG"]

  def test_load_stdout_mode_config(self, temp_dir):
    config_file = temp_dir / "probe-config.yaml"
    config_file.write_text("""stream:
  mode: stdout
""")
    os.environ["PROBE_CONFIG"] = str(config_file)
    try:
      cfg = load_probe_config()
      assert cfg.stream.mode == "stdout"
    finally:
      del os.environ["PROBE_CONFIG"]

  def test_load_with_batch_settings(self, temp_dir):
    config_file = temp_dir / "probe-config.yaml"
    config_file.write_text("""stream:
  mode: grpc
  batchSize: 128
  queueLength: 2048
  ringBufferPages: 128
grpc:
  endpoint: detector:50051
  tlsEnabled: true
  caSecret: ca-cert
""")
    os.environ["PROBE_CONFIG"] = str(config_file)
    try:
      cfg = load_probe_config()
      assert cfg.stream.batch_size == 128
      assert cfg.stream.queue_length == 2048
      assert cfg.stream.ring_buffer_pages == 128
      assert cfg.stream.endpoint == "detector:50051"
      assert cfg.stream.tls_enabled is True
      assert cfg.stream.ca_secret == "ca-cert"
    finally:
      del os.environ["PROBE_CONFIG"]

  def test_default_values(self, temp_dir):
    config_file = temp_dir / "probe-config.yaml"
    config_file.write_text("{}")
    os.environ["PROBE_CONFIG"] = str(config_file)
    try:
      cfg = load_probe_config()
      assert cfg.log_level == "info"
      assert cfg.health_port == 9101
      assert cfg.metrics_enabled is True
      assert cfg.stream.mode == "grpc"
      assert cfg.stream.endpoint == "localhost:50051"
      assert cfg.stream.batch_size == 512  # Updated default for better performance
      assert cfg.stream.queue_length == 50000  # Updated default for better performance
      assert cfg.stream.ring_buffer_pages == 256  # Updated default for better performance
    finally:
      del os.environ["PROBE_CONFIG"]


class TestDetectorConfig:
  """Test detector configuration loading."""

  def test_load_default_config(self):
    # Clear env vars
    for key in ["DETECTOR_PORT", "DETECTOR_SCORE_MODE"]:
      os.environ.pop(key, None)
    cfg = load_detector_config()
    assert cfg.port == 50051
    assert cfg.score_mode == "raw"

  def test_load_from_env(self):
    os.environ["DETECTOR_PORT"] = "8080"
    os.environ["DETECTOR_MODEL_ALGORITHM"] = "halfspacetrees"
    os.environ["DETECTOR_THRESHOLD"] = "0.7"
    os.environ["DETECTOR_HST_N_TREES"] = "10"
    os.environ["DETECTOR_HST_HEIGHT"] = "12"
    os.environ["DETECTOR_HST_WINDOW_SIZE"] = "128"
    os.environ["DETECTOR_LODA_PROJECTIONS"] = "8"
    os.environ["DETECTOR_LODA_BINS"] = "16"
    os.environ["DETECTOR_LODA_RANGE"] = "2.5"
    os.environ["DETECTOR_LODA_EMA_ALPHA"] = "0.2"
    os.environ["DETECTOR_LODA_HIST_DECAY"] = "0.9"
    os.environ["DETECTOR_MEMSTREAM_MEMORY_SIZE"] = "64"
    os.environ["DETECTOR_MEMSTREAM_LR"] = "0.005"
    os.environ["DETECTOR_MEMSTREAM_BETA"] = "0.2"
    os.environ["DETECTOR_MEMSTREAM_K"] = "5"
    os.environ["DETECTOR_MEMSTREAM_GAMMA"] = "0.6"
    os.environ["DETECTOR_MEMSTREAM_INPUT_MODE"] = "freq1d_z"
    os.environ["DETECTOR_ZSCORE_MIN_COUNT"] = "15"
    os.environ["DETECTOR_ZSCORE_STD_FLOOR"] = "0.002"
    os.environ["DETECTOR_KNN_K"] = "7"
    os.environ["DETECTOR_KNN_MEMORY_SIZE"] = "2048"
    os.environ["DETECTOR_KNN_METRIC"] = "manhattan"
    os.environ["DETECTOR_FREQ1D_BINS"] = "33"
    os.environ["DETECTOR_FREQ1D_ALPHA"] = "0.75"
    os.environ["DETECTOR_FREQ1D_DECAY"] = "0.97"
    os.environ["DETECTOR_FREQ1D_MAX_CATEGORIES"] = "321"
    os.environ["DETECTOR_FREQ1D_AGGREGATION"] = "topk_mean"
    os.environ["DETECTOR_FREQ1D_TOPK"] = "5"
    os.environ["DETECTOR_FREQ1D_SOFT_TOPK_TEMPERATURE"] = "0.3"
    os.environ["DETECTOR_COPULATREE_U_CLAMP"] = "0.0002"
    os.environ["DETECTOR_COPULATREE_REG"] = "0.07"
    os.environ["DETECTOR_COPULATREE_MAX_FEATURES"] = "11"
    os.environ["DETECTOR_COPULATREE_IMPORTANCE_WINDOW"] = "77"
    os.environ["DETECTOR_COPULATREE_TREE_UPDATE_INTERVAL"] = "13"
    os.environ["DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION"] = "topk_mean"
    os.environ["DETECTOR_COPULATREE_EDGE_SCORE_TOPK"] = "4"
    os.environ["DETECTOR_LATENTCLUSTER_MAX_CLUSTERS"] = "9"
    os.environ["DETECTOR_LATENTCLUSTER_U_CLAMP"] = "0.0001"
    os.environ["DETECTOR_LATENTCLUSTER_REG"] = "0.5"
    os.environ["DETECTOR_LATENTCLUSTER_UPDATE_ALPHA"] = "0.2"
    os.environ["DETECTOR_LATENTCLUSTER_SPAWN_THRESHOLD"] = "7.5"
    os.environ["DETECTOR_SCORE_MODE"] = "scaled"
    os.environ["DETECTOR_MODEL_DEVICE"] = "cpu"
    os.environ["DETECTOR_MODEL_SEED"] = "7"
    try:
      cfg = load_detector_config()
      assert cfg.port == 8080
      assert cfg.model_algorithm == "halfspacetrees"
      assert cfg.threshold == 0.7
      assert cfg.hst_n_trees == 10
      assert cfg.hst_height == 12
      assert cfg.hst_window_size == 128
      assert cfg.loda_n_projections == 8
      assert cfg.loda_bins == 16
      assert cfg.loda_range == 2.5
      assert cfg.loda_ema_alpha == 0.2
      assert cfg.loda_hist_decay == 0.9
      assert cfg.mem_memory_size == 64
      assert cfg.mem_lr == 0.005
      assert cfg.mem_beta == 0.2
      assert cfg.mem_k == 5
      assert cfg.mem_gamma == 0.6
      assert cfg.mem_input_mode == "freq1d_z"
      assert cfg.zscore_min_count == 15
      assert cfg.zscore_std_floor == 0.002
      assert cfg.knn_k == 7
      assert cfg.knn_memory_size == 2048
      assert cfg.knn_metric == "manhattan"
      assert cfg.freq1d_bins == 33
      assert cfg.freq1d_alpha == 0.75
      assert cfg.freq1d_decay == 0.97
      assert cfg.freq1d_max_categories == 321
      assert cfg.freq1d_aggregation == "topk_mean"
      assert cfg.freq1d_topk == 5
      assert cfg.freq1d_soft_topk_temperature == 0.3
      assert cfg.copulatree_u_clamp == 0.0002
      assert cfg.copulatree_reg == 0.07
      assert cfg.copulatree_max_features == 11
      assert cfg.copulatree_importance_window == 77
      assert cfg.copulatree_tree_update_interval == 13
      assert cfg.copulatree_edge_score_aggregation == "topk_mean"
      assert cfg.copulatree_edge_score_topk == 4
      assert cfg.latentcluster_max_clusters == 9
      assert cfg.latentcluster_u_clamp == 0.0001
      assert cfg.latentcluster_reg == 0.5
      assert cfg.latentcluster_update_alpha == 0.2
      assert cfg.latentcluster_spawn_threshold == 7.5
      assert cfg.score_mode == "scaled"
      assert cfg.model_device == "cpu"
      assert cfg.model_seed == 7
    finally:
      for key in [
        "DETECTOR_PORT",
        "DETECTOR_MODEL_ALGORITHM",
        "DETECTOR_THRESHOLD",
        "DETECTOR_HST_N_TREES",
        "DETECTOR_HST_HEIGHT",
        "DETECTOR_HST_WINDOW_SIZE",
        "DETECTOR_LODA_PROJECTIONS",
        "DETECTOR_LODA_BINS",
        "DETECTOR_LODA_RANGE",
        "DETECTOR_LODA_EMA_ALPHA",
        "DETECTOR_LODA_HIST_DECAY",
        "DETECTOR_MEMSTREAM_MEMORY_SIZE",
        "DETECTOR_MEMSTREAM_LR",
        "DETECTOR_MEMSTREAM_BETA",
        "DETECTOR_MEMSTREAM_K",
        "DETECTOR_MEMSTREAM_GAMMA",
        "DETECTOR_MEMSTREAM_INPUT_MODE",
        "DETECTOR_ZSCORE_MIN_COUNT",
        "DETECTOR_ZSCORE_STD_FLOOR",
        "DETECTOR_KNN_K",
        "DETECTOR_KNN_MEMORY_SIZE",
        "DETECTOR_KNN_METRIC",
        "DETECTOR_FREQ1D_BINS",
        "DETECTOR_FREQ1D_ALPHA",
        "DETECTOR_FREQ1D_DECAY",
        "DETECTOR_FREQ1D_MAX_CATEGORIES",
        "DETECTOR_FREQ1D_AGGREGATION",
        "DETECTOR_FREQ1D_TOPK",
        "DETECTOR_FREQ1D_SOFT_TOPK_TEMPERATURE",
        "DETECTOR_COPULATREE_U_CLAMP",
        "DETECTOR_COPULATREE_REG",
        "DETECTOR_COPULATREE_MAX_FEATURES",
        "DETECTOR_COPULATREE_IMPORTANCE_WINDOW",
        "DETECTOR_COPULATREE_TREE_UPDATE_INTERVAL",
        "DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION",
        "DETECTOR_COPULATREE_EDGE_SCORE_TOPK",
        "DETECTOR_LATENTCLUSTER_MAX_CLUSTERS",
        "DETECTOR_LATENTCLUSTER_U_CLAMP",
        "DETECTOR_LATENTCLUSTER_REG",
        "DETECTOR_LATENTCLUSTER_UPDATE_ALPHA",
        "DETECTOR_LATENTCLUSTER_SPAWN_THRESHOLD",
        "DETECTOR_SCORE_MODE",
        "DETECTOR_MODEL_DEVICE",
        "DETECTOR_MODEL_SEED",
      ]:
        os.environ.pop(key, None)

  def test_invalid_copulatree_aggregation(self):
    os.environ["DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION"] = "median"
    try:
      try:
        load_detector_config()
        assert False, "expected ValueError"
      except ValueError as exc:
        assert "DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION" in str(exc)
    finally:
      os.environ.pop("DETECTOR_COPULATREE_EDGE_SCORE_AGGREGATION", None)
