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
    for key in ["DETECTOR_PORT"]:
      os.environ.pop(key, None)
    cfg = load_detector_config()
    assert cfg.port == 50051

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
    os.environ["DETECTOR_MEMSTREAM_HIDDEN_DIM"] = "12"
    os.environ["DETECTOR_MEMSTREAM_LATENT_DIM"] = "6"
    os.environ["DETECTOR_MEMSTREAM_MEMORY_SIZE"] = "64"
    os.environ["DETECTOR_MEMSTREAM_LR"] = "0.005"
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
      assert cfg.mem_hidden_dim == 12
      assert cfg.mem_latent_dim == 6
      assert cfg.mem_memory_size == 64
      assert cfg.mem_lr == 0.005
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
        "DETECTOR_MEMSTREAM_HIDDEN_DIM",
        "DETECTOR_MEMSTREAM_LATENT_DIM",
        "DETECTOR_MEMSTREAM_MEMORY_SIZE",
        "DETECTOR_MEMSTREAM_LR",
        "DETECTOR_MODEL_DEVICE",
        "DETECTOR_MODEL_SEED",
      ]:
        os.environ.pop(key, None)
