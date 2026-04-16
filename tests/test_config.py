"""Tests for probe/config.py and detector/config.py."""
import os

import pytest

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

  def test_load_from_env_overrides_representative_fields(self, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DETECTOR_PORT", "8080")
    monkeypatch.setenv("DETECTOR_PIPELINE_ID", "las_gas_fusion")
    monkeypatch.setenv("DETECTOR_THRESHOLD", "0.7")
    monkeypatch.setenv("DETECTOR_SCORE_MODE", "scaled")
    monkeypatch.setenv("DETECTOR_MODEL_DEVICE", "cpu")
    monkeypatch.setenv("DETECTOR_MODEL_SEED", "7")
    monkeypatch.setenv("DETECTOR_FREQ1D_AGGREGATION", "topk_mean")
    monkeypatch.setenv("DETECTOR_FREQ1D_TOPK", "5")
    monkeypatch.setenv("DETECTOR_MEMSTREAM_INPUT_MODE", "freq1d_z")
    monkeypatch.setenv("DETECTOR_SEQUENCE_NGRAM_LENGTH", "5")
    monkeypatch.setenv("DETECTOR_EMBEDDING_WORD2VEC_UPDATE_EVERY", "3")

    cfg = load_detector_config()

    assert cfg.port == 8080
    assert cfg.pipeline_id == "las_gas_fusion"
    assert cfg.threshold == 0.7
    assert cfg.score_mode == "scaled"
    assert cfg.model_device == "cpu"
    assert cfg.model_seed == 7
    assert cfg.freq1d_aggregation == "topk_mean"
    assert cfg.freq1d_topk == 5
    assert cfg.mem_input_mode == "freq1d_z"
    assert cfg.sequence_ngram_length == 5
    assert cfg.embedding_word2vec_update_every == 3

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

  @pytest.mark.parametrize(
    ("env_name", "env_value", "expected_fragment"),
    [
      ("DETECTOR_PIPELINE_ID", "legacy", "registered non-legacy pipeline"),
      ("DETECTOR_FREQ1D_TOPK", "0", "DETECTOR_FREQ1D_TOPK"),
      ("DETECTOR_SEQUENCE_NGRAM_LENGTH", "1", "DETECTOR_SEQUENCE_NGRAM_LENGTH"),
    ],
  )
  def test_invalid_detector_env_values_raise(
    self,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    expected_fragment: str,
  ):
    monkeypatch.setenv(env_name, env_value)
    with pytest.raises(ValueError, match=expected_fragment):
      load_detector_config()
