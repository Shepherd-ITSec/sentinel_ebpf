"""Shared pytest fixtures."""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
  """Create a temporary directory for test files."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield Path(tmpdir)


@pytest.fixture
def sample_rules_yaml(temp_dir):
  """Create a sample rules.yaml file."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text("""rules:
  - name: capture-all-opens
    enabled: true
    event: file_open
    match:
      pathPrefixes: ["/"]
  - name: capture-sensitive-opens
    enabled: true
    event: file_open
    match:
      pathPrefixes: ["/etc", "/bin"]
  - name: capture-specific-comm
    enabled: true
    event: file_open
    match:
      comms: ["bash", "sh"]
  - name: disabled-rule
    enabled: false
    event: file_open
    match:
      pathPrefixes: ["/tmp"]
""")
  return rules_file


@pytest.fixture
def sample_probe_config_yaml(temp_dir):
  """Create a sample probe-config.yaml file."""
  config_file = temp_dir / "probe-config.yaml"
  config_file.write_text("""logLevel: debug
health:
  port: 9101
metrics:
  enabled: true
  port: 9102
rulesFile: /etc/sentinel-ebpf/rules.yaml
stream:
  mode: grpc
  batchSize: 32
  queueLength: 512
  file:
    path: /var/log/sentinel-ebpf/events.bin
    rotateMaxBytes: 10485760
    rotateMaxFiles: 5
    compress: false
grpc:
  endpoint: localhost:50051
  tlsEnabled: false
probes:
  fileWrites:
    enabled: true
    pathPrefixes: []
""")
  return config_file
