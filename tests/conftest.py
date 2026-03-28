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
  rules_file.write_text("""lists:
  file_syscalls: [open, openat, openat2]
  process_syscalls: [execve]
  network_syscalls: [connect]
groups:
  file:
    syscalls: file_syscalls
  process:
    syscalls: process_syscalls
  network:
    syscalls: network_syscalls
rules:
  - name: capture-all-file-events
    enabled: true
    group: file
    condition: "syscall_name in (file_syscalls) and attributes.fd_path startswith /"
  - name: capture-sensitive-file-events
    enabled: true
    group: file
    condition: "syscall_name in (file_syscalls) and (attributes.fd_path startswith /etc or attributes.fd_path startswith /bin)"
  - name: capture-specific-comm
    enabled: true
    group: process
    condition: "syscall_name in (process_syscalls) and comm in (bash, sh)"
  - name: disabled-rule
    enabled: false
    group: network
    condition: "syscall_name in (network_syscalls) and attributes.fd_path startswith /tmp"
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
