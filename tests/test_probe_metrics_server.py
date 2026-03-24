"""Probe Prometheus endpoint on metrics port."""

import pytest
from urllib.request import urlopen

pytest.importorskip("bcc", reason="bcc not installed; probe module depends on bcc.")

from probe.probe_runner import start_metrics_server


def test_metrics_server_exposes_prometheus_text():
  srv = start_metrics_server(0, probe_runner=None)
  try:
    host, port = srv.server_address
    with urlopen(f"http://{host}:{port}/metrics", timeout=2) as resp:
      body = resp.read().decode("utf-8")
    assert "sentinel_ebpf_probe_queue_size" in body
    assert "# TYPE sentinel_ebpf_probe_queue_size gauge" in body
  finally:
    srv.shutdown()
