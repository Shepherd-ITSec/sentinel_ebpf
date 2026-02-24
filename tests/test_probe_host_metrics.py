"""Tests for probe host metrics sampler."""

import pytest

pytest.importorskip("bcc", reason="bcc not installed; probe module depends on bcc.")

from probe.probe_runner import HostMetricsSampler


def test_host_metrics_sampler_collect_shape():
  sampler = HostMetricsSampler()
  sample = sampler.collect()
  assert "cpu_usage_percent" in sample
  assert "memory_usage_percent" in sample
  assert "load1" in sample
  assert "cpu_count" in sample
  assert sample["cpu_count"] >= 1
