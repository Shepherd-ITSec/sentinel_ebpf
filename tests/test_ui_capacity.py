"""Tests for UI capacity and score-map helpers."""

from ui.server import _compute_capacity_summary, _parse_prometheus_metrics


def test_parse_prometheus_metrics_handles_labels():
  raw = """
# HELP demo Demo metric
demo_metric_total 12
demo_with_label{foo="bar"} 3.5
"""
  metrics = _parse_prometheus_metrics(raw)
  assert metrics["demo_metric_total"] == 12.0
  assert metrics["demo_with_label"] == 3.5


def test_compute_capacity_summary_ok_status():
  probe = {
    "sentinel_ebpf_probe_host_cpu_usage_percent": 35.0,
    "sentinel_ebpf_probe_host_memory_usage_percent": 42.0,
    "sentinel_ebpf_probe_host_load1": 0.8,
    "sentinel_ebpf_probe_host_cpu_count": 4,
    "sentinel_ebpf_probe_queue_size": 100,
    "sentinel_ebpf_probe_queue_capacity": 5000,
    "sentinel_ebpf_probe_events_dropped_total": 0,
  }
  detector = {"sentinel_ebpf_detector_events_total": 1000}
  summary = _compute_capacity_summary(probe, detector)
  assert summary["status"] == "OK"
  assert summary["safe_to_scale"] is True
  assert "replica_hint" in summary


def test_compute_capacity_summary_saturated_on_drops():
  base_probe = {
    "sentinel_ebpf_probe_host_cpu_usage_percent": 40.0,
    "sentinel_ebpf_probe_host_memory_usage_percent": 40.0,
    "sentinel_ebpf_probe_host_load1": 0.5,
    "sentinel_ebpf_probe_host_cpu_count": 4,
    "sentinel_ebpf_probe_queue_size": 10,
    "sentinel_ebpf_probe_queue_capacity": 5000,
    "sentinel_ebpf_probe_events_dropped_total": 0,
  }
  detector = {"sentinel_ebpf_detector_events_total": 1000}
  _compute_capacity_summary(base_probe, detector)

  probe_with_new_drops = dict(base_probe)
  probe_with_new_drops["sentinel_ebpf_probe_events_dropped_total"] = 5
  summary = _compute_capacity_summary(probe_with_new_drops, detector)
  assert summary["status"] == "Saturated"
  assert summary["safe_to_scale"] is False


def test_compute_capacity_summary_handles_missing_queue_capacity():
  probe = {
    "sentinel_ebpf_probe_host_cpu_usage_percent": 25.0,
    "sentinel_ebpf_probe_host_memory_usage_percent": 35.0,
    "sentinel_ebpf_probe_host_load1": 0.4,
    "sentinel_ebpf_probe_host_cpu_count": 2,
    "sentinel_ebpf_probe_queue_size": 50000,
    "sentinel_ebpf_probe_events_dropped_total": 0,
  }
  detector = {"sentinel_ebpf_detector_events_total": 2000}
  summary = _compute_capacity_summary(probe, detector)
  assert summary["pipeline"]["queue_capacity"] == 0
  assert summary["pipeline"]["queue_fill_ratio"] is None
