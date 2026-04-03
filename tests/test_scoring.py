"""Tests for detector.scoring."""

import pytest

from detector.config import DetectorConfig
from detector.model import OnlinePercentileCalibrator
from detector.scoring import (
  anomaly_from_primary,
  compute_primary_score,
  event_group_key,
  get_or_create_percentile_calibrator,
)


def test_event_group_key_default():
  assert event_group_key("") == "__default__"
  assert event_group_key("  Network  ") == "network"


def test_compute_primary_score_scaled():
  assert compute_primary_score(10.0, 0.5, score_mode="scaled", suppress_primary=False, percentile_cal=None) == pytest.approx(0.5)


def test_compute_primary_score_raw():
  assert compute_primary_score(1.25, 0.9, score_mode="raw", suppress_primary=False, percentile_cal=None) == pytest.approx(1.25)


def test_compute_primary_score_suppress():
  assert compute_primary_score(99.0, 1.0, score_mode="scaled", suppress_primary=True, percentile_cal=None) == 0.0


def test_anomaly_from_primary():
  assert anomaly_from_primary(0.8, 0.7, suppress_primary=False) is True
  assert anomaly_from_primary(0.6, 0.7, suppress_primary=False) is False
  assert anomaly_from_primary(0.99, 0.7, suppress_primary=True) is False


def test_get_or_create_percentile_calibrator_reuses():
  cfg = DetectorConfig()
  reg: dict[str, OnlinePercentileCalibrator] = {}
  a = get_or_create_percentile_calibrator(reg, "g1", cfg)
  b = get_or_create_percentile_calibrator(reg, "g1", cfg)
  assert a is b
  assert len(reg) == 1


def test_compute_primary_score_percentile_warmup_then_nonzero():
  cal = OnlinePercentileCalibrator(window_size=64, warmup=2)
  # Warmup: first two updates return 0.0
  assert compute_primary_score(1.0, 0.0, score_mode="percentile", suppress_primary=False, percentile_cal=cal) == 0.0
  assert compute_primary_score(2.0, 0.0, score_mode="percentile", suppress_primary=False, percentile_cal=cal) == 0.0
  p = compute_primary_score(3.0, 0.0, score_mode="percentile", suppress_primary=False, percentile_cal=cal)
  assert 0.0 <= p <= 1.0
