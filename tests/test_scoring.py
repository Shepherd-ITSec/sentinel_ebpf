"""Tests for scoring primitives."""

import pytest

from detector.building_blocks.core.base import DecisionOutput, ScoreOutput
from detector.building_blocks.primitives.scoring.calibration import OnlinePercentileCalibrator
from detector.building_blocks.primitives.scoring.decision import anomaly_from_primary
from detector.building_blocks.primitives.scoring.primary_score import compute_primary_score, event_group_key
from detector.building_blocks.primitives.scoring.response import response_score_from_decision


def test_event_group_key_default():
  assert event_group_key("") == "__default__"
  assert event_group_key("  Network  ") == "network"


def test_compute_primary_score_scaled():
  score = ScoreOutput(raw=10.0, scaled=0.5)
  assert compute_primary_score(score, score_mode="scaled", suppress_primary=False, percentile_cal=None) == pytest.approx(0.5)


def test_compute_primary_score_raw():
  score = ScoreOutput(raw=1.25, scaled=0.9)
  assert compute_primary_score(score, score_mode="raw", suppress_primary=False, percentile_cal=None) == pytest.approx(1.25)


def test_compute_primary_score_suppress():
  score = ScoreOutput(raw=99.0, scaled=1.0)
  assert compute_primary_score(score, score_mode="scaled", suppress_primary=True, percentile_cal=None) == 0.0


def test_anomaly_from_primary():
  assert anomaly_from_primary(0.8, 0.7, suppress_primary=False) is True
  assert anomaly_from_primary(0.6, 0.7, suppress_primary=False) is False
  assert anomaly_from_primary(0.99, 0.7, suppress_primary=True) is False


def test_compute_primary_score_percentile_warmup_then_nonzero():
  cal = OnlinePercentileCalibrator(window_size=64, warmup=2)
  # Warmup: first two updates return 0.0
  assert compute_primary_score(ScoreOutput(raw=1.0, scaled=0.0), score_mode="percentile", suppress_primary=False, percentile_cal=cal) == 0.0
  assert compute_primary_score(ScoreOutput(raw=2.0, scaled=0.0), score_mode="percentile", suppress_primary=False, percentile_cal=cal) == 0.0
  p = compute_primary_score(ScoreOutput(raw=3.0, scaled=0.0), score_mode="percentile", suppress_primary=False, percentile_cal=cal)
  assert 0.0 <= p <= 1.0


def test_response_score_from_decision_scaled_mode():
  out = DecisionOutput(raw=2.0, scaled=0.6, primary=0.6, score_mode="scaled", suppressed=False, threshold=0.7, anomaly=False)
  assert response_score_from_decision(out) == pytest.approx(0.6)


def test_response_score_from_decision_percentile_mode():
  out = DecisionOutput(raw=2.0, scaled=0.6, primary=0.95, score_mode="percentile", suppressed=False, threshold=0.7, anomaly=True)
  assert response_score_from_decision(out) == pytest.approx(0.95)


def test_response_score_from_decision_suppressed_returns_zero():
  out = DecisionOutput(raw=2.0, scaled=0.6, primary=0.0, score_mode="scaled", suppressed=True, threshold=0.7, anomaly=False)
  assert response_score_from_decision(out) == 0.0
