"""Tests that PySAD LODA (algorithm=loda) is rejected with a clear error."""

import pytest

from detector.model import OnlineAnomalyDetector


def test_loda_raises_runtime_error():
  with pytest.raises(RuntimeError) as exc_info:
    OnlineAnomalyDetector(algorithm="loda", loda_n_projections=16, loda_bins=32, seed=3)
  assert "loda" in str(exc_info.value).lower()
  assert "loda_ema" in str(exc_info.value).lower()
