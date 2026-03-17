"""Focused tests for PySAD KitNet wrapper behavior."""

import logging

import numpy as np

import detector.model as model_module
from detector.model import OnlineKitNet


class _FakeKitNet:
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.calls = []
    self.score_value = 2.0

  def score_partial(self, x):
    self.calls.append(("score_partial", x.copy()))
    return self.score_value

  def fit_partial(self, x, y=None):
    self.calls.append(("fit_partial", x.copy()))
    return self


class _ColdStartKitNet(_FakeKitNet):
  def score_partial(self, x):
    self.calls.append(("score_partial", x.copy()))
    raise AttributeError("'KitNet' object has no attribute 'model'")


def _make_features():
  return {"b": 2.0, "a": 1.0}


def test_kitnet_scores_before_learning(monkeypatch):
  monkeypatch.setattr(model_module, "Pysad_KitNet", _FakeKitNet)
  detector = OnlineKitNet(
    max_size_ae=10,
    grace_feature_mapping=5,
    grace_anomaly_detector=10,
    learning_rate=0.1,
    hidden_ratio=0.75,
    model_device="cpu",
    seed=0,
  )

  _, score = detector.score_and_learn(_make_features())
  assert np.isclose(score, 1.0 - np.exp(-2.0))

  assert detector._model is not None
  assert [name for name, _ in detector._model.calls] == ["score_partial", "fit_partial"]
  # For 2-D input, sub-autoencoder size is clamped to input dimensionality.
  assert detector._model.kwargs["max_size_ae"] == 2
  np.testing.assert_allclose(detector._model.calls[0][1], np.array([1.0, 2.0], dtype=np.float64))
  np.testing.assert_allclose(detector._model.calls[1][1], np.array([1.0, 2.0], dtype=np.float64))


def test_kitnet_non_finite_raw_score_falls_back_to_zero(monkeypatch, caplog):
  monkeypatch.setattr(model_module, "Pysad_KitNet", _FakeKitNet)
  detector = OnlineKitNet(
    max_size_ae=10,
    grace_feature_mapping=5,
    grace_anomaly_detector=10,
    learning_rate=0.1,
    hidden_ratio=0.75,
    model_device="cpu",
    seed=0,
  )
  assert detector._model is None
  detector._init_from_features(_make_features())
  assert detector._model is not None
  detector._model.score_value = float("nan")

  with caplog.at_level(logging.WARNING):
    _, score = detector.score_and_learn(_make_features())

  assert score == 0.0
  assert "KitNet produced non-finite score" in caplog.text


def test_kitnet_cold_start_missing_internal_model_attribute(monkeypatch):
  monkeypatch.setattr(model_module, "Pysad_KitNet", _ColdStartKitNet)
  detector = OnlineKitNet(
    max_size_ae=10,
    grace_feature_mapping=5,
    grace_anomaly_detector=10,
    learning_rate=0.1,
    hidden_ratio=0.75,
    model_device="cpu",
    seed=0,
  )

  _, score = detector.score_and_learn(_make_features())
  assert score == 0.0
  assert detector._model is not None
  assert [name for name, _ in detector._model.calls] == ["score_partial", "fit_partial"]


def test_kitnet_score_only_does_not_call_fit_partial(monkeypatch):
  monkeypatch.setattr(model_module, "Pysad_KitNet", _FakeKitNet)
  detector = OnlineKitNet(
    max_size_ae=10,
    grace_feature_mapping=5,
    grace_anomaly_detector=10,
    learning_rate=0.1,
    hidden_ratio=0.75,
    model_device="cpu",
    seed=0,
  )
  detector.score_and_learn(_make_features())
  detector._model.calls.clear()

  detector.score_only(_make_features())

  assert [name for name, _ in detector._model.calls] == ["score_partial"]
