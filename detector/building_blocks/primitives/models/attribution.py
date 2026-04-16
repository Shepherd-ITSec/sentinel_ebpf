from __future__ import annotations

from typing import Any, Dict, Tuple


def compute_feature_attribution(
  detector: Any,
  features: Dict[str, float],
  epsilon: float = 0.01,
  binary_threshold: float = 0.01,
  *,
  score_fn=None,
) -> Tuple[float, Dict[str, float]]:
  """
  Model-agnostic perturbation-based feature attribution.
  Returns (score, {feature_name: attribution}).
  Positive attribution = feature pushes score up (more anomalous).

  Binary features (value in [0, binary_threshold] or [1-binary_threshold, 1]) use flip:
  attribution = (score(feature=1) - score(feature=0)) * value.
  Avoids invalid interpolation and threshold effects from ±epsilon.

  Continuous features use finite difference:
  attribution = (s_plus - s_minus) / (2*epsilon) * value.
  """
  if score_fn is None:
    score_fn = lambda f: detector.score_only(f)[1]
  score = float(score_fn(features))
  names = sorted(features.keys())
  attribution: Dict[str, float] = {}
  for name in names:
    val = float(features[name])
    is_binary = val <= binary_threshold or val >= (1.0 - binary_threshold)
    if is_binary:
      features_0 = dict(features)
      features_0[name] = 0.0
      features_1 = dict(features)
      features_1[name] = 1.0
      s_0 = float(score_fn(features_0))
      s_1 = float(score_fn(features_1))
      attribution[name] = float((s_1 - s_0) * val)
    else:
      val_plus = max(0.0, min(1.0, val + epsilon))
      val_minus = max(0.0, min(1.0, val - epsilon))
      features_plus = dict(features)
      features_plus[name] = val_plus
      features_minus = dict(features)
      features_minus[name] = val_minus
      s_plus = float(score_fn(features_plus))
      s_minus = float(score_fn(features_minus))
      grad_approx = (s_plus - s_minus) / (2.0 * epsilon) if epsilon > 0 else 0.0
      attribution[name] = float(grad_approx * val)
  return (score, attribution)
