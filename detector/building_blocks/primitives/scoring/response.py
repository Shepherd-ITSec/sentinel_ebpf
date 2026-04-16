from __future__ import annotations

from detector.building_blocks.core.base import DecisionOutput


def response_score_from_decision(decision: DecisionOutput) -> float:
  """Return the score exposed to clients/UI from a final decision payload."""
  if str(decision.score_mode).strip().lower() == "percentile":
    return float(decision.primary)
  return 0.0 if decision.suppressed else float(decision.scaled)
