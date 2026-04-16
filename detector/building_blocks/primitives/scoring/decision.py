from __future__ import annotations


def anomaly_from_primary(
  score_primary: float,
  threshold: float,
  *,
  suppress_primary: bool,
) -> bool:
  return (not suppress_primary) and float(score_primary) >= float(threshold)

