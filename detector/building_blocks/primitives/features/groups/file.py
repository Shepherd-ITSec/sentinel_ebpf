from __future__ import annotations

from typing import Any, Dict

from detector.building_blocks.primitives.features.generic import _hash01
from detector.building_blocks.primitives.features.rules import _DEFAULT_SENSITIVE_PATH_PREFIXES, _DEFAULT_TMP_PATH_PREFIXES, _get_group_feature_value
from detector.building_blocks.primitives.features.views import _FeatureViewSpec


def _extract_file_group_features(evt: Any, view: _FeatureViewSpec, event_group: str) -> Dict[str, float]:
  attrs = dict(evt.attributes or {})
  path = str(attrs.get("fd_path", "") or "").strip()
  g = (event_group or "").strip().lower()
  out: Dict[str, float] = {}
  if view.include_file_sensitive_tmp:
    path_lower = path.lower()
    sensitive_prefixes = _get_group_feature_value(g, "sensitive_paths", _DEFAULT_SENSITIVE_PATH_PREFIXES)
    tmp_prefixes = _get_group_feature_value(g, "tmp_paths", _DEFAULT_TMP_PATH_PREFIXES)
    out["file_sensitive_path"] = 1.0 if any(path_lower.startswith(p) for p in sensitive_prefixes) else 0.0
    out["file_tmp_path"] = 1.0 if any(path_lower.startswith(p) for p in tmp_prefixes) else 0.0
  if view.include_file_flags:
    out["file_flags_hash"] = _hash01(attrs.get("flags") or "")
  return out
