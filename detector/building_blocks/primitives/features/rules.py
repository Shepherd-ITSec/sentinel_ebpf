from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from rules_config import RulesConfig, load_rules_config

_REPO_RULES_PATH = Path(__file__).resolve().parents[4] / "charts" / "sentinel-ebpf" / "rules.yaml"
_DEFAULT_SENSITIVE_PATH_PREFIXES = ("/etc", "/root", "/bin", "/usr/bin", "/sbin", "/usr/sbin")
_DEFAULT_TMP_PATH_PREFIXES = ("/tmp", "/var/tmp")


@lru_cache(maxsize=1)
def _detector_rules_path() -> Path:
  env_path = (os.environ.get("DETECTOR_RULES_PATH") or "").strip()
  if env_path:
    return Path(env_path)
  mounted_path = Path("/etc/sentinel-ebpf/rules.yaml")
  if mounted_path.exists():
    return mounted_path
  return _REPO_RULES_PATH


@lru_cache(maxsize=1)
def _rules_config() -> RulesConfig | None:
  try:
    return load_rules_config(_detector_rules_path())
  except (OSError, ValueError):
    return None


def _get_group_feature_value(event_group: str, feature_name: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
  cfg = _rules_config()
  fallback_values = tuple(sorted(set(fallback)))
  if cfg is None:
    return fallback_values
  group = cfg.groups.get((event_group or "").strip().lower())
  if group is None:
    return fallback_values
  values = group.features.get((feature_name or "").strip().lower(), ())
  if not values:
    return fallback_values
  return tuple(sorted(set(values)))
