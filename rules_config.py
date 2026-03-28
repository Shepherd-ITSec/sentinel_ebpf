from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuleGroupConfig:
  name: str
  features: dict[str, tuple[str, ...]]
  syscalls: tuple[str, ...]


@dataclass(frozen=True)
class RuleConfig:
  name: str
  enabled: bool
  group: str
  condition: str


@dataclass(frozen=True)
class RulesConfig:
  lists: dict[str, tuple[str, ...]]
  macros: dict[str, str]
  groups: dict[str, RuleGroupConfig]
  rules: tuple[RuleConfig, ...]


def _normalize_name(value: Any) -> str:
  return str(value or "").strip().lower()


def _normalize_string_list(raw: Any, *, field_name: str) -> tuple[str, ...]:
  if not isinstance(raw, list):
    raise ValueError(f"{field_name} must be a YAML list of strings")
  values: list[str] = []
  for item in raw:
    name = _normalize_name(item)
    if not name:
      raise ValueError(f"{field_name} must not contain empty values")
    values.append(name)
  if not values:
    raise ValueError(f"{field_name} must not be empty")
  return tuple(values)


def _normalize_lists(raw: Any) -> dict[str, tuple[str, ...]]:
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    raise ValueError("top-level 'lists' must be a mapping")
  lists: dict[str, tuple[str, ...]] = {}
  for name, values in raw.items():
    list_name = _normalize_name(name)
    if not list_name:
      raise ValueError("list name must not be empty")
    lists[list_name] = _normalize_string_list(values, field_name=f"lists.{list_name}")
  return lists


def _normalize_macros(raw: Any) -> dict[str, str]:
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    raise ValueError("top-level 'macros' must be a mapping")
  macros: dict[str, str] = {}
  for name, expr in raw.items():
    macro_name = _normalize_name(name)
    if not macro_name:
      raise ValueError("macro name must not be empty")
    macros[macro_name] = str(expr or "").strip()
  return macros


def _resolve_syscalls(raw: Any, *, lists: dict[str, tuple[str, ...]], field_name: str) -> tuple[str, ...]:
  if isinstance(raw, str):
    key = _normalize_name(raw)
    if key in lists:
      return lists[key]
    raise ValueError(f"{field_name} references unknown syscall list '{key}'")
  if not isinstance(raw, list):
    raise ValueError(f"{field_name} must be a YAML list of syscall names or a list reference")
  values: list[str] = []
  for item in raw:
    if isinstance(item, str):
      key = _normalize_name(item)
      if key in lists:
        values.extend(lists[key])
      else:
        if not key:
          raise ValueError(f"{field_name} must not contain empty values")
        values.append(key)
    else:
      raise ValueError(f"{field_name} must contain only strings")
  if not values:
    raise ValueError(f"{field_name} must not be empty")
  return tuple(values)


def _normalize_features(raw: Any, *, field_name: str) -> dict[str, tuple[str, ...]]:
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    raise ValueError(f"{field_name} must be a mapping")
  features: dict[str, tuple[str, ...]] = {}
  for name, values in raw.items():
    feature_name = _normalize_name(name)
    if not feature_name:
      raise ValueError(f"{field_name} contains an empty feature name")
    features[feature_name] = _normalize_string_list(values, field_name=f"{field_name}.{feature_name}")
  return features


def _normalize_groups(raw: Any, *, lists: dict[str, tuple[str, ...]]) -> dict[str, RuleGroupConfig]:
  if raw is None:
    return {}
  if not isinstance(raw, dict):
    raise ValueError("top-level 'groups' must be a mapping")
  groups: dict[str, RuleGroupConfig] = {}
  for name, body in raw.items():
    group_name = _normalize_name(name)
    if not group_name:
      raise ValueError("group name must not be empty")
    if not isinstance(body, dict):
      raise ValueError(f"group '{group_name}' must be a mapping")
    if "feature_kind" in body:
      raise ValueError(
        f"group '{group_name}' uses removed 'feature_kind'; detector routing uses syscall shape internally"
      )
    features = _normalize_features(body.get("features", {}), field_name=f"groups.{group_name}.features")
    try:
      syscalls = _resolve_syscalls(
        body.get("syscalls"), lists=lists, field_name=f"groups.{group_name}.syscalls"
      )
    except ValueError as e:
      if "must not be empty" in str(e) or "must be a YAML list" in str(e):
        raise ValueError(f"group '{group_name}' must declare non-empty 'syscalls'") from e
      raise
    # Names need not exist in probe/events.py yet: reserved for model vocabulary / forward names; BPF uses known IDs only.
    groups[group_name] = RuleGroupConfig(name=group_name, features=features, syscalls=syscalls)
  return groups


def _normalize_rules(
  raw: Any,
  *,
  groups: dict[str, RuleGroupConfig],
) -> tuple[RuleConfig, ...]:
  if not isinstance(raw, list):
    raise ValueError("rules file must contain a top-level 'rules' list")
  rules: list[RuleConfig] = []
  for idx, item in enumerate(raw):
    if not isinstance(item, dict):
      raise ValueError(f"rules[{idx}] must be a mapping")
    name = str(item.get("name") or "").strip()
    enabled = bool(item.get("enabled", True))
    if "type" in item:
      raise ValueError(f"Rule '{name or idx}' uses deprecated 'type'; use 'group'")
    if "event_type" in item:
      raise ValueError(f"Rule '{name or idx}' uses deprecated 'event_type'; use 'group'")
    if "event_group" in item:
      raise ValueError(f"Rule '{name or idx}' uses deprecated 'event_group'; use 'group'")
    if "syscalls" in item:
      raise ValueError(
        f"Rule '{name or idx}' uses removed 'syscalls'; declare syscalls on the group under 'groups.<name>.syscalls'"
      )
    group = _normalize_name(item.get("group"))
    if not group:
      raise ValueError(f"Rule '{name or idx}' is missing required 'group'")
    if group not in groups:
      raise ValueError(
        f"Rule '{name or idx}' references undefined group {group!r}; declare it under top-level 'groups'"
      )
    condition = str(item.get("condition") or "").strip()
    rules.append(RuleConfig(name=name, enabled=enabled, group=group, condition=condition))
  return tuple(rules)


def _validate_enabled_group_syscall_overlap(groups: dict[str, RuleGroupConfig], rules: tuple[RuleConfig, ...]) -> None:
  enabled_groups = {r.group for r in rules if r.enabled}
  syscall_to_groups: dict[str, set[str]] = {}
  for gname in enabled_groups:
    gcfg = groups[gname]
    for sc in gcfg.syscalls:
      syscall_to_groups.setdefault(sc, set()).add(gname)
  conflicts = sorted((sc, sorted(gs)) for sc, gs in syscall_to_groups.items() if len(gs) > 1)
  if not conflicts:
    return
  detail = "; ".join(f"{sc!r} -> {gs}" for sc, gs in conflicts[:12])
  more = f" ({len(conflicts) - 12} more)" if len(conflicts) > 12 else ""
  raise ValueError(
    "The same syscall appears in groups that both have enabled rules (ambiguous event_group routing): "
    f"{detail}{more}"
  )


def load_rules_config(path: str | Path) -> RulesConfig:
  path = Path(path)
  with path.open("r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
  if not isinstance(raw, dict):
    raise ValueError("rules file must be a mapping with keys including rules")
  if "pathPrefixExcludes" in raw:
    raise ValueError("pathPrefixExcludes is not supported; express exclusions in rule conditions")
  lists = _normalize_lists(raw.get("lists"))
  macros = _normalize_macros(raw.get("macros"))
  groups = _normalize_groups(raw.get("groups"), lists=lists)
  rules = _normalize_rules(raw.get("rules"), groups=groups)
  _validate_enabled_group_syscall_overlap(groups, rules)
  return RulesConfig(lists=lists, macros=macros, groups=groups, rules=rules)
