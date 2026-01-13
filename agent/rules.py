import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class RuleMatch:
  path_prefixes: List[str] = field(default_factory=list)
  comms: List[str] = field(default_factory=list)


@dataclass
class Rule:
  name: str
  event: str
  enabled: bool = True
  match: RuleMatch = field(default_factory=RuleMatch)

  def matches(self, event_type: str, filename: str, comm: str) -> bool:
    if not self.enabled:
      return False
    if self.event != event_type:
      return False
    if self.match.path_prefixes:
      if not any(filename.startswith(p) for p in self.match.path_prefixes):
        return False
    if self.match.comms:
      if comm not in self.match.comms:
        return False
    return True


def _load_rules_yaml(path: str) -> List[Rule]:
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  rules = []
  for raw in data.get("rules", []):
    rules.append(
      Rule(
        name=str(raw.get("name", "")),
        event=str(raw.get("event", "")),
        enabled=bool(raw.get("enabled", True)),
        match=RuleMatch(
          path_prefixes=list(raw.get("match", {}).get("pathPrefixes", [])),
          comms=list(raw.get("match", {}).get("comms", [])),
        ),
      )
    )
  return rules


class RuleEngine:
  def __init__(self, path: str):
    self.path = path
    self.rules = _load_rules_yaml(path)

  def reload(self):
    self.rules = _load_rules_yaml(self.path)

  def allow(self, event_type: str, filename: str, comm: str) -> bool:
    return any(rule.matches(event_type, filename, comm) for rule in self.rules)
