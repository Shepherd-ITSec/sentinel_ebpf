import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from probe.events import EVENT_NAME_TO_ID, UNKNOWN_EVENT_ID
from rules_config import load_rules_config

# Sentinel for "match any" in kernel rules; must match probe.bpf.c WILDCARD (0xFFFFFFFF).
# Avoids collision with event_id 0 (read), uid 0 (root), pid/tid 0.
WILDCARD = 0xFFFFFFFF

SUPPORTED_FIELDS = {
  "event_name",
  "event_id",
  "path",
  "comm",
  "pid",
  "tid",
  "uid",
  "flags",
  "arg0",
  "arg1",
  "arg_flags",
  "return_value",
  "hostname",
  "namespace",
}

@dataclass
class ConditionRule:
  name: str
  condition: str
  enabled: bool = True
  group: str = ""
  syscalls: tuple[str, ...] = ()
  event_ids: frozenset[int] = frozenset()
  ast: Optional["Expr"] = None


@dataclass
class CompileStats:
  compiled_predicates: int = 0
  fallback_predicates: int = 0
  branches_total: int = 0
  branches_compiled: int = 0
  branches_impossible: int = 0


class Expr:
  pass


@dataclass
class Predicate(Expr):
  field: str
  op: str
  value: Any


@dataclass
class And(Expr):
  left: Expr
  right: Expr


@dataclass
class Or(Expr):
  left: Expr
  right: Expr


@dataclass
class Not(Expr):
  inner: Expr


def _strip_quotes(token: str) -> str:
  if len(token) >= 2 and ((token[0] == '"' and token[-1] == '"') or (token[0] == "'" and token[-1] == "'")):
    return token[1:-1]
  return token


def _tokenize_condition(condition: str) -> List[str]:
  # quoted strings | parentheses | commas | equals | words/symbol chunks
  tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\(|\)|,|=|[^\s(),=]+', condition)
  return tokens


class _ConditionParser:
  def __init__(self, tokens: List[str], lists: Dict[str, tuple[str, ...]]):
    self.tokens = tokens
    self.pos = 0
    self.lists = lists

  def _peek(self) -> Optional[str]:
    if self.pos >= len(self.tokens):
      return None
    return self.tokens[self.pos]

  def _next(self) -> str:
    tok = self._peek()
    if tok is None:
      raise ValueError("Unexpected end of condition")
    self.pos += 1
    return tok

  def _consume(self, expected: str) -> None:
    tok = self._next()
    if tok.lower() != expected.lower():
      raise ValueError(f"Expected '{expected}', got '{tok}'")

  def parse(self) -> Expr:
    expr = self._parse_or()
    if self._peek() is not None:
      raise ValueError(f"Unexpected token '{self._peek()}'")
    return expr

  def _parse_or(self) -> Expr:
    left = self._parse_and()
    while (self._peek() or "").lower() == "or":
      self._next()
      right = self._parse_and()
      left = Or(left=left, right=right)
    return left

  def _parse_and(self) -> Expr:
    left = self._parse_not()
    while (self._peek() or "").lower() == "and":
      self._next()
      right = self._parse_not()
      left = And(left=left, right=right)
    return left

  def _parse_not(self) -> Expr:
    if (self._peek() or "").lower() == "not":
      self._next()
      return Not(inner=self._parse_not())
    return self._parse_primary()

  def _parse_primary(self) -> Expr:
    if self._peek() == "(":
      self._next()
      expr = self._parse_or()
      self._consume(")")
      return expr
    return self._parse_predicate()

  def _parse_list_items(self) -> List[Any]:
    items: List[Any] = []
    while True:
      tok = self._next()
      if tok == ")":
        break
      if tok == ",":
        continue
      list_key = _strip_quotes(tok).strip().lower()
      if list_key in self.lists:
        items.extend(self.lists[list_key])
      else:
        items.append(_strip_quotes(tok))
    return items

  def _parse_predicate(self) -> Predicate:
    field = _strip_quotes(self._next())
    if field not in SUPPORTED_FIELDS:
      raise ValueError(f"Unsupported field '{field}' in condition")
    op_tok = (self._next() or "").lower()

    if op_tok == "=":
      value = _strip_quotes(self._next())
      return Predicate(field=field, op="eq", value=value)

    if op_tok in ("startswith", "contains"):
      value = _strip_quotes(self._next())
      return Predicate(field=field, op=op_tok, value=value)

    if op_tok == "in":
      if self._peek() == "(":
        self._next()
        values = self._parse_list_items()
      else:
        tok = _strip_quotes(self._next())
        list_key = tok.strip().lower()
        values = list(self.lists.get(list_key, (tok,)))
      return Predicate(field=field, op="in", value=values)

    if op_tok == "startswithin":
      self._consume("(")
      values = self._parse_list_items()
      return Predicate(field=field, op="startswithin", value=values)

    raise ValueError(f"Unsupported operator '{op_tok}' in condition")


def _expand_macros(condition: str, macros: Dict[str, str]) -> str:
  cache: Dict[str, str] = {}

  def resolve(name: str, stack: List[str]) -> str:
    if name in cache:
      return cache[name]
    if name in stack:
      raise ValueError(f"Macro cycle detected: {' -> '.join(stack + [name])}")
    expr = macros.get(name, name)
    if name not in macros:
      return expr
    out = expr
    for other in sorted(macros.keys(), key=len, reverse=True):
      pattern = r"\b" + re.escape(other) + r"\b"
      if re.search(pattern, out):
        out = re.sub(pattern, f"({resolve(other, stack + [name])})", out)
    cache[name] = out
    return out

  expanded = condition
  for mname in sorted(macros.keys(), key=len, reverse=True):
    pattern = r"\b" + re.escape(mname) + r"\b"
    if re.search(pattern, expanded):
      expanded = re.sub(pattern, f"({resolve(mname, [])})", expanded)
  return expanded


class RuleEngine:
  def __init__(self, path: str):
    self.path = path
    self.condition_rules: List[ConditionRule] = []
    self.compile_stats = CompileStats()
    self.reload()

  def reload(self):
    cfg = load_rules_config(self.path)
    self.condition_rules = []

    for raw in cfg.rules:
      ast: Optional[Expr] = None
      if raw.condition:
        expanded = _expand_macros(raw.condition, cfg.macros)
        tokens = _tokenize_condition(expanded)
        parser = _ConditionParser(tokens=tokens, lists=cfg.lists)
        ast = parser.parse()
      missing = [name for name in raw.syscalls if name not in EVENT_NAME_TO_ID]
      if missing:
        raise ValueError(f"Rule '{raw.name or '<unnamed>'}' references unknown syscall(s): {', '.join(missing)}")
      self.condition_rules.append(
        ConditionRule(
          name=raw.name,
          condition=raw.condition,
          enabled=raw.enabled,
          group=raw.group,
          syscalls=raw.syscalls,
          event_ids=frozenset(EVENT_NAME_TO_ID[name] for name in raw.syscalls),
          ast=ast,
        )
      )

  @staticmethod
  def _to_int_maybe(value: Any) -> Optional[int]:
    try:
      return int(value)
    except (TypeError, ValueError):
      return None

  def _eval_predicate(self, pred: Predicate, ctx: Dict[str, Any]) -> bool:
    field = pred.field
    op = pred.op
    value = ctx.get(field)
    if field == "path" and value is None:
      value = ctx.get("filename")
    if field == "event_name":
      value = ctx.get("event_name")
    if field == "arg_flags" and value is None:
      value = ctx.get("flags")
    if value is None:
      return False

    if field in ("pid", "tid", "uid", "event_id", "arg0", "arg1", "return_value"):
      value_int = self._to_int_maybe(value)
      if value_int is None:
        return False
      if op == "eq":
        rhs = self._to_int_maybe(pred.value)
        return rhs is not None and value_int == rhs
      if op == "in":
        rhs_vals = [self._to_int_maybe(v) for v in pred.value]
        rhs_norm = {v for v in rhs_vals if v is not None}
        return value_int in rhs_norm
      return False

    value_str = str(value)
    if op == "eq":
      return value_str == str(pred.value)
    if op == "startswith":
      return value_str.startswith(str(pred.value))
    if op == "startswithin":
      return any(value_str.startswith(str(p)) for p in pred.value)
    if op == "contains":
      return str(pred.value) in value_str
    if op == "in":
      return value_str in {str(v) for v in pred.value}
    return False

  def _eval_expr(self, expr: Expr, ctx: Dict[str, Any]) -> bool:
    if isinstance(expr, Predicate):
      return self._eval_predicate(expr, ctx)
    if isinstance(expr, And):
      return self._eval_expr(expr.left, ctx) and self._eval_expr(expr.right, ctx)
    if isinstance(expr, Or):
      return self._eval_expr(expr.left, ctx) or self._eval_expr(expr.right, ctx)
    if isinstance(expr, Not):
      return not self._eval_expr(expr.inner, ctx)
    return False

  def _ctx_event_id(self, ctx: Dict[str, Any]) -> Optional[int]:
    event_id = self._to_int_maybe(ctx.get("event_id"))
    if event_id is not None:
      return event_id
    event_name = str(ctx.get("event_name") or "").strip().lower()
    if not event_name:
      return None
    mapped = EVENT_NAME_TO_ID.get(event_name, UNKNOWN_EVENT_ID)
    return None if mapped == UNKNOWN_EVENT_ID else mapped

  def _matches_rule_syscalls(self, rule: ConditionRule, ctx: Dict[str, Any]) -> bool:
    event_name = str(ctx.get("event_name") or "").strip().lower()
    if event_name and event_name in rule.syscalls:
      return True
    event_id = self._ctx_event_id(ctx)
    return event_id is not None and event_id in rule.event_ids

  def allow(self, event_name: str, filename: str, comm: str) -> bool:
    """event_name = syscall name (e.g. openat, connect). Returns True if any rule matches."""
    return self.allow_event(
      {
        "event_name": event_name,
        "event_id": EVENT_NAME_TO_ID.get(event_name, UNKNOWN_EVENT_ID),
        "path": filename,
        "filename": filename,
        "comm": comm,
        "pid": None,
        "tid": None,
        "uid": None,
        "flags": "",
        "arg0": None,
        "arg1": None,
        "return_value": None,
        "hostname": "",
        "namespace": "",
      }
    )

  def classify_event(self, ctx: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Evaluate rules against an event context.

    Returns:
      (allowed, group) where group is the first matching rule's group.

    This is a superset of allow_event; allow_event keeps the original bool-only
    behavior for compatibility.
    """
    for rule in self.condition_rules:
      if not rule.enabled:
        continue
      if not self._matches_rule_syscalls(rule, ctx):
        continue
      if rule.ast is None or self._eval_expr(rule.ast, ctx):
        return True, (rule.group or None)
    return False, None

  def allow_event(self, ctx: Dict[str, Any]) -> bool:
    allowed, _ = self.classify_event(ctx)
    return allowed

  def _expr_to_dnf(self, expr: Expr) -> List[List[Expr]]:
    if isinstance(expr, Or):
      return self._expr_to_dnf(expr.left) + self._expr_to_dnf(expr.right)
    if isinstance(expr, And):
      left = self._expr_to_dnf(expr.left)
      right = self._expr_to_dnf(expr.right)
      out: List[List[Expr]] = []
      for l in left:
        for r in right:
          out.append(l + r)
      return out
    return [[expr]]

  def compile_kernel_rules(self) -> Tuple[List[Dict[str, Any]], CompileStats]:
    stats = CompileStats()
    compiled: List[Dict[str, Any]] = []

    def _event_ids_from_values(raw_vals: List[Any]) -> set[int]:
      ids: set[int] = set()
      for raw in raw_vals:
        sval = str(raw)
        if sval in EVENT_NAME_TO_ID:
          ids.add(EVENT_NAME_TO_ID[sval])
          continue
        nval = self._to_int_maybe(raw)
        if nval is not None and nval >= 0:
          ids.add(nval)
      return ids

    # Explicit syscall membership is always part of the kernel filter; condition adds extra branch-wise predicates.
    for rule in self.condition_rules:
      if not rule.enabled:
        continue
      branches = self._expr_to_dnf(rule.ast) if rule.ast is not None else [[]]
      stats.branches_total += len(branches)

      for branch in branches:
        event_ids: Optional[set[int]] = set(rule.event_ids)
        prefixes: Optional[set[str]] = None
        comms: Optional[set[str]] = None
        pids: Optional[set[int]] = None
        tids: Optional[set[int]] = None
        uids: Optional[set[int]] = None
        possible = True
        compiled_here = 0

        for lit in branch:
          if isinstance(lit, Not):
            stats.fallback_predicates += 1
            continue
          if not isinstance(lit, Predicate):
            stats.fallback_predicates += 1
            continue

          field = lit.field
          op = lit.op
          val = lit.value
          if field == "event_name":
            if op == "eq":
              es = _event_ids_from_values([val])
            elif op == "in":
              es = _event_ids_from_values(list(val))
            else:
              stats.fallback_predicates += 1
              continue
            if len(es) == 0:
              possible = False
              break
            event_ids = es if event_ids is None else (event_ids & es)
            if event_ids is not None and len(event_ids) == 0:
              possible = False
              break
            compiled_here += 1
          elif field == "event_id" and op in ("eq", "in"):
            try:
              es = {int(val)} if op == "eq" else {int(v) for v in val}
            except (TypeError, ValueError):
              stats.fallback_predicates += 1
              continue
            es = {ev for ev in es if ev >= 0}
            if len(es) == 0:
              possible = False
              break
            event_ids = es if event_ids is None else (event_ids & es)
            if event_ids is not None and len(event_ids) == 0:
              possible = False
              break
            compiled_here += 1
          elif field == "path" and op in ("startswith", "startswithin"):
            pset = {str(val)} if op == "startswith" else {str(v) for v in val}
            prefixes = pset if prefixes is None else {p for p in prefixes if any(p.startswith(x) or x.startswith(p) for x in pset)}
            if prefixes is not None and len(prefixes) == 0:
              possible = False
              break
            compiled_here += 1
          elif field == "comm" and op in ("eq", "in"):
            cset = {str(val)} if op == "eq" else {str(v) for v in val}
            comms = cset if comms is None else (comms & cset)
            if comms is not None and len(comms) == 0:
              possible = False
              break
            compiled_here += 1
          elif field in ("pid", "tid", "uid") and op in ("eq", "in"):
            try:
              nset = {int(val)} if op == "eq" else {int(v) for v in val}
            except (TypeError, ValueError):
              stats.fallback_predicates += 1
              continue
            target = None
            if field == "pid":
              target = pids
            elif field == "tid":
              target = tids
            else:
              target = uids
            target = nset if target is None else (target & nset)
            if target is not None and len(target) == 0:
              possible = False
              break
            if field == "pid":
              pids = target
            elif field == "tid":
              tids = target
            else:
              uids = target
            compiled_here += 1
          else:
            stats.fallback_predicates += 1

        if not possible:
          stats.branches_impossible += 1
          continue

        event_vals = list(event_ids) if event_ids else [WILDCARD]
        prefix_vals = list(prefixes) if prefixes else [""]
        comm_vals = list(comms) if comms else [""]
        pid_vals = list(pids) if pids else [WILDCARD]
        tid_vals = list(tids) if tids else [WILDCARD]
        uid_vals = list(uids) if uids else [WILDCARD]
        for event_id in event_vals:
          for prefix in prefix_vals:
            for comm in comm_vals:
              for pid in pid_vals:
                for tid in tid_vals:
                  for uid in uid_vals:
                    compiled.append(
                      {
                        "event_id": int(event_id),
                        "prefix": prefix,
                        "comm": comm,
                        "pid": int(pid),
                        "tid": int(tid),
                        "uid": int(uid),
                      }
                    )
        stats.compiled_predicates += compiled_here
        stats.branches_compiled += 1

    self.compile_stats = stats
    return compiled, stats
