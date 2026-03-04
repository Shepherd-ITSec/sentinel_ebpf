"""Tests for Falco-like condition DSL and kernel compile behavior."""

from probe.rules import RuleEngine


def test_condition_rules_with_lists_and_macros(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  shell_comms: ["bash", "sh"]
macros:
  file_evt: "event_name in (open, openat, openat2)"
  sensitive_open: "path startswith /etc or path startswith /root"
rules:
  - name: dsl-rule
    enabled: true
    condition: "file_evt and sensitive_open and comm in (shell_comms)"
"""
  )
  engine = RuleEngine(str(rules_file))
  assert engine.allow_event(
    {
      "event_name": "openat",
      "path": "/etc/passwd",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "open_flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert not engine.allow_event(
    {
      "event_name": "openat",
      "path": "/tmp/file",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "open_flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )


def test_rule_type_label_is_exposed_via_classify_event(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """rules:
  - name: capture-network-connectivity
    enabled: true
    type: network
    condition: "event_name in (socket, connect)"
"""
  )
  engine = RuleEngine(str(rules_file))
  allowed, rule_type = engine.classify_event(
    {
      "event_name": "connect",
      "event_id": 42,
      "path": "",
      "filename": "",
      "comm": "curl",
      "pid": 123,
      "tid": 123,
      "uid": 1000,
      "open_flags": "",
      "arg0": 3,
      "arg1": 16,
      "arg_flags": "",
      "return_value": 0,
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert allowed
  assert rule_type == "network"


def test_kernel_compile_with_userspace_fallback_predicates(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """rules:
  - name: mixed-rule
    enabled: true
    condition: "event_name = openat and path startswith /etc and open_flags contains O_RDONLY and hostname = node-a"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, stats = engine.compile_kernel_rules()
  assert len(compiled) >= 1
  assert stats.compiled_predicates >= 2  # event_name + path startswith
  assert stats.fallback_predicates >= 2  # open_flags/hostname
