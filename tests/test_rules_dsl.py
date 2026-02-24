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
      "event_type": "openat",
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
      "event_type": "openat",
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
