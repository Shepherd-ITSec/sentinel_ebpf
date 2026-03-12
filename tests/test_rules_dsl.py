"""Tests for Falco-like condition DSL and kernel compile behavior."""

from probe.bpf_build import build_bpf_program, enabled_event_ids_from_rules
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


def test_kernel_compile_includes_read_write_from_file_events(temp_dir):
  """Read and write are in EVENT_NAME_TO_ID; rules with file_events should compile BPF rules for them."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  file_events: [open, openat, read, write]
rules:
  - name: capture-file
    enabled: true
    type: file
    condition: "event_name in (file_events)"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  event_ids = {r["event_id"] for r in compiled}
  assert 0 in event_ids, "read (syscall 0) should be in compiled rules"
  assert 1 in event_ids, "write (syscall 1) should be in compiled rules"


def test_enabled_event_ids_selective_probes(temp_dir):
  """Only event_ids in rules are enabled for probe attachment."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """rules:
  - name: openat-only
    enabled: true
    condition: "event_name = openat"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_event_ids_from_rules(compiled)
  assert 257 in enabled, "openat (257) should be enabled"
  assert 0 not in enabled, "read (0) should not be enabled when not in rules"
  assert 59 not in enabled, "execve (59) should not be enabled"

  # Build program and verify ENABLE_OPENAT=1; read/write probes disabled when not in rules
  prog = build_bpf_program(256, enabled_event_ids=enabled)
  assert "#define ENABLE_OPENAT 1" in prog
  assert "#define ENABLE_READ 0" in prog  # read probe not attached when not in rules


def test_enabled_event_ids_read_only_includes_fd_map_deps(temp_dir):
  """Rules with only read (or write) get open/openat/openat2/close for fd->path cache."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """rules:
  - name: read-only
    enabled: true
    condition: "event_name = read"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_event_ids_from_rules(compiled)
  assert 0 in enabled
  assert 2 in enabled, "open needed for fd_map"
  assert 3 in enabled, "close needed for fd_map"
  assert 257 in enabled, "openat needed for fd_map"
  assert 437 in enabled, "openat2 needed for fd_map"
  assert 57 in enabled, "fork needed for pid_to_parent (inherited fds)"


def test_enabled_event_ids_wildcard_enables_all(temp_dir):
  """Rule with event_id wildcard enables all probes."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """rules:
  - name: any-event
    enabled: true
    condition: "comm = bash"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_event_ids_from_rules(compiled)
  assert 257 in enabled and 59 in enabled, "wildcard should enable all"
  assert 0 in enabled and 1 in enabled, "wildcard enables read/write (enriched via fd->path)"


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


def test_startswithin_with_list(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  noisy_paths: [/proc, /sys, /tmp]
macros:
  noisy_path: "path startswithin (noisy_paths)"
rules:
  - name: capture-non-noisy
    enabled: true
    condition: "event_name = openat and not noisy_path"
"""
  )
  engine = RuleEngine(str(rules_file))
  # /etc/passwd does not start with any noisy prefix -> allowed
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
  # /proc/self/status starts with /proc -> noisy, not allowed
  assert not engine.allow_event(
    {
      "event_name": "openat",
      "path": "/proc/self/status",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "open_flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  # /tmp/foo starts with /tmp -> noisy, not allowed
  assert not engine.allow_event(
    {
      "event_name": "openat",
      "path": "/tmp/foo",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "open_flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
