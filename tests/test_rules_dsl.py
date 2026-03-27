"""Tests for explicit-syscall rules and DSL/kernel compile behavior."""

from probe.bpf_build import build_bpf_program, enabled_syscall_nrs_from_rules
from probe.rules import RuleEngine


def test_condition_rules_with_lists_and_macros(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  file_syscalls: [open, openat, openat2]
  shell_comms: [bash, sh]
macros:
  sensitive_open: "path startswith /etc or path startswith /root"
groups:
  file: {}
rules:
  - name: dsl-rule
    enabled: true
    group: file
    syscalls: file_syscalls
    condition: "sensitive_open and comm in (shell_comms)"
"""
  )
  engine = RuleEngine(str(rules_file))
  assert engine.allow_event(
    {
      "syscall_name": "openat",
      "path": "/etc/passwd",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert not engine.allow_event(
    {
      "syscall_name": "openat",
      "path": "/tmp/file",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )


def test_rule_group_label_is_exposed_via_classify_event(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  network: {}
rules:
  - name: capture-network-connectivity
    enabled: true
    group: network
    syscalls: [socket, connect]
"""
  )
  engine = RuleEngine(str(rules_file))
  allowed, group = engine.classify_event(
    {
      "syscall_name": "connect",
      "syscall_nr": 42,
      "path": "",
      "filename": "",
      "comm": "curl",
      "pid": 123,
      "tid": 123,
      "uid": 1000,
      "flags": "",
      "arg0": 3,
      "arg1": 16,
      "arg_flags": "",
      "return_value": 0,
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert allowed
  assert group == "network"


def test_kernel_compile_includes_read_write_from_rule_syscalls(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  file_syscalls: [open, openat, read, write]
groups:
  file: {}
rules:
  - name: capture-file
    enabled: true
    group: file
    syscalls: file_syscalls
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  syscall_nrs = {r["syscall_nr"] for r in compiled}
  assert 0 in syscall_nrs
  assert 1 in syscall_nrs


def test_enabled_syscall_nrs_selective_probes(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file: {}
rules:
  - name: openat-only
    enabled: true
    group: file
    syscalls: [openat]
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_syscall_nrs_from_rules(compiled)
  assert 257 in enabled
  assert 0 not in enabled
  assert 59 not in enabled

  prog = build_bpf_program(256, enabled_syscall_nrs=enabled)
  assert "#define ENABLE_OPENAT 1" in prog
  assert "#define ENABLE_READ 0" in prog


def test_enabled_syscall_nrs_read_only_includes_fd_map_deps(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file: {}
rules:
  - name: read-only
    enabled: true
    group: file
    syscalls: [read]
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_syscall_nrs_from_rules(compiled)
  assert 0 in enabled
  assert 2 in enabled
  assert 3 in enabled
  assert 257 in enabled
  assert 437 in enabled
  assert 57 in enabled


def test_enabled_syscall_nrs_without_syscall_wildcard(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file: {}
rules:
  - name: any-file-open
    enabled: true
    group: file
    syscalls: [openat]
    condition: "comm = bash"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, _ = engine.compile_kernel_rules()
  enabled = enabled_syscall_nrs_from_rules(compiled)
  assert enabled == {257}


def test_kernel_compile_with_userspace_fallback_predicates(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file: {}
rules:
  - name: mixed-rule
    enabled: true
    group: file
    syscalls: [openat]
    condition: "path startswith /etc and flags contains O_RDONLY and hostname = node-a"
"""
  )
  engine = RuleEngine(str(rules_file))
  compiled, stats = engine.compile_kernel_rules()
  assert len(compiled) >= 1
  assert stats.compiled_predicates >= 1
  assert stats.fallback_predicates >= 2


def test_startswithin_with_list(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  noisy_paths: [/proc, /sys, /tmp]
macros:
  noisy_path: "path startswithin (noisy_paths)"
groups:
  file: {}
rules:
  - name: capture-non-noisy
    enabled: true
    group: file
    syscalls: [openat]
    condition: "not noisy_path"
"""
  )
  engine = RuleEngine(str(rules_file))
  assert engine.allow_event(
    {
      "syscall_name": "openat",
      "path": "/etc/passwd",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert not engine.allow_event(
    {
      "syscall_name": "openat",
      "path": "/proc/self/status",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
  assert not engine.allow_event(
    {
      "syscall_name": "openat",
      "path": "/tmp/foo",
      "comm": "bash",
      "pid": 1,
      "tid": 1,
      "uid": 0,
      "flags": "O_RDONLY",
      "hostname": "node-a",
      "namespace": "default",
    }
  )
