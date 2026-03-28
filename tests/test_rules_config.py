"""Tests for rules_config.load_rules_config validation."""

import pytest

from rules_config import load_rules_config


def test_rule_group_must_be_declared(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file:
    syscalls: [openat]
rules:
  - name: r1
    group: file
  - name: r2
    group: network
"""
  )
  with pytest.raises(ValueError, match="undefined group"):
    load_rules_config(rules_file)


def test_rule_syscalls_removed(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file:
    syscalls: [openat]
rules:
  - name: r
    group: file
    syscalls: [openat]
"""
  )
  with pytest.raises(ValueError, match="removed 'syscalls'"):
    load_rules_config(rules_file)


def test_group_must_declare_syscalls(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  file:
    features:
      sensitive_paths: [/etc]
rules:
  - name: r
    group: file
"""
  )
  with pytest.raises(ValueError, match="non-empty 'syscalls'"):
    load_rules_config(rules_file)


def test_feature_kind_removed(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  foo:
    feature_kind: file
    syscalls: [openat]
rules:
  - name: r
    group: foo
"""
  )
  with pytest.raises(ValueError, match="removed 'feature_kind'"):
    load_rules_config(rules_file)


def test_syscall_overlap_between_enabled_groups_rejected(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  a:
    syscalls: [openat, connect]
  b:
    syscalls: [connect, bind]
rules:
  - name: r1
    enabled: true
    group: a
  - name: r2
    enabled: true
    group: b
"""
  )
  with pytest.raises(ValueError, match="ambiguous"):
    load_rules_config(rules_file)


def test_declared_groups_roundtrip(temp_dir):
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """groups:
  storage:
    syscalls: [openat, read]
    features:
      sensitive_paths: [/var/lib]
rules:
  - name: r
    group: storage
"""
  )
  cfg = load_rules_config(rules_file)
  assert cfg.groups["storage"].syscalls == ("openat", "read")
  assert "/var/lib" in cfg.groups["storage"].features["sensitive_paths"]


def test_unknown_syscalls_allowed_in_group(temp_dir):
  """Group syscalls are model vocabulary; names need not exist in probe.events yet."""
  rules_file = temp_dir / "rules.yaml"
  rules_file.write_text(
    """lists:
  file_syscalls: [openat]
groups:
  file:
    syscalls: [openat, future_reserved_syscall]
rules:
  - name: r
    group: file
    condition: "syscall_name in (file_syscalls)"
"""
  )
  cfg = load_rules_config(rules_file)
  assert cfg.groups["file"].syscalls == ("openat", "future_reserved_syscall")
