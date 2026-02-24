"""Tests for probe/rules.py in DSL-only mode."""

import pytest

from probe.rules import RuleEngine


class TestRuleEngineDsl:
  def test_allow_matches_any_rule(self, sample_rules_yaml):
    engine = RuleEngine(str(sample_rules_yaml))
    assert engine.allow("openat", "/tmp/test", "bash")

  def test_allow_path_prefix_filter(self, sample_rules_yaml):
    engine = RuleEngine(str(sample_rules_yaml))
    assert engine.allow("openat", "/etc/hosts", "bash")
    assert engine.allow("open", "/bin/ls", "bash")
    assert engine.allow("openat2", "/tmp/test", "python")  # capture-all-file-events

  def test_allow_comm_filter(self, sample_rules_yaml):
    engine = RuleEngine(str(sample_rules_yaml))
    assert engine.allow("execve", "/tmp/test", "bash")
    assert engine.allow("execve", "/tmp/test", "sh")
    assert engine.allow("openat", "/tmp/test", "python")  # capture-all-file-events

  def test_allow_disabled_rule_ignored(self, sample_rules_yaml):
    engine = RuleEngine(str(sample_rules_yaml))
    assert engine.allow("openat", "/tmp/test", "python")  # capture-all-file-events

  def test_reload(self, temp_dir):
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: rule1
    condition: "event_name = openat and path startswith /"
""")
    engine = RuleEngine(str(rules_file))
    assert len(engine.condition_rules) == 1

    rules_file.write_text("""rules:
  - name: rule1
    condition: "event_name = openat and path startswith /"
  - name: rule2
    condition: "event_name = execve and comm = bash"
""")
    engine.reload()
    assert len(engine.condition_rules) == 2

  def test_rejects_path_prefix_excludes(self, temp_dir):
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""pathPrefixExcludes:
  - "/proc"
  - "/sys"
rules:
  - name: capture-all
    condition: "event_name = openat and path startswith /"
""")
    with pytest.raises(ValueError, match="pathPrefixExcludes is not supported"):
      RuleEngine(str(rules_file))
