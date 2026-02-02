"""Additional tests for probe/rules.py with filtered rules (no catch-all)."""
import tempfile
from pathlib import Path

import pytest

from probe.rules import RuleEngine


class TestRuleEngineFiltered:
  """Test RuleEngine with rules that don't have catch-all."""

  def test_path_prefix_filtering_without_catchall(self, temp_dir):
    """Test that path prefix filtering works when there's no catch-all rule."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-sensitive-opens
    enabled: true
    event: file_open
    match:
      pathPrefixes: ["/etc", "/bin"]
""")
    engine = RuleEngine(str(rules_file))
    # Should match sensitive paths
    assert engine.allow("file_open", "/etc/hosts", "bash")
    assert engine.allow("file_open", "/bin/ls", "bash")
    # Should NOT match other paths (no catch-all)
    assert not engine.allow("file_open", "/tmp/test", "bash")
    assert not engine.allow("file_open", "/home/user/file", "bash")

  def test_comm_filtering_without_catchall(self, temp_dir):
    """Test that comm filtering works when there's no catch-all rule."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-specific-comm
    enabled: true
    event: file_open
    match:
      comms: ["bash", "sh"]
""")
    engine = RuleEngine(str(rules_file))
    # Should match allowed comms
    assert engine.allow("file_open", "/tmp/test", "bash")
    assert engine.allow("file_open", "/tmp/test", "sh")
    # Should NOT match other comms (no catch-all)
    assert not engine.allow("file_open", "/tmp/test", "python")
    assert not engine.allow("file_open", "/tmp/test", "node")

  def test_combined_filters_without_catchall(self, temp_dir):
    """Test combined path and comm filters without catch-all."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-sensitive-bash-opens
    enabled: true
    event: file_open
    match:
      pathPrefixes: ["/etc"]
      comms: ["bash"]
""")
    engine = RuleEngine(str(rules_file))
    # Should match: both path and comm match
    assert engine.allow("file_open", "/etc/hosts", "bash")
    # Should NOT match: path matches but comm doesn't
    assert not engine.allow("file_open", "/etc/hosts", "python")
    # Should NOT match: comm matches but path doesn't
    assert not engine.allow("file_open", "/tmp/test", "bash")
    # Should NOT match: neither matches
    assert not engine.allow("file_open", "/tmp/test", "python")

  def test_disabled_rule_truly_ignored(self, temp_dir):
    """Test that disabled rules are completely ignored."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-tmp-opens
    enabled: false
    event: file_open
    match:
      pathPrefixes: ["/tmp"]
""")
    engine = RuleEngine(str(rules_file))
    # Should NOT match because rule is disabled
    assert not engine.allow("file_open", "/tmp/test", "bash")
