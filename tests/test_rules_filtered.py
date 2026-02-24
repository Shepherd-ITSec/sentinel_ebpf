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
    condition: "event_name in (open, openat, openat2) and (path startswith /etc or path startswith /bin)"
""")
    engine = RuleEngine(str(rules_file))
    # Should match sensitive paths
    assert engine.allow("openat", "/etc/hosts", "bash")
    assert engine.allow("open", "/bin/ls", "bash")
    # Should NOT match other paths (no catch-all)
    assert not engine.allow("openat", "/tmp/test", "bash")
    assert not engine.allow("openat", "/home/user/file", "bash")

  def test_comm_filtering_without_catchall(self, temp_dir):
    """Test that comm filtering works when there's no catch-all rule."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-specific-comm
    enabled: true
    condition: "event_name = execve and comm in (bash, sh)"
""")
    engine = RuleEngine(str(rules_file))
    # Should match allowed comms
    assert engine.allow("execve", "/tmp/test", "bash")
    assert engine.allow("execve", "/tmp/test", "sh")
    # Should NOT match other comms (no catch-all)
    assert not engine.allow("execve", "/tmp/test", "python")
    assert not engine.allow("execve", "/tmp/test", "node")

  def test_combined_filters_without_catchall(self, temp_dir):
    """Test combined path and comm filters without catch-all."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-sensitive-bash-opens
    enabled: true
    condition: "event_name = openat and path startswith /etc and comm = bash"
""")
    engine = RuleEngine(str(rules_file))
    # Should match: both path and comm match
    assert engine.allow("openat", "/etc/hosts", "bash")
    # Should NOT match: path matches but comm doesn't
    assert not engine.allow("openat", "/etc/hosts", "python")
    # Should NOT match: comm matches but path doesn't
    assert not engine.allow("openat", "/tmp/test", "bash")
    # Should NOT match: neither matches
    assert not engine.allow("openat", "/tmp/test", "python")

  def test_disabled_rule_truly_ignored(self, temp_dir):
    """Test that disabled rules are completely ignored."""
    rules_file = temp_dir / "rules.yaml"
    rules_file.write_text("""rules:
  - name: capture-tmp-opens
    enabled: false
    condition: "event_name = openat and path startswith /tmp"
""")
    engine = RuleEngine(str(rules_file))
    # Should NOT match because rule is disabled
    assert not engine.allow("openat", "/tmp/test", "bash")
