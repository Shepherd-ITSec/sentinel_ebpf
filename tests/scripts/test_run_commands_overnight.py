"""Tests for scripts/run_commands_overnight.sh."""

import json
import subprocess
import sys
import time
from pathlib import Path


def _python_write_command(target: Path, text: str, sleep_s: float = 0.0) -> str:
  payload = json.dumps(text)
  target_path = json.dumps(str(target))
  return (
    f"{sys.executable} -c 'import pathlib,time; "
    f"time.sleep({sleep_s}); "
    f"pathlib.Path({target_path}).write_text({payload}, encoding=\"utf-8\")'"
  )


def test_file_backed_runner_allows_editing_pending_commands(temp_dir):
  repo_root = Path(__file__).resolve().parent.parent.parent
  script = repo_root / "scripts" / "run_commands_overnight.sh"
  log_path = temp_dir / "overnight.log"
  cmd_file = temp_dir / "commands.txt"
  first_out = temp_dir / "first.txt"
  old_second_out = temp_dir / "old_second.txt"
  new_second_out = temp_dir / "new_second.txt"

  cmd_file.write_text(
    "\n".join([
      _python_write_command(first_out, "first", sleep_s=2.0),
      _python_write_command(old_second_out, "old-second"),
      "",
    ]),
    encoding="utf-8",
  )

  proc = subprocess.Popen(
    ["bash", str(script), "--nohup-inner", str(log_path), "--file", str(cmd_file)],
    cwd=repo_root,
  )
  try:
    pid_file = Path(f"{log_path}.pid")
    for _ in range(50):
      if pid_file.exists():
        break
      time.sleep(0.1)
    assert pid_file.exists(), "runner PID file was not created"

    cmd_file.write_text(
      "\n".join([
        _python_write_command(first_out, "first", sleep_s=2.0),
        _python_write_command(new_second_out, "new-second"),
        "",
      ]),
      encoding="utf-8",
    )

    proc.wait(timeout=15)
    assert proc.returncode == 0
  finally:
    if proc.poll() is None:
      proc.terminate()
      proc.wait(timeout=5)

  assert first_out.read_text(encoding="utf-8") == "first"
  assert not old_second_out.exists()
  assert new_second_out.read_text(encoding="utf-8") == "new-second"

  log_text = log_path.read_text(encoding="utf-8")
  assert "Refreshed" in log_text
