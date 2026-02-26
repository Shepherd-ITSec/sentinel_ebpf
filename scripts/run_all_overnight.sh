#!/usr/bin/env bash
# Run full BETH eval matrix overnight. Survives SSH disconnect.
#
# Background (recommended): start and log out; job keeps running.
#   ./scripts/run_all_overnight.sh
#   tail -f run_all.log   # watch while connected
#   # disconnect SSH; reattach later and check run_all.log or test_data/beth/run_all_*/
#
# Foreground: run in tmux/screen so you can detach.
#   tmux new -s beth
#   cd /path/to/sentinel_ebpf && uv run python scripts/run_beth_train_test_eval.py --run-all --pace fast

set -e
cd "$(dirname "$0")/.."
LOG="${1:-run_all.log}"
echo "Run-all started at $(date -Iseconds). Log: $LOG"
nohup uv run python scripts/run_beth_train_test_eval.py --run-all --pace fast >> "$LOG" 2>&1 &
echo "Started PID=$!. Watch: tail -f $LOG. Results under test_data/beth/run_all_*/"
