#!/usr/bin/env bash
# Run multiple commands one after another overnight. Backgrounds with nohup so
# it survives SSH disconnect (same pattern as run_synthetic_overnight.sh).
#
# Usage:
#   ./scripts/run_commands_overnight.sh -f <command-file> [logfile]
#       One command per line; empty lines and # comments ignored.
#   ./scripts/run_commands_overnight.sh [logfile] "command 1" "command 2" ...
#   tail -f overnight.log
#
# Log file defaults to commands_overnight.log. If a command fails, it is
# logged and the next command runs; a summary of failures is printed at the end.
# PID file: <log>.pid has runner PID on line 1, then one PID per command (process group).
# To stop everything: kill $(head -1 <log>.pid); for p in $(tail -n +2 <log>.pid); do kill -TERM -$p 2>/dev/null; done

set -e
cd "$(dirname "$0")/.."

usage() {
  echo "Usage: $0 -f <command-file> [logfile]"
  echo "   or: $0 [logfile] \"command 1\" \"command 2\" ..."
  echo "  Commands from file: one per line; empty lines and # lines ignored."
  exit 1
}

# Parse LOG and COMMANDS (same for inner nohup invocation)
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
elif [[ "$1" == "--nohup-inner" ]]; then
  shift
  LOG="$1"
  shift
  if [[ "$1" == "--file" ]]; then
    shift
    CMDFILE="$1"
    COMMANDS=()
    buf=""
    # Read commands from file, supporting '\' line continuation.
    # Empty lines and lines starting with '#' (when not in a continuation)
    # are ignored.
    while IFS= read -r line || [[ -n "$line" ]]; do
      if [[ -z "$buf" ]]; then
        [[ -z "$line" ]] && continue
        [[ "${line:0:1}" == "#" ]] && continue
      fi
      if [[ "$line" == *\\ ]]; then
        buf+="${line%\\} "
        continue
      else
        buf+="$line"
        cmd="$buf"
        # Trim leading/trailing whitespace
        cmd="${cmd#"${cmd%%[![:space:]]*}"}"
        cmd="${cmd%"${cmd##*[![:space:]]}"}"
        [[ -z "$cmd" ]] && { buf=""; continue; }
        COMMANDS+=("$cmd")
        buf=""
      fi
    done < "$CMDFILE"
  else
    COMMANDS=("$@")
  fi
elif [[ "$1" == "-f" || "$1" == "--file" ]]; then
  shift
  [[ $# -lt 1 ]] && usage
  CMDFILE="$1"
  shift
  LOG="${1:-commands_overnight.log}"
  [[ -n "$1" ]] && shift
  [[ ! -f "$CMDFILE" ]] && { echo "No such file: $CMDFILE"; exit 1; }
  echo "Commands overnight started at $(date -Iseconds). Log: $LOG (from $CMDFILE)"
  nohup "$0" --nohup-inner "$LOG" --file "$CMDFILE" >> "$LOG" 2>&1 &
  _nohup_pid=$!
  for _i in {1..50}; do
    [[ -f "$LOG.pid" ]] && break
    sleep 0.1
  done
  if [[ -f "$LOG.pid" ]]; then
    _pid=$(head -n1 "$LOG.pid")
    echo "Started runner PID=$_pid. Watch: tail -f $LOG"
    echo "  To stop all: kill \$(head -1 $LOG.pid); for p in \$(tail -n +2 $LOG.pid); do kill -TERM -\$p 2>/dev/null; done"
  else
    echo "Started (PID file not found). Watch: tail -f $LOG"
  fi
  exit 0
else
  if [[ $# -eq 0 ]]; then
    usage
  fi
  if [[ $# -eq 1 ]]; then
    LOG="commands_overnight.log"
    COMMANDS=("$1")
  else
    LOG="$1"
    shift
    COMMANDS=("$@")
  fi
  echo "Commands overnight started at $(date -Iseconds). Log: $LOG"
  nohup "$0" --nohup-inner "$LOG" "${COMMANDS[@]}" >> "$LOG" 2>&1 &
  _nohup_pid=$!
  for _i in {1..50}; do
    [[ -f "$LOG.pid" ]] && break
    sleep 0.1
  done
  if [[ -f "$LOG.pid" ]]; then
    _pid=$(head -n1 "$LOG.pid")
    echo "Started runner PID=$_pid. Watch: tail -f $LOG"
    echo "  To stop all: kill \$(head -1 $LOG.pid); for p in \$(tail -n +2 $LOG.pid); do kill -TERM -\$p 2>/dev/null; done"
  else
    echo "Started (PID file not found). Watch: tail -f $LOG"
  fi
  exit 0
fi

# Inner process: write our PID (runner) so outer can report it, then run commands.
# With set -m, each background job gets its own process group (PGID = job PID), so
# killing that PID with kill -TERM -$pid stops the whole job tree (uv, python, etc.).
set -m
echo $$ > "$LOG.pid"
exec >> "$LOG" 2>&1
echo "==== Started at $(date -Iseconds) ===="
FAILED=()

for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$i]}"
  n=$((i + 1))
  echo ""
  echo "==== Command $n/${#COMMANDS[@]} at $(date -Iseconds) ===="
  echo "> $cmd"
  ( eval "$cmd" ) &
  cmd_pid=$!
  echo "Command $n PID=$cmd_pid (process group; use kill -TERM -$cmd_pid to stop this job)"
  echo "$cmd_pid" >> "$LOG.pid"
  if wait $cmd_pid; then
    echo "==== Command $n finished OK at $(date -Iseconds) ===="
  else
    ex=$?
    echo "==== Command $n FAILED (exit $ex) at $(date -Iseconds) ===="
    FAILED+=("$n (exit $ex)")
  fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "==== All ${#COMMANDS[@]} commands finished OK at $(date -Iseconds) ===="
else
  echo "==== Finished at $(date -Iseconds): ${#COMMANDS[@]} commands, ${#FAILED[@]} failed: ${FAILED[*]} ===="
fi
