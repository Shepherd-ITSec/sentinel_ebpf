#!/usr/bin/env bash
# Run full synthetic EVT1 eval matrix overnight. Survives SSH disconnect.
# Uses run_detector_eval.py in single-stream mode.
#
#   ./scripts/run_synthetic_overnight.sh --generate test_data/synthetic/run4 [total_events]
#       Generate new EVT1 + labels, then run full matrix on them. Optional total_events (default 100000).
#   ./scripts/run_synthetic_overnight.sh
#       Use default evt1/labels (test_data/synthetic/run3.evt1)
#   ./scripts/run_synthetic_overnight.sh my.evt1
#       evt1 + derived labels path (my.labels.ndjson)
#   ./scripts/run_synthetic_overnight.sh my.evt1 my.labels.ndjson [log]
#
# Results under test_data/synthetic/run_all_*/

set -e

cd "$(dirname "$0")/.."

SYNTH_DIR="test_data/synthetic"
DEFAULT_EVT1="${SYNTH_DIR}/run3.evt1"
DEFAULT_LABELS="${SYNTH_DIR}/run3.labels.ndjson"

if [[ $# -ge 1 && "$1" == "--generate" ]]; then
  OUT_PREFIX="${2:?Usage: run_synthetic_overnight.sh --generate <out-prefix> [total-events]}"
  TOTAL_EVENTS="${3:-100000}"
  echo "Generating synthetic data: prefix=$OUT_PREFIX total_events=$TOTAL_EVENTS"
  uv run python scripts/generate_synthetic_evt1_dataset.py \
    --out-prefix "$OUT_PREFIX" \
    --total-events "$TOTAL_EVENTS" \
    --positive-fraction 0.01 \
    --warmup-fraction 0.75
  EVT1_PATH="${OUT_PREFIX}.evt1"
  LABELS_PATH="${OUT_PREFIX}.labels.ndjson"
  LOG="$(basename "$OUT_PREFIX").log"
elif [[ $# -ge 2 ]]; then
  EVT1_PATH="$1"
  LABELS_PATH="$2"
  LOG="${3:-synthetic_run_all.log}"
elif [[ $# -eq 1 ]]; then
  EVT1_PATH="$1"
  base="${1%.evt1}"
  LABELS_PATH="${base}.labels.ndjson"
  LOG="synthetic_run_all.log"
else
  EVT1_PATH="${DEFAULT_EVT1}"
  LABELS_PATH="${DEFAULT_LABELS}"
  LOG="synthetic_run_all.log"
fi

echo "Synthetic run-all started at $(date -Iseconds). Log: $LOG"
echo "  evt1=$EVT1_PATH labels=$LABELS_PATH"
nohup uv run python scripts/run_detector_eval.py \
  --evt1 "${EVT1_PATH}" \
  --labels "${LABELS_PATH}" \
  --out-dir test_data/synthetic/eval \
  --run-all \
  --pace fast >> "$LOG" 2>&1 &
echo "Started PID=$!. Watch: tail -f $LOG. Results under test_data/synthetic/run_all_*/"
