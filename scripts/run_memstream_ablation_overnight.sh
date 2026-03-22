#!/usr/bin/env bash
# Run focused MemStream input-ablation diagnostics overnight on a real event stream.
# Uses run_commands_overnight.sh to chain commands so the run survives disconnects.

set -e

cd "$(dirname "$0")/.."

EVENTS_PATH="${1:-events_17_03_26.jsonl}"
LIMIT="${2:-500000}"
LOG="${3:-memstream_ablation_overnight.log}"
BASE_OUT="test_data/memstream_ablation_overnight"
CMD_FILE="${BASE_OUT}/commands.txt"

mkdir -p "$BASE_OUT"

cat > "$CMD_FILE" <<EOF
env DETECTOR_MEMSTREAM_INPUT_MODE=raw uv run python scripts/memstream_diagnostic.py --limit ${LIMIT} --out-dir ${BASE_OUT}/raw ${EVENTS_PATH}
env DETECTOR_MEMSTREAM_INPUT_MODE=freq1d_u uv run python scripts/memstream_diagnostic.py --limit ${LIMIT} --out-dir ${BASE_OUT}/freq1d_u ${EVENTS_PATH}
env DETECTOR_MEMSTREAM_INPUT_MODE=freq1d_z uv run python scripts/memstream_diagnostic.py --limit ${LIMIT} --out-dir ${BASE_OUT}/freq1d_z ${EVENTS_PATH}
env DETECTOR_MEMSTREAM_INPUT_MODE=freq1d_surprisal uv run python scripts/memstream_diagnostic.py --limit ${LIMIT} --out-dir ${BASE_OUT}/freq1d_surprisal ${EVENTS_PATH}
EOF

echo "Prepared MemStream ablation commands in ${CMD_FILE}"
echo "  events=${EVENTS_PATH}"
echo "  limit=${LIMIT}"
echo "  log=${LOG}"

./scripts/run_commands_overnight.sh -f "${CMD_FILE}" "${LOG}"
