#!/usr/bin/env bash
set -euo pipefail

# Run the activity generator in a Kubernetes pod and verify which generated
# events appear in the detector's anomaly log (or event dump).
# Usage: ./scripts/run-activity-generator.sh [namespace]
#   namespace: Kubernetes namespace (default: default)
#
# Requires: detector running with ANOMALY_LOG_PATH and/or EVENT_DUMP_PATH set.
# Verification uses path matching (data[8] in canonical event vector).
#
# EVENTS_TAIL: only fetch last N lines from events.jsonl (default 50000). Avoids copying huge files.

NAMESPACE="${1:-default}"
EVENTS_TAIL="${EVENTS_TAIL:-50000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAIT_AFTER_GEN="${WAIT_AFTER_GEN:-5}"  # Seconds to wait for events to reach detector

echo "Running activity generator in namespace: ${NAMESPACE}"
echo ""

# Record start time for event filtering (only events after this count as "ours")
GEN_START=$(date +%s.%N)

SENSITIVE_OPS="${SENSITIVE_OPS:-3}"
NORMAL_OPS="${NORMAL_OPS:-10}"

# Read the script and pass it to busybox pod (env must match verification)
kubectl run activity-generator \
  --rm -i --restart=Never \
  --image=busybox:1.36 \
  --namespace="${NAMESPACE}" \
  --env="SENSITIVE_OPS=${SENSITIVE_OPS}" \
  --env="NORMAL_OPS=${NORMAL_OPS}" \
  -- sh -c "$(cat "${ROOT_DIR}/scripts/generate-activity.sh")"

echo ""
echo "Waiting ${WAIT_AFTER_GEN}s for events to propagate..."
sleep "${WAIT_AFTER_GEN}"

# Fetch anomaly log and/or event dump from detector pod
DETECTOR_POD=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/component=detector -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
VERIFY_DIR=$(mktemp -d)
trap 'rm -rf "${VERIFY_DIR}"' EXIT

ANOMALY_LOG=""
EVENT_DUMP=""

if [ -n "${DETECTOR_POD}" ]; then
  # Prefer event dump (all events including benign); has anomaly=True per entry. Use tail for speed.
  if kubectl -n "${NAMESPACE}" exec "${DETECTOR_POD}" -c detector -- test -f /var/log/sentinel-ebpf/events.jsonl 2>/dev/null; then
    kubectl -n "${NAMESPACE}" exec "${DETECTOR_POD}" -c detector -- tail -n "${EVENTS_TAIL}" /var/log/sentinel-ebpf/events.jsonl > "${VERIFY_DIR}/events.jsonl" 2>/dev/null || true
    [ -s "${VERIFY_DIR}/events.jsonl" ] && EVENT_DUMP="${VERIFY_DIR}/events.jsonl"
  fi
  # Anomaly log (only flagged events) as fallback or supplement
  if kubectl -n "${NAMESPACE}" exec "${DETECTOR_POD}" -c detector -- test -f /var/log/sentinel-ebpf/anomalies.jsonl 2>/dev/null; then
    kubectl -n "${NAMESPACE}" cp "${DETECTOR_POD}:/var/log/sentinel-ebpf/anomalies.jsonl" "${VERIFY_DIR}/anomalies.jsonl" -c detector 2>/dev/null || true
    [ -f "${VERIFY_DIR}/anomalies.jsonl" ] && ANOMALY_LOG="${VERIFY_DIR}/anomalies.jsonl"
  fi
fi

# Run verification
# Only expect sensitive paths the generator actually accessed (first SENSITIVE_OPS from SENSITIVE_FILES)
SENSITIVE_FILES="/etc/passwd /etc/shadow /etc/group /etc/sudoers /etc/hosts /etc/ssh/sshd_config /root/.ssh/id_rsa /etc/ssl/private"
SENSITIVE_EXPECTED=""
count=0
for f in ${SENSITIVE_FILES}; do
  [ "${count}" -ge "${SENSITIVE_OPS}" ] && break
  SENSITIVE_EXPECTED="${SENSITIVE_EXPECTED} ${f}"
  count=$((count + 1))
done
SENSITIVE_EXPECTED="${SENSITIVE_EXPECTED# }"  # trim leading space

if [ -n "${ANOMALY_LOG}" ] || [ -n "${EVENT_DUMP}" ]; then
  echo ""
  LOG_FILE="${ROOT_DIR}/activity_verify_$(date +%Y%m%d_%H%M%S).csv"
  ARGS=()
  [ -n "${ANOMALY_LOG}" ] && ARGS+=(--anomaly-log "${ANOMALY_LOG}")
  [ -n "${EVENT_DUMP}" ] && ARGS+=(--event-dump "${EVENT_DUMP}")
  ARGS+=(--log-file "${LOG_FILE}")
  ARGS+=(--after "${GEN_START}" --within 120 --comm "cat,head,ls" --normal-ops "${NORMAL_OPS:-10}")
  [ -n "${SENSITIVE_EXPECTED}" ] && ARGS+=(--sensitive ${SENSITIVE_EXPECTED})
  uv run python "${ROOT_DIR}/scripts/verify_activity_in_anomaly_log.py" "${ARGS[@]}" || true
  VERIFY_EXIT=$?
else
  echo ""
  echo "Could not fetch anomaly log or event dump from detector."
  echo "Ensure detector has ANOMALY_LOG_PATH and/or EVENT_DUMP_PATH set (e.g. anomalyLogPath, eventDumpPath in Helm values)."
  echo "Detector pod: ${DETECTOR_POD:-none found}"
  VERIFY_EXIT=1
fi

echo ""
echo "Activity generation complete."
[ "${VERIFY_EXIT}" -ne 0 ] && exit "${VERIFY_EXIT}"
