#!/usr/bin/env bash
set -euo pipefail

# Run the activity generator in a Kubernetes pod.
# Usage: ./scripts/run-activity-generator.sh [namespace]
#   namespace: Kubernetes namespace (default: default)

NAMESPACE="${1:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Running activity generator in namespace: ${NAMESPACE}"
echo ""

# Read the script and pass it to busybox pod
kubectl run activity-generator \
  --rm -i --restart=Never \
  --image=busybox:1.36 \
  --namespace="${NAMESPACE}" \
  -- sh -c "$(cat "${ROOT_DIR}/scripts/generate-activity.sh")"

echo ""
echo "Activity generation complete. Check detector logs for anomalies:"
echo "  kubectl -n ${NAMESPACE} logs -l app.kubernetes.io/component=detector -c detector --tail=50"
