#!/usr/bin/env bash
set -euo pipefail

# Deploy sentinel-ebpf agent in file-only mode for data collection.
# Defaults target the local chart; override via env:
#   NAMESPACE=default
#   CHART_PATH=./charts/sentinel-ebpf
#   AGENT_IMAGE=ghcr.io/example/sentinel-ebpf-agent
#   AGENT_TAG=latest
#   PVC_SIZE=5Gi
#   PVC_CLASS=""

NAMESPACE="${NAMESPACE:-default}"
CHART_PATH="${CHART_PATH:-./charts/sentinel-ebpf}"
RELEASE="${RELEASE:-sentinel-ebpf}"
AGENT_IMAGE="${AGENT_IMAGE:-ghcr.io/example/sentinel-ebpf-agent}"
AGENT_TAG="${AGENT_TAG:-latest}"
PVC_SIZE="${PVC_SIZE:-5Gi}"
PVC_CLASS="${PVC_CLASS:-}"

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }

require kubectl
require helm

echo "[+] Checking cluster access"
kubectl cluster-info >/dev/null

echo "[+] Checking permissions in namespace ${NAMESPACE}"
kubectl auth can-i create configmaps -n "${NAMESPACE}" >/dev/null
kubectl auth can-i create daemonsets -n "${NAMESPACE}" >/dev/null
kubectl auth can-i create persistentvolumeclaims -n "${NAMESPACE}" >/dev/null
kubectl auth can-i create serviceaccounts -n "${NAMESPACE}" >/dev/null
echo "[+] Checking cluster-wide permissions (for RBAC install)"
kubectl auth can-i create clusterrole >/dev/null
kubectl auth can-i create clusterrolebinding >/dev/null

echo "[+] Deploying (file mode)"
helm upgrade --install "${RELEASE}" "${CHART_PATH}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set detector.enabled=false \
  --set agent.stream.mode=file \
  --set agent.stream.file.path=/var/log/sentinel-ebpf/events.bin \
  --set agent.stream.file.rotateMaxBytes=52428800 \
  --set agent.stream.file.rotateMaxFiles=5 \
  --set agent.stream.file.compress=false \
  --set agent.storage.pvc.enabled=true \
  --set agent.storage.pvc.size="${PVC_SIZE}" \
  --set agent.storage.pvc.storageClassName="${PVC_CLASS}" \
  --set agent.image.repository="${AGENT_IMAGE}" \
  --set agent.image.tag="${AGENT_TAG}"

echo "[+] Waiting for DaemonSet rollout"
kubectl -n "${NAMESPACE}" rollout status daemonset -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=agent --timeout=180s

echo "[+] Done. Logs will accumulate under /var/log/sentinel-ebpf/ in the mounted PVC."
