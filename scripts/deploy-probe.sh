#!/usr/bin/env bash
set -euo pipefail

# Deploy sentinel-ebpf probe in file-only mode for data collection.
# Defaults target the local chart; override via env:
#   NAMESPACE=default
#   CHART_PATH=./charts/sentinel-ebpf
#   PROBE_IMAGE=ghcr.io/Shepherd-ITSec/sentinel-ebpf-probe
#   PROBE_TAG=latest
#   PVC_SIZE=5Gi
#   PVC_CLASS=""
#   CLUSTER_NAME=""

NAMESPACE="${NAMESPACE:-default}"
CHART_PATH="${CHART_PATH:-./charts/sentinel-ebpf}"
RELEASE="${RELEASE:-sentinel-ebpf}"
PROBE_IMAGE="${PROBE_IMAGE:-ghcr.io/Shepherd-ITSec/sentinel-ebpf-probe}"
PROBE_TAG="${PROBE_TAG:-latest}"
PVC_SIZE="${PVC_SIZE:-5Gi}"
PVC_CLASS="${PVC_CLASS:-}"
CLUSTER_NAME="${CLUSTER_NAME:-}"

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
  --set probe.stream.mode=file \
  --set probe.stream.file.path=/var/log/sentinel-ebpf/events.bin \
  --set probe.stream.file.rotateMaxBytes=52428800 \
  --set probe.stream.file.rotateMaxFiles=5 \
  --set probe.stream.file.compress=false \
  --set probe.storage.pvc.enabled=true \
  --set probe.storage.pvc.size="${PVC_SIZE}" \
  --set probe.storage.pvc.storageClassName="${PVC_CLASS}" \
  --set probe.image.repository="${PROBE_IMAGE}" \
  --set probe.image.tag="${PROBE_TAG}" \
  --set probe.clusterName="${CLUSTER_NAME}"

echo "[+] Waiting for DaemonSet rollout"
kubectl -n "${NAMESPACE}" rollout status daemonset -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe --timeout=180s

echo "[+] Done. Logs will accumulate under /var/log/sentinel-ebpf/ in the mounted PVC."
