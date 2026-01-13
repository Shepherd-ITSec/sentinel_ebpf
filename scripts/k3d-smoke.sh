#!/usr/bin/env bash
set -euo pipefail

# Quick k3d-based smoke test for sentinel-ebpf on a single-node cluster.
# Prereqs: docker, k3d, helm. Runs best on Linux where /lib/modules and
# /sys/kernel/debug can be mounted into k3d nodes for eBPF.

CLUSTER_NAME="${CLUSTER_NAME:-sentinel-ebpf}"
AGENT_IMAGE="${AGENT_IMAGE:-sentinel-ebpf-agent:latest}"
DETECTOR_IMAGE="${DETECTOR_IMAGE:-sentinel-ebpf-detector:latest}"
CHART_PATH="${CHART_PATH:-./charts/sentinel-ebpf}"

info() { echo "[+] $*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

require docker
require k3d
require helm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/pyproject.toml" || ! -f "${ROOT_DIR}/uv.lock" ]]; then
  echo "pyproject.toml or uv.lock missing in ${ROOT_DIR}; run from repo root" >&2
  exit 1
fi

# Prefer BuildKit only if buildx binary exists and works.
BUILDX_BIN="${BUILDX_BIN:-/usr/local/lib/docker/cli-plugins/docker-buildx}"
if [[ -x "$BUILDX_BIN" ]] && docker buildx version >/dev/null 2>&1; then
  info "Building images with BuildKit (buildx available)"
  export DOCKER_BUILDKIT=1
else
  info "Buildx missing or broken; forcing legacy builder (DOCKER_BUILDKIT=0)"
  export DOCKER_BUILDKIT=0
fi

docker build -f agent/Dockerfile -t "${AGENT_IMAGE}" .
docker build -f detector/Dockerfile -t "${DETECTOR_IMAGE}" .

unset DOCKER_BUILDKIT

info "Creating k3d cluster ${CLUSTER_NAME}"
k3d cluster delete "${CLUSTER_NAME}" >/dev/null 2>&1 || true
k3d cluster create "${CLUSTER_NAME}" \
  --agents 1 \
  --servers 1 \
  --k3s-arg "--disable=traefik@server:0" \
  --volume /lib/modules:/lib/modules:ro@all \
  --volume /sys/kernel/debug:/sys/kernel/debug:rw@all

info "Importing images into cluster"
k3d image import -c "${CLUSTER_NAME}" "${AGENT_IMAGE}" "${DETECTOR_IMAGE}"

info "Ensuring kubeconfig is set"
export KUBECONFIG="$(k3d kubeconfig write "${CLUSTER_NAME}")"

info "Installing Helm chart"
helm install sentinel-ebpf "${CHART_PATH}" \
  --set agent.image.repository="${AGENT_IMAGE}" \
  --set detector.image.repository="${DETECTOR_IMAGE}"

wait_for_resource() {
  local kind="$1"; shift
  local selector="$1"; shift
  local timeout="${1:-60}"
  local start
  start=$(date +%s)
  while true; do
    if kubectl get "$kind" -l "$selector" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      echo "Timed out waiting for $kind with selector: $selector" >&2
      return 1
    fi
    sleep 2
  done
}

info "Waiting for daemonset and detector"
wait_for_resource "daemonset" "app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=agent" 90
kubectl rollout status daemonset -l app.kubernetes.io/name=sentinel-ebpf -l app.kubernetes.io/component=agent --timeout=180s
wait_for_resource "deploy" "app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector" 90
kubectl rollout status deploy -l app.kubernetes.io/name=sentinel-ebpf -l app.kubernetes.io/component=detector --timeout=180s

info "Triggering a test write/read from a pod"
kubectl run tester --rm -i --tty --image=busybox --restart=Never -- sh -c "echo x >> /etc/hosts && cat /etc/hosts >/dev/null"

info "Recent agent logs"
kubectl logs daemonset/sentinel-ebpf-agent -c agent --tail=40

info "Recent detector logs"
kubectl logs deploy/sentinel-ebpf-detector -c detector --tail=40

info "Done. Delete cluster with: k3d cluster delete ${CLUSTER_NAME}"
