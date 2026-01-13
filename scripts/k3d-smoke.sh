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

info "Building images"
docker build -t "${AGENT_IMAGE}" ./agent
docker build -t "${DETECTOR_IMAGE}" ./detector

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

info "Waiting for daemonset and detector"
kubectl rollout status daemonset/sentinel-ebpf-agent --timeout=90s
kubectl rollout status deploy/sentinel-ebpf-detector --timeout=90s

info "Triggering a test write/read from a pod"
kubectl run tester --rm -i --tty --image=busybox --restart=Never -- sh -c "echo x >> /etc/hosts && cat /etc/hosts >/dev/null"

info "Recent agent logs"
kubectl logs daemonset/sentinel-ebpf-agent -c agent --tail=40

info "Recent detector logs"
kubectl logs deploy/sentinel-ebpf-detector -c detector --tail=40

info "Done. Delete cluster with: k3d cluster delete ${CLUSTER_NAME}"
