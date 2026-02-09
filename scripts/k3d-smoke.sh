#!/usr/bin/env bash
set -euo pipefail

# Quick k3d-based smoke test for sentinel-ebpf on a single-node cluster.
# Prereqs: docker, k3d, helm. Runs best on Linux where /lib/modules and
# /sys/kernel/debug can be mounted into k3d nodes for eBPF.
#
# By default, uses images from ghcr.io/Shepherd-ITSec/ (checks locally first, then pulls).
# Use --build flag to build images locally instead.

CLUSTER_NAME="${CLUSTER_NAME:-sentinel-ebpf}"
PROBE_IMAGE="${PROBE_IMAGE:-ghcr.io/Shepherd-ITSec/sentinel-ebpf-probe:latest}"
DETECTOR_IMAGE="${DETECTOR_IMAGE:-ghcr.io/Shepherd-ITSec/sentinel-ebpf-detector:latest}"
CHART_PATH="${CHART_PATH:-./charts/sentinel-ebpf}"
NAMESPACE="${NAMESPACE:-default}"
BUILD_IMAGES="${BUILD_IMAGES:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      BUILD_IMAGES=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--build]" >&2
      exit 1
      ;;
  esac
done

info() { echo "[+] $*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

require docker
require k3d
require helm

# Check Docker access early
if ! docker info >/dev/null 2>&1; then
  echo "Error: Cannot access Docker daemon. Permission denied." >&2
  echo "  - Ensure your user is in the 'docker' group: sudo usermod -aG docker $USER" >&2
  echo "  - Or run with sudo (not recommended)" >&2
  echo "  - After adding to docker group, log out and back in" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

#region debug probe log setup
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/.debug.log}"
SESSION_ID="debug-session"
RUN_ID="smoke-$(date +%s)"
log_debug() {
  local hypo="$1"; shift
  local msg="$1"; shift
  local data="$1"; shift
  mkdir -p "$(dirname "${LOG_PATH}")" 2>/dev/null || true
  touch "${LOG_PATH}" 2>/dev/null || true
  # shellcheck disable=SC2129
  echo "{\"sessionId\":\"${SESSION_ID}\",\"runId\":\"${RUN_ID}\",\"hypothesisId\":\"${hypo}\",\"location\":\"scripts/k3d-smoke.sh\",\"message\":\"${msg}\",\"data\":${data},\"timestamp\":$(date +%s%3N)}" >> "${LOG_PATH}" 2>/dev/null || true
  # also emit to stdout for immediate visibility
  echo "[log-debug ${hypo}] ${msg} ${data}" || true
}
#endregion

if [[ ! -f "${ROOT_DIR}/pyproject.toml" || ! -f "${ROOT_DIR}/uv.lock" ]]; then
  echo "pyproject.toml or uv.lock missing in ${ROOT_DIR}; run from repo root" >&2
  exit 1
fi

# Handle image building/pulling
if [[ "${BUILD_IMAGES}" == "true" ]]; then
  info "Building images locally (--build flag provided)"
  # When building locally, use lowercase local tags (Docker requires lowercase)
  # Registry names with uppercase are fine for pulling, but not for building
  LOCAL_PROBE_IMAGE="${PROBE_IMAGE}"
  LOCAL_DETECTOR_IMAGE="${DETECTOR_IMAGE}"
  # If using registry names, convert to local lowercase tags for building
  if [[ "${PROBE_IMAGE}" == *"/"* ]]; then
    LOCAL_PROBE_IMAGE="sentinel-ebpf-probe:latest"
  fi
  if [[ "${DETECTOR_IMAGE}" == *"/"* ]]; then
    LOCAL_DETECTOR_IMAGE="sentinel-ebpf-detector:latest"
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
  log_debug "H1" "build_start" "{\"builder\":\"$(if [[ ${DOCKER_BUILDKIT:-0} -eq 1 ]]; then echo buildkit; else echo legacy; fi)\",\"probe_image\":\"${LOCAL_PROBE_IMAGE}\",\"detector_image\":\"${LOCAL_DETECTOR_IMAGE}\"}"
  docker build -f probe/Dockerfile -t "${LOCAL_PROBE_IMAGE}" .
  docker build -f detector/Dockerfile -t "${LOCAL_DETECTOR_IMAGE}" .
  # Update image variables to use local tags for k3d import
  PROBE_IMAGE="${LOCAL_PROBE_IMAGE}"
  DETECTOR_IMAGE="${LOCAL_DETECTOR_IMAGE}"
  unset DOCKER_BUILDKIT
else
  info "Checking for images locally or pulling from registry"
  # Check if probe image exists locally
  if ! docker image inspect "${PROBE_IMAGE}" >/dev/null 2>&1; then
    info "Probe image not found locally, pulling from registry: ${PROBE_IMAGE}"
    docker pull "${PROBE_IMAGE}" || {
      echo "Failed to pull ${PROBE_IMAGE}. Use --build to build locally or ensure images are available." >&2
      exit 1
    }
  else
    info "Using existing local probe image: ${PROBE_IMAGE}"
  fi
  
  # Check if detector image exists locally
  if ! docker image inspect "${DETECTOR_IMAGE}" >/dev/null 2>&1; then
    info "Detector image not found locally, pulling from registry: ${DETECTOR_IMAGE}"
    docker pull "${DETECTOR_IMAGE}" || {
      echo "Failed to pull ${DETECTOR_IMAGE}. Use --build to build locally or ensure images are available." >&2
      exit 1
    }
  else
    info "Using existing local detector image: ${DETECTOR_IMAGE}"
  fi
fi

info "Creating k3d cluster ${CLUSTER_NAME}"
k3d cluster delete "${CLUSTER_NAME}" >/dev/null 2>&1 || true
k3d cluster create "${CLUSTER_NAME}" \
  --agents 1 \
  --servers 1 \
  --k3s-arg "--disable=traefik@server:0" \
  --volume /lib/modules:/lib/modules:ro@all \
  --volume /sys/kernel/debug:/sys/kernel/debug:rw@all

info "Importing images into cluster"
k3d image import -c "${CLUSTER_NAME}" "${PROBE_IMAGE}" "${DETECTOR_IMAGE}"

info "Ensuring kubeconfig is set"
export KUBECONFIG="$(k3d kubeconfig write "${CLUSTER_NAME}")"

info "Installing Helm chart"
helm install sentinel-ebpf "${CHART_PATH}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set probe.image.repository="$(cut -d: -f1 <<< "${PROBE_IMAGE}")" \
  --set probe.image.tag="$(cut -d: -f2 <<< "${PROBE_IMAGE}")" \
  --set probe.clusterName="${CLUSTER_NAME}" \
  --set detector.image.repository="$(cut -d: -f1 <<< "${DETECTOR_IMAGE}")" \
  --set detector.image.tag="$(cut -d: -f2 <<< "${DETECTOR_IMAGE}")"
INSTALL_RC=$?
log_debug "H2" "helm_install" "{\"rc\":${INSTALL_RC},\"namespace\":\"${NAMESPACE}\"}"
if [[ ${INSTALL_RC} -ne 0 ]]; then
  log_debug "H2" "helm_install_failed" "{\"rc\":${INSTALL_RC}}"
  exit ${INSTALL_RC}
fi

all_ds=$(kubectl get ds -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name} {end}' || echo "")
all_dep=$(kubectl get deploy -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name} {end}' || echo "")
log_debug "H2" "post_helm_resources" "$(printf '{\"ds\":\"%s\",\"deploy\":\"%s\"}' "${all_ds}" "${all_dep}")"

wait_for_resource() {
  local kind="$1"; shift
  local selector="$1"; shift
  local timeout="${1:-30}"
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
    sleep 1
  done
}

info "Waiting for daemonset and detector"
if wait_for_resource "daemonset" "app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe" 30; then
  count_ds=$(kubectl -n "${NAMESPACE}" get ds -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -o name | wc -l || true)
  log_debug "H3" "ds_found" "{\"count\":${count_ds}}"
  kubectl rollout status daemonset -l app.kubernetes.io/name=sentinel-ebpf -l app.kubernetes.io/component=probe --timeout=60s -n "${NAMESPACE}" || true
else
  log_debug "H3" "ds_missing" "{\"namespace\":\"${NAMESPACE}\"}"
fi

if wait_for_resource "deploy" "app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector" 30; then
  count_dep=$(kubectl -n "${NAMESPACE}" get deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o name | wc -l || true)
  log_debug "H4" "deploy_found" "{\"count\":${count_dep}}"
  kubectl rollout status deploy -l app.kubernetes.io/name=sentinel-ebpf -l app.kubernetes.io/component=detector --timeout=60s -n "${NAMESPACE}" || true
else
  log_debug "H4" "deploy_missing" "{\"namespace\":\"${NAMESPACE}\"}"
fi

all_ds2=$(kubectl get ds -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name} {end}' || echo "")
all_dep2=$(kubectl get deploy -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name} {end}' || echo "")
log_debug "H5" "post_wait_resources" "$(printf '{\"ds\":\"%s\",\"deploy\":\"%s\"}' "${all_ds2}" "${all_dep2}")"

# Inspect DS labels if present
ds_name=$(kubectl -n "${NAMESPACE}" get ds -l app.kubernetes.io/name=sentinel-ebpf -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [[ -n "${ds_name}" ]]; then
  ds_labels=$(kubectl -n "${NAMESPACE}" get ds "${ds_name}" -o jsonpath='{.metadata.labels}' 2>/dev/null || echo "")
  log_debug "H6" "ds_labels" "$(printf '{\"name\":\"%s\",\"labels\":\"%s\"}' "${ds_name}" "${ds_labels}")"
fi

dep_name=$(kubectl -n "${NAMESPACE}" get deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [[ -n "${dep_name}" ]]; then
  dep_labels=$(kubectl -n "${NAMESPACE}" get deploy "${dep_name}" -o jsonpath='{.metadata.labels}' 2>/dev/null || echo "")
  log_debug "H6" "dep_labels" "$(printf '{\"name\":\"%s\",\"labels\":\"%s\"}' "${dep_name}" "${dep_labels}")"
fi

# Pod status snapshots
probe_pods=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -o jsonpath='{range .items[*]}{.metadata.name}:{.status.phase}:{range .status.containerStatuses[*]}{.name}:{.ready}:{.state}{\";\"}{end} {end}' 2>/dev/null || echo "")
log_debug "H7" "probe_pod_status" "$(printf '{\"pods\":\"%s\"}' "${probe_pods}")"
det_pods=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o jsonpath='{range .items[*]}{.metadata.name}:{.status.phase}:{range .status.containerStatuses[*]}{.name}:{.ready}:{.state}{\";\"}{end} {end}' 2>/dev/null || echo "")
log_debug "H8" "detector_pod_status" "$(printf '{\"pods\":\"%s\"}' "${det_pods}")"

# DS/Deploy status fields
if [[ -n "${ds_name}" ]]; then
  ds_status=$(kubectl -n "${NAMESPACE}" get ds "${ds_name}" -o jsonpath='{.status.numberAvailable}/{.status.numberReady}/{.status.desiredNumberScheduled}/{.status.currentNumberScheduled}' 2>/dev/null || echo "")
  log_debug "H9" "ds_status" "$(printf '{\"name\":\"%s\",\"status\":\"%s\"}' "${ds_name}" "${ds_status}")"
fi
if [[ -n "${dep_name}" ]]; then
  dep_status=$(kubectl -n "${NAMESPACE}" get deploy "${dep_name}" -o jsonpath='{.status.availableReplicas}/{.status.readyReplicas}/{.status.updatedReplicas}/{.status.replicas}' 2>/dev/null || echo "")
  log_debug "H10" "deploy_status" "$(printf '{\"name\":\"%s\",\"status\":\"%s\"}' "${dep_name}" "${dep_status}")"
fi

# Pod list across namespaces for sanity
probe_pods_all=$(kubectl get pods -A -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}:{.status.phase};{end}' 2>/dev/null || echo "")
det_pods_all=$(kubectl get pods -A -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}:{.status.phase};{end}' 2>/dev/null || echo "")
log_debug "H11" "pod_status_all" "$(printf '{\"probe\":\"%s\",\"detector\":\"%s\"}' "${probe_pods_all}" "${det_pods_all}")"

# Pod waiting reasons / conditions
probe_wait=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -o jsonpath='{range .items[*]}{.metadata.name}:{range .status.containerStatuses[*]}{.state.waiting.reason}:{.state.waiting.message};{end} {end}' 2>/dev/null || echo "")
det_wait=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o jsonpath='{range .items[*]}{.metadata.name}:{range .status.containerStatuses[*]}{.state.waiting.reason}:{.state.waiting.message};{end} {end}' 2>/dev/null || echo "")
log_debug "H12" "pod_waiting" "$(printf '{\"probe\":\"%s\",\"detector\":\"%s\"}' "${probe_wait}" "${det_wait}")"

probe_cond=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -o jsonpath='{range .items[*]}{.metadata.name}:{range .status.conditions[*]}{.type}={.status}:{.reason}:{.message};{end} {end}' 2>/dev/null || echo "")
det_cond=$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -o jsonpath='{range .items[*]}{.metadata.name}:{range .status.conditions[*]}{.type}={.status}:{.reason}:{.message};{end} {end}' 2>/dev/null || echo "")
log_debug "H13" "pod_conditions" "$(printf '{\"probe\":\"%s\",\"detector\":\"%s\"}' "${probe_cond}" "${det_cond}")"

info "Triggering a test write/read from a pod"
kubectl run tester --rm -i --tty --image=busybox --restart=Never -- sh -c "echo x >> /etc/hosts && cat /etc/hosts >/dev/null"

info "Recent probe logs"
kubectl logs daemonset/sentinel-ebpf-probe -c probe --tail=40

info "Recent detector logs"
kubectl logs deploy/sentinel-ebpf-detector -c detector --tail=40

info "Done. Delete cluster with: k3d cluster delete ${CLUSTER_NAME}"
