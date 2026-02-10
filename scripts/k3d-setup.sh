#!/usr/bin/env bash
set -euo pipefail

# Quick k3d-based dev environment setup for sentinel-ebpf.
# Spins up/updates a k3d cluster, deploys sentinel-ebpf, and optionally forwards the UI.
# Prereqs: docker, k3d, helm. Install missing tools with: ./scripts/install-tools.sh --all
# Runs best on Linux where /lib/modules and /sys/kernel/debug can be mounted into k3d nodes for eBPF.
#
# Usage:
#   ./scripts/k3d-setup.sh                    # Update existing cluster or create if missing
#   ./scripts/k3d-setup.sh --build            # Build images locally and deploy
#   ./scripts/k3d-setup.sh --build --ui       # Build images including UI and enable UI
#   ./scripts/k3d-setup.sh --purge            # Delete existing cluster and create fresh
#   ./scripts/k3d-setup.sh --build --purge --ui  # Build + fresh cluster + UI

CLUSTER_NAME="${CLUSTER_NAME:-sentinel-ebpf}"
PROBE_IMAGE="${PROBE_IMAGE:-ghcr.io/shepherd-itsec/sentinel-ebpf-probe:latest}"
DETECTOR_IMAGE="${DETECTOR_IMAGE:-ghcr.io/shepherd-itsec/sentinel-ebpf-detector:latest}"
UI_IMAGE="${UI_IMAGE:-ghcr.io/shepherd-itsec/sentinel-ebpf-ui:latest}"
CHART_PATH="${CHART_PATH:-./charts/sentinel-ebpf}"
NAMESPACE="${NAMESPACE:-default}"
BUILD_IMAGES="${BUILD_IMAGES:-false}"
PURGE_CLUSTER="${PURGE_CLUSTER:-false}"
ENABLE_UI="${ENABLE_UI:-false}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_USERNAME="${GITHUB_USERNAME:-}"
UI_PORT="${UI_PORT:-8080}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      BUILD_IMAGES=true
      shift
      ;;
    --purge)
      PURGE_CLUSTER=true
      shift
      ;;
    --ui|--enable-ui)
      ENABLE_UI=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--build] [--purge] [--ui]" >&2
      echo "  --build      Build images locally instead of pulling from registry" >&2
      echo "  --purge      Delete existing cluster and create fresh (default: update existing)" >&2
      echo "  --ui         Enable UI deployment and port-forward (requires --build to build UI image)" >&2
      exit 1
      ;;
  esac
done

info() { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

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

# Probe needs kernel headers at /lib/modules/$(uname -r)/build (mounted into k3d nodes)
KERNEL_RELEASE="$(uname -r)"
KERNEL_BUILD="/lib/modules/${KERNEL_RELEASE}/build"
if [[ ! -d "${KERNEL_BUILD}" ]]; then
  echo "Error: Kernel headers not found at ${KERNEL_BUILD}" >&2
  echo "  The probe compiles BPF at runtime and needs kernel headers on the host." >&2
  echo "  Install them, then re-run this script:" >&2
  echo "    sudo apt install linux-headers-${KERNEL_RELEASE}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Check if cluster exists
CLUSTER_EXISTS=false
if k3d cluster list "${CLUSTER_NAME}" >/dev/null 2>&1; then
  CLUSTER_EXISTS=true
fi

# Handle cluster creation/update
if [[ "${PURGE_CLUSTER}" == "true" ]]; then
  if [[ "${CLUSTER_EXISTS}" == "true" ]]; then
    info "Purging existing cluster ${CLUSTER_NAME}"
    k3d cluster delete "${CLUSTER_NAME}" || true
  fi
  info "Creating fresh k3d cluster ${CLUSTER_NAME}"
  k3d cluster create "${CLUSTER_NAME}" \
    --agents 1 \
    --servers 1 \
    --k3s-arg "--disable=traefik@server:0" \
    --volume /lib/modules:/lib/modules:ro@all \
    --volume /sys/kernel/debug:/sys/kernel/debug:rw@all
elif [[ "${CLUSTER_EXISTS}" == "false" ]]; then
  info "Creating k3d cluster ${CLUSTER_NAME}"
  k3d cluster create "${CLUSTER_NAME}" \
    --agents 1 \
    --servers 1 \
    --k3s-arg "--disable=traefik@server:0" \
    --volume /lib/modules:/lib/modules:ro@all \
    --volume /sys/kernel/debug:/sys/kernel/debug:rw@all
else
  info "Using existing cluster ${CLUSTER_NAME}"
fi

# Ensure kubeconfig is set
export KUBECONFIG="$(k3d kubeconfig write "${CLUSTER_NAME}")"

# Handle image building/pulling
if [[ "${BUILD_IMAGES}" == "true" ]]; then
  info "Building images locally"
  LOCAL_PROBE_IMAGE="sentinel-ebpf-probe:latest"
  LOCAL_DETECTOR_IMAGE="sentinel-ebpf-detector:latest"
  
  # Prefer BuildKit if available
  BUILDX_BIN="${BUILDX_BIN:-/usr/local/lib/docker/cli-plugins/docker-buildx}"
  if [[ -x "$BUILDX_BIN" ]] && docker buildx version >/dev/null 2>&1; then
    export DOCKER_BUILDKIT=1
  else
    export DOCKER_BUILDKIT=0
  fi
  
  docker build -f probe/Dockerfile -t "${LOCAL_PROBE_IMAGE}" .
  docker build -f detector/Dockerfile -t "${LOCAL_DETECTOR_IMAGE}" .
  
  PROBE_IMAGE="${LOCAL_PROBE_IMAGE}"
  DETECTOR_IMAGE="${LOCAL_DETECTOR_IMAGE}"
  
  # Build UI image only if UI is enabled
  if [[ "${ENABLE_UI}" == "true" ]]; then
    info "Building UI image..."
    docker build -f ui/Dockerfile -t sentinel-ebpf-ui:latest .
  fi
  
  unset DOCKER_BUILDKIT
  
  # When building locally, always import the newly built images
  # (k3d image import will replace existing images with the same tag)
  info "Importing newly built images into cluster..."
  IMAGES_TO_IMPORT=("${PROBE_IMAGE}" "${DETECTOR_IMAGE}")
  
  # Import UI image only if UI is enabled
  if [[ "${ENABLE_UI}" == "true" ]]; then
    IMAGES_TO_IMPORT+=("sentinel-ebpf-ui:latest")
  fi
  
  info "Importing ${#IMAGES_TO_IMPORT[@]} image(s) into cluster: ${IMAGES_TO_IMPORT[*]}"
  k3d image import -c "${CLUSTER_NAME}" "${IMAGES_TO_IMPORT[@]}"
else
  info "Checking/pulling images from registry"
  
  # Auto-authenticate with GitHub Container Registry if token is available
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    if [[ -z "${GITHUB_USERNAME:-}" ]]; then
      GITHUB_USERNAME="${GITHUB_USERNAME:-$(whoami)}"
    fi
    info "Authenticating with GitHub Container Registry"
    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GITHUB_USERNAME}" --password-stdin >/dev/null 2>&1 || {
      warn "Failed to authenticate with ghcr.io. Continuing anyway..."
    }
  fi
  
  # Pull probe image if not local
  if ! docker image inspect "${PROBE_IMAGE}" >/dev/null 2>&1; then
    info "Pulling probe image: ${PROBE_IMAGE}"
    docker pull "${PROBE_IMAGE}" || {
      echo "Failed to pull ${PROBE_IMAGE}." >&2
      echo "For private images, set GITHUB_TOKEN env var or use --build flag." >&2
      exit 1
    }
  fi
  
  # Pull detector image if not local
  if ! docker image inspect "${DETECTOR_IMAGE}" >/dev/null 2>&1; then
    info "Pulling detector image: ${DETECTOR_IMAGE}"
    docker pull "${DETECTOR_IMAGE}" || {
      echo "Failed to pull ${DETECTOR_IMAGE}." >&2
      echo "For private images, set GITHUB_TOKEN env var or use --build flag." >&2
      exit 1
    }
  fi
  
  # Pull UI image if UI is enabled and not local
  if [[ "${ENABLE_UI}" == "true" ]]; then
    if ! docker image inspect "${UI_IMAGE}" >/dev/null 2>&1; then
      info "Pulling UI image: ${UI_IMAGE}"
      docker pull "${UI_IMAGE}" || {
        echo "Failed to pull ${UI_IMAGE}." >&2
        echo "For private images, set GITHUB_TOKEN env var or use --build flag." >&2
        exit 1
      }
    fi
  fi
  
  # Check if images already exist in cluster before importing
  info "Checking if images are already in cluster..."
  IMAGES_TO_IMPORT=()
  SERVER_NODE="k3d-${CLUSTER_NAME}-server-0"
  
  check_image_in_cluster() {
    local img="$1"
    # Try crictl first (containerd)
    if docker exec "${SERVER_NODE}" crictl images --quiet "${img}" >/dev/null 2>&1; then
      return 0
    fi
    # Fallback: check via ctr (containerd CLI)
    if docker exec "${SERVER_NODE}" sh -c "ctr -n k8s.io images ls | grep -q '${img}'" 2>/dev/null; then
      return 0
    fi
    return 1
  }
  
  # Always check probe and detector images
  for img in "${PROBE_IMAGE}" "${DETECTOR_IMAGE}"; do
    if check_image_in_cluster "${img}"; then
      info "  Image ${img} already in cluster, skipping"
    else
      IMAGES_TO_IMPORT+=("${img}")
    fi
  done
  
  # Check UI image if UI is enabled
  if [[ "${ENABLE_UI}" == "true" ]]; then
    if check_image_in_cluster "${UI_IMAGE}"; then
      info "  Image ${UI_IMAGE} already in cluster, skipping"
    else
      IMAGES_TO_IMPORT+=("${UI_IMAGE}")
    fi
  fi
  
  if [[ ${#IMAGES_TO_IMPORT[@]} -gt 0 ]]; then
    info "Importing ${#IMAGES_TO_IMPORT[@]} image(s) into cluster: ${IMAGES_TO_IMPORT[*]}"
    k3d image import -c "${CLUSTER_NAME}" "${IMAGES_TO_IMPORT[@]}" || true
  else
    info "All images already in cluster, skipping import"
  fi
fi

# Install/upgrade Helm chart
info "Installing/upgrading Helm chart"
UI_SET_ARGS=()
if [[ "${ENABLE_UI}" == "true" ]]; then
  if [[ "${BUILD_IMAGES}" == "true" ]]; then
    # When building locally, use local image name and set pullPolicy to Never
    # since the image is imported into cluster via k3d image import (not from registry)
    UI_SET_ARGS=(
      --set ui.enabled=true
      --set ui.image.repository=sentinel-ebpf-ui
      --set ui.image.tag=latest
      --set ui.image.pullPolicy=Never
    )
  else
    # When using registry, use registry image (default from values.yaml)
    UI_SET_ARGS=(
      --set ui.enabled=true
      --set ui.image.repository="$(cut -d: -f1 <<< "${UI_IMAGE}")"
      --set ui.image.tag="$(cut -d: -f2 <<< "${UI_IMAGE}")"
    )
  fi
  info "UI will be enabled"
fi

# Prepare Helm set arguments
HELM_SET_ARGS=(
  --set probe.image.repository="$(cut -d: -f1 <<< "${PROBE_IMAGE}")"
  --set probe.image.tag="$(cut -d: -f2 <<< "${PROBE_IMAGE}")"
  --set probe.clusterName="${CLUSTER_NAME}"
  --set detector.image.repository="$(cut -d: -f1 <<< "${DETECTOR_IMAGE}")"
  --set detector.image.tag="$(cut -d: -f2 <<< "${DETECTOR_IMAGE}")"
)

# When building locally, set imagePullPolicy to Never for probe and detector
# since images are imported into cluster via k3d (not from registry)
if [[ "${BUILD_IMAGES}" == "true" ]]; then
  HELM_SET_ARGS+=(
    --set probe.image.pullPolicy=Never
    --set detector.image.pullPolicy=Never
  )
fi

helm upgrade --install sentinel-ebpf "${CHART_PATH}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  "${HELM_SET_ARGS[@]}" \
  "${UI_SET_ARGS[@]}"

# Give Helm a moment to create resources
sleep 2

# Wait for resources to be ready
info "Waiting for probe and detector to be ready..."
rollout_ds_pid=""
rollout_ds_failed=false
rollout_deploy_failed=false

if kubectl get daemonset -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -q .; then
  info "Waiting for probe DaemonSet rollout..."
  kubectl rollout status daemonset -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=probe --timeout=90s -n "${NAMESPACE}" &
  rollout_ds_pid=$!
else
  warn "Probe DaemonSet not found"
fi

if kubectl get deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -q .; then
  info "Waiting for detector Deployment rollout..."
  if ! kubectl rollout status deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=detector --timeout=90s -n "${NAMESPACE}"; then
    rollout_deploy_failed=true
    warn "Detector deployment rollout timed out or failed"
    warn "Check pod status: kubectl get pods -l app.kubernetes.io/component=detector -n ${NAMESPACE}"
  fi
else
  warn "Detector Deployment not found"
fi

if [[ -n "${rollout_ds_pid:-}" ]]; then
  if ! wait "${rollout_ds_pid}" 2>/dev/null; then
    rollout_ds_failed=true
    warn "Probe DaemonSet rollout timed out or failed"
    warn "Check pod status: kubectl get pods -l app.kubernetes.io/component=probe -n ${NAMESPACE}"
  fi
fi

if [[ "${rollout_ds_failed}" == "true" ]] || [[ "${rollout_deploy_failed}" == "true" ]]; then
  warn "Some rollouts failed or timed out. Continuing anyway..."
fi

# Check if UI is enabled (verify resources actually exist)
UI_ENABLED=false
UI_DEPLOYMENT=""
UI_SVC="sentinel-ebpf-sentinel-ebpf-ui"

# Check if UI was requested via flag
if [[ "${ENABLE_UI}" == "true" ]]; then
  # Check if UI deployment exists (verify resources are returned, not just command success)
  if kubectl get deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=ui -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -q .; then
    UI_ENABLED=true
    UI_DEPLOYMENT="$(kubectl get deploy -l app.kubernetes.io/name=sentinel-ebpf,app.kubernetes.io/component=ui -n "${NAMESPACE}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")"
    if [[ -n "${UI_DEPLOYMENT}" ]]; then
      # If we built images locally, restart the deployment to pick up the new image
      # (imagePullPolicy=Always ensures it pulls the updated image)
      if [[ "${BUILD_IMAGES}" == "true" ]]; then
        info "Restarting UI deployment to pick up new image..."
        kubectl rollout restart "deploy/${UI_DEPLOYMENT}" -n "${NAMESPACE}"
        sleep 2
      fi
      info "Waiting for UI deployment '${UI_DEPLOYMENT}' to be ready..."
      if ! kubectl rollout status "deploy/${UI_DEPLOYMENT}" --timeout=60s -n "${NAMESPACE}"; then
        warn "UI deployment rollout timed out or failed"
        warn "Check pod status: kubectl get pods -l app.kubernetes.io/component=ui -n ${NAMESPACE}"
      fi
      # Readiness probe ensures HTTP server is ready, but give it a moment for port-forward
      sleep 1
    fi
  else
    warn "UI was requested (--ui flag) but deployment not found"
    warn "This may indicate a Helm chart issue. Checking Helm values..."
    helm get values sentinel-ebpf -n "${NAMESPACE}" 2>/dev/null | grep -A 5 "ui:" || warn "UI section not found in Helm values"
  fi
fi

# Port-forward UI if enabled and service exists
if [[ "${UI_ENABLED}" == "true" ]]; then
  # Verify service exists before port-forwarding
  if kubectl get svc "${UI_SVC}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    info "UI is enabled. Starting port-forward to http://localhost:${UI_PORT}"
    info "Press Ctrl+C to stop port-forwarding (cluster will remain running)"
    echo ""
    kubectl port-forward "svc/${UI_SVC}" "${UI_PORT}:8080" -n "${NAMESPACE}"
  else
    warn "UI deployment exists but service '${UI_SVC}' not found"
    warn "This may indicate the Helm chart didn't create the service properly"
    warn "Check Helm values: helm get values sentinel-ebpf -n ${NAMESPACE}"
  fi
else
  info "UI is not enabled. Enable it with:"
  echo "  helm upgrade sentinel-ebpf ${CHART_PATH} -n ${NAMESPACE} --set ui.enabled=true"
  echo "  kubectl port-forward svc/${UI_SVC} ${UI_PORT}:8080 -n ${NAMESPACE}"
fi
