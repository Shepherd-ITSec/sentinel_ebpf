#!/usr/bin/env bash
set -euo pipefail

# Setup script for k3d + toolchain on Debian/Ubuntu (WSL-friendly).
# Run with sudo: sudo bash scripts/k3d-setup.sh [--install-ebpf]

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo bash scripts/k3d-setup.sh)" >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-}"
if [[ -z "$TARGET_USER" ]]; then
  TARGET_USER="$(logname 2>/dev/null || true)"
fi
if [[ -z "$TARGET_USER" || "$TARGET_USER" == "root" ]]; then
  TARGET_USER=""
fi

INSTALL_EBPF=0
for arg in "$@"; do
  case "$arg" in
    --install-ebpf|-e) INSTALL_EBPF=1 ;;
    --help|-h)
      echo "Usage: sudo bash scripts/k3d-setup.sh [--install-ebpf]"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

apt update
apt install -y \
  docker.io \
  curl \
  ca-certificates \
  gnupg \
  lsb-release \
  bash

if [[ "${INSTALL_EBPF}" == "1" ]]; then
  apt install -y linux-headers-$(uname -r) bpfcc-tools || true
else
  echo "Skipping kernel headers/bpfcc-tools (pass --install-ebpf to attempt)"
fi

systemctl enable docker >/dev/null 2>&1 || true
systemctl start docker >/dev/null 2>&1 || true

if ! command -v k3d >/dev/null 2>&1; then
  curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
fi

install_buildx() {
  local version="${BUILDX_VERSION:-v0.14.0}"
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) arch="amd64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) echo "Unsupported arch for buildx: $arch" >&2; return 1 ;;
  esac

  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -sSL "https://github.com/docker/buildx/releases/download/${version}/buildx-${version}.linux-${arch}" \
    -o /usr/local/lib/docker/cli-plugins/docker-buildx
  chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
}

if ! docker buildx version >/dev/null 2>&1; then
  if apt-get install -y docker-buildx-plugin >/dev/null 2>&1; then
    echo "Installed docker-buildx-plugin via apt"
  else
    echo "Falling back to manual buildx install"
    install_buildx || echo "Failed to install buildx; builds may require DOCKER_BUILDKIT=0"
  fi
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "WARNING: buildx is still unavailable. You can force legacy builds with DOCKER_BUILDKIT=0."
fi

if ! command -v kubectl >/dev/null 2>&1; then
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  install -m 0755 kubectl /usr/local/bin/kubectl
  rm kubectl
fi

if ! command -v helm >/dev/null 2>&1; then
  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Ensure PATH includes \$HOME/.local/bin in your shell (e.g., echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc)"
if [[ -n "$TARGET_USER" ]]; then
  echo "To enable non-root docker: sudo usermod -aG docker $TARGET_USER && newgrp docker"
else
  echo "To enable non-root docker: sudo usermod -aG docker <username> && newgrp docker"
fi
echo "If buildx is still missing, retry: BUILDX_VERSION=v0.14.0 sudo bash scripts/k3d-setup.sh"
echo "Setup complete."
