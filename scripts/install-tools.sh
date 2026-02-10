#!/usr/bin/env bash
set -euo pipefail

# Install basic tools required for sentinel-ebpf development and testing.
# Installs: docker, k3d, helm, kubectl (if missing).
# Run with --help for more options.

INSTALL_DOCKER="${INSTALL_DOCKER:-false}"
INSTALL_K3D="${INSTALL_K3D:-false}"
INSTALL_HELM="${INSTALL_HELM:-false}"
INSTALL_KUBECTL="${INSTALL_KUBECTL:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --docker)
      INSTALL_DOCKER=true
      shift
      ;;
    --k3d)
      INSTALL_K3D=true
      shift
      ;;
    --helm)
      INSTALL_HELM=true
      shift
      ;;
    --kubectl)
      INSTALL_KUBECTL=true
      shift
      ;;
    --all)
      INSTALL_DOCKER=true
      INSTALL_K3D=true
      INSTALL_HELM=true
      INSTALL_KUBECTL=true
      shift
      ;;
    --help)
      echo "Usage: $0 [--docker] [--k3d] [--helm] [--kubectl] [--all]"
      echo ""
      echo "Install required tools for sentinel-ebpf development."
      echo ""
      echo "Options:"
      echo "  --docker   Install Docker"
      echo "  --k3d      Install k3d (Kubernetes in Docker)"
      echo "  --helm     Install Helm"
      echo "  --kubectl  Install kubectl"
      echo "  --all      Install all tools"
      echo ""
      echo "If no flags are provided, checks what's missing and prompts to install."
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage" >&2
      exit 1
      ;;
  esac
done

info() { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

# Detect OS
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  OS="${ID}"
else
  warn "Cannot detect OS. Assuming Debian/Ubuntu."
  OS="ubuntu"
fi

check_command() {
  command -v "$1" >/dev/null 2>&1
}

install_docker() {
  if check_command docker; then
    info "Docker is already installed: $(docker --version)"
    return 0
  fi
  
  info "Installing Docker..."
  if [[ "${OS}" == "ubuntu" ]] || [[ "${OS}" == "debian" ]]; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/${OS}/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${OS} $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER" || true
    info "Docker installed. You may need to log out and back in for group changes to take effect."
  else
    warn "Automatic Docker installation not supported for ${OS}. Please install manually:"
    echo "  https://docs.docker.com/engine/install/"
    return 1
  fi
}

install_k3d() {
  if check_command k3d; then
    info "k3d is already installed: $(k3d version | head -1)"
    return 0
  fi
  
  info "Installing k3d..."
  curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
  info "k3d installed"
}

install_helm() {
  if check_command helm; then
    info "Helm is already installed: $(helm version --short)"
    return 0
  fi
  
  info "Installing Helm..."
  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  info "Helm installed"
}

install_kubectl() {
  if check_command kubectl; then
    info "kubectl is already installed: $(kubectl version --client --short 2>/dev/null || echo 'installed')"
    return 0
  fi
  
  info "Installing kubectl..."
  KUBECTL_VERSION="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
  curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
  sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
  rm kubectl
  info "kubectl installed"
}

# If no flags, check what's missing and prompt
if [[ "${INSTALL_DOCKER}" == "false" ]] && \
   [[ "${INSTALL_K3D}" == "false" ]] && \
   [[ "${INSTALL_HELM}" == "false" ]] && \
   [[ "${INSTALL_KUBECTL}" == "false" ]]; then
  MISSING=()
  check_command docker || MISSING+=("docker")
  check_command k3d || MISSING+=("k3d")
  check_command helm || MISSING+=("helm")
  check_command kubectl || MISSING+=("kubectl")
  
  if [[ ${#MISSING[@]} -eq 0 ]]; then
    info "All required tools are installed!"
    exit 0
  fi
  
  warn "Missing tools: ${MISSING[*]}"
  echo "Run with --all to install all, or specify individual tools:"
  echo "  $0 --all"
  echo "  $0 --docker --k3d --helm"
  exit 1
fi

# Install requested tools
[[ "${INSTALL_DOCKER}" == "true" ]] && install_docker
[[ "${INSTALL_K3D}" == "true" ]] && install_k3d
[[ "${INSTALL_HELM}" == "true" ]] && install_helm
[[ "${INSTALL_KUBECTL}" == "true" ]] && install_kubectl

info "Tool installation complete!"
