#!/usr/bin/env bash
set -euo pipefail

ok=1

say() { echo "[*] $*"; }
warn() { echo "[!] $*" >&2; ok=0; }

say "Kernel: $(uname -r)"

# Kernel headers
KERNEL_DIR="/lib/modules/$(uname -r)"
if [[ -d "${KERNEL_DIR}" ]]; then
  say "Kernel modules directory present: ${KERNEL_DIR}"
else
  warn "Missing kernel modules directory: ${KERNEL_DIR}"
fi

# Debugfs mount
if awk '$3 == "debugfs" { found=1 } END { exit(found ? 0 : 1) }' /proc/mounts; then
  say "debugfs is mounted"
else
  warn "debugfs is not mounted (try: sudo mount -t debugfs debugfs /sys/kernel/debug)"
fi

# BTF (optional but helpful)
if [[ -f /sys/kernel/btf/vmlinux ]]; then
  say "BTF found: /sys/kernel/btf/vmlinux"
else
  say "BTF not found (CO-RE not required for BCC, but helpful)"
fi

# BCC python bindings
if python3 - <<'PY' >/dev/null 2>&1
import bcc  # noqa: F401
PY
then
  say "python3-bpfcc import ok"
else
  warn "python3-bpfcc not importable (missing python3-bpfcc/bpfcc-tools)"
fi

# BCC tools (best-effort check)
if command -v opensnoop-bpfcc >/dev/null 2>&1 || command -v opensnoop >/dev/null 2>&1; then
  say "bcc tools present"
else
  warn "bcc tools not found in PATH (bpfcc-tools/bcc-tools missing?)"
fi

if [[ ${ok} -eq 1 ]]; then
  say "Preflight OK"
  exit 0
fi

warn "Preflight failed"
exit 1
