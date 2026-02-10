#!/usr/bin/env bash
set -euo pipefail

# Activity generator script for testing sentinel-ebpf probe and detector.
# Generates both normal file activity and suspicious accesses (e.g., /etc/passwd) to trigger anomalies.
# Run in a Kubernetes pod: kubectl run activity-gen --rm -i --restart=Never --image=busybox -- sh -c "$(cat scripts/generate-activity.sh)"

# Configuration
NORMAL_OPS=${NORMAL_OPS:-10}  # Number of normal file operations
SENSITIVE_OPS=${SENSITIVE_OPS:-3}  # Number of sensitive file accesses
DELAY=${DELAY:-0.5}  # Delay between operations (seconds)

echo "=== sentinel-ebpf activity generator ==="
echo "Normal operations: ${NORMAL_OPS}"
echo "Sensitive operations: ${SENSITIVE_OPS}"
echo "Delay: ${DELAY}s"
echo ""

# Normal file operations (should be low anomaly score)
echo "[1/2] Generating normal file activity..."
for i in $(seq 1 "${NORMAL_OPS}"); do
  echo "  Normal op ${i}/${NORMAL_OPS}: Creating/reading temp files..."
  # Create and read files in /tmp (normal location)
  echo "test data ${i}" > "/tmp/sentinel-test-${i}.txt"
  cat "/tmp/sentinel-test-${i}.txt" > /dev/null
  # Read proc files (normal, not sensitive)
  if [ -f /proc/self/status ]; then
    head -5 /proc/self/status > /dev/null
  fi
  if [ -f /proc/version ]; then
    cat /proc/version > /dev/null
  fi
  # Normal directory operations
  ls /tmp > /dev/null 2>&1 || true
  sleep "${DELAY}"
done
echo "✓ Normal operations complete"
echo ""

# Sensitive file accesses (should trigger anomalies)
echo "[2/2] Generating suspicious file activity..."
# Use a space-separated list instead of bash array for POSIX compatibility
SENSITIVE_FILES="/etc/passwd /etc/shadow /etc/group /etc/sudoers /etc/hosts /etc/ssh/sshd_config /root/.ssh/id_rsa /etc/ssl/private"

count=0
for file in ${SENSITIVE_FILES}; do
  if [ ${count} -ge ${SENSITIVE_OPS} ]; then
    break
  fi
  echo "  Suspicious op $((count + 1))/${SENSITIVE_OPS}: Accessing ${file}..."
  # Try to read (will fail for some, but probe still sees the open attempt)
  # Use multiple access patterns to increase visibility
  cat "${file}" > /dev/null 2>&1 || true
  # Also try to stat/ls (different syscall)
  ls -la "${file}" > /dev/null 2>&1 || true
  # Try to open for reading explicitly
  [ -r "${file}" ] && head -1 "${file}" > /dev/null 2>&1 || true
  count=$((count + 1))
  sleep "${DELAY}"
done

# Ensure /etc/passwd is always accessed (most common sensitive file)
if [ ! -f /etc/passwd ]; then
  echo "  Warning: /etc/passwd not found (unusual)"
else
  echo "  Force access to /etc/passwd (guaranteed anomaly trigger)..."
  cat /etc/passwd > /dev/null
  head -5 /etc/passwd > /dev/null
fi

# Also try writing to sensitive locations
if [ -w /etc ]; then
  echo "  Attempting write to /etc/test (should fail or be suspicious)..."
  echo "test" > /etc/sentinel-test-write.txt 2>&1 || true
  rm -f /etc/sentinel-test-write.txt 2>&1 || true
fi

echo "✓ Suspicious operations complete"
echo ""
echo "=== Activity generation finished ==="
echo "Check detector logs for anomaly detections!"
