#!/usr/bin/env bash
# Test: run probe + detector, trigger cat /etc/hosts on host, check if detector notices.
set -euo pipefail

cd "$(dirname "$0")/.."
DETECTOR_LOG=$(mktemp)
PROBE_LOG=$(mktemp)
PROBE_CFG=$(mktemp -d)
trap "rm -f $DETECTOR_LOG $PROBE_LOG; rm -rf $PROBE_CFG" EXIT

# Probe config with grpc mode, pointing to localhost:50051
cp -r /tmp/probe-test/* "$PROBE_CFG/" 2>/dev/null || mkdir -p "$PROBE_CFG"
cp charts/sentinel-ebpf/rules.yaml "$PROBE_CFG/"
cat > "$PROBE_CFG/probe-config.yaml" << 'CFG'
logLevel: INFO
rulesFile: /etc/sentinel-ebpf/rules.yaml
health: { port: 9101 }
metrics: { enabled: true, port: 9102 }
grpc: { endpoint: localhost:50051, tlsEnabled: false, caSecret: "" }
stream:
  mode: grpc
  batchSize: 512
  queueLength: 50000
  ringBufferPages: 8
CFG

echo "[1/4] Starting detector..."
uv run python -m detector.server 2>&1 | tee "$DETECTOR_LOG" &
DETECTOR_PID=$!
sleep 3
if ! kill -0 $DETECTOR_PID 2>/dev/null; then
  echo "Detector failed to start"
  exit 1
fi

echo "[2/4] Starting probe (docker, traces host)..."
# Probe needs to reach detector: use host network so localhost:50051 works
docker run -d --rm --name sentinel-probe-test --privileged --network host \
  -v /lib/modules:/lib/modules:ro -v /sys/kernel/debug:/sys/kernel/debug:ro \
  -v "$PROBE_CFG":/etc/sentinel-ebpf:ro \
  -e PROBE_CONFIG=/etc/sentinel-ebpf/probe-config.yaml \
  sentinel-ebpf-probe:latest 2>/dev/null || true
sleep 3

echo "[3/4] Triggering cat /etc/hosts on host..."
cat /etc/hosts > /dev/null
sleep 2

echo "[4/4] Checking detector for anomalies..."
kill $DETECTOR_PID 2>/dev/null || true
docker stop sentinel-probe-test 2>/dev/null || true

if grep -q "anomaly" "$DETECTOR_LOG" 2>/dev/null; then
  echo "✓ Detector noticed anomaly (cat /etc/hosts)"
  grep -i anomaly "$DETECTOR_LOG" | tail -5
else
  echo "✗ No anomaly logged. Detector may need more baseline events, or /etc/hosts is not flagged."
  echo "  Last 20 detector log lines:"
  tail -20 "$DETECTOR_LOG"
fi
