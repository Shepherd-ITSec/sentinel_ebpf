# Container Architecture Explanation

This document explains the containers deployed by the sentinel-ebpf Helm chart.

## Overview

The sentinel-ebpf system consists of **3 main containers** (plus 1 init container):

1. **Probe** (DaemonSet) - eBPF-based event collector
2. **Detector** (Deployment) - Anomaly detection service
3. **UI** (Deployment, optional) - Debug web interface

---

## 1. Probe Container (`sentinel-ebpf-probe`)

### Type: DaemonSet
**Purpose**: Runs on every node to capture filesystem events using eBPF

### Key Characteristics:
- **Deployment**: DaemonSet (one pod per Kubernetes node)
- **Image**: `ghcr.io/shepherd-itsec/sentinel-ebpf-probe:latest`
- **Privileges**: Requires `privileged: true` and `hostPID: true` for eBPF access
- **Capabilities**: `SYS_ADMIN`, `SYS_RESOURCE`, `SYS_PTRACE`

### What it does:
1. **Loads eBPF program** into the kernel that hooks into `open()` and `openat()` syscalls
2. **Captures file access events** (filename, process info, flags, timestamps)
3. **Filters events** based on rules (e.g., only capture files matching certain paths)
4. **Streams events** to the detector via gRPC (or writes to file/stdout)

### Volumes mounted:
- `/lib/modules` (hostPath, read-only) - Kernel headers for eBPF compilation
- `/sys/kernel/debug` (hostPath, read-only) - Kernel debugging interface
- `/etc/sentinel-ebpf/probe-config.yaml` (ConfigMap) - Probe configuration
- `/etc/sentinel-ebpf/rules.yaml` (ConfigMap) - Filtering rules
- `/var/log/sentinel-ebpf` (emptyDir or PVC) - Log storage (if file mode enabled)

### Ports:
- **9101** (health) - Health check endpoint (`/healthz`)
- **9102** (metrics) - Prometheus metrics endpoint (`/metrics`)

### Init Container:
- **`wait-for-detector`** (busybox) - Waits for detector service to be ready before probe starts

### Configuration:
- **Stream mode**: `grpc` (default), `stdout`, or `file`
- **Rules**: YAML file defining which file paths/processes to monitor
- **Ring buffer**: Configurable size for event buffering

---

## 2. Detector Container (`sentinel-ebpf-detector`)

### Type: Deployment
**Purpose**: Receives events from probes and performs anomaly detection

### Key Characteristics:
- **Deployment**: Deployment (default: 1 replica, can be scaled)
- **Image**: `ghcr.io/shepherd-itsec/sentinel-ebpf-detector:latest`
- **Privileges**: Normal (no special permissions needed)

### What it does:
1. **Receives events** from probes via gRPC bidirectional streaming
2. **Extracts features** from events (path hash, process info, timestamps, etc.)
3. **Runs anomaly detection** using online ML models (Half-Space Trees, LODA, or MemStream)
4. **Scores events** and marks anomalies based on threshold
5. **Stores recent events** in memory buffer for UI log tail
6. **Exposes metrics** in Prometheus format

### Ports:
- **50051** (grpc) - gRPC service for receiving events from probes
- **50052** (events) - HTTP API for:
  - `/recent_events` - Recent event log (for UI)
  - `/metrics` - Prometheus metrics

### Environment Variables (Model Configuration):
- `DETECTOR_MODEL_ALGORITHM` - Algorithm: `halfspacetrees`, `loda`, or `memstream`
- `DETECTOR_THRESHOLD` - Anomaly score threshold (0-1)
- `DETECTOR_HST_*` - Half-Space Trees parameters
- `DETECTOR_LODA_*` - LODA parameters
- `DETECTOR_MEMSTREAM_*` - MemStream parameters

### Health Checks:
- **Liveness**: TCP check on gRPC port (20s initial delay)
- **Readiness**: TCP check on gRPC port (15s initial delay)

### Features:
- **Online learning**: Models adapt continuously without separate training phase
- **Thread-safe**: Handles concurrent gRPC streams from multiple probes
- **Metrics**: Tracks events processed, anomalies detected, errors

---

## 3. UI Container (`sentinel-ebpf-ui`)

### Type: Deployment (Optional)
**Purpose**: Web-based debug interface for monitoring the system

### Key Characteristics:
- **Deployment**: Deployment (1 replica)
- **Image**: `ghcr.io/shepherd-itsec/sentinel-ebpf-ui:latest`
- **Enabled**: `false` by default (enable with `--set ui.enabled=true` or `--ui` flag)

### What it does:
1. **Serves web UI** on port 8080 (HTML/CSS/JavaScript)
2. **Fetches status** from probe health endpoints
3. **Displays metrics** from probe and detector (Prometheus format)
4. **Shows event log** - Recent events from detector API (gRPC mode) or log file (file mode)
5. **Auto-refreshes** every 10 seconds (configurable)

### Ports:
- **8080** (http) - Web interface

### Environment Variables:
- `DETECTOR_GRPC_ADDR` - Detector gRPC address (for health check)
- `PROBE_HEALTH_URL` - Probe health endpoint
- `PROBE_METRICS_URL` - Probe metrics endpoint
- `DETECTOR_METRICS_URL` - Detector metrics endpoint
- `DETECTOR_EVENTS_URL` - Detector recent events API
- `LOG_PATH` - Path to log file (if file mode)
- `UI_POLL_SECONDS` - Refresh interval (default: 10)
- `UI_LOG_LIMIT` - Max events to display (default: 50)

### Volumes:
- `/var/log/sentinel-ebpf` (emptyDir or PVC) - Shared with probe for log file access (if file mode)

### Features:
- **Dark/Light mode** toggle
- **Sortable log table** (by time, type, anomaly, score, etc.)
- **Real-time updates** via polling
- **Formatted metrics** display (not raw Prometheus text)

---

## Container Communication Flow

```
┌─────────────────┐         ┌─────────────────┐
│  Probe Pod       │         │  Probe Pod       │
│  (DaemonSet)     │         │  (DaemonSet)     │
│  Node 1          │         │  Node 2          │
│  - eBPF hooks    │         │  - eBPF hooks    │
│  - Event capture │         │  - Event capture │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │ gRPC stream                │ gRPC stream
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌─────────────────────┐
         │  Detector Pod        │
         │  (Deployment)        │
         │  - Anomaly detection │
         │  - gRPC service      │
         │  - HTTP API          │
         └──────────┬───────────┘
                    │
                    │ HTTP
                    │
         ┌──────────▼───────────┐
         │   UI Pod             │
         │   (Deployment)       │
         │   - Web interface    │
         │   - Status display   │
         │   - Metrics viewer   │
         │   - Event log tail   │
         └──────────────────────┘
```

**Note**: 
- **Probe pods**: One per Kubernetes node (DaemonSet)
- **Detector pod**: One replica by default (can be scaled with HPA)
- **UI pod**: One replica (optional, disabled by default)

---

## Resource Requirements

### Probe:
- **CPU**: No limits by default (eBPF is lightweight)
- **Memory**: No limits by default
- **Storage**: Optional PVC for log persistence (default: emptyDir)

### Detector:
- **CPU**: No limits by default
- **Memory**: Depends on model (MemStream uses more memory)
- **Storage**: None (in-memory only)

### UI:
- **CPU**: 50m request, 200m limit
- **Memory**: 64Mi request, 256Mi limit
- **Storage**: Shared with probe (if file mode)

---

## Security Considerations

### Probe:
- **Requires privileged access** - Needed for eBPF kernel hooks
- **hostPID: true** - Accesses host process namespace
- **Mounts kernel modules** - Needs `/lib/modules` for eBPF compilation
- **Runs as root** - Required for eBPF operations

### Detector & UI:
- **Normal privileges** - No special permissions needed
- **Network access only** - Communicates via gRPC/HTTP

---

## Configuration Modes

### Stream Modes (Probe):
1. **gRPC** (default) - Streams events to detector in real-time
2. **stdout** - Debug mode, prints events to logs
3. **file** - Writes events to binary log file (for offline analysis)

### Detection Algorithms (Detector):
1. **halfspacetrees** - Fast, memory-efficient streaming anomaly detection
2. **loda** - Random projection + histogram-based
3. **memstream** (default) - Autoencoder with memory buffer (more accurate, uses more memory)

---

## Typical Deployment

When you run `./scripts/k3d-setup.sh --build --ui`, you get:

- **2 Probe pods** (one per k3d node - server + agent)
- **1 Detector pod** (scalable deployment)
- **1 UI pod** (if `--ui` flag used)

All containers communicate via Kubernetes services (DNS-based service discovery).
