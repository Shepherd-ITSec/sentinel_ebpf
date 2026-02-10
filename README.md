# sentinel-ebpf (eBPF + detector)

Minimal Falco-style pipeline (not affiliated with Falco) with two components:
- Python/bcc eBPF probe (DaemonSet) that emits file read/write events over gRPC (or stdout for debugging) using generic event_type + data fields.
- Pluggable detector service (default pass-through gRPC server); swap in your anomaly model to decide what is anomalous.
- Deployable via the Helm chart in `charts/sentinel-ebpf/`.

## Layout
- `proto/events.proto` – event envelope + detector RPC.
- `probe/` – bcc-based probe runner, health/metrics endpoints, Dockerfile, rule loader.
- `detector/` – reference rule-based detector + Dockerfile.
- `charts/sentinel-ebpf/` – Helm chart wiring the DaemonSet + detector Deployment, ConfigMap, RBAC, HPA option.
- `scripts/`
  - `install-tools.sh` – install docker/k3d/helm/kubectl on Debian/Ubuntu. Run `./scripts/install-tools.sh --all` to install all missing tools.
  - `k3d-setup.sh` – dev environment setup: spins up/updates k3d cluster, deploys sentinel-ebpf, forwards UI. Use `--build` to build images locally, `--purge` for fresh cluster.
  - `k3d-smoke.sh` – smoke test script: creates k3d cluster, installs chart, runs activity generator, shows logs.
  - `generate-activity.sh` – activity generator: creates normal file ops and suspicious accesses (e.g., /etc/passwd) to trigger anomalies.
  - `run-activity-generator.sh` – wrapper to run activity generator in a Kubernetes pod.
  - `deploy-probe.sh` – deploy probe only (file logging) into an existing cluster with permission checks.
  - `preflight-check.sh` – local node checks for BCC/kernel prerequisites.
  - `decode_logs.py` – EVT1 → NDJSON decoder for offline training.
  - `replay_logs.py` – replay EVT1 logs into a DetectorService gRPC endpoint.

## Build & generate stubs
```bash
cd /home/user/master
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-dev
uv run make proto
```

Build images (Dockerfiles use uv internally):
```bash
docker build -t ghcr.io/shepherd-itsec/sentinel-ebpf-probe:latest ./probe
docker build -t ghcr.io/shepherd-itsec/sentinel-ebpf-detector:latest ./detector
docker build -t ghcr.io/shepherd-itsec/sentinel-ebpf-ui:latest ./ui
```

## Quick Start (Dev Environment)

**Install prerequisites:**
```bash
./scripts/install-tools.sh --all
```

**Spin up dev environment:**
```bash
# Update existing cluster or create if missing
./scripts/k3d-setup.sh

# Build images locally and deploy
./scripts/k3d-setup.sh --build

# Enable UI using registry images (no build required)
./scripts/k3d-setup.sh --ui

# Build images including UI and enable UI
./scripts/k3d-setup.sh --build --ui

# Fresh cluster (delete and recreate)
./scripts/k3d-setup.sh --purge

# Build + fresh cluster + UI
./scripts/k3d-setup.sh --build --purge --ui
```

The `k3d-setup.sh` script will:
- Check/update existing k3d cluster (or create if missing)
- Build or pull images as needed (UI image pulled from registry if `--ui` is used without `--build`, or built locally if `--build --ui` is used)
- Import images into cluster (skips if already present)
- Install/upgrade Helm chart
- Wait for probe and detector to be ready
- **Automatically port-forward UI** to `http://localhost:8080` if `--ui` flag is used

## Helm install (manual)

```bash
helm install sentinel-ebpf ./charts/sentinel-ebpf \
  --set probe.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-probe \
  --set detector.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-detector
```

Key values and config (see `charts/sentinel-ebpf/values.yaml`):
- `probe.stream.mode`: `grpc`, `stdout`, or `file`.
- `probe.stream.file.*`: `path` (default `/var/log/sentinel-ebpf/events.bin`), rotation (`rotateMaxBytes`, `rotateMaxFiles`), `compress` (gzip whole file).
- `probe.grpc.endpoint`: detector address (defaults to service name from chart fullname).
- `probe.stream.ringBufferPages`: ring buffer pages per CPU (power of two, default 64).
- Rule file: `rules.yaml` (mounted at `/etc/sentinel-ebpf/rules.yaml`) controls what events are forwarded; default rules capture all file opens.
- PVC for logs: `probe.storage.pvc.enabled/size/storageClassName/accessModes` (if disabled, uses `emptyDir`).
- `detector.enabled`: disable and point probe to an external detector image.
- `detector.model.*`: model algorithm and parameters (see Model Configuration below).
- `detector.autoscaling.*`: enable HPA for detector.
- `ui.enabled`: enable debug UI (default: false).

## Runtime notes
- Probe runs privileged, hostPID, mounts `/lib/modules` and `/sys/kernel/debug`; kernel headers must be available on nodes.
- Health/readiness: probe HTTP `/healthz` (9101), metrics placeholder `/metrics` (9102). Detector TCP probe on gRPC port.
- Config is mounted from ConfigMap to `/etc/sentinel-ebpf/probe-config.yaml`; rules file mounted to `/etc/sentinel-ebpf/rules.yaml`.
- File mode logs land at `probe.stream.file.path`; use PVC for persistence in clusters.
- Optional debug UI is disabled by default; enable it only when needed.

## Rule file (falco-like, minimal)
Mounted at `/etc/sentinel-ebpf/rules.yaml` via ConfigMap. Example (default):
```yaml
rules:
  - name: capture-all-opens
    enabled: true
    event: file_open
    match:
      pathPrefixes: ["/"]
```
Supported fields: `event` (`file_open`), `pathPrefixes` (prefix match), `comms` (process name allowlist), and optional `pids`, `tids`, `uids`. The probe forwards any event matching at least one enabled rule; the detector decides anomalies.

Payload format sent by probe:
- `event_type`: string (e.g., `file_open`)
- `data`: ordered vector of strings. For `file_open`: `[filename, flags, comm, pid, tid, uid]`
- `attributes.open_flags`: decoded open flags (userspace enrichment)

## Model Configuration

The detector uses **online** anomaly detection. Each event is scored and immediately learned, so it continuously adapts without a separate training phase.

Configure via Helm values (see `charts/sentinel-ebpf/values.yaml`):

```yaml
detector:
  model:
    algorithm: "memstream"  # halfspacetrees | loda | memstream (default: memstream)
    threshold: 0.5  # Anomaly score threshold (0-1)
    # Half-Space Trees parameters
    hst_n_trees: 25
    hst_height: 15
    hst_window_size: 250
    # LODA parameters
    loda_n_projections: 20
    loda_bins: 64
    loda_range: 3.0
    loda_ema_alpha: 0.01
    loda_hist_decay: 1.0
    # MemStream parameters
    mem_hidden_dim: 32
    mem_latent_dim: 8
    mem_memory_size: 128
    mem_lr: 0.001
    seed: 42
```

Or override via Helm `--set`:
```bash
helm upgrade sentinel-ebpf ./charts/sentinel-ebpf \
  --set detector.model.algorithm=loda \
  --set detector.model.threshold=0.6
```

Alternatively, set environment variables in the detector deployment (the chart sets these from values automatically).

### Available Algorithms

- **halfspacetrees**: Streaming anomaly detection with online updates (River).
- **loda**: Random projection + online histogram density estimation.
- **memstream**: Online autoencoder with memory buffer.

### Feature Extraction

Events are converted to 10-dimensional feature vectors:
- Open flags (log-normalized)
- Path hash (normalized)
- Command hash (normalized)
- PID/TID (normalized)
- UID (normalized)
- Sensitive path indicator
- Path depth
- Hour/minute of day

## Extending detector
- Implement the `DetectorService` gRPC contract (`proto/events.proto`), package your model in an image, and deploy via `--set detector.image.repository=...`.
- The probe sends `EventEnvelope` with `event_type` (e.g., `file_open`) and `data` ordered vector containing `[filename, flags, comm, pid, tid, uid]` (for file events), metadata (node/pod), and UUID `event_id`. Replace `detector/model.py` and `detector/features.py` to use your own anomaly detection model.

## File mode & offline decoding
- Enable file mode: `--set probe.stream.mode=file --set probe.storage.pvc.enabled=true` (plus PVC size/class).
- Logs are binary records (magic `EVT1`, little-endian length, JSON payload). If `compress=true`, file is gzip-compressed.
- Decode to NDJSON for training:
  ```bash
  uv run python scripts/decode_logs.py /var/log/sentinel-ebpf/events.bin > events.ndjson
  ```

## Replay recorded logs into the detector (test mode)
- Replay client: `uv run python scripts/replay_logs.py events.bin --target localhost:50051 --pace realtime|fast [--start-ms ... --end-ms ...]`
- Run detector with embedded replay (loopback):
  ```bash
  uv run python -m detector.server --replay-log events.bin --replay-pace realtime
  ```
  This starts the detector gRPC server and streams the log into it.

## Testing

**Unit tests:**
Run tests with pytest (dev = pytest; detector = numpy, river, torch for ML tests):
```bash
uv sync --extra dev --extra detector
uv run python -m pytest tests/ -v
```

Optional model sanity checks (synthetic data):
```bash
RUN_OPTIONAL_MODEL_TESTS=1 uv run python -m pytest tests/test_models_optional.py -v
```

Test coverage:
- `tests/test_rules.py`: Rule engine matching logic
- `tests/test_config.py`: Configuration loading (probe and detector)
- `tests/test_rules_filtered.py`: Rule filtering without catch-all rules
- `tests/test_decode_logs.py`: EVT1 log decoding
- `tests/test_filesink.py`: File logging and rotation
- `tests/test_events.py`: EventEnvelope serialization
- `tests/test_detector.py`: Detector service (River online scoring)
- `tests/test_replay_logs.py`: Log replay functionality

**Integration testing:**
Generate filesystem activity to test the probe and trigger anomalies:
```bash
# Run activity generator locally
./scripts/generate-activity.sh

# Or run in a Kubernetes pod (requires cluster to be running)
./scripts/run-activity-generator.sh
```

The activity generator creates normal file operations and suspicious accesses (e.g., `/etc/passwd`) to test anomaly detection.

## Scripts quick reference
- `scripts/install-tools.sh`: Install prerequisites (docker, k3d, helm, kubectl). Run `./scripts/install-tools.sh --all` to install all missing tools, or `--docker`, `--k3d`, `--helm`, `--kubectl` for specific tools.
- `scripts/k3d-setup.sh`: Dev environment setup script. Updates existing k3d cluster (or creates if missing), builds/pulls images, imports into cluster (skips if already present), installs/upgrades Helm chart, waits for readiness, and optionally port-forwards UI. Use `--build` to build images locally, `--purge` for fresh cluster, `--ui` to enable UI deployment and port-forwarding.
- `scripts/k3d-smoke.sh`: Smoke test script. Creates k3d cluster, installs chart, runs activity generator, shows logs. Use `--build` to build images locally.
- `scripts/generate-activity.sh`: Generate filesystem activity for testing (normal ops + suspicious accesses like `/etc/passwd`).
- `scripts/run-activity-generator.sh`: Wrapper to run activity generator in a Kubernetes pod.
- `scripts/deploy-probe.sh`: Deploy probe in file mode to an existing cluster with permission checks; overrides via env (`NAMESPACE`, `PROBE_IMAGE`, `PROBE_TAG`, `PVC_SIZE`, `PVC_CLASS`, etc.).
- `scripts/decode_logs.py`: Decode EVT1 (or .gz) to NDJSON for offline training.
- `scripts/replay_logs.py`: Feed EVT1 logs into a DetectorService endpoint; supports realtime pacing and time-window filters.

## Optional debug UI (in-cluster)
- Enable UI: `--set ui.enabled=true` (or use `k3d-setup.sh --ui` to pull from registry, or `k3d-setup.sh --build --ui` to build locally)
- Access: `kubectl -n <ns> port-forward svc/sentinel-ebpf-ui 8080:8080` (or use `k3d-setup.sh --ui` which does this automatically)
- UI shows:
  - **Status**: Probe and detector health, connection status
  - **Metrics**: Probe Prometheus metrics and detector metrics (events/anomalies counters)
  - **Logs**: Sortable table of recent events (from detector API when using gRPC mode, or log file when using file mode)
- Features: Dark/light mode toggle, auto-refresh, sortable log columns
- Tune: `ui.pollSeconds`, `ui.logLimit`, `ui.logPath` in Helm values
