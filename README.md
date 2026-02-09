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
  - `k3d-setup.sh` – install docker/k3d/helm/uv (optional bpf headers) on Debian/Ubuntu.
  - `k3d-smoke.sh` – uses registry images by default (or --build to build locally), spins up k3d, installs chart, runs a quick write/read test.
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
docker build -t ghcr.io/Shepherd-ITSec/sentinel-ebpf-probe:latest ./probe
docker build -t ghcr.io/Shepherd-ITSec/sentinel-ebpf-detector:latest ./detector
docker build -t ghcr.io/Shepherd-ITSec/sentinel-ebpf-ui:latest ./ui
```

## Helm install
```bash
helm install sentinel-ebpf ./charts/sentinel-ebpf \
  --set probe.image.repository=ghcr.io/Shepherd-ITSec/sentinel-ebpf-probe \
  --set detector.image.repository=ghcr.io/Shepherd-ITSec/sentinel-ebpf-detector
```

Key values and config (see `charts/sentinel-ebpf/values.yaml`):
- `probe.stream.mode`: `grpc`, `stdout`, or `file`.
- `probe.stream.file.*`: `path` (default `/var/log/sentinel-ebpf/events.bin`), rotation (`rotateMaxBytes`, `rotateMaxFiles`), `compress` (gzip whole file).
- `probe.grpc.endpoint`: detector address (default service).
- `probe.stream.ringBufferPages`: ring buffer pages per CPU (power of two, default 64).
- Rule file: `rules.yaml` (mounted at `/etc/sentinel-ebpf/rules.yaml`) controls what events are forwarded; default rules capture all file opens.
- PVC for logs: `probe.storage.pvc.enabled/size/storageClassName/accessModes` (if disabled, uses `emptyDir`).
- `detector.enabled`: disable and point probe to an external detector image.
- `detector.autoscaling.*`: enable HPA for detector.

## Runtime notes
- Probe runs privileged, hostPID, mounts `/lib/modules` and `/sys/kernel/debug`; kernel headers must be available on nodes.
- Health/readiness: probe HTTP `/healthz` (9101), metrics placeholder `/metrics` (9102). Detector TCP probe on gRPC port.
- Config is mounted from ConfigMap to `/etc/sentinel-ebpf/probe-config.yaml`; rules file mounted to `/etc/sentinel-ebpf/rules.yaml`.
- File mode logs land at `probe.stream.file.path`; use PVC for persistence in clusters.
- Optional debug UI is disabled by default; enable it only when needed.

## Testing (kind/minikube quick start)
1. Build/push images to a registry reachable by the cluster.
2. `helm install` as above; wait for probe DaemonSet and detector pods to be ready.
3. For stdout mode debugging: `--set probe.stream.mode=stdout --set detector.enabled=false` then check pod logs.
4. Trigger an open on a sensitive path (e.g., `cat /etc/hosts`) in a test pod; detector should log an anomaly.

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

## Online Detection Models

The detector **online** anomaly detection. Each event is scored and immediately learned, so it continuously adapts without a separate training phase.

Halfspacetrees configuration:

```bash
export DETECTOR_MODEL_ALGORITHM=halfspacetrees
export DETECTOR_THRESHOLD=0.5  # Anomaly score threshold (0-1)
export DETECTOR_HST_N_TREES=25
export DETECTOR_HST_HEIGHT=15
export DETECTOR_HST_WINDOW_SIZE=250
```

LODA configuration:

```bash
export DETECTOR_MODEL_ALGORITHM=loda
export DETECTOR_THRESHOLD=0.5
export DETECTOR_LODA_PROJECTIONS=20
export DETECTOR_LODA_BINS=64
export DETECTOR_LODA_RANGE=3.0
export DETECTOR_LODA_EMA_ALPHA=0.01
export DETECTOR_LODA_HIST_DECAY=1.0
```

MemStream configuration:

```bash
export DETECTOR_MODEL_ALGORITHM=memstream
export DETECTOR_THRESHOLD=0.5
export DETECTOR_MEMSTREAM_HIDDEN_DIM=32
export DETECTOR_MEMSTREAM_LATENT_DIM=8
export DETECTOR_MEMSTREAM_MEMORY_SIZE=128
export DETECTOR_MEMSTREAM_LR=0.001
```

### Available Algorithms

- **halfspacetrees** (default): Streaming anomaly detection with online updates (River).
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

Run tests with pytest (dev extras install pytest into the venv):
```bash
uv sync --extra dev
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

## Scripts quick reference
- `scripts/k3d-setup.sh`: `sudo bash scripts/k3d-setup.sh [--install-ebpf]` installs docker/helm/k3d/uv (and optionally bpf headers). Warns if buildx missing.
- `scripts/k3d-smoke.sh`: From repo root, uses registry images by default (checks locally, then pulls from ghcr.io), creates k3d cluster, installs chart, triggers test read/write, tails logs. Use `--build` flag to build images locally instead. Falls back to legacy builder if buildx missing.
- `scripts/deploy-probe.sh`: Deploy probe in file mode to an existing cluster with permission checks; overrides via env (`NAMESPACE`, `PROBE_IMAGE`, `PROBE_TAG`, `PVC_SIZE`, `PVC_CLASS`, etc.).
- `scripts/decode_logs.py`: Decode EVT1 (or .gz) to NDJSON for offline training.
- `scripts/replay_logs.py`: Feed EVT1 logs into a DetectorService endpoint; supports realtime pacing and time-window filters.

## Optional debug UI (in-cluster)
- Enable UI: `--set ui.enabled=true`
- Access: `kubectl -n <ns> port-forward svc/sentinel-ebpf-ui 8080:8080`
- UI shows probe health, detector gRPC health, probe metrics, and log tail.
- Tune: `ui.pollSeconds`, `ui.logLimit`, `ui.logPath`
