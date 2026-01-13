# sentinel-ebpf (eBPF + detector)

Minimal Falco-style pipeline (not affiliated with Falco) with two components:
- Python/bcc eBPF agent (DaemonSet) that emits file read/write events over gRPC (or stdout for debugging) using generic event_type + data fields.
- Pluggable detector service (default pass-through gRPC server); swap in your anomaly model to decide what is anomalous.
- Deployable via the Helm chart in `charts/sentinel-ebpf/`.

## Layout
- `proto/events.proto` – event envelope + detector RPC.
- `agent/` – bcc-based probe runner, health/metrics endpoints, Dockerfile, rule loader.
- `detector/` – reference rule-based detector + Dockerfile.
- `charts/sentinel-ebpf/` – Helm chart wiring the DaemonSet + detector Deployment, ConfigMap, RBAC, HPA option.
- `scripts/`
  - `k3d-setup.sh` – install docker/k3d/helm/uv (optional bpf headers) on Debian/Ubuntu.
  - `k3d-smoke.sh` – build images, spin up k3d, install chart, run a quick write/read test.
  - `deploy-probe.sh` – deploy agent only (file logging) into an existing cluster with permission checks.
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
docker build -t ghcr.io/example/sentinel-ebpf-agent:latest ./agent
docker build -t ghcr.io/example/sentinel-ebpf-detector:latest ./detector
```

## Helm install
```bash
helm install sentinel-ebpf ./charts/sentinel-ebpf \
  --set agent.image.repository=ghcr.io/example/sentinel-ebpf-agent \
  --set detector.image.repository=ghcr.io/example/sentinel-ebpf-detector
```

Key values and config (see `charts/sentinel-ebpf/values.yaml`):
- `agent.stream.mode`: `grpc`, `stdout`, or `file`.
- `agent.stream.file.*`: `path` (default `/var/log/sentinel-ebpf/events.bin`), rotation (`rotateMaxBytes`, `rotateMaxFiles`), `compress` (gzip whole file).
- `agent.grpc.endpoint`: detector address (default service).
- Rule file: `rules.yaml` (mounted at `/etc/sentinel-ebpf/rules.yaml`) controls what events are forwarded; default rules capture all file reads and writes.
- PVC for logs: `agent.storage.pvc.enabled/size/storageClassName/accessModes` (if disabled, uses `emptyDir`).
- `detector.enabled`: disable and point agent to an external detector image.
- `detector.autoscaling.*`: enable HPA for detector.

## Runtime notes
- Agent runs privileged, hostPID, mounts `/lib/modules` and `/sys/kernel/debug`; kernel headers must be available on nodes.
- Health/readiness: agent HTTP `/healthz` (9101), metrics placeholder `/metrics` (9102). Detector TCP probe on gRPC port.
- Config is mounted from ConfigMap to `/etc/sentinel-ebpf/agent-config.yaml`; rules file mounted to `/etc/sentinel-ebpf/rules.yaml`.
- File mode logs land at `agent.stream.file.path`; use PVC for persistence in clusters.

## Testing (kind/minikube quick start)
1. Build/push images to a registry reachable by the cluster.
2. `helm install` as above; wait for agent DaemonSet and detector pods to be ready.
3. For stdout mode debugging: `--set agent.stream.mode=stdout --set detector.enabled=false` then check pod logs.
4. Trigger a write to a sensitive path (e.g., `/etc/hosts`) in a test pod; detector should log an anomaly.

## Rule file (falco-like, minimal)
Mounted at `/etc/sentinel-ebpf/rules.yaml` via ConfigMap. Example (default):
```yaml
rules:
  - name: capture-all-writes
    enabled: true
    event: file_write
    match:
      pathPrefixes: ["/"]
  - name: capture-all-reads
    enabled: true
    event: file_read
    match:
      pathPrefixes: ["/"]
```
Supported fields: `event` (`file_read`|`file_write`), `pathPrefixes` (prefix match), and `comms` (process name allowlist). Agent forwards any event matching at least one enabled rule; the detector decides anomalies.

Payload format sent by agent:
- `event_type`: string (e.g., `file_read`, `file_write`)
- `data`: map with `filename`, `bytes`, `comm`, `pid`, `tid` as strings

## Extending detector
- Implement the `DetectorService` gRPC contract (`proto/events.proto`), package your model in an image, and deploy via `--set detector.image.repository=...`.
- The agent sends `EventEnvelope` with `event_type` (e.g., `file_read`/`file_write`) and `data` map containing filename/bytes/comm/pid/tid, metadata (node/pod), and UUID `event_id`. The default detector just logs; your anomaly model should decide.

## File mode & offline decoding
- Enable file mode: `--set agent.stream.mode=file --set agent.storage.pvc.enabled=true` (plus PVC size/class).
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

## Scripts quick reference
- `scripts/k3d-setup.sh`: `sudo bash scripts/k3d-setup.sh [--install-ebpf]` installs docker/helm/k3d/uv (and optionally bpf headers). Warns if buildx missing.
- `scripts/k3d-smoke.sh`: From repo root, builds images, creates k3d cluster, installs chart, triggers test read/write, tails logs. Falls back to legacy builder if buildx missing.
- `scripts/deploy-probe.sh`: Deploy agent in file mode to an existing cluster with permission checks; overrides via env (`NAMESPACE`, `AGENT_IMAGE`, `AGENT_TAG`, `PVC_SIZE`, `PVC_CLASS`, etc.).
- `scripts/decode_logs.py`: Decode EVT1 (or .gz) to NDJSON for offline training.
- `scripts/replay_logs.py`: Feed EVT1 logs into a DetectorService endpoint; supports realtime pacing and time-window filters.
