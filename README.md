# sentinel-ebpf

Minimal eBPF security pipeline (Falco-style, not affiliated with Falco):

- `probe`: eBPF DaemonSet (Python + BCC) capturing file-open events.
- `detector`: gRPC service for anomaly scoring (rule-based reference + pluggable models).
- `ui`: optional debug dashboard.
- `charts/sentinel-ebpf`: Helm chart to deploy all components.

Detailed reference and advanced explanations: `docs/README_DETAILED.md`

---

## Quickstart: choose your path

### 1) Full stack (probe + detector, fastest local dev)

Use this when you want a working end-to-end pipeline quickly.

```bash
./scripts/install-tools.sh --all
./scripts/k3d-setup.sh --build
```

Optional UI:

```bash
./scripts/k3d-setup.sh --build --ui
```

Then generate activity:

```bash
./scripts/run-activity-generator.sh
kubectl -n default logs -l app.kubernetes.io/component=detector -c detector --tail=50
```

### 2) Probe-only (file logging, no detector)

Use this for data collection or offline model work.

```bash
./scripts/deploy-probe.sh
```

This deploys probe in `file` mode with PVC storage and writes EVT1 logs to:
`/var/log/sentinel-ebpf/events.bin`

Decode to NDJSON:

```bash
uv run python scripts/decode_logs.py /var/log/sentinel-ebpf/events.bin > events.ndjson
```

### 3) Offline replay into detector

Use this to test detector behavior with recorded logs.

```bash
uv run python scripts/replay_logs.py events.bin --target localhost:50051 --pace fast
```

Or run detector with embedded replay:

```bash
uv run python -m detector.server --replay-log events.bin --replay-pace realtime
```

### 4) Quick smoke test

Use this for a clean one-shot cluster test.

```bash
./scripts/k3d-smoke.sh --build
```

---

## Local development

### Python environment and proto stubs

```bash
uv sync --extra dev --extra detector
uv run make proto
```

### Run tests

```bash
uv run python -m pytest tests/ -v
```

Optional model tests:

```bash
RUN_OPTIONAL_MODEL_TESTS=1 uv run python -m pytest tests/test_models_optional.py -v
```

---

## Helm install (manual)

```bash
helm upgrade --install sentinel-ebpf ./charts/sentinel-ebpf \
  --namespace default \
  --create-namespace \
  --set probe.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-probe \
  --set detector.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-detector
```

Important values in `charts/sentinel-ebpf/values.yaml`:

- `probe.stream.mode`: `grpc`, `stdout`, or `file`
- `probe.stream.file.*`: file path + rotation/compression settings
- `probe.grpc.endpoint`: detector address
- `probe.storage.pvc.*`: persistent storage for file mode
- `detector.enabled`: disable detector when running probe-only
- `detector.model.*`: algorithm + thresholds/hyperparameters
- `ui.enabled`: debug UI (off by default)

---

## Rules DSL (Falco-like)

Rules are mounted at `/etc/sentinel-ebpf/rules.yaml`.
Detailed rules authoring and execution guide: `docs/RULES_GUIDE.md`.

Path exclusions should be expressed in DSL macros/conditions (Falco-like style),
for example `not noisy_path` in file-focused rules.

Rule format is DSL-only: `lists`, `macros`, `condition`.

Minimal example:

```yaml
lists:
  file_events: [open, openat, openat2, unlink, unlinkat, rename, renameat]
  shell_comms: [bash, sh, zsh]

macros:
  file_evt: "event_name in (file_events)"
  sensitive_path: "path startswith /etc or path startswith /root"

rules:
  - name: capture-sensitive-file-events
    enabled: true
    condition: "file_evt and sensitive_path and comm in (shell_comms)"

  - name: capture-network-connectivity
    enabled: true
    condition: "event_name in (socket, connect)"
```

Supported fields: `event_name`, `event_id`, `path`, `comm`, `pid`, `tid`, `uid`, `open_flags`, `arg0`, `arg1`, `arg_flags`, `return_value`, `hostname`, `namespace`

Supported operators: `=`, `in`, `startswith`, `contains`, `and`, `or`, `not`, parentheses

---

## Model config (detector)

Set via `detector.model.*` in Helm values:

- `algorithm`: `halfspacetrees`, `loda`, or `memstream`
- `threshold`: anomaly threshold
- per-algorithm params: `hst_*`, `loda_*`, `mem_*`, plus `seed`


The detector scores events online (streaming), no separate batch training phase required.

---

## Project layout

- `probe/`: eBPF probe runtime, rules, metrics, config loading
- `detector/`: gRPC detector service, feature extraction, models
- `ui/`: optional in-cluster debug UI
- `proto/events.proto`: event envelope + detector gRPC contract
- `charts/sentinel-ebpf/`: Helm chart
- `scripts/`: setup, deploy, replay, and test helpers

---

## Notes and requirements

- Probe needs privileged access, `/lib/modules`, and `/sys/kernel/debug`.
- Kernel headers must exist on cluster nodes (`/lib/modules/$(uname -r)/build`).
- Use `./scripts/preflight-check.sh` on Linux hosts for a quick prerequisites check.
