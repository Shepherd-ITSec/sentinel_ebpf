# sentinel-ebpf detailed guide

This document contains the full reference and detailed explanations that were intentionally removed from `README.md` to keep onboarding short.

## Architecture

`sentinel-ebpf` is a minimal Falco-style pipeline (not affiliated with Falco) with these components:

- Python/BCC eBPF probe (DaemonSet) that emits file-open events via gRPC (or stdout/file for debugging and offline workflows).
- Pluggable detector service (default rule-based detector) that can be replaced with your own anomaly model.
- Optional UI for status, metrics, and recent logs.
- Helm chart in `charts/sentinel-ebpf/` wiring DaemonSet, Deployment, ConfigMaps, RBAC, and optional HPA.

## Repository layout

- `proto/events.proto`: event envelope and detector RPC contract.
- `probe/`: probe runtime, rule loading, health and metrics endpoints, Dockerfile.
- `detector/`: detector gRPC server, model and feature extraction, Dockerfile.
- `ui/`: optional debug UI and API server.
- `charts/sentinel-ebpf/`: Helm chart and values.
- `scripts/`: dev setup, smoke tests, activity generation, deployment helpers, decode and replay tools.

## Build and generate stubs

Use `uv` for dependency management:

```bash
uv sync --no-dev
uv run make proto
```

Build container images:

```bash
docker build -f probe/Dockerfile -t ghcr.io/shepherd-itsec/sentinel-ebpf-probe:latest .
docker build -f detector/Dockerfile -t ghcr.io/shepherd-itsec/sentinel-ebpf-detector:latest .
docker build -f ui/Dockerfile -t ghcr.io/shepherd-itsec/sentinel-ebpf-ui:latest .
```

## Dev environment setup details

`./scripts/k3d-setup.sh` can:

- Reuse an existing cluster or create one if missing.
- Build images locally (`--build`) or pull/import from registry.
- Enable and port-forward UI (`--ui`) to `http://localhost:8080`.
- Recreate the cluster from scratch (`--purge`).

Examples:

```bash
./scripts/k3d-setup.sh
./scripts/k3d-setup.sh --build
./scripts/k3d-setup.sh --ui
./scripts/k3d-setup.sh --build --ui
./scripts/k3d-setup.sh --purge
./scripts/k3d-setup.sh --build --purge --ui
```

## Manual Helm install and key values

```bash
helm install sentinel-ebpf ./charts/sentinel-ebpf \
  --set probe.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-probe \
  --set detector.image.repository=ghcr.io/shepherd-itsec/sentinel-ebpf-detector
```

Important values in `charts/sentinel-ebpf/values.yaml`:

- `probe.stream.mode`: `grpc`, `stdout`, or `file`.
- `probe.stream.file.*`: file path (default `/var/log/sentinel-ebpf/events.bin`), rotation, and compression options.
- `probe.grpc.endpoint`: detector gRPC address.
- `probe.stream.ringBufferPages`: ring buffer pages per CPU (power of two).
- `probe.storage.pvc.*`: PVC settings for persistent file-mode logs.
- `detector.enabled`: disable in-cluster detector if using external detector.
- `detector.model.*`: detector algorithm and model parameters.
- `detector.autoscaling.*`: HPA options.
- `ui.enabled`: optional debug UI.

## Runtime notes

- Probe runs privileged with `hostPID`, and mounts `/lib/modules` and `/sys/kernel/debug`.
- Kernel headers must be present on nodes (`/lib/modules/$(uname -r)/build`).
- Probe endpoints:
  - health: `/healthz` on port `9101`
  - metrics: `/metrics` on port `9102`
- Detector has gRPC health check on its service port.
- Config mounts:
  - probe config: `/etc/sentinel-ebpf/probe-config.yaml`
  - rules file: `/etc/sentinel-ebpf/rules.yaml`
- File mode writes to `probe.stream.file.path`; use PVC for persistence.

## Rule file (Falco-like DSL)

Rules are mounted at `/etc/sentinel-ebpf/rules.yaml` via ConfigMap.
Deep-dive authoring and execution details: `docs/RULES_GUIDE.md`.

Supported style:

- Hybrid rules with explicit `rules[].syscalls`, optional reusable `lists`/`macros`, boolean `condition`, and `groups` metadata.
- Path exclusions should be encoded in DSL macros/conditions (Falco-like style),
  for example `not noisy_path` in file-focused rules.

Example:

```yaml
lists:
  file_syscalls: [open, openat, openat2, unlink, unlinkat, rename, renameat]
  shell_comms: [bash, sh, zsh]

macros:
  sensitive_path: "path startswith /etc or path startswith /root"

groups:
  file: {}
  network: {}

rules:
  - name: capture-sensitive-shell-file-events
    enabled: true
    group: file
    syscalls: file_syscalls
    condition: "sensitive_path and comm in (shell_comms)"

  - name: capture-network-connectivity
    enabled: true
    group: network
    syscalls: [socket, connect]
```

Supported condition fields:

- `syscall_name`, `syscall_nr`, `event_id` (correlation id), `path`, `comm`, `pid`, `tid`, `uid`, `flags`, `arg0`, `arg1`, `arg_flags`, `return_value`, `hostname`, `namespace`

Supported operators:

- `=`, `in`, `startswith`, `contains`, `and`, `or`, `not`, parentheses

### Hybrid compile behavior (kernel + userspace)

- Probe compiles supported predicates into kernel prefilter tuples (`syscall_name`/`syscall_nr`, `path startswith`, `comm`, numeric IDs).
- Unsupported predicates remain in userspace fallback evaluation.
- Events are forwarded only after full condition evaluation.
- Compile metrics are exposed on probe `/metrics`:
  - `sentinel_ebpf_probe_kernel_compiled_predicates`
  - `sentinel_ebpf_probe_kernel_fallback_predicates`
  - `sentinel_ebpf_probe_kernel_branches_total`
  - `sentinel_ebpf_probe_kernel_branches_compiled`
  - `sentinel_ebpf_probe_kernel_branches_impossible`

## Probe event payload format

Probe sends `EventEnvelope` with:

- `syscall_name`: Linux syscall name (e.g. `openat`, `execve`, `socket`, `connect`).
- `event_group`: rule-defined category (e.g. `network`, `file`, `process`) or empty.
- `data`: ordered string vector with canonical format:
  `[syscall_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]`.
- `attributes.flags`: userspace-decoded flags for open/openat/openat2 events.
- metadata: hostname/pod/namespace/container information and `event_id`.

## Detector model configuration

Detector uses online anomaly detection: each event is scored and immediately learned (streaming behavior, no separate offline training phase required for baseline operation).

Configure in Helm values:

```yaml
detector:
  model:
    algorithm: "memstream" # halfspacetrees | loda | loda_ema | memstream | freq1d | copulatree | latentcluster
    threshold: 0.5
    # Half-Space Trees
    hst_n_trees: 25
    hst_height: 15
    hst_window_size: 250
    # LODA
    loda_n_projections: 20
    loda_bins: 64
    loda_range: 3.0
    loda_ema_alpha: 0.01
    loda_hist_decay: 1.0
    # MemStream (paper-aligned: K-NN discounted L1, latent=2×input_dim)
    mem_memory_size: 512
    mem_lr: 0.01
    mem_beta: 0.1
    mem_k: 3
    mem_gamma: 0.5
    seed: 42
```

Override from CLI:

```bash
helm upgrade sentinel-ebpf ./charts/sentinel-ebpf \
  --set detector.model.algorithm=loda \
  --set detector.model.threshold=0.6
```

Available algorithms:

- `halfspacetrees`: streaming tree-based anomaly detection (River).
- `loda`: PySAD LODA wrapper; paper-faithful, infinite window.
- `loda_ema`: custom LODA with EMA-based adaptive normalization; streaming from first event; supports CUDA.
- `memstream`: online autoencoder with latent-memory scoring and threshold-gated FIFO memory updates.
- `freq1d`: per-feature rarity baseline with configurable score aggregation.
- `copulatree`: streaming copula tree on top of `freq1d` marginals; tracks pairwise Gaussian copula edges and refreshes a maximum-spanning tree online.
- `latentcluster`: online latent clustering in `freq1d`-normalized space.


`memstream` implementation note:
- Paper-aligned: single-layer encoder/decoder (latent=2×input_dim, Tanh), K-NN discounted L1 scoring, FIFO memory when score ≤ β. Optional `mem_warmup_path` for normal-only warmup.
- References:
  - Paper: https://arxiv.org/abs/2106.03837
  - Official repo: https://github.com/Stream-AD/MemStream

`loda` implementation note:
- Uses PySAD LODA (library-based wrapper). Paper-faithful, infinite window.
- References: PySAD https://pysad.readthedocs.io/en/latest/_modules/pysad/models/loda.html

`loda_ema` implementation note:
- Custom in-repo variant with EMA-based adaptive normalization; streaming from first event.
- References: Paper https://doi.org/10.1007/s10994-015-5521-0, PyOD/PySAD notes


### Feature extraction summary

Events are converted to numeric feature vectors. The **feature view** (chosen per algorithm: default, frequency, loda, memstream) controls which optional blocks are included. All string identity uses scalar hashes (`*_hash`, `event_id_norm`); there are no sparse bucket banks. **`default`** keeps **`group_syscall_*`** one-hots and file/process path flags (`file_*`, `proc_*`); **`frequency`** drops those in favor of **`day_fraction_norm`** time encoding. **LODA** / **memstream** drop general comm/hostname/path hashes; **memstream** also drops some **`net_*`** hashes compared to **LODA** (see `detector/README.md`).

## Extending detector

- Implement `DetectorService` in `proto/events.proto`.
- Build your detector image and point Helm at it with `detector.image.repository`.
- Replace or extend the model primitives, feature primitives, or pipeline assembly modules as needed.

## File mode, decoding, and replay

### File mode for collection

- Enable with `probe.stream.mode=file` and PVC enabled.
- Binary log format: records with magic `EVT1`, little-endian length, JSON payload.
- If compression is enabled, output is gzip-compressed.

Decode to NDJSON:

```bash
uv run python scripts/decode_logs.py /var/log/sentinel-ebpf/events.bin > events.ndjson
```

### Replay logs into detector

```bash
uv run python scripts/replay_logs.py events.bin --target localhost:50051 --pace realtime
```

Or run detector in replay mode:

```bash
uv run python -m detector.server --replay-log events.bin --replay-pace realtime
```

## Testing details

Install test dependencies:

```bash
uv sync --extra dev --extra detector
```

Run full unit test suite:

```bash
uv run python -m pytest tests/ -v
```

Optional environment-dependent checks:

```bash
uv run python -m pytest tests/test_cuda.py -m gpu -v
uv run python -m pytest tests/test_speed.py -m perf -v
```

Representative test areas:

- rule matching and filtering,
- probe and detector config loading,
- EVT1 decode and replay tooling,
- file sink and rotation behavior,
- protobuf envelope serialization,
- detector stream behavior and scoring.

## Script reference (detailed)

- `scripts/install-tools.sh`: installs Docker, k3d, Helm, and kubectl (Debian/Ubuntu automation).
- `scripts/k3d-setup.sh`: update/create k3d cluster, build/pull/import images, deploy chart, optional UI forwarding.
- `scripts/k3d-smoke.sh`: clean smoke path for cluster create + deploy + activity + logs.
- `scripts/generate-activity.sh`: mixed benign and suspicious file activity.
- `scripts/run-activity-generator.sh`: run activity generation in-cluster.
- `scripts/deploy-probe.sh`: deploy probe-only in file mode with PVC and permission checks.
- `scripts/preflight-check.sh`: host checks for kernel/BCC prerequisites.
- `scripts/decode_logs.py`: convert EVT1 (or gzip EVT1) to NDJSON.
- `scripts/replay_logs.py`: replay EVT1 to detector gRPC endpoint with `fast`/`realtime` pacing and optional time-window filters.
- `scripts/replay_lidds.py`: convert LID-DS recordings to detector JSONL and optionally replay them.
- `scripts/train_detector_checkpoint.py`: train a detector offline from JSONL or EVT1 and save a checkpoint.
- `scripts/score_from_checkpoint.py`: load a detector checkpoint and score a JSONL or EVT1 stream.
- `scripts/evaluate_replay.py`: evaluate replay anomaly outputs against labels NDJSON.
- `scripts/run_detector_eval.py`: run detector eval (requires --evt1 and --labels); replay, report metrics.
- `scripts/generate_synthetic_evt1_dataset.py`: generate synthetic EVT1 plus labels for controlled eval runs.
- `scripts/run_synthetic_overnight.sh`: launch the synthetic eval matrix under `nohup`.
- `scripts/feature_attribution.py`: compute perturbation-based feature attribution for detector events.
- `scripts/model_diagnostic.py`: run a compact unlabeled detector diagnostic over a replay stream.

## Optional debug UI

Enable:

```bash
./scripts/k3d-setup.sh --ui
```

or:

```bash
helm upgrade --install sentinel-ebpf ./charts/sentinel-ebpf --set ui.enabled=true
kubectl -n default port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
```

UI provides:

- status: probe and detector health/connectivity,
- metrics: probe and detector counters,
- logs: recent events (detector API in gRPC mode or file source in file mode),
- convenience features such as dark/light mode, auto-refresh, and sortable log columns.

