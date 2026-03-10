## Scripts overview

This directory contains helper scripts for working with sentinel-ebpf. They are grouped by purpose below.

### BETH / detector evaluation

- **`convert_beth_to_evt1.py`**: Convert BETH CSV logs into EVT1 binary plus labels NDJSON, used for BETH replay/evaluation.
- **`run_detector_eval.py`**: Unified eval script. **BETH mode** (default): convert train/test CSV, warm up on train, replay test, evaluate; **single-stream mode** (`--evt1` + `--labels`): replay one EVT1 (e.g. synthetic), evaluate. Supports single-run and `--run-all` matrix; single-stream matrix is 12 runs (evil_only only), BETH matrix is 24 runs.
- **`evaluate_beth_replay.py`**: Compute classification metrics (precision, recall, F1, etc.) by comparing a detector anomaly log (`anomalies.jsonl`) against BETH-style labels.

### EVT1 utilities

- **`replay_logs.py`**: Replay EVT1 logs to the detector over gRPC, either as fast as possible or in realtime pacing.
- **`decode_logs.py`**: Decode EVT1 binary logs (optionally gzipped) into human-readable NDJSON for inspection or debugging.
- **`compare_replay_scores.py`**: Take a slice of `detector-events` JSONL, replay it to a fresh detector instance, and compare original vs replayed scores to sanity‑check determinism and state handling.

### Synthetic / activity generation

- **`generate_synthetic_evt1_dataset.py`**: Generate a synthetic EVT1 log plus matching labels with configurable total size, positive fraction, and warmup region (no positives at the start). By default it samples only network syscall rows from BETH (`--network-only`, disable with `--no-network-only`). Useful for controlled FPR/recall experiments.

**Testing the detector on a new synthetic network run**

1. **Generate a new synthetic EVT1 + labels** (network-only rows by default, with `event_type: network`):
   ```bash
   uv run python scripts/generate_synthetic_evt1_dataset.py \
     --out-prefix test_data/synthetic/run4 \
     --total-events 100000 \
     --positive-fraction 0.01 \
     --warmup-fraction 0.75
   ```
   This writes `test_data/synthetic/run4.evt1` and `test_data/synthetic/run4.labels.ndjson`.

2. **Single eval run** (one algorithm/threshold):
   ```bash
   uv run python scripts/run_detector_eval.py \
     --evt1 test_data/synthetic/run4.evt1 \
     --labels test_data/synthetic/run4.labels.ndjson \
     --out-dir test_data/synthetic/eval
   ```
   The script starts the detector, replays the EVT1 to it over gRPC, then evaluates detector anomalies against the labels. Output: `test_data/synthetic/eval/metrics.json` (precision, recall, F1, etc.).

3. **Full matrix (all algorithms × thresholds)**:
   ```bash
   ./scripts/run_synthetic_overnight.sh test_data/synthetic/run4.evt1 test_data/synthetic/run4.labels.ndjson run4.log
   ```
   Or with defaults (run3.evt1 / run3.labels.ndjson): `./scripts/run_synthetic_overnight.sh`. Results go under `test_data/synthetic/run_all_<timestamp>/` with one run per (algorithm, threshold); each run has `metrics.json` and the manifest summarizes precision/recall/F1.

4. **Generate new data and run the full matrix in one go**:
   ```bash
   ./scripts/run_synthetic_overnight.sh --generate test_data/synthetic/run4
   ```
   Optional second argument is total events (default 100000): `./scripts/run_synthetic_overnight.sh --generate test_data/synthetic/run5 50000`. The script generates the EVT1 and labels, then starts the matrix in the background; watch with `tail -f run4.log` (log name from the output prefix).
- **`generate-activity.sh`**: Run inside a pod or host to generate a mix of normal and suspicious file activity (e.g., reading `/etc/passwd`) to exercise the probe and detector.
- **`run-activity-generator.sh`**: Convenience wrapper for running the activity generator against a cluster or environment (see script for invocation details).
- **`verify_activity_in_anomaly_log.py`**: Given an anomaly log and/or full event dump, report which generated activity paths were flagged, and check that benign vs sensitive paths match expectations (helps validate the activity generator and detector behavior).

### Long-running / orchestration helpers

- **`run_beth_overnight.sh`**: Fire-and-forget wrapper that starts `run_detector_eval.py --run-all` (BETH mode) under `nohup`, logs to `run_all.log`.
- **`run_synthetic_overnight.sh`**: Same in single-stream mode: `run_detector_eval.py --evt1/--labels --run-all`. No args = defaults to `test_data/synthetic/run3.evt1` and `run3.labels.ndjson`; optional args: `[evt1] [labels] [log]`. Warmup is in the data: generate the EVT1 with `--warmup-fraction 0.75` so the first part of the stream has no positives and the detector can learn normal behavior.

### Analysis / visualization

- **`analyze_dataset.py`**: Load detector event dump JSONL, plot loss/anomaly score over samples, and optionally count near-duplicate records (0/1/2/3‑field differences) to inspect dataset quality and anomaly distribution.
- **`feature_attribution.py`**: Compute perturbation‑based feature attribution for detector events. Replays the stream up to a target event, then generates bar‑chart PNGs (and optional JSON reports) under `test_data/attribution/` by default; supports attributing multiple consecutive events via `--num-events`.

### Cluster / tooling setup

- **`install-tools.sh`**: Install external dependencies such as Docker, k3d, Helm, and kubectl (driven by flags like `--docker`, `--k3d`, `--helm`, `--kubectl`, or `--all`).
- **`k3d-setup.sh`**: Create and configure a k3d Kubernetes cluster suitable for running sentinel-ebpf.
- **`k3d-smoke.sh`**: Run a smoke test against a k3d cluster to verify basic deployment and detector/probe behavior.
- **`deploy-probe.sh`**: Deploy the eBPF probe and detector components into a Kubernetes cluster (typically k3d) using the Helm chart.

### Environment / kernel checks

- **`preflight-check.sh`**: Check that the local environment is ready for eBPF development (kernel version, modules directory, debugfs, BTF presence, BCC bindings/tools).

### Protobuf / code generation

- **`gen_proto.sh`**: Regenerate Python gRPC/protobuf stubs from the `events.proto` definition used by the detector and replay tooling.

