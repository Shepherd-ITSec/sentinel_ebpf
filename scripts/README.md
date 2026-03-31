## Supported Scripts

This directory now focuses on the current supported workflows. Old experiment manifests and one-off helper scripts have been removed from the main surface.

### Replay and evaluation

- `decode_logs.py`: Decode EVT1 binary logs to NDJSON.
- `replay_logs.py`: Replay EVT1 or detector JSONL to the detector over gRPC.
- `evaluate_replay.py`: Evaluate anomaly outputs against labels NDJSON.
- `run_detector_eval.py`: Start a detector, replay EVT1, and write metrics for one run or a matrix of runs.
- `compare_replay_scores.py`: Sanity-check replay determinism by comparing original and replayed scores.

### LID-DS and checkpoint workflow

- `replay_lidds.py`: Convert LID-DS recordings to detector-compatible JSONL and optionally replay them.
- `train_detector_checkpoint.py`: Train a detector offline from JSONL or EVT1 and save a checkpoint.
- `score_from_checkpoint.py`: Load a detector checkpoint and score a JSONL or EVT1 stream.

Example sequence-model flow:

```bash
uv run python -m scripts.train_detector_checkpoint \
  test_data/lidds/cve-2012-2122.train.jsonl \
  --algorithm sequence_mlp \
  --out test_data/lidds/sequence_mlp_training.pkl

uv run python -m scripts.score_from_checkpoint \
  test_data/lidds/cve-2012-2122.test.jsonl \
  --algorithm sequence_mlp \
  --checkpoint test_data/lidds/sequence_mlp_training.pkl \
  --out test_data/lidds/sequence_mlp_test_scores.jsonl
```

### Synthetic data and overnight runs

- `generate_synthetic_evt1_dataset.py`: Generate synthetic EVT1 plus matching labels.
- `run_synthetic_overnight.sh`: Launch the synthetic eval matrix under `nohup`.
- `run_commands_overnight.sh`: Run a file of commands or an explicit command list sequentially overnight.

### Analysis and diagnostics

- `analyze_dataset.py`: Plot and inspect replay dumps or detector event dumps.
- `feature_attribution.py`: Compute perturbation-based feature attribution for detector events.
- `model_diagnostic.py`: Run a compact unlabeled diagnostic over a replay stream.

### Cluster and activity helpers

- `install-tools.sh`: Install local dependencies such as Docker, k3d, Helm, and kubectl.
- `k3d-setup.sh`: Create or update a k3d cluster and deploy sentinel-ebpf.
- `k3d-smoke.sh`: Run an end-to-end k3d smoke test.
- `deploy-probe.sh`: Deploy probe-only mode for file logging.
- `generate-activity.sh`: Produce mixed benign and suspicious activity.
- `run-activity-generator.sh`: Run the activity generator against a cluster.
- `verify_activity_in_anomaly_log.py`: Check which generated activities were flagged.
- `preflight-check.sh`: Verify the local host is ready for eBPF development.
- `gen_proto.sh`: Regenerate Python protobuf and gRPC stubs.

