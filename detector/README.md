# Detector

The detector is an anomaly detection service that consumes kernel-level events (e.g. syscalls) over gRPC, extracts features, scores them with an online ML model, and returns anomaly decisions. It can be run standalone (`python -m detector.server`) or in Docker.

## Event model

**syscall_name = Linux syscall name; event_group = category.** There is no separate “network” vs “syscall” category: `syscall_name` is the syscall name (e.g. `openat`, `connect`, `socket`, `execve`); the registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`). The rule-defined category is **event_group** (e.g. `network`, `file`, `process`) and can be empty; it is carried in the protobuf field `EventEnvelope.event_group`.

**One envelope shape for all.** Syscalls are like function calls with defined inputs and outputs. The `EventEnvelope` is the single contract: every producer (eBPF probe, EVT1 replay, future sources) must send the same named syscall fields and attribute keys. For a given event only the relevant fields are non-empty (e.g. `path` for openat, empty for connect). Raw open/socket flag bits are already in `arg0` (open) or `arg1` (openat, openat2, socket) as emitted by the probe; decoded names go in `attributes["flags"]` when present.

---

## Feature Primitives

**Role:** Turn each `EventEnvelope` into numeric features for tabular models, or sequence-context payloads for sequence models. The current implementation lives under `detector/building_blocks/primitives/features/`, mainly in:

- `extractor.py`
- `views.py`
- `generic.py`
- `groups/file.py`
- `groups/network.py`
- `rules.py`

Extraction is layered:

1. **General** — shared numeric context (optionally enabled by the selected feature view): ids/args, calendar time, (optionally) path depth/prefix hash, and (optionally) return code normalization. Some views also enable scalar hashes (`*_hash`).
2. **Group-specific** — only when `evt.event_group` is non-empty and that name exists under `groups` in `rules.yaml`:
  - **File group** (`event_group == "file"`): `file_*` flags and path-prefix checks.
  - **Network group** (`event_group == "network"`): `net_*` features derived from socket attributes/args.
  - **Process group** (`event_group == "process"`): currently no extra features are emitted.

**Group-scoped features** are added when `evt.event_group` is non-empty and that name exists under `groups` in `rules.yaml`. The server uses one model per `event_group`; each model sees a stable key set for its group.

**Event format (protobuf / JSON):**

- `event_id` (string): correlation id for the message (not the Linux syscall number).
- `syscall_name`: Linux syscall name string (e.g. `openat`, `connect`); used for flag-slot routing, network/socket parsing, and group-local one-hot labels.
- `syscall_nr` (uint32): Linux syscall number (e.g. `257` for `openat`); currently not used by the feature extractor.
- `hostname` (string): host identity for `**hostname_hash`** in the `**frequency`** view.
- `comm`, `pid`, `tid`, `uid` (strings): process context; `pid`/`tid`/`uid` use decimal string encoding as emitted by the probe.
- `arg0`, `arg1` (strings): syscall arguments; also used as raw flag numeric strings for `open` → `arg0`, `openat`/`openat2`/`socket` → `arg1` when deriving `file_flags_*` / `file_flags_hash` (file groups) or `proc_flags_*` / `proc_flags_hash` (process group). `**net_***` numeric norms use these args; endpoint IP/port/family for `**net_***` hashes come from `**fd_sock_***` attributes (`_parse_sockaddr_from_evt`), not from parsing `arg1`.
- Path strings for feature extraction: `**attributes["fd_path"]**` (the legacy `path` field is reserved in `events.proto`).
- `ts_unix_nano` (uint64): event time (UTC) for calendar time features.

`evt.event_group` stores the rule-defined **event_group** (e.g. `file`, `network`, `process`, or empty) for routing to the per-group model and for type-specific features.

JSON/EVT1 records use the same names; see `event_envelope_codec.envelope_to_dict` / `envelope_from_dict`.

Optional `attributes` (map, producer-filled when available):

- `return_value`: syscall return (non-negative = success, negative = errno-style); drives `return_success` / `return_errno_norm` (missing → treated as `0`).
- `flags`: decoded flags string for open family / socket; preferred over raw flag slot for `**file_flags_*`** / `**file_flags_hash`** (or `proc_flags_*` / `proc_flags_hash`) when present.
- `cluster`, `node`: optional deployment tags (probe env); not consumed by the current feature primitives today.
- Network enrichment (preferred for `**net_***`): `fd_sock_remote_port`, `fd_sock_remote_addr`, `fd_sock_family` — read in `_parse_sockaddr_from_evt` (legacy aliases `sin_port` / `dest_port`, etc. are not used by the current extractor).

**Rules file (`rules.yaml`):** Loaded from `DETECTOR_RULES_PATH`, else `/etc/sentinel-ebpf/rules.yaml` if present, else repo `charts/sentinel-ebpf/rules.yaml`. Supplies optional `**groups.<name>.features`** (e.g. `**sensitive_paths`** / `**tmp_paths**`) used by the file layer (`file_sensitive_path` / `file_tmp_path`). Which events are captured is decided by probe **rules** (`condition`), not by the group list alone—see `docs/RULES_GUIDE.md`.

**Feature design:** Based on work on syscall-argument anomaly detection (e.g. Krügel/Mutz, “On the Detection of Anomalous System Call Arguments”, ESORICS 2003): identity and numeric/semantic features for call type, process, path, arguments, return value, and context (host, time).

**Public API:**

- `extract_feature_dict(evt, feature_view="default") -> Dict[str, float]`: extracts features for a single event, and appends group-specific features only when `evt.event_group` is present in the loaded rules config. Key set can differ per **event_group** and per **feature view**.
- `feature_view_for_algorithm(algorithm) -> str`: maps algorithm to feature view. `freq1d`, `copulatree`, `latentcluster`, `zscore` → `frequency`; `sequence_mlp` / `sequence_transformer` → `sequence`; else → `default`.

**Feature views:** One extractor; the view toggles which optional blocks are included (`_FeatureViewSpec` in `detector/building_blocks/primitives/features/views.py`).

- `**default`**: minimal generic context + time features. Does **not** include process id/uid norms, path depth/prefix features, return code features, string hashes, network port norms, or file flag hashes.
- `**frequency`**: hash-heavy view for frequency-style models. Emits `*_hash` general hashes plus network hashes, and uses `day_fraction_norm` instead of the sin/cos pair. Omits most other generic context.
- `**full`**: emits all optional generic blocks (process norms, time bucket, path depth/prefix hash, return code features) and includes file/network norms where applicable.
- `**sequence**`: separate path: produces only sequence-context features under the `sequence_ctx_*` prefix (no generic/group blocks).

Empty `**event_group**`: no group-specific block.

**Views** column: only the listed feature views include that row (primary named views: **default**, **frequency**, **full**).


| Feature / pattern                                                         | Views         | Encoded as         | Range   | Source                                                                                   |
| ------------------------------------------------------------------------- | ------------- | ------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `day_cycle_sin`, `day_cycle_cos`                                          | default, full | cyclic sin/cos     | [-1, 1] | UTC time → sin/cos of 2π × (nanoseconds mod day) / day length                            |
| `day_fraction_norm`                                                       | frequency     | scalar             | [0, 1]  | Same clock as day-cycle, but one scalar position in the UTC day                          |
| `week_of_month_norm`                                                      | full          | normalized integer | [0, 1]  | `evt.ts_unix_nano` → week-of-month bucket                                                |
| `pid_norm`, `tid_norm`                                                    | full          | normalized integer | [0, 1]  | `evt.pid` / `evt.tid` → int, scaled by `_PID_MAX`                                        |
| `uid_norm`                                                                | full          | normalized integer | [0, 1]  | `evt.uid` → int, scaled by `_UID_MAX`                                                    |
| `arg0_norm`, `arg1_norm`                                                  | full          | log1p-magnitude    | [0, 1]  | `evt.arg0` / `evt.arg1` → int, `_norm_arg`                                               |
| `path_depth_norm`                                                         | full          | normalized integer | [0, 1]  | Slash-split `attributes["fd_path"]` → component count                                    |
| `path_prefix_hash`                                                        | full          | scalar hash        | [0, 1)  | `_hash01(first path component)` from `fd_path`                                           |
| `return_success`                                                          | full          | binary             | {0, 1}  | `int(attributes["return_value"], default 0) >= 0`                                        |
| `return_errno_norm`                                                       | full          | log1p-magnitude    | [0, 1]  | `log1p(abs(int(attributes["return_value"], default 0)))` scaled by `_RETURN_ERRNO_SCALE` |
| `comm_hash`, `hostname_hash`, `pid_hash`, `path_hash`, `path_prefix_hash` | frequency     | scalar hash        | [0, 1)  | `_hash01` of respective string fields                                                    |


**Group-scoped features** — appended when `evt.event_group` is non-empty and matches a `**groups.<name>`** entry in `rules.yaml`. Sensitive/tmp prefix lists come from `**groups.<name>.features`** (fallbacks: `_DEFAULT_SENSITIVE_PATH_PREFIXES`, `_DEFAULT_TMP_PATH_PREFIXES`). Exactly one of the **File** / **Process** / **Network** layers applies: **File** for `event_group=file`; **Process** for `event_group=process` (currently emits no extra features); **Network** for `event_group=network`. `**net_`*** use `attributes` via `_parse_sockaddr_from_evt` (`fd_sock_`* keys); values are 0 when fields are missing.


| Layer   | Feature / pattern                                       | Views         | Encoded as  | Range  | Source                                                                  |
| ------- | ------------------------------------------------------- | ------------- | ----------- | ------ | ----------------------------------------------------------------------- |
| File    | `file_sensitive_path`, `file_tmp_path`                  | default, full | binary      | {0, 1} | Lowercased `attributes["fd_path"]` vs configured sensitive/tmp prefixes |
| File    | `file_flags_hash`                                       | full          | scalar hash | [0, 1) | `_hash01(attributes["flags"])` (decoded flags string when present)      |
| Network | `net_socket_family_norm`                                | full          | numeric     | [0, 1] | socket family from `socket()` args or `fd_sock_family` attribute        |
| Network | `net_dport_norm`                                        | full          | numeric     | [0, 1] | remote port from `attributes["fd_sock_remote_port"]`                    |
| Network | `net_socket_type_hash`, `net_daddr_hash`, `net_af_hash` | frequency     | scalar hash | [0, 1) | `_hash01` of socket type value, remote addr, AF string                  |


Missing or invalid fields use safe defaults. Each per-event_group model sees a consistent feature set for its group, and the **feature view** (chosen per algorithm) is fixed for that model instance.

---

## config.py

**Role:** Central configuration for the detector: ports, model choice, score space, and all algorithm hyperparameters. Used by `server.py` and (for eval) by scripts that start the detector with overrides.

**Public API:**

- `DetectorConfig`: dataclass of default values (port, events_http_port, recent_events_buffer_size, model_algorithm, threshold, score_mode, and all HST/LODA/KitNet/MemStream/ZScore/KNN/Freq1D parameters, plus **sequence** fields: `sequence_ngram_length`, `sequence_thread_aware`, `sequence_mlp_hidden_size`, `sequence_mlp_hidden_layers`, `sequence_mlp_lr`, `DETECTOR_SEQUENCE_`* envs). The feature view is derived from `model_algorithm` via `feature_view_for_algorithm`.
- `load_config() -> DetectorConfig`: builds a `DetectorConfig` from the environment. Environment variables override defaults (e.g. `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, `DETECTOR_SCORE_MODE`, `DETECTOR_KITNET_*`, `DETECTOR_MEMSTREAM_*`, `DETECTOR_SEQUENCE_*`, etc.).

No config file is read; configuration is env-only so it works well in containers and eval runs.

---

## Building Blocks Layout

The detector runtime is organized around explicit building blocks.

- `detector/building_blocks/core/`: runtime primitives such as `BuildingBlock`, `BlockContext`, `BuildingBlockManager`, checkpoint loading/saving, and `OnlineIDS`
- `detector/building_blocks/blocks/`: graph nodes such as feature extraction, model blocks, fusion, scaling, sequence blocks, and scoring/decision blocks
- `detector/building_blocks/primitives/`: low-level reusable implementations for models, features, embeddings, sequence handling, and scoring helpers
- `detector/building_blocks/pipelines/`: named pipeline builders selected by `cfg.pipeline_id`

Built-in pipeline ids are registered in `detector/pipelines/registry.py`. `build_final_bb(cfg)` returns the final graph node for the configured pipeline, and `OnlineIDS.run_event(evt)` executes that graph and returns the final payload, which is currently expected to be a `DecisionOutput` for detector-facing pipelines.

---

## Model Primitives

**Role:** Online anomaly detection implementations that consume feature dicts, keep state internally, and produce anomaly scores. They now live under `detector/building_blocks/primitives/models/`, and are instantiated through `detector/building_blocks/primitives/models/factory.py`.

**Algorithms:**

1. **Half-Space Trees (River)** – Tree ensemble over the feature vector; CPU-only, no PyTorch. Good baseline, low dependency.
2. **LODA-EMA** – Custom implementation with EMA-based adaptive normalization; streaming from first event; PyTorch, supports CPU/CUDA. Recommended LODA-style baseline.
3. **KitNet (PySAD)** – Ensemble of small autoencoders; learns a mapping then an anomaly detector in two grace phases. CPU-backed wrapper.
4. **MemStream** – Paper-aligned (WWW'22): single-layer encoder/decoder with latent = 2×input_dim, Tanh, K-NN discounted L1 scoring, FIFO memory when score ≤ β. PyTorch, CPU/CUDA.
5. **ZScore** – Per-feature online running mean/std with event score `mean(abs(z_i))`. Intentionally simple baseline; CPU-only.
6. **KNN (scikit-learn)** – Sliding-memory nearest-neighbor detector; score is mean distance to k nearest historical events. CPU-only.
7. **Freq1D** – Per-feature 1D frequency baseline: numeric features use fixed-bin histograms, categorical/hash features use capped count tables. Scores by configurable aggregation over excess surprisal (sum / mean / top-k mean / soft top-k mean). CPU-only.
8. **CopulaTree** – Streaming copula-tree detector on top of `freq1d`: marginals come from `freq1d`, pairwise dependence is tracked online in Gaussianized space, and a maximum-spanning tree is refreshed periodically. CPU-only.
9. **LatentCluster** – Online latent clustering on top of `freq1d` marginals: events are mapped to CDF/probit coordinates, scored against a small bank of latent clusters with diagonal variance, and only likely-normal points update clusters. CPU-only.

`**sequence_mlp`** is no longer a special top-level model module. It is assembled from explicit building blocks and primitives:

- feature extraction block
- `Word2VecEmbeddingBlock`
- `SequenceNgramContextBlock`
- `SequenceNextTokenMLPBlock`
- scoring / threshold blocks

The underlying sequence and embedding primitives live under:

- `detector/building_blocks/primitives/embeddings/`
- `detector/building_blocks/primitives/sequence/`

Device selection is via config (`model_device`: `auto` / `cpu` / `cuda`).

**Deferred idea for later:** a **learned mixture-of-regimes** model on top of `freq1d`-normalized inputs. The intended design is: keep `freq1d` for marginals, transform each event to CDF/probit coordinates, use a small gating network to softly assign the event to one of several latent regimes, and score the event by how well at least one regime explains it. Unlike hard routing by `comm`/event_group, the partition is learned from data and can differ between systems. Compared with `latentcluster`, the mixture keeps *soft* assignments and regime-specific density models instead of just nearest-cluster geometry.

---

## server.py

**Role:** gRPC + optional HTTP server that receives events, runs the configured pipeline graph, and streams back detection responses. Also maintains in-memory buffers and optional file logging for anomalies.

**Flow:**

1. **Startup:** `load_config()`, create `RECENT_EVENTS` deque (size from config), optionally enable anomaly log file via `ANOMALY_LOG_PATH`, instantiate `RuleBasedDetector(cfg)` / `DeterministicScorer`.
2. **gRPC:** Implements `DetectorService`: `StreamEvents(stream EventEnvelope) -> stream DetectionResponse`. For each event: pick/create the per-`event_group` `OnlineIDS`, run the final building-block graph for the event, and consume its final `DecisionOutput` to build `DetectionResponse` (event_id, anomaly, reason, score, ts). Responses are yielded back to the client. Anomalies and recent events are pushed to in-memory buffers; if `ANOMALY_LOG_PATH` is set, anomalies are also appended as JSONL.
3. **DeterministicScorer:** Holds the config and one pipeline runtime per `event_group`. Warmup suppression, score-mode selection, calibration, scaling, and thresholding are part of the explicit scoring building blocks in the graph, not separate legacy scoring logic outside the pipeline.
4. **HTTP (optional):** If `events_http_port` > 0, a small HTTP server serves:
  - `GET /recent_events?limit=N` – last N events (for UI log tail)
  - `GET /metrics` – Prometheus-style metrics (events_total, anomalies_total, etc.)
  - `GET /anomalies?limit=N` – last N anomalies

**Concurrency:** One gRPC stream is processed sequentially per connection; the lock in `DeterministicScorer.score_event` ensures a single model update per event. HTTP is served from a background thread.

**Entrypoint:** `python -m detector.server` (or Docker `ENTRYPOINT`). gRPC port from `DETECTOR_PORT`, HTTP from `DETECTOR_EVENTS_PORT` (config).

**Do we need a general layer or autoencoder to combine multiple models?** The current architecture can already express that if we want it. Built-in pipelines still use one final decision path per event, but the building-block graph supports composing multiple upstream score producers and combining them via explicit fusion/model blocks before thresholding. The existing `las_gas_fusion` pipeline is the reference example of that style.

---

## **init**.py

Package marker only; no public exports.

---

## Dockerfile

**Role:** Build a minimal image to run the detector as a service.

- Base: `python:3.11-slim`.
- Installs `uv`, then copies project deps and detector (and probe/scripts and generated gRPC stubs as needed), runs `uv sync --no-dev --frozen --extra detector`.
- `ENTRYPOINT`: `python -m detector.server`.

Configuration is via environment variables (see `config.py`). For eval or custom ports, set `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, etc., when running the container.