# Detector

The detector is an anomaly detection service that consumes kernel-level events (e.g. syscalls) over gRPC, extracts features, scores them with an online ML model, and returns anomaly decisions. It can be run standalone (`python -m detector.server`) or in Docker.

## Event model

**event_name = syscall name; event_group = category.** There is no separate “network” vs “syscall” category: `event_name` is the syscall/event name (e.g. `openat`, `connect`, `socket`, `execve`); the registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`). The rule-defined category is **event_group** (e.g. `network`, `file`, `process`) and can be empty; it is carried in the protobuf field `EventEnvelope.event_group`.

**One envelope shape for all.** Syscalls are like function calls with defined inputs and outputs. The `EventEnvelope` is the single contract: every producer (eBPF probe, EVT1 replay, future sources) must send the same canonical `data` layout and the same attribute keys. For a given event only the relevant slots are non-empty (e.g. `path` for openat, empty for connect; `flags` for open*, empty for others). That way the detector can support different feature extraction per event_group later (e.g. path semantics for file, addr/port for network) while the envelope always carries everything any syscall can contain.

---

## features.py

**Role:** Turn each `EventEnvelope` into numeric features (a dict of floats) for the anomaly models. **General features** are always added; **type-specific features** are added when the event has an **event_group** (stored in `evt.event_group`, e.g. `file`, `network`, `process`). The resulting dict can have different features (and thus different “size”) per event; the server uses one model per event_group so each model sees the same feature set for its group.

**Event format:** Events follow the canonical `data` vector order (strings):

`[event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]`

- `event_name` (`data[0]`): syscall name (e.g. `openat`, `connect`, `socket`).
- `event_id` (`data[1]`): numeric syscall ID as string (e.g. `"257"`, `"42"`). Used mainly by legacy (`hash`) mode.
- `comm` (`data[2]`): process name.
- `pid`, `tid`, `uid` (`data[3:6]`): process/thread/user IDs as strings.
- `arg0`, `arg1` (`data[6]`, `data[7]`): syscall-specific arguments as strings (e.g. fd, family/type, addrlen, mode).
- `path` (`data[8]`): path for path-based calls (open, exec, unlink, etc.). Empty string for non-path events.
- `flags` (`data[9]`): legacy flags slot (often open flags), may be empty.

`evt.event_group` is **not** part of `data`: it stores the rule-defined **event_group** (e.g. `file`, `network`, `process`, or empty) used for routing to the per-group model and for selecting type-specific features.

Optional `attributes` (preferred for structured fields when present):

- `return_value`: syscall return value (>=0 success, <0 errno-style failure)
- `mount_namespace`: mount namespace ID (container/isolation context)
- `flags`: normalized flags string for open/openat/openat2 (preferred over `data[9]` when available)
- network fields (when available): `sin_port`/`dest_port`, `sin_addr`/`dest_ip`, `sa_family`

**Feature design:** Based on work on syscall-argument anomaly detection (e.g. Krügel/Mutz, “On the Detection of Anomalous System Call Arguments”, ESORICS 2003): identity and numeric/semantic features for call type, process, path, arguments, return value, and context (host, namespace, time).

**Public API:**

- `extract_feature_dict(evt, feature_view="default") -> Dict[str, float]`: general features always; if the event has an **event_group** (in `evt.event_group`) of `file`, `network`, or `process`, appends type-specific features. Key set (and “vector size”) can differ per event_group and per **feature view**.
- `feature_view_for_algorithm(algorithm) -> str`: maps algorithm to feature view. `freq1d`, `gausscop`, `copulatree`, `latentcluster` → hash (scalar hashes); `loda` / `loda_ema` → loda; `memstream` → memstream; else → default.

**Feature views:** A single extraction path produces features; the view controls which categorical encodings are used (one-hot/bucket vs scalar hash). `default` = one-hot + bucket banks (AE-family); `hash` = scalar hashes (`*_hash`) + `event_id_norm` (freq1d only); `loda` = compact identity blocks; `memstream` = minimal identity blocks. The view is chosen per algorithm via `feature_view_for_algorithm(algorithm)`.

**General features:** Always present. Base features come from `evt` (data list, `ts_unix_nano`, attributes, hostname). Shared online stats (proc/host rate, proc interarrival) use five decay windows (`01`=0.1s, `1`, `5`, `30`, `120`s).

*Present when:* `all` = every view; `default` / `loda` / `memstream` / `hash` = that view only.

| Feature / pattern | Present when | Encoded as | Range | Represents / source |
|---|---|---|---|---|
| `pid_norm` | all | normalized integer | [0, 1] | PID scaled by Linux PID max |
| `tid_norm` | all | normalized integer | [0, 1] | TID scaled by Linux PID max |
| `uid_norm` | all | normalized integer | [0, 1] | UID scaled by \(2^{32}-1\) |
| `arg0_norm`, `arg1_norm` | all | log1p-magnitude | [0, 1] | Event argument magnitudes (`data[6]`, `data[7]`) |
| `hour_sin`, `hour_cos` | all | cyclic sin/cos | [-1, 1] | Hour-of-day from timestamp |
| `minute_sin`, `minute_cos` | all | cyclic sin/cos | [-1, 1] | Minute-of-hour from timestamp |
| `weekday_sin`, `weekday_cos` | all | cyclic sin/cos | [-1, 1] | Day-of-week from timestamp |
| `week_of_month_norm` | all | normalized integer | [0, 1] | Month quarter (1–4) from timestamp |
| `path_depth_norm` | all | normalized integer | [0, 1] | Path component count derived from `data[8]` |
| `return_success` | all | binary | {0, 1} | 1 if `attributes["return_value"] >= 0` |
| `return_errno_norm` | all | log1p-magnitude | [0, 1] | Abs errno from `attributes["return_value"]` |
| `event_name_*` | default, loda, memstream | one-hot | {0, 1} | Event identity; values come from event lists in `rules.yaml` |
| `comm_bucket_*` | default | hashed sparse bank | {0, 1} | Process name (`data[2]`) mapped to fixed bucket bank |
| `hostname_bucket_*` | default | hashed sparse bank | {0, 1} | Hostname (`evt.hostname`) mapped to fixed bucket bank |
| `mount_ns_bucket_*` | default | hashed sparse bank | {0, 1} | Mount namespace (`attributes["mount_namespace"]`) mapped to fixed bucket bank |
| `path_tok_d{0..3}_bucket_*` | default | depth-positioned hashed multi-hot banks | {0, 1} | Tokenized path (`data[8]`) mapped to fixed bucket banks per depth |
| `event_hash` | hash | scalar hash | [0, 1) | Hash of `event_name` |
| `comm_hash` | hash | scalar hash | [0, 1) | Hash of `comm` |
| `path_hash` | hash | scalar hash | [0, 1) | Hash of full path string |
| `flags_hash` | hash | scalar hash | [0, 1) | Hash of `flags` (`data[9]`) |
| `path_prefix_hash` | hash | scalar hash | [0, 1) | Hash of first path component (e.g. `etc`) |
| `mount_ns_hash` | hash | scalar hash | [0, 1) | Hash of mount namespace |
| `hostname_hash` | hash | scalar hash | [0, 1) | Hash of hostname |
| `event_id_norm` | hash | normalized integer | [0, 1] | Numeric syscall ID (`data[1]`) normalized |
| `proc_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Process rate (per `comm`), 5 decay windows |
| `host_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Host rate, 5 decay windows |
| `proc_interarrival_{wn}` | all | decayed mean | [0, 1] | Inter-arrival mean, 5 decay windows |
| `proc_interarrival_std_{wn}` | all | decayed std | [0, 1] | Inter-arrival std, 5 decay windows |

**Type-specific features:** Added only when `evt.event_group` matches.

Feature coverage is intentionally scoped to the currently supported/traced syscall families behind `event_group = file|network|process`; we do not claim uniform syscall-specific semantics across the full Linux syscall surface.

### File (`event_group == "file"`)

`file_sensitive_path` and `file_tmp_path` prefixes are configured via `groups.file.features.sensitive_paths` and `groups.file.features.tmp_paths` in `rules.yaml`.

| Feature / pattern | Present when | Encoded as | Range | Represents / source |
|---|---|---|---|---|
| `file_sensitive_path` | all | binary | {0, 1} | Path under any prefix in `groups.file.features.sensitive_paths` |
| `file_tmp_path` | all | binary | {0, 1} | Path under any prefix in `groups.file.features.tmp_paths` |
| `file_arg0_norm`, `file_arg1_norm` | all | log1p-magnitude | [0, 1] | Event-specific numeric args (`data[6]`, `data[7]`) |
| `file_event_name_*` | default | one-hot | {0, 1} | File syscall identity; values from `rules[].syscalls` where `event_group = file` |
| `file_extension_bucket_*` | default | hashed sparse bank | {0, 1} | File extension bucket bank (derived from `data[8]`) |
| `file_flags_bucket_*` | default | hashed multi-hot bank | {0, 1} | Tokenized file flags string from `attributes["flags"]` or `data[9]`; syscall-agnostic within the file extractor |
| `file_extension_hash` | hash | scalar hash | [0, 1) | Hash of file extension |
| `file_event_name_hash` | hash | scalar hash | [0, 1) | Hash of syscall name |
| `file_flags_hash` | hash | scalar hash | [0, 1) | Hash of file flags string from `attributes["flags"]` or `data[9]` |
| `file_path_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Per-path event rate |
| `file_pair_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Per (comm, path) event rate |
| `file_pair_interarrival_{wn}` | all | decayed mean | [0, 1] | Inter-arrival mean for (comm, path) |
| `file_pair_interarrival_std_{wn}` | all | decayed std | [0, 1] | Inter-arrival std for (comm, path) |
| `file_proc_path_depth_mean_{wn}`, `file_proc_path_depth_std_{wn}` | all | decayed mean/std | [0, 1] | Path-depth stats per process |
| `file_host_path_depth_mean_{wn}`, `file_host_path_depth_std_{wn}` | all | decayed mean/std | [0, 1] | Path-depth stats per host |
| `file_pair_path_depth_mean_{wn}`, `file_pair_path_depth_std_{wn}` | all | decayed mean/std | [0, 1] | Path-depth stats per (comm, path) |

### Network (`event_group == "network"`)

| Feature / pattern | Present when | Encoded as | Range | Represents / source |
|---|---|---|---|---|
| `net_fd_norm` | all | log1p-magnitude | [0, 1] | FD or socket arg0 (depends on syscall) |
| `net_addrlen_norm` | all | log1p-magnitude | [0, 1] | Addrlen (connect arg1) |
| `net_socket_family_norm` | all | normalized integer | [0, 1] | Address family (AF_INET/AF_INET6) |
| `net_dport_norm` | all | normalized integer | [0, 1] | Destination port (from attributes / parsed sockaddr) |
| `net_socket_type_bucket_*` | default, loda | hashed sparse bank | {0, 1} | Socket type bucket bank (socket arg1) |
| `net_daddr_bucket_*` | default | hashed sparse bank | {0, 1} | Destination IP bucket bank |
| `net_af_*` | default, loda, memstream | one-hot | {0, 1} | Address family label (`af_inet`, `af_inet6`, `af_unix`, `af_netlink`, `af_other`) |
| `net_socket_type_hash` | hash | scalar hash | [0, 1) | Hash of socket type |
| `net_daddr_hash` | hash | scalar hash | [0, 1) | Hash of destination IP |
| `net_af_hash` | hash | scalar hash | [0, 1) | Hash of address family |
| `net_pair_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Per (comm, daddr, dport) rate |
| `net_daddr_rate_{wn}` | all | decayed log-scaled rate | [0, 1] | Per-destination IP rate |
| `net_pair_interarrival_{wn}`, `net_pair_interarrival_std_{wn}` | all | decayed mean/std | [0, 1] | Inter-arrival mean/std for pair |
| `net_proc_dport_mean_{wn}`, `net_proc_dport_std_{wn}` | all | decayed mean/std | [0, 1] | Port stats per process |
| `net_proc_addrlen_mean_{wn}`, `net_proc_addrlen_std_{wn}` | all | decayed mean/std | [0, 1] | Addrlen stats per process |
| `net_host_dport_mean_{wn}`, `net_host_dport_std_{wn}` | all | decayed mean/std | [0, 1] | Port stats per host |
| `net_host_addrlen_mean_{wn}`, `net_host_addrlen_std_{wn}` | all | decayed mean/std | [0, 1] | Addrlen stats per host |
| `net_daddr_dport_mean_{wn}`, `net_daddr_dport_std_{wn}` | all | decayed mean/std | [0, 1] | Port stats per destination IP |
| `net_proc_daddr_dport_mean_{wn}`, `net_proc_daddr_dport_std_{wn}` | all | decayed mean/std | [0, 1] | Port stats per (process, destination IP) |


**Comparison to Kitsune (NDSS 2018):** Kitsune uses packet-level streams with 5 decay windows and ~115 features. We work at syscall level (connect/socket), so we have no packet size or MAC. We mirror the idea with **five decay windows** (0.1s, 1s, 5s, 30s, 120s), **rate + interarrival mean/std + value mean/std** per stream. Shared online stats (proc/host rate, proc interarrival) apply to all events; type-specific stats add pair, daddr, path, and value distributions per type.

**Still doable with the same syscall-level data (not yet implemented):** (1) **2D / correlation**: covariance between two values (e.g. dport vs addrlen) per stream. (2) **Rarity / novelty**: binary or decayed "first time (comm, daddr)" in window. (3) **Per-protocol rate**: rate of TCP vs UDP per window. (4) **Longer windows**: e.g. 300s, 600s. (5) **Daddr addrlen**: mean/std of addrlen per destination IP.

**Process** (`event_group == "process"`):


| Feature             | Source                 | Description                            | Discrete | Range  |
| ------------------- | ---------------------- | -------------------------------------- | -------- | ------ |
| `process_is_execve` | `data[0]` (event_name) | 1.0 if event_name is execve; else 0.0. | yes      | {0, 1} |
| `process_is_fork`   | `data[0]` (event_name) | 1.0 if event_name is fork; else 0.0.   | yes      | {0, 1} |


Missing or invalid fields use safe defaults. Each per-event_group model sees a consistent feature set for its group, and the **feature view** (chosen per algorithm) is fixed for that model instance.

---

## config.py

**Role:** Central configuration for the detector: ports, model choice, score space, and all algorithm hyperparameters. Used by `server.py` and (for eval) by scripts that start the detector with overrides.

**Public API:**

- `DetectorConfig`: dataclass of default values (port, events_http_port, recent_events_buffer_size, model_algorithm, threshold, score_mode, and all HST/LODA/KitNet/MemStream/ZScore/KNN/Freq1D parameters). The feature view is derived from `model_algorithm` via `feature_view_for_algorithm`.
- `load_config() -> DetectorConfig`: builds a `DetectorConfig` from the environment. Environment variables override defaults (e.g. `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, `DETECTOR_SCORE_MODE`, `DETECTOR_KITNET_*`, `DETECTOR_MEMSTREAM_*`, `DETECTOR_ZSCORE_*`, `DETECTOR_KNN_*`, `DETECTOR_FREQ1D_*`, etc.).

No config file is read; configuration is env-only so it works well in containers and eval runs.

---

## model.py

**Role:** Online anomaly detection models that consume the feature dict from `features.py`, update their internal state (learn), and return an anomaly score per event. All implement the same interface: `score_and_learn(features: Dict[str, float]) -> float`.

**Public API:** `OnlineAnomalyDetector` is a factory. You pass `algorithm` (`"halfspacetrees"`, `"loda"`, `"loda_ema"`, `"kitnet"`, `"memstream"`, `"zscore"`, `"knn"`, `"freq1d"`, `"gausscop"`, `"copulatree"`, `"latentcluster"`) plus optional hyperparameters; it instantiates the corresponding implementation and exposes `score_and_learn(features)`.

**Algorithms:**

1. **Half-Space Trees (River)** – Tree ensemble over the feature vector; CPU-only, no PyTorch. Good baseline, low dependency.
2. **LODA (PySAD)** – PySAD LODA wrapper. **Broken**: fit_partial overwrites histograms instead of accumulating; scores are often ~0. Not recommended.
3. **LODA-EMA** – Custom implementation with EMA-based adaptive normalization; streaming from first event; PyTorch, supports CPU/CUDA. **Recommended for LODA.**
4. **KitNet (PySAD)** – Ensemble of small autoencoders; learns a mapping then an anomaly detector in two grace phases. PyTorch, CPU/CUDA.
5. **MemStream** – Paper-aligned (WWW'22): single-layer encoder/decoder with latent = 2×input_dim, Tanh, K-NN discounted L1 scoring, FIFO memory when score ≤ β. PyTorch, CPU/CUDA.
6. **ZScore** – Per-feature online running mean/std with event score `mean(abs(z_i))`. Intentionally simple baseline; CPU-only.
7. **KNN (scikit-learn)** – Sliding-memory nearest-neighbor detector; score is mean distance to k nearest historical events. CPU-only.
8. **Freq1D** – Per-feature 1D frequency baseline: numeric features use fixed-bin histograms, categorical/hash features use capped count tables. Scores by configurable aggregation over excess surprisal (sum / mean / top-k mean / soft top-k mean). CPU-only.
9. **GaussCop** – Gaussian-copula extension of `freq1d`: marginals come from `freq1d`, then dependence is modeled in Gaussianized space. CPU-only and more expensive than `freq1d`.
10. **CopulaTree** – Streaming copula-tree detector on top of `freq1d`: marginals come from `freq1d`, pairwise dependence is tracked online in Gaussianized space, and a maximum-spanning tree is refreshed periodically. This keeps the sparse-tree idea from the paper without requiring offline family fitting or Monte Carlo calibration. CPU-only.
11. **LatentCluster** – Online latent clustering on top of `freq1d` marginals: events are mapped to CDF/probit coordinates, scored against a small bank of latent clusters with diagonal variance, and only likely-normal points update clusters. CPU-only.

Device selection is via config (`model_device`: `auto` / `cpu` / `cuda`). The server uses one model per event_group and calls `score_and_learn` under a lock so only one thread updates any model per event.

**Deferred idea for later:** a **learned mixture-of-regimes** model on top of `freq1d`-normalized inputs. The intended design is: keep `freq1d` for marginals, transform each event to CDF/probit coordinates, use a small gating network to softly assign the event to one of several latent regimes, and score the event by how well at least one regime explains it. Unlike hard routing by `comm`/event_group, the partition is learned from data and can differ between systems. Compared with `latentcluster`, the mixture keeps *soft* assignments and regime-specific density models instead of just nearest-cluster geometry.

---

## server.py

**Role:** gRPC + optional HTTP server that receives events, runs the feature extractor and model, and streams back detection responses. Also maintains in-memory buffers and optional file logging for anomalies.

**Flow:**

1. **Startup:** `load_config()`, create `RECENT_EVENTS` deque (size from config), optionally enable anomaly log file via `ANOMALY_LOG_PATH`, instantiate `RuleBasedDetector(cfg)` (which wraps `DeterministicScorer` and one `OnlineAnomalyDetector` **per event_group**).
2. **gRPC:** Implements `DetectorService`: `StreamEvents(stream EventEnvelope) -> stream DetectionResponse`. For each event: `extract_feature_dict(evt, feature_view=feature_view_for_algorithm(cfg.model_algorithm))` → pick/create the model for `evt.event_group` → `score_and_learn(features)` → build `DetectionResponse` (event_id, anomaly, reason, score, ts). Responses are yielded back to the client. Anomalies and recent events are pushed to in-memory buffers; if `ANOMALY_LOG_PATH` is set, anomalies are also appended as JSONL.
3. **DeterministicScorer:** Holds the config and the `OnlineAnomalyDetector`. `score_event(evt)` calls `extract_feature_dict(evt, feature_view=feature_view_for_algorithm(cfg.model_algorithm))` then `score_and_learn(features)` which returns `(raw, scaled)`, compares the selected score (based on `cfg.score_mode`) to `cfg.threshold`, and returns a `DetectionResponse` (anomaly = score ≥ threshold). Exceptions are caught and returned as a non-anomaly response with an error reason.
4. **HTTP (optional):** If `events_http_port` > 0, a small HTTP server serves:
  - `GET /recent_events?limit=N` – last N events (for UI log tail)
  - `GET /metrics` – Prometheus-style metrics (events_total, anomalies_total, etc.)
  - `GET /anomalies?limit=N` – last N anomalies

**Concurrency:** One gRPC stream is processed sequentially per connection; the lock in `DeterministicScorer.score_event` ensures a single model update per event. HTTP is served from a background thread.

**Entrypoint:** `python -m detector.server` (or Docker `ENTRYPOINT`). gRPC port from `DETECTOR_PORT`, HTTP from `DETECTOR_EVENTS_PORT` (config).

**Do we need a general layer or autoencoder to combine multiple models?** No, for the current design. We have **one model per event_group**; each event is routed to exactly one model and gets **one score**. There are no “multiple online extractors” whose outputs we must fuse per event. A fusion layer or meta-model would only be needed if we changed the architecture, for example: (1) **Ensemble**: run the same event through several models (e.g. a general + a group-specific) and combine scores (e.g. max, mean, or a small learned combiner). (2) **Score calibration**: normalize scores across groups so a single threshold behaves similarly for network vs file vs default (e.g. per-group running mean/std or a tiny calibration model). (3) **Single shared model**: one autoencoder or detector that sees all events in a common representation (e.g. per-group encoders mapping to a fixed latent size, then one decoder). That would require a common embedding size and more complexity. The current per-group design keeps groups independent and avoids mixing score distributions; we keep it as-is unless we explicitly add ensemble or calibration.

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