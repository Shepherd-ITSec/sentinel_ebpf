# Detector

The detector is an anomaly detection service that consumes kernel-level events (e.g. syscalls) over gRPC, extracts features, scores them with an online ML model, and returns anomaly decisions. It can be run standalone (`python -m detector.server`) or in Docker.

## Event model

**event_name = syscall name; event_type = category.** There is no separate “network” vs “syscall” category: `event_name` is the syscall/event name (e.g. `openat`, `connect`, `socket`, `execve`); the registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`). `event_type` is the rule-defined category (e.g. `network`, `file`, `process`) and can be empty.

**One envelope shape for all.** Syscalls are like function calls with defined inputs and outputs. The `EventEnvelope` is the single contract: every producer (eBPF probe, BETH converter, future sources) must send the same canonical `data` layout and the same attribute keys. For a given event only the relevant slots are non-empty (e.g. `path` for openat, empty for connect; `flags` for open*, empty for others). That way the detector can support different feature extraction per event_type (category) later (e.g. path semantics for openat, addr/port for connect) while the envelope always carries everything any syscall can contain.

---

## features.py

**Role:** Turn each `EventEnvelope` into numeric features (a dict of floats) for the anomaly models. **General features** are always added; **type-specific features** are added when `evt.event_type` is set (e.g. `file`, `network`, `process`). The resulting dict can have different features (and thus different “size”) per event; the server uses one model per event_type so each model sees the same feature set for its type.

**Event format:** Events follow the canonical `data` vector order:

`[event_name, event_id, comm, pid, tid, uid, arg0, arg1, path, flags]`

- `event_name`: syscall name (e.g. `openat`, `connect`, `socket`)
- `event_type`: category from rules (e.g. `network`, `file`, or empty)
- `event_id`: numeric syscall ID (e.g. 257, 42)
- `comm`: process name
- `pid`, `tid`, `uid`: process/thread/user IDs
- `arg0`, `arg1`: event-specific arguments (e.g. fd, sockaddr, flags)
- `path`: path for path-based calls (open, exec, unlink, etc.)
- `flags`: e.g. open flags

Optional `attributes` (e.g. from BETH/converter): `return_value`, `mount_namespace`, etc.

**Feature design:** Based on work on syscall-argument anomaly detection (e.g. Krügel/Mutz, “On the Detection of Anomalous System Call Arguments”, ESORICS 2003): identity and numeric/semantic features for call type, process, path, arguments, return value, and context (host, namespace, time).

**Public API:**

- `extract_feature_dict(evt) -> Dict[str, float]`: general features always; if `evt.event_type` is `file`, `network`, or `process`, appends type-specific features. Key set (and “vector size”) can differ per event.

**General features (18 base + 20 shared online stats = 38):** Always present. Base features from `evt` (data list, ts_unix_nano, attributes, hostname). Shared online stats (proc/host rate, proc interarrival) use five decay windows (01=0.1s, 1, 5, 30, 120s).

| Feature | Source | Description | Discrete | Range |
|---------|--------|-------------|----------|-------|
| `event_hash` | `data[0]` | Hash of event name. | yes | [0, 1) |
| `comm_hash` | `data[2]` | Hash of process name. | yes | [0, 1) |
| `path_hash` | `data[8]` | Hash of path string. | yes | [0, 1) |
| `pid_norm` | `data[3]` | PID normalized. | no | [0, 1] |
| `tid_norm` | `data[4]` | TID normalized. | no | [0, 1] |
| `uid_norm` | `data[5]` | UID normalized. | no | [0, 1] |
| `arg0_norm` | `data[6]` | Log-scaled magnitude of first arg. | no | [0, 1] |
| `arg1_norm` | `data[7]` | Log-scaled magnitude of second arg. | no | [0, 1] |
| `hour_norm` | `evt.ts_unix_nano` | Hour of day. | no | [0, 1] |
| `minute_norm` | `evt.ts_unix_nano` | Minute of hour. | no | [0, 1] |
| `event_id_norm` | `data[1]` | Numeric syscall ID normalized (e.g. 42, 257). | no | [0, 1] |
| `flags_hash` | `data[9]` | Hash of flags string (e.g. open mode). | yes | [0, 1) |
| `path_depth_norm` | derived from `data[8]` (path) | Number of path components, normalized. | no | [0, 1] |
| `path_prefix_hash` | derived from `data[8]` (path) | Hash of first path component (e.g. `etc`, `tmp`). | yes | [0, 1) |
| `return_success` | `evt.attributes["return_value"]` | 1.0 if return ≥ 0, else 0.0. | yes | {0, 1} |
| `return_errno_norm` | `evt.attributes["return_value"]` | Log-scaled magnitude of errno. | no | [0, 1] |
| `mount_ns_hash` | `evt.attributes["mount_namespace"]` | Container/isolation context. | yes | [0, 1) |
| `hostname_hash` | `evt.hostname` | Hash of the hostname of the machine/node that emitted the event. | yes | [0, 1) |
| *Shared online stats (5 windows: 01, 1, 5, 30, 120s)* | | | | |
| `proc_rate_{wn}` | decayed rate | Process (comm) event rate. | no | [0, 1) |
| `host_rate_{wn}` | decayed rate | Host event rate. | no | [0, 1) |
| `proc_interarrival_{wn}` | decayed mean | Normalized inter-arrival for process. | no | [0, 1] |
| `proc_interarrival_std_{wn}` | decayed std | Inter-arrival std (burstiness) for process. | no | [0, 1] |

**Type-specific features:** Added only when `evt.event_type` matches. One table per type.

**File** (`event_type == "file"`): 7 static + online stats per window. Covers open, openat, openat2, unlink, unlinkat, rename, renameat, chmod, chown.

| Feature | Source | Description | Discrete | Range |
|---------|--------|-------------|----------|-------|
| `file_sensitive_path` | derived from `data[8]` (path) | 1.0 if path under /etc, /root, /bin, /sbin, /usr/bin, /usr/sbin; else 0.0. | yes | {0, 1} |
| `file_tmp_path` | derived from `data[8]` (path) | 1.0 if path under /tmp or /var/tmp; else 0.0. | yes | {0, 1} |
| `file_extension_hash` | derived from `data[8]` (path) | Hash of file extension (e.g. .so, .conf); empty → 0.0. | yes | [0, 1) |
| `file_event_name_hash` | `data[0]` | Hash of syscall name (open vs unlink vs chmod etc.). | yes | [0, 1) |
| `file_open_flags_hash` | `attributes["open_flags"]` or `data[9]` | Hash of open flags; only for open/openat/openat2, else 0.0. | yes | [0, 1) |
| `file_arg0_norm` | `data[6]` | Log-scaled arg0 (dfd for openat, etc.). | no | [0, 1] |
| `file_arg1_norm` | `data[7]` | Log-scaled arg1 (flags for openat, mode for chmod, etc.). | no | [0, 1] |
| *Online stats (5 windows)* | | | | |
| `file_path_rate_{wn}` | decayed rate | Per-path event rate. | no | [0, 1) |
| `file_pair_rate_{wn}` | decayed rate | Per (comm, path) pair rate. | no | [0, 1) |
| `file_pair_interarrival_{wn}`, `file_pair_interarrival_std_{wn}` | decayed | Inter-arrival mean/std for (comm, path). | no | [0, 1] |
| `file_proc_path_depth_mean_{wn}`, `file_proc_path_depth_std_{wn}` | decayed | Path depth mean/std per process. | no | [0, 1] |
| `file_host_path_depth_mean_{wn}`, `file_host_path_depth_std_{wn}` | decayed | Path depth mean/std per host. | no | [0, 1] |
| `file_pair_path_depth_mean_{wn}`, `file_pair_path_depth_std_{wn}` | decayed | Path depth mean/std per (comm, path). | no | [0, 1] |

**Network** (`event_type == "network"`): 7 static + online stats per window. Covers connect, socket.

| Feature | Source | Description | Discrete | Range |
|---------|--------|-------------|----------|-------|
| `net_addrlen_norm` | `data[7]` | Log-scaled addrlen (connect arg1). | no | [0, 1] |
| `net_fd_norm` | `data[6]` | Log-scaled fd (connect arg0). | no | [0, 1] |
| `net_socket_family_norm` | `data[6]` (socket) or parsed sockaddr | AF_INET(2)/AF_INET6(10) normalized. | no | [0, 1] |
| `net_socket_type_hash` | `data[7]` (socket) | Hash of socket type (SOCK_STREAM=1, SOCK_DGRAM=2). | yes | [0, 1) |
| `net_dport_norm` | attributes or parsed `data[7]` (BETH) | Destination port 0–65535 → 0–1. | no | [0, 1] |
| `net_daddr_hash` | attributes or parsed sockaddr | Hash of destination IP. | yes | [0, 1) |
| `net_af_hash` | attributes or parsed sockaddr | Hash of address family. | yes | [0, 1) |
| *Online stats (5 windows)* | | | | |
| `net_pair_rate_{wn}` | decayed rate | Per (comm, daddr, dport) pair rate. | no | [0, 1) |
| `net_daddr_rate_{wn}` | decayed rate | Per-destination IP rate. | no | [0, 1) |
| `net_pair_interarrival_{wn}`, `net_pair_interarrival_std_{wn}` | decayed | Inter-arrival mean/std for pair. | no | [0, 1] |
| `net_proc_dport_mean_{wn}`, `net_proc_dport_std_{wn}` | decayed | Destination port mean/std per process. | no | [0, 1] |
| `net_proc_addrlen_mean_{wn}`, `net_proc_addrlen_std_{wn}` | decayed | Addrlen mean/std per process. | no | [0, 1] |
| `net_host_dport_mean_{wn}`, `net_host_dport_std_{wn}` | decayed | Destination port mean/std per host. | no | [0, 1] |
| `net_host_addrlen_mean_{wn}`, `net_host_addrlen_std_{wn}` | decayed | Addrlen mean/std per host. | no | [0, 1] |
| `net_daddr_dport_mean_{wn}`, `net_daddr_dport_std_{wn}` | decayed | Port mean/std per destination IP. | no | [0, 1] |
| `net_proc_daddr_dport_mean_{wn}`, `net_proc_daddr_dport_std_{wn}` | decayed | Port mean/std per (process, destination IP). | no | [0, 1] |

**Comparison to Kitsune (NDSS 2018):** Kitsune uses packet-level streams with 5 decay windows and ~115 features. We work at syscall level (connect/socket), so we have no packet size or MAC. We mirror the idea with **five decay windows** (0.1s, 1s, 5s, 30s, 120s), **rate + interarrival mean/std + value mean/std** per stream. Shared online stats (proc/host rate, proc interarrival) apply to all events; type-specific stats add pair, daddr, path, and value distributions per type.

**Still doable with the same syscall-level data (not yet implemented):** (1) **2D / correlation**: covariance between two values (e.g. dport vs addrlen) per stream. (2) **Rarity / novelty**: binary or decayed "first time (comm, daddr)" in window. (3) **Per-protocol rate**: rate of TCP vs UDP per window. (4) **Longer windows**: e.g. 300s, 600s. (5) **Daddr addrlen**: mean/std of addrlen per destination IP.

**Process** (`event_type == "process"`):

| Feature | Source | Description | Discrete | Range |
|---------|--------|-------------|----------|-------|
| `process_is_execve` | `data[0]` (event_name) | 1.0 if event_name is execve; else 0.0. | yes | {0, 1} |

Missing or invalid fields use safe defaults (0.0, empty hash). Each per-event_type model sees a consistent feature set for its type.

---

## config.py

**Role:** Central configuration for the detector: ports, model choice, and all algorithm hyperparameters. Used by `server.py` and (for eval) by scripts that start the detector with overrides.

**Public API:**

- `DetectorConfig`: dataclass of default values (port, events_http_port, recent_events_buffer_size, model_algorithm, threshold, and all HST/LODA/KitNet/MemStream parameters).
- `load_config() -> DetectorConfig`: builds a `DetectorConfig` from the environment. Environment variables override defaults (e.g. `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, `DETECTOR_KITNET_*`, `DETECTOR_MEMSTREAM_*`, etc.).

No config file is read; configuration is env-only so it works well in containers and eval runs.

---

## model.py

**Role:** Online anomaly detection models that consume the feature dict from `features.py`, update their internal state (learn), and return an anomaly score per event. All implement the same interface: `score_and_learn(features: Dict[str, float]) -> float`.

**Public API:** `OnlineAnomalyDetector` is a factory. You pass `algorithm` (`"halfspacetrees"`, `"loda"`, `"kitnet"`, `"memstream"`) plus optional hyperparameters; it instantiates the corresponding implementation and exposes `score_and_learn(features)`.

**Algorithms:**

1. **Half-Space Trees (River)** – Tree ensemble over the feature vector; CPU-only, no PyTorch. Good baseline, low dependency.
2. **LODA** – Sparse random projections + online histograms; PyTorch, supports CPU/CUDA. Lightweight density estimate.
3. **KitNet (PySAD)** – Ensemble of small autoencoders; learns a mapping then an anomaly detector in two grace phases. PyTorch, CPU/CUDA.
4. **MemStream** – Autoencoder + latent memory; memory updated only when the score is below an adaptive threshold to limit poisoning. PyTorch, CPU/CUDA.

Device selection is via config (`model_device`: `auto` / `cpu` / `cuda`). The server uses one model per event_type and calls `score_and_learn` under a lock so only one thread updates any model per event.

---

## server.py

**Role:** gRPC + optional HTTP server that receives events, runs the feature extractor and model, and streams back detection responses. Also maintains in-memory buffers and optional file logging for anomalies.

**Flow:**

1. **Startup:** `load_config()`, create `RECENT_EVENTS` deque (size from config), optionally enable anomaly log file via `ANOMALY_LOG_PATH`, instantiate `RuleBasedDetector(cfg)` (which wraps `DeterministicScorer` and one `OnlineAnomalyDetector` **per event_type**).
2. **gRPC:** Implements `DetectorService`: `StreamEvents(stream EventEnvelope) -> stream DetectionResponse`. For each event: `extract_feature_dict(evt)` → pick/create the model for `evt.event_type` → `score_and_learn(features)` → build `DetectionResponse` (event_id, anomaly, reason, score, ts). Responses are yielded back to the client. Anomalies and recent events are pushed to in-memory buffers; if `ANOMALY_LOG_PATH` is set, anomalies are also appended as JSONL.
3. **DeterministicScorer:** Holds the config and the `OnlineAnomalyDetector`. `score_event(evt)` calls `extract_feature_dict(evt)` then `score_and_learn(features)`, compares score to `cfg.threshold`, and returns a `DetectionResponse` (anomaly = score ≥ threshold). Exceptions are caught and returned as a non-anomaly response with an error reason.
4. **HTTP (optional):** If `events_http_port` > 0, a small HTTP server serves:
   - `GET /recent_events?limit=N` – last N events (for UI log tail)
   - `GET /metrics` – Prometheus-style metrics (events_total, anomalies_total, etc.)
   - `GET /anomalies?limit=N` – last N anomalies

**Concurrency:** One gRPC stream is processed sequentially per connection; the lock in `DeterministicScorer.score_event` ensures a single model update per event. HTTP is served from a background thread.

**Entrypoint:** `python -m detector.server` (or Docker `ENTRYPOINT`). gRPC port from `DETECTOR_PORT`, HTTP from `DETECTOR_EVENTS_PORT` (config).

**Do we need a general layer or autoencoder to combine multiple models?** No, for the current design. We have **one model per event_type**; each event is routed to exactly one model and gets **one score**. There are no “multiple online extractors” whose outputs we must fuse per event. A fusion layer or meta-model would only be needed if we changed the architecture, for example: (1) **Ensemble**: run the same event through several models (e.g. a general + a type-specific) and combine scores (e.g. max, mean, or a small learned combiner). (2) **Score calibration**: normalize scores across event types so a single threshold behaves similarly for network vs file vs default (e.g. per-type running mean/std or a tiny calibration model). (3) **Single shared model**: one autoencoder or detector that sees all events in a common representation (e.g. per-type encoders mapping to a fixed latent size, then one decoder). That would require a common embedding size and more complexity. The current per-type design keeps types independent and avoids mixing score distributions; we keep it as-is unless we explicitly add ensemble or calibration.

---

## __init__.py

Package marker only; no public exports.

---

## Dockerfile

**Role:** Build a minimal image to run the detector as a service.

- Base: `python:3.11-slim`.
- Installs `uv`, then copies project deps and detector (and probe/scripts and generated gRPC stubs as needed), runs `uv sync --no-dev --frozen --extra detector`.
- `ENTRYPOINT`: `python -m detector.server`.

Configuration is via environment variables (see `config.py`). For eval or custom ports, set `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, etc., when running the container.
