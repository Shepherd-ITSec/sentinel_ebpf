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

**General features (18):** Always present. Each is read from `evt` (data list, ts_unix_nano, attributes, hostname).

| Feature | Source | Description |
|---------|--------|-------------|
| `event_hash` | `data[0]` | Hash of event name (0–1). |
| `comm_hash` | `data[2]` | Hash of process name. |
| `path_hash` | `data[8]` | Hash of path string. |
| `pid_norm` | `data[3]` | PID normalized (0–1). |
| `tid_norm` | `data[4]` | TID normalized (0–1). |
| `uid_norm` | `data[5]` | UID normalized (0–1). |
| `arg0_norm` | `data[6]` | Log-scaled magnitude of first arg. |
| `arg1_norm` | `data[7]` | Log-scaled magnitude of second arg. |
| `hour_norm` | `evt.ts_unix_nano` | Hour of day (0–1). |
| `minute_norm` | `evt.ts_unix_nano` | Minute of hour (0–1). |
| `event_id_norm` | `data[1]` | Numeric syscall ID normalized (e.g. 42, 257). |
| `flags_hash` | `data[9]` | Hash of flags string (e.g. open mode). |
| `path_depth_norm` | derived from `data[8]` (path) | Number of path components, normalized. |
| `path_prefix_hash` | derived from `data[8]` (path) | Hash of first path component (e.g. `etc`, `tmp`). |
| `return_success` | `evt.attributes["return_value"]` | 1.0 if return ≥ 0, else 0.0. |
| `return_errno_norm` | `evt.attributes["return_value"]` | Log-scaled magnitude of errno. |
| `mount_ns_hash` | `evt.attributes["mount_namespace"]` | Container/isolation context. |
| `hostname_hash` | `evt.hostname` | Hash of the hostname of the machine/node that emitted the event. |

**Type-specific features:** Added only when `evt.event_type` matches. One table per type.

**File** (`event_type == "file"`):

| Feature | Source | Description |
|---------|--------|-------------|
| `file_sensitive_path` | derived from `data[8]` (path) | 1.0 if path under /etc, /root, /bin, /sbin, /usr/bin, /usr/sbin; else 0.0. |
| `file_tmp_path` | derived from `data[8]` (path) | 1.0 if path under /tmp or /var/tmp; else 0.0. |
| `file_extension_hash` | derived from `data[8]` (path) | Hash of file extension (e.g. .so, .conf); empty → 0.0. |

**Network** (`event_type == "network"`):

| Feature | Source | Description |
|---------|--------|-------------|
| `net_addrlen_norm` | `data[7]` | Log-scaled addrlen (connect arg1). |
| `net_fd_norm` | `data[6]` | Log-scaled fd (connect arg0). |
| `net_socket_family_norm` | `data[6]` (socket) or parsed sockaddr | AF_INET(2)/AF_INET6(10) normalized (0–1). |
| `net_socket_type_hash` | `data[7]` (socket) | Hash of socket type (SOCK_STREAM=1, SOCK_DGRAM=2). |
| `net_dport_norm` | attributes or parsed `data[7]` (BETH) | Destination port 0–65535 → 0–1. |
| `net_daddr_hash` | attributes or parsed sockaddr | Hash of destination IP (socket-pair / host identity). |
| `net_af_hash` | attributes or parsed sockaddr | Hash of address family. |
| *Online stats (5 windows: 01=0.1s, 1, 5, 30, 120s)* | | |
| `net_*_rate_{wn}` | decayed rate | Process, host, pair (`comm\|daddr\|dport`), and per-destination (daddr) rate in [0,1). |
| `net_proc_interarrival_{wn}` | decayed mean | Normalized inter-arrival for process and for pair. |
| `net_proc_interarrival_std_{wn}` | decayed std | Inter-arrival std (burstiness) for process and for pair. |
| `net_proc_dport_mean_{wn}`, `net_proc_dport_std_{wn}` | decayed | Destination port mean/std per process. |
| `net_proc_addrlen_mean_{wn}`, `net_proc_addrlen_std_{wn}` | decayed | Addrlen mean/std per process. |
| `net_host_dport_mean_{wn}`, `net_host_dport_std_{wn}` | decayed | Destination port mean/std per host. |
| `net_host_addrlen_mean_{wn}`, `net_host_addrlen_std_{wn}` | decayed | Addrlen mean/std per host. |
| `net_daddr_dport_mean_{wn}`, `net_daddr_dport_std_{wn}` | decayed | Destination port mean/std per destination IP. |
| `net_proc_daddr_dport_mean_{wn}`, `net_proc_daddr_dport_std_{wn}` | decayed | Port mean/std per (process, destination IP) i.e. port distribution for that process→IP. |

**Comparison to Kitsune (NDSS 2018):** Kitsune uses packet-level streams with 5 decay windows (λ ∈ {5, 3, 1, 0.1, 0.01} s) and ~115 features: per-IP and MAC–IP packet-size stats (3 each), host–host and socket–socket (7 each), and per-IP jitter (3). We work at syscall level (connect/socket), so we have no packet size or MAC; we mirror the idea with **five decay windows** (0.1s, 1s, 5s, 30s, 120s), **rate + interarrival mean/std + value mean/std** per stream, and **process, host, pair, daddr, host value stats, daddr value stats, and process→IP (no port) value stats**. Remaining differences: input is syscalls not packets (no packet-size or MAC); we use one decay formulation (exponential in time delta) rather than Kitsune’s exact λ schedule.

**Still doable with the same syscall-level data (not yet implemented):** (1) **2D / correlation**: covariance or correlation between two values (e.g. dport vs addrlen) per stream would need sum_xy in the decayed stats. (2) **Rarity / novelty**: binary or decayed “first time (comm, daddr)” in window. (3) **Per-protocol rate**: rate of TCP vs UDP (or proto hash bucket) per window. (4) **Longer windows**: e.g. 300s, 600s for very long baseline. (5) **Daddr addrlen**: mean/std of addrlen per destination IP (we have daddr dport only).

**Process** (`event_type == "process"`):

| Feature | Source | Description |
|---------|--------|-------------|
| `process_is_execve` | `data[0]` (event_name) | 1.0 if event_name is execve; else 0.0. |

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
