# Detector

The detector is an anomaly detection service that consumes kernel-level events (e.g. syscalls) over gRPC, extracts features, scores them with an online ML model, and returns anomaly decisions. It can be run standalone (`python -m detector.server`) or in Docker.

## Event model

**event_name = syscall name; event_group = category.** There is no separate “network” vs “syscall” category: `event_name` is the syscall/event name (e.g. `openat`, `connect`, `socket`, `execve`); the registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`). The rule-defined category is **event_group** (e.g. `network`, `file`, `process`) and can be empty; it is carried in the protobuf field `EventEnvelope.event_group`.

**One envelope shape for all.** Syscalls are like function calls with defined inputs and outputs. The `EventEnvelope` is the single contract: every producer (eBPF probe, EVT1 replay, future sources) must send the same named syscall fields and attribute keys. For a given event only the relevant fields are non-empty (e.g. `path` for openat, empty for connect). Raw open/socket flag bits are already in `arg0` (open) or `arg1` (openat, openat2, socket) as emitted by the probe; decoded names go in `attributes["flags"]` when present.

---

## features.py

**Role:** Turn each `EventEnvelope` into numeric features (a dict of floats) for the anomaly models. **General features** are always added; **type-specific features** are added when the event has an **event_group** (stored in `evt.event_group`, e.g. `file`, `network`, `process`). The resulting dict can have different features (and thus different “size”) per event; the server uses one model per event_group so each model sees the same feature set for its group.

**Event format (protobuf / JSON):**

- `event_id` (string): correlation id for the message (not the Linux syscall number).
- `event_name`: syscall name string (e.g. `openat`, `connect`); used for flag-slot routing, network/socket parsing, and group-local one-hot labels.
- `syscall_nr` (uint32): Linux syscall number (e.g. `257` for `openat`); used for **`event_id_norm`** in the **frequency** feature view (`min(syscall_nr / 500, 1)` in code).
- `hostname` (string): host identity for bucket/hash features and online stats keys.
- `comm`, `pid`, `tid`, `uid` (strings): process context; `pid`/`tid`/`uid` use decimal string encoding as emitted by the probe.
- `arg0`, `arg1` (strings): syscall arguments; also used as raw flag numeric strings for `open` → `arg0`, `openat`/`openat2`/`socket` → `arg1` when deriving file-flag features (`file_flags_*` / `file_flags_hash`). `arg1` may hold a stringified `dict` (synthetic replay) for sockaddr fallback in network parsing.
- `path`: path for path-based syscalls; empty otherwise.
- `ts_unix_nano` (uint64): event time (UTC) for calendar and online-stat decay.

`evt.event_group` stores the rule-defined **event_group** (e.g. `file`, `network`, `process`, or empty) for routing to the per-group model and for type-specific features.

JSON/EVT1 records use the same names; see `event_envelope_codec.envelope_to_dict` / `envelope_from_dict`.

Optional `attributes` (map, producer-filled when available):

- `return_value`: syscall return (non-negative = success, negative = errno-style); drives `return_success` / `return_errno_norm` (missing → treated as `0`).
- `flags`: decoded flags string for open family / socket; preferred over raw flag slot for **`file_flags_*`** / **`file_flags_hash`** when present.
- `cluster`, `node`: optional deployment tags (probe env); not consumed by `features.py` today.
- Network enrichment: `sin_port` or `dest_port`, `sin_addr` or `dest_ip`, `sa_family` — merged with sockaddr parsed from `arg1` in `_parse_sockaddr_from_evt` for **`net_*`** address/port/family features.

**Rules file (`rules.yaml`):** Loaded from `DETECTOR_RULES_PATH`, else `/etc/sentinel-ebpf/rules.yaml` if present, else repo `charts/sentinel-ebpf/rules.yaml`. Supplies (1) which syscalls belong to each **`event_group`** for **`file_event_name_*`** / **`net_event_name_*`** one-hot vocabularies (with code fallbacks in `features.py` if a group is empty), and (2) **`groups.file.features.sensitive_paths`** / **`tmp_paths`** for **`file_sensitive_path`** / **`file_tmp_path`**.

**Online features:** `proc_rate_*`, `host_rate_*`, `proc_interarrival_*`, and type-specific `file_*` / `net_*` rate and moment streams come from an in-process **`_OnlineFeatureStats`** store (not from a single field on the event). They are emitted only in **default**, **loda**, and **memstream** (not **frequency**). Keys use `comm`, `hostname`, `path`, `(comm, path)`, `(comm, daddr, dport)`, etc., and `evt.ts_unix_nano`.

**Feature design:** Based on work on syscall-argument anomaly detection (e.g. Krügel/Mutz, “On the Detection of Anomalous System Call Arguments”, ESORICS 2003): identity and numeric/semantic features for call type, process, path, arguments, return value, and context (host, time).

**Public API:**

- `extract_feature_dict(evt, feature_view="default") -> Dict[str, float]`: general features always; if the event has an **event_group** (in `evt.event_group`) of `file`, `network`, or `process`, appends type-specific features. Key set (and “vector size”) can differ per event_group and per **feature view**.
- `feature_view_for_algorithm(algorithm) -> str`: maps algorithm to feature view. `freq1d`, `copulatree`, `latentcluster` → **`frequency`**; `loda` / `loda_ema` → `loda`; `memstream` → `memstream`; else → `default`.

**Feature views:** One extractor; the view toggles encodings (`_FeatureViewSpec` in `features.py`). **`default`**: sparse buckets + one-hots where listed below. **`frequency`**: MD5-derived scalars (`_hash01`) for comm/hostname/path + **`event_id_norm`** from **`syscall_nr`**, plus type-specific `*_hash` fields where applicable. It **does not** emit **`flags_hash`** (removed globally), **`file_sensitive_path`** / **`file_tmp_path`**, or **any online** rate/interarrival/moment streams (general, file, or network). Used by freq1d / copulatree / latentcluster. **`loda`** / **`loda_ema`**: drops general `comm`/`hostname`/`path` banks; for **file**, drops `file_event_name_*`, `file_extension_bucket_*`, `file_flags_bucket_*`; for **network**, drops `net_event_name_*` and `net_daddr_bucket_*`. **`memstream`**: same as loda for file/network, and also drops **`net_socket_type_bucket_*`**.

**General features:** No global syscall one-hot. Per-group syscall identity uses **`file_event_name_*`** / **`net_event_name_*`** (**default** only) or **`event_id_norm`** (**frequency** only). Empty **`event_group`**: no syscall one-hot; **frequency** still has **`event_id_norm`**. Online stats (**`proc_rate_*`**, **`host_rate_*`**, **`proc_interarrival_*`**) use five decay windows (`01` = 0.1s, `1`, `5`, `30`, `120` s) and appear only in **default**, **loda**, **memstream**.

**Views** column: only the listed feature views include that row (four views exist: **default**, **frequency**, **loda**, **memstream**).

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `pid_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | `evt.pid` → int, scaled by `_PID_MAX` |
| `tid_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | `evt.tid` → int, scaled by `_PID_MAX` |
| `uid_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | `evt.uid` → int, scaled by `_UID_MAX` |
| `arg0_norm`, `arg1_norm` | default, frequency, loda, memstream | log1p-magnitude | [0, 1] | `evt.arg0` / `evt.arg1` → int, `_norm_arg` |
| `hour_sin`, `hour_cos` | default, loda, memstream | cyclic sin/cos | [-1, 1] | `evt.ts_unix_nano` → UTC hour |
| `minute_sin`, `minute_cos` | default, loda, memstream | cyclic sin/cos | [-1, 1] | `evt.ts_unix_nano` → minute-of-hour |
| `weekday_sin`, `weekday_cos` | default, loda, memstream | cyclic sin/cos | [-1, 1] | `evt.ts_unix_nano` → weekday |
| `hour_norm`, `minute_norm`, `weekday_norm` | frequency | normalized integer | [0, 1] | UTC hour / 23, minute / 59, Python `weekday()` / 6 |
| `week_of_month_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | `evt.ts_unix_nano` → week-of-month bucket |
| `path_depth_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | Slash-split `evt.path` → component count |
| `return_success` | default, frequency, loda, memstream | binary | {0, 1} | `int(attributes["return_value"], default 0) >= 0` |
| `return_errno_norm` | default, frequency, loda, memstream | log1p-magnitude | [0, 1] | `log1p(abs(int(attributes["return_value"], default 0)))` scaled by `_RETURN_ERRNO_SCALE` (same formula success or fail) |
| `comm_bucket_*` | default | hashed sparse bank | {0, 1} | MD5 bucket of `evt.comm` (`_COMM_BUCKETS`) |
| `hostname_bucket_*` | default | hashed sparse bank | {0, 1} | MD5 bucket of `evt.hostname` (`_HOSTNAME_BUCKETS`) |
| `path_tok_d{0..3}_bucket_*` | default | depth multi-hot banks | {0, 1} | Tokenized `evt.path` per depth (`_path_components` + `_tokenize_path_by_depth`) |
| `comm_hash` | frequency | scalar hash | [0, 1) | `_hash01(evt.comm)` |
| `path_hash` | frequency | scalar hash | [0, 1) | `_hash01(evt.path)` |
| `path_prefix_hash` | frequency | scalar hash | [0, 1) | First path component after split, `_hash01` |
| `hostname_hash` | frequency | scalar hash | [0, 1) | `_hash01(evt.hostname)` |
| `event_id_norm` | frequency | normalized integer | [0, 1] | `evt.syscall_nr` → int, `min(nr / _EVENT_ID_MAX, 1)` (500 in code) |
| `proc_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | `_ONLINE_STATS`, stream key `comm`, group `general` |
| `host_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | `_ONLINE_STATS`, stream key `hostname`, group `general` |
| `proc_interarrival_{wn}` | default, loda, memstream | decayed mean | [0, 1] | `_ONLINE_STATS` interarrival on `comm` |
| `proc_interarrival_std_{wn}` | default, loda, memstream | decayed std | [0, 1] | Same stream as `proc_interarrival_*` |

**Type-specific features:** Added only when `evt.event_group` matches.

Feature coverage is intentionally scoped to the currently supported/traced syscall families behind `event_group = file|network|process`; we do not claim uniform syscall-specific semantics across the full Linux syscall surface.

### File (`event_group == "file"`)

Prefix lists for **`file_sensitive_path`** / **`file_tmp_path`** come from **`groups.file.features`** in `rules.yaml` (fallbacks: `_DEFAULT_SENSITIVE_PATH_PREFIXES`, `_DEFAULT_TMP_PATH_PREFIXES` in code).

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `file_sensitive_path` | default, loda, memstream | binary | {0, 1} | Lowercased `evt.path` starts with any configured sensitive prefix |
| `file_tmp_path` | default, loda, memstream | binary | {0, 1} | Lowercased `evt.path` starts with any configured tmp prefix |
| `file_event_name_*` | default | one-hot | {0, 1} | Lowercased `evt.event_name` vs `_group_syscalls("file", …)` from rules (fallback `_DEFAULT_FILE_EVENT_NAMES`) |
| `file_extension_bucket_*` | default | hashed sparse bank | {0, 1} | Last `path` component: extension after `.`, else `""`; MD5 bucket `_FILE_EXTENSION_BUCKETS` |
| `file_flags_bucket_*` | default | hashed multi-hot | {0, 1} | `attributes["flags"]` if set, else `_syscall_flags_numeric_string(evt)`; tokens → `_FILE_FLAGS_BUCKETS` |
| `file_extension_hash` | frequency | scalar hash | [0, 1) | `_hash01` of extension string (same derivation as buckets) |
| `file_flags_hash` | frequency | scalar hash | [0, 1) | `_hash01` of full flags string (same as `file_flags_bucket_*` input) |
| `file_path_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | `_ONLINE_STATS` group `file`, stream = `evt.path` or `"unknown"` |
| `file_pair_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | Stream = `f"{comm}|{path}"` (`comm` from `evt.comm`) |
| `file_pair_interarrival_{wn}` | default, loda, memstream | decayed mean | [0, 1] | Interarrival on same pair key |
| `file_pair_interarrival_std_{wn}` | default, loda, memstream | decayed std | [0, 1] | Same pair key |
| `file_proc_path_depth_mean_{wn}`, `file_proc_path_depth_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `path_depth_norm` from `evt.path` streamed by `comm` |
| `file_host_path_depth_mean_{wn}`, `file_host_path_depth_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | Same depth signal streamed by `evt.hostname` |
| `file_pair_path_depth_mean_{wn}`, `file_pair_path_depth_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | Same depth signal streamed by `comm|path` |

### Network (`event_group == "network"`)

Sockaddr fields are built in **`_parse_sockaddr_from_evt`**: `attributes` (`sin_port`/`dest_port`, `sin_addr`/`dest_ip`, `sa_family`) first, else **`ast.literal_eval(evt.arg1)`** if `arg1` looks like a `dict` string.

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `net_fd_norm` | default, frequency, loda, memstream | log1p-magnitude | [0, 1] | `evt.arg0` → int, `_norm_arg` |
| `net_addrlen_norm` | default, frequency, loda, memstream | log1p-magnitude | [0, 1] | `evt.arg1` → int (addrlen magnitude), `_ADDRLEN_SCALE` |
| `net_socket_family_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | If `event_name=="socket"`: `arg0` as family; else numeric hint from sockaddr `sa_family` / INET vs INET6 strings |
| `net_dport_norm` | default, frequency, loda, memstream | normalized integer | [0, 1] | Parsed `sin_port` → int / 65535 |
| `net_event_name_*` | default | one-hot | {0, 1} | Lowercased `evt.event_name` vs `_group_syscalls("network", …)` (fallback `_DEFAULT_NET_EVENT_NAMES`) |
| `net_socket_type_bucket_*` | default, loda | hashed sparse bank | {0, 1} | If `event_name=="socket"`: `str(arg1)` into `_NET_SOCKET_TYPE_BUCKETS`; else `str(0)` |
| `net_daddr_bucket_*` | default | hashed sparse bank | {0, 1} | MD5 bucket of parsed destination IP string |
| `net_af_*` | default, loda, memstream | one-hot | {0, 1} | `_normalize_af_label(sa_family string, family_val)` vs `_NET_AF_VALUES` |
| `net_socket_type_hash` | frequency | scalar hash | [0, 1) | `_hash01(str(type_val))` (same `type_val` as buckets) |
| `net_daddr_hash` | frequency | scalar hash | [0, 1) | `_hash01` parsed daddr |
| `net_af_hash` | frequency | scalar hash | [0, 1) | `_hash01` resolved AF string (sockaddr or inferred from `family_val`) |
| `net_pair_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | `_ONLINE_STATS` group `network`, key `comm|daddr|dport` |
| `net_daddr_rate_{wn}` | default, loda, memstream | decayed log-scaled rate | [0, 1] | Key = daddr or `"unknown"` |
| `net_pair_interarrival_{wn}`, `net_pair_interarrival_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | Same pair key as `net_pair_rate_*` |
| `net_proc_dport_mean_{wn}`, `net_proc_dport_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_dport_norm` streamed by `comm` |
| `net_proc_addrlen_mean_{wn}`, `net_proc_addrlen_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_addrlen_norm` by `comm` |
| `net_host_dport_mean_{wn}`, `net_host_dport_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_dport_norm` by `hostname` |
| `net_host_addrlen_mean_{wn}`, `net_host_addrlen_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_addrlen_norm` by `hostname` |
| `net_daddr_dport_mean_{wn}`, `net_daddr_dport_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_dport_norm` by daddr key |
| `net_proc_daddr_dport_mean_{wn}`, `net_proc_daddr_dport_std_{wn}` | default, loda, memstream | decayed mean/std | [0, 1] | `net_dport_norm` by `comm|daddr` |

**Comparison to Kitsune (NDSS 2018):** Kitsune uses packet-level streams with 5 decay windows and ~115 features. We work at syscall level (connect/socket), so we have no packet size or MAC. We mirror the idea with **five decay windows** (0.1s, 1s, 5s, 30s, 120s), **rate + interarrival mean/std + value mean/std** per stream. Those online streams appear in **default**, **loda**, and **memstream** only (**frequency** omits them). Type-specific stats add pair, daddr, path, and value distributions per type (same three views).

**Still doable with the same syscall-level data (not yet implemented):** (1) **2D / correlation**: covariance between two values (e.g. dport vs addrlen) per stream. (2) **Rarity / novelty**: binary or decayed "first time (comm, daddr)" in window. (3) **Per-protocol rate**: rate of TCP vs UDP per window. (4) **Longer windows**: e.g. 300s, 600s. (5) **Daddr addrlen**: mean/std of addrlen per destination IP.

### Process (`event_group == "process"`)

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `process_is_execve` | default, frequency, loda, memstream | binary | {0, 1} | `1.0` if `evt.event_name == "execve"` (exact string), else `0.0` |
| `process_is_fork` | default, frequency, loda, memstream | binary | {0, 1} | `1.0` if `evt.event_name == "fork"`, else `0.0` |

View-independent; same in **frequency** as in **default** (`_extract_process_features` ignores view).


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

**Public API:** `OnlineAnomalyDetector` is a factory. You pass `algorithm` (`"halfspacetrees"`, `"loda"`, `"loda_ema"`, `"kitnet"`, `"memstream"`, `"zscore"`, `"knn"`, `"freq1d"`, `"copulatree"`, `"latentcluster"`) plus optional hyperparameters; it instantiates the corresponding implementation and exposes `score_and_learn(features)`.

**Algorithms:**

1. **Half-Space Trees (River)** – Tree ensemble over the feature vector; CPU-only, no PyTorch. Good baseline, low dependency.
2. **LODA (PySAD)** – PySAD LODA wrapper. **Broken**: fit_partial overwrites histograms instead of accumulating; scores are often ~0. Not recommended.
3. **LODA-EMA** – Custom implementation with EMA-based adaptive normalization; streaming from first event; PyTorch, supports CPU/CUDA. **Recommended for LODA.**
4. **KitNet (PySAD)** – Ensemble of small autoencoders; learns a mapping then an anomaly detector in two grace phases. PyTorch, CPU/CUDA.
5. **MemStream** – Paper-aligned (WWW'22): single-layer encoder/decoder with latent = 2×input_dim, Tanh, K-NN discounted L1 scoring, FIFO memory when score ≤ β. PyTorch, CPU/CUDA.
6. **ZScore** – Per-feature online running mean/std with event score `mean(abs(z_i))`. Intentionally simple baseline; CPU-only.
7. **KNN (scikit-learn)** – Sliding-memory nearest-neighbor detector; score is mean distance to k nearest historical events. CPU-only.
8. **Freq1D** – Per-feature 1D frequency baseline: numeric features use fixed-bin histograms, categorical/hash features use capped count tables. Scores by configurable aggregation over excess surprisal (sum / mean / top-k mean / soft top-k mean). CPU-only.
9. **CopulaTree** – Streaming copula-tree detector on top of `freq1d`: marginals come from `freq1d`, pairwise dependence is tracked online in Gaussianized space, and a maximum-spanning tree is refreshed periodically. This keeps the sparse-tree idea from the paper without requiring offline family fitting or Monte Carlo calibration. CPU-only.
10. **LatentCluster** – Online latent clustering on top of `freq1d` marginals: events are mapped to CDF/probit coordinates, scored against a small bank of latent clusters with diagonal variance, and only likely-normal points update clusters. CPU-only.

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