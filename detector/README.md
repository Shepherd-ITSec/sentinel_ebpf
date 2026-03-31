# Detector

The detector is an anomaly detection service that consumes kernel-level events (e.g. syscalls) over gRPC, extracts features, scores them with an online ML model, and returns anomaly decisions. It can be run standalone (`python -m detector.server`) or in Docker.

## Event model

**syscall_name = Linux syscall name; event_group = category.** There is no separate “network” vs “syscall” category: `syscall_name` is the syscall name (e.g. `openat`, `connect`, `socket`, `execve`); the registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`). The rule-defined category is **event_group** (e.g. `network`, `file`, `process`) and can be empty; it is carried in the protobuf field `EventEnvelope.event_group`.

**One envelope shape for all.** Syscalls are like function calls with defined inputs and outputs. The `EventEnvelope` is the single contract: every producer (eBPF probe, EVT1 replay, future sources) must send the same named syscall fields and attribute keys. For a given event only the relevant fields are non-empty (e.g. `path` for openat, empty for connect). Raw open/socket flag bits are already in `arg0` (open) or `arg1` (openat, openat2, socket) as emitted by the probe; decoded names go in `attributes["flags"]` when present.

---

## features.py

**Role:** Turn each `EventEnvelope` into numeric features (a dict of floats) for the anomaly models. **General features** are always added; **group-scoped features** are added when `evt.event_group` is non-empty and that name exists under `groups` in `rules.yaml`. The server uses one model per `event_group`; each model sees a stable key set for its group (vocabulary size follows that group’s declared `syscalls` list).

**Event format (protobuf / JSON):**

- `event_id` (string): correlation id for the message (not the Linux syscall number).
- `syscall_name`: Linux syscall name string (e.g. `openat`, `connect`); used for flag-slot routing, network/socket parsing, and group-local one-hot labels.
- `syscall_nr` (uint32): Linux syscall number (e.g. `257` for `openat`); used for **`event_id_norm`** in the **frequency** feature view (`min(syscall_nr / 500, 1)` in code).
- `hostname` (string): host identity for bucket/hash features and online stats keys.
- `comm`, `pid`, `tid`, `uid` (strings): process context; `pid`/`tid`/`uid` use decimal string encoding as emitted by the probe.
- `arg0`, `arg1` (strings): syscall arguments; also used as raw flag numeric strings for `open` → `arg0`, `openat`/`openat2`/`socket` → `arg1` when deriving `group_flags_*` / `group_flags_hash`. `arg1` may hold a stringified `dict` (synthetic replay) for sockaddr fallback in **`net_*`** parsing.
- `path`: path for path-based syscalls; empty otherwise.
- `ts_unix_nano` (uint64): event time (UTC) for calendar and online-stat decay.

`evt.event_group` stores the rule-defined **event_group** (e.g. `file`, `network`, `process`, or empty) for routing to the per-group model and for type-specific features.

JSON/EVT1 records use the same names; see `event_envelope_codec.envelope_to_dict` / `envelope_from_dict`.

Optional `attributes` (map, producer-filled when available):

- `return_value`: syscall return (non-negative = success, negative = errno-style); drives `return_success` / `return_errno_norm` (missing → treated as `0`).
- `flags`: decoded flags string for open family / socket; preferred over raw flag slot for **`group_flags_*`** / **`group_flags_hash`** when present.
- `cluster`, `node`: optional deployment tags (probe env); not consumed by `features.py` today.
- Network enrichment: `sin_port` or `dest_port`, `sin_addr` or `dest_ip`, `sa_family` — merged with sockaddr parsed from `arg1` in `_parse_sockaddr_from_evt` for **`net_*`** address/port/family features.

**Rules file (`rules.yaml`):** Loaded from `DETECTOR_RULES_PATH`, else `/etc/sentinel-ebpf/rules.yaml` if present, else repo `charts/sentinel-ebpf/rules.yaml`. Supplies (1) **`groups.<name>.syscalls`** for **`group_syscall_*`** one-hot vocabularies (names may include placeholders not yet in `probe/events.py`; those get a column with value 0 until the name appears on events), and (2) optional **`groups.<name>.features`** (e.g. **`sensitive_paths`** / **`tmp_paths`**) for **`group_sensitive_path`** / **`group_tmp_path`**. Which events are captured is decided by probe **rules** (`condition`), not by the group list alone—see `docs/RULES_GUIDE.md`.

**Online features:** `proc_rate_*`, `host_rate_*`, `proc_interarrival_*`, plus per-`event_group` **`group_path_*`** / **`net_*`** rate and moment streams from **`_OnlineFeatureStats`**. Emitted only in **default**, **loda**, and **memstream** (not **frequency**). Stream keys use `comm`, `hostname`, `path`, `(comm, path)`, `(comm, daddr, dport)`, etc., and `evt.ts_unix_nano`.

**Feature design:** Based on work on syscall-argument anomaly detection (e.g. Krügel/Mutz, “On the Detection of Anomalous System Call Arguments”, ESORICS 2003): identity and numeric/semantic features for call type, process, path, arguments, return value, and context (host, time).

**Public API:**

- `extract_feature_dict(evt, feature_view="default") -> Dict[str, float]`: general features always; if **`evt.event_group`** matches a **`groups`** entry in rules, appends group-scoped features (same key template for every event in that group). Key set can differ per **event_group** and per **feature view**.
- `feature_view_for_algorithm(algorithm) -> str`: maps algorithm to feature view. `freq1d`, `copulatree`, `latentcluster` → **`frequency`**; `loda` / `loda_ema` → `loda`; `memstream` → `memstream`; **`sequence_mlp`** / **`sequence_transformer`** → **`sequence`**; else → `default`.

**Feature views:** One extractor; the view toggles encodings (`_FeatureViewSpec` in `features.py`). **`default`**: sparse buckets + **`group_syscall_*`** one-hots (from rules) and the tables below. **`frequency`**: MD5-derived scalars for comm/hostname/path + **`event_id_norm`**, plus **`group_ext_hash`**, **`group_flags_hash`**, **`net_*_hash`** as applicable. Omits **`flags_hash`**, **`group_sensitive_path`** / **`group_tmp_path`**, and **all** online streams. **`loda`** / **`loda_ema`**: drops general `comm`/`hostname`/`path` banks; drops **`group_syscall_*`**, **`group_ext_bucket_*`**, **`group_flags_bucket_*`**, **`net_daddr_bucket_*`**. **`memstream`**: same as loda and also drops **`net_socket_type_bucket_*`**.

**General features:** No global syscall one-hot. Per-group syscall identity uses **`group_syscall_*`** (**default** only) or **`event_id_norm`** (**frequency** only). Empty **`event_group`**: no group block. Online stats (**`proc_rate_*`**, **`host_rate_*`**, **`proc_interarrival_*`**) use five decay windows (`01` = 0.1s, `1`, `5`, `30`, `120` s) and appear only in **default**, **loda**, **memstream**.

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

**Group-scoped block** (same template for every non-empty `event_group` defined in rules; syscall vocabulary is **`groups.<name>.syscalls`**, not a hardcoded taxonomy):

Prefix lists for **`group_sensitive_path`** / **`group_tmp_path`** come from **`groups.<name>.features`** (fallbacks: `_DEFAULT_SENSITIVE_PATH_PREFIXES`, `_DEFAULT_TMP_PATH_PREFIXES`).

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `group_sensitive_path` | default, loda, memstream | binary | {0, 1} | Lowercased `evt.path` starts with any configured sensitive prefix |
| `group_tmp_path` | default, loda, memstream | binary | {0, 1} | Lowercased `evt.path` starts with any configured tmp prefix |
| `group_syscall_*` | default | one-hot | {0, 1} | `evt.syscall_name` vs sorted `groups.<name>.syscalls` |
| `group_ext_bucket_*` | default | hashed sparse bank | {0, 1} | Extension from last `path` component |
| `group_flags_bucket_*` | default | hashed multi-hot | {0, 1} | `attributes["flags"]` or `_syscall_flags_numeric_string(evt)` |
| `group_ext_hash` | frequency | scalar hash | [0, 1) | `_hash01` of extension string |
| `group_flags_hash` | frequency | scalar hash | [0, 1) | `_hash01` of flags string |
| `group_path_rate_{wn}` | default, loda, memstream | decayed rate | [0, 1] | `_ONLINE_STATS`, stream = `path` or `"unknown"`, group = `event_group` |
| `group_pair_rate_{wn}` | default, loda, memstream | decayed rate | [0, 1] | Stream `comm|path` |
| `group_pair_interarrival_*`, `group_proc_path_depth_*`, `group_host_path_depth_*`, `group_pair_path_depth_*` | default, loda, memstream | decayed stats | [0, 1] | Path-depth / interarrival on same keys |

**`net_*` block** (always merged for group events; values are 0 / empty when sockaddr fields are missing). Sockaddr from `attributes` or **`ast.literal_eval(evt.arg1)`** when `arg1` is a dict string.

| Feature / pattern | Views | Encoded as | Range | Source |
|---|---|---|---|---|
| `net_fd_norm`, `net_addrlen_norm`, `net_socket_family_norm`, `net_dport_norm` | default, frequency, loda, memstream | numeric | [0, 1] | Args + parsed sockaddr |
| `net_socket_type_bucket_*`, `net_daddr_bucket_*`, `net_af_*` | default / loda / memstream | buckets / one-hot | {0, 1} | Socket type, daddr, AF label |
| `net_*_hash` | frequency | scalar hash | [0, 1) | MD5 scalars for type/daddr/af |
| `net_pair_rate_*`, `net_daddr_rate_*`, `net_*_mean/std_*` | default, loda, memstream | decayed stats | [0, 1] | `_ONLINE_STATS` metrics `net_rate`, `net_interarrival`, `net_dport`, `net_addrlen` under stream keys derived from `comm`, `hostname`, daddr |

**Comparison to Kitsune (NDSS 2018):** (unchanged) five decay windows (0.1s–120s), rate + interarrival + value moments on typed streams; **frequency** omits online streams.

Missing or invalid fields use safe defaults. Each per-event_group model sees a consistent feature set for its group, and the **feature view** (chosen per algorithm) is fixed for that model instance.

---

## config.py

**Role:** Central configuration for the detector: ports, model choice, score space, and all algorithm hyperparameters. Used by `server.py` and (for eval) by scripts that start the detector with overrides.

**Public API:**

- `DetectorConfig`: dataclass of default values (port, events_http_port, recent_events_buffer_size, model_algorithm, threshold, score_mode, and all HST/LODA/KitNet/MemStream/ZScore/KNN/Freq1D parameters, plus **sequence** fields: `sequence_ngram_length`, `sequence_thread_aware`, `sequence_mlp_hidden_size`, `sequence_mlp_hidden_layers`, `sequence_mlp_lr`, `DETECTOR_SEQUENCE_*` envs). The feature view is derived from `model_algorithm` via `feature_view_for_algorithm`.
- `load_config() -> DetectorConfig`: builds a `DetectorConfig` from the environment. Environment variables override defaults (e.g. `DETECTOR_PORT`, `DETECTOR_MODEL_ALGORITHM`, `DETECTOR_THRESHOLD`, `DETECTOR_SCORE_MODE`, `DETECTOR_KITNET_*`, `DETECTOR_MEMSTREAM_*`, `DETECTOR_SEQUENCE_*`, etc.).

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

**`sequence_mlp`** — thread-aware syscall n-gram + gensim Word2Vec + PyTorch MLP (score `1 − p(correct syscall)`), ported from LID-DS building blocks. Implemented as a normal `OnlineAnomalyDetector` algorithm that consumes the **`sequence`** feature view like any other feature-dict model. Word2Vec lives in `detector/embeddings/` so it can be reused by other models/views. Requires `gensim` (`uv sync --extra detector`). Logic is GPL-3.0-derived from vendored `third_party/LID-DS`.

Device selection is via config (`model_device`: `auto` / `cpu` / `cuda`). The server uses one model per event_group and calls `score_and_learn` under a lock so only one thread updates any model per event.

**Deferred idea for later:** a **learned mixture-of-regimes** model on top of `freq1d`-normalized inputs. The intended design is: keep `freq1d` for marginals, transform each event to CDF/probit coordinates, use a small gating network to softly assign the event to one of several latent regimes, and score the event by how well at least one regime explains it. Unlike hard routing by `comm`/event_group, the partition is learned from data and can differ between systems. Compared with `latentcluster`, the mixture keeps *soft* assignments and regime-specific density models instead of just nearest-cluster geometry.

---

## server.py

**Role:** gRPC + optional HTTP server that receives events, runs the feature extractor and model, and streams back detection responses. Also maintains in-memory buffers and optional file logging for anomalies.

**Flow:**

1. **Startup:** `load_config()`, create `RECENT_EVENTS` deque (size from config), optionally enable anomaly log file via `ANOMALY_LOG_PATH`, instantiate `RuleBasedDetector(cfg)` (which wraps `DeterministicScorer` and one scorer **per event_group**).
2. **gRPC:** Implements `DetectorService`: `StreamEvents(stream EventEnvelope) -> stream DetectionResponse`. For each event: pick/create the per-`event_group` model, extract the configured feature view, and call `model.score_and_learn_event(evt, feature_fn=extract_feature_dict)`. Build `DetectionResponse` (event_id, anomaly, reason, score, ts). Responses are yielded back to the client. Anomalies and recent events are pushed to in-memory buffers; if `ANOMALY_LOG_PATH` is set, anomalies are also appended as JSONL.
3. **DeterministicScorer:** Holds the config and one model per `event_group`. `score_event(evt)` applies an optional **model-independent** warmup gate (`DETECTOR_WARMUP_EVENTS` + `DETECTOR_SUPPRESS_ANOMALIES_DURING_WARMUP`) to suppress anomaly decisions and UI score early in a run.
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