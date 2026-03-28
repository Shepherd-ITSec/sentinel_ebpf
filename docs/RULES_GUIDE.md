# Rules Guide (Falco-like DSL)

This guide explains how Sentinel eBPF rules are authored and how they execute across kernel and userspace.

## Supported syscalls

The probe traces the following syscalls with dedicated tracepoints. These are the `syscall_name` values you can use in rules. The full registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`).

| syscall_name | syscall_nr | fd_path | arg0 | arg1 | notes |
|------------|----------|---------|------|------|-------|
| **File I/O** |
| `read` | 0 | ✓ (`attributes.fd_path`, fd cache) | fd | count | `return_value` = bytes read |
| `write` | 1 | ✓ (`attributes.fd_path`, fd cache) | fd | count | `return_value` = bytes written |
| `open` | 2 | ✓ (`attributes.fd_path`) | flags | — | Legacy; most programs use `openat`; `return_value` = fd or -errno |
| `openat` | 257 | ✓ (`attributes.fd_path`) | dfd | flags | Primary file-open syscall; `return_value` = fd or -errno |
| `openat2` | 437 | ✓ (`attributes.fd_path`) | dfd | usize | Extended open flags; `return_value` = fd or -errno |
| `close` | 3 | — | fd | — | Removes fd from path cache; `return_value` = 0 or -errno |
| **File metadata / deletion** |
| `unlink` | 87 | ✓ | — | — | |
| `unlinkat` | 263 | ✓ | dfd | flag | |
| `rename` | 82 | ✓ (old) | — | — | |
| `renameat` | 264 | ✓ (old) | olddfd | newdfd | |
| `chmod` | 90 | ✓ | mode | — | |
| `fchmod` | 91 | ✓ (from fd cache) | fd | mode | |
| `chown` | 92 | ✓ | user | group | |
| `fchown` | 93 | ✓ (from fd cache) | fd | user | |
| **Process** |
| `execve` | 59 | ✓ | — | — | Program path; `return_value` = -errno on failure |
| `fork` | 57 | — | — | — | Process creation; `return_value` = child pid or -errno |
| **Network** |
| `socket` | 41 | — | family | type | `return_value` = fd or -errno |
| `connect` | 42 | ✓ (`attributes.fd_sock_*` IPv4 peer) | fd | addrlen | `return_value` = 0 or -errno |
| `bind` | 49 | — | fd | addrlen | `return_value` = 0 or -errno |
| `listen` | 50 | — | fd | backlog | `return_value` = 0 or -errno |
| `accept` | 43 | — | fd | — | addrlen is pointer; `return_value` = fd or -errno |
| `accept4` | 288 | — | fd | flags | `return_value` = fd or -errno |
| **Additional (common LID-DS / mixed workloads)** |
| `fstat` | 5 | ✓ (from fd cache) | fd | — | `return_value` = 0 or -errno |
| `poll` | 7 | — | nfds | timeout | `return_value` = count or -errno |
| `mmap` | 9 | — | len | flags | `return_value` = address or -errno |
| `mprotect` | 10 | — | start | len | `return_value` = 0 or -errno |
| `ioctl` | 16 | ✓ (from fd cache) | fd | cmd | `return_value` = ioctl result |
| `sendto` | 44 | — | fd | len | `return_value` = bytes sent or -errno |
| `recvfrom` | 45 | — | fd | size | `return_value` = bytes recv or -errno |
| `getsockname` | 51 | — | fd | — | `return_value` = 0 or -errno |
| `getpeername` | 52 | — | fd | — | `return_value` = 0 or -errno |
| `setsockopt` | 54 | — | fd | optname | `return_value` = 0 or -errno |
| `fcntl` | 72 | ✓ (from fd cache) | fd | cmd | `return_value` = depends on cmd |
| `set_robust_list` | 273 | — | head | len | `return_value` = 0 or -errno |
| `sendmmsg` | 307 | — | fd | vlen | `return_value` = count or -errno |

Other syscall names in `probe/events.py` (e.g. `stat`, `access`) may be used in rules but are not traced by dedicated tracepoints; they only appear via the generic raw syscall enter path, with minimal data (arg0/arg1 only, no fd enrichment, no exit `return_value`).

**Source:** `probe/events.py` (registry), `probe/probe.bpf.c` (tracepoints).

## Event `attributes` (fd enrichment)

Enrichment is always about the **relevant file descriptor** (or open path). Canonical keys (probe and `replay_lidds`):

| Key | Purpose |
|-----|---------|
| `fd_resource_kind` | `file`, `tcp`, `udp`, `unix`, `pipe`, or `unknown` |
| `fd_path` | VFS path when the fd refers to a file (not TCP tuples) |
| `fd_sock_local_addr`, `fd_sock_local_port`, `fd_sock_remote_addr`, `fd_sock_remote_port` | Socket endpoints when known (ports decimal strings) |
| `fd_sock_family` | e.g. `2` (AF_INET) when known |

Rules and macros use **`attributes.fd_path`** (not `path`). Legacy keys `attributes.path`, `sin_*`, `dest_*` are not used.

## Source of truth

- Chart deployments read rules from `charts/sentinel-ebpf/rules.yaml`.
- Rules are mounted into the probe and detector at `/etc/sentinel-ebpf/rules.yaml`.
- Rules use: `groups` (model contract), `rules` (filters into a group), optional `lists` and `macros`.

## Group model

Each `groups.<name>` entry declares:

- **`syscalls`**: non-empty list of syscall names (or a `lists` reference). This is the **expected syscall vocabulary** for that `event_group`: detector `group_syscall_*` one-hots, overlap checks between enabled groups, and userspace matching for events whose name appears in this list. Names **need not** exist in `probe/events.py` yet (reserved for forward use); the probe only attaches tracepoints for **known** IDs. **Do not** treat this list as the capture filter—narrow capture in each rule’s `condition` (see below).
- **`features`**: optional maps of string lists (for example `sensitive_paths` / `tmp_paths` for path-prefix booleans in the detector).

## Rule model

Each rule has:

- `group`: the downstream category attached to matching events (written into `EventEnvelope.event_group`)
- `condition`: optional filter in the DSL (empty means “match all syscalls in the group” once group membership passes). Prefer including **`syscall_name in (…)`** (via a shared `lists` entry) so capture is explicit; the group’s `syscalls` field is not a substitute for that filter.
- `lists` / `macros`: optional helpers for reuse and readability

The `condition` part is a boolean expression over event fields.

- **Supported operators:** `=`, `in`, `startswith`, `contains`, `and`, `or`, `not`, parentheses
- **Supported fields:** `syscall_name`, `syscall_nr`, `event_id` (correlation id string), `attributes.fd_path`, `comm`, `pid`, `tid`, `uid`, `flags`, `arg0`, `arg1`, `arg_flags`, `return_value`, `hostname`, `namespace` (and optional `attributes.fd_sock_*`, `attributes.fd_resource_kind` for rules that need them)
- `arg_flags` is evaluated as an alias of `flags`.

## Selective probe attachment

At startup, the probe compiles **enabled** rules into kernel tuples. Per DNF branch, the syscall dimension comes from compilable **`syscall_name` / `syscall_nr`** predicates in the `condition` when present; if a branch has no such predicate, it falls back to the **known** syscall numbers from `groups[rule.group].syscalls`. The union of compiled syscall numbers (or `WILDCARD` when unavoidable) drives which tracepoints attach (`#if ENABLE_*`).

## Execution model: kernel prefilter + userspace final decision

Rules run in two stages:

1. **Kernel prefilter (BPF tuple match):**
   - Event can be dropped early in kernel if it does not match compiled tuples.
   - Compilable predicate types are positive checks on:
     - `syscall_name` / `syscall_nr`
     - `attributes.fd_path startswith`
     - `comm`
     - numeric IDs (`pid`, `tid`, `uid`)

2. **Userspace fallback + final evaluation:**
   - Full DSL AST is evaluated in Python.
   - Anything not kernel-compilable stays here (for example `not ...`, `contains`, some string logic).
   - Event is forwarded only if full condition evaluation is true.

## Why `not noisy_path` is userspace fallback

Even if `noisy_path` itself only uses `attributes.fd_path startswith`, `not noisy_path` introduces negation.
The current kernel tuple model only stores positive match dimensions, so negated predicates are not compiled.

## Why a few DSL rules can become many kernel tuples

Compiler output is not 1:1 with rule count. It expands conditions into DNF branches and then into tuple combinations.

Typical expansion pattern for the optional `condition`:

- `syscall_name in (...)` multiplies by number of syscalls
- `attributes.fd_path startswith A or attributes.fd_path startswith B` splits branches
- `comm in (...)` multiplies by number of comm values

Example from current default chart rules:

- `capture-file-events`: group `file` declares vocabulary; `condition` uses `syscall_name in (file_syscalls)` plus path filters
- `capture-shell-process-events`: group `process` declares vocabulary; `condition` uses `syscall_name in (process_syscalls)` plus comm filter
- `capture-network-events`: group `network` declares vocabulary; `condition` uses `syscall_name in (network_syscalls)`

## Limits and truncation

- Kernel tuple table is capped by `MAX_RULES` in BPF/probe runner (currently 24).
- If compiled tuples exceed cap, extra tuples are dropped.
- Probe exposes truncation and compile metrics so this is observable.

Key metrics:

- `sentinel_ebpf_probe_rules_compiled_total`
- `sentinel_ebpf_probe_rules_loaded_total`
- `sentinel_ebpf_probe_rules_truncated_total`
- `sentinel_ebpf_probe_kernel_compiled_predicates`
- `sentinel_ebpf_probe_kernel_fallback_predicates`
- `sentinel_ebpf_probe_kernel_branches_total`
- `sentinel_ebpf_probe_kernel_branches_compiled`
- `sentinel_ebpf_probe_kernel_branches_impossible`

## Authoring guidance

- Declare the per-group syscall vocabulary under `groups.<name>.syscalls` (and reuse the same names in `lists` where helpful). **Also** constrain capture per rule with `syscall_name in (…)` in `condition` when you want an explicit filter (recommended for default chart rules).
- Prefer reusable `lists` and `macros` for readability when conditions repeat.
- Keep conditions selective early (specific prefixes/comms).
- Use `not ...` when needed for clarity, but expect userspace fallback for that part.
- Watch tuple growth when adding wide `in (...)` lists combined with `or` branches.
- Check metrics after rule changes to detect truncation or high fallback share.

## Minimal pattern (recommended style)

```yaml
lists:
  file_syscalls: [open, openat, openat2]
  noisy_paths: [/proc, /sys]

macros:
  sensitive_path: "attributes.fd_path startswith /etc or attributes.fd_path startswith /root"
  noisy_path: "attributes.fd_path startswithin (noisy_paths)"

groups:
  file:
    syscalls: file_syscalls

rules:
  - name: capture-sensitive-file-events
    enabled: true
    group: file
    condition: "syscall_name in (file_syscalls) and sensitive_path and not noisy_path"
```
