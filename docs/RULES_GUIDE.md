# Rules Guide (Falco-like DSL)

This guide explains how Sentinel eBPF rules are authored and how they execute across kernel and userspace.

## Supported syscalls

The probe traces the following syscalls with dedicated tracepoints. These are the `event_name` values you can use in rules. The full registry lives in `probe/events.py` (`EVENT_NAME_TO_ID`).

| event_name | event_id | path | arg0 | arg1 | notes |
|------------|----------|------|------|------|-------|
| **File I/O** |
| `read` | 0 | ✓ (from fd cache) | fd | count | Path from fd cache; `return_value` = bytes read |
| `write` | 1 | ✓ (from fd cache) | fd | count | Path from fd cache; `return_value` = bytes written |
| `open` | 2 | ✓ | flags | — | Legacy; most programs use `openat`; `return_value` = fd or -errno |
| `openat` | 257 | ✓ | dfd | flags | Primary file-open syscall; `return_value` = fd or -errno |
| `openat2` | 437 | ✓ | dfd | usize | Extended open flags; `return_value` = fd or -errno |
| `close` | 3 | — | fd | — | Removes fd from path cache; `return_value` = 0 or -errno |
| **File metadata / deletion** |
| `unlink` | 87 | ✓ | — | — | |
| `unlinkat` | 263 | ✓ | dfd | flag | |
| `rename` | 82 | ✓ (old) | — | — | |
| `renameat` | 264 | ✓ (old) | olddfd | newdfd | |
| `chmod` | 90 | ✓ | mode | — | |
| `fchmod` | 91 | — | fd | mode | |
| `chown` | 92 | ✓ | user | group | |
| `fchown` | 93 | — | fd | user | |
| **Process** |
| `execve` | 59 | ✓ | — | — | Program path; `return_value` = -errno on failure |
| `fork` | 57 | — | — | — | Process creation; `return_value` = child pid or -errno |
| **Network** |
| `socket` | 41 | — | family | type | `return_value` = fd or -errno |
| `connect` | 42 | — | fd | addrlen | `return_value` = 0 or -errno |
| `bind` | 49 | — | fd | addrlen | `return_value` = 0 or -errno |
| `listen` | 50 | — | fd | backlog | `return_value` = 0 or -errno |
| `accept` | 43 | — | fd | — | addrlen is pointer; `return_value` = fd or -errno |
| `accept4` | 288 | — | fd | flags | `return_value` = fd or -errno |

Other event names in `probe/events.py` (e.g. `stat`, `access`) may be used in rules but are not traced by dedicated tracepoints; they would only appear if routed through the generic raw syscall path, with minimal data (arg0/arg1 only, no path).

**Source:** `probe/events.py` (registry), `probe/probe.bpf.c` (tracepoints).

## Source of truth

- Chart deployments read rules from `charts/sentinel-ebpf/rules.yaml`.
- Rules are mounted into the probe and detector at `/etc/sentinel-ebpf/rules.yaml`.
- Rules use a hybrid format: explicit `rules[].syscalls`, optional `lists` and `macros`, optional `condition`, and optional `groups` metadata.

## Rule model

Each rule has:

- `group`: the downstream category attached to matching events (written into `EventEnvelope.event_group`)
- `syscalls`: the required syscall set for that rule; this is the source of selective probe attachment
- `condition`: optional extra filter expressed in the DSL
- `lists` / `macros`: optional helpers for reuse and readability
- `groups`: optional metadata keyed by `group` (for example detector feature config)

The `condition` part is a boolean expression over event fields.

- **Supported operators:** `=`, `in`, `startswith`, `contains`, `and`, `or`, `not`, parentheses
- **Supported fields:** `event_name`, `event_id`, `path`, `comm`, `pid`, `tid`, `uid`, `flags`, `arg0`, `arg1`, `arg_flags`, `return_value`, `hostname`, `namespace`
- `arg_flags` is evaluated as an alias of `flags`.

## Selective probe attachment

At startup, the probe compiles the explicit `syscalls` from enabled rules and extracts the `event_id`s needed. Only tracepoints for those syscalls are attached (via preprocessor `#if ENABLE_*`). For example, if rules only declare `openat`, the read/write/execve probes are not attached, reducing kernel overhead.

## Execution model: kernel prefilter + userspace final decision

Rules run in two stages:

1. **Kernel prefilter (BPF tuple match):**
   - Event can be dropped early in kernel if it does not match compiled tuples.
   - Compilable predicate types are positive checks on:
     - `event_name` / `event_id`
     - `path startswith`
     - `comm`
     - numeric IDs (`pid`, `tid`, `uid`)

2. **Userspace fallback + final evaluation:**
   - Full DSL AST is evaluated in Python.
   - Anything not kernel-compilable stays here (for example `not ...`, `contains`, some string logic).
   - Event is forwarded only if full condition evaluation is true.

## Why `not noisy_path` is userspace fallback

Even if `noisy_path` itself only uses `path startswith`, `not noisy_path` introduces negation.
The current kernel tuple model only stores positive match dimensions, so negated predicates are not compiled.

## Why a few DSL rules can become many kernel tuples

Compiler output is not 1:1 with rule count. It expands conditions into DNF branches and then into tuple combinations.

Typical expansion pattern for the optional `condition`:

- `event_name in (...)` multiplies by number of events
- `path startswith A or path startswith B` splits branches
- `comm in (...)` multiplies by number of comm values

Example from current default chart rules:

- `capture-file-events`: explicit file syscall set, then extra path filters from the condition
- `capture-shell-process-events`: explicit process syscall set, then extra comm filter
- `capture-network-events`: explicit network syscall set, no extra condition

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

- Put syscall membership in `rules[].syscalls`, not hidden inside `condition`.
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
  sensitive_path: "path startswith /etc or path startswith /root"
  noisy_path: "path startswithin (noisy_paths)"

groups:
  file: {}

rules:
  - name: capture-sensitive-file-events
    enabled: true
    group: file
    syscalls: file_syscalls
    condition: "sensitive_path and not noisy_path"
```
