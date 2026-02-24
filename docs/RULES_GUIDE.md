# Rules Guide (Falco-like DSL)

This guide explains how Sentinel eBPF rules are authored and how they execute across kernel and userspace.

## Source of truth

- Chart deployments read rules from `charts/sentinel-ebpf/rules.yaml`.
- Rules are mounted into the probe at `/etc/sentinel-ebpf/rules.yaml`.
- Rules are DSL-only (`lists`, `macros`, `rules[].condition`).

## DSL model

Rules are boolean expressions over event fields.

- **Supported operators:** `=`, `in`, `startswith`, `contains`, `and`, `or`, `not`, parentheses
- **Supported fields:** `event_name`, `event_id`, `path`, `comm`, `pid`, `tid`, `uid`, `open_flags`, `arg0`, `arg1`, `arg_flags`, `return_value`, `hostname`, `namespace`
- `arg_flags` is evaluated as an alias of `open_flags`.

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

Typical expansion pattern:

- `event_name in (...)` multiplies by number of events
- `path startswith A or path startswith B` splits branches
- `comm in (...)` multiplies by number of comm values

Example from current default chart rules:

- `capture-sensitive-file-events`: `9 event_names * 2 prefixes = 18 tuples`
- `capture-shell-execve`: `1 event * 3 comms = 3 tuples`
- `capture-network-connectivity`: `2 events = 2 tuples`
- **Total compiled tuples: 23**

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

- Prefer reusable macros for readability and Falco-like style.
- Keep expressions selective early (`event_name` + specific prefixes/comms).
- Use `not ...` when needed for clarity, but expect userspace fallback for that part.
- Watch tuple growth when adding wide `in (...)` lists combined with `or` branches.
- Check metrics after rule changes to detect truncation or high fallback share.

## Minimal pattern (recommended style)

```yaml
lists:
  file_events: [open, openat, openat2]

macros:
  file_evt: "event_name in (file_events)"
  sensitive_path: "path startswith /etc or path startswith /root"
  noisy_path: "path startswith /proc or path startswith /sys"

rules:
  - name: capture-sensitive-file-events
    enabled: true
    condition: "file_evt and sensitive_path and not noisy_path"
```
