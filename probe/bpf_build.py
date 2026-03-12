"""BPF program building: selective probe compilation based on rules."""

from pathlib import Path
from typing import Optional

from probe.rules import WILDCARD

# event_id -> preprocessor define for conditional compilation.
# read(0) and write(1) enriched via fd->path cache from open/openat/openat2/close.
EVENT_ID_TO_DEFINE = {
    0: "ENABLE_READ",
    1: "ENABLE_WRITE",
    2: "ENABLE_OPEN",
    3: "ENABLE_CLOSE",
    41: "ENABLE_SOCKET",
    42: "ENABLE_CONNECT",
    43: "ENABLE_ACCEPT",
    49: "ENABLE_BIND",
    50: "ENABLE_LISTEN",
    57: "ENABLE_FORK",
    59: "ENABLE_EXECVE",
    82: "ENABLE_RENAME",
    87: "ENABLE_UNLINK",
    90: "ENABLE_CHMOD",
    91: "ENABLE_FCHMOD",
    92: "ENABLE_CHOWN",
    93: "ENABLE_FCHOWN",
    257: "ENABLE_OPENAT",
    263: "ENABLE_UNLINKAT",
    264: "ENABLE_RENAMEAT",
    288: "ENABLE_ACCEPT4",
    437: "ENABLE_OPENAT2",
}


# fd_map needs open/openat/openat2/close to populate path cache for read/write.
# fork needed for pid_to_parent so child's read/write can resolve path from inherited fds.
FD_MAP_DEPS = {2, 3, 257, 437, 57}  # open, close, openat, openat2, fork


def enabled_event_ids_from_rules(compiled: list) -> set[int]:
    """Extract event_ids needed by rules; WILDCARD means enable all probes."""
    all_ids = set(EVENT_ID_TO_DEFINE.keys())
    ids: set[int] = set()
    for r in compiled:
        eid = r.get("event_id")
        if eid == WILDCARD:
            return all_ids
        ids.add(eid)
    ids = ids if ids else all_ids
    # When read or write is enabled, fd_map needs open/openat/openat2/close to populate path.
    if 0 in ids or 1 in ids:
        ids |= FD_MAP_DEPS
    return ids


def load_bpf_program() -> str:
    """Load BPF program from separate file."""
    bpffile = Path(__file__).parent / "probe.bpf.c"
    if not bpffile.exists():
        raise FileNotFoundError(f"BPF program file not found: {bpffile}")
    return bpffile.read_text(encoding="utf-8")


def build_bpf_program(
    ring_buffer_pages: int, enabled_event_ids: Optional[set[int]] = None
) -> str:
    """Build BPF program with ring buffer pages and selective probe enables.

    enabled_event_ids: set of event_ids to attach probes for. If None, all probes enabled.
    """
    pages = max(1, int(ring_buffer_pages))
    program = load_bpf_program()
    program = program.replace("__RINGBUF_PAGES__", str(pages))

    if enabled_event_ids is None:
        enabled_event_ids = set(EVENT_ID_TO_DEFINE.keys())
    defines = []
    for eid, define in sorted(EVENT_ID_TO_DEFINE.items()):
        val = 1 if eid in enabled_event_ids else 0
        defines.append(f"#define {define} {val}")
    program = "\n".join(defines) + "\n\n" + program
    return program
