"""BPF program building: selective probe compilation based on rules."""

from pathlib import Path
from typing import Optional

from probe.rules import WILDCARD

# Linux syscall_nr -> preprocessor define for conditional compilation.
# read(0) and write(1) enriched via fd->path cache from open/openat/openat2/close.
SYSCALL_NR_TO_DEFINE = {
    0: "ENABLE_READ",
    1: "ENABLE_WRITE",
    2: "ENABLE_OPEN",
    3: "ENABLE_CLOSE",
    5: "ENABLE_FSTAT",
    7: "ENABLE_POLL",
    9: "ENABLE_MMAP",
    10: "ENABLE_MPROTECT",
    16: "ENABLE_IOCTL",
    41: "ENABLE_SOCKET",
    42: "ENABLE_CONNECT",
    43: "ENABLE_ACCEPT",
    44: "ENABLE_SENDTO",
    45: "ENABLE_RECVFROM",
    49: "ENABLE_BIND",
    50: "ENABLE_LISTEN",
    51: "ENABLE_GETSOCKNAME",
    52: "ENABLE_GETPEERNAME",
    54: "ENABLE_SETSOCKOPT",
    57: "ENABLE_FORK",
    59: "ENABLE_EXECVE",
    72: "ENABLE_FCNTL",
    82: "ENABLE_RENAME",
    87: "ENABLE_UNLINK",
    90: "ENABLE_CHMOD",
    91: "ENABLE_FCHMOD",
    92: "ENABLE_CHOWN",
    93: "ENABLE_FCHOWN",
    257: "ENABLE_OPENAT",
    263: "ENABLE_UNLINKAT",
    264: "ENABLE_RENAMEAT",
    273: "ENABLE_SET_ROBUST_LIST",
    288: "ENABLE_ACCEPT4",
    307: "ENABLE_SENDMMSG",
    437: "ENABLE_OPENAT2",
}


# fd_map needs open/openat/openat2/close to populate path cache for read/write and fd-backed syscalls.
# fork needed for pid_to_parent so a child can resolve paths for inherited fds.
FD_MAP_DEPS = {2, 3, 257, 437, 57}  # open, close, openat, openat2, fork

# Syscalls that fill envelope path from fd_to_path (same cache as read/write).
_SYSCALL_NRS_USING_FD_PATH = frozenset({0, 1, 5, 16, 72, 91, 93})


def enabled_syscall_nrs_from_rules(compiled: list) -> set[int]:
    """Extract syscall_nr values needed by rules; WILDCARD means enable all probes."""
    all_nrs = set(SYSCALL_NR_TO_DEFINE.keys())
    nrs: set[int] = set()
    for r in compiled:
        snr = r.get("syscall_nr")
        if snr == WILDCARD:
            return all_nrs
        nrs.add(snr)
    nrs = nrs if nrs else all_nrs
    if nrs & _SYSCALL_NRS_USING_FD_PATH:
        nrs |= FD_MAP_DEPS
    return nrs


def load_bpf_program() -> str:
    """Load BPF program from separate file."""
    bpffile = Path(__file__).parent / "probe.bpf.c"
    if not bpffile.exists():
        raise FileNotFoundError(f"BPF program file not found: {bpffile}")
    return bpffile.read_text(encoding="utf-8")


def build_bpf_program(
    ring_buffer_pages: int, enabled_syscall_nrs: Optional[set[int]] = None
) -> str:
    """Build BPF program with ring buffer pages and selective probe enables.

    enabled_syscall_nrs: Linux syscall numbers to attach probes for. If None, all probes enabled.
    """
    pages = max(1, int(ring_buffer_pages))
    program = load_bpf_program()
    program = program.replace("__RINGBUF_PAGES__", str(pages))

    if enabled_syscall_nrs is None:
        enabled_syscall_nrs = set(SYSCALL_NR_TO_DEFINE.keys())
    defines = []
    for snr, define in sorted(SYSCALL_NR_TO_DEFINE.items()):
        val = 1 if snr in enabled_syscall_nrs else 0
        defines.append(f"#define {define} {val}")
    program = "\n".join(defines) + "\n\n" + program
    return program
