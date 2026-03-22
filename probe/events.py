"""Curated syscall/event registry shared by probe components."""

from typing import Dict

# Placeholder for rows with missing or unrecognized event names (not a syscall number).
UNKNOWN_EVENT_ID = 9999

# Curated events captured with enriched args/path.
# IDs follow Linux x86_64 syscall numbers where applicable.
EVENT_NAME_TO_ID: Dict[str, int] = {
  "read": 0,
  "write": 1,
  "open": 2,
  "close": 3,
  "stat": 4,
  "fstat": 5,
  "lstat": 6,
  "access": 21,
  "dup": 32,
  "dup2": 33,
  "socket": 41,
  "accept": 43,
  "bind": 49,
  "listen": 50,
  "getsockname": 51,
  "clone": 56,
  "fork": 57,
  "connect": 42,
  "execve": 59,
  "kill": 62,
  "rename": 82,
  "symlink": 88,
  "unlink": 87,
  "chmod": 90,
  "fchmod": 91,
  "chown": 92,
  "lchown": 94,
  "setuid": 105,
  "setgid": 106,
  "setreuid": 113,
  "setregid": 114,
  "setfsuid": 122,
  "setfsgid": 123,
  "prctl": 157,
  "mount": 165,
  "umount": 166,
  "getdents64": 217,
  "mknod": 133,
  "fchown": 93,
  "openat": 257,
  "fchownat": 260,
  "unlinkat": 263,
  "fchmodat": 268,
  "faccessat": 269,
  "renameat": 264,
  "accept4": 288,
  "dup3": 292,
  "memfd_create": 319,
  "bpf": 321,
  "openat2": 437,
  # LSM-style event id (not a syscall number).
  "cap_capable": 1003,
  "security_bprm_check": 1004,
  "security_file_open": 1005,
  "security_inode_unlink": 1006,
  "mem_prot_alert": 1009,
  "sched_process_exit": 1010,
  "unknown": UNKNOWN_EVENT_ID,
}

EVENT_ID_TO_NAME: Dict[int, str] = {v: k for k, v in EVENT_NAME_TO_ID.items()}

# Syscalls instrumented at sys_exit with meaningful return value (fd, errno, etc.).
EVENT_IDS_WITH_RETURN_VALUE: frozenset[int] = frozenset({
    0, 1, 2, 3, 41, 42, 43, 49, 50, 57, 59, 257, 288, 437,
})

