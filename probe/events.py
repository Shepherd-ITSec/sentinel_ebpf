"""Curated syscall/event registry shared by probe components."""

from typing import Dict

# Curated events captured with enriched args/path.
# IDs follow Linux x86_64 syscall numbers where applicable.
EVENT_NAME_TO_ID: Dict[str, int] = {
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
  # BETH/LSM-style event id (not a syscall number).
  "cap_capable": 1003,
  "security_bprm_check": 1004,
  "security_file_open": 1005,
  "security_inode_unlink": 1006,
  "mem_prot_alert": 1009,
  "sched_process_exit": 1010,
  # Placeholder for rows with missing event names.
  "unknown": 0,
}

EVENT_ID_TO_NAME: Dict[int, str] = {v: k for k, v in EVENT_NAME_TO_ID.items()}

