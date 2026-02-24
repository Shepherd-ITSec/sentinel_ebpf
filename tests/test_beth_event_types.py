"""Regression tests for BETH event type compatibility."""

from probe.events import EVENT_NAME_TO_ID


# Event name -> eventId mappings observed in test_data/beth/*.csv.
BETH_EVENT_NAME_TO_ID = {
  "accept": 43,
  "accept4": 288,
  "access": 21,
  "bind": 49,
  "bpf": 321,
  "cap_capable": 1003,
  "chmod": 90,
  "chown": 92,
  "clone": 56,
  "close": 3,
  "connect": 42,
  "dup": 32,
  "dup2": 33,
  "dup3": 292,
  "execve": 59,
  "faccessat": 269,
  "fchmod": 91,
  "fchmodat": 268,
  "fchownat": 260,
  "fstat": 5,
  "getdents64": 217,
  "getsockname": 51,
  "kill": 62,
  "lchown": 94,
  "listen": 50,
  "lstat": 6,
  "mem_prot_alert": 1009,
  "memfd_create": 319,
  "mknod": 133,
  "mount": 165,
  "open": 2,
  "openat": 257,
  "prctl": 157,
  "sched_process_exit": 1010,
  "security_bprm_check": 1004,
  "security_file_open": 1005,
  "security_inode_unlink": 1006,
  "setfsgid": 123,
  "setfsuid": 122,
  "setgid": 106,
  "setregid": 114,
  "setreuid": 113,
  "setuid": 105,
  "socket": 41,
  "stat": 4,
  "symlink": 88,
  "umount": 166,
  "unknown": 0,
  "unlink": 87,
  "unlinkat": 263,
}


def test_registry_covers_all_beth_event_types():
  assert BETH_EVENT_NAME_TO_ID.keys() <= EVENT_NAME_TO_ID.keys()


def test_registry_uses_expected_beth_event_ids():
  for name, expected_id in BETH_EVENT_NAME_TO_ID.items():
    assert EVENT_NAME_TO_ID.get(name) == expected_id
