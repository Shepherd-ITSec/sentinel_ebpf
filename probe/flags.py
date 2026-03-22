import os
import socket
from functools import lru_cache
from typing import Optional

# Event IDs that have a flags/type field we can decode. open, openat, openat2, socket.
SUPPORTED_FLAG_EVENT_IDS = (2, 257, 437, 41)


def decode_flags(flags: int, event_id: int) -> Optional[str]:
  """Decode flags/type into a compact string. Returns None if event_id is not supported."""
  if event_id not in SUPPORTED_FLAG_EVENT_IDS:
    return None
  if event_id == 41:
    return _decode_socket_type(flags)
  return _decode_open_flags(flags)


@lru_cache(maxsize=256)
def _decode_open_flags(flags: int) -> str:
  """Decode Linux open(2) flags into a compact string."""
  flag_names = []

  # Access mode is encoded in a masked field, not independent bits.
  access_mode = flags & os.O_ACCMODE
  if access_mode == os.O_RDONLY:
    flag_names.append("O_RDONLY")
  elif access_mode == os.O_WRONLY:
    flag_names.append("O_WRONLY")
  elif access_mode == os.O_RDWR:
    flag_names.append("O_RDWR")

  for name in [
    "O_CREAT",
    "O_TRUNC",
    "O_APPEND",
    "O_CLOEXEC",
    "O_EXCL",
    "O_DIRECTORY",
    "O_NOFOLLOW",
    "O_SYNC",
    "O_DSYNC",
  ]:
    if hasattr(os, name) and (flags & getattr(os, name)):
      flag_names.append(name)

  return "|".join(flag_names) if flag_names else "0"


_SOCKET_TYPE_NAMES = {
  1: "SOCK_STREAM",
  2: "SOCK_DGRAM",
  3: "SOCK_RAW",
  4: "SOCK_RDM",
  5: "SOCK_SEQPACKET",
  6: "SOCK_DCCP",
  10: "SOCK_PACKET",
}


@lru_cache(maxsize=256)
def _decode_socket_type(flags: int) -> str:
  """Decode Linux socket(2) type into a compact string."""
  parts = []
  base_type = flags & 0xFF
  if base_type in _SOCKET_TYPE_NAMES:
    parts.append(_SOCKET_TYPE_NAMES[base_type])
  else:
    parts.append(str(base_type))
  if flags & getattr(socket, "SOCK_CLOEXEC", 0x80000):
    parts.append("SOCK_CLOEXEC")
  if flags & getattr(socket, "SOCK_NONBLOCK", 0x800):
    parts.append("SOCK_NONBLOCK")
  return "|".join(parts)
