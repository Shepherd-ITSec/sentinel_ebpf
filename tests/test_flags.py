import os
import socket

from probe.flags import SUPPORTED_FLAG_EVENT_IDS, decode_flags


def test_decode_flags_access_modes_are_exclusive():
  assert decode_flags(os.O_RDONLY, 257) == "O_RDONLY"
  assert decode_flags(os.O_WRONLY, 257) == "O_WRONLY"
  assert decode_flags(os.O_RDWR, 257) == "O_RDWR"


def test_decode_flags_includes_modifier_bits():
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
  decoded = decode_flags(flags, 257)
  assert "O_WRONLY" in decoded
  assert "O_CREAT" in decoded
  assert "O_TRUNC" in decoded


def test_decode_flags_socket_types():
  assert decode_flags(1, 41) == "SOCK_STREAM"
  assert decode_flags(2, 41) == "SOCK_DGRAM"
  assert decode_flags(3, 41) == "SOCK_RAW"


def test_decode_flags_socket_with_modifiers():
  flags = socket.SOCK_STREAM | socket.SOCK_CLOEXEC | socket.SOCK_NONBLOCK
  decoded = decode_flags(flags, 41)
  assert "SOCK_STREAM" in decoded
  assert "SOCK_CLOEXEC" in decoded
  assert "SOCK_NONBLOCK" in decoded


def test_decode_flags_returns_none_for_unsupported_event_id():
  assert decode_flags(0, 0) is None
  assert decode_flags(123, 42) is None
  assert decode_flags(os.O_RDONLY, 99) is None


def test_supported_flag_event_ids():
  assert 2 in SUPPORTED_FLAG_EVENT_IDS
  assert 257 in SUPPORTED_FLAG_EVENT_IDS
  assert 41 in SUPPORTED_FLAG_EVENT_IDS
