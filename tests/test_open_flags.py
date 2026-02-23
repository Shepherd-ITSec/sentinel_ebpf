import os

from probe.open_flags import decode_open_flags


def test_decode_open_flags_access_modes_are_exclusive():
  assert decode_open_flags(os.O_RDONLY) == "O_RDONLY"
  assert decode_open_flags(os.O_WRONLY) == "O_WRONLY"
  assert decode_open_flags(os.O_RDWR) == "O_RDWR"


def test_decode_open_flags_includes_modifier_bits():
  flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
  decoded = decode_open_flags(flags)
  assert "O_WRONLY" in decoded
  assert "O_CREAT" in decoded
  assert "O_TRUNC" in decoded
