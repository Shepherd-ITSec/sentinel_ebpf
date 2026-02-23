import os


def decode_open_flags(flags: int) -> str:
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
