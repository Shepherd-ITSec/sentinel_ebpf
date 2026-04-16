from __future__ import annotations

from typing import Any, Dict

from detector.building_blocks.primitives.features.generic import (
  _PORT_MAX,
  _SOCKET_FAMILY_MAX,
  _hash01,
  _parse_sockaddr_from_evt,
  _safe_int,
)
from detector.building_blocks.primitives.features.views import _FeatureViewSpec


def _extract_network_group_features(evt: Any, view: _FeatureViewSpec) -> Dict[str, float]:
  out: Dict[str, float] = {}
  arg0_val = _safe_int(evt.arg0 or "0", default=0)
  arg1_val = _safe_int(evt.arg1 or "0", default=0)
  syscall_display = evt.syscall_name or ""
  sockaddr = _parse_sockaddr_from_evt(evt)
  if view.include_network_socket_type:
    family_val = 0
    if syscall_display == "socket":
      family_val = arg0_val
    else:
      af = (sockaddr.get("sa_family") or "").upper()
      if "INET6" in af or af == "10":
        family_val = 10
      elif "INET" in af or af == "2":
        family_val = 2
    out["net_socket_family_norm"] = min(family_val / _SOCKET_FAMILY_MAX, 1.0)
  type_val = arg1_val if syscall_display == "socket" else 0
  port_str = sockaddr.get("sin_port") or ""
  dport = _safe_int(port_str, default=0) if port_str else 0
  if view.include_network_ports:
    out["net_dport_norm"] = min(dport / _PORT_MAX, 1.0) if dport else 0.0
  daddr = sockaddr.get("sin_addr") or ""
  if view.include_hashes:
    out["net_socket_type_hash"] = _hash01(str(type_val))
    out["net_daddr_hash"] = _hash01(daddr)
    af_str = sockaddr.get("sa_family") or ("AF_INET" if family_val == 2 else "AF_INET6" if family_val == 10 else "")
    out["net_af_hash"] = _hash01(af_str)
  return out
