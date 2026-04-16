from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _FeatureViewSpec:
  include_sequence: bool = False
  include_hashes: bool = False
  include_general_process_context: bool = True
  include_general_time_context: bool = True
  include_general_path_context: bool = True
  include_general_return_context: bool = True
  include_file_sensitive_tmp: bool = True
  include_file_flags: bool = True
  include_network_ports: bool = True
  include_network_socket_type: bool = True
  time_feature_mode: str = "day_cycle"


_SEQUENCE_CONTEXT_PREFIX = "sequence_ctx"
_FULL_FEATURE_VIEW = _FeatureViewSpec()
_DEFAULT_FEATURE_VIEW = _FeatureViewSpec(
  include_general_process_context=False,
  include_general_time_context=False,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_flags=False,
  include_network_ports=False,
)
_FREQUENCY_FEATURE_VIEW = _FeatureViewSpec(
  include_hashes=True,
  include_general_process_context=False,
  include_general_time_context=False,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_sensitive_tmp=False,
  include_file_flags=False,
  include_network_ports=False,
  include_network_socket_type=False,
  time_feature_mode="day_fraction",
)
_SEQUENCE_FEATURE_VIEW = _FeatureViewSpec(
  include_sequence=True,
  include_hashes=False,
  include_general_process_context=False,
  include_general_time_context=True,
  include_general_path_context=False,
  include_general_return_context=False,
  include_file_sensitive_tmp=False,
  include_file_flags=False,
  include_network_ports=False,
  include_network_socket_type=False,
  time_feature_mode="day_cycle",
)


def feature_view_for_algorithm(algorithm: str | None) -> str:
  algo = (algorithm or "").strip().lower()
  if algo in ("sequence_mlp", "sequence_transformer"):
    return "sequence"
  if algo in ("freq1d", "copulatree", "latentcluster", "zscore"):
    return "frequency"
  return "default"


def _feature_view_spec(feature_view: str | None) -> _FeatureViewSpec:
  normalized = (feature_view or "default").strip().lower()
  if normalized == "sequence":
    return _SEQUENCE_FEATURE_VIEW
  if normalized == "frequency":
    return _FREQUENCY_FEATURE_VIEW
  if normalized in ("full", "memstream"):
    return _FULL_FEATURE_VIEW
  return _DEFAULT_FEATURE_VIEW
