from __future__ import annotations

GENERIC_ALWAYS_FEATURES: tuple[str, ...] = ("syscall_nr_norm",)
GENERAL_PROCESS_FEATURES: tuple[str, ...] = (
  "pid_norm",
  "tid_norm",
  "uid_norm",
  "arg0_norm",
  "arg1_norm",
)
GENERAL_DAY_CYCLE_FEATURES: tuple[str, ...] = ("day_cycle_sin", "day_cycle_cos")
GENERAL_DAY_FRACTION_FEATURES: tuple[str, ...] = ("day_fraction_norm",)
GENERAL_TIME_BUCKET_FEATURES: tuple[str, ...] = ("week_of_month_norm",)
GENERAL_PATH_FEATURES: tuple[str, ...] = ("path_depth_norm", "path_prefix_hash")
GENERAL_RETURN_FEATURES: tuple[str, ...] = ("return_success", "return_errno_norm")
GENERAL_HASH_FEATURES: tuple[str, ...] = ("hostname_hash", "pid_hash", "path_hash", "path_prefix_hash")
FILE_FEATURES: tuple[str, ...] = ("file_sensitive_path", "file_tmp_path", "file_flags_hash")
NETWORK_NUMERIC_FEATURES: tuple[str, ...] = ("net_socket_family_norm", "net_dport_norm")
NETWORK_HASH_FEATURES: tuple[str, ...] = ("net_socket_type_hash", "net_daddr_hash", "net_af_hash")


def sequence_feature_names() -> tuple[str, ...]:
  return ("sequence_ctx_*",)


def default_tabular_feature_names() -> tuple[str, ...]:
  return GENERIC_ALWAYS_FEATURES + GENERAL_DAY_CYCLE_FEATURES


def frequency_feature_names() -> tuple[str, ...]:
  return (
    GENERIC_ALWAYS_FEATURES
    + GENERAL_DAY_FRACTION_FEATURES
    + GENERAL_HASH_FEATURES
    + NETWORK_HASH_FEATURES
  )


def full_tabular_feature_names() -> tuple[str, ...]:
  return (
    GENERIC_ALWAYS_FEATURES
    + GENERAL_PROCESS_FEATURES
    + GENERAL_DAY_CYCLE_FEATURES
    + GENERAL_TIME_BUCKET_FEATURES
    + GENERAL_PATH_FEATURES
    + GENERAL_RETURN_FEATURES
    + FILE_FEATURES
    + NETWORK_NUMERIC_FEATURES
  )


def feature_names_for_algorithm(cfg: object, algorithm: str | None) -> tuple[str, ...]:
  algo = (algorithm or "").strip().lower()
  if algo in ("sequence_mlp", "sequence_transformer"):
    return sequence_feature_names()
  if algo in ("freq1d", "copulatree", "latentcluster", "zscore"):
    return frequency_feature_names()
  return default_tabular_feature_names()
