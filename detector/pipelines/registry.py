"""Register named pipelines (builder returns ``final_bb``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from detector.building_blocks.core.base import BuildingBlock
from detector.pipelines.sequence_frequency_fusion import build_sequence_frequency_fusion
from detector.pipelines.sequence_mlp import build_sequence_mlp
from detector.pipelines.single_model import build_single_model

if TYPE_CHECKING:
  from detector.config import DetectorConfig

log = logging.getLogger(__name__)

_PIPELINE_BUILDERS: dict[str, Callable[["DetectorConfig"], BuildingBlock]] = {}


def register_pipeline(pipeline_id: str, builder: Callable[["DetectorConfig"], BuildingBlock]) -> None:
  pid = (pipeline_id or "").strip()
  if not pid or pid.lower() == "legacy":
    raise ValueError("pipeline_id must be non-empty and not 'legacy'")
  _PIPELINE_BUILDERS[pid] = builder
  log.debug("Registered pipeline_id=%r", pid)


def build_final_bb(cfg: "DetectorConfig") -> BuildingBlock:
  pid = (getattr(cfg, "pipeline_id", "") or "").strip()
  if not pid or pid.lower() == "legacy":
    raise ValueError("build_final_bb requires cfg.pipeline_id != 'legacy'")
  fn = _PIPELINE_BUILDERS.get(pid)
  if fn is None:
    raise ValueError("Unknown pipeline_id=%r (registered: %s)" % (pid, sorted(_PIPELINE_BUILDERS)))
  return fn(cfg)


def registered_pipeline_ids() -> tuple[str, ...]:
  return tuple(sorted(_PIPELINE_BUILDERS))


def _build_single_model(cfg: "DetectorConfig") -> BuildingBlock:
  algorithm = str(getattr(cfg, "model_algorithm", "")).strip().lower()
  if algorithm == "sequence_mlp":
    return build_sequence_mlp(cfg)
  return build_single_model(cfg, algorithm=algorithm)


def _register_builtin_pipelines() -> None:
  register_pipeline("single_model", _build_single_model)
  register_pipeline("sequence_mlp", build_sequence_mlp)
  register_pipeline("las_gas_fusion", build_sequence_frequency_fusion)


_register_builtin_pipelines()
