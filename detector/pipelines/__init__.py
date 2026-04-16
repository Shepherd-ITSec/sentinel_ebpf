"""Pipeline assembly recipes built from building blocks."""

from detector.pipelines.registry import build_final_bb, register_pipeline, registered_pipeline_ids

__all__ = [
  "build_final_bb",
  "register_pipeline",
  "registered_pipeline_ids",
]
