from __future__ import annotations

from typing import TYPE_CHECKING

from detector.building_blocks.blocks.model_blocks import DictFeatureModelBlock
from detector.building_blocks.blocks.scoring import PrimaryScoreBlock, ThresholdDecisionBlock
from detector.building_blocks.blocks.tabular import FeatureDictExtractorBlock
from detector.building_blocks.core.base import BuildingBlock
from detector.building_blocks.primitives.features.extractor import build_feature_extractor
from detector.pipelines.feature_sets import feature_names_for_algorithm

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def build_single_model(cfg: "DetectorConfig", *, algorithm: str | None = None) -> BuildingBlock:
  algo = (algorithm if algorithm is not None else getattr(cfg, "model_algorithm", "")).strip().lower()
  requested_features = feature_names_for_algorithm(cfg, algo)
  extractor = build_feature_extractor(cfg)
  feats = FeatureDictExtractorBlock(extractor, requested_features, block_uid=f"pipeline:single_model:{algo}:features")
  model = DictFeatureModelBlock(feats, cfg, algorithm=algo, block_uid=f"pipeline:single_model:{algo}:model")
  primary = PrimaryScoreBlock(model, cfg, block_uid=f"pipeline:single_model:{algo}:primary_score")
  return ThresholdDecisionBlock(primary, float(cfg.threshold), block_uid=f"pipeline:single_model:{algo}:decision")
