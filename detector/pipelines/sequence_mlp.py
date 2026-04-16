from __future__ import annotations

from typing import TYPE_CHECKING

from detector.building_blocks.blocks.scoring import PrimaryScoreBlock, ThresholdDecisionBlock
from detector.building_blocks.blocks.sequence import SequenceNextTokenMLPBlock
from detector.building_blocks.blocks.tabular import FeatureDictExtractorBlock
from detector.building_blocks.core.base import BuildingBlock
from detector.building_blocks.primitives.features.extractor import build_feature_extractor
from detector.pipelines.feature_sets import sequence_feature_names

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def build_sequence_mlp(cfg: "DetectorConfig") -> BuildingBlock:
  extractor = build_feature_extractor(cfg)
  seq = FeatureDictExtractorBlock(extractor, sequence_feature_names(), block_uid="pipeline:sequence_mlp:sequence_dict")
  model = SequenceNextTokenMLPBlock(seq, cfg, block_uid="pipeline:sequence_mlp:model")
  primary = PrimaryScoreBlock(model, cfg, block_uid="pipeline:sequence_mlp:primary_score")
  return ThresholdDecisionBlock(primary, float(cfg.threshold), block_uid="pipeline:sequence_mlp:decision")
