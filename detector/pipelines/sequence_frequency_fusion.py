from __future__ import annotations

from typing import TYPE_CHECKING

from detector.building_blocks.blocks.fusion import FusionLinearBlock
from detector.building_blocks.blocks.model_blocks import DictFeatureModelBlock
from detector.building_blocks.blocks.scoring import PrimaryScoreBlock, ThresholdDecisionBlock
from detector.building_blocks.blocks.sequence import SequenceNextTokenMLPBlock
from detector.building_blocks.blocks.tabular import FeatureDictExtractorBlock
from detector.building_blocks.core.base import BuildingBlock
from detector.building_blocks.primitives.features.extractor import build_feature_extractor
from detector.pipelines.feature_sets import frequency_feature_names, sequence_feature_names

if TYPE_CHECKING:
  from detector.config import DetectorConfig


def build_sequence_frequency_fusion(cfg: "DetectorConfig") -> BuildingBlock:
  seq_extractor = build_feature_extractor(cfg)
  seq = FeatureDictExtractorBlock(seq_extractor, sequence_feature_names(), block_uid="pipeline:las_gas:sequence_dict")
  las = SequenceNextTokenMLPBlock(seq, cfg, block_uid="pipeline:las_gas:las_mlp")

  extractor = build_feature_extractor(cfg)
  fe = FeatureDictExtractorBlock(extractor, frequency_feature_names(), block_uid="pipeline:las_gas:freq_dict")
  gas = DictFeatureModelBlock(fe, cfg, algorithm="zscore", block_uid="pipeline:las_gas:gas_z")

  fusion = FusionLinearBlock([las, gas], seed=int(cfg.model_seed), block_uid="pipeline:las_gas:fusion")
  primary = PrimaryScoreBlock(fusion, cfg, block_uid="pipeline:las_gas:primary_score")
  return ThresholdDecisionBlock(primary, float(cfg.threshold), block_uid="pipeline:las_gas:decision")
