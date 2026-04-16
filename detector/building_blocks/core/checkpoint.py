"""Pickle checkpoint for composed pipelines (per ``block_uid``)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from detector.building_blocks.core.base import BuildingBlock
from detector.building_blocks.core.manager import BuildingBlockManager

log = logging.getLogger(__name__)


def save_pipeline_checkpoint(
  path: Path,
  *,
  pipeline_id: str,
  manager: BuildingBlockManager,
  checkpoint_index: int,
  extra: dict[str, Any] | None = None,
) -> None:
  blocks: dict[str, dict[str, Any]] = {}
  for bb in manager.all_blocks():
    st = bb.get_state()
    if st:
      blocks[bb.block_uid] = st
  payload = {
    "format": "building_blocks_v1",
    "pipeline_id": pipeline_id,
    "checkpoint_index": int(checkpoint_index),
    "blocks": blocks,
    "extra": extra or {},
  }
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
  log.info("Saved pipeline checkpoint index=%d to %s", checkpoint_index, path)


def load_pipeline_checkpoint(path: Path, manager: BuildingBlockManager) -> tuple[int, dict[str, Any]]:
  path = Path(path)
  with path.open("rb") as f:
    payload = pickle.load(f)
  if payload.get("format") != "building_blocks_v1":
    raise ValueError("Unknown checkpoint format: %r" % payload.get("format"))
  blocks = payload.get("blocks") or {}
  uid_to_bb: dict[str, BuildingBlock] = {bb.block_uid: bb for bb in manager.all_blocks()}
  for uid, st in blocks.items():
    bb = uid_to_bb.get(uid)
    if bb is None:
      log.warning("Checkpoint has unknown block_uid=%s; skipping", uid)
      continue
    if isinstance(st, dict):
      bb.set_state(st)
  idx = int(payload.get("checkpoint_index", 0))
  extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
  log.info("Loaded pipeline checkpoint from %s index=%d", path, idx)
  return idx, extra

