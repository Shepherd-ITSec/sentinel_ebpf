from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from detector.building_blocks.primitives.embeddings.word2vec import OnlineGensimWord2Vec
from detector.building_blocks.primitives.features.generic import _extract_generic_features
from detector.building_blocks.primitives.features.groups.file import _extract_file_group_features
from detector.building_blocks.primitives.features.groups.network import _extract_network_group_features
from detector.building_blocks.primitives.features.views import _FeatureViewSpec, _SEQUENCE_CONTEXT_PREFIX, _feature_view_spec
from detector.building_blocks.primitives.sequence.sequence_context import SequenceFeatureMeta, SequenceVectorContext, TokenClassTable
from detector.config import load_config


class FeatureExtractor:
  def __init__(
    self,
    *,
    sequence_w2v: OnlineGensimWord2Vec,
    sequence_classes: TokenClassTable,
    sequence_context: SequenceVectorContext,
  ) -> None:
    self._sequence_w2v = sequence_w2v
    self._sequence_classes = sequence_classes
    self._sequence_context = sequence_context

  @classmethod
  def from_config(cls, cfg: Any) -> "FeatureExtractor":
    emb_dim = int(getattr(cfg, "embedding_word2vec_dim"))
    return cls(
      sequence_w2v=OnlineGensimWord2Vec(
        vector_size=emb_dim,
        sentence_len=int(getattr(cfg, "embedding_word2vec_sentence_len")),
        seed=int(getattr(cfg, "model_seed")),
        w2v_window=int(getattr(cfg, "embedding_word2vec_window")),
        w2v_sg=int(getattr(cfg, "embedding_word2vec_sg")),
        update_every=int(getattr(cfg, "embedding_word2vec_update_every")),
        epochs=int(getattr(cfg, "embedding_word2vec_epochs")),
        warmup_events=int(getattr(cfg, "warmup_events")),
        post_warmup_lr_scale=float(getattr(cfg, "embedding_word2vec_post_warmup_lr_scale")),
      ),
      sequence_classes=TokenClassTable(),
      sequence_context=SequenceVectorContext(
        element_width=emb_dim,
        ngram_length=int(getattr(cfg, "sequence_ngram_length")),
        thread_aware=bool(getattr(cfg, "sequence_thread_aware")),
        feature_prefix=_SEQUENCE_CONTEXT_PREFIX,
      ),
    )

  def _extract_sequence_features(self, evt: Any) -> tuple[Dict[str, float], SequenceFeatureMeta]:
    try:
      actor_id = int((getattr(evt, "tid", "0")).strip())
    except ValueError:
      actor_id = 0
    token = (getattr(evt, "syscall_name", "")).strip().lower()
    target_id = self._sequence_classes.get_label_idx(token)
    w2v_emb = self._sequence_w2v.observe_and_vector(int(actor_id), token)
    out, meta = self._sequence_context.observe_vector(
      stream_id=int(actor_id),
      vector=w2v_emb.tolist(),
      target_id=int(target_id),
      num_classes=int(self._sequence_classes.num_classes),
    )
    return out, meta

  def extract_feature_dict(self, evt: Any, feature_view: str = "default") -> tuple[Dict[str, float], SequenceFeatureMeta | None]:
    meta: SequenceFeatureMeta | None = None
    view = _feature_view_spec(feature_view or "default")
    out = _extract_generic_features(evt, view)
    if view.include_sequence:
      seq, meta = self._extract_sequence_features(evt)
      out.update(seq)
    event_group = (evt.event_group or "").strip().lower() or "__empty__"
    out.update(_extract_group_features(evt, view, event_group))
    return out, meta

  def get_state(self) -> dict:
    return {
      "sequence_context": {
        "classes": self._sequence_classes.to_serializable(),
        "w2v": self._sequence_w2v.get_state(),
        "context": self._sequence_context.get_state(),
      },
    }

  def set_state(self, state: dict) -> None:
    seq_state = state.get("sequence_context", None)
    if isinstance(seq_state, dict):
      self._sequence_classes.load_from_pairs(list(seq_state.get("classes", []) or []))
      w2v_state = seq_state.get("w2v", None)
      if isinstance(w2v_state, dict):
        self._sequence_w2v.set_state(w2v_state)
      ctx_state = seq_state.get("context", None)
      if isinstance(ctx_state, dict):
        self._sequence_context.set_state(ctx_state)


def _extract_group_features(evt: Any, view: _FeatureViewSpec, event_group: str) -> Dict[str, float]:
  group = (event_group or "").strip().lower()
  out: Dict[str, float] = {}
  if group == "network":
    out.update(_extract_network_group_features(evt, view))
  elif group == "file":
    out.update(_extract_file_group_features(evt, view, event_group))
  return out


def build_feature_extractor(cfg: Any | None = None) -> FeatureExtractor:
  if cfg is None:
    cfg = load_config()
  return FeatureExtractor.from_config(cfg)


@lru_cache(maxsize=1)
def _default_feature_extractor() -> FeatureExtractor:
  return build_feature_extractor()


def extract_feature_dict(evt: Any, feature_view: str = "default") -> tuple[Dict[str, float], SequenceFeatureMeta | None]:
  return _default_feature_extractor().extract_feature_dict(evt, feature_view=feature_view)
