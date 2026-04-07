#!/usr/bin/env python3
"""
Visualize the online syscall Word2Vec by projecting embeddings to 2D.

Inputs:
- A detector log (JSONL/EVT1): replay + train the online Word2Vec, then plot.
- A detector checkpoint (.pkl): load the saved feature_state (incl. Word2Vec), then plot.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

try:
  import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
  plt = None

try:
  from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
  tqdm = None

from detector.config import load_config
from detector.sequence.context import SequenceContextFeatureExtractor
from scripts.replay_logs import _detect_format, iter_events, iter_events_jsonl

log = logging.getLogger(Path(__file__).stem)


def _make_extractor() -> SequenceContextFeatureExtractor:
  cfg = load_config()
  return SequenceContextFeatureExtractor(
    vector_size=int(getattr(cfg, "embedding_word2vec_dim", 5)),
    sentence_len=int(getattr(cfg, "embedding_word2vec_sentence_len", 7)),
    seed=int(getattr(cfg, "model_seed", 42)),
    w2v_window=int(getattr(cfg, "embedding_word2vec_window", 5)),
    w2v_sg=int(getattr(cfg, "embedding_word2vec_sg", 1)),
    update_every=int(getattr(cfg, "embedding_word2vec_update_every", 25)),
    epochs=int(getattr(cfg, "embedding_word2vec_epochs", 1)),
    post_warmup_lr_scale=float(getattr(cfg, "embedding_word2vec_post_warmup_lr_scale", 0.1)),
    warmup_events=int(getattr(cfg, "warmup_events", 0)),
    ngram_length=int(getattr(cfg, "sequence_ngram_length", 8)),
    thread_aware=bool(getattr(cfg, "sequence_thread_aware", True)),
    feature_prefix="sequence_ctx",
  )

def _load_word2vec_from_checkpoint(path: Path, *, limit_tokens: int | None) -> tuple[list[str], np.ndarray, int]:
  path = Path(path)
  with path.open("rb") as f:
    state = pickle.load(f)
  feature_state = state.get("feature_state", None)
  if not isinstance(feature_state, dict):
    raise SystemExit(f"Checkpoint has no feature_state: {path}")

  seq = feature_state.get("sequence_context", None)
  if not isinstance(seq, dict):
    raise SystemExit(f"Checkpoint feature_state missing sequence_context: {path}")

  w2v = seq.get("w2v", None)
  if not isinstance(w2v, dict):
    raise SystemExit(f"Checkpoint sequence_context missing w2v: {path}")

  model = w2v.get("model", None)
  if model is None:
    raise SystemExit(f"Checkpoint has no saved Word2Vec model: {path}")

  try:
    keys = [str(k) for k in list(model.wv.index_to_key) if k is not None]
  except Exception as e:
    raise SystemExit(f"Could not read gensim Word2Vec vocab from checkpoint: {e}")

  if limit_tokens is not None:
    keys = keys[: max(0, int(limit_tokens))]

  if not keys:
    return [], np.zeros((0, 0), dtype=np.float32), int(state.get("checkpoint_index", 0))

  mat = np.asarray([model.wv[k] for k in keys], dtype=np.float32)
  return keys, mat, int(state.get("checkpoint_index", 0))


def _project_2d(x: np.ndarray, *, method: str, seed: int) -> np.ndarray:
  if x.shape[0] == 0:
    return np.zeros((0, 2), dtype=np.float32)
  m = (method or "pca").strip().lower()

  if m == "pca":
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(x).astype(np.float32)

  if m in ("tsne", "t-sne"):
    from sklearn.manifold import TSNE

    # Perplexity must be < n_samples; pick a conservative value.
    n = int(x.shape[0])
    perplexity = float(min(30.0, max(2.0, (n - 1) / 3.0)))
    return (
      TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
      )
      .fit_transform(x)
      .astype(np.float32)
    )

  raise ValueError("Unknown projection method: %r (use: pca, tsne)" % method)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
  ap = argparse.ArgumentParser(
    description="Plot syscall Word2Vec embeddings from a log replay (JSONL/EVT1) or from a checkpoint (.pkl)."
  )
  ap.add_argument(
    "input",
    type=Path,
    help="Path to detector-events.jsonl / EVT1 events.bin(.gz) OR a detector checkpoint .pkl (with feature_state).",
  )
  ap.add_argument("--out", type=Path, required=True, help="Output PNG path.")
  ap.add_argument("--max-events", type=int, default=None, help="Stop after N events (for quicker iteration).")
  ap.add_argument("--skip", type=int, default=None, help="Skip first N events.")
  ap.add_argument("--limit-tokens", type=int, default=250, help="Plot only the first N vocab tokens (default: 250).")
  ap.add_argument("--method", default="tsne", help="2D projection: pca or tsne (default: tsne).")
  ap.add_argument("--label-topk", type=int, default=80, help="Label only the first K tokens to reduce clutter.")
  ap.add_argument("--seed", type=int, default=0, help="Random seed for projection (default: 0).")
  args = ap.parse_args()

  if plt is None:
    raise SystemExit("matplotlib is required (install the 'dev' extra).")

  path = Path(args.input)
  if not path.exists():
    raise SystemExit(f"File not found: {path}")

  n = 0
  tokens: list[str]
  x: np.ndarray
  if path.suffix.lower() == ".pkl":
    tokens, x, n = _load_word2vec_from_checkpoint(path, limit_tokens=args.limit_tokens)
    log.info("Loaded Word2Vec from checkpoint %s (checkpoint_index=%d)", path, n)
    if x.shape[0] == 0:
      raise SystemExit("Checkpoint Word2Vec has empty vocabulary.")
  else:
    extractor = _make_extractor()
    fmt = _detect_format(path)
    event_iter = (
      iter_events_jsonl(path, max_events=args.max_events, skip=args.skip)
      if fmt == "jsonl"
      else iter_events(path, max_events=args.max_events, skip=args.skip)
    )

    if tqdm is not None:
      total = args.max_events if args.max_events is not None else None
      event_iter = tqdm(event_iter, total=total, desc="Replay", unit=" evt")
    else:
      log.info("tqdm not installed; replaying without progress bar")

    for obj in event_iter:
      # obj is a dict (jsonl) or a protobuf-ish dict via replay_logs; both have these keys.
      syscall = (obj.get("syscall_name") or "").strip().lower() or "__empty__"
      tid_raw = obj.get("tid") or "0"
      try:
        tid = int(str(tid_raw).strip() or "0")
      except ValueError:
        tid = 0
      extractor.observe_embedding(stream_id=tid, token=syscall)
      n += 1
      if tqdm is None and (n % 200000) == 0:
        log.info("Replayed %d events...", n)

    tokens, mat_list = extractor.export_word2vec_matrix(limit=args.limit_tokens)
    x = np.asarray(mat_list, dtype=np.float32)
    if x.shape[0] == 0:
      raise SystemExit("Word2Vec has empty vocabulary. Try increasing --max-events or check the input log.")

  xy = _project_2d(x, method=args.method, seed=int(args.seed))

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)

  plt.figure(figsize=(14, 10))
  plt.scatter(xy[:, 0], xy[:, 1], s=18, alpha=0.75)

  label_k = max(0, min(int(args.label_topk), len(tokens)))
  for i in range(label_k):
    plt.text(xy[i, 0], xy[i, 1], tokens[i], fontsize=8, alpha=0.9)

  plt.title(f"Syscall Word2Vec ({args.method}) — tokens={len(tokens)} events={n}")
  plt.axis("off")
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  log.info("Wrote %s", out)


if __name__ == "__main__":
  main()

