from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from detector.sequence.context import SequenceFeatureDict

logger = logging.getLogger(__name__)


def _resolve_device(preference: str) -> torch.device:
  pref = (preference or "auto").strip().lower()
  if pref == "cpu":
    return torch.device("cpu")
  if pref == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda")
  if pref == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  raise ValueError(f"Unknown device preference: {preference!r}")


class _FeedforwardCore(nn.Module):
  def __init__(
    self,
    input_size: int,
    hidden_size: int,
    output_size: int,
    hidden_layers: int,
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = [
      nn.Linear(input_size, hidden_size),
      nn.Dropout(p=0.5),
      nn.ReLU(),
    ]
    for _ in range(int(hidden_layers)):
      layers.extend(
        [
          nn.Linear(hidden_size, hidden_size),
          nn.Dropout(p=0.5),
          nn.ReLU(),
        ]
      )
    layers.extend(
      [
        nn.Linear(hidden_size, output_size),
        nn.Dropout(p=0.5),
        nn.Softmax(dim=-1),
      ]
    )
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class OnlineSequenceMLP:
  """Online next-token predictor consuming the generic sequence feature view."""

  def __init__(
    self,
    *,
    hidden_size: int,
    hidden_layers: int,
    learning_rate: float,
    model_device: str,
    seed: int,
  ) -> None:
    self.algorithm = "sequence_mlp"
    self._hidden_size = int(hidden_size)
    self._hidden_layers = int(hidden_layers)
    self._lr = float(learning_rate)
    self._device = _resolve_device(model_device)
    self._seed = int(seed)
    self._feature_names: list[str] | None = None
    self._input_dim = 0
    self._num_classes = 0
    self._model: _FeedforwardCore | None = None
    self._optimizer: optim.Optimizer | None = None
    torch.manual_seed(self._seed)

  def _init_from_features(self, features: dict[str, float]) -> None:
    self._feature_names = sorted(features.keys())
    self._input_dim = len(self._feature_names)

  def _vectorize(self, features: dict[str, float]) -> np.ndarray:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("SequenceMLP feature names not initialized")
    return np.array([float(features[k]) for k in self._feature_names], dtype=np.float32)

  def _ensure_model(self, num_classes: int) -> None:
    n = int(num_classes)
    if n <= 0:
      return
    if self._model is None:
      self._num_classes = n
      self._model = _FeedforwardCore(
        self._input_dim,
        self._hidden_size,
        n,
        self._hidden_layers,
      ).to(self._device)
      self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
      logger.info(
        "SequenceMLP init input_dim=%d hidden=%d layers=%d classes=%d device=%s",
        self._input_dim,
        self._hidden_size,
        self._hidden_layers,
        n,
        self._device,
      )
      return
    if n <= self._num_classes:
      return
    self._grow_output_linear(n)
    self._num_classes = n

  def _grow_output_linear(self, new_out: int) -> None:
    assert self._model is not None and self._optimizer is not None
    mods = list(self._model.net.children())
    idx_last = None
    for i, mod in enumerate(mods):
      if isinstance(mod, nn.Linear):
        idx_last = i
    if idx_last is None:
      raise RuntimeError("SequenceMLP: no Linear layer found")
    last = mods[idx_last]
    assert isinstance(last, nn.Linear)
    old_out = last.out_features
    if new_out <= old_out:
      return
    in_features = last.in_features
    new_lin = nn.Linear(in_features, new_out, bias=True).to(self._device)
    with torch.no_grad():
      new_lin.weight[:old_out, :] = last.weight
      new_lin.bias[:old_out] = last.bias
      nn.init.normal_(new_lin.weight[old_out:, :], std=0.01)
      nn.init.zeros_(new_lin.bias[old_out:])
    new_mods = mods[:idx_last] + [new_lin] + mods[idx_last + 1 :]
    self._model.net = nn.Sequential(*new_mods).to(self._device)
    self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

  @staticmethod
  def _meta_from_features(features: dict[str, float]) -> Any | None:
    meta = getattr(features, "sequence_meta", None)
    if meta is None and isinstance(features, SequenceFeatureDict):
      meta = features.sequence_meta
    return meta

  def score_only_raw(self, features: dict[str, float]) -> float:
    meta = self._meta_from_features(features)
    if meta is None or not bool(meta.ready):
      return 0.0
    x = self._vectorize(features)
    self._ensure_model(int(meta.num_classes))
    if self._model is None:
      return 0.0
    xv = torch.tensor(x, device=self._device).view(1, -1)
    self._model.eval()
    with torch.no_grad():
      probs = self._model(xv).view(-1)
      target_id = int(meta.target_id)
      if target_id < 0 or target_id >= probs.numel():
        return 0.0
      py = float(probs[target_id].clamp(1e-8, 1.0).item())
    return float(1.0 - py)

  def score_only(self, features: dict[str, float]) -> tuple[float, float]:
    raw = self.score_only_raw(features)
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return float(raw), float(scaled)

  def score_and_learn_raw(self, features: dict[str, float]) -> float:
    meta = self._meta_from_features(features)
    if meta is None or not bool(meta.ready):
      return 0.0
    x = self._vectorize(features)
    self._ensure_model(int(meta.num_classes))
    if self._model is None or self._optimizer is None:
      return 0.0

    xv = torch.tensor(x, device=self._device).view(1, -1)
    if xv.shape[1] != self._input_dim:
      raise ValueError(f"Expected input dim {self._input_dim}, got {xv.shape[1]}")

    target_id = int(meta.target_id)
    self._model.train()
    self._optimizer.zero_grad()
    probs = self._model(xv).view(-1)
    if target_id < 0 or target_id >= probs.numel():
      return 0.0
    py = probs[target_id].clamp(1e-8, 1.0)
    loss = -torch.log(py)
    loss.backward()
    self._optimizer.step()

    self._model.eval()
    with torch.no_grad():
      p_eval = self._model(xv).view(-1)
      py_eval = float(p_eval[target_id].clamp(1e-8, 1.0).item())
    return float(1.0 - py_eval)

  def score_and_learn(self, features: dict[str, float]) -> tuple[float, float]:
    raw = self.score_and_learn_raw(features)
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return float(raw), float(scaled)

  def get_state(self) -> dict[str, Any]:
    return {
      "feature_names": list(self._feature_names) if self._feature_names is not None else None,
      "input_dim": self._input_dim,
      "hidden_size": self._hidden_size,
      "hidden_layers": self._hidden_layers,
      "lr": self._lr,
      "num_classes": self._num_classes,
      "model": self._model.state_dict() if self._model is not None else None,
      "optim": self._optimizer.state_dict() if self._optimizer is not None else None,
      "torch_rng_state": torch.get_rng_state(),
      "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

  def set_state(self, state: dict[str, Any]) -> None:
    feature_names = state.get("feature_names")
    self._feature_names = list(feature_names) if feature_names is not None else None
    self._input_dim = int(state.get("input_dim", 0))
    n = int(state.get("num_classes", 0))
    self._model = None
    self._optimizer = None
    self._num_classes = 0
    if n <= 0 or state.get("model") is None or self._input_dim <= 0:
      return
    self._ensure_model(n)
    assert self._model is not None and self._optimizer is not None
    self._model.load_state_dict(state["model"])
    if state.get("optim") is not None:
      try:
        self._optimizer.load_state_dict(state["optim"])
      except Exception:
        logger.warning("SequenceMLP: could not load optimizer state; using fresh Adam moments")

    rng = state.get("torch_rng_state", None)
    if rng is not None:
      try:
        torch.set_rng_state(rng)
      except Exception:
        logger.warning("SequenceMLP: could not restore torch RNG state")
    cuda_rng = state.get("torch_cuda_rng_state_all", None)
    if cuda_rng is not None and torch.cuda.is_available():
      try:
        torch.cuda.set_rng_state_all(cuda_rng)
      except Exception:
        logger.warning("SequenceMLP: could not restore CUDA RNG state")
