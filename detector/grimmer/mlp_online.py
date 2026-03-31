"""
Online MLP next-syscall predictor with anomaly score ``1 - p(correct)``.

Architecture and scoring match the LID-DS MLP building block (PyTorch softmax output,
score from predicted probability of the true class).

References:
  - ``third_party/LID-DS/algorithms/decision_engines/mlp.py`` — ``MLP._cached_results``
  - ``third_party/LID-DS/algorithms/decision_engines/mlp.py`` — ``Feedforward._get_mlp_sequence``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
  """
  Same layer pattern as LID-DS ``Feedforward`` (Linear, Dropout, ReLU × depth, then Linear + Softmax).

  Reference: ``third_party/LID-DS/algorithms/decision_engines/mlp.py`` — ``Feedforward``.
  """

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


class GrimmerOnlineMLP:
  """
  One gradient step per event after the n-gram window is warm.

  Uses softmax outputs and raw anomaly score ``1 - p[y]`` like LID-DS ``MLP``.
  When new syscalls appear, the final linear layer is expanded (new rows/cols).
  """

  def __init__(
    self,
    *,
    input_dim: int,
    hidden_size: int,
    hidden_layers: int,
    learning_rate: float,
    model_device: str,
    seed: int,
  ) -> None:
    self._input_dim = int(input_dim)
    self._hidden_size = int(hidden_size)
    self._hidden_layers = int(hidden_layers)
    self._lr = float(learning_rate)
    self._device = _resolve_device(model_device)
    self._seed = int(seed)
    self._num_classes = 0
    self._model: _FeedforwardCore | None = None
    self._optimizer: optim.Optimizer | None = None
    torch.manual_seed(self._seed)

  @property
  def algorithm(self) -> str:
    return "grimmer_mlp"

  def ensure_num_classes(self, num_classes: int) -> None:
    """Grow the softmax layer if the syscall vocabulary grew."""
    self._ensure_model(int(num_classes))

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
        "GrimmerOnlineMLP init input_dim=%d hidden=%d layers=%d classes=%d device=%s",
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
    """Expand the final Linear + rebuild Softmax tail (LID-DS fixed vocab at train time; we grow online)."""
    assert self._model is not None and self._optimizer is not None
    mods = list(self._model.net.children())
    idx_last = None
    for i, m in enumerate(mods):
      if isinstance(m, nn.Linear):
        idx_last = i
    if idx_last is None:
      raise RuntimeError("GrimmerOnlineMLP: no Linear layer found")
    last = mods[idx_last]
    assert isinstance(last, nn.Linear)
    old_out = last.out_features
    if new_out <= old_out:
      return
    in_f = last.in_features
    new_lin = nn.Linear(in_f, new_out, bias=True).to(self._device)
    with torch.no_grad():
      new_lin.weight[:old_out, :] = last.weight
      new_lin.bias[:old_out] = last.bias
      nn.init.normal_(new_lin.weight[old_out:, :], std=0.01)
      nn.init.zeros_(new_lin.bias[old_out:])
    new_mods = mods[:idx_last] + [new_lin] + mods[idx_last + 1 :]
    self._model.net = nn.Sequential(*new_mods).to(self._device)
    self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

  def score_and_learn(
    self,
    x: np.ndarray | list[float],
    y_class: int,
  ) -> tuple[float, float]:
    """
    Returns ``(raw, scaled)`` with raw = ``1 - p[y]`` (LID-DS MLP).

    Reference: ``third_party/LID-DS/algorithms/decision_engines/mlp.py`` — ``_cached_results``.
    """
    y = int(y_class)
    self._ensure_model(max(y + 1, 1))
    if self._model is None or self._optimizer is None:
      return 0.0, 0.0

    xv = torch.tensor(np.asarray(x, dtype=np.float32), device=self._device).view(1, -1)
    if xv.shape[1] != self._input_dim:
      raise ValueError(f"Expected input dim {self._input_dim}, got {xv.shape[1]}")

    self._model.train()
    self._optimizer.zero_grad()
    probs = self._model(xv).view(-1)
    if y < 0 or y >= probs.numel():
      return 0.0, 0.0
    # Numerical safety: clamp probability used for score
    py = probs[y].clamp(1e-8, 1.0)
    loss = -torch.log(py)
    loss.backward()
    self._optimizer.step()

    self._model.eval()
    with torch.no_grad():
      p_eval = self._model(xv).view(-1)
      py_eval = float(p_eval[y].clamp(1e-8, 1.0).item())
    raw = 1.0 - py_eval
    scaled = 1.0 - float(np.exp(-max(0.0, raw)))
    return float(raw), float(scaled)

  def get_state(self) -> dict[str, Any]:
    return {
      "input_dim": self._input_dim,
      "hidden_size": self._hidden_size,
      "hidden_layers": self._hidden_layers,
      "lr": self._lr,
      "num_classes": self._num_classes,
      "model": self._model.state_dict() if self._model is not None else None,
      "optim": self._optimizer.state_dict() if self._optimizer is not None else None,
    }

  def set_state(self, state: dict[str, Any]) -> None:
    n = int(state.get("num_classes", 0))
    self._model = None
    self._optimizer = None
    self._num_classes = 0
    if n <= 0 or state.get("model") is None:
      return
    self._ensure_model(n)
    assert self._model is not None and self._optimizer is not None
    self._model.load_state_dict(state["model"])
    if state.get("optim") is not None:
      try:
        self._optimizer.load_state_dict(state["optim"])
      except Exception:
        logger.warning("GrimmerOnlineMLP: could not load optimizer state; using fresh Adam moments")
