from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import norm as scipy_norm

from detector.building_blocks.primitives.models.common import _BothScoresMixin, _resolve_torch_device
from detector.building_blocks.primitives.models.statistical import OnlineFreq1D


class _MemStreamAutoEncoder(torch.nn.Module):
  def __init__(self, input_dim: int):
    super().__init__()
    latent_dim = 2 * input_dim
    self.encoder = torch.nn.Sequential(torch.nn.Linear(input_dim, latent_dim), torch.nn.Tanh())
    self.decoder = torch.nn.Linear(latent_dim, input_dim)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = self.encoder(x)
    recon = self.decoder(z)
    return z, recon


class OnlineMemStream(_BothScoresMixin):
  def __init__(
    self,
    memory_size: int,
    lr: float,
    beta: float,
    k: int,
    gamma: float,
    input_mode: str,
    freq1d_bins: int,
    freq1d_alpha: float,
    freq1d_decay: float,
    freq1d_max_categories: int,
    model_device: str,
    seed: int,
    warmup_accept: int = 512,
  ):
    self.algorithm = "memstream"
    self.memory_size = memory_size
    self.lr = lr
    self.beta = beta
    self.k = k
    self.gamma = gamma
    self.seed = seed
    self.input_mode = str(input_mode).strip().lower()
    if self.input_mode not in ("raw", "freq1d_u", "freq1d_z", "freq1d_surprisal", "freq1d_z_surprisal"):
      raise ValueError("memstream input_mode must be one of: raw, freq1d_u, freq1d_z, freq1d_surprisal, freq1d_z_surprisal")
    self._frontend_u_clamp = 1e-6
    self._frontend_z_clip = 8.0
    self._frontend_surprisal_clip = 12.0
    self.model_device = model_device
    self._device = _resolve_torch_device(model_device)
    self._feature_names: Optional[List[str]] = None
    self._model: Optional[_MemStreamAutoEncoder] = None
    self._optimizer: Optional[torch.optim.Optimizer] = None
    self._memory_latent: Optional[torch.Tensor] = None
    self._memory_input: Optional[torch.Tensor] = None
    self._norm_mean: Optional[torch.Tensor] = None
    self._norm_std: Optional[torch.Tensor] = None
    self._mem_index = 0
    self._mem_filled = 0
    self._warmup_accept = warmup_accept
    self._noise_std = 1e-3
    self._accepted_updates = 0
    self._rejected_updates = 0
    self._overwrite_updates = 0
    self._last_debug: Dict[str, Any] = {}
    self._frontend_marginals: Optional[OnlineFreq1D] = None
    if self.input_mode != "raw":
      self._frontend_marginals = OnlineFreq1D(
        bins=freq1d_bins,
        alpha=freq1d_alpha,
        decay=freq1d_decay,
        max_categories=freq1d_max_categories,
        aggregation="mean",
        topk=1,
        soft_topk_temperature=1.0,
        model_device="cpu",
        seed=0,
      )
    self._exp: Optional[torch.Tensor] = None

  def _frontend_transform(self, features: Dict[str, float]) -> Dict[str, float]:
    if self.input_mode == "raw":
      return dict(features)
    if self._frontend_marginals is None:
      raise RuntimeError("MemStream frontend marginals not initialized")
    u = self._frontend_marginals.get_cdf_vector(features)
    names = self._frontend_marginals._feature_names
    if names is None:
      raise RuntimeError("MemStream frontend feature names not initialized")
    if self.input_mode == "freq1d_u":
      return {f"u::{name}": float(u[i]) for i, name in enumerate(names)}
    u_clipped = np.clip(u, self._frontend_u_clamp, 1.0 - self._frontend_u_clamp)
    z = np.clip(scipy_norm.ppf(u_clipped), -self._frontend_z_clip, self._frontend_z_clip)
    if self.input_mode == "freq1d_z":
      return {f"z::{name}": float(z[i]) for i, name in enumerate(names)}
    excess = np.clip(self._frontend_marginals.get_excess_vector(features), 0.0, self._frontend_surprisal_clip)
    if self.input_mode == "freq1d_surprisal":
      return {f"s::{name}": float(excess[i]) for i, name in enumerate(names)}
    transformed: Dict[str, float] = {}
    for i, name in enumerate(names):
      transformed[f"z::{name}"] = float(z[i])
      transformed[f"s::{name}"] = float(excess[i])
    return transformed

  def _init_model_for_feature_names(self, feature_names: List[str]) -> None:
    self._feature_names = list(feature_names)
    input_dim = len(self._feature_names)
    latent_dim = 2 * input_dim
    torch.manual_seed(self.seed)
    if self._device.type == "cuda":
      torch.cuda.manual_seed_all(self.seed)
    self._model = _MemStreamAutoEncoder(input_dim=input_dim).to(self._device)
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    self._memory_latent = torch.zeros(self.memory_size, latent_dim, dtype=torch.float32, device=self._device)
    self._memory_input = torch.zeros(self.memory_size, input_dim, dtype=torch.float32, device=self._device)
    self._norm_mean = torch.zeros(input_dim, dtype=torch.float32, device=self._device)
    self._norm_std = torch.ones(input_dim, dtype=torch.float32, device=self._device)
    self._exp = torch.tensor([self.gamma ** i for i in range(self.k)], dtype=torch.float32, device=self._device)

  def _init_from_features(self, features: Dict[str, float]) -> None:
    transformed = self._frontend_transform(features)
    self._init_model_for_feature_names(sorted(transformed.keys()))

  def _vectorize(self, features: Dict[str, float]) -> torch.Tensor:
    if self._feature_names is None:
      self._init_from_features(features)
    if self._feature_names is None:
      raise RuntimeError("MemStream feature names not initialized")
    transformed = self._frontend_transform(features)
    vec = np.array([float(transformed[k]) for k in self._feature_names], dtype=np.float32)
    return torch.from_numpy(vec).to(self._device)

  def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    if self._norm_mean is None or self._norm_std is None:
      return x
    safe_std = torch.where(self._norm_std > 1e-6, self._norm_std, torch.ones_like(self._norm_std))
    return (x - self._norm_mean) / safe_std

  def _refresh_norm_from_memory(self) -> None:
    if self._memory_input is None or self._mem_filled == 0:
      return
    mem = self._memory_input[: self._mem_filled]
    self._norm_mean = mem.mean(dim=0)
    std = mem.std(dim=0, unbiased=False)
    self._norm_std = torch.where(std > 1e-6, std, torch.ones_like(std))

  def _memory_distance(self, z: torch.Tensor) -> float:
    if self._memory_latent is None or self._mem_filled == 0 or self._exp is None:
      return 0.0
    memory = self._memory_latent[: self._mem_filled]
    dists = torch.norm(memory - z.unsqueeze(0), p=1, dim=1)
    k_eff = max(1, min(self.k, int(self._mem_filled)))
    topk_vals = torch.topk(dists, k=k_eff, largest=False).values
    exp = self._exp[:k_eff]
    return float((topk_vals * exp).sum().item() / exp.sum().item())

  def _should_update_memory(self, score_raw: float) -> bool:
    if self._mem_filled < self._warmup_accept:
      return True
    return score_raw <= self.beta

  def _write_memory(self, x: torch.Tensor, z: torch.Tensor) -> Dict[str, Any]:
    if self._memory_latent is None or self._memory_input is None:
      return {"memory_slot": None, "overwrite": False, "mem_filled_after": self._mem_filled, "mem_index_after": self._mem_index}
    overwrite = self._mem_filled >= self.memory_size
    if self._mem_filled < self.memory_size:
      idx = self._mem_filled
      self._mem_filled += 1
    else:
      idx = self._mem_index
      self._mem_index = (self._mem_index + 1) % self.memory_size
    self._memory_latent[idx] = z.detach()
    self._memory_input[idx] = x.detach()
    self._refresh_norm_from_memory()
    return {"memory_slot": int(idx), "overwrite": overwrite, "mem_filled_after": int(self._mem_filled), "mem_index_after": int(self._mem_index)}

  def get_last_debug(self) -> Dict[str, Any]:
    return dict(self._last_debug)

  def score_only_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")
    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.eval()
    with torch.no_grad():
      z_eval, _ = self._model(x_norm)
      score_raw = self._memory_distance(z_eval.squeeze(0))
    return max(0.0, float(score_raw))

  def score_and_learn_raw(self, features: Dict[str, float], *, meta: Any | None = None) -> float:
    if self._model is None or self._optimizer is None:
      self._init_from_features(features)
    if self._model is None or self._optimizer is None:
      raise RuntimeError("MemStream not initialized")
    x = self._vectorize(features)
    x_norm = self._normalize(x).unsqueeze(0)
    self._model.train()
    with torch.no_grad():
      z_eval, _ = self._model(x_norm)
      score_raw = self._memory_distance(z_eval.squeeze(0))
    update_allowed = self._should_update_memory(score_raw)
    if update_allowed:
      noisy = x_norm + (self._noise_std * torch.randn_like(x_norm))
      self._optimizer.zero_grad()
      z_train, recon_train = self._model(noisy)
      loss = torch.mean((x_norm - recon_train) ** 2)
      loss.backward()
      self._optimizer.step()
      write_info = self._write_memory(x, z_train.squeeze(0))
      self._accepted_updates += 1
      if bool(write_info["overwrite"]):
        self._overwrite_updates += 1
    else:
      self._rejected_updates += 1
    return max(0.0, float(score_raw))

  def get_state(self) -> Dict[str, Any]:
    if self._model is None:
      raise RuntimeError("MemStream not initialized; cannot save empty state")
    return {
      "feature_names": list(self._feature_names),
      "model_state": {k: v.cpu() for k, v in self._model.state_dict().items()},
      "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
      "memory_latent": self._memory_latent.cpu().numpy() if self._memory_latent is not None else None,
      "memory_input": self._memory_input.cpu().numpy() if self._memory_input is not None else None,
      "norm_mean": self._norm_mean.cpu().numpy() if self._norm_mean is not None else None,
      "norm_std": self._norm_std.cpu().numpy() if self._norm_std is not None else None,
      "input_mode": self.input_mode,
      "frontend_marginals_state": self._frontend_marginals.get_state() if self._frontend_marginals is not None and self.input_mode != "raw" else None,
      "mem_index": self._mem_index,
      "mem_filled": self._mem_filled,
      "accepted_updates": self._accepted_updates,
      "rejected_updates": self._rejected_updates,
      "overwrite_updates": self._overwrite_updates,
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    self.input_mode = str(state.get("input_mode", self.input_mode))
    frontend_state = state.get("frontend_marginals_state")
    if frontend_state is not None:
      if self._frontend_marginals is None:
        raise RuntimeError("MemStream frontend state present but frontend is disabled")
      self._frontend_marginals.set_state(frontend_state)
    self._feature_names = list(state["feature_names"])
    self._init_model_for_feature_names(self._feature_names)
    self._model.load_state_dict({k: v.to(self._device) for k, v in state["model_state"].items()})
    if state["optimizer_state"] and self._optimizer:
      self._optimizer.load_state_dict(state["optimizer_state"])
    if state["memory_latent"] is not None:
      self._memory_latent = torch.from_numpy(state["memory_latent"]).to(self._device)
    if state["memory_input"] is not None:
      self._memory_input = torch.from_numpy(state["memory_input"]).to(self._device)
    if state["norm_mean"] is not None:
      self._norm_mean = torch.from_numpy(state["norm_mean"]).to(self._device)
    if state["norm_std"] is not None:
      self._norm_std = torch.from_numpy(state["norm_std"]).to(self._device)
    self._mem_index = int(state["mem_index"])
    self._mem_filled = int(state["mem_filled"])
    self._accepted_updates = int(state.get("accepted_updates", 0))
    self._rejected_updates = int(state.get("rejected_updates", 0))
    self._overwrite_updates = int(state.get("overwrite_updates", 0))
