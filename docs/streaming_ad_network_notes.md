# Streaming anomaly detection: network design notes

## Our MemStream vs original (Stream-AD/MemStream)

| Aspect | Original (paper/code) | Our implementation |
|--------|----------------------|---------------------|
| **Encoder** | 1 layer: `Linear(in, 2*in)` + Tanh | 1 layer: `Linear(in, 2*in)` + Tanh |
| **Decoder** | 1 layer: `Linear(2*in, in)` | 1 layer: `Linear(2*in, in)` |
| **Latent** | **2× input dim** | **2× input dim** |
| **Memory size** | 256–2048 in demos | 512 (configurable) |
| **Warmup** | Offline: train AE on **normal-only** subset, then init memory | Online by default; optional `mem_warmup_path` for normal-only warmup |
| **Score** | K-NN discounted L1: `(topk L1 × γ^i).sum() / exp.sum()` | Same: K-NN discounted L1 only |
| **Update** | FIFO when score ≤ β | FIFO when score ≤ β (fixed β=0.1) |

Our implementation is now **aligned with the paper**. Config: `mem_beta`, `mem_k`, `mem_gamma` for scoring; `mem_memory_size`, `mem_lr`; optional `mem_warmup_path` for normal-only warmup.

## Other network ideas (from quick research)

- **LSTM encoder–decoder**  
  For sequence-aware streaming AD: encode a sliding window of recent events, decode and use reconstruction error. Fits time-series streams; our stream is event-by-event so we’d need to define a window (e.g. last N events or last T seconds). More complex and more hyperparameters.

- **Deeper (2–3 layer) AE**  
  Keep the same interface (single-vector in, score out) but use a deeper MLP, e.g. `in → 64 → 32 → latent → 32 → 64 → in` with ReLU. Gives more capacity than our current 32→8 and may help if the data is highly nonlinear.

- **MAE instead of MSE**  
  Some work suggests Mean Absolute Error for reconstruction is more robust to outliers when training on contaminated streams. Easy to try alongside MSE.

- **Paper-closer MemStream**  
  Implemented: single-layer encoder/decoder with latent = 2×input_dim, Tanh, K-NN discounted L1 scoring, FIFO memory when score ≤ β. Optional `mem_warmup_path` for normal-only warmup.

## Recommended next steps

1. **Add a deeper AE variant**  
   - e.g. 10 → 64 → 32 → 16 → 32 → 64 → 10 (or 10 → 64 → 32 → 64 → 10).  
   - Same memory + k-NN scoring on latent as now.  
   - Expose as `memstream_deep` (or similar) and tune depth/width via config.

2. **Optional: MAE loss**  
   - Config flag to use MAE instead of MSE for AE training; compare on synthetic eval.

3. **Later: LSTM/sequence**  
   - Only if we want to exploit temporal order explicitly; requires windowing and more engineering.

Implementing (1) gives an additional network backend to compare on the same eval pipeline (evil_only / sus_or_evil) without changing the rest of the stack.
