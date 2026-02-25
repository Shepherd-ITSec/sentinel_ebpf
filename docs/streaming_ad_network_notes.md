# Streaming anomaly detection: network design notes

## Our MemStream vs original (Stream-AD/MemStream)

| Aspect | Original (paper/code) | Our implementation |
|--------|----------------------|---------------------|
| **Encoder** | 1 layer: `Linear(in, 2*in)` + Tanh | 2 layers: Linear(10, 32) → ReLU → Linear(32, 8) |
| **Decoder** | 1 layer: `Linear(2*in, in)` | 2 layers: Linear(8, 32) → ReLU → Linear(32, 10) |
| **Latent** | **2× input dim** (e.g. 20 for 10 features) | **8** (smaller bottleneck) |
| **Memory size** | 256–2048 in demos | 128 |
| **Warmup** | Offline: train AE on **normal-only** subset, then init memory | Online only: no normal subset; memory filled as stream arrives |
| **Score** | min L1 distance(encoder(x), memory) | 0.8 × mean k-NN distance + 0.2 × recon error |
| **Update** | FIFO when score ≤ β | Adaptive β; FIFO when score ≤ β |

So we are **shallower in width** (smaller latent, smaller memory) and **fully online** (no normal-only warmup). The paper’s setup assumes an initial batch of normal data to pretrain the AE and seed memory; we don’t have that, so our AE and memory are trained on a mix that can include anomalies (we mitigate with threshold-gated updates).

## Other network ideas (from quick research)

- **LSTM encoder–decoder**  
  For sequence-aware streaming AD: encode a sliding window of recent events, decode and use reconstruction error. Fits time-series streams; our stream is event-by-event so we’d need to define a window (e.g. last N events or last T seconds). More complex and more hyperparameters.

- **Deeper (2–3 layer) AE**  
  Keep the same interface (single-vector in, score out) but use a deeper MLP, e.g. `in → 64 → 32 → latent → 32 → 64 → in` with ReLU. Gives more capacity than our current 32→8 and may help if the data is highly nonlinear.

- **MAE instead of MSE**  
  Some work suggests Mean Absolute Error for reconstruction is more robust to outliers when training on contaminated streams. Easy to try alongside MSE.

- **Paper-closer MemStream**  
  Single-layer encoder/decoder with **latent_dim = 2 × input_dim**, Tanh, and larger memory (e.g. 512–2048). Keeps the same “min distance to memory” style scoring. Optionally support a “normal-only” warmup phase if we ever have labels or a trusted initial window.

## Recommended next steps

1. **Add a “paper-like” MemStream variant**  
   - One hidden layer, width = 2× input_dim, Tanh.  
   - Larger default memory (e.g. 512).  
   - Score = min L1 distance to memory (like original); keep optional recon term for stability.  
   - Expose as a config option (e.g. `memstream_paper` or `memstream_wide`) so we can A/B against current MemStream.

2. **Add a deeper AE variant**  
   - e.g. 10 → 64 → 32 → 16 → 32 → 64 → 10 (or 10 → 64 → 32 → 64 → 10).  
   - Same memory + k-NN scoring on latent as now.  
   - Expose as `memstream_deep` (or similar) and tune depth/width via config.

3. **Optional: MAE loss**  
   - Config flag to use MAE instead of MSE for AE training; compare on BETH.

4. **Later: LSTM/sequence**  
   - Only if we want to exploit temporal order explicitly; requires windowing and more engineering.

Implementing (1) and (2) gives two additional network backends we can compare on the same eval pipeline (BETH, evil_only / sus_or_evil) without changing the rest of the stack.
