# Fractal Agents — Research Program

## Mission
Autonomously discover and visualize fractal properties in neural network weights, training dynamics, and loss landscapes.

## Setup
```bash
uv run prepare.py     # download data, tokenize (once)
uv run train.py       # train small GPT, saves checkpoints every 200 steps
uv run analyze.py all # run all fractal analyses on saved checkpoints
```

## Research Agenda

### Phase 1: Weight Spectra ✅
- [x] Eigenvalue spectra of weight matrices across training
- [x] Power-law exponent fitting — α ≈ 1.1–4.0, R² > 0.9, confirms Martin & Mahoney
- [x] Compare spectra across layer types — attention develops steeper tails than MLP
- [x] Singular value log-log plots — clear power-law in trained vs Marchenko-Pastur in init

### Phase 2: Self-Similarity ✅
- [x] Cross-layer spectral similarity — cosine sim >0.99 across all layer pairs
- [x] Box-counting fractal dimension — D≈1.99, attention drifts down during training
- [x] Attention weight correlations — nested block-diagonal structure (best visual)
- [x] Weight zoom — horizontal banding persists at multiple scales
- [ ] Multifractal spectrum (Rényi dimensions) ← NEXT

### Phase 3: Training Dynamics
- [x] Fractal structure emergence — tracked across 21 checkpoints, power-law develops monotonically
- [x] Multifractal spectrum — attempted, 1D flattening destroys spatial structure. Need 2D MFDFA/wavelet leaders.
- [x] Fractal dimension of SGD trajectory — H=0.753 (D=1.247), persistent fractal path through weight space
- [x] Loss landscape cross-sections — smooth (H≈1.05), not fractal at this scale/regime

### Phase 4: Attention & Activations ✅
- [x] Fractal structure in attention maps — power-law singular values, self-similar across heads
- [x] Self-similarity of hidden representations across layers — CKA gradient + spectral sim >0.99
- [x] Fractal dimension of activation distributions — correlation dim grows 0.5D → 16D through network

### Phase 5: Synthesis & Extensions
- [x] Gradient fractal structure — kurtosis 10–12 in c_attn layers, confirms heavy-tailed SGD theory
- [x] Compare init vs trained — every metric moves from random → structured/power-law during training
- [x] Scale comparison — YES, all fractal signatures universal across 5M/12M/19M. Hurst ≈ 0.76 is near-constant.

## Constraints
- MPS (Apple Silicon) — keep everything single-GPU friendly
- Each experiment should complete in < 5 minutes
- All findings go in experiments/ with plots and findings.md

## How the Agent Works
1. Pick next item from the agenda
2. Implement the analysis in analyze.py (add a new experiment function)
3. Run it, generate plots
4. Write up findings
5. Update this agenda, repeat

## ArXiv 2026 Experiment Roadmap (Last-6-Month Literature Driven)

### Priority Execution Order
1. Critical-LR fractal basin maps
2. RMT-denoised spectral metrics
3. Attention graph spectral diagnostics
4. Seed identity persistence
5. Connectivity vs confinement

### 1) Critical-LR Fractal Basin Maps
- Goal: test whether convergence boundaries become fractal near instability.
- Protocol: for each model scale, estimate critical LR `lr*`, then run dense sweeps around `lr*` (`~40` LR values, log-spaced) across `~32` seeds for `~2k` steps.
- Metrics: class labels (`stable`, `slow`, `diverged`), boundary box-counting dimension, boundary length vs resolution, bootstrap confidence intervals.
- Controls: convex-model baseline and shuffled-label baseline.
- Success criterion: boundary dimension significantly above 1 near `lr*`, robust across scales.

### 2) Seed Identity Persistence
- Goal: test if initialization leaves persistent signatures after training.
- Protocol: train `30-50` seeds per scale, save checkpoints every `500` steps, extract spectral + attention + gradient feature vectors.
- Metrics: seed classification accuracy (linear probe + nearest centroid), mutual information between seed and feature vector over time.
- Controls: random labels, permutation baselines, train/test split by data chunk.
- Success criterion: above-chance seed identification remains at late checkpoints.

### 3) Universal Subspace Across Scales
- Goal: test whether models share a common low-dimensional weight/spectral subspace.
- Protocol: collect checkpoints across small/medium/large models; align by layer type and run PCA/CCA/Procrustes on standardized vectors or spectra.
- Metrics: shared variance explained, principal-angle overlap, cross-scale reconstruction error.
- Controls: random orthonormal basis and layer-shuffle baseline.
- Success criterion: shared basis explains substantially more variance than controls.

### 4) Connectivity vs Confinement
- Goal: test whether low-loss paths coexist with local confinement around minima.
- Protocol: train model pairs per configuration; measure interpolation barriers and optimize curved connecting paths; estimate local curvature and noisy transition behavior.
- Metrics: interpolation barrier, connected-path max loss, curvature proxy, transition probability under noise.
- Controls: SWA/flatter-solution comparator.
- Success criterion: low-loss connectivity plus measurable local confinement.

### 5) RMT-Denoised Spectral Metrics
- Goal: separate random bulk from informative tail before heavy-tail inference.
- Protocol: apply singular-value denoising (MP edge or robust thresholding), then fit tails with MLE + model comparison.
- Metrics: KS goodness-of-fit, likelihood-ratio tests (power law vs alternatives), estimate stability across checkpoints/seeds.
- Controls: synthetic random matrices and planted heavy-tail matrices.
- Success criterion: denoising improves fit diagnostics and conclusion stability.

### 6) Gradient Spectral Anisotropy Over Training
- Goal: test if gradient anisotropy predicts later fractal structure in weights.
- Protocol: at fixed intervals, compute gradients on a fixed probe set and estimate covariance spectra with low-rank methods.
- Metrics: `lambda1/trace`, participation ratio, effective rank, lagged correlation with later weight-spectrum changes.
- Controls: clipping on/off, batch-size sweeps, random probe batches.
- Success criterion: anisotropy metrics are reliable early indicators of later spectral changes.

### 7) Attention Graph Spectral Diagnostics
- Goal: quantify attention "nested structure" spectrally rather than visually only.
- Protocol: treat attention maps as weighted graphs and track Laplacian-based statistics over training.
- Metrics: Fiedler value, spectral entropy, high-frequency energy ratio, smoothness, modularity.
- Controls: degree-preserving graph randomization and token-order shuffles where valid.
- Success criterion: consistent trained-vs-init shifts and monotonic training trends.

### 8) Causal-Sensitivity Probe
- Goal: test whether causal lag structure emerges alongside fractal signatures.
- Protocol: compute gradient sensitivity of next-token logits to earlier token positions (lag-k influence curves) over layers/checkpoints.
- Metrics: lag-decay exponent, sparsity/concentration index, head diversity, correlation with attention spectral metrics and validation behavior.
- Controls: randomized-sequence baselines and synthetic known-lag data.
- Success criterion: learned lag structure differs clearly from init/random baselines and aligns with geometry metrics.

### Paper Narrative (Recommended)
- Optimization phase transition: critical-LR fractal basin maps.
- Robust structure estimate: RMT-denoised spectra + statistical tail tests.
- Functional structure: attention graph spectral diagnostics.
- Memory of initialization: seed identity persistence.
- Landscape interpretation: connectivity vs confinement.
