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

### Phase 4: Attention & Activations (current)
- [x] Fractal structure in attention maps — power-law singular values, self-similar across heads
- [ ] Self-similarity of hidden representations across layers
- [ ] Fractal dimension of activation distributions

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
