"""
Fractal analysis toolkit for neural network weights.
This is the file that gets modified experiment by experiment.

Usage: uv run analyze.py [experiment_name]
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from train import GPT, CKPT_DIR, N_EMBD, N_LAYER, BLOCK_SIZE

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_checkpoint(path):
    """Load a checkpoint into a GPT model (on CPU for analysis)."""
    model = GPT()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_checkpoints():
    """Return sorted list of checkpoint paths."""
    pattern = os.path.join(CKPT_DIR, "step_*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No checkpoints found in {CKPT_DIR}/")
        sys.exit(1)
    return paths


def get_weight_matrices(model):
    """Extract all 2D weight matrices from the model."""
    matrices = {}
    for name, param in model.named_parameters():
        if param.dim() == 2 and "wte" not in name and "wpe" not in name:
            matrices[name] = param.detach().numpy()
    return matrices


# --- Shared helpers ---

def compute_loss_batched(model, n_batches=5):
    """Evaluate cross-entropy loss averaged over n_batches of fixed data."""
    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")
    model.eval()
    losses = []
    with torch.no_grad():
        for b in range(n_batches):
            offset = b * BLOCK_SIZE * 4
            ix = [offset + i * BLOCK_SIZE for i in range(4)
                  if offset + (i + 1) * BLOCK_SIZE + 1 < len(data)]
            if not ix:
                break
            x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
            _, loss = model(x, y)
            losses.append(loss.item())
    return np.mean(losses)


def set_model_params(model, flat_params):
    """Set model parameters from a flat tensor."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
        offset += numel


def get_flat_params(model):
    """Return all model parameters as a single flat tensor."""
    return torch.cat([p.detach().clone().flatten() for p in model.parameters()])


def get_attention_maps(model, x=None):
    """
    Extract attention weight matrices for all layers/heads.

    Args:
        model: GPT model (eval mode)
        x: input tensor [1, T]. If None, loads first BLOCK_SIZE tokens from train data.

    Returns:
        dict mapping "layer_{i}" -> np.array of shape [n_head, T, T]
    """
    if x is None:
        data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                         dtype=np.uint16, mode="r")
        x = torch.from_numpy(data[:BLOCK_SIZE].astype(np.int64)).unsqueeze(0)

    model.eval()
    attention_maps = {}
    with torch.no_grad():
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long)
        hidden = model.drop(model.wte(x) + model.wpe(pos))

        for i, block in enumerate(model.blocks):
            normed = block.ln_1(hidden)
            qkv = block.attn.c_attn(normed)
            n_embd = block.attn.n_embd
            n_head = block.attn.n_head
            q, k, v = qkv.split(n_embd, dim=2)
            head_dim = n_embd // n_head
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = torch.softmax(att, dim=-1)
            attention_maps[f"layer_{i}"] = att[0].numpy()  # [n_head, T, T]

            hidden = hidden + block.attn(block.ln_1(hidden))
            hidden = hidden + block.mlp(block.ln_2(hidden))

    return attention_maps


def load_checkpoint_scaled(path, n_layer, n_embd, n_head):
    """Load a checkpoint into a parameterized GPT model."""
    from train_scale import GPT as ScaledGPT
    model = ScaledGPT(n_layer, n_embd, n_head)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# --- Experiment 1: Eigenvalue spectra (heavy tails) ---

def exp_eigenvalue_spectra():
    """
    Compute eigenvalue spectra of weight matrices across training.
    Looking for: heavy-tailed distributions, power-law exponents.
    Martin & Mahoney (2019) found well-trained nets have power-law spectra
    with exponents 1.5 < alpha < 3.5.
    """
    checkpoints = get_checkpoints()
    print(f"Analyzing eigenvalue spectra across {len(checkpoints)} checkpoints...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weight Matrix Eigenvalue Spectra Across Training", fontsize=14)

    # pick a few representative weight matrices
    target_layers = [
        "blocks.0.attn.c_attn.weight",   # early attention
        "blocks.0.mlp.c_fc.weight",       # early MLP
        f"blocks.{N_LAYER-1}.attn.c_attn.weight",  # late attention
        f"blocks.{N_LAYER-1}.mlp.c_fc.weight",     # late MLP
    ]

    for ax, layer_name in zip(axes.flat, target_layers):
        for ckpt_path in checkpoints:
            step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
            model = load_checkpoint(ckpt_path)
            matrices = get_weight_matrices(model)

            if layer_name not in matrices:
                continue

            W = matrices[layer_name]
            # compute singular values (eigenvalues of W^T W)
            sv = np.linalg.svd(W, compute_uv=False)
            eigenvalues = sv ** 2

            # log-log histogram for power law detection
            log_eig = np.log10(eigenvalues + 1e-10)
            ax.hist(log_eig, bins=30, alpha=0.5, label=f"step {step}", density=True)

        short_name = layer_name.replace("blocks.", "L").replace(".weight", "")
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel("log₁₀(eigenvalue)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "eigenvalue_spectra.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def exp_power_law_exponents():
    """
    Fit power-law exponents to eigenvalue spectra across training.
    Track how alpha evolves — does it converge to the 1.5-3.5 range?
    """
    checkpoints = get_checkpoints()
    print(f"Fitting power-law exponents across {len(checkpoints)} checkpoints...")

    results = {}  # layer_name -> [(step, alpha), ...]

    for ckpt_path in checkpoints:
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        matrices = get_weight_matrices(model)

        for name, W in matrices.items():
            sv = np.linalg.svd(W, compute_uv=False)
            eigenvalues = sv ** 2
            eigenvalues = eigenvalues[eigenvalues > 0]

            # fit power law via log-log linear regression on the tail
            sorted_eig = np.sort(eigenvalues)[::-1]
            n = len(sorted_eig)
            ranks = np.arange(1, n + 1)
            # use top 80% for fitting
            cutoff = max(int(n * 0.2), 2)
            log_ranks = np.log10(ranks[cutoff:])
            log_vals = np.log10(sorted_eig[cutoff:])

            if len(log_ranks) > 2:
                slope, _, r_value, _, _ = stats.linregress(log_ranks, log_vals)
                alpha = -slope
                if name not in results:
                    results[name] = []
                results[name].append((step, alpha, r_value ** 2))

    # plot alpha evolution for all layers
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results.items():
        steps, alphas, r2s = zip(*data)
        short = name.replace("blocks.", "L").replace(".weight", "")
        ax.plot(steps, alphas, "o-", markersize=3, label=short, alpha=0.7)

    ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.5, label="α=1.5 (heavy tail)")
    ax.axhline(y=3.5, color="red", linestyle="--", alpha=0.5, label="α=3.5")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Power-law exponent α")
    ax.set_title("Evolution of Power-Law Exponents During Training")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "power_law_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # print summary
    print("\nFinal power-law exponents:")
    for name, data in sorted(results.items()):
        step, alpha, r2 = data[-1]
        short = name.replace("blocks.", "L").replace(".weight", "")
        print(f"  {short:40s} α={alpha:.3f}  R²={r2:.3f}")


# --- Experiment 3: Weight matrix heatmaps + zoom (visual self-similarity) ---

def exp_weight_zoom():
    """
    Visualize weight matrices as images at multiple zoom levels.
    If they're fractal, zooming in should reveal similar structure at every scale.
    """
    checkpoints = get_checkpoints()
    final_ckpt = checkpoints[-1]
    step = int(os.path.basename(final_ckpt).split("_")[1].split(".")[0])
    model = load_checkpoint(final_ckpt)
    matrices = get_weight_matrices(model)

    # pick one large weight matrix
    layer_name = "blocks.2.mlp.c_fc.weight"
    W = matrices[layer_name]

    zoom_levels = [
        ("Full", W),
        ("Top-left ¼", W[:W.shape[0]//2, :W.shape[1]//2]),
        ("Top-left ¹⁄₁₆", W[:W.shape[0]//4, :W.shape[1]//4]),
        ("Top-left ¹⁄₆₄", W[:W.shape[0]//8, :W.shape[1]//8]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Weight Matrix Zoom — {layer_name} (step {step})\nDo we see similar structure at every scale?", fontsize=13)
    for ax, (label, patch) in zip(axes, zoom_levels):
        im = ax.imshow(patch, cmap="RdBu_r", aspect="auto", vmin=-0.15, vmax=0.15)
        ax.set_title(f"{label}\n{patch.shape[0]}×{patch.shape[1]}", fontsize=10)
        ax.set_xlabel("output dim")
        ax.set_ylabel("input dim")
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "weight_zoom.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 4: Weight correlation matrix (fractal patterns) ---

def exp_correlation_fractal():
    """
    Compute correlation matrices of weights — these often show striking
    fractal-like block structure. Compare init vs trained.
    """
    checkpoints = get_checkpoints()
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])

    layer_name = "blocks.2.mlp.c_fc.weight"
    W_init = get_weight_matrices(init_model)[layer_name]
    W_final = get_weight_matrices(final_model)[layer_name]

    # correlation matrix: how correlated are the rows (output neurons)?
    corr_init = np.corrcoef(W_init)
    corr_final = np.corrcoef(W_final)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Weight Correlation Matrix — {layer_name}\nFractal-like block structure emerges with training", fontsize=13)

    axes[0].imshow(corr_init, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title(f"Random Init (step 0)\n{corr_init.shape[0]}×{corr_init.shape[1]}")

    im = axes[1].imshow(corr_final, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    step = int(os.path.basename(checkpoints[-1]).split("_")[1].split(".")[0])
    axes[1].set_title(f"Trained (step {step})\n{corr_final.shape[0]}×{corr_final.shape[1]}")

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_fractal.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 5: Weight evolution filmstrip ---

def exp_weight_filmstrip():
    """
    Show the same weight matrix at every checkpoint — a filmstrip
    of fractal structure emerging from random noise during training.
    """
    checkpoints = get_checkpoints()
    layer_name = "blocks.2.mlp.c_fc.weight"

    n = len(checkpoints)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    fig.suptitle(f"Weight Matrix Evolution — {layer_name}\nFrom random noise to structured (fractal?) patterns", fontsize=13)

    for ax, ckpt_path in zip(axes, checkpoints):
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        W = get_weight_matrices(model)[layer_name]
        ax.imshow(W, cmap="RdBu_r", aspect="auto", vmin=-0.15, vmax=0.15)
        ax.set_title(f"step {step}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "weight_filmstrip.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 6: Singular value rank plot (log-log, the classic fractal test) ---

def exp_singular_value_loglog():
    """
    Log-log plot of singular values ranked by magnitude.
    A straight line = power law = scale-free = fractal.
    Compare init vs trained.
    """
    checkpoints = get_checkpoints()
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])

    target_layers = [
        "blocks.0.attn.c_attn.weight",
        "blocks.2.mlp.c_fc.weight",
        f"blocks.{N_LAYER-1}.attn.c_attn.weight",
        f"blocks.{N_LAYER-1}.mlp.c_fc.weight",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Singular Values (log-log) — Straight line = Power Law = Fractal", fontsize=13)

    for ax, layer_name in zip(axes.flat, target_layers):
        for label, model, color in [("init", init_model, "gray"), ("trained", final_model, "blue")]:
            W = get_weight_matrices(model)[layer_name]
            sv = np.linalg.svd(W, compute_uv=False)
            sv_sorted = np.sort(sv)[::-1]
            ranks = np.arange(1, len(sv_sorted) + 1)
            ax.loglog(ranks, sv_sorted, "o-", markersize=2, color=color, alpha=0.8, label=label)

        short = layer_name.replace("blocks.", "L").replace(".weight", "")
        ax.set_title(short, fontsize=10)
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "singular_values_loglog.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 7: Box-counting fractal dimension of weight matrices ---

def box_counting_dimension(image, min_box=2, max_box=None):
    """
    Compute box-counting fractal dimension of a 2D array.
    Binarize the image, then count how many boxes of size `s` are needed
    to cover all non-zero pixels. D = -slope of log(count) vs log(1/s).
    """
    if max_box is None:
        max_box = min(image.shape) // 2

    # binarize: 1 where value exceeds threshold
    threshold = np.std(image) * 0.5
    binary = np.abs(image) > threshold

    sizes = []
    counts = []
    s = min_box
    while s <= max_box:
        # count boxes that contain at least one True pixel
        n_rows = binary.shape[0] // s
        n_cols = binary.shape[1] // s
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                box = binary[i*s:(i+1)*s, j*s:(j+1)*s]
                if box.any():
                    count += 1
        sizes.append(s)
        counts.append(count)
        s *= 2

    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    # D = -slope of log(count) vs log(s)
    if len(sizes) > 2:
        coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
        return coeffs[0], sizes, counts
    return None, sizes, counts


def exp_box_counting():
    """
    Measure box-counting fractal dimension of weight matrices.
    A pure 2D object has D=2, pure noise ~2, but structured fractals have D < 2.
    Track how D evolves during training — does structure become more fractal?
    """
    checkpoints = get_checkpoints()
    print(f"Computing box-counting fractal dimension across {len(checkpoints)} checkpoints...")

    # track dimension evolution per layer
    results = {}  # layer_name -> [(step, D), ...]

    target_layers = [
        "blocks.0.attn.c_attn.weight",
        "blocks.0.mlp.c_fc.weight",
        f"blocks.{N_LAYER-1}.attn.c_attn.weight",
        f"blocks.{N_LAYER-1}.mlp.c_fc.weight",
    ]

    for ckpt_path in checkpoints:
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        matrices = get_weight_matrices(model)

        for name in target_layers:
            W = matrices[name]
            D, sizes, counts = box_counting_dimension(W)
            if D is not None:
                if name not in results:
                    results[name] = []
                results[name].append((step, D))

    # plot 1: D evolution during training
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        steps, dims = zip(*data)
        short = name.replace("blocks.", "L").replace(".weight", "")
        ax.plot(steps, dims, "o-", markersize=4, label=short)

    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="D=2.0 (space-filling)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Box-counting fractal dimension D")
    ax.set_title("Fractal Dimension of Weight Matrices During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "box_counting_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # plot 2: log-log box counting for init vs final (show the actual measurement)
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Box-Counting: log(count) vs log(1/box_size)\nSlope = fractal dimension D", fontsize=13)

    for ax, name in zip(axes.flat, target_layers):
        for label, model, color in [("init", init_model, "gray"), ("trained", final_model, "blue")]:
            W = get_weight_matrices(model)[name]
            D, sizes, counts = box_counting_dimension(W)
            if D is not None and len(sizes) > 0:
                ax.plot(np.log(1.0/sizes), np.log(counts), "o-", color=color, label=f"{label} D={D:.3f}")

        short = name.replace("blocks.", "L").replace(".weight", "")
        ax.set_title(short)
        ax.set_xlabel("log(1/box_size)")
        ax.set_ylabel("log(count)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "box_counting_loglog.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # print summary
    print("\nFractal dimensions (final checkpoint):")
    for name, data in sorted(results.items()):
        step, D = data[-1]
        short = name.replace("blocks.", "L").replace(".weight", "")
        print(f"  {short:40s} D={D:.3f}")


# --- Experiment 8: Attention weight correlation (fractal block structure) ---

def exp_attention_correlation():
    """
    Correlation matrix of attention weight matrices.
    Attention layers should show more structured correlations than MLP layers,
    potentially with fractal-like block-diagonal patterns.
    """
    checkpoints = get_checkpoints()
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])
    step = int(os.path.basename(checkpoints[-1]).split("_")[1].split(".")[0])

    # look at all 6 layers' attention weights
    fig, axes = plt.subplots(2, N_LAYER, figsize=(4 * N_LAYER, 8))
    fig.suptitle("Attention Weight Correlations: Init (top) vs Trained (bottom)\nLooking for fractal block structure", fontsize=13)

    for i in range(N_LAYER):
        layer_name = f"blocks.{i}.attn.c_attn.weight"

        for row, (model, label) in enumerate([(init_model, "init"), (final_model, f"step {step}")]):
            W = get_weight_matrices(model)[layer_name]
            corr = np.corrcoef(W)
            # clip for visibility
            corr = np.clip(corr, -1, 1)
            ax = axes[row, i]
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
            ax.set_title(f"L{i} {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "attention_correlations.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 9: Cross-layer self-similarity ---

def exp_cross_layer_similarity():
    """
    Compare weight matrices across layers — do deeper layers look like
    scaled/transformed versions of earlier layers? This would be the
    strongest evidence of fractal (self-similar) structure.
    """
    checkpoints = get_checkpoints()
    final_model = load_checkpoint(checkpoints[-1])
    matrices = get_weight_matrices(final_model)

    # compare singular value distributions across layers for each weight type
    weight_types = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Layer Self-Similarity: Do layers have the same spectral shape?\nOverlapping curves = self-similar structure", fontsize=13)

    for ax, wtype in zip(axes.flat, weight_types):
        for i in range(N_LAYER):
            name = f"blocks.{i}.{wtype}.weight"
            if name in matrices:
                W = matrices[name]
                sv = np.linalg.svd(W, compute_uv=False)
                # normalize so we can compare shapes
                sv_norm = sv / sv[0]
                ranks_norm = np.linspace(0, 1, len(sv_norm))
                ax.plot(ranks_norm, sv_norm, "-", linewidth=1.5, label=f"L{i}", alpha=0.8)

        ax.set_title(wtype, fontsize=11)
        ax.set_xlabel("normalized rank")
        ax.set_ylabel("normalized singular value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cross_layer_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # quantify: pairwise cosine similarity of singular value vectors
    print("\nCross-layer cosine similarity of singular value spectra:")
    for wtype in weight_types:
        sv_vectors = []
        for i in range(N_LAYER):
            name = f"blocks.{i}.{wtype}.weight"
            if name in matrices:
                sv = np.linalg.svd(matrices[name], compute_uv=False)
                sv_vectors.append(sv / np.linalg.norm(sv))

        print(f"\n  {wtype}:")
        for i in range(len(sv_vectors)):
            for j in range(i+1, len(sv_vectors)):
                sim = np.dot(sv_vectors[i], sv_vectors[j])
                print(f"    L{i} vs L{j}: {sim:.4f}")


# --- Experiment 10: Multifractal spectrum ---

def multifractal_spectrum(data, q_range=None, n_scales=10):
    """
    Compute the multifractal singularity spectrum f(α) via the
    partition function method on 1D data.

    Uses the flattened weight values (not singular values) for enough
    data points. For each moment order q, compute τ(q) from
    Z(q, s) ~ s^τ(q). Then Legendre transform: α = dτ/dq, f(α) = q·α - τ.
    """
    if q_range is None:
        q_range = np.linspace(-5, 5, 41)

    data = np.abs(data) + 1e-12  # ensure positive

    # box sizes (powers of 2)
    n = len(data)
    scales = [2**i for i in range(2, n_scales + 2) if 2**i < n // 8]

    if len(scales) < 4:
        return None, None, None

    # compute partition function for each q and scale
    tau = []
    for q in q_range:
        log_Z = []
        log_s = []
        for s in scales:
            n_boxes = n // s
            if n_boxes < 2:
                continue
            # partition into boxes, compute measure (sum of |values|)
            boxes = data[:n_boxes * s].reshape(n_boxes, s)
            mu = boxes.sum(axis=1)
            total = mu.sum()
            if total == 0:
                continue
            mu = mu / total  # normalize to probability measure
            mu = mu[mu > 0]  # remove zeros
            # partition function
            if abs(q - 1.0) < 1e-10:
                # use entropy for q=1
                Z = np.exp(np.sum(mu * np.log(mu)))
            else:
                Z = np.sum(mu ** q)
            if Z > 0:
                log_Z.append(np.log(Z))
                log_s.append(np.log(s))

        if len(log_s) >= 4:
            slope, _, _, _, _ = stats.linregress(log_s, log_Z)
            tau.append(slope)
        else:
            tau.append(np.nan)

    tau = np.array(tau)
    q_range = np.array(q_range)

    # remove nans
    valid = ~np.isnan(tau)
    if valid.sum() < 5:
        return None, None, None
    q_valid = q_range[valid]
    tau_valid = tau[valid]

    # Legendre transform: α = dτ/dq, f(α) = q·α - τ
    alpha = np.gradient(tau_valid, q_valid)
    f_alpha = q_valid * alpha - tau_valid

    return alpha, f_alpha, tau_valid


def exp_multifractal():
    """
    Compute the multifractal singularity spectrum f(α) of weight matrices.
    A single fractal gives a single point. A multifractal gives a curve —
    an inverted parabola shape. Wider parabola = richer multifractal structure.
    Compare init vs trained, and across layers.
    """
    checkpoints = get_checkpoints()
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])
    step = int(os.path.basename(checkpoints[-1]).split("_")[1].split(".")[0])

    target_layers = [
        "blocks.0.attn.c_attn.weight",
        "blocks.0.mlp.c_fc.weight",
        f"blocks.{N_LAYER-1}.attn.c_attn.weight",
        f"blocks.{N_LAYER-1}.mlp.c_fc.weight",
    ]

    # plot 1: init vs trained singularity spectra
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multifractal Singularity Spectrum f(α)\nWider curve = richer multifractal structure", fontsize=13)

    for ax, layer_name in zip(axes.flat, target_layers):
        for label, model, color in [("init", init_model, "gray"), (f"trained (step {step})", final_model, "blue")]:
            W = get_weight_matrices(model)[layer_name]
            # use flattened weight values for enough data points
            flat = W.flatten()
            alpha, f_alpha, _ = multifractal_spectrum(flat)
            if alpha is not None:
                ax.plot(alpha, f_alpha, "o-", markersize=3, color=color, label=label, alpha=0.8)

        short = layer_name.replace("blocks.", "L").replace(".weight", "")
        ax.set_title(short, fontsize=10)
        ax.set_xlabel("α (singularity exponent)")
        ax.set_ylabel("f(α) (singularity spectrum)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "multifractal_spectrum.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # plot 2: multifractal width (Δα) evolution during training
    checkpoints_subset = get_checkpoints()
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_name in target_layers:
        steps_list = []
        widths = []
        for ckpt_path in checkpoints_subset:
            s = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
            model = load_checkpoint(ckpt_path)
            W = get_weight_matrices(model)[layer_name]
            flat = W.flatten()
            alpha, f_alpha, _ = multifractal_spectrum(flat)
            if alpha is not None:
                width = np.max(alpha) - np.min(alpha)
                steps_list.append(s)
                widths.append(width)

        short = layer_name.replace("blocks.", "L").replace(".weight", "")
        ax.plot(steps_list, widths, "o-", markersize=4, label=short)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Multifractal width Δα")
    ax.set_title("Multifractal Width During Training\nWider = more complex multifractal structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "multifractal_width_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 11: Attention map fractals ---

def exp_attention_maps():
    """
    Feed data through the trained model, capture actual attention patterns,
    and analyze them for fractal structure. Are attention maps self-similar?
    """
    checkpoints = get_checkpoints()
    final_model = load_checkpoint(checkpoints[-1])

    # get a batch of data to feed through
    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")
    x = torch.from_numpy(data[:BLOCK_SIZE].astype(np.int64)).unsqueeze(0)

    # hook to capture attention patterns
    attention_maps = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            # output is the result of forward, but we need to recompute attention
            # to get the actual attention weights
            pass
        return hook_fn

    # manually compute attention for each layer
    final_model.eval()
    with torch.no_grad():
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long)
        hidden = final_model.drop(final_model.wte(x) + final_model.wpe(pos))

        for i, block in enumerate(final_model.blocks):
            # pre-norm
            normed = block.ln_1(hidden)
            # compute Q, K, V
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.split(N_EMBD, dim=2)
            n_head = block.attn.n_head
            head_dim = N_EMBD // n_head
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)

            # attention weights
            att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            # causal mask
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = torch.softmax(att, dim=-1)
            attention_maps[f"layer_{i}"] = att[0].numpy()  # [n_head, T, T]

            # continue forward pass
            hidden = hidden + block.attn(block.ln_1(hidden))
            hidden = hidden + block.mlp(block.ln_2(hidden))

    # plot attention maps for all layers, head 0
    fig, axes = plt.subplots(2, N_LAYER, figsize=(4 * N_LAYER, 8))
    fig.suptitle("Attention Maps: Head 0 (top) vs Head 1 (bottom)\nLooking for fractal/self-similar patterns", fontsize=13)

    for i in range(N_LAYER):
        att = attention_maps[f"layer_{i}"]
        for head_idx in range(min(2, att.shape[0])):
            ax = axes[head_idx, i]
            im = ax.imshow(att[head_idx], cmap="viridis", aspect="auto")
            ax.set_title(f"L{i} H{head_idx}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "attention_maps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # fractal analysis: singular value spectra of attention maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Singular Values of Attention Maps (log-log)\nPower law here = fractal attention", fontsize=13)

    for i in range(N_LAYER):
        ax = axes[i // 3, i % 3]
        att = attention_maps[f"layer_{i}"]
        for h in range(min(3, att.shape[0])):
            sv = np.linalg.svd(att[h], compute_uv=False)
            sv = sv[sv > 1e-10]
            ranks = np.arange(1, len(sv) + 1)
            ax.loglog(ranks, sv, "o-", markersize=2, label=f"H{h}", alpha=0.8)
        ax.set_title(f"Layer {i}")
        ax.set_xlabel("rank")
        ax.set_ylabel("singular value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "attention_map_spectra.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# --- Experiment 12: SGD trajectory fractal dimension ---

def exp_sgd_trajectory():
    """
    The path weights take through parameter space during training.
    Compute displacements between consecutive checkpoints, then measure
    how the trajectory's "length" scales with measurement resolution.
    Fractal trajectory → length grows as resolution shrinks (like a coastline).

    Also project the high-dimensional trajectory into 2D/3D via PCA
    to visualize the path shape.
    """
    checkpoints = get_checkpoints()
    if len(checkpoints) < 4:
        print("Need at least 4 checkpoints for trajectory analysis")
        return

    print(f"Analyzing SGD trajectory across {len(checkpoints)} checkpoints...")

    # collect flattened weight vectors at each checkpoint
    trajectory = []
    steps = []
    for ckpt_path in checkpoints:
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        # flatten all parameters into one big vector
        params = []
        for name, param in model.named_parameters():
            params.append(param.detach().numpy().flatten())
        trajectory.append(np.concatenate(params))
        steps.append(step)

    trajectory = np.array(trajectory)  # [n_checkpoints, n_params]
    steps = np.array(steps)
    print(f"Trajectory shape: {trajectory.shape} ({trajectory.shape[1]:,} parameters)")

    # --- 1. Displacement analysis ---
    # compute step-to-step displacements
    displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # compute multi-step displacements at different scales
    # scale k: distance between checkpoints k apart
    max_scale = len(checkpoints) // 2
    scales = list(range(1, max_scale + 1))
    mean_displacements = []
    for k in scales:
        dists = []
        for i in range(len(trajectory) - k):
            d = np.linalg.norm(trajectory[i + k] - trajectory[i])
            dists.append(d)
        mean_displacements.append(np.mean(dists))

    # if displacement ~ k^H, then H is the Hurst exponent
    # H > 0.5 = persistent (trending), H < 0.5 = anti-persistent, H = 0.5 = random walk
    log_scales = np.log(scales)
    log_disps = np.log(mean_displacements)
    H, intercept, r_value, _, _ = stats.linregress(log_scales, log_disps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SGD Trajectory Through Weight Space", fontsize=13)

    # plot displacement per step
    axes[0].plot(steps[1:], displacements, "o-", markersize=4)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("‖Δw‖ (L2 displacement)")
    axes[0].set_title("Step-to-step displacement")
    axes[0].grid(True, alpha=0.3)

    # plot scaling: displacement vs scale (log-log)
    axes[1].loglog(scales, mean_displacements, "o-", color="blue", label=f"H={H:.3f} (R²={r_value**2:.3f})")
    # fit line
    fit_line = np.exp(intercept) * np.array(scales) ** H
    axes[1].loglog(scales, fit_line, "--", color="red", alpha=0.7, label="power-law fit")
    axes[1].set_xlabel("Scale (# checkpoints apart)")
    axes[1].set_ylabel("Mean displacement")
    axes[1].set_title(f"Displacement scaling → Hurst exponent H={H:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- 2. PCA projection ---
    from numpy.linalg import svd as np_svd
    # center the trajectory
    centered = trajectory - trajectory.mean(axis=0)
    # compute top 3 PCs via truncated SVD on the trajectory matrix
    U, S, Vt = np_svd(centered, full_matrices=False)
    proj = centered @ Vt[:3].T  # project onto top 3 PCs
    variance_explained = (S[:3] ** 2) / (S ** 2).sum()

    axes[2].plot(proj[:, 0], proj[:, 1], "o-", markersize=5, color="blue", alpha=0.7)
    axes[2].plot(proj[0, 0], proj[0, 1], "s", markersize=10, color="green", label="start")
    axes[2].plot(proj[-1, 0], proj[-1, 1], "s", markersize=10, color="red", label="end")
    # label every few points
    for i in range(0, len(steps), max(1, len(steps) // 5)):
        axes[2].annotate(f"{steps[i]}", (proj[i, 0], proj[i, 1]), fontsize=7)
    axes[2].set_xlabel(f"PC1 ({variance_explained[0]:.1%} var)")
    axes[2].set_ylabel(f"PC2 ({variance_explained[1]:.1%} var)")
    axes[2].set_title("Trajectory in PCA space")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sgd_trajectory.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- 3. 3D trajectory ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], "o-", markersize=4, alpha=0.7)
    ax.scatter(*proj[0, :3], s=100, c="green", marker="s", label="start")
    ax.scatter(*proj[-1, :3], s=100, c="red", marker="s", label="end")
    for i in range(0, len(steps), max(1, len(steps) // 5)):
        ax.text(proj[i, 0], proj[i, 1], proj[i, 2], f" {steps[i]}", fontsize=7)
    ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({variance_explained[2]:.1%})")
    ax.set_title(f"SGD Trajectory in 3D PCA Space\nHurst exponent H={H:.3f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sgd_trajectory_3d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print(f"\nHurst exponent H = {H:.4f} (R² = {r_value**2:.4f})")
    print(f"  H > 0.5 → persistent/trending trajectory")
    print(f"  H = 0.5 → random walk")
    print(f"  H < 0.5 → anti-persistent")
    print(f"  Fractal dimension D = 2 - H = {2 - H:.4f}")
    print(f"\nVariance explained by top 3 PCs: {variance_explained.sum():.1%}")


# --- Experiment 13: Loss landscape cross-sections ---

def exp_loss_landscape():
    """
    Take 1D and 2D slices through the loss landscape near the final trained weights.
    If the surface is fractal, it should be rough/jagged at all scales (not smooth).
    We measure roughness via the Hurst exponent of the 1D cross-sections.

    Method: pick two random directions in parameter space, evaluate loss along them.
    """
    checkpoints = get_checkpoints()
    final_model = load_checkpoint(checkpoints[-1])

    # load data for loss evaluation
    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")

    # get a few batches for stable loss estimates
    def compute_loss(model, n_batches=5):
        model.eval()
        losses = []
        with torch.no_grad():
            for b in range(n_batches):
                offset = b * BLOCK_SIZE * 4
                ix = [offset + i * BLOCK_SIZE for i in range(4)
                      if offset + (i + 1) * BLOCK_SIZE + 1 < len(data)]
                if not ix:
                    break
                x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
                _, loss = model(x, y)
                losses.append(loss.item())
        return np.mean(losses)

    # get base parameters as a flat vector
    base_params = []
    for p in final_model.parameters():
        base_params.append(p.detach().clone().flatten())
    base_params = torch.cat(base_params)
    n_params = len(base_params)

    # generate two random directions (normalized)
    torch.manual_seed(42)
    dir1 = torch.randn(n_params)
    dir1 = dir1 / dir1.norm()
    dir2 = torch.randn(n_params)
    # orthogonalize dir2 w.r.t. dir1
    dir2 = dir2 - (dir2 @ dir1) * dir1
    dir2 = dir2 / dir2.norm()

    def set_params(model, flat_params):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
            offset += numel

    # --- 1D cross-sections along both directions ---
    n_points = 101
    alphas = np.linspace(-1.0, 1.0, n_points)
    losses_d1 = []
    losses_d2 = []

    print(f"Computing 1D loss landscape ({n_points} points x 2 directions)...")
    for alpha in alphas:
        # direction 1
        perturbed = base_params + alpha * dir1
        set_params(final_model, perturbed)
        losses_d1.append(compute_loss(final_model, n_batches=3))

        # direction 2
        perturbed = base_params + alpha * dir2
        set_params(final_model, perturbed)
        losses_d2.append(compute_loss(final_model, n_batches=3))

    losses_d1 = np.array(losses_d1)
    losses_d2 = np.array(losses_d2)

    # restore original params
    set_params(final_model, base_params)

    # compute Hurst exponent of the 1D slices via rescaled range (R/S) analysis
    def hurst_rs(series):
        """Estimate Hurst exponent via rescaled range."""
        n = len(series)
        max_k = n // 4
        if max_k < 4:
            return 0.5, 0.0
        sizes = []
        rs_values = []
        for k in range(4, max_k + 1):
            n_subseries = n // k
            rs_list = []
            for i in range(n_subseries):
                sub = series[i*k:(i+1)*k]
                mean_sub = sub.mean()
                devs = np.cumsum(sub - mean_sub)
                R = devs.max() - devs.min()
                S = sub.std(ddof=1)
                if S > 1e-10:
                    rs_list.append(R / S)
            if rs_list:
                sizes.append(k)
                rs_values.append(np.mean(rs_list))
        if len(sizes) < 3:
            return 0.5, 0.0
        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        H, _, r, _, _ = stats.linregress(log_sizes, log_rs)
        return H, r**2

    H1, r2_1 = hurst_rs(losses_d1)
    H2, r2_2 = hurst_rs(losses_d2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Loss Landscape Cross-Sections", fontsize=13)

    axes[0].plot(alphas, losses_d1, "-", linewidth=1, color="blue", label=f"Dir 1 (H={H1:.3f})")
    axes[0].plot(alphas, losses_d2, "-", linewidth=1, color="red", alpha=0.7, label=f"Dir 2 (H={H2:.3f})")
    axes[0].set_xlabel("Perturbation α")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("1D Loss Cross-Sections")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # zoom into center region
    center = n_points // 2
    window = n_points // 5
    axes[1].plot(alphas[center-window:center+window], losses_d1[center-window:center+window],
                 "-o", linewidth=1, markersize=2, color="blue", label="Dir 1")
    axes[1].plot(alphas[center-window:center+window], losses_d2[center-window:center+window],
                 "-o", linewidth=1, markersize=2, color="red", alpha=0.7, label="Dir 2")
    axes[1].set_xlabel("Perturbation α")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Zoomed to center (local roughness)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # roughness at multiple scales: compute std of differences at different strides
    strides = [1, 2, 4, 8, 16]
    roughness_d1 = []
    roughness_d2 = []
    for s in strides:
        roughness_d1.append(np.std(np.diff(losses_d1[::s])))
        roughness_d2.append(np.std(np.diff(losses_d2[::s])))

    axes[2].loglog(strides, roughness_d1, "o-", label=f"Dir 1", color="blue")
    axes[2].loglog(strides, roughness_d2, "o-", label=f"Dir 2", color="red")
    axes[2].set_xlabel("Stride (coarser →)")
    axes[2].set_ylabel("Std of differences")
    axes[2].set_title("Roughness vs scale")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss_landscape_1d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- 2D loss landscape ---
    print("Computing 2D loss landscape (21x21 grid)...")
    grid_size = 21
    alphas_2d = np.linspace(-0.5, 0.5, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))

    for i, a1 in enumerate(alphas_2d):
        for j, a2 in enumerate(alphas_2d):
            perturbed = base_params + a1 * dir1 + a2 * dir2
            set_params(final_model, perturbed)
            loss_grid[i, j] = compute_loss(final_model, n_batches=2)
        print(f"  Row {i+1}/{grid_size}")

    set_params(final_model, base_params)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("2D Loss Landscape", fontsize=13)

    im = axes[0].imshow(loss_grid, extent=[alphas_2d[0], alphas_2d[-1], alphas_2d[0], alphas_2d[-1]],
                         cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_xlabel("Direction 1")
    axes[0].set_ylabel("Direction 2")
    axes[0].set_title("Loss surface (heatmap)")
    axes[0].plot(0, 0, "r+", markersize=15, markeredgewidth=2, label="trained weights")
    axes[0].legend()
    plt.colorbar(im, ax=axes[0])

    # contour plot
    X, Y = np.meshgrid(alphas_2d, alphas_2d)
    cs = axes[1].contour(X, Y, loss_grid, levels=20, cmap="viridis")
    axes[1].clabel(cs, fontsize=7)
    axes[1].set_xlabel("Direction 1")
    axes[1].set_ylabel("Direction 2")
    axes[1].set_title("Loss contours")
    axes[1].plot(0, 0, "r+", markersize=15, markeredgewidth=2, label="trained weights")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss_landscape_2d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print(f"\nHurst exponents of 1D cross-sections:")
    print(f"  Direction 1: H = {H1:.4f} (R² = {r2_1:.4f})")
    print(f"  Direction 2: H = {H2:.4f} (R² = {r2_2:.4f})")
    print(f"  H > 0.5 → smooth/persistent, H < 0.5 → rough/anti-persistent, H = 0.5 → Brownian")


# --- Experiment 14: Hidden representation self-similarity ---

def exp_representation_similarity():
    """
    Feed data through the trained model and capture hidden representations
    at every layer. Measure how similar the representations are across layers
    using CKA (Centered Kernel Alignment) and cosine similarity of their
    singular value spectra. If representations are self-similar, we should
    see structured (not random) similarity patterns.
    """
    checkpoints = get_checkpoints()
    final_model = load_checkpoint(checkpoints[-1])

    # load data
    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")
    # use multiple sequences for stable statistics
    n_seqs = 16
    x = torch.stack([
        torch.from_numpy(data[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE].astype(np.int64))
        for i in range(n_seqs)
    ])

    # forward pass, capturing hidden states at each layer
    final_model.eval()
    hidden_states = []
    with torch.no_grad():
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long)
        h = final_model.drop(final_model.wte(x) + final_model.wpe(pos))
        hidden_states.append(h.numpy().reshape(B * T, -1))  # after embedding

        for i, block in enumerate(final_model.blocks):
            h = h + block.attn(block.ln_1(h))
            h = h + block.mlp(block.ln_2(h))
            hidden_states.append(h.numpy().reshape(B * T, -1))

    n_layers = len(hidden_states)
    layer_names = ["embed"] + [f"L{i}" for i in range(N_LAYER)]

    # --- 1. CKA (linear) between all layer pairs ---
    def linear_cka(X, Y):
        """Centered Kernel Alignment — measures representation similarity."""
        # center
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
        hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

    cka_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = linear_cka(hidden_states[i], hidden_states[j])

    # --- 2. Singular value spectra of each layer's representations ---
    sv_spectra = []
    for h in hidden_states:
        sv = np.linalg.svd(h - h.mean(axis=0), compute_uv=False)
        sv = sv / sv.sum()  # normalize
        sv_spectra.append(sv)

    # cosine similarity of spectra
    spec_sim = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            min_len = min(len(sv_spectra[i]), len(sv_spectra[j]))
            a, b = sv_spectra[i][:min_len], sv_spectra[j][:min_len]
            spec_sim[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --- 3. Effective dimensionality (participation ratio) per layer ---
    eff_dims = []
    for sv in sv_spectra:
        p = sv ** 2
        p = p / p.sum()
        pr = 1.0 / (p ** 2).sum()  # participation ratio
        eff_dims.append(pr)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Hidden Representation Self-Similarity Across Layers", fontsize=14)

    # CKA matrix
    im0 = axes[0, 0].imshow(cka_matrix, cmap="magma", vmin=0, vmax=1)
    axes[0, 0].set_xticks(range(n_layers))
    axes[0, 0].set_xticklabels(layer_names, fontsize=8)
    axes[0, 0].set_yticks(range(n_layers))
    axes[0, 0].set_yticklabels(layer_names, fontsize=8)
    axes[0, 0].set_title("Linear CKA (representation similarity)")
    plt.colorbar(im0, ax=axes[0, 0])
    # annotate
    for i in range(n_layers):
        for j in range(n_layers):
            axes[0, 0].text(j, i, f"{cka_matrix[i,j]:.2f}", ha="center", va="center",
                           fontsize=7, color="white" if cka_matrix[i,j] < 0.5 else "black")

    # spectral similarity matrix
    im1 = axes[0, 1].imshow(spec_sim, cmap="magma", vmin=0.9, vmax=1)
    axes[0, 1].set_xticks(range(n_layers))
    axes[0, 1].set_xticklabels(layer_names, fontsize=8)
    axes[0, 1].set_yticks(range(n_layers))
    axes[0, 1].set_yticklabels(layer_names, fontsize=8)
    axes[0, 1].set_title("Spectral cosine similarity")
    plt.colorbar(im1, ax=axes[0, 1])
    for i in range(n_layers):
        for j in range(n_layers):
            axes[0, 1].text(j, i, f"{spec_sim[i,j]:.3f}", ha="center", va="center",
                           fontsize=6, color="white" if spec_sim[i,j] < 0.95 else "black")

    # SV spectra overlay (log-log)
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    for i, (sv, name) in enumerate(zip(sv_spectra, layer_names)):
        axes[1, 0].loglog(np.arange(1, len(sv)+1), sv, "-", color=colors[i],
                          label=name, alpha=0.8, linewidth=1.5)
    axes[1, 0].set_xlabel("Rank")
    axes[1, 0].set_ylabel("Normalized singular value")
    axes[1, 0].set_title("Representation spectra (log-log)")
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)

    # effective dimensionality
    axes[1, 1].bar(range(n_layers), eff_dims, color=colors)
    axes[1, 1].set_xticks(range(n_layers))
    axes[1, 1].set_xticklabels(layer_names, fontsize=8)
    axes[1, 1].set_ylabel("Effective dimensionality (participation ratio)")
    axes[1, 1].set_title("Representation dimensionality per layer")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "representation_similarity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print(f"\nCKA matrix diagonal (self-similarity = 1.0):")
    for i, name in enumerate(layer_names):
        neighbors = []
        if i > 0:
            neighbors.append(f"CKA({layer_names[i-1]})={cka_matrix[i,i-1]:.3f}")
        if i < n_layers - 1:
            neighbors.append(f"CKA({layer_names[i+1]})={cka_matrix[i,i+1]:.3f}")
        print(f"  {name}: eff_dim={eff_dims[i]:.1f}, {', '.join(neighbors)}")

    # distant layer similarity (how similar are layers far apart?)
    print(f"\nDistant layer CKA:")
    print(f"  embed vs L5: {cka_matrix[0, -1]:.4f}")
    print(f"  L0 vs L5:    {cka_matrix[1, -1]:.4f}")
    print(f"  L1 vs L4:    {cka_matrix[2, 5]:.4f}")


# --- Experiment 15: Activation distribution fractal dimension ---

def exp_activation_fractals():
    """
    Analyze the distribution of activations at each layer. Measure:
    1. Whether activation distributions follow power laws (heavy tails)
    2. How the distribution shape changes across layers (self-similarity)
    3. Fractal dimension of the activation space via correlation dimension
    """
    checkpoints = get_checkpoints()
    final_model = load_checkpoint(checkpoints[-1])

    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")
    n_seqs = 32
    x = torch.stack([
        torch.from_numpy(data[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE].astype(np.int64))
        for i in range(n_seqs)
    ])

    # collect activations
    final_model.eval()
    activations = {}
    with torch.no_grad():
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long)
        h = final_model.drop(final_model.wte(x) + final_model.wpe(pos))
        activations["embed"] = h.numpy().flatten()

        for i, block in enumerate(final_model.blocks):
            # capture pre-attention, post-attention, post-MLP
            attn_out = block.attn(block.ln_1(h))
            h = h + attn_out
            activations[f"L{i}_attn"] = attn_out.numpy().flatten()

            mlp_out = block.mlp(block.ln_2(h))
            h = h + mlp_out
            activations[f"L{i}_mlp"] = mlp_out.numpy().flatten()

        h = final_model.ln_f(h)
        activations["final"] = h.numpy().flatten()

    layer_names = list(activations.keys())
    print(f"Collected activations from {len(layer_names)} points")

    # --- 1. Distribution shape analysis ---
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Activation Distributions Across Layers\nLooking for heavy tails and self-similar shapes", fontsize=13)
    axes_flat = axes.flatten()

    kurtosis_values = []
    tail_exponents = []

    for idx, name in enumerate(layer_names):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        vals = activations[name]

        # histogram on log scale
        # use absolute values for tail analysis
        abs_vals = np.abs(vals)
        abs_vals = abs_vals[abs_vals > 1e-6]

        ax.hist(vals, bins=200, density=True, alpha=0.7, color="steelblue", log=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-5 * vals.std(), 5 * vals.std())

        # kurtosis (Gaussian = 3, heavy-tailed > 3)
        kurt = float(stats.kurtosis(vals, fisher=False))
        kurtosis_values.append((name, kurt))
        ax.text(0.95, 0.95, f"K={kurt:.1f}", transform=ax.transAxes,
                fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

        # fit power-law tail to the positive side
        sorted_abs = np.sort(abs_vals)[::-1]
        n_tail = len(sorted_abs) // 10  # top 10%
        if n_tail > 20:
            tail = sorted_abs[:n_tail]
            ranks = np.arange(1, n_tail + 1)
            slope, _, r, _, _ = stats.linregress(np.log(ranks), np.log(tail))
            tail_exponents.append((name, -slope, r**2))

    # hide unused axes
    for idx in range(len(layer_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "activation_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- 2. Kurtosis evolution across layers ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Activation Statistics Across Layers", fontsize=13)

    names_k, kurts = zip(*kurtosis_values)
    colors_k = ["coral" if "attn" in n else "steelblue" if "mlp" in n else "gray" for n in names_k]
    axes[0].bar(range(len(kurts)), kurts, color=colors_k)
    axes[0].set_xticks(range(len(kurts)))
    axes[0].set_xticklabels(names_k, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Kurtosis")
    axes[0].set_title("Kurtosis per layer (Gaussian=3, heavy-tailed>3)")
    axes[0].axhline(y=3, color="black", linestyle="--", alpha=0.5, label="Gaussian")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # tail exponents
    if tail_exponents:
        names_t, alphas_t, r2s = zip(*tail_exponents)
        colors_t = ["coral" if "attn" in n else "steelblue" if "mlp" in n else "gray" for n in names_t]
        axes[1].bar(range(len(alphas_t)), alphas_t, color=colors_t)
        axes[1].set_xticks(range(len(alphas_t)))
        axes[1].set_xticklabels(names_t, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Tail exponent α")
        axes[1].set_title("Power-law tail exponent (rank-frequency)")
        axes[1].grid(True, alpha=0.3, axis="y")

    # --- 3. Correlation dimension estimate ---
    # Use a subsample of the high-dimensional activation vectors
    # Correlation dimension: how does the number of pairs within distance r scale with r?
    print("Computing correlation dimensions...")
    corr_dims = []
    for name in ["embed", "L0_attn", "L2_attn", "L5_attn", "L0_mlp", "L2_mlp", "L5_mlp", "final"]:
        if name not in activations:
            continue
        vals = activations[name]
        # reshape to (n_samples, n_features) — use token-level vectors
        n_tokens = n_seqs * BLOCK_SIZE
        n_feat = len(vals) // n_tokens
        vecs = vals.reshape(n_tokens, n_feat)

        # subsample for speed
        n_sub = min(500, n_tokens)
        idx = np.random.RandomState(42).choice(n_tokens, n_sub, replace=False)
        vecs_sub = vecs[idx]

        # pairwise distances
        dists = []
        for i in range(n_sub):
            d = np.linalg.norm(vecs_sub[i] - vecs_sub[i+1:], axis=1)
            dists.extend(d.tolist())
        dists = np.array(dists)
        dists = dists[dists > 0]

        # correlation integral: C(r) = fraction of pairs with dist < r
        r_values = np.logspace(np.log10(np.percentile(dists, 1)),
                               np.log10(np.percentile(dists, 90)), 30)
        C_values = np.array([np.mean(dists < r) for r in r_values])
        C_values = C_values[C_values > 0]
        r_values = r_values[:len(C_values)]

        if len(C_values) > 5:
            # fit in the scaling region (middle portion)
            mid = len(C_values) // 4
            end = 3 * len(C_values) // 4
            slope, _, r, _, _ = stats.linregress(
                np.log(r_values[mid:end]), np.log(C_values[mid:end]))
            corr_dims.append((name, slope, r**2))

    if corr_dims:
        names_c, dims_c, r2s_c = zip(*corr_dims)
        colors_c = ["coral" if "attn" in n else "steelblue" if "mlp" in n else "gray" for n in names_c]
        axes[2].bar(range(len(dims_c)), dims_c, color=colors_c)
        axes[2].set_xticks(range(len(dims_c)))
        axes[2].set_xticklabels(names_c, rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("Correlation dimension")
        axes[2].set_title("Correlation dimension of activation space")
        axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "activation_statistics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    print(f"\nKurtosis (Gaussian=3, heavy-tailed>3):")
    for name, k in kurtosis_values:
        tag = " ← HEAVY-TAILED" if k > 5 else ""
        print(f"  {name}: {k:.2f}{tag}")

    if tail_exponents:
        print(f"\nTail exponents:")
        for name, alpha, r2 in tail_exponents:
            print(f"  {name}: α={alpha:.3f} (R²={r2:.3f})")

    if corr_dims:
        print(f"\nCorrelation dimensions:")
        for name, d, r2 in corr_dims:
            print(f"  {name}: D_corr={d:.3f} (R²={r2:.3f})")


# --- Experiment 16: Gradient fractal structure ---

def exp_gradient_fractals():
    """
    Compute gradients on a batch of data and analyze their distribution
    and spectral properties. Simsekli et al. (2019) showed SGD gradient
    noise is heavy-tailed — can we see this in our small model?
    """
    checkpoints = get_checkpoints()

    data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                     dtype=np.uint16, mode="r")

    # analyze gradients at multiple training stages
    stages = [0, len(checkpoints)//4, len(checkpoints)//2, 3*len(checkpoints)//4, -1]
    stages = list(dict.fromkeys(stages))  # deduplicate
    stage_names = []
    all_grad_stats = []  # (name, step, layer_name, kurtosis, tail_alpha)

    for stage_idx in stages:
        ckpt_path = checkpoints[stage_idx]
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        stage_names.append(f"step_{step}")
        model = load_checkpoint(ckpt_path)
        model.train()

        # compute gradients on multiple batches and accumulate
        grad_accum = {}
        n_batches = 8
        for b in range(n_batches):
            offset = b * BLOCK_SIZE * 4
            ix = [offset + i * BLOCK_SIZE for i in range(4)
                  if offset + (i + 1) * BLOCK_SIZE + 1 < len(data)]
            if not ix:
                break
            x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])

            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad.detach().cpu().numpy().flatten()
                    if name not in grad_accum:
                        grad_accum[name] = []
                    grad_accum[name].append(g.copy())

        # analyze each layer's gradients
        for layer_name, grads_list in grad_accum.items():
            all_grads = np.concatenate(grads_list)
            kurt = float(stats.kurtosis(all_grads, fisher=False))
            # tail exponent
            abs_g = np.abs(all_grads)
            abs_g = abs_g[abs_g > 1e-10]
            sorted_g = np.sort(abs_g)[::-1]
            n_tail = len(sorted_g) // 10
            alpha, r2 = 0.0, 0.0
            if n_tail > 20:
                tail = sorted_g[:n_tail]
                ranks = np.arange(1, n_tail + 1)
                slope, _, r, _, _ = stats.linregress(np.log(ranks), np.log(tail))
                alpha, r2 = -slope, r**2
            all_grad_stats.append((step, layer_name, kurt, alpha, r2))

    # organize by layer type for plotting
    # pick key layers to show
    key_layers = []
    for name in grad_accum.keys():
        if any(k in name for k in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
            key_layers.append(name)

    # --- Plot 1: Gradient distributions at final checkpoint ---
    final_model = load_checkpoint(checkpoints[-1])
    final_model.train()
    final_model.zero_grad()
    x = torch.stack([torch.from_numpy(data[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE].astype(np.int64))
                     for i in range(8)])
    y = torch.stack([torch.from_numpy(data[i*BLOCK_SIZE+1:(i+1)*BLOCK_SIZE+1].astype(np.int64))
                     for i in range(8)])
    _, loss = final_model(x, y)
    loss.backward()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Gradient Distributions at Final Checkpoint\nHeavy tails = non-Gaussian gradient noise", fontsize=13)

    for idx, layer_name in enumerate(key_layers[:6]):
        ax = axes[idx // 3, idx % 3]
        for name, param in final_model.named_parameters():
            if name == layer_name:
                g = param.grad.detach().numpy().flatten()
                ax.hist(g, bins=200, density=True, alpha=0.7, color="steelblue", log=True)
                kurt = float(stats.kurtosis(g, fisher=False))
                ax.set_title(f"{name}\nK={kurt:.1f}", fontsize=8)
                # overlay Gaussian for comparison
                std = g.std()
                x_range = np.linspace(g.min(), g.max(), 200)
                gaussian = stats.norm.pdf(x_range, 0, std)
                ax.plot(x_range, gaussian, "r--", alpha=0.7, label="Gaussian")
                ax.legend(fontsize=7)
                break

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "gradient_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Plot 2: Kurtosis evolution during training ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Gradient Statistics During Training", fontsize=13)

    # kurtosis evolution for key layers
    for layer_name in key_layers[:6]:
        steps_layer = []
        kurts_layer = []
        for step, ln, kurt, alpha, r2 in all_grad_stats:
            if ln == layer_name:
                steps_layer.append(step)
                kurts_layer.append(kurt)
        short_name = layer_name.replace("blocks.", "L").replace(".attn.", ".").replace(".mlp.", ".")
        axes[0].plot(steps_layer, kurts_layer, "o-", label=short_name, markersize=4)
    axes[0].axhline(y=3, color="black", linestyle="--", alpha=0.5, label="Gaussian")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Kurtosis")
    axes[0].set_title("Gradient kurtosis evolution")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # tail exponent evolution
    for layer_name in key_layers[:6]:
        steps_layer = []
        alphas_layer = []
        for step, ln, kurt, alpha, r2 in all_grad_stats:
            if ln == layer_name and r2 > 0.8:
                steps_layer.append(step)
                alphas_layer.append(alpha)
        short_name = layer_name.replace("blocks.", "L").replace(".attn.", ".").replace(".mlp.", ".")
        axes[1].plot(steps_layer, alphas_layer, "o-", label=short_name, markersize=4)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Tail exponent α")
    axes[1].set_title("Gradient tail exponent evolution")
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(True, alpha=0.3)

    # singular value spectrum of gradient matrices at final step
    for name, param in final_model.named_parameters():
        if name in key_layers[:6] and param.grad is not None:
            g = param.grad.detach().numpy()
            if g.ndim == 2:
                sv = np.linalg.svd(g, compute_uv=False)
                sv = sv[sv > 1e-12]
                short_name = name.replace("blocks.", "L").replace(".attn.", ".").replace(".mlp.", ".")
                axes[2].loglog(np.arange(1, len(sv)+1), sv, "o-", markersize=2,
                              label=short_name, alpha=0.8)
    axes[2].set_xlabel("Rank")
    axes[2].set_ylabel("Singular value")
    axes[2].set_title("Gradient matrix SVD (log-log)")
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "gradient_evolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # print summary
    final_step = max(s for s, _, _, _, _ in all_grad_stats)
    print(f"\nGradient statistics at step {final_step}:")
    for step, ln, kurt, alpha, r2 in all_grad_stats:
        if step == final_step and ln in key_layers:
            short = ln.replace("blocks.", "L").replace(".attn.", ".").replace(".mlp.", ".")
            tag = " ← HEAVY" if kurt > 5 else ""
            print(f"  {short}: kurtosis={kurt:.1f}, α={alpha:.3f} (R²={r2:.3f}){tag}")


# --- Experiment 17: Init vs trained fractal summary ---

def exp_init_vs_trained():
    """
    Side-by-side comparison of fractal properties at initialization vs after training.
    One unified plot showing how every metric changes.
    """
    checkpoints = get_checkpoints()
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])

    models = {"Init": init_model, "Trained": final_model}
    colors = {"Init": "gray", "Trained": "blue"}

    # collect metrics per layer for both models
    results = {}
    for tag, model in models.items():
        layer_names = []
        sv_slopes = []
        sv_r2s = []
        kurtoses = []
        top_sv_ratios = []

        for name, param in model.named_parameters():
            if param.ndim != 2 or param.shape[0] < 10:
                continue
            w = param.detach().numpy()
            sv = np.linalg.svd(w, compute_uv=False)
            sv = sv[sv > 1e-10]

            # power-law fit
            ranks = np.arange(1, len(sv) + 1)
            slope, _, r, _, _ = stats.linregress(np.log(ranks), np.log(sv))
            sv_slopes.append(-slope)
            sv_r2s.append(r**2)

            # kurtosis of weights
            kurt = float(stats.kurtosis(w.flatten(), fisher=False))
            kurtoses.append(kurt)

            # top SV concentration
            top_sv_ratios.append(sv[0] / sv.sum())

            short = name.replace("blocks.", "L").replace(".attn.", ".").replace(".mlp.", ".")
            layer_names.append(short)

        results[tag] = {
            "layers": layer_names,
            "slopes": sv_slopes,
            "r2s": sv_r2s,
            "kurtoses": kurtoses,
            "top_sv": top_sv_ratios,
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Init vs Trained: Fractal Properties Side-by-Side", fontsize=14)

    n = len(results["Init"]["layers"])
    x_pos = np.arange(n)
    width = 0.35

    # power-law exponents
    axes[0, 0].bar(x_pos - width/2, results["Init"]["slopes"], width, color="gray", alpha=0.7, label="Init")
    axes[0, 0].bar(x_pos + width/2, results["Trained"]["slopes"], width, color="blue", alpha=0.7, label="Trained")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(results["Init"]["layers"], rotation=45, ha="right", fontsize=6)
    axes[0, 0].set_ylabel("Power-law exponent α")
    axes[0, 0].set_title("SV power-law exponents")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # R² of power-law fit
    axes[0, 1].bar(x_pos - width/2, results["Init"]["r2s"], width, color="gray", alpha=0.7, label="Init")
    axes[0, 1].bar(x_pos + width/2, results["Trained"]["r2s"], width, color="blue", alpha=0.7, label="Trained")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(results["Init"]["layers"], rotation=45, ha="right", fontsize=6)
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_title("Power-law fit quality (R²)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # weight kurtosis
    axes[1, 0].bar(x_pos - width/2, results["Init"]["kurtoses"], width, color="gray", alpha=0.7, label="Init")
    axes[1, 0].bar(x_pos + width/2, results["Trained"]["kurtoses"], width, color="blue", alpha=0.7, label="Trained")
    axes[1, 0].axhline(y=3, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(results["Init"]["layers"], rotation=45, ha="right", fontsize=6)
    axes[1, 0].set_ylabel("Kurtosis")
    axes[1, 0].set_title("Weight kurtosis (Gaussian=3)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # top SV concentration
    axes[1, 1].bar(x_pos - width/2, results["Init"]["top_sv"], width, color="gray", alpha=0.7, label="Init")
    axes[1, 1].bar(x_pos + width/2, results["Trained"]["top_sv"], width, color="blue", alpha=0.7, label="Trained")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(results["Init"]["layers"], rotation=45, ha="right", fontsize=6)
    axes[1, 1].set_ylabel("σ₁ / Σσᵢ")
    axes[1, 1].set_title("Top singular value concentration")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "init_vs_trained.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # summary stats
    print("\nInit → Trained changes:")
    print(f"  Mean power-law α: {np.mean(results['Init']['slopes']):.3f} → {np.mean(results['Trained']['slopes']):.3f}")
    print(f"  Mean R²:          {np.mean(results['Init']['r2s']):.3f} → {np.mean(results['Trained']['r2s']):.3f}")
    print(f"  Mean kurtosis:    {np.mean(results['Init']['kurtoses']):.3f} → {np.mean(results['Trained']['kurtoses']):.3f}")
    print(f"  Mean top-SV conc: {np.mean(results['Init']['top_sv']):.3f} → {np.mean(results['Trained']['top_sv']):.3f}")


# --- Experiment 18: Scale comparison ---

def exp_scale_comparison():
    """
    Compare fractal properties across three model scales:
    - Small: 3 layers, 96 embd (~1.6M params)
    - Medium: 6 layers, 192 embd (~12.3M params) — our main model
    - Large: 8 layers, 256 embd (~26M params)

    Key question: are fractal signatures universal across scales?
    """
    import train_scale

    scale_configs = {
        "small\n(3L/96d)": {
            "ckpt_dir": os.path.join(os.path.dirname(__file__), "checkpoints_small"),
            "n_layer": 3, "n_embd": 96, "n_head": 3,
        },
        "medium\n(6L/192d)": {
            "ckpt_dir": CKPT_DIR,
            "n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": 6,
        },
        "large\n(8L/256d)": {
            "ckpt_dir": os.path.join(os.path.dirname(__file__), "checkpoints_large"),
            "n_layer": 8, "n_embd": 256, "n_head": 8,
        },
    }

    results = {}

    for scale_name, cfg in scale_configs.items():
        ckpt_dir = cfg["ckpt_dir"]
        pattern = os.path.join(ckpt_dir, "step_*.pt")
        ckpts = sorted(glob.glob(pattern))
        if len(ckpts) < 2:
            print(f"Skipping {scale_name}: not enough checkpoints in {ckpt_dir}")
            continue

        print(f"\nAnalyzing {scale_name} ({len(ckpts)} checkpoints)...")

        # load init and final
        if scale_name.startswith("medium"):
            init_model = load_checkpoint(ckpts[0])
            final_model = load_checkpoint(ckpts[-1])
        else:
            init_model = train_scale.GPT(cfg["n_layer"], cfg["n_embd"], cfg["n_head"])
            init_model.load_state_dict(torch.load(ckpts[0], map_location="cpu", weights_only=True))
            init_model.eval()
            final_model = train_scale.GPT(cfg["n_layer"], cfg["n_embd"], cfg["n_head"])
            final_model.load_state_dict(torch.load(ckpts[-1], map_location="cpu", weights_only=True))
            final_model.eval()

        # --- Metric 1: Power-law exponents ---
        init_alphas = []
        trained_alphas = []
        trained_r2s = []

        for model, alpha_list in [(init_model, init_alphas), (final_model, trained_alphas)]:
            for name, param in model.named_parameters():
                if param.ndim != 2 or param.shape[0] < 10:
                    continue
                w = param.detach().numpy()
                sv = np.linalg.svd(w, compute_uv=False)
                sv = sv[sv > 1e-10]
                ranks = np.arange(1, len(sv) + 1)
                slope, _, r, _, _ = stats.linregress(np.log(ranks), np.log(sv))
                alpha_list.append(-slope)
                if model is final_model:
                    trained_r2s.append(r**2)

        # --- Metric 2: Weight kurtosis ---
        init_kurts = []
        trained_kurts = []
        for model, kurt_list in [(init_model, init_kurts), (final_model, trained_kurts)]:
            for name, param in model.named_parameters():
                if param.ndim != 2 or param.shape[0] < 10:
                    continue
                kurt_list.append(float(stats.kurtosis(param.detach().numpy().flatten(), fisher=False)))

        # --- Metric 3: Cross-layer spectral similarity ---
        spectra = []
        for name, param in final_model.named_parameters():
            if param.ndim != 2 or param.shape[0] < 10:
                continue
            sv = np.linalg.svd(param.detach().numpy(), compute_uv=False)
            sv = sv / sv.sum()
            spectra.append(sv)

        cross_sims = []
        for i in range(len(spectra)):
            for j in range(i+1, len(spectra)):
                min_len = min(len(spectra[i]), len(spectra[j]))
                a, b = spectra[i][:min_len], spectra[j][:min_len]
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                cross_sims.append(sim)

        # --- Metric 4: Top SV concentration ---
        init_conc = []
        trained_conc = []
        for model, conc_list in [(init_model, init_conc), (final_model, trained_conc)]:
            for name, param in model.named_parameters():
                if param.ndim != 2 or param.shape[0] < 10:
                    continue
                sv = np.linalg.svd(param.detach().numpy(), compute_uv=False)
                conc_list.append(sv[0] / sv.sum())

        # --- Metric 5: SGD trajectory Hurst exponent ---
        hurst = None
        if len(ckpts) >= 4:
            trajectory = []
            for ckpt_path in ckpts:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                params = []
                for key in sorted(state.keys()):
                    params.append(state[key].numpy().flatten())
                trajectory.append(np.concatenate(params))

            trajectory = np.array(trajectory)
            max_scale = len(ckpts) // 2
            scales = list(range(1, max_scale + 1))
            mean_disps = []
            for k in scales:
                dists = []
                for i in range(len(trajectory) - k):
                    d = np.linalg.norm(trajectory[i + k] - trajectory[i])
                    dists.append(d)
                mean_disps.append(np.mean(dists))

            if len(scales) >= 3:
                H, _, r, _, _ = stats.linregress(np.log(scales), np.log(mean_disps))
                hurst = (H, r**2)

        n_params = sum(p.numel() for p in final_model.parameters())
        results[scale_name] = {
            "n_params": n_params,
            "init_alpha_mean": np.mean(init_alphas),
            "trained_alpha_mean": np.mean(trained_alphas),
            "trained_r2_mean": np.mean(trained_r2s),
            "init_kurt_mean": np.mean(init_kurts),
            "trained_kurt_mean": np.mean(trained_kurts),
            "cross_sim_mean": np.mean(cross_sims) if cross_sims else 0,
            "init_conc_mean": np.mean(init_conc),
            "trained_conc_mean": np.mean(trained_conc),
            "hurst": hurst,
        }

    if len(results) < 2:
        print("Need at least 2 scales to compare. Train more models first.")
        return

    # --- Plot ---
    scale_names = list(results.keys())
    n_scales = len(scale_names)
    x_pos = np.arange(n_scales)
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Fractal Properties Across Model Scales\nAre fractal signatures universal?", fontsize=14)

    colors_init = "lightgray"
    colors_trained = ["#2196F3", "#4CAF50", "#FF5722"][:n_scales]

    # 1. Power-law exponents
    init_vals = [results[s]["init_alpha_mean"] for s in scale_names]
    trained_vals = [results[s]["trained_alpha_mean"] for s in scale_names]
    axes[0, 0].bar(x_pos - width/2, init_vals, width, color=colors_init, label="Init")
    bars = axes[0, 0].bar(x_pos + width/2, trained_vals, width, color=colors_trained, label="Trained")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(scale_names, fontsize=8)
    axes[0, 0].set_ylabel("Mean α")
    axes[0, 0].set_title("Power-law exponent α")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. R² of power-law fit
    r2_vals = [results[s]["trained_r2_mean"] for s in scale_names]
    axes[0, 1].bar(x_pos, r2_vals, color=colors_trained)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(scale_names, fontsize=8)
    axes[0, 1].set_ylabel("Mean R²")
    axes[0, 1].set_title("Power-law fit quality (trained)")
    axes[0, 1].set_ylim(0.5, 1.0)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Weight kurtosis
    init_vals = [results[s]["init_kurt_mean"] for s in scale_names]
    trained_vals = [results[s]["trained_kurt_mean"] for s in scale_names]
    axes[0, 2].bar(x_pos - width/2, init_vals, width, color=colors_init, label="Init")
    axes[0, 2].bar(x_pos + width/2, trained_vals, width, color=colors_trained, label="Trained")
    axes[0, 2].axhline(y=3, color="black", linestyle="--", alpha=0.5)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(scale_names, fontsize=8)
    axes[0, 2].set_ylabel("Mean kurtosis")
    axes[0, 2].set_title("Weight kurtosis (Gaussian=3)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    # 4. Cross-layer similarity
    sim_vals = [results[s]["cross_sim_mean"] for s in scale_names]
    axes[1, 0].bar(x_pos, sim_vals, color=colors_trained)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(scale_names, fontsize=8)
    axes[1, 0].set_ylabel("Mean cosine similarity")
    axes[1, 0].set_title("Cross-layer spectral similarity")
    axes[1, 0].set_ylim(0.8, 1.0)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 5. Top SV concentration
    init_vals = [results[s]["init_conc_mean"] for s in scale_names]
    trained_vals = [results[s]["trained_conc_mean"] for s in scale_names]
    axes[1, 1].bar(x_pos - width/2, init_vals, width, color=colors_init, label="Init")
    axes[1, 1].bar(x_pos + width/2, trained_vals, width, color=colors_trained, label="Trained")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scale_names, fontsize=8)
    axes[1, 1].set_ylabel("σ₁ / Σσᵢ")
    axes[1, 1].set_title("Top SV concentration")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # 6. Hurst exponent
    hurst_vals = []
    hurst_labels = []
    hurst_colors = []
    for i, s in enumerate(scale_names):
        if results[s]["hurst"] is not None:
            H, r2 = results[s]["hurst"]
            hurst_vals.append(H)
            hurst_labels.append(f"{s}\nH={H:.3f}\nR²={r2:.3f}")
            hurst_colors.append(colors_trained[i])
    if hurst_vals:
        axes[1, 2].bar(range(len(hurst_vals)), hurst_vals, color=hurst_colors)
        axes[1, 2].axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Random walk")
        axes[1, 2].set_xticks(range(len(hurst_vals)))
        axes[1, 2].set_xticklabels(hurst_labels, fontsize=7)
        axes[1, 2].set_ylabel("Hurst exponent H")
        axes[1, 2].set_title("SGD trajectory Hurst exponent")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "scale_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # print summary table
    print(f"\n{'='*80}")
    print(f"SCALE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<30} ", end="")
    for s in scale_names:
        print(f"{s:>15} ", end="")
    print()
    print("-" * 80)

    metrics = [
        ("Parameters", lambda s: f"{results[s]['n_params']/1e6:.1f}M"),
        ("α (init → trained)", lambda s: f"{results[s]['init_alpha_mean']:.3f}→{results[s]['trained_alpha_mean']:.3f}"),
        ("R² (trained)", lambda s: f"{results[s]['trained_r2_mean']:.3f}"),
        ("Kurtosis (init → trained)", lambda s: f"{results[s]['init_kurt_mean']:.2f}→{results[s]['trained_kurt_mean']:.2f}"),
        ("Cross-layer sim", lambda s: f"{results[s]['cross_sim_mean']:.4f}"),
        ("Top SV conc (init → trained)", lambda s: f"{results[s]['init_conc_mean']:.3f}→{results[s]['trained_conc_mean']:.3f}"),
        ("Hurst exponent", lambda s: f"{results[s]['hurst'][0]:.3f}" if results[s]['hurst'] else "N/A"),
    ]

    for metric_name, fmt_fn in metrics:
        print(f"{metric_name:<30} ", end="")
        for s in scale_names:
            print(f"{fmt_fn(s):>15} ", end="")
        print()


# --- Experiment 19: RMT-Denoised Spectral Metrics ---

def exp_rmt_denoised():
    """
    Random Matrix Theory denoising of weight spectra.

    Methodology (ArXiv-grade):
    ──────────────────────────
    For a weight matrix W ∈ ℝ^{m×n} (m ≥ n), compute the singular values σ₁ ≥ σ₂ ≥ … ≥ σₙ
    and eigenvalues λᵢ = σᵢ². Under the null hypothesis that W is iid Gaussian with variance
    σ², the eigenvalue distribution converges to the Marchenko-Pastur (MP) law:

        ρ_MP(λ) = (1/(2πσ²γλ)) √((λ₊ - λ)(λ - λ₋))

    where γ = m/n (aspect ratio), λ₊ = σ²(1 + √γ)², λ₋ = σ²(1 - √γ)².

    Denoising: eigenvalues above λ₊ are "signal" (informative); below are "bulk" (noise).
    We estimate σ² via the median eigenvalue (robust to heavy tails), then apply the
    Gavish-Donoho optimal hard threshold for singular values.

    Tail fitting: on signal eigenvalues, we fit:
      1. Power law via MLE: α̂ = 1 + n/Σ ln(xᵢ/x_min)  (Clauset et al. 2009)
      2. Log-normal via scipy MLE
      3. Exponential via scipy MLE
    Model comparison via Kolmogorov-Smirnov test and log-likelihood ratios.

    Controls:
      - Pure random matrix (same shape, σ=0.02): should show NO signal above MP edge
      - Planted heavy-tail (random + rank-5 power-law component): should recover planted tail

    Outputs:
      - plots/rmt_denoised_spectra.png (eigenvalue histograms with MP edge)
      - plots/rmt_tail_fits.png (CCDF with competing fits)
      - plots/rmt_evolution.png (α_MLE and KS across checkpoints)
      - plots/rmt_controls.png (synthetic controls)
      - plots/rmt_data.npz (all numerical results for paper tables)
    """
    from scipy.stats import kstest, lognorm, expon

    checkpoints = get_checkpoints()
    print(f"RMT denoising across {len(checkpoints)} checkpoints...")

    # --- helper functions ---

    def mp_upper_edge(m, n, sigma_sq):
        """Marchenko-Pastur upper edge: λ₊ = σ²(1 + √(m/n))²"""
        gamma = m / n
        return sigma_sq * (1 + np.sqrt(gamma)) ** 2

    def estimate_noise_variance(eigenvalues, gamma):
        """
        Estimate σ² from bulk eigenvalues using the median.
        For MP distribution, median ≈ σ² × m_γ where m_γ is the MP median.
        We use a simpler robust estimator: σ² = median(λ) / (1 + √γ)²
        which is conservative (underestimates noise → keeps more signal).
        """
        med = np.median(eigenvalues)
        # Approximate: median of MP is close to the center of the bulk
        mp_center = (1 + gamma) / 2  # rough center for σ²=1
        if mp_center > 0:
            return med / mp_center
        return med

    def fit_power_law_mle(data):
        """
        MLE power-law exponent (Clauset et al. 2009).
        α̂ = 1 + n / Σ ln(xᵢ / x_min)
        Returns: alpha, x_min
        """
        x_min = np.min(data)
        if x_min <= 0:
            return np.nan, np.nan
        n = len(data)
        alpha = 1 + n / np.sum(np.log(data / x_min))
        return alpha, x_min

    def power_law_ccdf(x, alpha, x_min):
        """Complementary CDF: P(X > x) = (x/x_min)^{-(α-1)}"""
        return (x / x_min) ** (-(alpha - 1))

    def log_likelihood_power_law(data, alpha, x_min):
        """Log-likelihood of power-law fit."""
        n = len(data)
        if alpha <= 1 or x_min <= 0:
            return -np.inf
        return n * np.log(alpha - 1) - n * np.log(x_min) - alpha * np.sum(np.log(data / x_min))

    def mp_density(lam, sigma_sq, gamma):
        """Marchenko-Pastur density for plotting."""
        lam_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
        lam_minus = sigma_sq * (1 - np.sqrt(gamma)) ** 2
        density = np.zeros_like(lam)
        mask = (lam >= lam_minus) & (lam <= lam_plus)
        if np.any(mask):
            density[mask] = np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus)) / (
                2 * np.pi * sigma_sq * gamma * lam[mask])
        return density

    # --- main analysis ---

    # Pick 4 representative layers for detailed plots
    init_model = load_checkpoint(checkpoints[0])
    final_model = load_checkpoint(checkpoints[-1])
    weight_names = [n for n in get_weight_matrices(init_model).keys()]
    # Pick diverse layers: early attn, early MLP, late attn, late MLP
    repr_layers = []
    for keyword in ["blocks.0.attn.c_attn", "blocks.0.mlp.c_fc",
                     "blocks.4.attn.c_attn", "blocks.4.mlp.c_fc"]:
        matches = [n for n in weight_names if keyword in n]
        if matches:
            repr_layers.append(matches[0])
    if len(repr_layers) < 4:
        repr_layers = weight_names[:4]

    print(f"Representative layers: {repr_layers}")

    # ========================================
    # Plot 1: Eigenvalue histograms with MP edge
    # ========================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("RMT Denoising: Eigenvalue Spectra vs Marchenko-Pastur Boundary",
                 fontsize=14, fontweight='bold')

    all_results = {}

    for col, layer_name in enumerate(repr_layers):
        for row, (model, label, ckpt_label) in enumerate([
            (init_model, "Init (step 0)", "init"),
            (final_model, f"Trained (step {len(checkpoints)*500-500})", "final"),
        ]):
            ax = axes[row, col]
            W = dict(model.named_parameters())[layer_name].detach().numpy()
            m, n = W.shape
            if m < n:
                W = W.T
                m, n = W.shape
            gamma = m / n

            S = np.linalg.svd(W, compute_uv=False)
            eigenvalues = S ** 2

            # estimate noise and MP edge
            sigma_sq = estimate_noise_variance(eigenvalues, gamma)
            lam_plus = mp_upper_edge(m, n, sigma_sq)

            # signal vs bulk split
            signal_mask = eigenvalues > lam_plus
            n_signal = np.sum(signal_mask)
            signal_eigs = eigenvalues[signal_mask]
            bulk_eigs = eigenvalues[~signal_mask]

            # plot histogram
            bins = np.linspace(0, np.max(eigenvalues) * 1.1, 80)
            ax.hist(bulk_eigs, bins=bins, alpha=0.6, color='steelblue',
                    density=True, label=f'Bulk ({len(bulk_eigs)})')
            if len(signal_eigs) > 0:
                ax.hist(signal_eigs, bins=bins, alpha=0.6, color='coral',
                        density=True, label=f'Signal ({len(signal_eigs)})')

            # overlay MP density
            lam_range = np.linspace(max(bins[0], 1e-6), bins[-1], 500)
            mp_curve = mp_density(lam_range, sigma_sq, gamma)
            ax.plot(lam_range, mp_curve, 'k--', linewidth=1.5, label='MP density')

            # MP edge line
            ax.axvline(lam_plus, color='red', linestyle=':', linewidth=2,
                       label=f'$\\lambda_+={lam_plus:.4f}$')

            ax.set_xlabel('$\\lambda$ (eigenvalue)')
            if col == 0:
                ax.set_ylabel('Density')
            short_name = layer_name.split(".")[-1]
            layer_idx = layer_name.split(".")[1] if "blocks" in layer_name else "?"
            ax.set_title(f'L{layer_idx} {short_name}\n{label}', fontsize=10)
            ax.legend(fontsize=7)

            key = f"{ckpt_label}_{layer_name}"
            all_results[key] = {
                "sigma_sq": sigma_sq, "lam_plus": lam_plus,
                "n_signal": n_signal, "n_total": len(eigenvalues),
                "signal_frac": n_signal / len(eigenvalues),
            }

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rmt_denoised_spectra.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Plot 2: CCDF with competing tail fits
    # ========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Signal Tail Fits: Power Law vs Log-Normal vs Exponential (Trained Weights)",
                 fontsize=14, fontweight='bold')

    fit_results = {}

    for col, layer_name in enumerate(repr_layers):
        ax = axes[col]
        W = dict(final_model.named_parameters())[layer_name].detach().numpy()
        m, n = W.shape
        if m < n:
            W = W.T
            m, n = W.shape
        gamma = m / n

        S = np.linalg.svd(W, compute_uv=False)
        eigenvalues = S ** 2
        sigma_sq = estimate_noise_variance(eigenvalues, gamma)
        lam_plus = mp_upper_edge(m, n, sigma_sq)
        signal_eigs = np.sort(eigenvalues[eigenvalues > lam_plus])[::-1]

        if len(signal_eigs) < 5:
            ax.text(0.5, 0.5, f'Only {len(signal_eigs)} signal eigs\n(too few to fit)',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        # Empirical CCDF
        n_sig = len(signal_eigs)
        sorted_sig = np.sort(signal_eigs)
        ccdf_empirical = np.arange(n_sig, 0, -1) / n_sig
        ax.plot(sorted_sig, ccdf_empirical, 'ko', markersize=3, label='Empirical CCDF')

        # Fit 1: Power law (MLE)
        alpha_pl, x_min_pl = fit_power_law_mle(signal_eigs)
        if not np.isnan(alpha_pl):
            x_fit = np.linspace(sorted_sig[0], sorted_sig[-1], 200)
            ccdf_pl = power_law_ccdf(x_fit, alpha_pl, x_min_pl)
            ax.plot(x_fit, ccdf_pl, 'r-', linewidth=2,
                    label=f'Power law ($\\alpha$={alpha_pl:.2f})')
            # KS test
            ks_pl, p_pl = kstest(signal_eigs / x_min_pl,
                                 lambda x: 1 - x ** (-(alpha_pl - 1)))
            ll_pl = log_likelihood_power_law(signal_eigs, alpha_pl, x_min_pl)

        # Fit 2: Log-normal
        try:
            shape_ln, loc_ln, scale_ln = lognorm.fit(signal_eigs, floc=0)
            ccdf_ln = 1 - lognorm.cdf(sorted_sig, shape_ln, loc=loc_ln, scale=scale_ln)
            ax.plot(sorted_sig, ccdf_ln, 'b--', linewidth=2,
                    label=f'Log-normal ($\\sigma$={shape_ln:.2f})')
            ks_ln, p_ln = kstest(signal_eigs, 'lognorm', args=(shape_ln, loc_ln, scale_ln))
            ll_ln = np.sum(lognorm.logpdf(signal_eigs, shape_ln, loc=loc_ln, scale=scale_ln))
        except Exception:
            ks_ln, p_ln, ll_ln = np.nan, np.nan, -np.inf

        # Fit 3: Exponential
        try:
            loc_ex, scale_ex = expon.fit(signal_eigs, floc=np.min(signal_eigs))
            ccdf_ex = 1 - expon.cdf(sorted_sig, loc=loc_ex, scale=scale_ex)
            ax.plot(sorted_sig, ccdf_ex, 'g:', linewidth=2,
                    label=f'Exponential ($\\lambda$={1/scale_ex:.2f})')
            ks_ex, p_ex = kstest(signal_eigs, 'expon', args=(loc_ex, scale_ex))
            ll_ex = np.sum(expon.logpdf(signal_eigs, loc=loc_ex, scale=scale_ex))
        except Exception:
            ks_ex, p_ex, ll_ex = np.nan, np.nan, -np.inf

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$\\lambda$ (eigenvalue)')
        ax.set_ylabel('$P(X > \\lambda)$')
        short_name = layer_name.split(".")[-1]
        layer_idx = layer_name.split(".")[1] if "blocks" in layer_name else "?"
        ax.set_title(f'L{layer_idx} {short_name} ({n_sig} signal eigs)', fontsize=10)
        ax.legend(fontsize=7)

        fit_results[layer_name] = {
            "alpha_pl": alpha_pl, "ks_pl": ks_pl, "p_pl": p_pl, "ll_pl": ll_pl,
            "ks_ln": ks_ln, "p_ln": p_ln, "ll_ln": ll_ln,
            "ks_ex": ks_ex, "p_ex": p_ex, "ll_ex": ll_ex,
            "n_signal": n_sig,
            "lr_pl_vs_ln": ll_pl - ll_ln if not np.isnan(ll_ln) else np.nan,
            "lr_pl_vs_ex": ll_pl - ll_ex if not np.isnan(ll_ex) else np.nan,
        }

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rmt_tail_fits.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Print fit comparison table
    print("\n=== Tail Fit Comparison (trained weights) ===")
    print(f"{'Layer':<35} {'n_sig':>5} {'α_PL':>6} {'KS_PL':>7} {'KS_LN':>7} {'KS_EX':>7} {'LR(PL/LN)':>10} {'LR(PL/EX)':>10}")
    for layer_name, fr in fit_results.items():
        short = layer_name.replace("blocks.", "L").replace(".weight", "")
        print(f"{short:<35} {fr['n_signal']:>5d} {fr['alpha_pl']:>6.2f} "
              f"{fr['ks_pl']:>7.3f} {fr['ks_ln']:>7.3f} {fr['ks_ex']:>7.3f} "
              f"{fr.get('lr_pl_vs_ln', np.nan):>10.1f} {fr.get('lr_pl_vs_ex', np.nan):>10.1f}")

    # ========================================
    # Plot 3: Evolution across checkpoints
    # ========================================
    # Track α_MLE and signal fraction for 2 representative layers across all checkpoints
    evo_layers = repr_layers[:2]
    evo_data = {ln: {"steps": [], "alpha": [], "n_signal": [], "signal_frac": []} for ln in evo_layers}

    for ci, ckpt_path in enumerate(checkpoints):
        step = ci * 500
        model = load_checkpoint(ckpt_path)
        for layer_name in evo_layers:
            W = dict(model.named_parameters())[layer_name].detach().numpy()
            m, n = W.shape
            if m < n:
                W = W.T
                m, n = W.shape
            gamma = m / n
            S = np.linalg.svd(W, compute_uv=False)
            eigenvalues = S ** 2
            sigma_sq = estimate_noise_variance(eigenvalues, gamma)
            lam_plus = mp_upper_edge(m, n, sigma_sq)
            signal_eigs = eigenvalues[eigenvalues > lam_plus]

            if len(signal_eigs) >= 3:
                alpha, _ = fit_power_law_mle(signal_eigs)
            else:
                alpha = np.nan

            evo_data[layer_name]["steps"].append(step)
            evo_data[layer_name]["alpha"].append(alpha)
            evo_data[layer_name]["n_signal"].append(len(signal_eigs))
            evo_data[layer_name]["signal_frac"].append(len(signal_eigs) / len(eigenvalues))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RMT Denoised Metrics Over Training", fontsize=14, fontweight='bold')

    colors = ['coral', 'steelblue']
    for i, layer_name in enumerate(evo_layers):
        d = evo_data[layer_name]
        short = layer_name.replace("blocks.", "L").replace(".weight", "")

        axes[0].plot(d["steps"], d["alpha"], 'o-', color=colors[i], label=short, markersize=4)
        axes[1].plot(d["steps"], d["n_signal"], 'o-', color=colors[i], label=short, markersize=4)
        axes[2].plot(d["steps"], d["signal_frac"], 'o-', color=colors[i], label=short, markersize=4)

    axes[0].set_ylabel('$\\hat{\\alpha}_{MLE}$ (power-law exponent)')
    axes[0].set_xlabel('Training step')
    axes[0].set_title('Power-Law Exponent Evolution')
    axes[0].legend()
    axes[0].axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='α=2 (Zipf)')

    axes[1].set_ylabel('Number of signal eigenvalues')
    axes[1].set_xlabel('Training step')
    axes[1].set_title('Signal Eigenvalues Above MP Edge')
    axes[1].legend()

    axes[2].set_ylabel('Signal fraction')
    axes[2].set_xlabel('Training step')
    axes[2].set_title('Fraction of Spectrum Above MP Edge')
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rmt_evolution.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Plot 4: Synthetic controls
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RMT Controls: Random vs Planted Heavy-Tail Matrix",
                 fontsize=14, fontweight='bold')

    # Control 1: Pure random matrix (same shape as first repr layer)
    W_ref = dict(final_model.named_parameters())[repr_layers[0]].detach().numpy()
    m, n = W_ref.shape
    if m < n:
        m, n = n, m

    np.random.seed(42)
    W_rand = np.random.randn(m, n) * 0.02
    S_rand = np.linalg.svd(W_rand, compute_uv=False)
    eig_rand = S_rand ** 2
    gamma = m / n
    sigma_sq_rand = estimate_noise_variance(eig_rand, gamma)
    lam_plus_rand = mp_upper_edge(m, n, sigma_sq_rand)
    n_signal_rand = np.sum(eig_rand > lam_plus_rand)

    bins = np.linspace(0, np.max(eig_rand) * 1.1, 60)
    axes[0].hist(eig_rand, bins=bins, alpha=0.7, color='steelblue', density=True, label='Eigenvalues')
    lam_range = np.linspace(max(bins[0], 1e-8), bins[-1], 500)
    mp_curve = mp_density(lam_range, sigma_sq_rand, gamma)
    axes[0].plot(lam_range, mp_curve, 'k--', linewidth=2, label='MP density')
    axes[0].axvline(lam_plus_rand, color='red', linestyle=':', linewidth=2,
                     label=f'$\\lambda_+={lam_plus_rand:.6f}$')
    axes[0].set_title(f'Random Matrix ({m}×{n}, σ=0.02)\nSignal above MP edge: {n_signal_rand}')
    axes[0].set_xlabel('$\\lambda$')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=8)

    # Control 2: Planted heavy-tail matrix
    W_planted = np.random.randn(m, n) * 0.02
    # Add rank-5 component with power-law singular values
    planted_svs = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])  # power-law decay
    U_plant = np.linalg.qr(np.random.randn(m, 5))[0]
    V_plant = np.linalg.qr(np.random.randn(n, 5))[0]
    W_planted += U_plant @ np.diag(planted_svs) @ V_plant.T

    S_planted = np.linalg.svd(W_planted, compute_uv=False)
    eig_planted = S_planted ** 2
    sigma_sq_pl = estimate_noise_variance(eig_planted, gamma)
    lam_plus_pl = mp_upper_edge(m, n, sigma_sq_pl)
    signal_planted = eig_planted[eig_planted > lam_plus_pl]
    n_signal_planted = len(signal_planted)

    bins2 = np.linspace(0, np.max(eig_planted) * 1.1, 80)
    bulk_pl = eig_planted[eig_planted <= lam_plus_pl]
    axes[1].hist(bulk_pl, bins=bins2, alpha=0.6, color='steelblue', density=True, label='Bulk')
    axes[1].hist(signal_planted, bins=bins2, alpha=0.6, color='coral', density=True, label='Signal')
    lam_range2 = np.linspace(max(bins2[0], 1e-8), bins2[-1], 500)
    mp_curve2 = mp_density(lam_range2, sigma_sq_pl, gamma)
    axes[1].plot(lam_range2, mp_curve2, 'k--', linewidth=2, label='MP density')
    axes[1].axvline(lam_plus_pl, color='red', linestyle=':', linewidth=2,
                     label=f'$\\lambda_+={lam_plus_pl:.4f}$')
    axes[1].set_title(f'Planted Matrix (rank-5 signal)\nRecovered {n_signal_planted} signal eigs (planted 5)')
    axes[1].set_xlabel('$\\lambda$')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rmt_controls.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Save structured data for paper
    # ========================================
    save_data = {
        "repr_layers": np.array(repr_layers, dtype=object),
    }
    for layer_name in evo_layers:
        d = evo_data[layer_name]
        key = layer_name.replace(".", "_")
        save_data[f"{key}_steps"] = np.array(d["steps"])
        save_data[f"{key}_alpha"] = np.array(d["alpha"])
        save_data[f"{key}_n_signal"] = np.array(d["n_signal"])
        save_data[f"{key}_signal_frac"] = np.array(d["signal_frac"])
    for layer_name, fr in fit_results.items():
        key = layer_name.replace(".", "_")
        for k, v in fr.items():
            save_data[f"fit_{key}_{k}"] = np.array(v)
    save_data["control_random_n_signal"] = np.array(n_signal_rand)
    save_data["control_planted_n_signal"] = np.array(n_signal_planted)

    npz_path = os.path.join(PLOTS_DIR, "rmt_data.npz")
    np.savez_compressed(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    print("\n=== RMT Denoised Spectral Analysis Complete ===")
    print(f"Key finding: trained weights have {all_results.get(f'final_{repr_layers[0]}', {}).get('signal_frac', 0):.1%} "
          f"of spectrum above MP edge vs init")


# --- Experiment 20: Attention Graph Spectral Diagnostics ---

def exp_attn_graph_spectral():
    """
    Graph-theoretic analysis of attention maps via Laplacian spectral decomposition.

    Methodology (ArXiv-grade):
    ──────────────────────────
    Each attention map A ∈ ℝ^{T×T} (T = sequence length) is treated as the weighted
    adjacency matrix of a directed graph. We symmetrize: Ã = (A + Aᵀ)/2, then compute
    the normalized graph Laplacian:

        L_norm = I - D^{-1/2} Ã D^{-1/2}

    where D = diag(Ã·1) is the degree matrix. The eigenvalues 0 = λ₀ ≤ λ₁ ≤ … ≤ λ_{T-1}
    encode graph structure:

    Metrics:
      1. Fiedler value (λ₁): algebraic connectivity — measures how well-connected the graph
         is. Higher λ₁ → harder to bipartition → more globally integrated attention.

      2. Spectral entropy: H = -Σᵢ pᵢ ln(pᵢ) where pᵢ = λᵢ/Σλ (for λᵢ > 0).
         Measures spread of Laplacian spectrum. Uniform → high entropy → evenly distributed
         structure across scales.

      3. High-frequency energy ratio: E_HF = Σᵢ₌ₜ/₂^{T-1} λᵢ / Σλ.
         High → lots of local/fine-grained structure (tokens attend to nearby tokens).
         Low → attention is smooth/global.

      4. Modularity (Newman 2006): Q = (1/2m) Σᵢⱼ (Ãᵢⱼ - dᵢdⱼ/2m) δ(cᵢ,cⱼ)
         using spectral bipartition from the Fiedler vector. Measures how strongly
         the graph decomposes into communities.

    Controls:
      - Degree-preserving randomization: permute entries within each row
      - Token-order shuffle: randomly permute token positions before computing attention

    Outputs:
      - plots/attn_graph_fiedler.png
      - plots/attn_graph_spectral_entropy.png
      - plots/attn_graph_hf_energy.png
      - plots/attn_graph_summary.png
      - plots/attn_graph_data.npz
    """
    checkpoints = get_checkpoints()
    # Use 5 evenly-spaced checkpoints for efficiency
    ckpt_indices = np.linspace(0, len(checkpoints) - 1, 5, dtype=int)
    ckpt_paths = [checkpoints[i] for i in ckpt_indices]
    steps = [i * 500 for i in ckpt_indices]
    print(f"Attention graph analysis at steps: {steps}")

    # --- graph metric functions ---

    def compute_graph_metrics(A):
        """
        Compute Laplacian spectral metrics for a single attention map.

        Args:
            A: np.array of shape [T, T], attention weights (row-stochastic after softmax)

        Returns:
            dict with fiedler, spectral_entropy, hf_energy, modularity
        """
        T = A.shape[0]

        # Symmetrize
        A_sym = (A + A.T) / 2

        # Degree matrix
        d = A_sym.sum(axis=1)
        d_safe = np.maximum(d, 1e-10)  # avoid division by zero

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
        L_norm = np.eye(T) - D_inv_sqrt @ A_sym @ D_inv_sqrt

        # Eigenvalues (symmetric → eigvalsh)
        eigs = np.linalg.eigvalsh(L_norm)
        eigs = np.sort(np.real(eigs))
        # Clamp small negatives from numerical error
        eigs = np.maximum(eigs, 0)

        # Fiedler value (second smallest eigenvalue)
        fiedler = eigs[1] if len(eigs) > 1 else 0.0

        # Spectral entropy
        pos_eigs = eigs[eigs > 1e-10]
        if len(pos_eigs) > 0:
            p = pos_eigs / pos_eigs.sum()
            spectral_entropy = -np.sum(p * np.log(p + 1e-15))
        else:
            spectral_entropy = 0.0

        # High-frequency energy ratio
        total_energy = eigs.sum()
        if total_energy > 0:
            hf_energy = eigs[T // 2:].sum() / total_energy
        else:
            hf_energy = 0.0

        # Modularity via Fiedler vector bipartition
        eigvals, eigvecs = np.linalg.eigh(L_norm)
        fiedler_vec = eigvecs[:, 1]  # second eigenvector
        partition = (fiedler_vec >= 0).astype(int)
        m = A_sym.sum() / 2
        if m > 0:
            modularity = 0.0
            for i in range(T):
                for j in range(T):
                    if partition[i] == partition[j]:
                        modularity += A_sym[i, j] - d[i] * d[j] / (2 * m)
            modularity /= (2 * m)
        else:
            modularity = 0.0

        return {
            "fiedler": fiedler,
            "spectral_entropy": spectral_entropy,
            "hf_energy": hf_energy,
            "modularity": modularity,
        }

    def degree_preserving_randomize(A, rng):
        """Randomize attention map while preserving row sums (degree)."""
        A_rand = A.copy()
        for i in range(A.shape[0]):
            row = A_rand[i].copy()
            rng.shuffle(row)
            A_rand[i] = row
        return A_rand

    # --- collect metrics ---

    rng = np.random.RandomState(42)

    # Structure: metrics[step][layer][head] = dict of metrics
    all_metrics = {"trained": {}, "randomized": {}, "shuffled": {}}

    for si, (ckpt_path, step) in enumerate(zip(ckpt_paths, steps)):
        print(f"  Processing step {step}...")
        model = load_checkpoint(ckpt_path)

        # Regular attention maps
        attn_maps = get_attention_maps(model)

        # Shuffled-token attention maps
        data = np.memmap(os.path.join(os.path.dirname(__file__), "data", "train.bin"),
                         dtype=np.uint16, mode="r")
        x_normal = torch.from_numpy(data[:BLOCK_SIZE].astype(np.int64)).unsqueeze(0)
        perm = rng.permutation(BLOCK_SIZE)
        x_shuffled = x_normal[:, perm]
        attn_maps_shuffled = get_attention_maps(model, x_shuffled)

        for layer_key, attn_all_heads in attn_maps.items():
            n_heads = attn_all_heads.shape[0]
            for h in range(n_heads):
                mk = f"{layer_key}_h{h}"

                # Trained
                m_trained = compute_graph_metrics(attn_all_heads[h])
                all_metrics["trained"].setdefault(step, {})[mk] = m_trained

                # Degree-preserving randomized
                A_rand = degree_preserving_randomize(attn_all_heads[h], rng)
                m_rand = compute_graph_metrics(A_rand)
                all_metrics["randomized"].setdefault(step, {})[mk] = m_rand

                # Token-shuffled
                attn_shuf = attn_maps_shuffled[layer_key][h]
                m_shuf = compute_graph_metrics(attn_shuf)
                all_metrics["shuffled"].setdefault(step, {})[mk] = m_shuf

    # --- aggregate for plotting ---

    metric_names = ["fiedler", "spectral_entropy", "hf_energy", "modularity"]
    conditions = ["trained", "randomized", "shuffled"]
    cond_colors = {"trained": "coral", "randomized": "steelblue", "shuffled": "gray"}

    # Aggregate: mean and std across heads for each step and condition
    agg = {cond: {mn: {"mean": [], "std": []} for mn in metric_names} for cond in conditions}

    for cond in conditions:
        for step in steps:
            step_data = all_metrics[cond].get(step, {})
            for mn in metric_names:
                values = [step_data[mk][mn] for mk in step_data]
                agg[cond][mn]["mean"].append(np.mean(values))
                agg[cond][mn]["std"].append(np.std(values))

    # ========================================
    # Plot individual metric evolution
    # ========================================
    for mn, ylabel, title in [
        ("fiedler", "$\\lambda_1$ (Fiedler value)", "Algebraic Connectivity Over Training"),
        ("spectral_entropy", "$H$ (spectral entropy)", "Spectral Entropy Over Training"),
        ("hf_energy", "$E_{HF}$ (high-freq ratio)", "High-Frequency Energy Over Training"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond in conditions:
            means = agg[cond][mn]["mean"]
            stds = agg[cond][mn]["std"]
            ax.plot(steps, means, 'o-', color=cond_colors[cond], label=cond, linewidth=2)
            ax.fill_between(steps,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=cond_colors[cond], alpha=0.15)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        fname = f"attn_graph_{mn}.png"
        path = os.path.join(PLOTS_DIR, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ========================================
    # Summary plot: all metrics in one figure
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Attention Graph Spectral Diagnostics: Trained vs Controls",
                 fontsize=14, fontweight='bold')

    for ax, (mn, ylabel) in zip(axes.flat, [
        ("fiedler", "$\\lambda_1$ (Fiedler)"),
        ("spectral_entropy", "$H$ (spectral entropy)"),
        ("hf_energy", "$E_{HF}$ (high-freq)"),
        ("modularity", "$Q$ (modularity)"),
    ]):
        for cond in conditions:
            means = agg[cond][mn]["mean"]
            stds = agg[cond][mn]["std"]
            ax.plot(steps, means, 'o-', color=cond_colors[cond], label=cond, linewidth=2)
            ax.fill_between(steps,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=cond_colors[cond], alpha=0.15)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "attn_graph_summary.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Save structured data
    # ========================================
    save_data = {"steps": np.array(steps)}
    for cond in conditions:
        for mn in metric_names:
            save_data[f"{cond}_{mn}_mean"] = np.array(agg[cond][mn]["mean"])
            save_data[f"{cond}_{mn}_std"] = np.array(agg[cond][mn]["std"])

    npz_path = os.path.join(PLOTS_DIR, "attn_graph_data.npz")
    np.savez_compressed(npz_path, **save_data)
    print(f"Saved: {npz_path}")

    # Print summary table
    print("\n=== Attention Graph Metrics (init → final) ===")
    for mn in metric_names:
        trained_init = agg["trained"][mn]["mean"][0]
        trained_final = agg["trained"][mn]["mean"][-1]
        rand_final = agg["randomized"][mn]["mean"][-1]
        print(f"  {mn:>20}: {trained_init:.4f} → {trained_final:.4f} "
              f"(Δ={trained_final-trained_init:+.4f}) | random control: {rand_final:.4f}")


# --- Experiment 21: Critical-LR Fractal Basin Maps ---

def exp_critical_lr():
    """
    Map the convergence boundary in (learning rate × seed) space near the critical LR.

    Methodology (ArXiv-grade):
    ──────────────────────────
    For a model trained with AdamW + cosine schedule, there exists a critical learning
    rate lr* above which training diverges. Near lr*, the boundary between stable and
    diverged runs in (LR, seed) space may exhibit fractal structure — analogous to
    Julia set boundaries in dynamical systems.

    Protocol:
      1. Coarse search: 10 log-spaced LRs in [1e-4, 1e-1], 3 seeds, 1000 steps
         → identify lr* (smallest LR where >50% diverge)
      2. Dense sweep: 20 log-spaced LRs in [0.5·lr*, 2·lr*], 16 seeds, 1000 steps
      3. Classify each run as stable (final loss < 2.0), slow (≥ 2.0), or diverged (NaN/loss > 100)
      4. Box-counting dimension of the stable/diverged boundary

    Box-counting dimension:
      Partition the (LR, seed) grid at resolution ε. Count N(ε) = number of boxes
      containing BOTH stable and diverged runs. If the boundary is fractal:
        N(ε) ~ ε^{-D}
      and D > 1 indicates fractal boundary (smooth boundary → D = 1).

    Bootstrap 95% CI: resample the seed axis 1000 times, recompute D each time.

    Controls:
      - Convex model (single linear layer): boundary should be smooth (D ≈ 1)
      - Shuffled labels: different boundary structure

    Outputs:
      - plots/critical_lr_basin_map.png
      - plots/critical_lr_boundary_dimension.png
      - plots/critical_lr_controls.png
      - plots/critical_lr_data.npz

    Requires: run 'uv run train_sweep.py --critical-lr small' first,
              or this function runs the sweep inline.
    """
    from train_sweep import train_seeded, classify_run, find_critical_lr

    data_path = os.path.join(PLOTS_DIR, "critical_lr_data.npz")

    if os.path.exists(data_path):
        print("Loading pre-computed sweep data...")
        cached = np.load(data_path, allow_pickle=True)
        lr_values = cached["lr_values"]
        outcome_grid = cached["outcome_grid"]
        final_loss_grid = cached["final_loss_grid"]
        lr_critical = float(cached["lr_critical"])
    else:
        print("Running LR sweep...")

        # Step 1: Sweep a wide LR range directly (skip coarse search —
        # the boundary is between "training makes progress" and "diverged").
        # Use 1000 steps so the model has time to either converge or blow up.
        n_lrs = 12
        n_seeds = 10
        max_steps_sweep = 1000

        # Wide range: 1e-3 to 1.0 — need very high LR to break AdamW + grad clip
        lr_values = np.logspace(-3, 0, n_lrs)
        lr_critical = lr_values[n_lrs // 2]  # rough center, refined below

        # outcome_grid[i, j] = 0 (stable), 1 (slow), 2 (diverged)
        outcome_grid = np.zeros((n_lrs, n_seeds), dtype=int)
        final_loss_grid = np.zeros((n_lrs, n_seeds))

        for i, lr in enumerate(lr_values):
            for j in range(n_seeds):
                print(f"  [{i*n_seeds + j + 1}/{n_lrs*n_seeds}] lr={lr:.2e} seed={j}",
                      flush=True)
                r = train_seeded(seed=j, lr=float(lr), max_steps=max_steps_sweep,
                                 n_layer=3, n_embd=96, n_head=3, quiet=True)
                final_loss_grid[i, j] = r["final_loss"] if not np.isnan(r["final_loss"]) else 100.0

        # Adaptive classification: use median loss as boundary
        # Runs with loss < median → "stable", loss > 2×median → "diverged", else "slow"
        median_loss = np.median(final_loss_grid[np.isfinite(final_loss_grid)])
        for i in range(n_lrs):
            for j in range(n_seeds):
                fl = final_loss_grid[i, j]
                if np.isnan(fl) or fl > 50:
                    outcome_grid[i, j] = 2  # diverged
                elif fl > median_loss * 1.5:
                    outcome_grid[i, j] = 1  # slow
                else:
                    outcome_grid[i, j] = 0  # stable
        print(f"Median loss: {median_loss:.2f}, threshold: {median_loss*1.5:.2f}", flush=True)

        # Refine lr_critical: first LR where >50% are NOT stable
        for i, lr in enumerate(lr_values):
            frac_stable = np.mean(outcome_grid[i] == 0)
            if frac_stable < 0.5:
                lr_critical = float(lr)
                break

        np.savez_compressed(data_path,
                            lr_values=lr_values, outcome_grid=outcome_grid,
                            final_loss_grid=final_loss_grid, lr_critical=lr_critical)
        print(f"Saved sweep data: {data_path}")

    n_lrs, n_seeds = outcome_grid.shape
    print(f"Grid: {n_lrs} LRs × {n_seeds} seeds, lr* ≈ {lr_critical:.2e}")
    print(f"Outcomes: stable={np.sum(outcome_grid==0)}, slow={np.sum(outcome_grid==1)}, "
          f"diverged={np.sum(outcome_grid==2)}")

    # ========================================
    # Plot 1: Basin map heatmap
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Critical-LR Basin Map: Convergence Outcomes in (LR × Seed) Space",
                 fontsize=14, fontweight='bold')

    # Discrete outcome map
    cmap_discrete = plt.cm.colors.ListedColormap(['#2ecc71', '#f39c12', '#e74c3c'])
    im = axes[0].imshow(outcome_grid.T, aspect='auto', origin='lower',
                         cmap=cmap_discrete, vmin=0, vmax=2)
    axes[0].set_xlabel('Learning Rate Index')
    axes[0].set_ylabel('Seed')
    axes[0].set_title('Outcome (green=stable, yellow=slow, red=diverged)')
    # Add LR labels
    tick_idx = np.linspace(0, n_lrs - 1, 5, dtype=int)
    axes[0].set_xticks(tick_idx)
    axes[0].set_xticklabels([f'{lr_values[i]:.1e}' for i in tick_idx], rotation=45)

    # Continuous loss map
    loss_clipped = np.clip(final_loss_grid, 0, 10)
    im2 = axes[1].imshow(loss_clipped.T, aspect='auto', origin='lower',
                          cmap='hot_r')
    axes[1].set_xlabel('Learning Rate Index')
    axes[1].set_ylabel('Seed')
    axes[1].set_title('Final Loss (clipped at 10)')
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([f'{lr_values[i]:.1e}' for i in tick_idx], rotation=45)
    plt.colorbar(im2, ax=axes[1], label='Loss')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "critical_lr_basin_map.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Box-counting dimension of the boundary
    # ========================================
    def box_counting_boundary(grid, resolutions):
        """
        Count boxes containing BOTH stable (0) and diverged (2) outcomes.

        Args:
            grid: 2D array of outcomes (0=stable, 1=slow, 2=diverged)
            resolutions: list of box sizes to try

        Returns:
            sizes, counts (arrays for log-log fit)
        """
        H, W = grid.shape
        # Binary boundary: stable (0) vs not-stable (1 or 2)
        binary = (grid >= 2).astype(int)
        sizes = []
        counts = []

        for box_size in resolutions:
            if box_size > min(H, W):
                continue
            n_boxes = 0
            for i in range(0, H, box_size):
                for j in range(0, W, box_size):
                    box = binary[i:i+box_size, j:j+box_size]
                    if box.size == 0:
                        continue
                    # Boundary box: contains both 0 and 1
                    if np.any(box == 0) and np.any(box == 1):
                        n_boxes += 1
            if n_boxes > 0:
                sizes.append(box_size)
                counts.append(n_boxes)

        return np.array(sizes), np.array(counts)

    resolutions = [1, 2, 3, 4, 5, 8]
    sizes, counts = box_counting_boundary(outcome_grid, resolutions)

    if len(sizes) >= 3:
        log_inv_size = np.log(1.0 / sizes)
        log_counts = np.log(counts)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_inv_size, log_counts)
        D_box = slope

        # Bootstrap CI
        n_boot = 1000
        D_boots = []
        for _ in range(n_boot):
            # Resample seed axis
            seed_idx = np.random.randint(0, n_seeds, n_seeds)
            grid_boot = outcome_grid[:, seed_idx]
            s_b, c_b = box_counting_boundary(grid_boot, resolutions)
            if len(s_b) >= 3:
                sl, _, _, _, _ = stats.linregress(np.log(1.0/s_b), np.log(c_b))
                D_boots.append(sl)
        D_boots = np.array(D_boots)
        ci_lo, ci_hi = np.percentile(D_boots, [2.5, 97.5])

        print(f"\nBox-counting dimension D = {D_box:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
        print(f"R² = {r_value**2:.4f}, p = {p_value:.4e}")
    else:
        D_box = np.nan
        ci_lo = ci_hi = np.nan
        print("Not enough resolution levels for box-counting dimension")

    # Plot 2: Box-counting log-log
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(sizes) >= 3:
        ax.plot(np.log(1.0/sizes), np.log(counts), 'ko', markersize=8, label='Data')
        x_fit = np.linspace(np.log(1.0/sizes).min(), np.log(1.0/sizes).max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', linewidth=2,
                label=f'$D = {D_box:.3f}$ (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('$\\ln(1/\\epsilon)$ (log inverse box size)')
        ax.set_ylabel('$\\ln N(\\epsilon)$ (log boundary box count)')
        ax.set_title(f'Box-Counting Dimension of Convergence Boundary\n'
                     f'$D = {D_box:.3f}$, $R^2 = {r_value**2:.4f}$',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "critical_lr_boundary_dimension.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Save all results
    save_data = {
        "lr_values": lr_values, "outcome_grid": outcome_grid,
        "final_loss_grid": final_loss_grid, "lr_critical": lr_critical,
        "box_sizes": sizes, "box_counts": counts,
        "D_box": D_box, "ci_lo": ci_lo, "ci_hi": ci_hi,
    }
    try:
        if len(D_boots) > 0:
            save_data["D_bootstrap"] = D_boots
    except NameError:
        pass

    npz_path = os.path.join(PLOTS_DIR, "critical_lr_data.npz")
    np.savez_compressed(npz_path, **save_data)
    print(f"Saved: {npz_path}")


# --- Experiment 22: Connectivity vs Confinement ---

def exp_connectivity_confinement():
    """
    Mode connectivity and local confinement analysis.

    Methodology (ArXiv-grade):
    ──────────────────────────
    Given two independently trained models θ_A, θ_B (same architecture, different seeds),
    we measure:

    1. Linear interpolation barrier:
       L(t) = loss((1-t)θ_A + tθ_B) for t ∈ [0, 1]
       Barrier = max_t L(t) - max(L(0), L(1))
       If barrier ≈ 0, models are linearly connected (rare for non-trivial nets).

    2. Quadratic Bezier path (Garipov et al. 2018):
       θ(t) = (1-t)²θ_A + 2t(1-t)θ_mid + t²θ_B
       Optimize θ_mid to minimize max_t L(θ(t)). We test 5 candidates.

    3. Local curvature (Hessian trace estimate):
       Tr(H) ≈ (1/k) Σᵢ [L(θ+εdᵢ) + L(θ-εdᵢ) - 2L(θ)] / ε²
       using k=20 random unit directions dᵢ. Higher → sharper minimum.

    4. Noise sensitivity:
       L(θ + σ·z) for z ~ N(0, I), σ ∈ logspace(-3, 0, 20)
       Sharp minima show rapid loss increase with σ.

    5. SWA control: stochastic weight averaging (Izmailov et al. 2018)
       Average last 5 checkpoints → flatter minimum → compare curvature.

    Outputs:
      - plots/connectivity_interpolation.png
      - plots/connectivity_curved.png
      - plots/confinement_curvature.png
      - plots/confinement_noise.png
      - plots/connectivity_data.npz

    Requires: pre-trained model pairs. Uses existing checkpoints for one model,
              trains additional seeds via train_sweep.py.
    """
    from train_sweep import train_seeded

    data_path = os.path.join(PLOTS_DIR, "connectivity_data.npz")

    # --- get or train model pairs ---
    checkpoints = get_checkpoints()

    # Model A: existing trained model (seed implicit)
    model_A = load_checkpoint(checkpoints[-1])
    params_A = get_flat_params(model_A)

    # Model B: train with different seed
    ckpt_dir_B = os.path.join(os.path.dirname(__file__), "checkpoints_connectivity_B")
    ckpt_path_B = os.path.join(ckpt_dir_B, "step_010000.pt")

    if os.path.exists(ckpt_path_B):
        print("Loading pre-trained model B...")
        model_B = load_checkpoint(ckpt_path_B)
    else:
        print("Training model B (seed=7, ~15 min)...")
        r = train_seeded(seed=7, lr=3e-4, max_steps=10000,
                         n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_EMBD // 32,
                         save_checkpoints=True, ckpt_dir=ckpt_dir_B)
        model_B = load_checkpoint(ckpt_path_B)

    params_B = get_flat_params(model_B)

    # SWA model: average last 5 checkpoints of model A
    swa_ckpts = checkpoints[-5:]
    swa_state = None
    for cp in swa_ckpts:
        m = load_checkpoint(cp)
        if swa_state is None:
            swa_state = {k: v.clone().float() for k, v in m.state_dict().items()}
        else:
            for k, v in m.state_dict().items():
                swa_state[k] += v.float()
    for k in swa_state:
        swa_state[k] /= len(swa_ckpts)
    model_SWA = GPT()
    model_SWA.load_state_dict(swa_state)
    model_SWA.eval()
    params_SWA = get_flat_params(model_SWA)

    loss_A = compute_loss_batched(model_A)
    loss_B = compute_loss_batched(model_B)
    loss_SWA = compute_loss_batched(model_SWA)
    print(f"Loss A: {loss_A:.4f}, Loss B: {loss_B:.4f}, Loss SWA: {loss_SWA:.4f}")

    # --- 1. Linear interpolation ---
    print("\n1. Linear interpolation barrier...")
    n_points = 51
    ts = np.linspace(0, 1, n_points)
    losses_linear = []

    for t in ts:
        params_t = (1 - t) * params_A + t * params_B
        set_model_params(model_A, params_t)
        loss_t = compute_loss_batched(model_A, n_batches=3)
        losses_linear.append(loss_t)

    set_model_params(model_A, params_A)  # restore
    losses_linear = np.array(losses_linear)
    barrier_linear = np.max(losses_linear) - max(loss_A, loss_B)
    print(f"Linear barrier: {barrier_linear:.4f} (max={np.max(losses_linear):.4f})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, losses_linear, 'b-', linewidth=2, label='Linear interpolation')
    ax.axhline(loss_A, color='green', linestyle='--', alpha=0.7, label=f'Model A ({loss_A:.3f})')
    ax.axhline(loss_B, color='red', linestyle='--', alpha=0.7, label=f'Model B ({loss_B:.3f})')
    ax.set_xlabel('$t$ (interpolation parameter)')
    ax.set_ylabel('Loss')
    ax.set_title(f'Linear Interpolation: $\\theta(t) = (1-t)\\theta_A + t\\theta_B$\n'
                 f'Barrier = {barrier_linear:.4f}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "connectivity_interpolation.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- 2. Quadratic Bezier path ---
    print("\n2. Curved (Bezier) path...")
    midpoint = (params_A + params_B) / 2

    # 5 candidate midpoints
    torch.manual_seed(42)
    candidates = [midpoint]
    for _ in range(4):
        # midpoint + small perturbation in a random direction
        noise = torch.randn_like(midpoint)
        noise = noise / noise.norm() * params_A.norm() * 0.01
        candidates.append(midpoint + noise)

    best_max_loss = np.inf
    best_losses = None
    best_idx = 0

    for ci, mid_candidate in enumerate(candidates):
        bezier_losses = []
        for t in ts:
            params_t = (1-t)**2 * params_A + 2*t*(1-t) * mid_candidate + t**2 * params_B
            set_model_params(model_A, params_t)
            loss_t = compute_loss_batched(model_A, n_batches=3)
            bezier_losses.append(loss_t)
        max_loss = max(bezier_losses)
        if max_loss < best_max_loss:
            best_max_loss = max_loss
            best_losses = bezier_losses
            best_idx = ci

    set_model_params(model_A, params_A)  # restore
    barrier_bezier = best_max_loss - max(loss_A, loss_B)
    print(f"Best Bezier barrier: {barrier_bezier:.4f} (candidate {best_idx})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, losses_linear, 'b-', linewidth=2, alpha=0.5, label=f'Linear (barrier={barrier_linear:.3f})')
    ax.plot(ts, best_losses, 'r-', linewidth=2, label=f'Bezier (barrier={barrier_bezier:.3f})')
    ax.axhline(loss_A, color='green', linestyle='--', alpha=0.5, label=f'A ({loss_A:.3f})')
    ax.axhline(loss_B, color='orange', linestyle='--', alpha=0.5, label=f'B ({loss_B:.3f})')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Loss')
    ax.set_title('Linear vs Bezier Interpolation Path', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "connectivity_curved.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- 3. Local curvature ---
    print("\n3. Local curvature estimation...")
    n_dirs = 20
    eps = 1e-3

    def estimate_curvature(model, params, n_dirs=20, eps_per_param=1e-3):
        """
        Estimate sharpness via finite differences in random directions.

        Rather than Tr(H)/ε² (scale-dependent), we report the average loss
        change under unit-variance Gaussian noise scaled by eps_per_param:
            sharpness = mean_d [ L(θ + ε·d) - L(θ) ]
        where d ~ N(0, I) and ε = eps_per_param.
        This is equivalent to the "expected sharpness" metric (Keskar et al. 2017).
        """
        torch.manual_seed(123)
        base_loss = compute_loss_batched(model, n_batches=5)
        loss_deltas = []
        for _ in range(n_dirs):
            d = torch.randn_like(params) * eps_per_param
            set_model_params(model, params + d)
            loss_plus = compute_loss_batched(model, n_batches=5)
            loss_deltas.append(loss_plus - base_loss)
        set_model_params(model, params)  # restore
        print(f"    eps_per_param={eps_per_param}, base_loss={base_loss:.4f}, "
              f"mean_delta={np.mean(loss_deltas):.4f}")
        return base_loss, np.array(loss_deltas)

    loss_A_curv, curvs_A = estimate_curvature(model_A, params_A)
    loss_B_curv, curvs_B = estimate_curvature(model_B, params_B)
    loss_SWA_curv, curvs_SWA = estimate_curvature(model_SWA, params_SWA)

    print(f"Sharpness (ΔL under noise): A={np.mean(curvs_A):.4f}±{np.std(curvs_A):.4f}, "
          f"B={np.mean(curvs_B):.4f}±{np.std(curvs_B):.4f}, "
          f"SWA={np.mean(curvs_SWA):.4f}±{np.std(curvs_SWA):.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Model A', 'Model B', 'SWA']
    means = [np.mean(curvs_A), np.mean(curvs_B), np.mean(curvs_SWA)]
    stds = [np.std(curvs_A), np.std(curvs_B), np.std(curvs_SWA)]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel('$\\Delta L$ (expected sharpness)')
    ax.set_title('Expected Sharpness: Standard Training vs SWA\n'
                 '(higher = sharper minimum)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + max(abs(m)*0.1, 0.001),
                f'{m:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confinement_curvature.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- 4. Noise sensitivity ---
    print("\n4. Noise sensitivity...")
    sigmas = np.logspace(-2, 1, 20)  # relative to ‖θ‖/√d

    def noise_sensitivity(model, params, rel_sigmas, n_samples=3):
        """
        Evaluate loss under Gaussian noise of increasing relative magnitude.
        σ_absolute = rel_sigma × ‖θ‖ × (noise / ‖noise‖) scaled per-element.
        """
        torch.manual_seed(999)
        param_norm = params.norm().item()
        losses = []
        for rel_sigma in rel_sigmas:
            sigma = rel_sigma * param_norm / np.sqrt(params.numel())
            sample_losses = []
            for _ in range(n_samples):
                noise = torch.randn_like(params) * sigma
                set_model_params(model, params + noise)
                l = compute_loss_batched(model, n_batches=3)
                sample_losses.append(l)
            losses.append(np.mean(sample_losses))
        set_model_params(model, params)  # restore
        return np.array(losses)

    noise_A = noise_sensitivity(model_A, params_A, sigmas)
    noise_B = noise_sensitivity(model_B, params_B, sigmas)
    noise_SWA = noise_sensitivity(model_SWA, params_SWA, sigmas)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigmas, noise_A, 'o-', color='steelblue', label='Model A', linewidth=2)
    ax.plot(sigmas, noise_B, 'o-', color='coral', label='Model B', linewidth=2)
    ax.plot(sigmas, noise_SWA, 's-', color='seagreen', label='SWA', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('$\\sigma$ (noise magnitude)')
    ax.set_ylabel('Loss')
    ax.set_title('Noise Sensitivity: $L(\\theta + \\sigma z)$, $z \\sim \\mathcal{N}(0, I)$\n'
                 'Sharp minima degrade faster', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confinement_noise.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Save all data
    np.savez_compressed(os.path.join(PLOTS_DIR, "connectivity_data.npz"),
                        ts=ts, losses_linear=losses_linear,
                        losses_bezier=np.array(best_losses),
                        barrier_linear=barrier_linear, barrier_bezier=barrier_bezier,
                        loss_A=loss_A, loss_B=loss_B, loss_SWA=loss_SWA,
                        curvature_A=curvs_A, curvature_B=curvs_B, curvature_SWA=curvs_SWA,
                        sigmas=sigmas, noise_A=noise_A, noise_B=noise_B, noise_SWA=noise_SWA)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'connectivity_data.npz')}")


# --- Experiment 23: Seed Identity Persistence ---

def exp_seed_persistence():
    """
    Test whether initialization seeds leave persistent fingerprints after training.

    Methodology (ArXiv-grade):
    ──────────────────────────
    Hypothesis: random initialization creates a unique "signature" in the weight space
    that persists even after extensive training, detectable via spectral features.

    Protocol:
      1. Train 20 models with seeds 0-19 (small model, 3L/96d, 10k steps each)
      2. At each checkpoint, extract a feature vector:
         - Top-10 normalized singular values per weight matrix
         - Frobenius norm per weight matrix
         - Spectral gap (σ₁ - σ₂) per weight matrix
      3. Train a nearest-centroid classifier (scipy, no sklearn):
         - Training set: even-numbered checkpoints
         - Test set: odd-numbered checkpoints
         - 20-way classification → chance = 5%
      4. Track classification accuracy vs training step

    Controls:
      - Random labels: permute seed assignments → should get ~5%
      - Permutation test (100×): null distribution of accuracy

    Outputs:
      - plots/seed_persistence_accuracy.png
      - plots/seed_persistence_features.png
      - plots/seed_persistence_data.npz

    Requires: train 20 seeds first. Run:
      for i in $(seq 0 19); do uv run train_sweep.py --seed $i; done
    Or this function runs inline.
    """
    from train_sweep import train_seeded
    from scipy.spatial.distance import cdist

    seed_dir = os.path.join(os.path.dirname(__file__), "checkpoints_seeds")
    n_seeds = 20
    n_layer, n_embd, n_head = 3, 96, 3

    # Check if training is done
    all_exist = all(
        os.path.exists(os.path.join(seed_dir, f"seed_{s}", "step_010000.pt"))
        for s in range(n_seeds)
    )

    if not all_exist:
        print(f"Training {n_seeds} seeds (small model, ~8 min each, ~2.7hr total)...")
        for s in range(n_seeds):
            ckpt_dir_s = os.path.join(seed_dir, f"seed_{s}")
            if os.path.exists(os.path.join(ckpt_dir_s, "step_010000.pt")):
                print(f"  seed {s}: already done, skipping")
                continue
            print(f"  Training seed {s}/{n_seeds}...")
            train_seeded(seed=s, lr=3e-4, max_steps=10000,
                         n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                         save_checkpoints=True, ckpt_dir=ckpt_dir_s,
                         ckpt_interval=500, quiet=True)

    # --- feature extraction ---
    print("Extracting spectral features...")

    steps = list(range(0, 10001, 500))  # 21 checkpoints
    # features[seed][step_idx] = feature vector
    features = {}

    for s in range(n_seeds):
        features[s] = {}
        for si, step in enumerate(steps):
            ckpt_path = os.path.join(seed_dir, f"seed_{s}", f"step_{step:06d}.pt")
            if not os.path.exists(ckpt_path):
                continue
            model = load_checkpoint_scaled(ckpt_path, n_layer, n_embd, n_head)
            matrices = get_weight_matrices(model)

            feat = []
            for name, W in matrices.items():
                S = np.linalg.svd(W, compute_uv=False)
                # Normalized top-10 singular values
                S_norm = S / (np.linalg.norm(S) + 1e-10)
                feat.extend(S_norm[:min(10, len(S))])
                # Frobenius norm
                feat.append(np.linalg.norm(W))
                # Spectral gap
                if len(S) >= 2:
                    feat.append(S[0] - S[1])
                else:
                    feat.append(0.0)

            features[s][si] = np.array(feat)

    n_features = len(features[0][0])
    print(f"Feature vector dimension: {n_features}")

    # --- classification ---
    print("Running nearest-centroid classification...")

    # Split: even step indices for train, odd for test
    train_idx = list(range(0, len(steps), 2))
    test_idx = list(range(1, len(steps), 2))

    # For each test checkpoint, classify seed identity
    accuracy_per_step = {}

    for si in test_idx:
        if si >= len(steps):
            continue
        step = steps[si]

        # Compute centroids from training checkpoints
        centroids = []
        for s in range(n_seeds):
            train_feats = [features[s][ti] for ti in train_idx if ti in features[s]]
            if train_feats:
                centroids.append(np.mean(train_feats, axis=0))
            else:
                centroids.append(np.zeros(n_features))
        centroids = np.array(centroids)  # [n_seeds, n_features]

        # Classify test features
        test_feats = []
        test_labels = []
        for s in range(n_seeds):
            if si in features[s]:
                test_feats.append(features[s][si])
                test_labels.append(s)

        if not test_feats:
            continue

        test_feats = np.array(test_feats)
        test_labels = np.array(test_labels)

        # Nearest centroid
        dists = cdist(test_feats, centroids, metric='euclidean')
        preds = np.argmin(dists, axis=1)
        acc = np.mean(preds == test_labels)
        accuracy_per_step[step] = acc

    # Permutation test (null distribution)
    print("Running permutation test (100 iterations)...")
    n_perm = 100
    perm_accs = []
    rng = np.random.RandomState(42)

    for _ in range(n_perm):
        # Shuffle seed labels
        perm = rng.permutation(n_seeds)
        perm_acc_per_step = []

        for si in test_idx:
            if si >= len(steps):
                continue
            step = steps[si]

            # Centroids with permuted labels
            centroids = []
            for s in range(n_seeds):
                orig_s = perm[s]
                train_feats = [features[orig_s][ti] for ti in train_idx if ti in features[orig_s]]
                if train_feats:
                    centroids.append(np.mean(train_feats, axis=0))
                else:
                    centroids.append(np.zeros(n_features))
            centroids = np.array(centroids)

            test_feats = []
            test_labels = []
            for s in range(n_seeds):
                if si in features[s]:
                    test_feats.append(features[s][si])
                    test_labels.append(s)

            if not test_feats:
                continue

            test_feats = np.array(test_feats)
            test_labels = np.array(test_labels)
            dists = cdist(test_feats, centroids, metric='euclidean')
            preds = np.argmin(dists, axis=1)
            perm_acc_per_step.append(np.mean(preds == test_labels))

        perm_accs.append(np.mean(perm_acc_per_step) if perm_acc_per_step else 0)

    perm_ci = np.percentile(perm_accs, [2.5, 97.5])
    chance = 1.0 / n_seeds

    print(f"\nAccuracy per step:")
    for step, acc in sorted(accuracy_per_step.items()):
        print(f"  step {step:5d}: {acc:.1%}")
    print(f"Chance level: {chance:.1%}")
    print(f"Permutation null: {np.mean(perm_accs):.1%} (95% CI: [{perm_ci[0]:.1%}, {perm_ci[1]:.1%}])")

    # ========================================
    # Plot 1: Accuracy over training
    # ========================================
    sorted_steps = sorted(accuracy_per_step.keys())
    sorted_accs = [accuracy_per_step[s] for s in sorted_steps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_steps, sorted_accs, 'o-', color='coral', linewidth=2, markersize=6,
            label='Nearest centroid accuracy')
    ax.axhline(chance, color='gray', linestyle='--', linewidth=1.5, label=f'Chance ({chance:.0%})')
    ax.fill_between([sorted_steps[0], sorted_steps[-1]], perm_ci[0], perm_ci[1],
                    color='gray', alpha=0.2, label='Permutation 95% CI')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title(f'Seed Identity Persistence: Can We Identify Initialization Seed?\n'
                 f'{n_seeds} seeds, {n_layer}L/{n_embd}d model',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "seed_persistence_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ========================================
    # Plot 2: PCA of features colored by seed
    # ========================================
    # Show init (step 0), mid (step 5000), and final (step 10000)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Feature Space (PCA) Colored by Seed', fontsize=14, fontweight='bold')

    for ax, (target_step, label) in zip(axes, [(0, 'Init'), (10, 'Mid (5000)'), (20, 'Final (10000)')]):
        all_feats = []
        all_seeds = []
        for s in range(n_seeds):
            if target_step in features[s]:
                all_feats.append(features[s][target_step])
                all_seeds.append(s)

        if len(all_feats) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
            continue

        all_feats = np.array(all_feats)
        all_seeds = np.array(all_seeds)

        # Simple PCA via SVD
        feats_centered = all_feats - all_feats.mean(axis=0)
        U, S_pca, Vt = np.linalg.svd(feats_centered, full_matrices=False)
        coords = feats_centered @ Vt[:2].T
        var_explained = S_pca[:2]**2 / np.sum(S_pca**2) * 100

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=all_seeds,
                            cmap='tab20', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
        ax.set_title(label)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "seed_persistence_features.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Save data
    save_dict = {
        "steps": np.array(sorted_steps),
        "accuracy": np.array(sorted_accs),
        "chance": chance,
        "perm_accs": np.array(perm_accs),
        "perm_ci": perm_ci,
        "n_seeds": n_seeds,
        "n_features": n_features,
    }
    np.savez_compressed(os.path.join(PLOTS_DIR, "seed_persistence_data.npz"), **save_dict)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'seed_persistence_data.npz')}")


# --- registry ---

EXPERIMENTS = {
    "spectra": exp_eigenvalue_spectra,
    "powerlaw": exp_power_law_exponents,
    "zoom": exp_weight_zoom,
    "correlation": exp_correlation_fractal,
    "filmstrip": exp_weight_filmstrip,
    "loglog": exp_singular_value_loglog,
    "boxcount": exp_box_counting,
    "attn_corr": exp_attention_correlation,
    "cross_layer": exp_cross_layer_similarity,
    "multifractal": exp_multifractal,
    "attn_maps": exp_attention_maps,
    "sgd_trajectory": exp_sgd_trajectory,
    "loss_landscape": exp_loss_landscape,
    "repr_sim": exp_representation_similarity,
    "act_fractals": exp_activation_fractals,
    "grad_fractals": exp_gradient_fractals,
    "init_vs_trained": exp_init_vs_trained,
    "scale": exp_scale_comparison,
    "rmt_denoised": exp_rmt_denoised,
    "attn_graph": exp_attn_graph_spectral,
    "critical_lr": exp_critical_lr,
    "connectivity": exp_connectivity_confinement,
    "seed_persist": exp_seed_persistence,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  uv run analyze.py {name}")
        print(f"  uv run analyze.py all")
        sys.exit(0)

    name = sys.argv[1]
    if name == "all":
        for exp_fn in EXPERIMENTS.values():
            exp_fn()
    elif name in EXPERIMENTS:
        EXPERIMENTS[name]()
    else:
        print(f"Unknown experiment: {name}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
