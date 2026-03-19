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
