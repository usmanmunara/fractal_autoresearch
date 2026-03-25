"""
Export real experiment data into scene-ready NPZ files.

Each evidence scene in the Manim video loads from these files.
No synthetic data — every number on screen is traceable to real checkpoints.

Usage:
    uv run python video/export_data.py
"""

import glob
import os
import sys

import numpy as np
import torch
from scipy import stats

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from train import GPT, CKPT_DIR, N_EMBD, N_LAYER, BLOCK_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_checkpoint(path):
    model = GPT()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_checkpoints(ckpt_dir=CKPT_DIR):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if not paths:
        print(f"ERROR: No checkpoints in {ckpt_dir}")
        sys.exit(1)
    return paths


def get_weight_matrices(model):
    matrices = {}
    for name, param in model.named_parameters():
        if param.dim() == 2 and "wte" not in name and "wpe" not in name:
            matrices[name] = param.detach().numpy()
    return matrices


# ============================================================
# 1. Spectra: singular values per layer at init and trained
# ============================================================
def export_spectra():
    print("Exporting singular value spectra...")
    ckpts = get_checkpoints()
    init_model = load_checkpoint(ckpts[0])
    final_model = load_checkpoint(ckpts[-1])
    final_step = int(os.path.basename(ckpts[-1]).split("_")[1].split(".")[0])

    # All 2D weight layers at init and trained
    init_matrices = get_weight_matrices(init_model)
    final_matrices = get_weight_matrices(final_model)

    layer_names = sorted(init_matrices.keys())
    data = {
        "layer_names": np.array(layer_names, dtype=object),
        "final_step": np.array(final_step),
    }

    for name in layer_names:
        safe = name.replace(".", "_")
        W_init = init_matrices[name]
        W_final = final_matrices[name]
        sv_init = np.linalg.svd(W_init, compute_uv=False)
        sv_final = np.linalg.svd(W_final, compute_uv=False)

        # Fit power law: log(sv) = -alpha * log(rank) + const
        ranks = np.arange(1, len(sv_final) + 1)
        mask = sv_final > 1e-10
        slope, intercept, r_val, _, _ = stats.linregress(
            np.log(ranks[mask]), np.log(sv_final[mask])
        )
        alpha = -slope
        r2 = r_val ** 2

        data[f"sv_init_{safe}"] = sv_init
        data[f"sv_final_{safe}"] = sv_final
        data[f"alpha_{safe}"] = np.array(alpha)
        data[f"r2_{safe}"] = np.array(r2)

    # Also export spectra evolution across all checkpoints for one representative layer
    repr_layer = f"blocks.{N_LAYER // 2}.mlp.c_fc.weight"
    evo_steps = []
    evo_svs = []
    evo_alphas = []
    for ckpt_path in ckpts:
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        W = get_weight_matrices(model)[repr_layer]
        sv = np.linalg.svd(W, compute_uv=False)
        ranks = np.arange(1, len(sv) + 1)
        mask = sv > 1e-10
        slope, _, r_val, _, _ = stats.linregress(np.log(ranks[mask]), np.log(sv[mask]))
        evo_steps.append(step)
        evo_svs.append(sv)
        evo_alphas.append(-slope)

    data["evo_layer"] = np.array(repr_layer, dtype=object)
    data["evo_steps"] = np.array(evo_steps)
    data["evo_svs"] = np.array(evo_svs)  # [n_checkpoints, n_sv]
    data["evo_alphas"] = np.array(evo_alphas)

    path = os.path.join(DATA_DIR, "spectra.npz")
    np.savez(path, **data)
    print(f"  Saved: {path} ({len(layer_names)} layers)")
    return data


# ============================================================
# 2. Attention correlations: real correlation matrices
# ============================================================
def export_attention_correlations():
    print("Exporting attention correlation matrices...")
    ckpts = get_checkpoints()
    init_model = load_checkpoint(ckpts[0])
    final_model = load_checkpoint(ckpts[-1])
    final_step = int(os.path.basename(ckpts[-1]).split("_")[1].split(".")[0])

    data = {"final_step": np.array(final_step), "n_layers": np.array(N_LAYER)}

    for i in range(N_LAYER):
        layer_name = f"blocks.{i}.attn.c_attn.weight"
        W_init = get_weight_matrices(init_model)[layer_name]
        W_final = get_weight_matrices(final_model)[layer_name]

        corr_init = np.corrcoef(W_init)
        corr_final = np.corrcoef(W_final)
        corr_init = np.clip(corr_init, -1, 1)
        corr_final = np.clip(corr_final, -1, 1)

        data[f"corr_init_L{i}"] = corr_init.astype(np.float32)
        data[f"corr_final_L{i}"] = corr_final.astype(np.float32)

    path = os.path.join(DATA_DIR, "attention_correlations.npz")
    np.savez_compressed(path, **data)
    print(f"  Saved: {path} (shape: {corr_init.shape})")
    return data


# ============================================================
# 3. Cross-layer similarity: cosine similarities + spectra overlay
# ============================================================
def export_cross_layer_similarity():
    print("Exporting cross-layer similarity data...")
    ckpts = get_checkpoints()
    final_model = load_checkpoint(ckpts[-1])
    matrices = get_weight_matrices(final_model)

    weight_types = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    data = {"n_layers": np.array(N_LAYER), "weight_types": np.array(weight_types, dtype=object)}

    for wtype in weight_types:
        safe_wtype = wtype.replace(".", "_")
        sv_list = []
        sv_norm_list = []
        for i in range(N_LAYER):
            name = f"blocks.{i}.{wtype}.weight"
            if name not in matrices:
                continue
            W = matrices[name]
            sv = np.linalg.svd(W, compute_uv=False)
            sv_norm = sv / sv[0]
            sv_list.append(sv)
            sv_norm_list.append(sv_norm)

        # Pairwise cosine similarities
        n = len(sv_list)
        sim_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a = sv_list[i] / np.linalg.norm(sv_list[i])
                b = sv_list[j] / np.linalg.norm(sv_list[j])
                min_len = min(len(a), len(b))
                sim = float(np.dot(a[:min_len], b[:min_len]))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Store per-layer to avoid object array issues
        data[f"n_{safe_wtype}"] = np.array(len(sv_list))
        for i, (sv, svn) in enumerate(zip(sv_list, sv_norm_list)):
            data[f"sv_{safe_wtype}_L{i}"] = sv
            data[f"sv_norm_{safe_wtype}_L{i}"] = svn
        data[f"sim_{safe_wtype}"] = sim_matrix

    path = os.path.join(DATA_DIR, "cross_layer_similarity.npz")
    np.savez(path, **data)
    print(f"  Saved: {path}")
    return data


# ============================================================
# 4. SGD trajectory: Hurst exponent + PCA projection
# ============================================================
def export_sgd_trajectory():
    print("Exporting SGD trajectory data...")
    ckpts = get_checkpoints()

    trajectory = []
    steps = []
    for ckpt_path in ckpts:
        step = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])
        model = load_checkpoint(ckpt_path)
        params = []
        for name, param in model.named_parameters():
            params.append(param.detach().numpy().flatten())
        trajectory.append(np.concatenate(params))
        steps.append(step)

    trajectory = np.array(trajectory)
    steps = np.array(steps)
    print(f"  Trajectory: {trajectory.shape} ({trajectory.shape[1]:,} params)")

    # Step-to-step displacements
    displacements = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

    # Multi-scale displacement analysis
    max_scale = len(ckpts) // 2
    scales = np.arange(1, max_scale + 1)
    mean_displacements = []
    for k in scales:
        dists = [np.linalg.norm(trajectory[i + k] - trajectory[i])
                 for i in range(len(trajectory) - k)]
        mean_displacements.append(np.mean(dists))
    mean_displacements = np.array(mean_displacements)

    # Hurst exponent fit
    log_scales = np.log(scales)
    log_disps = np.log(mean_displacements)
    H, intercept, r_value, _, _ = stats.linregress(log_scales, log_disps)
    r2 = r_value ** 2

    # PCA projection
    centered = trajectory - trajectory.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_3d = centered @ Vt[:3].T
    variance_explained = (S[:3] ** 2) / (S ** 2).sum()

    data = {
        "steps": steps,
        "displacements": displacements,
        "scales": scales,
        "mean_displacements": mean_displacements,
        "log_scales": log_scales,
        "log_disps": log_disps,
        "hurst_H": np.array(H),
        "hurst_intercept": np.array(intercept),
        "hurst_r2": np.array(r2),
        "fractal_dim": np.array(2 - H),
        "pca_3d": proj_3d,
        "variance_explained": variance_explained,
        "n_params": np.array(trajectory.shape[1]),
    }

    path = os.path.join(DATA_DIR, "sgd_trajectory.npz")
    np.savez(path, **data)
    print(f"  Saved: {path}")
    print(f"  H = {H:.4f} (R² = {r2:.4f}), D = {2-H:.4f}")
    return data


# ============================================================
# 5. Scale comparison: metrics across model sizes
# ============================================================
def export_scale_comparison():
    print("Exporting scale comparison metrics...")
    sys.path.insert(0, ROOT)

    scale_configs = [
        {
            "name": "small", "label": "Small (3L/96d)",
            "ckpt_dir": os.path.join(ROOT, "checkpoints_small"),
            "n_layer": 3, "n_embd": 96, "n_head": 3,
        },
        {
            "name": "medium", "label": "Medium (6L/192d)",
            "ckpt_dir": CKPT_DIR,
            "n_layer": N_LAYER, "n_embd": N_EMBD, "n_head": 6,
        },
        {
            "name": "large", "label": "Large (8L/256d)",
            "ckpt_dir": os.path.join(ROOT, "checkpoints_large"),
            "n_layer": 8, "n_embd": 256, "n_head": 8,
        },
    ]

    results = {}
    for cfg in scale_configs:
        ckpt_dir = cfg["ckpt_dir"]
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
        if len(ckpts) < 2:
            print(f"  Skipping {cfg['name']}: not enough checkpoints")
            continue

        print(f"  Processing {cfg['label']} ({len(ckpts)} checkpoints)...")

        # Load init and final
        if cfg["name"] == "medium":
            init_model = load_checkpoint(ckpts[0])
            final_model = load_checkpoint(ckpts[-1])
        else:
            import train_scale
            init_model = train_scale.GPT(cfg["n_layer"], cfg["n_embd"], cfg["n_head"])
            init_model.load_state_dict(torch.load(ckpts[0], map_location="cpu", weights_only=True))
            init_model.eval()
            final_model = train_scale.GPT(cfg["n_layer"], cfg["n_embd"], cfg["n_head"])
            final_model.load_state_dict(torch.load(ckpts[-1], map_location="cpu", weights_only=True))
            final_model.eval()

        # Power-law exponents
        trained_alphas = []
        trained_r2s = []
        for name, param in final_model.named_parameters():
            if param.ndim != 2 or param.shape[0] < 10:
                continue
            w = param.detach().numpy()
            sv = np.linalg.svd(w, compute_uv=False)
            sv = sv[sv > 1e-10]
            ranks = np.arange(1, len(sv) + 1)
            slope, _, r, _, _ = stats.linregress(np.log(ranks), np.log(sv))
            trained_alphas.append(-slope)
            trained_r2s.append(r ** 2)

        # Cross-layer spectral similarity
        spectra = []
        for name, param in final_model.named_parameters():
            if param.ndim != 2 or param.shape[0] < 10:
                continue
            sv = np.linalg.svd(param.detach().numpy(), compute_uv=False)
            sv = sv / sv.sum()
            spectra.append(sv)

        cross_sims = []
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                min_len = min(len(spectra[i]), len(spectra[j]))
                a, b = spectra[i][:min_len], spectra[j][:min_len]
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                cross_sims.append(sim)

        # Hurst exponent
        hurst_H, hurst_r2 = np.nan, np.nan
        if len(ckpts) >= 4:
            traj = []
            for p in ckpts:
                state = torch.load(p, map_location="cpu", weights_only=True)
                params = [state[k].numpy().flatten() for k in sorted(state.keys())]
                traj.append(np.concatenate(params))
            traj = np.array(traj)
            max_sc = len(ckpts) // 2
            sc = list(range(1, max_sc + 1))
            md = [np.mean([np.linalg.norm(traj[i + k] - traj[i])
                           for i in range(len(traj) - k)]) for k in sc]
            if len(sc) >= 3:
                H, _, r, _, _ = stats.linregress(np.log(sc), np.log(md))
                hurst_H, hurst_r2 = H, r ** 2

        n_params = sum(p.numel() for p in final_model.parameters())
        results[cfg["name"]] = {
            "label": cfg["label"],
            "n_params": n_params,
            "mean_alpha": np.mean(trained_alphas),
            "mean_r2": np.mean(trained_r2s),
            "cross_layer_sim": np.mean(cross_sims) if cross_sims else 0,
            "hurst_H": hurst_H,
            "hurst_r2": hurst_r2,
        }

    # Save as flat arrays for easy Manim consumption
    names = ["small", "medium", "large"]
    available = [n for n in names if n in results]
    data = {
        "scale_names": np.array(available, dtype=object),
        "scale_labels": np.array([results[n]["label"] for n in available], dtype=object),
        "n_params": np.array([results[n]["n_params"] for n in available]),
        "mean_alpha": np.array([results[n]["mean_alpha"] for n in available]),
        "mean_r2": np.array([results[n]["mean_r2"] for n in available]),
        "cross_layer_sim": np.array([results[n]["cross_layer_sim"] for n in available]),
        "hurst_H": np.array([results[n]["hurst_H"] for n in available]),
        "hurst_r2": np.array([results[n]["hurst_r2"] for n in available]),
    }

    path = os.path.join(DATA_DIR, "scale_comparison.npz")
    np.savez(path, **data)
    print(f"  Saved: {path}")
    for n in available:
        r = results[n]
        print(f"    {r['label']}: α={r['mean_alpha']:.3f}, "
              f"sim={r['cross_layer_sim']:.4f}, H={r['hurst_H']:.3f}")
    return data


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("EXPORTING REAL DATA FOR MANIM SCENES")
    print("=" * 60)

    export_spectra()
    print()
    export_attention_correlations()
    print()
    export_cross_layer_similarity()
    print()
    export_sgd_trajectory()
    print()
    export_scale_comparison()

    print()
    print("=" * 60)
    print("ALL DATA EXPORTED SUCCESSFULLY")
    print(f"Files in: {DATA_DIR}/")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".npz"):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"  {f} ({size / 1024:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
