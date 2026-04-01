"""
Seeded training with configurable LR for sweep experiments.

Usage:
    uv run train_sweep.py --test                     # quick smoke test (1 seed, 100 steps)
    uv run train_sweep.py --seeds 16 --lr 3e-4       # train 16 seeds at given LR
    uv run train_sweep.py --critical-lr small         # find critical LR for small model
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from prepare import load_tokens
from train_scale import GPT, BLOCK_SIZE, BATCH_SIZE, VOCAB_SIZE, DEVICE


def get_batch(split):
    """Load a batch (same as train_scale but importable)."""
    data = load_tokens(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train_seeded(seed, lr, max_steps, ckpt_dir=None,
                 n_layer=6, n_embd=192, n_head=6,
                 save_checkpoints=False, ckpt_interval=500,
                 eval_interval=500, quiet=False):
    """
    Train a GPT model with a fixed seed and learning rate.

    Returns:
        dict with keys: losses (list of floats every eval_interval steps),
                        final_loss (float), diverged (bool)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GPT(n_layer, n_embd, n_head).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    min_lr = lr * 0.1

    warmup_steps = min(100, max_steps // 5)

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    if ckpt_dir and save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "step_000000.pt"))

    losses = []
    t0 = time.time()

    for step in range(1, max_steps + 1):
        cur_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        x, y = get_batch("train")
        _, loss = model(x, y)

        if torch.isnan(loss) or loss.item() > 100:
            losses.append(float("nan"))
            return {"losses": losses, "final_loss": float("nan"), "diverged": True}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            losses.append(loss.item())
            if not quiet:
                print(f"  seed={seed} lr={lr:.1e} step={step} loss={loss.item():.4f} [{time.time()-t0:.0f}s]")

        if save_checkpoints and ckpt_dir and step % ckpt_interval == 0:
            path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            torch.save(model.state_dict(), path)

    final_loss = losses[-1] if losses else float("nan")
    return {"losses": losses, "final_loss": final_loss, "diverged": False}


def classify_run(result, stable_threshold=2.0):
    """
    Classify a training run outcome.

    Args:
        result: dict from train_seeded
        stable_threshold: final loss must be below this for "stable"

    Returns:
        "stable", "slow", or "diverged"
    """
    if result["diverged"]:
        return "diverged"
    if math.isnan(result["final_loss"]):
        return "diverged"
    if result["final_loss"] > stable_threshold:
        return "slow"
    return "stable"


def find_critical_lr(n_layer=3, n_embd=96, n_head=3, max_steps=1000, n_seeds=3):
    """
    Binary search for the critical LR where training starts to diverge.

    Returns:
        lr_critical (float), results (dict mapping lr -> list of outcomes)
    """
    print(f"Finding critical LR for {n_layer}L/{n_embd}d model...")

    # coarse sweep
    lrs = np.logspace(-4, -1, 10)
    results = {}

    for lr in lrs:
        outcomes = []
        for seed in range(n_seeds):
            r = train_seeded(seed, float(lr), max_steps,
                             n_layer=n_layer, n_embd=n_embd, n_head=n_head, quiet=True)
            outcomes.append(classify_run(r))
        frac_stable = sum(1 for o in outcomes if o == "stable") / len(outcomes)
        results[float(lr)] = outcomes
        print(f"  lr={lr:.1e}: {outcomes} ({frac_stable:.0%} stable)", flush=True)

    # find boundary: last LR where >50% stable
    sorted_lrs = sorted(results.keys())
    lr_critical = sorted_lrs[-1]
    for lr in sorted_lrs:
        frac_stable = sum(1 for o in results[lr] if o == "stable") / len(results[lr])
        if frac_stable < 0.5:
            lr_critical = lr
            break

    print(f"Critical LR estimate: {lr_critical:.2e}")
    return lr_critical, results


if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Smoke test: 1 seed, 100 steps, small model")
        r = train_seeded(seed=42, lr=3e-4, max_steps=100,
                         n_layer=3, n_embd=96, n_head=3, eval_interval=50)
        print(f"Result: final_loss={r['final_loss']:.4f}, diverged={r['diverged']}")
        print(f"Classification: {classify_run(r)}")

    elif "--critical-lr" in sys.argv:
        idx = sys.argv.index("--critical-lr")
        scale = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "small"
        from train_scale import SCALES
        cfg = SCALES.get(scale, SCALES["small"])
        find_critical_lr(**cfg)

    elif "--seeds" in sys.argv:
        idx = sys.argv.index("--seeds")
        n_seeds = int(sys.argv[idx + 1])
        lr = 3e-4
        if "--lr" in sys.argv:
            lr_idx = sys.argv.index("--lr")
            lr = float(sys.argv[lr_idx + 1])
        print(f"Training {n_seeds} seeds at lr={lr:.1e}")
        for seed in range(n_seeds):
            r = train_seeded(seed=seed, lr=lr, max_steps=10000,
                             n_layer=3, n_embd=96, n_head=3)
            print(f"  seed={seed}: {classify_run(r)}")

    else:
        print(__doc__)
