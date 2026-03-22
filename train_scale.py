"""
Train GPT models at different scales for fractal comparison.
Reuses the same architecture from train.py but with configurable sizes.

Usage:
    uv run train_scale.py small   # 3 layers, 96 embd (~1.6M params)
    uv run train_scale.py large   # 8 layers, 256 embd (~26M params)
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_tokens

# --- scale configs ---
SCALES = {
    "small": {"n_layer": 3, "n_embd": 96, "n_head": 3},
    "large": {"n_layer": 8, "n_embd": 256, "n_head": 8},
}

# shared hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 128
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_STEPS = 10000
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 500
EVAL_STEPS = 20
VOCAB_SIZE = 50257

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=DROPOUT if self.training else 0.0)
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_layer, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.wte = nn.Embedding(VOCAB_SIZE, n_embd)
        self.wpe = nn.Embedding(BLOCK_SIZE, n_embd)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB_SIZE, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(split):
    data = load_tokens(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def train(scale_name):
    cfg = SCALES[scale_name]
    ckpt_dir = os.path.join(os.path.dirname(__file__), f"checkpoints_{scale_name}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = GPT(cfg["n_layer"], cfg["n_embd"], cfg["n_head"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def get_lr(step):
        if step < 100:
            return LEARNING_RATE * step / 100
        decay_ratio = (step - 100) / (MAX_STEPS - 100)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE - LEARNING_RATE * 0.1)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "step_000000.pt"))
    print(f"Training {scale_name} model on {DEVICE} for {MAX_STEPS} steps...")
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if step % CHECKPOINT_INTERVAL == 0:
            path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
            torch.save(model.state_dict(), path)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"step_{MAX_STEPS:06d}.pt"))
    print(f"Done! {time.time() - t0:.1f}s total. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SCALES:
        print(f"Usage: uv run train_scale.py <{'|'.join(SCALES.keys())}>")
        sys.exit(1)
    train(sys.argv[1])
