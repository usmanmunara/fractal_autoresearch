"""
One-time data preparation: downloads tiny shakespeare and tokenizes it.
Run once before training: uv run prepare.py
"""

import os
import numpy as np
import requests
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, "input.txt")
    if os.path.exists(filepath):
        print(f"Data already exists at {filepath}")
        return filepath
    print("Downloading tiny shakespeare...")
    resp = requests.get(DATASET_URL)
    resp.raise_for_status()
    with open(filepath, "w") as f:
        f.write(resp.text)
    print(f"Downloaded {len(resp.text):,} characters to {filepath}")
    return filepath


def tokenize_and_split(filepath):
    """Tokenize with tiktoken gpt2 encoder, split 90/10 train/val."""
    enc = tiktoken.get_encoding("gpt2")
    with open(filepath, "r") as f:
        text = f.read()
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    n = len(tokens)
    split = int(n * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    print(f"Train: {len(train_tokens):,} tokens -> {train_path}")
    print(f"Val:   {len(val_tokens):,} tokens -> {val_path}")


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def load_tokens(split="train"):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


if __name__ == "__main__":
    filepath = download_data()
    tokenize_and_split(filepath)
    print("Done! Ready to train.")
