"""
fractal_agents — run the full pipeline: prepare data, train, analyze.
Usage: uv run main.py
"""

import subprocess
import sys


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    subprocess.run([sys.executable, cmd], check=True)


if __name__ == "__main__":
    run("prepare.py")
    run("train.py")
    run("analyze.py")
