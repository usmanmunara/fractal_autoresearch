"""
Render all scenes and stitch them into a single video.

Usage:
    uv run python video/render_all.py          # low quality (fast)
    uv run python video/render_all.py --hq     # high quality
    uv run python video/render_all.py --4k     # 4K quality
"""

import subprocess
import sys
import os
from pathlib import Path

SCENES = [
    "TheHook",
    "WhatIsFractal",
    "TheExperiment",
    "PowerLawsEmerge",
    "FractalAttention",
    "ScaledCopies",
    "HurstExponent",
    "FractalTrajectory",
    "Universality",
    "WhatDoesItMean",
]

SCRIPT = "video/scenes.py"
ROOT = Path(__file__).parent.parent


def main():
    quality = "-ql"
    quality_dir = "480p15"
    if "--hq" in sys.argv:
        quality = "-qh"
        quality_dir = "1080p60"
    elif "--4k" in sys.argv:
        quality = "-qk"
        quality_dir = "2160p60"

    media_dir = ROOT / "media" / "videos" / "scenes" / quality_dir

    print(f"Rendering {len(SCENES)} scenes at {quality} quality...\n")

    for i, scene in enumerate(SCENES, 1):
        print(f"[{i}/{len(SCENES)}] Rendering {scene}...")
        result = subprocess.run(
            ["uv", "run", "manim", quality, SCRIPT, scene],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR rendering {scene}:")
            print(result.stderr[-500:] if result.stderr else "No stderr")
            sys.exit(1)
        print(f"  Done.")

    # Create concat file list for ffmpeg
    concat_file = media_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for scene in SCENES:
            mp4 = media_dir / f"{scene}.mp4"
            if mp4.exists():
                f.write(f"file '{mp4.name}'\n")
            else:
                print(f"WARNING: {mp4} not found, skipping")

    # Stitch with ffmpeg
    output = media_dir / "NeuralNetworksAreFractals_FULL.mp4"
    print(f"\nStitching into {output}...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-500:]}")
        sys.exit(1)

    print(f"\nFull video: {output}")
    print(f"Duration: run `ffprobe -show_entries format=duration {output}` to check")


if __name__ == "__main__":
    main()
