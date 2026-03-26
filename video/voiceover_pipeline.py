"""
Generate ElevenLabs voiceover from video/voiceover_script.md and stitch onto the video.

Usage:
    uv run python video/voiceover_pipeline.py
    uv run python video/voiceover_pipeline.py --quality 1080p60
    uv run python video/voiceover_pipeline.py --voice-name "Rachel" --force-tts

Environment:
    ELEVENLABS_API_KEY=...
Optional:
    ELEVENLABS_VOICE_ID=...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "video"
VOICE_SCRIPT_PATH = VIDEO_DIR / "voiceover_script.md"
AUDIO_DIR = VIDEO_DIR / "voiceover_audio"
SCENE_AUDIO_DIR = AUDIO_DIR / "scenes"

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

TITLE_TO_SCENE = {
    "thehook": "TheHook",
    "whatisfractal": "WhatIsFractal",
    "theexperiment": "TheExperiment",
    "powerlawsemerge": "PowerLawsEmerge",
    "fractalattention": "FractalAttention",
    "scaledcopies": "ScaledCopies",
    "hurstexponent": "HurstExponent",
    "fractaltrajectory": "FractalTrajectory",
    "universality": "Universality",
    "whatdoesitmean": "WhatDoesItMean",
}


def read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def get_env(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    dotenv = read_env_file(ROOT / ".env")
    return dotenv.get(name)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def clean_scene_text(text: str) -> str:
    t = text.strip()
    if t.startswith('"'):
        t = t[1:]
    if t.endswith('"'):
        t = t[:-1]
    t = t.replace("“", '"').replace("”", '"')
    return t.strip()


def split_scene7_if_needed(text: str) -> tuple[str, str]:
    marker = "In 3D PCA space"
    if marker in text:
        left, right = text.split(marker, 1)
        a = left.strip()
        b = (marker + right).strip()
        if a and b:
            return a, b
    return text.strip(), text.strip()


def parse_voice_script(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing voice script: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    heading_re = re.compile(r"^##\s+Scene\s+[^—-]+[—-]\s+(.+?)(?:\s+\(~.*\))?\s*$")

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title is not None:
            sections.append((current_title, current_lines.copy()))
        current_title = None
        current_lines = []

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            flush()
            current_title = m.group(1).strip()
            current_lines = []
            continue
        if current_title is not None:
            if line.strip() == "---":
                continue
            current_lines.append(line)
    flush()

    scene_text: dict[str, str] = {}
    for title, content_lines in sections:
        raw_text = clean_scene_text("\n".join(content_lines))
        ntitle = normalize_title(title)

        if "hurstexponent" in ntitle and "fractaltrajectory" in ntitle:
            hurst_text, traj_text = split_scene7_if_needed(raw_text)
            scene_text["HurstExponent"] = hurst_text
            scene_text["FractalTrajectory"] = traj_text
            continue

        scene_name = TITLE_TO_SCENE.get(ntitle)
        if scene_name:
            scene_text[scene_name] = raw_text

    missing = [s for s in SCENES if s not in scene_text]
    if missing:
        raise ValueError(
            "Voice script parsing did not map all scenes. Missing: "
            + ", ".join(missing)
        )
    return scene_text


def eleven_get_voices(api_key: str) -> list[dict]:
    r = requests.get(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to list voices: {r.status_code} {r.text[:300]}")
    payload = r.json()
    return payload.get("voices", [])


def choose_voice_id(api_key: str, voice_id: str | None, voice_name: str | None) -> tuple[str, str]:
    if voice_id:
        # Avoid requiring voices_read permission when explicit ID is provided.
        return voice_id, "provided"

    try:
        voices = eleven_get_voices(api_key)
    except RuntimeError as e:
        if "missing_permissions" not in str(e):
            raise
        # Known public premade voice IDs as fallback when voices_read is unavailable.
        fallback = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        return fallback, "fallback-rachel"

    if not voices:
        raise RuntimeError("No ElevenLabs voices available for this API key.")

    if voice_name:
        low = voice_name.lower()
        for v in voices:
            if v.get("name", "").lower() == low:
                return v["voice_id"], v.get("name", "unknown")
        for v in voices:
            if low in v.get("name", "").lower():
                return v["voice_id"], v.get("name", "unknown")
        names = ", ".join(v.get("name", "") for v in voices[:10])
        raise ValueError(f"Voice name '{voice_name}' not found. Available examples: {names}")

    chosen = voices[0]
    return chosen["voice_id"], chosen.get("name", "unknown")


def synthesize_scene_audio(
    api_key: str,
    voice_id: str,
    text: str,
    out_path: Path,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True,
        },
    }
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": "mp3_44100_128"},
        headers={
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
        data=json.dumps(payload),
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs TTS failed for {out_path.name}: "
            f"{r.status_code} {r.text[:400]}"
        )
    out_path.write_bytes(r.content)


def synthesize_scene_audio_macos_say(
    text: str,
    out_path: Path,
    voice: str,
    rate: int,
) -> None:
    if shutil.which("say") is None:
        raise RuntimeError("macOS `say` command is not available on this machine.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        txt = td_path / "input.txt"
        aiff = td_path / "speech.aiff"
        txt.write_text(text, encoding="utf-8")

        run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                "-f",
                str(txt),
                "-o",
                str(aiff),
            ]
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(out_path),
            ]
        )


def try_vapi_tts_capability(vapi_private_key: str) -> None:
    """
    Validate whether Vapi currently exposes a direct TTS endpoint we can use.
    Raises RuntimeError if not available.
    """
    r = requests.get("https://api.vapi.ai/api-json", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Unable to read Vapi API schema: {r.status_code}")
    paths = r.json().get("paths", {})
    direct_tts_paths = [p for p in paths if any(k in p.lower() for k in ["/speech", "/tts", "/speak"])]
    if not direct_tts_paths:
        raise RuntimeError(
            "Vapi API schema does not expose a direct TTS endpoint for offline speech generation."
        )


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def probe_duration(path: Path) -> float:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(proc.stdout.strip())


def ensure_full_video(video_dir: Path, scenes: list[str]) -> Path:
    full = video_dir / "NeuralNetworksAreFractals_FULL.mp4"
    if full.exists():
        return full

    concat_list = video_dir / "concat_list_voiceover.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for scene in scenes:
            scene_path = video_dir / f"{scene}.mp4"
            if not scene_path.exists():
                raise FileNotFoundError(f"Missing scene video: {scene_path}")
            f.write(f"file '{scene_path.name}'\n")

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(full),
        ],
        cwd=video_dir,
    )
    return full


def concat_scene_audio(scene_audio_paths: list[Path], out_path: Path) -> None:
    concat_list = out_path.parent / "voice_concat_list.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for p in scene_audio_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing scene audio: {p}")
            f.write(f"file '{p.resolve()}'\n")

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(out_path),
        ]
    )


def mux_voiceover(video_path: Path, narration_mp3: Path, out_path: Path) -> None:
    video_dur = probe_duration(video_path)
    audio_dur = probe_duration(narration_mp3)

    working_video = video_path
    padded_video = out_path.parent / "tmp_video_padded.mp4"

    if audio_dur > video_dur:
        extra = audio_dur - video_dur + 0.05
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={extra:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(padded_video),
            ]
        )
        working_video = padded_video
        video_dur = probe_duration(working_video)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(working_video),
        "-i",
        str(narration_mp3),
        "-map",
        "0:v:0",
        "-filter_complex",
        f"[1:a]apad=pad_dur={max(video_dur - audio_dur, 0):.3f}[a]",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_dur:.3f}",
        str(out_path),
    ]
    run(cmd)

    if padded_video.exists():
        padded_video.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "elevenlabs", "vapi", "macos_say"],
        help="Audio generation provider.",
    )
    p.add_argument("--quality", default="480p15", help="manim quality folder under media/videos/scenes/")
    p.add_argument("--voice-id", default=None, help="Explicit ElevenLabs voice ID")
    p.add_argument("--voice-name", default=None, help="Voice name to match in your account")
    p.add_argument("--model-id", default="eleven_multilingual_v2")
    p.add_argument("--stability", type=float, default=0.62)
    p.add_argument("--similarity-boost", type=float, default=0.78)
    p.add_argument("--style", type=float, default=0.18)
    p.add_argument("--say-voice", default="Samantha", help="Voice for macOS `say` fallback.")
    p.add_argument("--say-rate", type=int, default=185, help="Words-per-minute rate for macOS `say` fallback.")
    p.add_argument("--force-tts", action="store_true", help="Regenerate scene mp3 even if present")
    p.add_argument(
        "--output",
        default=None,
        help="Final output path (default: media/videos/scenes/<quality>/NeuralNetworksAreFractals_FULL_voiceover.mp4)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found in PATH. Install ffmpeg first.")

    scene_text = parse_voice_script(VOICE_SCRIPT_PATH)
    chosen_provider = args.provider
    eleven_key = get_env("ELEVENLABS_API_KEY")
    eleven_voice_id = args.voice_id or get_env("ELEVENLABS_VOICE_ID")
    vapi_private_key = get_env("VAPI_PRIVATE_API_KEY")

    if chosen_provider == "auto":
        if eleven_key:
            chosen_provider = "elevenlabs"
        elif vapi_private_key:
            chosen_provider = "vapi"
        else:
            chosen_provider = "macos_say"

    provider_chain = [chosen_provider]
    if chosen_provider == "auto":
        provider_chain = ["elevenlabs", "vapi", "macos_say"]
    elif chosen_provider == "elevenlabs":
        provider_chain = ["elevenlabs", "vapi", "macos_say"]
    elif chosen_provider == "vapi":
        provider_chain = ["vapi", "macos_say"]
    elif chosen_provider == "macos_say":
        provider_chain = ["macos_say"]

    active_provider = None
    scene_audio_paths: list[Path] = []
    last_error: Exception | None = None

    for provider in provider_chain:
        try:
            scene_audio_dir = SCENE_AUDIO_DIR / provider
            scene_audio_dir.mkdir(parents=True, exist_ok=True)
            scene_audio_paths = []

            if provider == "elevenlabs":
                if not eleven_key:
                    raise RuntimeError("ELEVENLABS_API_KEY is missing.")
                voice_id, voice_name = choose_voice_id(
                    api_key=eleven_key,
                    voice_id=eleven_voice_id,
                    voice_name=args.voice_name,
                )
                print(f"Using ElevenLabs voice: {voice_name} ({voice_id})")

                for idx, scene in enumerate(SCENES, start=1):
                    out_mp3 = scene_audio_dir / f"{idx:02d}_{scene}.mp3"
                    scene_audio_paths.append(out_mp3)
                    if out_mp3.exists() and not args.force_tts:
                        print(f"[cached] {scene} -> {out_mp3.name}")
                        continue
                    print(f"[tts:{provider}] {scene}")
                    synthesize_scene_audio(
                        api_key=eleven_key,
                        voice_id=voice_id,
                        text=scene_text[scene],
                        out_path=out_mp3,
                        model_id=args.model_id,
                        stability=args.stability,
                        similarity_boost=args.similarity_boost,
                        style=args.style,
                    )

            elif provider == "vapi":
                if not vapi_private_key:
                    raise RuntimeError("VAPI_PRIVATE_API_KEY is missing.")
                try_vapi_tts_capability(vapi_private_key)
                raise RuntimeError("Vapi TTS path not implemented in this pipeline yet.")

            elif provider == "macos_say":
                for idx, scene in enumerate(SCENES, start=1):
                    out_mp3 = scene_audio_dir / f"{idx:02d}_{scene}.mp3"
                    scene_audio_paths.append(out_mp3)
                    if out_mp3.exists() and not args.force_tts:
                        print(f"[cached] {scene} -> {out_mp3.name}")
                        continue
                    print(f"[tts:{provider}] {scene}")
                    synthesize_scene_audio_macos_say(
                        text=scene_text[scene],
                        out_path=out_mp3,
                        voice=args.say_voice,
                        rate=args.say_rate,
                    )
            else:
                raise RuntimeError(f"Unknown provider: {provider}")

            active_provider = provider
            break
        except Exception as e:
            print(f"Provider '{provider}' failed: {e}")
            last_error = e
            continue

    if active_provider is None:
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    narration_full = AUDIO_DIR / f"NeuralNetworksAreFractals_voiceover_full_{active_provider}.mp3"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Concatenating scene audio ({active_provider})...")
    concat_scene_audio(scene_audio_paths, narration_full)

    video_dir = ROOT / "media" / "videos" / "scenes" / args.quality
    video_dir.mkdir(parents=True, exist_ok=True)
    full_video = ensure_full_video(video_dir, SCENES)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else video_dir / f"NeuralNetworksAreFractals_FULL_voiceover_{active_provider}.mp4"
    )
    print(f"Muxing voiceover into: {output_path}")
    mux_voiceover(full_video, narration_full, output_path)

    print("Done.")
    print(f"Narration audio: {narration_full}")
    print(f"Video with voiceover: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
