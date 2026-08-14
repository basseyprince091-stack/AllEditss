"""Thin, safe ffmpeg/ffprobe wrapper.

An LLM never emits a raw shell command here (Spec §11). Callers pass argument
lists that are built by typed code; this module only runs them and captures output.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .errors import ProbeError, RenderError

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def run(args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return proc


def ffmpeg(args: list[str], timeout: int = 1800) -> str:
    """Run ffmpeg with -y -hide_banner. Raises RenderError on non-zero exit."""
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = run(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[:2000]}"
                          f"\n  cmd: {' '.join(cmd[:40])}")
    return proc.stderr


def ffprobe_json(path: Path | str) -> dict:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = run(cmd, timeout=120)
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {proc.stderr.strip()[:500]}")
    return json.loads(proc.stdout)


def has_encoder(name: str) -> bool:
    proc = run([FFMPEG, "-hide_banner", "-encoders"], timeout=60)
    return name in proc.stdout


def has_filter(name: str) -> bool:
    proc = run([FFMPEG, "-hide_banner", "-filters"], timeout=60)
    return f" {name} " in proc.stdout
