"""Decode any audio-bearing file to mono float32 PCM via ffmpeg."""
from __future__ import annotations

import subprocess
import numpy as np
from pathlib import Path

from ..core.ffmpeg import FFMPEG
from ..core.errors import AnalysisError


def decode_pcm(path: Path | str, sr: int = 22050) -> tuple[np.ndarray, int]:
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0 or not proc.stdout:
        raise AnalysisError(f"audio decode failed for {path}: "
                            f"{proc.stderr.decode('utf8','ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr
