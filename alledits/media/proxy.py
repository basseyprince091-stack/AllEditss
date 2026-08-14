"""Proxy generation. Analysis and preview run on proxies; the original is
untouched and preserved (Principles 11, 12)."""
from __future__ import annotations

from pathlib import Path

from ..core.ffmpeg import ffmpeg
from .probe import MediaInfo

PROXY_HEIGHT = 540
PROXY_CRF = "26"


def needs_proxy(info: MediaInfo) -> bool:
    return info.height > PROXY_HEIGHT * 1.4 or info.bitrate > 20_000_000


def make_proxy(src: Path, dst: Path, info: MediaInfo, height: int = PROXY_HEIGHT) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    ffmpeg(["-i", str(src),
            "-vf", f"scale=-2:{height}:flags=bicubic",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", PROXY_CRF,
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            str(dst)])
    return dst
