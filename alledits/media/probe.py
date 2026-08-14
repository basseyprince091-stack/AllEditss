"""Container/stream probing -> normalized MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from fractions import Fraction
from pathlib import Path

from ..core.ffmpeg import ffprobe_json
from ..core.errors import UnsupportedMediaError


def _fps(s: str | None) -> float:
    if not s or s in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(s))
    except Exception:
        return 0.0


@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    pix_fmt: str = ""
    bitrate: int = 0
    sample_rate: int = 0
    channels: int = 0
    rotation: int = 0
    nb_frames: int = 0
    container: str = ""

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    @property
    def aspect_label(self) -> str:
        a = self.aspect
        if a == 0:
            return "unknown"
        for label, val in (("9:16", 9 / 16), ("1:1", 1.0), ("4:5", 0.8),
                           ("16:9", 16 / 9), ("4:3", 4 / 3), ("2.39:1", 2.39)):
            if abs(a - val) < 0.06:
                return label
        return f"{a:.2f}:1"

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    def to_dict(self):
        d = asdict(self)
        d["aspect_label"] = self.aspect_label
        return d


def probe(path: Path | str) -> MediaInfo:
    data = ffprobe_json(path)
    fmt = data.get("format", {})
    info = MediaInfo(path=str(path),
                     duration=float(fmt.get("duration") or 0.0),
                     bitrate=int(fmt.get("bit_rate") or 0),
                     container=fmt.get("format_name", ""))
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not info.has_video:
            info.has_video = True
            info.width = int(s.get("width") or 0)
            info.height = int(s.get("height") or 0)
            info.fps = _fps(s.get("avg_frame_rate")) or _fps(s.get("r_frame_rate"))
            info.codec = s.get("codec_name", "")
            info.pix_fmt = s.get("pix_fmt", "")
            info.nb_frames = int(s.get("nb_frames") or 0)
            if not info.duration:
                info.duration = float(s.get("duration") or 0.0)
            for sd in s.get("side_data_list", []) or []:
                if "rotation" in sd:
                    info.rotation = int(sd["rotation"])
        elif s.get("codec_type") == "audio" and not info.has_audio:
            info.has_audio = True
            info.sample_rate = int(s.get("sample_rate") or 0)
            info.channels = int(s.get("channels") or 0)
    if not info.has_video and not info.has_audio:
        raise UnsupportedMediaError(f"No usable streams in {path}")
    # Portrait footage recorded with rotation metadata: report display dimensions.
    if info.rotation in (90, -90, 270, -270):
        info.width, info.height = info.height, info.width
    return info
