"""MASTER: delivery profiles (Spec §12, §15).

A finished edit is not a deliverable. Each destination has its own container,
codec, resolution, frame rate, colour handling and loudness expectations, and
missing any of them means the platform re-encodes the file — undoing work that
was measured and verified upstream.

Two rules this module exists to enforce:

1. **A profile is a contract, not a suggestion.** Every field is checked against
   the encoded file afterwards (see `qc.py`). A mastering step that claims
   conformance without measuring it is the same failure as a mix that claims a
   loudness target it never hit.

2. **Never silently upscale.** If a profile asks for more pixels than the source
   has, the extra pixels are invented. That may be acceptable, but it must be
   declared (`resolution_provenance`), never presented as native capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class DeliveryProfile:
    name: str
    width: int
    height: int
    fps: float
    container: str = "mp4"
    video_codec: str = "libx264"
    video_codec_name: str = "h264"      # as ffprobe reports it
    pix_fmt: str = "yuv420p"
    crf: int | None = 18
    video_bitrate: str | None = None    # set for platforms that want CBR-ish
    max_bitrate: str | None = None
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    sample_rate: int = 48000
    channels: int = 2
    loudness_target: str = "social"     # key into audio.mix.LOUDNESS_TARGETS
    true_peak_db: float = -1.0
    faststart: bool = True
    max_duration: float | None = None   # platform hard limit, seconds
    notes: str = ""

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def to_dict(self):
        return asdict(self)


# Values reflect each platform's published delivery guidance at the time of
# writing. They are data, not logic — update them here rather than in the encoder.
PROFILES: dict = {
    "youtube_shorts": DeliveryProfile(
        name="youtube_shorts", width=1080, height=1920, fps=30,
        crf=18, max_duration=180.0, loudness_target="social",
        notes="vertical 9:16; YouTube normalises to about -14 LUFS"),
    "tiktok": DeliveryProfile(
        name="tiktok", width=1080, height=1920, fps=30,
        crf=20, video_bitrate="6M", max_duration=600.0,
        loudness_target="social",
        notes="vertical 9:16; heavier platform re-encode, so deliver clean"),
    "instagram_reels": DeliveryProfile(
        name="instagram_reels", width=1080, height=1920, fps=30,
        crf=20, max_duration=90.0, loudness_target="social",
        notes="vertical 9:16"),
    "youtube_1080p": DeliveryProfile(
        name="youtube_1080p", width=1920, height=1080, fps=30,
        crf=18, loudness_target="social",
        notes="landscape 16:9"),
    "broadcast_ebu": DeliveryProfile(
        name="broadcast_ebu", width=1920, height=1080, fps=25,
        crf=16, audio_bitrate="256k", loudness_target="broadcast",
        true_peak_db=-1.0,
        notes="EBU R128: -23 LUFS integrated, -1 dBTP ceiling"),
    "web_preview": DeliveryProfile(
        name="web_preview", width=720, height=1280, fps=30,
        crf=26, audio_bitrate="128k", loudness_target="social",
        notes="small, fast review copy — not a deliverable"),
}


def get_profile(name: str) -> DeliveryProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown delivery profile {name!r}; "
                       f"available: {', '.join(sorted(PROFILES))}")
    return PROFILES[name]


@dataclass
class ScalingDecision:
    """Whether reaching the profile requires inventing pixels, and by how much."""
    upscaling: bool
    factor: float
    provenance: str          # native | upscaled
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def plan_scaling(src_w: int, src_h: int, profile: DeliveryProfile) -> ScalingDecision:
    """Decide honestly whether the master invents pixels.

    Must mirror what the encoder actually does. The master fits the picture
    INSIDE the target frame and pads (never stretches), so the content is scaled
    by min(width_ratio, height_ratio) — not max. Using max reported a 1080x1920
    edit delivered to 1920x1080 as a 1.78x upscale when the picture is in fact
    downscaled and letterboxed: a false disclosure, which erodes trust in the
    real ones just as much as a missing disclosure does.
    """
    if src_w <= 0 or src_h <= 0:
        return ScalingDecision(False, 1.0, "native", "source dimensions unknown")
    factor = min(profile.width / src_w, profile.height / src_h)
    if factor > 1.01:
        return ScalingDecision(
            True, factor, "upscaled",
            f"source {src_w}x{src_h} is smaller than the {profile.width}x"
            f"{profile.height} target; the picture is enlarged {factor:.2f}x, so "
            "that much of the delivered detail is interpolated, not captured")
    return ScalingDecision(
        False, factor, "native",
        f"source {src_w}x{src_h} fits the {profile.width}x{profile.height} "
        f"target at {factor:.2f}x (no pixels invented)")
