"""MASTER: producing and verifying a deliverable.

`master()` transcodes a finished edit to a delivery profile and then measures the
result. The QC report is returned alongside the file, and a non-conformant master
is reported as such rather than handed over quietly.

Audio is re-normalised here rather than copied, because a transcode changes the
codec and bitrate and therefore the reconstructed peaks. Carrying over the
upstream measurement would be reporting a number that no longer describes the
file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.ffmpeg import ffmpeg
from ..media.probe import probe
from .profiles import DeliveryProfile, get_profile, plan_scaling
from .qc import run_qc, QCReport


@dataclass
class MasterResult:
    path: str
    profile: str
    qc: QCReport
    scaling: object = None
    warnings: list = field(default_factory=list)

    @property
    def conformant(self) -> bool:
        return self.qc.passed

    def to_dict(self):
        return {"path": self.path, "profile": self.profile,
                "conformant": self.conformant,
                "scaling": self.scaling.to_dict() if self.scaling else None,
                "qc": self.qc.to_dict(), "warnings": self.warnings}


def _scale_filter(profile: DeliveryProfile) -> str:
    """Fit to the target frame, padding rather than distorting.

    Aspect is never stretched to fill: a 16:9 source delivered to 9:16 is
    letterboxed, because distorting faces to avoid bars is the worse failure.
    """
    return (f"scale={profile.width}:{profile.height}"
            f":force_original_aspect_ratio=decrease,"
            f"pad={profile.width}:{profile.height}:-1:-1:color=black,"
            f"setsar=1,format={profile.pix_fmt}")


def master(src, out_path, profile_name: str = "youtube_shorts",
           loudness_gain_db: float | None = None,
           allow_upscale: bool = True, log=lambda *_: None) -> MasterResult:
    """Transcode `src` to a delivery profile and QC the result."""
    profile = get_profile(profile_name)
    src, out_path = Path(src), Path(out_path)
    if not src.exists():
        raise FileNotFoundError(src)

    info = probe(src)
    scaling = plan_scaling(info.width, info.height, profile)
    warnings = []
    if scaling.upscaling:
        if not allow_upscale:
            raise ValueError(
                f"profile {profile.name} needs {profile.width}x{profile.height} "
                f"but the source is {info.width}x{info.height}; refusing to "
                "invent pixels (pass allow_upscale=True to accept interpolation)")
        warnings.append(scaling.reason)
        log(f"    ! {scaling.reason}")

    vargs = ["-c:v", profile.video_codec, "-pix_fmt", profile.pix_fmt,
             "-r", f"{profile.fps:g}"]
    if profile.video_bitrate:
        vargs += ["-b:v", profile.video_bitrate]
        if profile.max_bitrate:
            vargs += ["-maxrate", profile.max_bitrate,
                      "-bufsize", profile.max_bitrate]
    else:
        vargs += ["-crf", str(profile.crf), "-preset", "medium"]

    aargs = ["-c:a", profile.audio_codec, "-b:a", profile.audio_bitrate,
             "-ar", str(profile.sample_rate), "-ac", str(profile.channels)]

    # Re-establish loudness on the transcoded audio. A new codec at a new bitrate
    # reconstructs peaks differently, so the upstream measurement no longer
    # describes this file.
    from ..audio.mix import LOUDNESS_TARGETS
    from ..render.ffmpeg_renderer import CODEC_OVERSHOOT_MARGIN_DB
    target = LOUDNESS_TARGETS.get(profile.loudness_target, -14.0)
    ceiling = profile.true_peak_db - CODEC_OVERSHOOT_MARGIN_DB
    afilter = (f"loudnorm=I={target:.1f}:TP={ceiling:.1f}:LRA=11,"
               f"alimiter=limit={10 ** (ceiling / 20):.4f}:level=disabled,"
               f"aformat=sample_fmts=fltp:sample_rates={profile.sample_rate}"
               f":channel_layouts=stereo")

    args = ["-i", str(src), "-vf", _scale_filter(profile), "-af", afilter,
            *vargs, *aargs]
    if profile.faststart and profile.container == "mp4":
        args += ["-movflags", "+faststart"]
    args += [str(out_path)]

    log(f"    mastering to {profile.name} "
        f"({profile.width}x{profile.height} @ {profile.fps:g})")
    ffmpeg(args)

    report = run_qc(out_path, profile, scaling=scaling)
    log(f"    QC: {report.summary()}")
    return MasterResult(path=str(out_path), profile=profile.name, qc=report,
                        scaling=scaling, warnings=warnings)
