"""QC: verifying a master against its delivery profile.

The point of this module is that nothing here trusts the encoder. Every check
re-measures the finished file — resolution, frame rate, codec, pixel format,
duration, loudness, true peak, faststart — and compares it to the contract.

A check has three outcomes and the distinction matters:
  PASS  — measured and conformant
  FAIL  — measured and NOT conformant; the file should not ship
  SKIP  — could not be measured, and says so rather than counting as a pass

The last one is the honest part. A QC report where an unmeasurable check quietly
becomes a tick is worse than no report, because it converts ignorance into
false assurance.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..core.ffmpeg import FFMPEG, FFPROBE
from ..media.probe import probe


PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclass
class Check:
    name: str
    status: str
    expected: str = ""
    measured: str = ""
    detail: str = ""

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[self.status]
        core = f"  {mark}  {self.name:22}"
        if self.expected or self.measured:
            core += f" expected {self.expected:<18} measured {self.measured}"
        return core + (f"   [{self.detail}]" if self.detail else "")


@dataclass
class QCReport:
    path: str
    profile: str
    checks: list = field(default_factory=list)
    resolution_provenance: str = "native"

    def add(self, *args, **kw):
        self.checks.append(Check(*args, **kw))

    @property
    def failed(self) -> list:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def skipped(self) -> list:
        return [c for c in self.checks if c.status == SKIP]

    @property
    def passed(self) -> bool:
        """Conformant only if nothing failed. Skips do NOT count as passes, but
        they do not block either — they are reported so a human can decide."""
        return not self.failed

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c.status == PASS)
        return (f"{n_pass} passed, {len(self.failed)} failed, "
                f"{len(self.skipped)} not measurable")

    def to_dict(self):
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        d["conformant"] = self.passed
        return d


def _measure_loudness(path):
    """Integrated loudness and true peak of the finished file."""
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
                            "-af", "ebur128=peak=true", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
        tail = r.stderr[r.stderr.rfind("Integrated loudness"):]
        I = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
        TP = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
        return (float(I.group(1)) if I else None,
                float(TP.group(1)) if TP else None)
    except Exception:
        return None, None


def _has_faststart(path) -> bool | None:
    """True when moov precedes mdat, so playback can begin before full download."""
    try:
        r = subprocess.run([FFPROBE, "-v", "trace", "-i", str(path)],
                           capture_output=True, text=True, timeout=120)
        text = r.stderr
        moov, mdat = text.find("'moov'"), text.find("'mdat'")
        if moov < 0 or mdat < 0:
            return None
        return moov < mdat
    except Exception:
        return None


def run_qc(path, profile, expect_loudness: bool = True,
           scaling=None) -> QCReport:
    """Measure a finished file against its delivery profile."""
    from ..audio.mix import LOUDNESS_TARGETS

    path = Path(path)
    rep = QCReport(path=str(path), profile=profile.name)
    if scaling is not None:
        rep.resolution_provenance = scaling.provenance

    if not path.exists():
        rep.add("file exists", FAIL, "present", "missing")
        return rep

    info = probe(path)

    # --- picture ---
    rep.add("resolution", PASS if (info.width, info.height) ==
            (profile.width, profile.height) else FAIL,
            f"{profile.width}x{profile.height}", f"{info.width}x{info.height}")

    fps_ok = abs(info.fps - profile.fps) <= 0.05
    rep.add("frame rate", PASS if fps_ok else FAIL,
            f"{profile.fps:g}", f"{info.fps:.3f}")

    rep.add("video codec", PASS if info.codec == profile.video_codec_name else FAIL,
            profile.video_codec_name, info.codec or "(none)")

    rep.add("pixel format", PASS if info.pix_fmt == profile.pix_fmt else FAIL,
            profile.pix_fmt, info.pix_fmt or "(none)")

    # --- audio presence ---
    rep.add("audio stream", PASS if info.has_audio else FAIL, "present",
            "present" if info.has_audio else "missing")
    if info.has_audio:
        rep.add("sample rate", PASS if info.sample_rate == profile.sample_rate
                else FAIL, str(profile.sample_rate), str(info.sample_rate))
        rep.add("channels", PASS if info.channels == profile.channels else FAIL,
                str(profile.channels), str(info.channels))

    # --- duration limit ---
    if profile.max_duration:
        ok = info.duration <= profile.max_duration + 0.05
        rep.add("duration limit", PASS if ok else FAIL,
                f"<= {profile.max_duration:g}s", f"{info.duration:.2f}s")

    # --- loudness (the reason the sound stage exists) ---
    if expect_loudness and info.has_audio:
        target = LOUDNESS_TARGETS.get(profile.loudness_target)
        I, TP = _measure_loudness(path)
        if I is None:
            rep.add("loudness", SKIP, f"{target:g} LUFS", "not measurable",
                    "ebur128 produced no reading")
        else:
            rep.add("loudness", PASS if abs(I - target) <= 1.0 else FAIL,
                    f"{target:g} +/-1 LUFS", f"{I:.1f} LUFS")
        if TP is None:
            rep.add("true peak", SKIP, f"<= {profile.true_peak_db:g} dBTP",
                    "not measurable")
        else:
            rep.add("true peak", PASS if TP <= profile.true_peak_db else FAIL,
                    f"<= {profile.true_peak_db:g} dBTP", f"{TP:.1f} dBFS")

    # --- container / streaming ---
    if profile.faststart and profile.container == "mp4":
        fs = _has_faststart(path)
        if fs is None:
            rep.add("faststart", SKIP, "moov before mdat", "not determinable")
        else:
            rep.add("faststart", PASS if fs else FAIL, "moov before mdat",
                    "moov first" if fs else "mdat first")

    # --- provenance, not conformance: never a FAIL, always disclosed ---
    if scaling is not None and scaling.upscaling:
        rep.add("resolution provenance", SKIP, "native capture", "upscaled",
                scaling.reason)

    return rep
