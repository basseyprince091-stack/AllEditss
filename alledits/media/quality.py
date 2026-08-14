"""Technical quality analysis + the two-score model (Spec §13, §25).

CRITICAL SPEC RULE: technical quality and creative value are SEPARATE scores.
A low-quality clip may be the perfect creative shot ("nothing is wasted"), and
already-good footage must not be blindly processed and degraded.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum

from .frames import FrameReader


class Handling(str, Enum):
    USE = "use"                    # good enough as-is; DO NOT process
    ENHANCE = "enhance"            # justified enhancement
    USE_BRIEFLY = "use_briefly"    # flash frame / transition / montage only
    REPLACE = "replace"            # find something better if possible
    REJECT = "reject"              # genuinely unusable


class Defect(str, Enum):
    NOISE = "noise"
    SOFTNESS = "softness"
    SHAKE = "shake"
    BLOCKINESS = "blockiness"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    LOW_CONTRAST = "low_contrast"


@dataclass
class QualityAnalysis:
    sharpness: float = 0.0          # 0..1 (variance of Laplacian, normalized)
    noise: float = 0.0              # 0..1 estimated
    blockiness: float = 0.0         # 0..1 compression artifacts
    exposure_health: float = 1.0    # 0..1 (penalizes clipping)
    dynamic_range: float = 0.0      # 0..1
    resolution_score: float = 0.0
    fps_score: float = 0.0
    bitrate_score: float = 0.0
    technical_quality: float = 0.0  # 0..1 composite
    creative_value: float = 0.0     # 0..1 composite, INDEPENDENT of the above
    handling: str = Handling.USE.value
    max_useful_duration: float | None = None   # for USE_BRIEFLY
    reasons: list = field(default_factory=list)
    defects: list = field(default_factory=list)
    salvage: list = field(default_factory=list)
    #   Creative functions this clip can still serve despite its score, each
    #   with a duration cap (Spec 25, "nothing is wasted").
    #   [{defect, severity 0..1, treatment}] — the SPECIFIC repairs justified.
    #   Enhancement is per-defect, never a blanket "make it better" pass, because
    #   applying a treatment a clip doesn't need degrades it (Spec §13).

    def to_dict(self):
        return asdict(self)


# Flat-region percentile for noise estimation. Measured across the corpus, a
# 55th-percentile mask let ordinary detail leak in, so clean footage scored
# 0.08-0.14 and overlapped with genuinely noisy material. At the 25th percentile
# every clean clip measures 0.000 while alls=20 noise reads 0.190 and alls=42
# reads 0.333 — a clean separation instead of an overlapping one.
NOISE_FLAT_PCT = 25


def _estimate_noise(gray: np.ndarray) -> float:
    """Noise level: residual after median filtering, measured ONLY in the
    flattest regions of the frame so that image detail is not counted as noise.

    Known limitation: this cannot distinguish sensor noise from extremely
    high-frequency CONTENT (e.g. nearest-neighbour-upscaled pixel art), which is
    uncorrelated at pixel scale by construction. Real camera footage does not
    have this property; synthetic material can. Denoise strength is capped so a
    false positive degrades detail modestly rather than destroying it.
    """
    med = cv2.medianBlur(gray, 3)
    resid = gray.astype(np.float32) - med.astype(np.float32)
    edges = cv2.Laplacian(med, cv2.CV_32F)
    flat = np.abs(edges) < np.percentile(np.abs(edges), NOISE_FLAT_PCT)
    if flat.sum() < 100:
        return 0.0
    return float(min(1.0, resid[flat].std() / 12.0))


def _estimate_blockiness(gray: np.ndarray) -> float:
    """Energy at 8-px block boundaries vs. off-boundary — classic DCT artifact tell."""
    g = gray.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    if dh.shape[1] < 16:
        return 0.0
    cols = np.arange(dh.shape[1])
    on = dh[:, (cols % 8) == 7].mean()
    off = dh[:, (cols % 8) != 7].mean() or 1e-6
    return float(np.clip((on / off - 1.0) / 0.6, 0, 1))


def analyze_quality(path, start: float, end: float, info, visual=None,
                    samples: int = 5) -> QualityAnalysis:
    q = QualityAnalysis()
    frames = []
    with FrameReader(path, analysis_fps=4.0, width=480) as fr:
        for _, f in fr.iter_frames(start=start, end=end):
            frames.append(f)
            if len(frames) >= samples:
                break
    if not frames:
        q.handling = Handling.REJECT.value
        q.reasons.append("no readable frames")
        return q

    sharps, noises, blocks, exps, drs = [], [], [], [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharps.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        noises.append(_estimate_noise(gray))
        blocks.append(_estimate_blockiness(gray))
        clip_hi = float((gray > 252).mean())
        clip_lo = float((gray < 3).mean())
        exps.append(1.0 - min(1.0, (clip_hi + clip_lo) * 4.0))
        lo, hi = np.percentile(gray, [2, 98])
        drs.append(float((hi - lo) / 255.0))

    q.sharpness = float(min(1.0, np.mean(sharps) / 450.0))
    q.noise = float(np.mean(noises))
    q.blockiness = float(np.mean(blocks))
    q.exposure_health = float(np.mean(exps))
    q.dynamic_range = float(np.mean(drs))

    px = info.width * info.height
    q.resolution_score = float(np.clip(np.log2(max(px, 1) / (640 * 360)) / 4.0, 0, 1))
    q.fps_score = float(np.clip((info.fps - 20.0) / 40.0, 0, 1))
    bpp = (info.bitrate / max(px * max(info.fps, 1), 1)) if info.bitrate else 0.0
    q.bitrate_score = float(np.clip(bpp / 0.12, 0, 1)) if bpp else 0.5

    q.technical_quality = float(np.clip(
        0.26 * q.sharpness + 0.20 * q.resolution_score + 0.12 * q.fps_score +
        0.12 * q.bitrate_score + 0.14 * q.exposure_health + 0.08 * q.dynamic_range +
        0.08 * (1.0 - q.blockiness) - 0.10 * q.noise, 0, 1))

    # ---- creative value: deliberately NOT a function of technical quality ----
    ve = getattr(visual, "visual_energy", 0.4) if visual else 0.4
    faces = getattr(visual, "faces", 0) if visual else 0
    thirds = getattr(visual, "thirds_alignment", 0.3) if visual else 0.3
    subj = getattr(visual, "subject_motion", 0.0) if visual else 0.0
    consistency = getattr(visual, "motion_consistency", 0.0) if visual else 0.0
    q.creative_value = float(np.clip(
        0.34 * ve + 0.18 * min(1.0, subj / 4.0) + 0.16 * thirds +
        0.16 * (1.0 if faces else 0.35) + 0.16 * consistency, 0, 1))

    # ---- handling decision ----
    tq, cv_ = q.technical_quality, q.creative_value
    if tq < 0.10 and cv_ < 0.30:
        q.handling = Handling.REJECT.value
        q.reasons.append("very low technical quality with no creative compensation")
    elif tq < 0.28 and cv_ >= 0.55:
        q.handling = Handling.USE_BRIEFLY.value
        q.max_useful_duration = 0.30
        q.reasons.append("weak technically but creatively valuable — usable as a "
                         "flash frame, transition or fast-montage beat")
    elif tq < 0.30:
        q.handling = Handling.REPLACE.value
        q.reasons.append("low technical quality and limited creative value")
    elif tq < 0.55:
        q.handling = Handling.ENHANCE.value
        if q.noise > 0.35:
            q.reasons.append("denoise justified")
        if q.sharpness < 0.30:
            q.reasons.append("detail enhancement justified")
        if q.exposure_health < 0.75:
            q.reasons.append("exposure correction justified")
        if not q.reasons:
            q.reasons.append("mild correction justified")
    else:
        q.handling = Handling.USE.value
        q.reasons.append("already good — no enhancement applied "
                         "(processing would risk degrading it)")

    q.defects = _detect_defects(q, visual)

    # "Nothing is wasted": a clip that fails as a hero shot is asked what else
    # it could do, BEFORE the handling verdict is allowed to exclude it.
    from .salvage import assess_salvage, salvage_cap, is_genuinely_unusable
    q.salvage = [s.to_dict() for s in assess_salvage(q, visual)]
    if q.salvage:
        cap = salvage_cap(q.salvage)
        if q.handling in (Handling.REPLACE.value, Handling.REJECT.value):
            q.handling = Handling.USE_BRIEFLY.value
            q.reasons.append(
                "not good enough to hold, but usable briefly as: "
                + ", ".join(s["role"] for s in q.salvage))
        q.max_useful_duration = (min(q.max_useful_duration, cap)
                                 if q.max_useful_duration else cap)
    elif q.handling == Handling.REPLACE.value:
        unusable, why = is_genuinely_unusable(q, visual)
        if unusable:
            q.handling = Handling.REJECT.value
            q.reasons.append(f"rejected: {why}")
    # A clip that is otherwise fine but carries a treatable defect is an
    # ENHANCE, not a USE — otherwise the prescription is computed and ignored.
    if q.handling == Handling.USE.value and q.defects:
        q.handling = Handling.ENHANCE.value
        q.reasons = [f"otherwise good, but treatable: "
                     + ", ".join(d["defect"] for d in q.defects)]
    return q


# Thresholds are the point at which a treatment does more good than harm.
# Below these, the correct action is to leave the footage alone.
DEFECT_THRESHOLDS = {
    Defect.NOISE:         0.12,   # calibrated: clean corpus measures 0.000
    Defect.SOFTNESS:      0.30,   # sharpness BELOW this
    Defect.SHAKE:         1.60,   # camera jitter above this
    Defect.BLOCKINESS:    0.35,
    Defect.UNDEREXPOSED:  0.030,  # fraction of crushed shadows
    Defect.OVEREXPOSED:   0.030,  # fraction of blown highlights
    Defect.LOW_CONTRAST:  0.30,   # dynamic range below this
}


def _severity(value, threshold, ceiling):
    """Map a measured defect onto 0..1 across the range where treatment helps.

    A defect just past its threshold still warrants real treatment — it is past
    the point where doing nothing is the better choice — so severity starts at a
    meaningful floor rather than at zero.
    """
    if ceiling <= threshold:
        return 1.0
    frac = (value - threshold) / (ceiling - threshold)
    return float(min(1.0, max(0.0, 0.35 + 0.65 * frac)))


def _severity_inverse(value, threshold, floor=0.0):
    """As above, for metrics where LOWER is worse (sharpness, dynamic range)."""
    if threshold <= floor:
        return 1.0
    frac = (threshold - value) / (threshold - floor)
    return float(min(1.0, max(0.0, 0.35 + 0.65 * frac)))


def _detect_defects(q: QualityAnalysis, visual) -> list:
    """List the specific, measurable defects worth treating.

    Detection is INDEPENDENT of the aggregate handling verdict. Gating it on
    handling meant a clip that scored well overall but had one strong fixable
    defect — heavy grain, or genuine handheld wobble — received no treatment at
    all, because its average was fine. Defects are judged on their own merits;
    the thresholds are what protect good footage, and clean footage still returns
    an empty list.
    """
    if q.handling == Handling.REJECT.value:
        return []

    out = []

    def add(defect, severity, treatment, detail):
        out.append({"defect": defect.value,
                    "severity": float(min(max(severity, 0.0), 1.0)),
                    "treatment": treatment, "detail": detail})

    if q.noise > DEFECT_THRESHOLDS[Defect.NOISE]:
        # Severity tracks ABSOLUTE defect magnitude, not merely the excess over
        # the threshold. Scaling by excess alone meant a genuinely noisy clip
        # sitting just past the threshold received an almost inert treatment,
        # because its severity rounded to ~0.02.
        sev = _severity(q.noise, DEFECT_THRESHOLDS[Defect.NOISE], ceiling=0.35)
        add(Defect.NOISE, sev, "denoise",
            f"noise {q.noise:.2f} above the {DEFECT_THRESHOLDS[Defect.NOISE]:.2f} "
            "threshold where denoising gains more detail than it costs")

    if q.sharpness < DEFECT_THRESHOLDS[Defect.SOFTNESS]:
        sev = _severity_inverse(q.sharpness, DEFECT_THRESHOLDS[Defect.SOFTNESS], floor=0.0)
        # Sharpening noise amplifies it; only justified once noise is treatable.
        add(Defect.SOFTNESS, sev, "sharpen",
            f"sharpness {q.sharpness:.2f} below {DEFECT_THRESHOLDS[Defect.SOFTNESS]:.2f}")

    if q.blockiness > DEFECT_THRESHOLDS[Defect.BLOCKINESS]:
        sev = _severity(q.blockiness, DEFECT_THRESHOLDS[Defect.BLOCKINESS], ceiling=1.0)
        add(Defect.BLOCKINESS, sev, "deblock",
            f"compression blocking {q.blockiness:.2f}")

    shake = getattr(visual, "shake", 0.0) if visual else 0.0
    consistency = getattr(visual, "motion_consistency", 1.0) if visual else 1.0
    if shake > DEFECT_THRESHOLDS[Defect.SHAKE] and consistency < 0.55:
        # Only stabilize genuine handheld wobble. A deliberate fast pan has high
        # jitter too, and stabilizing it would destroy the intended movement.
        add(Defect.SHAKE, _severity(shake, DEFECT_THRESHOLDS[Defect.SHAKE], ceiling=5.0),
            "stabilize",
            f"camera jitter {shake:.2f} with low directional consistency "
            f"{consistency:.2f} — handheld wobble, not an intended move")

    if q.dynamic_range < DEFECT_THRESHOLDS[Defect.LOW_CONTRAST]:
        add(Defect.LOW_CONTRAST,
            _severity_inverse(q.dynamic_range, DEFECT_THRESHOLDS[Defect.LOW_CONTRAST]),
            "expand_contrast", f"dynamic range {q.dynamic_range:.2f} is flat")

    return out
