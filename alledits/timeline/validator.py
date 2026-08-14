"""Timeline validator (Principle 15). NOTHING renders without passing this.

Errors block rendering. Warnings are surfaced to the critic and the user.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .schema import Timeline, EffectType, TransitionType


@dataclass
class Issue:
    severity: str      # "error" | "warning"
    code: str
    message: str
    subject: str = ""

    def __str__(self):
        return f"{self.severity.upper()}[{self.code}] {self.subject}: {self.message}"


# Effect parameter contracts: name -> (min, max)
PARAM_RANGES = {
    EffectType.PUSH_ZOOM.value:        {"from": (0.5, 3.0), "to": (0.5, 3.0)},
    EffectType.PULL_ZOOM.value:        {"from": (0.5, 3.0), "to": (0.5, 3.0)},
    EffectType.DIRECTIONAL_BLUR.value: {"strength": (0.0, 1.0), "angle": (-360.0, 360.0)},
    EffectType.GAUSSIAN_BLUR.value:    {"strength": (0.0, 1.0)},
    EffectType.RADIAL_BLUR.value:      {"strength": (0.0, 1.0)},
    EffectType.SHAKE.value:            {"amplitude": (0.0, 0.12), "frequency": (0.1, 30.0)},
    EffectType.VELOCITY_RAMP.value:    {"from": (0.1, 8.0), "to": (0.1, 8.0)},
    EffectType.FILM_GRAIN.value:       {"strength": (0.0, 1.0)},
    EffectType.VIGNETTE.value:         {"strength": (0.0, 1.0)},
    EffectType.GLOW.value:             {"strength": (0.0, 1.0)},
    EffectType.FLASH.value:            {"strength": (0.0, 1.0)},
    EffectType.COLOR_GRADE.value:      {"contrast": (0.0, 3.0), "saturation": (0.0, 3.0),
                                        "brightness": (-1.0, 1.0), "gamma": (0.1, 3.0),
                                        "temperature": (-1.0, 1.0)},
    EffectType.DRIFT.value:            {"dx": (-0.5, 0.5), "dy": (-0.5, 0.5)},
    EffectType.REFRAME.value:          {"x": (0.0, 1.0), "y": (0.0, 1.0), "scale": (0.2, 3.0)},
    EffectType.DENOISE.value:          {"strength": (0.0, 1.0)},
    EffectType.SHARPEN.value:          {"strength": (0.0, 1.0)},
    EffectType.DEBLOCK.value:          {"strength": (0.0, 1.0)},
    EffectType.STABILIZE.value:        {"strength": (0.0, 1.0), "zoom": (0.0, 0.2)},
    EffectType.EXPAND_CONTRAST.value:  {"strength": (0.0, 1.0)},
}

MAX_EFFECTS_PER_CLIP = 5
MIN_CLIP_DURATION = 0.08
MAX_TRANSITION_RATIO = 0.6     # transition may not exceed 60% of the shorter clip


def validate(tl: Timeline, beat_grid: list | None = None,
             beat_tolerance: float = 0.055) -> list[Issue]:
    issues: list[Issue] = []
    E = lambda c, m, s="": issues.append(Issue("error", c, m, s))
    W = lambda c, m, s="": issues.append(Issue("warning", c, m, s))

    if tl.schema_version.split(".")[0] != "1":
        E("schema", f"unsupported schema major version {tl.schema_version}")

    p = tl.project
    if p.width <= 0 or p.height <= 0:
        E("project", "invalid dimensions")
    if p.width % 2 or p.height % 2:
        E("project", f"dimensions must be even for yuv420p ({p.width}x{p.height})")
    if not (1 <= p.fps <= 240):
        E("project", f"implausible fps {p.fps}")

    if not tl.clips:
        E("empty", "timeline has no clips")
        return issues

    ids = set()
    prev_end = None
    for c in sorted(tl.clips, key=lambda c: c.timeline_start):
        s = c.id
        if c.id in ids:
            E("dup_id", "duplicate clip id", s)
        ids.add(c.id)

        if not c.source_path or not os.path.exists(c.source_path):
            E("missing_source", f"source not found: {c.source_path}", s)
        if c.source_out <= c.source_in:
            E("bad_range", f"source_out {c.source_out} <= source_in {c.source_in}", s)
        if c.source_in < 0:
            E("bad_range", "negative source_in", s)
        if c.timeline_duration < MIN_CLIP_DURATION:
            E("too_short", f"clip duration {c.timeline_duration:.3f}s below "
                           f"{MIN_CLIP_DURATION}s floor", s)
        if not (0.1 <= c.speed <= 8.0):
            E("bad_speed", f"speed {c.speed} out of range", s)

        # Spec §25: USE_BRIEFLY clips must actually be brief.
        if c.quality_handling == "use_briefly" and c.timeline_duration > 0.55:
            W("brief_overrun", f"clip marked use_briefly is {c.timeline_duration:.2f}s; "
                               "intended as a flash/transition beat", s)
        if c.quality_handling == "reject":
            E("rejected_clip", "a clip marked reject is on the timeline", s)

        if len(c.effects) > MAX_EFFECTS_PER_CLIP:
            W("effect_density", f"{len(c.effects)} effects on one clip — likely "
                                "over-processed", s)
        for eff in c.effects:
            etype = eff.type if hasattr(eff, "type") else eff.get("type")
            params = eff.params if hasattr(eff, "params") else eff.get("params", {})
            if etype not in {e.value for e in EffectType}:
                E("unknown_effect", f"unknown effect '{etype}'", s)
                continue
            for k, v in (params or {}).items():
                rng = PARAM_RANGES.get(etype, {}).get(k)
                if rng and not (rng[0] <= float(v) <= rng[1]):
                    E("param_range", f"{etype}.{k}={v} outside {rng}", s)

        t = c.transition_in
        if t is not None:
            ttype = t.type if hasattr(t, "type") else t.get("type")
            tdur = float(t.duration if hasattr(t, "duration") else t.get("duration", 0))
            if ttype not in {x.value for x in TransitionType}:
                E("unknown_transition", f"unknown transition '{ttype}'", s)
            if tdur < 0:
                E("bad_transition", "negative transition duration", s)
            if tdur > c.timeline_duration * MAX_TRANSITION_RATIO:
                E("transition_too_long",
                  f"transition {tdur:.2f}s exceeds {MAX_TRANSITION_RATIO:.0%} of "
                  f"clip ({c.timeline_duration:.2f}s)", s)

        if prev_end is not None:
            gap = c.timeline_start - prev_end
            if gap > 0.001:
                E("gap", f"{gap:.3f}s gap before this clip", s)
            elif gap < -0.001:
                # negative overlap is only legal where a transition consumes it
                tdur = float(getattr(c.transition_in, "duration", 0) or 0)
                if abs(gap) - tdur > 0.02:
                    E("overlap", f"{-gap:.3f}s unexplained overlap", s)
        prev_end = c.timeline_end

    for a in tl.audio:
        if not os.path.exists(a.source_path):
            E("missing_audio", f"audio source not found: {a.source_path}", a.id)
        if a.source_out <= a.source_in:
            E("bad_audio_range", "audio out <= in", a.id)

    for tx in tl.text:
        if tx.end <= tx.start:
            E("bad_text_range", "text end <= start", tx.id)
        if tx.end > tl.duration + 0.01:
            W("text_overrun", "text extends past the end of the video", tx.id)

    # beat alignment is a warning, never a hard error
    if beat_grid:
        import numpy as np
        grid = np.asarray(beat_grid)
        misses = 0
        for c in tl.clips:
            if not c.beat_locked:
                continue
            err = float(np.min(np.abs(grid - c.timeline_start)))
            if err > beat_tolerance:
                misses += 1
                W("beat_drift", f"cut is {err*1000:.0f}ms off the beat grid", c.id)
        if misses:
            W("beat_drift_total", f"{misses} beat-locked cuts drifted off grid")
    return issues


def errors(issues) -> list:
    return [i for i in issues if i.severity == "error"]


def warnings(issues) -> list:
    return [i for i in issues if i.severity == "warning"]
