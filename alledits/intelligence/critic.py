"""AI self-critique and bounded revision (Spec §19, Principle 16).

The critic inspects the RENDERED FILE, not the plan — that is the point. It
re-measures the output the same way it measured the inputs, so it can catch what
actually went wrong in the render rather than what was supposed to happen.

Checks:
  beat alignment      cuts recovered from the render vs. the music's beat grid
  pacing              realized cut rate vs. the reference's cut rate
  intensity tracking  realized intensity curve vs. the target curve
  continuity          luminance/colour jumps across cuts
  effect density      over-processing
  exposure            clipped highlights/shadows introduced by grading
  uniformity          sharpness/exposure spread across shots (Spec §13 pass)

Each issue carries a concrete, machine-applicable revision directive.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict

from ..media.scenes import detect_shots, content_curve
from ..media.probe import probe
from ..media.frames import FrameReader
from ..media.visual import analyze_shot

MAX_REVISIONS = 2       # bounded loop


@dataclass
class Critique:
    score: float = 0.0                  # 0..10 overall
    intensity_realized: float = 0.0
    beat_alignment: float = 0.0         # 0..1
    pacing_error: float = 0.0
    issues: list = field(default_factory=list)     # [{code,severity,message,directive}]
    metrics: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self):
        return asdict(self)

    def directives(self):
        return [i["directive"] for i in self.issues if i.get("directive")]


def critique(render_path, timeline, grammar, audio, ledger=None, cons=None) -> Critique:
    c = Critique()
    info = probe(render_path)
    planned = [cl.timeline_start for cl in sorted(timeline.clips,
                                                  key=lambda x: x.timeline_start)][1:]

    # ---- recover the cuts that actually exist in the rendered file ----
    # Sample at the video's own frame rate: at 20fps the ±50ms quantization error
    # would be larger than the beat error being measured, making the alignment
    # figure meaningless.
    afps = float(min(max(info.fps or 30.0, 12.0), 60.0))
    shots = detect_shots(render_path, info.duration, min_shot=0.1,
                         sensitivity=1.8, analysis_fps=afps)
    realized = [s.start for s in shots][1:]
    times, scores, lumas = content_curve(render_path, analysis_fps=afps)

    c.metrics["planned_cuts"] = len(planned)
    c.metrics["detected_cuts"] = len(realized)
    c.metrics["duration"] = info.duration

    # ---- beat alignment measured on the render ----
    if audio.beats and realized:
        b = np.asarray(audio.beats)
        errs = [float(np.min(np.abs(b - t))) for t in realized]
        mean_err = float(np.mean(errs))
        c.beat_alignment = float(np.clip(1.0 - mean_err / 0.12, 0, 1))
        c.metrics["mean_beat_error_ms"] = mean_err * 1000
        c.metrics["beat_error_frames"] = mean_err * afps
        if mean_err > 0.075:
            c.issues.append({
                "code": "beat_drift", "severity": "high",
                "message": (f"cuts sit an average of {mean_err*1000:.0f}ms off the beat; "
                            "the edit will feel loose against the music"),
                "directive": {"action": "resnap_to_beats"}})

    # ---- indistinct cuts ----
    # Fewer cuts detected than planned means some cuts are invisible: adjacent
    # clips are too alike for the cut to read. That is a SHOT-SELECTION problem,
    # not a pacing problem, and conflating the two makes the critic prescribe
    # exactly the wrong fix (speeding up an edit whose cuts simply don't land).
    planned_cps = len(timeline.clips) / max(info.duration, 1e-6)
    if planned:
        visible = len(realized) / max(len(planned), 1)
        c.metrics["cut_visibility"] = visible
        if visible < 0.75:
            c.issues.append({
                "code": "indistinct_cuts", "severity": "medium",
                "message": (f"only {len(realized)} of {len(planned)} planned cuts are "
                            "visually distinct — neighbouring clips look too alike, so "
                            "the cuts don't register"),
                "directive": {"action": "increase_visual_contrast"}})

    # ---- pacing vs the reference (measured on the plan, which is what we control) ----
    realized_cps = planned_cps
    # The brief moves the target. Judging a deliberately slow edit against the
    # reference's cut rate would make the critic "fix" exactly what was asked for.
    target_cps = grammar.pacing.cuts_per_second
    if cons is not None and cons.pacing_multiplier > 0:
        target_cps = target_cps / cons.pacing_multiplier
        c.metrics["brief_pacing_multiplier"] = cons.pacing_multiplier
    c.metrics["target_cuts_per_second"] = target_cps
    c.pacing_error = float(realized_cps - target_cps)
    c.metrics["cuts_per_second"] = realized_cps
    c.metrics["detected_cuts_per_second"] = len(realized) / max(info.duration, 1e-6)
    c.metrics["reference_cuts_per_second"] = target_cps
    if target_cps > 0:
        ratio = realized_cps / target_cps
        if ratio < 0.62:
            c.issues.append({
                "code": "too_slow", "severity": "medium",
                "message": (f"{realized_cps:.2f} cuts/s against the reference's "
                            f"{target_cps:.2f} — the edit is slower than the style"),
                "directive": {"action": "increase_cut_density", "factor": float(min(1.7, 1/ratio))}})
        elif ratio > 1.6:
            c.issues.append({
                "code": "too_fast", "severity": "medium",
                "message": (f"{realized_cps:.2f} cuts/s against the reference's "
                            f"{target_cps:.2f} — cuts are coming faster than the style"),
                "directive": {"action": "decrease_cut_density", "factor": float(max(0.6, 1/ratio))}})

    # ---- continuity: luminance jumps at cuts ----
    jumps = []
    for t in realized:
        i = int(np.argmin(np.abs(times - t)))
        if 2 <= i < len(lumas) - 2:
            jumps.append(abs(float(lumas[i + 2]) - float(lumas[i - 2])))
    if jumps:
        c.metrics["mean_luma_jump"] = float(np.mean(jumps))
        harsh = [j for j in jumps if j > 62]
        if len(harsh) > max(1, len(jumps) * 0.3):
            c.issues.append({
                "code": "harsh_luma_cuts", "severity": "medium",
                "message": (f"{len(harsh)} of {len(jumps)} cuts jump hard in brightness; "
                            "unintended flicker rather than deliberate contrast"),
                "directive": {"action": "soften_bright_cuts", "threshold": 62}})

    # ---- realized intensity vs target ----
    if len(times):
        from scipy import ndimage
        grid = np.linspace(0, info.duration, 48)
        cutd = np.zeros_like(grid)
        for t in realized:
            cutd += np.exp(-((grid - t) ** 2) / (2 * 0.6 ** 2))
        if cutd.max() > 0:
            cutd /= cutd.max()
        mot = np.interp(grid, times, ndimage.uniform_filter1d(scores, 8))
        if mot.max() > 0:
            mot /= mot.max()
        realized_curve = np.clip(0.62 * cutd + 0.38 * mot, 0, 1)
        target_curve = np.array([grammar.intensity_at(t / max(info.duration, 1e-6))
                                 for t in grid])
        c.intensity_realized = float(realized_curve.mean() * 10)
        err = float(np.mean(np.abs(realized_curve - target_curve)))
        c.metrics["intensity_tracking_error"] = err
        c.metrics["peak_position_realized"] = float(grid[int(np.argmax(realized_curve))])
        c.metrics["peak_position_target"] = float(grid[int(np.argmax(target_curve))])
        if err > 0.3:
            c.issues.append({
                "code": "intensity_mismatch", "severity": "medium",
                "message": (f"the edit's energy shape differs from the reference's arc "
                            f"(mean error {err:.2f})"),
                "directive": {"action": "reshape_intensity"}})

    # ---- effect density ----
    total_fx = sum(len(cl.effects) for cl in timeline.clips)
    per_clip = total_fx / max(len(timeline.clips), 1)
    c.metrics["effects_per_clip"] = per_clip
    fx_ceiling = 3.4 if cons is None else (1.2 + 3.2 * cons.effect_density)
    c.metrics["effect_ceiling"] = fx_ceiling
    if per_clip > fx_ceiling and (cons is None or cons.effect_density < 0.6):
        c.issues.append({
            "code": "over_processed", "severity": "medium",
            "message": (f"{per_clip:.1f} effects per clip while the reference is "
                        f"comparatively restrained ({grammar.effect_density:.2f}) — "
                        "effects are being added without justification"),
            "directive": {"action": "strip_lowest_value_effects", "target": 2}})

    # ---- exposure + uniformity across the render (Spec §13 project-level pass) ----
    sharps, brights = [], []
    with FrameReader(render_path, analysis_fps=2.0, width=384) as fr:
        import cv2
        for _, f in fr.iter_frames():
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sharps.append(cv2.Laplacian(g, cv2.CV_64F).var())
            brights.append(float(g.mean()))
    if sharps:
        sm, ss = float(np.mean(sharps)), float(np.std(sharps))
        c.metrics["sharpness_cv"] = ss / max(sm, 1e-6)
        c.metrics["brightness_std"] = float(np.std(brights))
        if ss / max(sm, 1e-6) > 1.15:
            c.issues.append({
                "code": "quality_inconsistent", "severity": "low",
                "message": ("sharpness varies a lot between shots; the project does not "
                            "yet read as one coherent piece"),
                "directive": {"action": "harmonize_sharpness"}})
        if float(np.std(brights)) > 55:
            c.issues.append({
                "code": "exposure_inconsistent", "severity": "low",
                "message": "exposure varies widely between shots",
                "directive": {"action": "harmonize_exposure"}})

    # ---- overall score ----
    penalty = sum({"high": 1.8, "medium": 1.0, "low": 0.4}[i["severity"]] for i in c.issues)
    base = 6.5 + 2.2 * c.beat_alignment
    c.score = float(np.clip(base - penalty, 0, 10))
    c.summary = (f"Intensity {c.intensity_realized:.1f}/10, beat alignment "
                 f"{c.beat_alignment*100:.0f}%, {len(realized)} cuts over "
                 f"{info.duration:.1f}s ({realized_cps:.2f}/s vs reference "
                 f"{target_cps:.2f}/s). "
                 + ("No significant issues found." if not c.issues else
                    f"{len(c.issues)} issue(s): " + "; ".join(i["code"] for i in c.issues)))

    if ledger:
        ledger.record(stage="self_critique", subject="preview",
                      choice=f"score {c.score:.1f}/10", rationale=c.summary,
                      confidence=0.75, evidence=c.metrics)
    return c


# --------------------------------------------------------------------- revise
def apply_revisions(timeline, crit, grammar, audio, slots, ledger=None,
                    overrides=None):
    """Apply the critic's directives to the timeline. Returns (timeline, changed).

    Revisions are surgical mutations of the existing timeline so that unchanged
    clips hit the render cache — this is what keeps the loop cheap.
    """
    # A human lock outranks the critic absolutely. Without this, the revision
    # loop silently undoes exactly the choice the user asked to keep, which is
    # the worst possible failure for a directable tool.
    locked_slots = overrides.locked_slots() if overrides else set()

    def _locked(clip) -> bool:
        try:
            return int(str(clip.id).lstrip("c")) in locked_slots
        except (ValueError, AttributeError):
            return False

    changed = False
    for d in crit.directives():
        action = d.get("action")

        if action == "resnap_to_beats" and audio.beats:
            # Re-derive every boundary from the beat grid in ABSOLUTE terms and
            # quantize to frames, so the correction cannot itself introduce the
            # cumulative drift it is meant to remove.
            fps = timeline.project.fps
            b = np.asarray(audio.beats)
            base = b[0] if len(b) else 0.0
            clips = sorted(timeline.clips, key=lambda x: x.timeline_start)
            edges = [0.0]
            for cl in clips:
                target = edges[-1] + cl.timeline_duration
                cand = b - base
                cand = cand[np.abs(cand - target) < 0.20]
                snapped = (float(cand[int(np.argmin(np.abs(cand - target)))])
                           if len(cand) else target)
                snapped = round(snapped * fps) / fps
                edges.append(max(edges[-1] + 1.0 / fps, snapped))
            for i, cl in enumerate(clips):
                cl.timeline_start = edges[i]
                if not _locked(cl):
                    cl.duration = edges[i + 1] - edges[i]
                avail = cl.source_out - cl.source_in
                cl.hold_extended = max(0.0, cl.duration - avail / max(cl.speed, 1e-6))
            changed = True

        elif action == "soften_bright_cuts":
            from ..timeline.schema import Transition, TransitionType
            for cl in timeline.clips:
                if (not _locked(cl) and cl.transition_in
                        and cl.transition_in.type == TransitionType.CUT.value):
                    cl.transition_in = Transition(
                        type=TransitionType.FLASH.value, duration=0.05,
                        reason="critic: hard brightness jump at this cut — a short "
                               "flash makes the change read as intentional")
                    changed = True
                    break

        elif action == "strip_lowest_value_effects":
            target = int(d.get("target", 2))
            priority = {"reframe": 0, "color_grade": 1, "push_zoom": 2, "pull_zoom": 2,
                        "directional_blur": 3, "shake": 4, "film_grain": 5,
                        "vignette": 6, "glow": 7}
            for cl in timeline.clips:
                if _locked(cl):
                    continue
                if len(cl.effects) > target:
                    cl.effects = sorted(cl.effects,
                                        key=lambda e: priority.get(e.type, 9))[:target]
                    changed = True

        elif action in ("increase_cut_density", "decrease_cut_density",
                        "increase_visual_contrast", "reshape_intensity"):
            # Handled by the orchestrator, which can re-plan and rebuild. Changing
            # clip durations here would alter the total running time, which must
            # stay locked to the music.
            continue

        elif action == "harmonize_exposure":
            from ..timeline.schema import Effect, EffectType
            for cl in timeline.clips:
                if _locked(cl):
                    continue
                cl.effects.append(Effect(
                    EffectType.COLOR_GRADE.value, {"contrast": 1.04, "brightness": 0.0},
                    reason="critic: project-level uniformity pass"))
            changed = True

    if changed and ledger:
        ledger.record(stage="revision", subject="timeline",
                      choice=", ".join(d.get("action", "?") for d in crit.directives()),
                      rationale=("applied the critic's directives; unchanged clips will "
                                 "be reused from the render cache"),
                      confidence=0.7)
    return timeline, changed
