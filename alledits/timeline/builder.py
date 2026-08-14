"""Timeline construction (Spec §10, §11).

Fills the slot plan with selected shots, then chooses editing-grammar primitives
whose parameters come from the reference's measured tendencies and the local
musical moment. Effects are applied because there is a reason, and the reason is
stored on the effect (Principle 13) — this is the opposite of a preset generator.
"""
from __future__ import annotations

import numpy as np

from .schema import (Timeline, TimelineClip, Effect, Transition, AudioTrack,
                     ProjectSettings, EffectType, TransitionType)
from ..intelligence.selector import select_for_slot


_SPEECH_CACHE: dict = {}


def _speech_windows(shot):
    """Speaking windows inside a shot, or [] if it has none.

    Cached per source file: one file usually supplies several shots, and
    re-analysing it per shot is pure waste. Any failure returns [] so that a
    missing or unreadable audio stream degrades to the old behaviour rather than
    breaking the build.
    """
    src = str(shot.source_path)
    if src not in _SPEECH_CACHE:
        try:
            from ..audio.speech import detect_speech
            sp = detect_speech(src)
            _SPEECH_CACHE[src] = sp.windows if sp.has_speech else []
        except Exception:
            _SPEECH_CACHE[src] = []
    return _SPEECH_CACHE[src]


def _pick_in_point(shot, need: float) -> tuple[float, float]:
    """Choose the sub-range of a shot to use.

    Default: the middle, where a shot is usually most stable.

    Exception: if the shot carries speech, land ON the speech. Observed in a
    real run — the middle-of-shot rule chose a 0.47s range ending 10 ms before
    the dialogue began, so the one talking clip in the edit contributed silence
    and the mix had nothing to duck for. Where someone is speaking, that is the
    part of the shot worth using.
    """
    avail = shot.duration
    cap = shot.quality.get("max_useful_duration")
    if cap:
        need = min(need, cap)
    if avail <= need:
        return shot.start, shot.end

    windows = [(a, b) for a, b in _speech_windows(shot)
               if b > shot.start and a < shot.end]
    if windows:
        # the speaking window with the most overlap inside this shot
        a, b = max(windows, key=lambda w: min(w[1], shot.end) - max(w[0], shot.start))
        a, b = max(a, shot.start), min(b, shot.end)
        if b - a >= min(need, 0.2):
            # centre the range on the speech, then clamp inside the shot
            centre = (a + b) / 2.0
            s_in = centre - need / 2.0
            s_in = max(shot.start, min(s_in, shot.end - need))
            return s_in, s_in + need

    # skip the first 12% (settling) unless that costs us the length
    lead = min(avail * 0.12, max(0.0, avail - need))
    mid_start = shot.start + lead + (avail - lead - need) * 0.35
    return mid_start, mid_start + need


def _transition_fit(prev_shot, shot, slot, intensity):
    """How well each non-cut transition would suit THIS boundary, 0..1.

    A transition is only appropriate where the footage gives it something to
    work with — a whip needs motion on both sides to hide inside, a flash needs
    an accent to punctuate, a dissolve needs a calm passage.
    """
    if prev_shot is None:
        return {}
    flow = shot.visual["flow_magnitude"]
    prev_flow = prev_shot.visual["flow_magnitude"]
    dbright = abs(shot.visual["brightness"] - prev_shot.visual["brightness"])
    same_dir = abs(((prev_shot.visual["flow_direction_deg"]
                     - shot.visual["flow_direction_deg"] + 180) % 360) - 180) < 60

    fit = {}
    fit[TransitionType.WHIP.value] = float(np.clip(
        min(flow, prev_flow) / 2.5 * (1.25 if same_dir else 0.8) * (0.6 + intensity), 0, 1))
    fit[TransitionType.FLASH.value] = float(np.clip(
        (1.0 if slot.on_drop else 0.0) * 0.6 + intensity * 0.5 + dbright * 1.1, 0, 1))
    fit[TransitionType.DISSOLVE.value] = float(np.clip(
        (1.0 - intensity) * 0.9 + (0.2 if slot.role == "release" else 0.0), 0, 1))
    return fit


def _assign_transitions(clips, fits, grammar, ledger=None, cons=None):
    """Reproduce the reference's transition MIX, placing each transition at the
    boundary that suits it best.

    Choosing the single best transition at each cut independently collapses to
    "hard cut" everywhere, because hard cuts dominate almost every reference —
    the edit then loses the reference's texture entirely. Allocating by quota
    keeps the reference's proportions while still putting each effect where the
    footage actually supports it.
    """
    tp = grammar.transitions
    n = len(clips) - 1
    if n <= 0:
        return
    bias = (cons.transition_bias if cons else {}) or {}
    cut_bias = float(bias.get("cut", 1.0))
    # A brief demanding hard cuts suppresses treated transitions proportionally.
    suppress = 1.0 / max(cut_bias, 1e-6) if cut_bias > 1.0 else 1.0
    quotas = {
        TransitionType.WHIP.value:
            int(round(tp.whip_share * n * float(bias.get("whip", 1.0)) * suppress)),
        TransitionType.FLASH.value:
            int(round(tp.flash_share * n * float(bias.get("flash", 1.0)) * suppress)),
        TransitionType.DISSOLVE.value:
            int(round(tp.dissolve_share * n * float(bias.get("dissolve", 1.0)) * suppress)),
    }
    cap = (cons.max_transition_share if cons else 0.6)
    total_special = sum(quotas.values())
    if total_special > n * cap:                      # never out-transition the brief
        scale = (n * cap) / max(total_special, 1)
        quotas = {k: int(v * scale) for k, v in quotas.items()}

    taken = set()
    durations = {TransitionType.WHIP.value: 0.13,
                 TransitionType.FLASH.value: 0.07,
                 TransitionType.DISSOLVE.value: 0.22}
    reasons = {
        TransitionType.WHIP.value: ("both shots carry real motion, so a whip hides the "
                                    "cut inside the movement instead of drawing "
                                    "attention to it"),
        TransitionType.FLASH.value: ("a musical accent with a bright jump across the cut "
                                     "— a flash punctuates it"),
        TransitionType.DISSOLVE.value: ("a calm passage, where a dissolve keeps the "
                                        "transition soft rather than abrupt"),
    }
    for ttype, quota in sorted(quotas.items(), key=lambda kv: -kv[1]):
        if quota <= 0:
            continue
        ranked = sorted((i for i in range(1, len(clips)) if i not in taken),
                        key=lambda i: -fits.get(i, {}).get(ttype, 0.0))
        for i in ranked[:quota]:
            if fits.get(i, {}).get(ttype, 0.0) < 0.25:
                continue                              # not supported by the footage
            dur = min(durations[ttype], clips[i].timeline_duration * 0.45)
            clips[i].transition_in = Transition(
                type=ttype, duration=float(dur),
                reason=(f"{reasons[ttype]} (reference uses this on "
                        f"{getattr(grammar.transitions, ttype + '_share', 0):.0%} of "
                        f"its cuts; fit here {fits[i][ttype]:.2f})"))
            taken.add(i)
            if ledger:
                ledger.record(stage="transition_choice", subject=clips[i].id,
                              choice=ttype, rationale=clips[i].transition_in.reason,
                              confidence=float(fits[i][ttype]),
                              evidence={"duration": dur})
    for i in range(1, len(clips)):
        if i not in taken:
            clips[i].transition_in = Transition(
                type=TransitionType.CUT.value, duration=0.0,
                reason=("straight cut — the reference cuts hard on "
                        f"{tp.hard_cut_share:.0%} of its edits and nothing here calls "
                        "for a treated transition"))


def _choose_effects(grammar, shot, slot, intensity, project, cons=None) -> list[Effect]:
    """Editing-grammar primitives, parameterized by intent."""
    fx: list[Effect] = []
    v, q = shot.visual, shot.quality
    density = 0.5 if cons is None else cons.effect_density
    motion_gate = 0.42 if cons is None else (1.0 - density) * 0.85
    allow_shake = True if cons is None else cons.allow_shake
    allow_grain = False if cons is None else cons.allow_grain
    max_fx = 5 if cons is None else cons.max_effects_per_clip

    # 1. Reframe when source and target aspect differ — subject-aware, not blind.
    src_ar = shot.visual.get("_src_aspect")
    if src_ar and abs(src_ar - project.width / project.height) > 0.05:
        fx.append(Effect(EffectType.REFRAME.value,
                         {"x": float(np.clip(v["subject_x"], 0.12, 0.88)),
                          "y": float(np.clip(v["subject_y"], 0.12, 0.88)),
                          "scale": 1.0},
                         reason=(f"source is {src_ar:.2f}:1 but the project is "
                                 f"{project.width/project.height:.2f}:1; the crop is "
                                 f"placed on the detected subject at "
                                 f"({v['subject_x']:.2f}, {v['subject_y']:.2f}) rather "
                                 f"than blindly on the frame centre")))

    # 2. Motion. Add movement where the shot lacks it, amplify where it has it.
    if v["camera_movement"] == "static" and intensity > motion_gate:
        amt = (0.04 + 0.10 * intensity) * (0.5 + density)
        fx.append(Effect(EffectType.PUSH_ZOOM.value, {"from": 1.0, "to": 1.0 + amt},
                         reason=("the shot is locked off but this moment needs forward "
                                 "energy, so a slow push supplies it")))
    elif v["camera_movement"] == "push_in" and intensity > 0.7:
        fx.append(Effect(EffectType.PUSH_ZOOM.value, {"from": 1.0, "to": 1.12},
                         reason="extends the camera's existing push at the peak"))
    elif v["camera_movement"] == "pull_out" and intensity < 0.4:
        fx.append(Effect(EffectType.PULL_ZOOM.value, {"from": 1.12, "to": 1.0},
                         reason="follows the shot's own pull-back during the release"))

    # 3. Shake, only at genuine peaks and only if the shot isn't already shaky.
    if (allow_shake and intensity > 0.82 - density * 0.25 and v["shake"] < 1.5
            and slot.role in ("climax", "escalation")):
        fx.append(Effect(EffectType.SHAKE.value,
                         {"amplitude": float((0.003 + 0.007 * intensity) * (0.5 + density)),
                          "frequency": 11.0},
                         reason="peak intensity — a small shake adds impact without "
                                "destroying readability"))

    # 4a. RESTORATION: one treatment per measured defect, never a blanket pass.
    #     Ordered so denoise precedes sharpen — sharpening noise amplifies it.
    for d in (q.get("defects") or []):
        t, sev = d["treatment"], float(d["severity"])
        etype = {"denoise": EffectType.DENOISE.value,
                 "sharpen": EffectType.SHARPEN.value,
                 "deblock": EffectType.DEBLOCK.value,
                 "stabilize": EffectType.STABILIZE.value,
                 "expand_contrast": EffectType.EXPAND_CONTRAST.value}.get(t)
        if not etype:
            continue
        # Sharpening a noisy clip that is NOT being denoised makes it worse.
        if etype == EffectType.SHARPEN.value:
            denoising = any(x["treatment"] == "denoise" for x in q["defects"])
            if q["noise"] > 0.45 and not denoising:
                continue
        params = ({"strength": float(np.clip(0.25 + 0.6 * sev, 0.1, 1.0))}
                  if etype != EffectType.STABILIZE.value
                  else {"strength": float(np.clip(0.3 + 0.6 * sev, 0.1, 1.0)),
                        "zoom": 0.03})
        fx.append(Effect(etype, params,
                         reason=f"restoration: {d['detail']}"))

    # 4b. Enhancement ONLY where quality analysis justified it (Spec §13).
    if q["handling"] == "enhance":
        grade = {}
        if v["contrast"] < grammar.color.contrast - 0.06:
            grade["contrast"] = float(np.clip(1.0 + (grammar.color.contrast - v["contrast"]) * 1.4, 1.0, 1.5))
        if v["saturation"] < grammar.color.saturation - 0.05:
            grade["saturation"] = float(np.clip(1.0 + (grammar.color.saturation - v["saturation"]) * 1.2, 1.0, 1.6))
        if q["exposure_health"] < 0.8:
            grade["brightness"] = 0.03
        if grade:
            fx.append(Effect(EffectType.COLOR_GRADE.value, grade,
                             reason="quality analysis flagged this clip for correction: "
                                    + "; ".join(q["reasons"])))

    # 5. Project-level look pulled from the reference's grading direction.
    look = {}
    tgt_warm = grammar.color.warmth + (cons.warmth_delta if cons else 0.0)
    tgt_contrast = grammar.color.contrast + (cons.contrast_delta if cons else 0.0)
    tgt_sat = grammar.color.saturation + (cons.saturation_delta if cons else 0.0)
    if abs(tgt_warm - v["warmth"]) > 0.10:
        look["temperature"] = float(np.clip((tgt_warm - v["warmth"]) * 0.6, -0.6, 0.6))
    if abs(tgt_contrast - 0.5) > 0.03:
        look["contrast"] = float(np.clip(1.0 + (tgt_contrast - 0.5) * 0.5, 0.7, 1.6))
    if abs(tgt_sat - v["saturation"]) > 0.06:
        look["saturation"] = float(np.clip(1.0 + (tgt_sat - v["saturation"]) * 0.7, 0.4, 1.8))
    if look:
        look.setdefault("saturation", float(np.clip(1.0 + (tgt_sat - v["saturation"]) * 0.5, 0.4, 1.8)))
        fx.append(Effect(EffectType.COLOR_GRADE.value, look,
                         reason=(("brief-adjusted " if cons and (cons.warmth_delta or cons.contrast_delta or cons.saturation_delta) else "")
                                 + "matches the reference's grading direction "
                                 f"(warmth {grammar.color.warmth:+.2f}, "
                                 f"contrast {grammar.color.contrast:.2f}) so mixed "
                                 "sources read as one project")))

    # 6. Directional blur on brief flash frames only.
    if q["handling"] == "use_briefly" and slot.duration <= 0.4:
        fx.append(Effect(EffectType.DIRECTIONAL_BLUR.value,
                         {"strength": 0.45, "angle": float(v["flow_direction_deg"])},
                         reason="low-quality frame used as a fast accent — blur along "
                                "its own motion vector hides artifacts and reads as speed"))
    if allow_grain and len(fx) < max_fx:
        fx.append(Effect(EffectType.FILM_GRAIN.value,
                         {"strength": float(0.12 + 0.2 * density)},
                         reason="the brief asks for a grainier, rawer texture"))
    return fx[:max_fx]


def _apply_duration_overrides(tl, overrides, clip_slot_index, fps, ledger=None):
    """Honour explicit "hold this longer/shorter" directives.

    Slot positions normally come straight from the beat plan, so changing one
    clip's length would leave a gap or an overlap — which the validator rejects.
    Everything downstream is therefore re-flowed.

    That has a real cost: shifted cuts no longer land on the musical grid. The
    human asked for this, so it is applied — but the affected clips are marked
    `beat_locked = False` and the reason is recorded, rather than leaving the
    timeline claiming a beat-lock it no longer honours.
    """
    forced = {}
    for idx, slot_index in clip_slot_index.items():
        d = overrides.duration_at(slot_index)
        if d is not None:
            forced[idx] = max(1.0 / fps, round(float(d) * fps) / fps)
    if not forced:
        return

    ordered = sorted(range(len(tl.clips)), key=lambda i: tl.clips[i].timeline_start)
    cursor = tl.clips[ordered[0]].timeline_start if ordered else 0.0
    first_change = min(forced)
    for i in ordered:
        clip = tl.clips[i]
        if i in forced:
            clip.duration = forced[i]
            avail = clip.source_out - clip.source_in
            clip.hold_extended = max(0.0, clip.duration - avail / max(clip.speed, 1e-6))
        new_start = round(cursor * fps) / fps
        if abs(new_start - clip.timeline_start) > 1e-6 and i >= first_change:
            clip.timeline_start = new_start
            if clip.beat_locked:
                clip.beat_locked = False
                clip.selection_reason += (
                    " (shifted off the beat grid by an explicit duration change "
                    "earlier in the timeline)")
        cursor = clip.timeline_start + clip.duration
        if ledger and i in forced:
            ledger.record(stage="human_override", subject=clip.id, actor="human",
                          choice=f"set_duration={forced[i]:.2f}s", confidence=1.0,
                          rationale="explicit duration set by the director")


def _selection_reason(best, overrides, slot) -> str:
    """Never let a human's choice be recorded as the system's own.

    The scorer still runs on a pinned shot (its numbers are useful context), but
    the stored reason must say plainly who decided, or the timeline misrepresents
    the provenance of the edit.
    """
    if overrides is not None and overrides.pinned_at(slot.index):
        return (f"pinned by the user for this slot; automatic ranking was not used "
                f"(for reference the automatic score would have been {best.score:.3f})")
    return best.explanation


def build_timeline(slots, index, grammar, audio, project: ProjectSettings,
                   music_path: str, music_start: float = 0.0,
                   intent: str = "", ledger=None, diversity: float = 0.0,
                   cons=None, overrides=None) -> Timeline:
    tl = Timeline(project=project, intent=intent, style_grammar_id=grammar.id)
    candidates = index.usable_shots(include_brief=True)
    if not candidates:
        return tl

    prev_shot = None
    used_ids, used_assets = set(), set()
    fits = {}
    clip_slot_index = {}
    fps = project.fps
    q = lambda t: round(t * fps) / fps      # frame quantization

    base = slots[0].start if slots else 0.0
    for slot in slots:
        # A shot may never immediately follow itself: the "cut" would be invisible
        # and the edit silently loses a beat.
        pool = [s for s in candidates if prev_shot is None or s.id != prev_shot.id]
        best, alts = select_for_slot(slot, pool or candidates, prev_shot, used_ids,
                                     used_assets, ledger=ledger,
                                     diversity=diversity + (cons.diversity if cons else 0.0),
                                     cons=cons, overrides=overrides)
        if best is None:
            continue
        shot = best.shot

        # absolute, frame-quantized position taken straight from the beat plan —
        # never accumulated, so error cannot compound along the timeline
        t_start = q(slot.start - base)
        t_end = q(slot.end - base)
        need = max(1.0 / fps, t_end - t_start)

        s_in, s_out = _pick_in_point(shot, need)
        actual = s_out - s_in
        speed = 1.0
        hold = 0.0
        if actual < need - 1e-4:
            # stretch time rather than shorten the slot (which would break sync)
            speed = float(np.clip(actual / need, 0.5, 1.0))
            still_short = need - actual / speed
            if still_short > 1e-4:
                hold = still_short      # renderer holds the last frame

        clip = TimelineClip(
            id=f"c{slot.index:03d}", source_id=shot.id,
            source_path=shot.source_path, source_in=s_in, source_out=s_out,
            timeline_start=t_start, duration=need, speed=speed, hold_extended=hold,
            role=slot.role, beat_locked=slot.beat_locked,
            quality_handling=shot.quality.get("handling", "use"),
            selection_reason=_selection_reason(best, overrides, slot))
        clip.effects = _choose_effects(grammar, shot, slot, slot.intensity, project, cons)
        clip.treatments = sorted({e.type for e in clip.effects}
                                 & {EffectType.DENOISE.value, EffectType.SHARPEN.value,
                                    EffectType.DEBLOCK.value, EffectType.STABILIZE.value,
                                    EffectType.EXPAND_CONTRAST.value})
        if overrides is not None:
            forced = overrides.forced_effects(slot.index)
            if forced is not None:
                clip.effects = [Effect(f["type"], f.get("params", {}),
                                       reason="set explicitly by the user")
                                for f in forced]
            banned = overrides.banned_effects()
            if banned:
                clip.effects = [e for e in clip.effects if e.type not in banned]
        tl.clips.append(clip)
        clip_slot_index[len(tl.clips) - 1] = slot.index
        fits[len(tl.clips) - 1] = _transition_fit(prev_shot, shot, slot, slot.intensity)

        if ledger and clip.effects:
            for e in clip.effects:
                ledger.record(stage="effect_choice", subject=clip.id, choice=e.type,
                              rationale=e.reason, confidence=0.7, evidence=e.params)

        prev_shot = shot
        used_ids.add(shot.id)
        used_assets.add(shot.asset_id)

    # Re-resolve human directives against the timeline that was ACTUALLY built.
    # Slot indices shift whenever the plan reshapes, so an anchored directive
    # follows its shot; one whose shot is gone is reported and NOT applied.
    if overrides is not None and hasattr(overrides, "bind"):
        binding = overrides.bind(tl.clips)
        tl.notes = list(getattr(tl, "notes", []) or [])
        for moved in binding.get("moved", []):
            tl.notes.append(f"directive followed its shot: {moved}")
            if ledger:
                ledger.record(stage="human_override", subject="anchor",
                              choice="followed", actor="system", confidence=1.0,
                              rationale=moved)
        for lost in binding.get("lost", []):
            tl.notes.append(f"directive not applied: {lost}")
            if ledger:
                ledger.record(stage="human_override", subject="anchor",
                              choice="lost", actor="system", confidence=1.0,
                              rationale=lost)

    _assign_transitions(tl.clips, fits, grammar, ledger=ledger, cons=cons)
    if overrides is not None:
        _apply_duration_overrides(tl, overrides, clip_slot_index, fps,
                                  ledger=ledger)
    if overrides is not None:
        # applied AFTER quota allocation so a user choice cannot be overwritten
        for i, clip in enumerate(tl.clips):
            forced = overrides.forced_transition(clip_slot_index.get(i, i))
            if forced:
                clip.transition_in = Transition(
                    type=str(forced), duration=0.0 if forced == "cut" else 0.12,
                    reason="set explicitly by the user")

    # music bed, cut to the edit
    if music_path and tl.clips:
        tl.audio.append(AudioTrack(
            id="music", source_path=music_path, source_in=music_start,
            source_out=music_start + tl.duration, timeline_start=0.0,
            fade_in=0.0, fade_out=min(0.6, tl.duration * 0.12), role="music"))
    return tl
