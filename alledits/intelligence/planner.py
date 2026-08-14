"""Edit planner — turns (style grammar + music analysis + brief) into a
beat-locked slot plan (Spec §11, §23).

A slot is a hole in the edit with requirements attached: how long it should be,
where it sits musically, what role it plays in the arc, and what kind of shot
would satisfy it. Selection then fills the holes.

Cut placement is quantized to the measured beat grid, so synchronisation is
real rather than approximate.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict

from ..timeline.schema import SlotRole


@dataclass
class Slot:
    index: int
    start: float
    end: float
    role: str
    intensity: float             # 0..1 target
    beat_locked: bool = True
    on_downbeat: bool = False
    on_drop: bool = False
    wants: dict = field(default_factory=dict)   # shot requirements
    reason: str = ""

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _role_at(t: float, total: float, grammar) -> str:
    """Map an absolute time onto the reference's narrative arc, scaled to our
    target duration. The arc is taken from the reference, not a fixed template."""
    if not grammar.structure or grammar.duration <= 0:
        frac = t / max(total, 1e-6)
        if frac < 0.12:
            return SlotRole.HOOK.value
        if frac < 0.45:
            return SlotRole.SETUP.value
        if frac < 0.75:
            return SlotRole.ESCALATION.value
        if frac < 0.92:
            return SlotRole.CLIMAX.value
        return SlotRole.RELEASE.value
    scale = total / grammar.duration
    for seg in grammar.structure:
        if seg["start"] * scale <= t < seg["end"] * scale:
            return seg["role"]
    return grammar.structure[-1]["role"]


def _wants_for(role: str, intensity: float, grammar, cons=None) -> dict:
    """What kind of shot should fill this slot."""
    w = {"min_energy": 0.0, "max_energy": 1.0, "prefer_motion": intensity,
         "prefer_faces": False, "prefer_move": None, "allow_brief": False}
    if cons is not None:
        # the brief shifts what kind of shot each slot wants
        w["prefer_motion"] = float(np.clip(
            w["prefer_motion"] + cons.motion_preference * 0.45, 0.0, 1.0))
        w["prefer_faces_weight"] = cons.prefer_faces
        w["shot_size_preference"] = cons.shot_size_preference
    if role == SlotRole.HOOK.value:
        w.update(min_energy=0.45, prefer_motion=max(0.6, intensity),
                 reason="the hook must earn attention immediately")
    elif role == SlotRole.SETUP.value:
        w.update(max_energy=0.75, prefer_faces=True,
                 reason="setup establishes subject and place; readable over kinetic")
    elif role == SlotRole.ANTICIPATION.value:
        w.update(prefer_move="push_in", reason="build tension by moving in")
    elif role == SlotRole.ESCALATION.value:
        w.update(min_energy=0.35, prefer_motion=min(1.0, intensity + 0.15),
                 allow_brief=True, reason="energy rises toward the peak")
    elif role == SlotRole.CLIMAX.value:
        w.update(min_energy=0.55, prefer_motion=1.0, allow_brief=True,
                 reason="peak intensity — strongest motion available")
    elif role == SlotRole.RELEASE.value:
        w.update(max_energy=0.6, prefer_motion=max(0.15, intensity - 0.25),
                 reason="release resolves the edit; let it breathe")
    if cons is not None:
        w["prefer_motion"] = float(np.clip(
            w["prefer_motion"] + cons.motion_preference * 0.45, 0.0, 1.0))
        w["prefer_faces_weight"] = cons.prefer_faces
        w["shot_size_preference"] = cons.shot_size_preference
        if cons.prefer_faces > 0.3:
            w["prefer_faces"] = True
    return w


# A punctuation slot is short enough that salvaged footage is legitimately
# usable in it (see media/salvage.py ROLE_MAX_DURATION).
PUNCTUATION_LEN = 0.20

# Whether a reference PUNCTUATES is relative to its own pacing, not an absolute
# cut length: a reference whose shortest shots match its median has uniform
# shots and no stabs, however fast it cuts. Requiring the short tail to be
# distinctly shorter than the middle is what "uses punctuation" actually means.
# (An absolute threshold was tried first and was the wrong question — the test
# reference sits at p10 = median = 0.467s: fast, but not punctuated.)
PUNCTUATION_RATIO = 0.6     # p10 must be this fraction of the median, or less
PUNCTUATION_MAX = 0.45      # and short enough to be a stab at all


def plan_slots(grammar, audio, target_duration: float,
               start_offset: float = 0.0, ledger=None, cons=None) -> list[Slot]:
    """Build the slot plan. Cuts land on beats; shot lengths follow the
    reference's own duration distribution scaled by local intensity."""
    beats = [b for b in (audio.beats or []) if b >= start_offset]
    bi = audio.beat_interval or (60.0 / (audio.bpm or 120.0))
    downbeats = set(round(b, 3) for b in (audio.downbeats or []))
    drops = audio.drops or []

    slots: list[Slot] = []
    t = start_offset
    idx = 0
    guard = 0
    while t < start_offset + target_duration - 0.12 and guard < 400:
        guard += 1
        rel = (t - start_offset)
        norm = rel / max(target_duration, 1e-6)

        # intensity = reference arc blended with the music's own energy,
        # then shaped by the brief
        ref_i = grammar.intensity_at(norm)
        mus_i = audio.energy_at(t)
        intensity = float(np.clip(0.55 * ref_i + 0.45 * mus_i, 0, 1))
        if cons is not None:
            intensity = float(np.clip(
                intensity * cons.intensity_gain + cons.intensity_offset, 0, 1))

        # near a drop, force the peak
        near_drop = any(abs(t - d) < 0.6 for d in drops)
        if near_drop:
            intensity = min(1.0, intensity + 0.3)

        target_len = grammar.target_shot_duration(intensity)
        if cons is not None:
            # THE pacing lever: the brief scales shot length directly, bounded by
            # the floor/ceiling it also declares.
            target_len = float(np.clip(target_len * cons.pacing_multiplier,
                                       cons.min_shot_floor, cons.max_shot_ceiling))

        # quantize to a whole number of beats
        if bi > 0:
            n_beats = max(1, int(round(target_len / bi)))
            target_len = n_beats * bi
        end = t + target_len

        # snap the boundary to the real (onset-refined) beat grid
        if beats:
            arr = np.asarray(beats)
            cand = arr[np.abs(arr - end) < bi * 0.55]
            if len(cand):
                end = float(cand[int(np.argmin(np.abs(cand - end)))])

        end = min(end, start_offset + target_duration)
        if end - t < 0.1:
            break

        # --- punctuation (Spec 25) ---
        # If the REFERENCE itself punctuates with very short shots, the plan
        # should too. Split a peak slot into a brief stab plus the remainder,
        # which is the only place salvaged footage can legitimately be used:
        # every ordinary slot is longer than a salvage clip's cap.
        # Style-derived, never invented — a reference that never cuts this fast
        # produces no punctuation slots at all.
        p10 = float(getattr(grammar.pacing, "p10_shot", 0.0) or 0.0)
        med = float(getattr(grammar.pacing, "median_shot", 0.0) or 0.0)
        allow_punct = (0.0 < p10 <= PUNCTUATION_MAX
                       and med > 0 and p10 <= PUNCTUATION_RATIO * med)
        if cons is not None and getattr(cons, "effect_density", 0.5) < 0.25:
            allow_punct = False          # a restrained brief does not stab
        if (allow_punct and intensity >= 0.72 and (end - t) >= PUNCTUATION_LEN * 3
                and idx > 0):
            punct_end = q_beat = t + PUNCTUATION_LEN
            slots.append(Slot(
                index=idx, start=t, end=punct_end, role="punctuation",
                intensity=min(1.0, intensity + 0.15), beat_locked=False,
                on_downbeat=round(t, 3) in downbeats, on_drop=near_drop,
                wants=_wants_for("punctuation", intensity, grammar, cons),
                reason=(f"punctuation: the reference's shortest shots ({p10:.2f}s) "
                        f"are well under its median ({med:.2f}s), so stabs are in "
                        f"its vocabulary; placed at intensity {intensity:.2f}")))
            idx += 1
            t = punct_end
            rel = t - start_offset

        role = _role_at(rel, target_duration, grammar)
        wants = _wants_for(role, intensity, grammar, cons)
        slot = Slot(index=idx, start=t, end=end, role=role, intensity=intensity,
                    beat_locked=bool(beats),
                    on_downbeat=round(t, 3) in downbeats,
                    on_drop=near_drop, wants=wants,
                    reason=(f"{role}: reference intensity {ref_i:.2f} x music energy "
                            f"{mus_i:.2f} -> {intensity:.2f}; reference pacing gives "
                            f"{target_len:.2f}s"
                            + (" (aligned to a musical drop)" if near_drop else "")))
        slots.append(slot)
        if ledger:
            ledger.record(stage="slot_plan", subject=f"slot_{idx:02d}",
                          choice=f"{role} {t:.2f}-{end:.2f}s ({target_len:.2f}s)",
                          rationale=slot.reason, confidence=0.8,
                          evidence={"reference_intensity": ref_i, "music_energy": mus_i,
                                    "on_drop": near_drop, "beat_locked": bool(beats)})
        t = end
        idx += 1
    return slots
