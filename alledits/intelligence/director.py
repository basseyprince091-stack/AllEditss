"""DIRECTOR: turning an editorial note into changes (Spec §17).

This is the mode where someone watches the cut and says what is wrong with it.
"lose the shaky one", "hold the third shot longer", "no flashes", "punch it up".

CAPABILITY HONESTY — the same rule as FIND.

There is no language model in this build, so this does not *understand* a note.
It matches editorial vocabulary against the timeline that already exists, and
resolves references three ways:

  by position   — "the third clip", "the last shot", "clip 2"
  by attribute  — "the shaky one", "the dark shot" (matched on MEASURED signal)
  globally      — "no flashes", "punch it up"

Anything it cannot resolve is returned in `unresolved` and reported to the user.
A note that half-worked is dangerous precisely because the half that failed is
invisible — the user believes the whole note was applied and re-watches looking
for a change that was never made.

Notes become DIRECTIVES against the project, not edits to a rendered file, so
they are auditable, reversible, and survive a re-plan — the Phase 2 guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.project import DirectiveKind


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
            "7th": 7, "8th": 8, "9th": 9, "10th": 10}

# Attribute references resolved against MEASURED signal on the timeline's clips.
ATTRIBUTE_REFS = {
    "shaky": ("shake", "max"), "wobbly": ("shake", "max"),
    "handheld": ("shake", "max"),
    "dark": ("brightness", "min"), "darkest": ("brightness", "min"),
    "bright": ("brightness", "max"), "brightest": ("brightness", "max"),
    "blurry": ("sharpness", "min"), "softest": ("sharpness", "min"),
    "soft": ("sharpness", "min"),
    "noisy": ("noise", "max"), "grainy": ("noise", "max"),
    "sharpest": ("sharpness", "max"),
    "longest": ("duration", "max"), "shortest": ("duration", "min"),
}

LONGER = ("longer", "hold", "linger", "extend", "stretch", "breathe")
SHORTER = ("shorter", "tighten", "trim", "quicker", "snappier", "shorten")

EFFECT_WORDS = {
    "flash": "flash", "flashes": "flash", "whip": "whip", "whips": "whip",
    "shake": "shake", "shakes": "shake", "zoom": "push_zoom",
    "zooms": "push_zoom", "grade": "color_grade", "grading": "color_grade",
    "dissolve": "dissolve", "dissolves": "dissolve",
}
TRANSITION_WORDS = {"cut": "cut", "dissolve": "dissolve", "flash": "flash",
                    "whip": "whip"}

# Notes that are about the whole piece rather than one shot pass straight
# through to the brief parser, so DIRECTOR and EDIT share one vocabulary.
GLOBAL_HINTS = ("punch", "faster", "slower", "calmer", "energetic", "restrained",
                "cinematic", "aggressive", "warmer", "colder", "moodier",
                "brighter", "snappy", "chaotic", "smooth", "gentle")

NEGATIONS = ("no ", "not ", "without ", "lose ", "drop ", "remove ", "kill ",
             "get rid of ", "stop ", "less ")


@dataclass
class ProposedChange:
    kind: str
    slot_index: int | None = None
    shot_id: str | None = None
    value: object = None
    note: str = ""
    rationale: str = ""

    def to_dict(self):
        return {"kind": self.kind, "slot_index": self.slot_index,
                "shot_id": self.shot_id, "value": self.value,
                "note": self.note, "rationale": self.rationale}


@dataclass
class DirectorPlan:
    note: str = ""
    changes: list = field(default_factory=list)
    brief_delta: str = ""
    unresolved: list = field(default_factory=list)
    is_llm: bool = False        # never true in this build

    @property
    def understood(self) -> bool:
        return bool(self.changes or self.brief_delta)

    def describe(self) -> str:
        parts = [c.rationale for c in self.changes]
        if self.brief_delta:
            parts.append(f"re-plan with: {self.brief_delta!r}")
        return "; ".join(parts) if parts else "nothing understood"

    def to_dict(self):
        return {"note": self.note, "brief_delta": self.brief_delta,
                "changes": [c.to_dict() for c in self.changes],
                "unresolved": self.unresolved, "is_llm": self.is_llm}


def _brief_would_act(text: str) -> bool:
    """Does the brief parser actually change anything for this phrase?"""
    try:
        from .brief import parse_brief
        from .constraints import CreativeConstraints
    except Exception:
        return True          # cannot check: prefer acting over blocking
    c, d = parse_brief(text), CreativeConstraints()
    return any(getattr(c, f) != getattr(d, f)
               for f in ("pacing_multiplier", "intensity_gain",
                         "intensity_offset", "effect_density",
                         "continuity_weight", "motion_preference",
                         "warmth_delta", "contrast_delta", "saturation_delta",
                         "diversity", "quality_weight"))


def _clip_attr(clip, name):
    if name == "duration":
        return clip.duration
    v = getattr(clip, name, None)
    if v is not None:
        return v
    for holder in ("visual", "quality"):
        d = getattr(clip, holder, None)
        if isinstance(d, dict) and name in d:
            return d[name]
    return None


def _resolve_attribute(clips, word, index):
    """Find the clip a phrase like 'the shaky one' refers to.

    Returns None when the signal is not present on the timeline rather than
    guessing — picking an arbitrary clip would apply the note to innocent footage.
    """
    attr, direction = ATTRIBUTE_REFS[word]
    scored = []
    for c in clips:
        v = _clip_attr(c, attr)
        if v is None and index is not None:
            shot = index.get(getattr(c, "source_id", None))
            if shot is not None:
                v = (shot.visual or {}).get(attr, (shot.quality or {}).get(attr))
        if v is not None:
            scored.append((float(v), c))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=(direction == "max"))
    return scored[0][1]


def _ordinal_target(text, clips):
    """Positional reference: 'the third clip', 'clip 2', 'the last shot'."""
    if re.search(r"\b(last|final)\b", text) and clips:
        return clips[-1]
    if re.search(r"\b(first|opening)\b", text) and clips:
        return clips[0]
    m = re.search(r"\b(?:clip|shot|cut)\s*#?\s*(\d+)\b", text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= len(clips):
            return clips[n - 1]
        return "out_of_range"
    for word, n in ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            if 1 <= n <= len(clips):
                return clips[n - 1]
            return "out_of_range"
    return None


def _slot_of(clip) -> int:
    """Directives are slot-indexed; clip ids are cNNN where NNN is the slot."""
    try:
        return int(str(clip.id).lstrip("c"))
    except (ValueError, AttributeError):
        return 0


def parse_note(text: str, timeline=None, index=None) -> DirectorPlan:
    """Turn an editorial note into typed directives plus an optional brief."""
    plan = DirectorPlan(note=text or "")
    if not text or not text.strip():
        return plan
    clips = sorted(getattr(timeline, "clips", []) or [],
                   key=lambda c: c.timeline_start)
    t = " " + text.lower().strip() + " "

    for clause in re.split(r"[,;.]| and ", t):
        clause = clause.strip()
        if not clause:
            continue
        negated = any(n in " " + clause + " " for n in NEGATIONS)
        target = _ordinal_target(clause, clips)
        if target == "out_of_range":
            plan.unresolved.append(
                f"{clause!r}: the edit has only {len(clips)} clips")
            continue
        if target is None:
            for word in ATTRIBUTE_REFS:
                if re.search(rf"\b{word}\b", clause):
                    target = _resolve_attribute(clips, word, index)
                    if target is None:
                        plan.unresolved.append(
                            f"{clause!r}: nothing on this timeline measures "
                            f"as {word}")
                    break

        handled = False

        # --- duration ---
        if target is not None and any(w in clause for w in LONGER + SHORTER):
            longer = any(w in clause for w in LONGER)
            factor = 1.5 if longer else 0.6
            m = re.search(r"([\d.]+)\s*(?:s|sec|seconds)\b", clause)
            new_dur = float(m.group(1)) if m else target.duration * factor
            plan.changes.append(ProposedChange(
                kind=DirectiveKind.SET_DURATION.value, slot_index=_slot_of(target),
                value=round(new_dur, 3), note=text,
                rationale=f"hold clip {_slot_of(target)} for {new_dur:.2f}s "
                          f"(was {target.duration:.2f}s)"))
            handled = True

        # --- drop a shot ---
        if not handled and target is not None and negated and not any(
                w in clause for w in EFFECT_WORDS):
            plan.changes.append(ProposedChange(
                kind=DirectiveKind.REJECT_SHOT.value,
                shot_id=getattr(target, "source_id", None), note=text,
                rationale=f"never use {getattr(target, 'source_id', '?')} again"))
            handled = True

        # --- lock ---
        if not handled and target is not None and re.search(
                r"\b(keep|lock|leave|don't touch|dont touch)\b", clause):
            plan.changes.append(ProposedChange(
                kind=DirectiveKind.LOCK_CLIP.value, slot_index=_slot_of(target),
                note=text,
                rationale=f"lock clip {_slot_of(target)} against further revision"))
            handled = True

        # --- effects ---
        if not handled:
            for word, eff in EFFECT_WORDS.items():
                if re.search(rf"\b{word}\b", clause) and negated:
                    plan.changes.append(ProposedChange(
                        kind=DirectiveKind.BAN_EFFECT.value, value=eff, note=text,
                        rationale=f"never apply {eff}"))
                    handled = True
                    break

        # --- transitions ---
        if not handled and target is not None:
            m = re.search(r"\b(?:use|make it|change to)?\s*a?\s*"
                          r"(cut|dissolve|flash|whip)\b", clause)
            if m and not negated:
                plan.changes.append(ProposedChange(
                    kind=DirectiveKind.FORCE_TRANSITION.value,
                    slot_index=_slot_of(target),
                    value=TRANSITION_WORDS[m.group(1)], note=text,
                    rationale=f"force a {m.group(1)} into clip "
                              f"{_slot_of(target)}"))
                handled = True

        # --- whole-piece note: hand to the brief parser, one shared vocabulary ---
        if not handled and any(h in clause for h in GLOBAL_HINTS):
            # Verify the brief parser will ACT on it. Six of these hints were
            # once absent from the lexicon, so DIRECTOR reported "re-plan with
            # 'punch it up'" and the re-plan changed nothing — a note that
            # looked applied and was not.
            if _brief_would_act(clause):
                plan.brief_delta = (plan.brief_delta + " " + clause).strip()
            else:
                plan.unresolved.append(
                    f"{clause!r}: recognised as a whole-piece note, but the "
                    "brief vocabulary has no setting for it")
            handled = True

        if not handled and clause not in ("", "the", "it"):
            plan.unresolved.append(clause)

    return plan


class _SavedClip:
    """Minimal clip view over a persisted timeline dict.

    DIRECTOR runs against a project on disk, where the timeline is JSON rather
    than live objects. Rehydrating the whole Timeline would need the media
    index; the note parser only reads position, duration and measured signal.
    """
    __slots__ = ("id", "source_id", "timeline_start", "duration", "visual",
                 "quality")

    def __init__(self, d: dict):
        self.id = d.get("id", "c000")
        self.source_id = d.get("source_id")
        self.timeline_start = float(d.get("timeline_start", 0.0))
        self.duration = float(d.get("duration", 0.0))
        self.visual = d.get("visual") or {}
        self.quality = d.get("quality") or {}


class _SavedTimeline:
    def __init__(self, d: dict):
        self.clips = [_SavedClip(c) for c in (d or {}).get("clips", [])]


def timeline_from_project(project) -> _SavedTimeline:
    return _SavedTimeline(getattr(project, "timeline", None) or {})
