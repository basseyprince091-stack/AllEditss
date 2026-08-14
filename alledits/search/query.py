"""FIND: querying a media library by what was actually measured (Spec §7, §8).

CAPABILITY HONESTY — read this before extending the module.

ALLEDITS has no vision-language model in this build, so it cannot do
open-vocabulary semantic search ("the shot where she looks relieved"). Rather
than keyword-guess over feature vectors and call it semantic, FIND does
something narrower and true: it maps a natural-language query onto the
attributes the analysers genuinely measured — camera movement, shot size,
exposure, motion, faces, speech, technical quality, colour — and executes that
as a typed, inspectable filter.

Every result can therefore state WHY it matched, in measured terms. Every term
the parser did not understand is reported as unmatched rather than silently
dropped, because a query that quietly ignores half of what was asked is worse
than one that admits the gap.

`semantic: False` is set on every result of this module. When a CLIP/SigLIP
provider is installed, MediaIndex.search_by_text() becomes available and should
be blended in — it is not a replacement for this, since measured attributes
("handheld", "underexposed", "4K") are exactly what embeddings are weakest at.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


# --- vocabulary -------------------------------------------------------------
# Each entry maps a phrase to (attribute, comparison, value). Multiword phrases
# are matched before single words so "close up" does not match "up".

CAMERA_TERMS = {
    "static": "static", "locked off": "static", "locked-off": "static",
    "tripod": "static", "still": "static",
    "pan": "pan", "panning": "pan", "pan left": "pan_left",
    "pan right": "pan_right", "whip pan": "pan",
    "tilt": "tilt", "tilt up": "tilt_up", "tilt down": "tilt_down",
    "push in": "push_in", "pushing in": "push_in", "dolly in": "push_in",
    "zoom in": "push_in",
    "pull out": "pull_out", "pulling out": "pull_out", "zoom out": "pull_out",
    "dolly out": "pull_out",
    "handheld": "handheld", "hand held": "handheld", "shaky": "handheld",
}

SHOT_SIZE_TERMS = {
    "close up": "close_up", "closeup": "close_up", "close-up": "close_up",
    "cu": "close_up", "tight": "close_up",
    "medium": "medium", "mid shot": "medium", "medium shot": "medium",
    "wide": "wide", "wide shot": "wide", "establishing": "wide",
    "long shot": "wide",
}

# (attribute, op, threshold, human description)
SCALAR_TERMS = {
    "dark":          ("brightness", "<", 0.35, "brightness below 0.35"),
    "low key":       ("brightness", "<", 0.35, "brightness below 0.35"),
    "low-key":       ("brightness", "<", 0.35, "brightness below 0.35"),
    "underexposed":  ("brightness", "<", 0.28, "brightness below 0.28"),
    "bright":        ("brightness", ">", 0.6, "brightness above 0.6"),
    "high key":      ("brightness", ">", 0.6, "brightness above 0.6"),
    "high-key":      ("brightness", ">", 0.6, "brightness above 0.6"),
    "overexposed":   ("clipped_highlights", ">", 0.02, "blown highlights"),
    "warm":          ("warmth", ">", 0.55, "warmth above 0.55"),
    "cold":          ("warmth", "<", 0.45, "warmth below 0.45"),
    "cool":          ("warmth", "<", 0.45, "warmth below 0.45"),
    "colourful":     ("colorfulness", ">", 0.6, "colourfulness above 0.6"),
    "colorful":      ("colorfulness", ">", 0.6, "colourfulness above 0.6"),
    "saturated":     ("saturation", ">", 0.6, "saturation above 0.6"),
    "muted":         ("saturation", "<", 0.4, "saturation below 0.4"),
    "flat":          ("contrast", "<", 0.35, "contrast below 0.35"),
    "contrasty":     ("contrast", ">", 0.65, "contrast above 0.65"),
    "energetic":     ("visual_energy", ">", 0.6, "visual energy above 0.6"),
    "high energy":   ("visual_energy", ">", 0.6, "visual energy above 0.6"),
    "calm":          ("visual_energy", "<", 0.4, "visual energy below 0.4"),
    "quiet":         ("visual_energy", "<", 0.4, "visual energy below 0.4"),
    "moving":        ("subject_motion", ">", 3.0, "subject motion above 3.0"),
    "action":        ("subject_motion", ">", 4.0, "subject motion above 4.0"),
}

QUALITY_TERMS = {
    "sharp":      ("sharpness", ">", 60.0, "sharpness above 60"),
    "soft":       ("sharpness", "<", 30.0, "sharpness below 30"),
    "blurry":     ("sharpness", "<", 20.0, "sharpness below 20"),
    "noisy":      ("noise", ">", 0.12, "noise above the treatable threshold"),
    "grainy":     ("noise", ">", 0.12, "noise above the treatable threshold"),
    "clean":      ("noise", "<", 0.05, "noise below 0.05"),
    "compressed": ("blockiness", ">", 0.35, "visible compression blocking"),
    "high quality": ("technical_quality", ">", 0.6, "technical quality above 0.6"),
    "good quality": ("technical_quality", ">", 0.6, "technical quality above 0.6"),
    "poor quality": ("technical_quality", "<", 0.35, "technical quality below 0.35"),
    "bad quality":  ("technical_quality", "<", 0.35, "technical quality below 0.35"),
    "usable":       ("technical_quality", ">", 0.45, "technical quality above 0.45"),
}

FLAG_TERMS = {
    "with faces": "faces", "faces": "faces", "person": "faces",
    "people": "faces", "someone": "faces", "portrait": "faces",
    "talking": "speech", "speaking": "speech", "dialogue": "speech",
    "dialog": "speech", "speech": "speech", "interview": "speech",
    "voice": "speech", "talking head": "speech",
}

SORT_TERMS = {
    "best": ("technical_quality", True), "highest quality": ("technical_quality", True),
    "best quality": ("technical_quality", True),
    "sharpest": ("sharpness", True), "brightest": ("brightness", True),
    "darkest": ("brightness", False), "longest": ("duration", True),
    "shortest": ("duration", False), "most energetic": ("visual_energy", True),
    "calmest": ("visual_energy", False),
}

NEGATIONS = ("no ", "not ", "without ", "excluding ", "avoid ", "non-")


@dataclass
class Criterion:
    """One measured condition, kept inspectable so a match can be explained."""
    attribute: str
    op: str
    value: object
    description: str
    negated: bool = False

    def describe(self) -> str:
        return ("not " if self.negated else "") + self.description

    def to_dict(self):
        return asdict(self)


@dataclass
class StructuredQuery:
    text: str = ""
    criteria: list = field(default_factory=list)
    sort_by: str | None = None
    sort_desc: bool = True
    limit: int = 10
    unmatched_terms: list = field(default_factory=list)
    semantic: bool = False       # never true in this build; see module docstring

    @property
    def is_empty(self) -> bool:
        return not self.criteria and self.sort_by is None

    def describe(self) -> str:
        parts = [c.describe() for c in self.criteria]
        if self.sort_by:
            parts.append(f"sorted by {self.sort_by} "
                         f"{'descending' if self.sort_desc else 'ascending'}")
        return "; ".join(parts) if parts else "no understood criteria"

    def to_dict(self):
        d = asdict(self)
        d["criteria"] = [c.to_dict() for c in self.criteria]
        return d


def _phrases(term_map):
    """Longest phrases first, so 'close up' wins over 'up'."""
    return sorted(term_map.items(), key=lambda kv: -len(kv[0]))


def _negated_at(text: str, pos: int) -> bool:
    """Is the term at `pos` negated, within its own clause?

    Clause-scoped, for the reason found in Phase 1: a negation in one clause was
    inverting a term in the next ('no faces, dark' must not mean 'not dark').
    """
    clause_start = max(text.rfind(",", 0, pos), text.rfind(" and ", 0, pos),
                       text.rfind(";", 0, pos), 0)
    window = text[clause_start:pos]
    return any(n in window for n in NEGATIONS)


def parse_query(text: str, limit: int = 10) -> StructuredQuery:
    """Turn a natural-language request into typed, measurable criteria."""
    q = StructuredQuery(text=text or "", limit=limit)
    if not text:
        return q
    t = " " + text.lower().strip() + " "
    consumed = [False] * len(t)

    def take(start, end):
        for i in range(start, end):
            consumed[i] = True

    def find_terms(term_map, handler):
        for phrase, val in _phrases(term_map):
            idx = t.find(" " + phrase + " ")
            if idx < 0:
                # also match when followed by punctuation
                for sep in (",", ".", ";"):
                    idx = t.find(" " + phrase + sep)
                    if idx >= 0:
                        break
            if idx < 0:
                continue
            s0, s1 = idx + 1, idx + 1 + len(phrase)
            if any(consumed[s0:s1]):
                continue
            take(s0, s1)
            handler(phrase, val, _negated_at(t, s0))

    def add(attr, op, value, desc, neg):
        # Several phrases map to the same measurement ("talking", "interview",
        # "talking head" all mean speech). Recording it once keeps the score
        # denominator honest — otherwise a synonym pair would double-weight one
        # condition against everything else in the query.
        for c in q.criteria:
            if (c.attribute, c.op, str(c.value), c.negated) == \
                    (attr, op, str(value), neg):
                return
        q.criteria.append(Criterion(attr, op, value, desc, neg))

    find_terms(SORT_TERMS, lambda p, v, n: setattr(q, "sort_by", v[0])
               or setattr(q, "sort_desc", v[1]))
    find_terms(CAMERA_TERMS, lambda p, v, n: add(
        "camera_movement", "prefix" if v in ("pan", "tilt") else "==", v,
        f"camera movement is {v.replace('_', ' ')}", n))
    find_terms(SHOT_SIZE_TERMS, lambda p, v, n: add(
        "shot_size", "==", v, f"shot size is {v.replace('_', ' ')}", n))
    find_terms(FLAG_TERMS, lambda p, v, n: add(
        v, "flag", True, f"contains {v}", n))
    find_terms(QUALITY_TERMS, lambda p, v, n: add(v[0], v[1], v[2], v[3], n))
    find_terms(SCALAR_TERMS, lambda p, v, n: add(v[0], v[1], v[2], v[3], n))

    # Report what was NOT understood, so the caller can see the gap.
    stop = {"the", "a", "an", "of", "in", "on", "with", "and", "or", "to",
            "shots", "shot", "clips", "clip", "footage", "find", "show",
            "me", "any", "all", "some", "that", "is", "are", "no", "not",
            "without", "excluding", "avoid", "give", "get", "look", "for"}
    word, start = "", 0
    for i, ch in enumerate(t):
        if ch.isalnum() or ch == "-":
            if not word:
                start = i
            word += ch
        else:
            if word and not any(consumed[start:i]) and word not in stop:
                q.unmatched_terms.append(word)
            word = ""
    return q
