"""StyleGrammar — the structured, versioned representation of an editing language
extracted from a reference edit (Spec §9, §10).

RIGHTS NOTE (Spec §30): this stores measured *characteristics* — pacing statistics,
intensity curves, transition tendencies, grading direction. It stores no frames,
no audio and no content of the reference. It describes HOW something was edited,
which is what gets applied to the user's own footage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields

GRAMMAR_VERSION = "1.0.0"


@dataclass
class PacingProfile:
    cuts_per_second: float = 0.0
    mean_shot: float = 0.0
    median_shot: float = 0.0
    p10_shot: float = 0.0
    p90_shot: float = 0.0
    shot_duration_std: float = 0.0
    rhythm: str = "steady"          # steady | accelerating | decelerating | bursty
    fastest_window: tuple = (0.0, 0.0)
    duration_histogram: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class MotionProfile:
    mean_flow: float = 0.0
    motion_variance: float = 0.0
    dominant_moves: list = field(default_factory=list)   # [(move, share)]
    zoom_tendency: float = 0.0
    shake_level: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class ColorProfile:
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    warmth: float = 0.0
    colorfulness: float = 0.0
    key: str = "mid"
    palette: list = field(default_factory=list)
    black_level: float = 0.0
    highlight_level: float = 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class TransitionProfile:
    hard_cut_share: float = 1.0
    flash_share: float = 0.0
    whip_share: float = 0.0
    dissolve_share: float = 0.0
    mean_transition_duration: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class StyleGrammar:
    """Everything the builder needs to edit *in this style*."""
    id: str = ""
    version: str = GRAMMAR_VERSION
    source_label: str = ""              # user-facing name, not the protected work's content
    duration: float = 0.0
    pacing: PacingProfile = field(default_factory=PacingProfile)
    motion: MotionProfile = field(default_factory=MotionProfile)
    color: ColorProfile = field(default_factory=ColorProfile)
    transitions: TransitionProfile = field(default_factory=TransitionProfile)
    intensity_curve: list = field(default_factory=list)     # [{t, value}] normalized 0..1
    structure: list = field(default_factory=list)           # [{role, start, end}]
    effect_density: float = 0.0        # effects-per-shot tendency, 0..1
    beat_sync_strength: float = 0.0    # how tightly cuts landed on musical beats
    text_presence: float = 0.0
    pacing_multiplier: float = 1.0   # revision knob: <1 tightens, >1 loosens pacing
    notes: list = field(default_factory=list)

    def intensity_at(self, t_norm: float) -> float:
        """Intensity at a normalized (0..1) position in the edit."""
        if not self.intensity_curve:
            return 0.5
        pts = self.intensity_curve
        tn = min(max(t_norm, 0.0), 1.0)
        i = int(tn * (len(pts) - 1))
        return float(pts[i]["value"])

    def target_shot_duration(self, intensity: float) -> float:
        """Higher intensity -> shorter shots, bounded by the reference's own range."""
        m = max(0.35, float(self.pacing_multiplier))
        lo = max(0.12, (self.pacing.p10_shot or 0.3) * m)
        hi = max(lo + 0.05, (self.pacing.p90_shot or 1.2) * m)
        return float(hi - (hi - lo) * min(max(intensity, 0.0), 1.0))

    def to_dict(self):
        d = asdict(self)
        for k in ("pacing", "motion", "color", "transitions"):
            d[k] = getattr(self, k).to_dict()
        return d

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleGrammar":
        """Rebuild a grammar from its serialised form.

        Version is checked rather than assumed: a grammar written by an older
        extractor may be missing fields the builder now relies on, and silently
        filling them with defaults would present a stale style as a current one.
        """
        d = dict(d or {})
        ver = d.get("version", "")
        if ver and ver != GRAMMAR_VERSION:
            raise ValueError(
                f"style grammar version {ver!r} does not match this build's "
                f"{GRAMMAR_VERSION!r}; re-analyse the reference rather than "
                "loading a stale style")
        sub = {"pacing": PacingProfile, "motion": MotionProfile,
               "color": ColorProfile, "transitions": TransitionProfile}
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name in sub and isinstance(v, dict):
                klass = sub[f.name]
                known = {ff.name for ff in fields(klass)}
                v = klass(**{k: val for k, val in v.items() if k in known})
            kwargs[f.name] = v
        g = cls(**kwargs)
        # tuples survive a JSON round-trip as lists
        if isinstance(g.pacing.fastest_window, list):
            g.pacing.fastest_window = tuple(g.pacing.fastest_window)
        return g

    @classmethod
    def from_json(cls, text: str) -> "StyleGrammar":
        return cls.from_dict(json.loads(text))
