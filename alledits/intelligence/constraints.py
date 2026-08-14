"""CreativeConstraints — the typed contract between a natural-language brief and
every editorial decision the system makes (Spec §5, §23).

This is deliberately a DATA structure, not prose. The brief is interpreted once,
into measurable knobs; from then on the planner, selector, builder, renderer and
critic all read the same numbers. That is what makes "cinematic and restrained"
versus "chaotic and fast" produce provably different timelines rather than the
same edit with different marketing.

Every field has a neutral default equal to "no opinion", so an empty brief
reproduces the pre-Phase-1 behaviour exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

CONSTRAINTS_VERSION = "1.0.0"


@dataclass
class CreativeConstraints:
    # ---- pacing -------------------------------------------------------------
    pacing_multiplier: float = 1.0      # <1 shorter shots (faster), >1 longer
    min_shot_floor: float = 0.12        # hard floor on any shot length, seconds
    max_shot_ceiling: float = 4.0

    # ---- energy / intensity -------------------------------------------------
    intensity_gain: float = 1.0         # scales the reference intensity curve
    intensity_offset: float = 0.0       # shifts it
    motion_preference: float = 0.0      # -1 prefer stillness .. +1 prefer kinetic

    # ---- selection ----------------------------------------------------------
    continuity_weight: float = 1.0      # >1 smooth/invisible cuts, <1 collision cuts
    quality_weight: float = 1.0         # >1 favour technically clean footage
    diversity: float = 0.0              # >0 forces visually distinct neighbours
    prefer_faces: float = 0.0           # -1 avoid .. +1 favour shots with people
    shot_size_preference: str | None = None   # close_up | medium | wide

    # ---- effects ------------------------------------------------------------
    effect_density: float = 0.5         # 0 none .. 1 heavy
    max_effects_per_clip: int = 4
    allow_shake: bool = True
    allow_speed_ramp: bool = True
    allow_grain: bool = False
    motion_effect_threshold: float = 0.42   # intensity above which motion is added

    # ---- transitions --------------------------------------------------------
    # multipliers applied to the reference's own transition shares
    transition_bias: dict = field(default_factory=lambda: {
        "cut": 1.0, "whip": 1.0, "flash": 1.0, "dissolve": 1.0})
    max_transition_share: float = 0.6

    # ---- look ---------------------------------------------------------------
    contrast_delta: float = 0.0         # added to the reference's grading direction
    saturation_delta: float = 0.0
    warmth_delta: float = 0.0

    # ---- provenance ---------------------------------------------------------
    brief: str = ""
    actor: str = "rule_based_brief_parser"
    is_llm: bool = False
    matched_terms: list = field(default_factory=list)   # [{term, field, delta}]
    notes: list = field(default_factory=list)
    version: str = CONSTRAINTS_VERSION

    # ------------------------------------------------------------------ helpers
    def clamp(self) -> "CreativeConstraints":
        """Keep every knob inside a range the rest of the system can honour.
        A brief must never be able to produce an unrenderable edit."""
        def c(v, lo, hi):
            return float(min(max(v, lo), hi))
        self.pacing_multiplier = c(self.pacing_multiplier, 0.35, 3.0)
        self.min_shot_floor = c(self.min_shot_floor, 0.10, 1.5)
        self.max_shot_ceiling = c(self.max_shot_ceiling, self.min_shot_floor + 0.1, 8.0)
        self.intensity_gain = c(self.intensity_gain, 0.3, 2.2)
        self.intensity_offset = c(self.intensity_offset, -0.5, 0.5)
        self.motion_preference = c(self.motion_preference, -1.0, 1.0)
        self.continuity_weight = c(self.continuity_weight, 0.2, 2.5)
        self.quality_weight = c(self.quality_weight, 0.3, 2.5)
        self.diversity = c(self.diversity, 0.0, 3.0)
        self.prefer_faces = c(self.prefer_faces, -1.0, 1.0)
        self.effect_density = c(self.effect_density, 0.0, 1.0)
        self.max_effects_per_clip = int(c(self.max_effects_per_clip, 0, 5))
        self.motion_effect_threshold = c(self.motion_effect_threshold, 0.0, 1.0)
        self.max_transition_share = c(self.max_transition_share, 0.0, 0.85)
        self.contrast_delta = c(self.contrast_delta, -0.4, 0.4)
        self.saturation_delta = c(self.saturation_delta, -0.6, 0.6)
        self.warmth_delta = c(self.warmth_delta, -0.6, 0.6)
        for k in list(self.transition_bias):
            self.transition_bias[k] = c(self.transition_bias[k], 0.0, 4.0)
        return self

    def summary(self) -> str:
        """One-line human-readable account of what the brief was taken to mean."""
        bits = []
        if self.pacing_multiplier < 0.85:
            bits.append(f"faster cutting (x{self.pacing_multiplier:.2f} shot length)")
        elif self.pacing_multiplier > 1.15:
            bits.append(f"slower cutting (x{self.pacing_multiplier:.2f} shot length)")
        if self.intensity_gain > 1.1 or self.intensity_offset > 0.05:
            bits.append("higher intensity")
        elif self.intensity_gain < 0.9 or self.intensity_offset < -0.05:
            bits.append("lower intensity")
        if self.motion_preference > 0.15:
            bits.append("kinetic shots")
        elif self.motion_preference < -0.15:
            bits.append("still shots")
        if self.continuity_weight > 1.15:
            bits.append("smooth continuity")
        elif self.continuity_weight < 0.85:
            bits.append("collision cutting")
        if self.effect_density > 0.65:
            bits.append("heavy effects")
        elif self.effect_density < 0.35:
            bits.append("restrained effects")
        strong = [k for k, v in self.transition_bias.items() if v > 1.4 and k != "cut"]
        if strong:
            bits.append("favours " + "/".join(strong))
        if self.transition_bias.get("cut", 1.0) > 1.4:
            bits.append("hard cuts")
        return "; ".join(bits) if bits else "no strong preferences detected"

    def to_dict(self):
        return asdict(self)
