"""Brief interpretation: natural language -> CreativeConstraints (Spec §5, §23).

Two interpreters, one output schema:

  LLMBriefParser        uses the reasoning tier when a provider is configured.
                        Emits JSON conforming to the constraint schema, which is
                        then CLAMPED — a model cannot push the editor outside
                        renderable ranges.

  LexiconBriefParser    deterministic, always available, honestly labelled
                        `rule_based_brief_parser` (is_llm=False).

The lexicon is not keyword-spotting-as-theatre: it handles negation ("not too
fast", "no shake"), intensifiers ("very", "slightly"), and multi-word phrases,
and it records every term it matched so the interpretation is auditable. Terms
compose additively, so "fast but restrained" lands between the two rather than
snapping to whichever word came last.
"""
from __future__ import annotations

import json
import re

from .constraints import CreativeConstraints
from .providers.base import Tier
from ..core.errors import ProviderError, ProviderUnavailable

# ---------------------------------------------------------------------------
# Each entry: term -> dict of field deltas.
# Multiplicative fields use "*" prefix, additive use "+".
# ---------------------------------------------------------------------------
LEXICON: dict[str, dict] = {
    # ---- speed / pacing ----
    "fast": {"*pacing_multiplier": 0.62, "+intensity_offset": 0.10},
    "faster": {"*pacing_multiplier": 0.62, "+intensity_offset": 0.10},
    "quick": {"*pacing_multiplier": 0.70},
    "rapid": {"*pacing_multiplier": 0.58, "+intensity_offset": 0.10},
    "snappy": {"*pacing_multiplier": 0.68},
    # Comparatives and colloquialisms a director actually says. These were
    # missing, so DIRECTOR could route a note like "punch it up" to the brief
    # and the brief would quietly do nothing — the note appeared to be applied
    # and was not.
    "punch": {"*pacing_multiplier": 0.78, "+intensity_gain": 0.15,
              "+effect_density": 0.1},
    "punchy": {"*pacing_multiplier": 0.78, "+intensity_gain": 0.15,
               "+effect_density": 0.1},
    "calmer": {"*pacing_multiplier": 1.3, "+intensity_gain": -0.15,
               "+effect_density": -0.1},
    "warmer": {"+warmth_delta": 0.25},
    "colder": {"+warmth_delta": -0.25},
    "cooler": {"+warmth_delta": -0.25},
    # There is no brightness knob in CreativeConstraints, so "moodier" maps to
    # the levers that DO exist (contrast and saturation) rather than to an
    # invented one that would silently do nothing.
    "moodier": {"+contrast_delta": 0.15, "+saturation_delta": -0.12},
    "punchy": {"*pacing_multiplier": 0.70, "+effect_density": 0.10},
    "breakneck": {"*pacing_multiplier": 0.48, "+intensity_offset": 0.18},
    "frantic": {"*pacing_multiplier": 0.52, "+intensity_offset": 0.18, "*continuity_weight": 0.7},
    "slow": {"*pacing_multiplier": 1.65, "+intensity_offset": -0.10},
    "slower": {"*pacing_multiplier": 1.65, "+intensity_offset": -0.10},
    "unhurried": {"*pacing_multiplier": 1.7, "+intensity_offset": -0.12},
    "patient": {"*pacing_multiplier": 1.6, "+intensity_offset": -0.10},
    "languid": {"*pacing_multiplier": 1.8, "+intensity_offset": -0.15},
    "meditative": {"*pacing_multiplier": 1.9, "+intensity_offset": -0.20, "+motion_preference": -0.35},
    "lingering": {"*pacing_multiplier": 1.7, "+motion_preference": -0.2},

    # ---- energy ----
    "energetic": {"+intensity_offset": 0.15, "+motion_preference": 0.30},
    "high-energy": {"+intensity_offset": 0.18, "+motion_preference": 0.35, "*pacing_multiplier": 0.75},
    "aggressive": {"+intensity_offset": 0.20, "+motion_preference": 0.35,
                   "*continuity_weight": 0.65, "+effect_density": 0.18},
    "intense": {"*intensity_gain": 1.3, "+intensity_offset": 0.12},
    "explosive": {"+intensity_offset": 0.22, "+effect_density": 0.20, "*pacing_multiplier": 0.65},
    "chaotic": {"*continuity_weight": 0.45, "+diversity": 1.2, "+intensity_offset": 0.15,
                "*pacing_multiplier": 0.70, "+effect_density": 0.20},
    "frenetic": {"*pacing_multiplier": 0.55, "+intensity_offset": 0.18, "*continuity_weight": 0.6},
    "hype": {"+intensity_offset": 0.18, "+effect_density": 0.15, "*pacing_multiplier": 0.72},
    "calm": {"+intensity_offset": -0.18, "+motion_preference": -0.35, "*pacing_multiplier": 1.4},
    "gentle": {"+intensity_offset": -0.15, "+motion_preference": -0.30, "*pacing_multiplier": 1.35},
    "relaxed": {"+intensity_offset": -0.15, "*pacing_multiplier": 1.4},
    "serene": {"+intensity_offset": -0.20, "+motion_preference": -0.40, "*pacing_multiplier": 1.5},
    "moody": {"+contrast_delta": 0.10, "+saturation_delta": -0.12, "+intensity_offset": -0.05},
    "dreamy": {"*pacing_multiplier": 1.5, "+motion_preference": -0.25,
               "transition:dissolve": 2.2, "+saturation_delta": -0.05},

    # ---- restraint / polish ----
    "restrained": {"+effect_density": -0.28, "*continuity_weight": 1.35,
                   "max_effects_per_clip": 2, "allow_shake": False,
                   "transition:flash": 0.35, "transition:whip": 0.5},
    "minimal": {"+effect_density": -0.32, "max_effects_per_clip": 2, "allow_shake": False,
                "transition:flash": 0.25, "transition:whip": 0.35},
    "understated": {"+effect_density": -0.28, "*continuity_weight": 1.3, "allow_shake": False,
                    "transition:flash": 0.35, "transition:whip": 0.5},
    "subtle": {"+effect_density": -0.25, "*continuity_weight": 1.25,
               "transition:flash": 0.4},
    "clean": {"+effect_density": -0.20, "*quality_weight": 1.35, "allow_grain": False},
    "polished": {"*quality_weight": 1.4, "+effect_density": -0.10},
    "raw": {"*quality_weight": 0.65, "allow_grain": True, "+effect_density": 0.05},
    "gritty": {"*quality_weight": 0.6, "allow_grain": True, "+contrast_delta": 0.12,
               "+saturation_delta": -0.15},
    "maximal": {"+effect_density": 0.32, "max_effects_per_clip": 5},
    "flashy": {"+effect_density": 0.28, "transition:flash": 2.0},

    # ---- style / genre ----
    "cinematic": {"*pacing_multiplier": 1.30, "*continuity_weight": 1.45,
                  "+contrast_delta": 0.10, "+effect_density": -0.12,
                  "+motion_preference": -0.10, "transition:flash": 0.5},
    "filmic": {"*pacing_multiplier": 1.25, "*continuity_weight": 1.4, "+contrast_delta": 0.08},
    "documentary": {"*continuity_weight": 1.2, "+effect_density": -0.25,
                    "*quality_weight": 0.9, "allow_shake": False},
    "vlog": {"*pacing_multiplier": 0.9, "+prefer_faces": 0.6},
    "trailer": {"*intensity_gain": 1.35, "+effect_density": 0.15, "transition:flash": 1.8},
    "montage": {"*pacing_multiplier": 0.8, "+diversity": 0.8},
    "music video": {"*pacing_multiplier": 0.8, "+effect_density": 0.12},
    "commercial": {"*quality_weight": 1.4, "+effect_density": -0.05},
    "vintage": {"allow_grain": True, "+saturation_delta": -0.15, "+warmth_delta": 0.20},
    "retro": {"allow_grain": True, "+warmth_delta": 0.18, "+saturation_delta": -0.10},

    # ---- look ----
    "warm": {"+warmth_delta": 0.28},
    "cold": {"+warmth_delta": -0.28},
    "cool": {"+warmth_delta": -0.22},
    "vibrant": {"+saturation_delta": 0.28, "+contrast_delta": 0.08},
    "saturated": {"+saturation_delta": 0.28},
    "colourful": {"+saturation_delta": 0.25},
    "colorful": {"+saturation_delta": 0.25},
    "desaturated": {"+saturation_delta": -0.30},
    "muted": {"+saturation_delta": -0.25, "+contrast_delta": -0.05},
    "monochrome": {"+saturation_delta": -0.55},
    "bleak": {"+saturation_delta": -0.30, "+contrast_delta": 0.10},
    "high contrast": {"+contrast_delta": 0.22},
    "flat": {"+contrast_delta": -0.18},
    "dark": {"+contrast_delta": 0.12, "+saturation_delta": -0.08},
    "bright": {"+saturation_delta": 0.12},

    # ---- framing / subject ----
    "intimate": {"shot_size_preference": "close_up", "+prefer_faces": 0.7,
                 "*pacing_multiplier": 1.3},
    "close up": {"shot_size_preference": "close_up"},
    "close-up": {"shot_size_preference": "close_up"},
    "epic": {"shot_size_preference": "wide", "*pacing_multiplier": 1.25,
             "*intensity_gain": 1.2, "+contrast_delta": 0.10},
    "sweeping": {"shot_size_preference": "wide", "*pacing_multiplier": 1.35,
                 "+motion_preference": 0.15},
    "wide": {"shot_size_preference": "wide"},
    "people": {"+prefer_faces": 0.7},
    "faces": {"+prefer_faces": 0.8},
    "portrait": {"+prefer_faces": 0.7, "shot_size_preference": "close_up"},

    # ---- transitions ----
    "hard cuts": {"transition:cut": 2.2, "transition:dissolve": 0.2,
                  "transition:whip": 0.4, "transition:flash": 0.5},
    "hard cut": {"transition:cut": 2.2, "transition:dissolve": 0.2},
    "straight cuts": {"transition:cut": 2.2, "transition:dissolve": 0.2},
    "whip": {"transition:whip": 2.4},
    "whips": {"transition:whip": 2.4},
    "whip pan": {"transition:whip": 2.6},
    "flash": {"transition:flash": 2.2},
    "flashes": {"transition:flash": 2.2},
    "strobe": {"transition:flash": 2.6, "+effect_density": 0.15},
    "dissolve": {"transition:dissolve": 2.4},
    "dissolves": {"transition:dissolve": 2.4},
    "crossfade": {"transition:dissolve": 2.4},
    "smooth transitions": {"transition:dissolve": 1.8, "*continuity_weight": 1.3},
    "jump cuts": {"*continuity_weight": 0.4, "+diversity": 0.8},
    "seamless": {"*continuity_weight": 1.6, "transition:dissolve": 1.4},

    # ---- motion / camera ----
    "shake": {"allow_shake": True},
    "grain": {"allow_grain": True},
    "grainy": {"allow_grain": True, "+saturation_delta": -0.08},
    "effects": {"+effect_density": 0.15},
    "shaky": {"allow_shake": True, "+effect_density": 0.12},
    "handheld": {"allow_shake": True, "*quality_weight": 0.85},
    "smooth": {"allow_shake": False, "*continuity_weight": 1.3,
               "transition:whip": 0.4, "transition:flash": 0.4, "transition:dissolve": 1.6},
    "static": {"+motion_preference": -0.45, "allow_shake": False},
    "still": {"+motion_preference": -0.45},
    "kinetic": {"+motion_preference": 0.45},
    "dynamic": {"+motion_preference": 0.35},
    "movement": {"+motion_preference": 0.30},
}

INTENSIFIERS = {
    "very": 1.6, "extremely": 2.0, "really": 1.4, "super": 1.6, "ultra": 1.9,
    "highly": 1.5, "totally": 1.6, "insanely": 2.0, "incredibly": 1.8,
    "slightly": 0.45, "somewhat": 0.55, "a bit": 0.5, "a little": 0.45,
    "mildly": 0.5, "fairly": 0.75, "quite": 1.15, "rather": 0.9, "mostly": 1.1,
}
NEGATORS = {"not", "no", "never", "without", "avoid", "less", "isn't", "dont",
            "don't", "doesn't", "aren't", "minus", "except", "nothing"}

BOOL_FIELDS = {"allow_shake", "allow_speed_ramp", "allow_grain"}
STR_FIELDS = {"shot_size_preference"}
INT_FIELDS = {"max_effects_per_clip"}


def _tokenize(text: str):
    return re.findall(r"[a-z0-9'-]+", text.lower())


class LexiconBriefParser:
    """Deterministic brief interpretation. Honest about what it is."""
    name = "rule_based_brief_parser"
    is_llm = False

    def parse(self, brief: str) -> CreativeConstraints:
        c = CreativeConstraints(brief=brief or "", actor=self.name, is_llm=False)
        if not brief or not brief.strip():
            c.notes.append("empty brief — neutral constraints, reference style governs")
            return c.clamp()

        low = " " + brief.lower().strip() + " "
        tokens = _tokenize(brief)

        # longest phrases first so "hard cuts" wins over "cuts"
        for term in sorted(LEXICON, key=lambda t: -len(t)):
            pattern = r"(?<![a-z])" + re.escape(term) + r"(?![a-z])"
            for m in re.finditer(pattern, low):
                weight, negated = self._context(low, m.start(), tokens, term)
                if negated:
                    weight = -abs(weight)
                self._apply(c, term, LEXICON[term], weight)

        if not c.matched_terms:
            c.notes.append("no known style terms recognised — reference style governs")
        return c.clamp()

    # -------------------------------------------------------------- internals
    CLAUSE_BREAK = re.compile(r"[,.;:!?]|\b(?:and|but|then|with|however|while)\b")

    def _context(self, low: str, idx: int, tokens, term):
        """Look back for an intensifier and/or a negator, stopping at the clause
        boundary.

        Without the clause stop, "not too fast, very warm, hard cuts" leaks the
        negation from the first clause onto "warm" and inverts the colour
        direction — the brief asks for warm and gets cold.
        """
        window = low[max(0, idx - 60):idx]
        breaks = list(self.CLAUSE_BREAK.finditer(window))
        if breaks:
            window = window[breaks[-1].end():]
        words = _tokenize(window)[-3:]

        weight = 1.0
        joined = " ".join(words)
        for phrase, mult in INTENSIFIERS.items():
            if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", joined):
                weight *= mult
        negated = any(w in NEGATORS for w in words)
        # "not too fast" -> negate; "too fast" alone -> intensify
        if "too" in words and not negated:
            weight *= 1.25
        return weight, negated

    def _apply(self, c: CreativeConstraints, term, deltas: dict, weight: float):
        for key, val in deltas.items():
            if key.startswith("transition:"):
                t = key.split(":", 1)[1]
                cur = c.transition_bias.get(t, 1.0)
                # weight scales how far the bias moves away from neutral
                c.transition_bias[t] = max(0.0, 1.0 + (val - 1.0) * weight)
                c.matched_terms.append({"term": term, "field": f"transition_bias.{t}",
                                        "from": cur, "to": c.transition_bias[t]})
            elif key in BOOL_FIELDS:
                newv = bool(val) if weight > 0 else (not bool(val))
                setattr(c, key, newv)
                c.matched_terms.append({"term": term, "field": key, "to": newv})
            elif key in STR_FIELDS:
                if weight > 0:
                    setattr(c, key, val)
                    c.matched_terms.append({"term": term, "field": key, "to": val})
            elif key in INT_FIELDS:
                if weight > 0:
                    cur = getattr(c, key)
                    setattr(c, key, min(cur, int(val)))
                    c.matched_terms.append({"term": term, "field": key,
                                            "from": cur, "to": getattr(c, key)})
            elif key.startswith("*"):
                f = key[1:]
                cur = getattr(c, f)
                # move proportionally from 1.0 by the term's factor, scaled by weight
                factor = 1.0 + (float(val) - 1.0) * weight
                factor = max(0.05, factor)
                setattr(c, f, cur * factor)
                c.matched_terms.append({"term": term, "field": f,
                                        "from": cur, "to": getattr(c, f)})
            elif key.startswith("+"):
                f = key[1:]
                cur = getattr(c, f)
                setattr(c, f, cur + float(val) * weight)
                c.matched_terms.append({"term": term, "field": f,
                                        "from": cur, "to": getattr(c, f)})


CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {
        "pacing_multiplier": {"type": "number",
                              "description": "shot-length scale; <1 faster cutting, >1 slower"},
        "intensity_gain": {"type": "number"},
        "intensity_offset": {"type": "number"},
        "motion_preference": {"type": "number",
                              "description": "-1 prefer still shots, +1 prefer kinetic"},
        "continuity_weight": {"type": "number",
                              "description": ">1 smooth invisible cuts, <1 collision cutting"},
        "quality_weight": {"type": "number"},
        "diversity": {"type": "number"},
        "prefer_faces": {"type": "number"},
        "shot_size_preference": {"type": ["string", "null"],
                                 "enum": ["close_up", "medium", "wide", None]},
        "effect_density": {"type": "number", "description": "0 none .. 1 heavy"},
        "max_effects_per_clip": {"type": "integer"},
        "allow_shake": {"type": "boolean"},
        "allow_grain": {"type": "boolean"},
        "transition_bias": {"type": "object",
                            "description": "multipliers for cut/whip/flash/dissolve"},
        "contrast_delta": {"type": "number"},
        "saturation_delta": {"type": "number"},
        "warmth_delta": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["pacing_multiplier", "effect_density", "rationale"],
}

SYSTEM = """You are the creative director of a video editing system. You convert a
user's brief into measurable editing constraints.

You do not write prose about the video and you do not choose shots. You only set
the knobs that govern how the edit is assembled. Be decisive: if the brief says
"slow and restrained", pacing_multiplier should be clearly above 1 and
effect_density clearly low. If it says "chaotic and fast", the opposite.

Neutral values mean "no opinion": pacing_multiplier 1.0, intensity_gain 1.0,
intensity_offset 0, motion_preference 0, continuity_weight 1.0, effect_density
0.5, transition_bias all 1.0."""


class LLMBriefParser:
    """Uses the reasoning tier. Falls back explicitly, never silently."""
    name = "llm_brief_parser"
    is_llm = True

    def __init__(self, provider, fallback=None):
        self.provider = provider
        self.fallback = fallback or LexiconBriefParser()

    def parse(self, brief: str) -> CreativeConstraints:
        if not (self.provider and self.provider.available()):
            c = self.fallback.parse(brief)
            c.notes.append("no language model available; brief interpreted by the "
                           "rule-based parser")
            return c
        try:
            r = self.provider.complete(SYSTEM, f"Brief: {brief}", tier=Tier.REASONING,
                                       json_schema=CONSTRAINT_SCHEMA, max_tokens=1200)
        except (ProviderError, ProviderUnavailable) as e:
            c = self.fallback.parse(brief)
            c.notes.append(f"language model unavailable ({e}); used rule-based parser")
            return c

        data = r.data or {}
        c = CreativeConstraints(brief=brief, actor=r.actor, is_llm=True)
        for k, v in data.items():
            if k == "rationale":
                c.notes.append(str(v))
            elif k == "transition_bias" and isinstance(v, dict):
                c.transition_bias.update({str(a): float(b) for a, b in v.items()})
            elif hasattr(c, k) and v is not None:
                try:
                    cur = getattr(c, k)
                    setattr(c, k, type(cur)(v) if not isinstance(cur, (list, dict)) else v)
                except (TypeError, ValueError):
                    pass
        # A model may not push the editor outside renderable ranges.
        return c.clamp()


def parse_brief(brief: str, provider=None) -> CreativeConstraints:
    """Interpret a brief with the best available interpreter."""
    if provider is not None and getattr(provider, "available", lambda: False)() \
            and getattr(provider, "name", "") != "rule_based_planner":
        return LLMBriefParser(provider).parse(brief)
    return LexiconBriefParser().parse(brief)
