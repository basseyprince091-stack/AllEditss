"""ALLEDITS Timeline DSL — schema version 1.0.0  (Spec §11, Principle 5)

An LLM never emits rendering commands. It emits (or influences) this structure,
which is typed, versioned, validated, human-readable and human-editable.
The renderer is the only component that turns it into ffmpeg arguments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum

SCHEMA_VERSION = "1.0.0"


class EffectType(str, Enum):
    # MOTION
    PUSH_ZOOM = "push_zoom"
    PULL_ZOOM = "pull_zoom"
    DRIFT = "drift"
    SHAKE = "shake"
    VELOCITY_RAMP = "velocity_ramp"
    # BLUR
    DIRECTIONAL_BLUR = "directional_blur"
    GAUSSIAN_BLUR = "gaussian_blur"
    RADIAL_BLUR = "radial_blur"
    # COLOR
    COLOR_GRADE = "color_grade"
    # COMPOSITING / TEXTURE
    FILM_GRAIN = "film_grain"
    VIGNETTE = "vignette"
    GLOW = "glow"
    FLASH = "flash"
    # RESTORATION (applied only where a measured defect justifies it)
    DENOISE = "denoise"
    SHARPEN = "sharpen"
    DEBLOCK = "deblock"
    STABILIZE = "stabilize"
    EXPAND_CONTRAST = "expand_contrast"
    # FRAMING
    REFRAME = "reframe"


class TransitionType(str, Enum):
    CUT = "cut"
    WHIP = "whip"
    FLASH = "flash"
    DISSOLVE = "dissolve"
    ZOOM = "zoom"
    SLIDE = "slide"
    MATCH_MOVEMENT = "match_movement"


class SlotRole(str, Enum):
    HOOK = "hook"
    SETUP = "setup"
    ANTICIPATION = "anticipation"
    ESCALATION = "escalation"
    CLIMAX = "climax"
    RELEASE = "release"


@dataclass
class Effect:
    type: str
    params: dict = field(default_factory=dict)
    reason: str = ""            # why this effect is here — inspectable

    def to_dict(self):
        return asdict(self)


@dataclass
class Transition:
    type: str = TransitionType.CUT.value
    duration: float = 0.0       # seconds; 0 for a hard cut
    params: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TimelineClip:
    id: str
    source_id: str              # asset id in the media library
    source_path: str            # resolved at build time
    source_in: float            # in-point within the source
    source_out: float
    timeline_start: float       # position on the timeline (absolute, frame-quantized)
    duration: float = 0.0       # AUTHORITATIVE timeline duration.
    #   Deriving duration from (source_out-source_in)/speed accumulates rounding
    #   error across a long timeline and silently walks the edit off the beat.
    #   The plan owns the duration; the source range says what to read into it.
    #   If the source range is shorter, the renderer holds the last frame and the
    #   shortfall is recorded in `hold_extended`.
    speed: float = 1.0
    hold_extended: float = 0.0
    role: str = SlotRole.SETUP.value
    effects: list = field(default_factory=list)         # list[Effect]
    transition_in: Transition | None = None
    beat_locked: bool = False
    selection_reason: str = ""
    quality_handling: str = "use"
    treatments: list = field(default_factory=list)
    #   Restoration applied to this clip, for provenance (Spec §14): the output
    #   must never be presented as untouched camera original when it isn't.

    @property
    def source_duration(self) -> float:
        return max(0.0, self.source_out - self.source_in)

    @property
    def timeline_duration(self) -> float:
        if self.duration > 0:
            return self.duration
        return self.source_duration / max(self.speed, 1e-6)

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.timeline_duration

    def to_dict(self):
        d = asdict(self)
        d["effects"] = [e.to_dict() if isinstance(e, Effect) else e for e in self.effects]
        d["transition_in"] = (self.transition_in.to_dict()
                              if isinstance(self.transition_in, Transition)
                              else self.transition_in)
        d["timeline_duration"] = self.timeline_duration
        d["timeline_end"] = self.timeline_end
        return d


@dataclass
class AudioTrack:
    id: str
    source_path: str
    source_in: float = 0.0
    source_out: float = 0.0
    timeline_start: float = 0.0
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.5
    role: str = "music"          # music/sfx/voice/ambience
    duck_under: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class TextElement:
    id: str
    text: str
    start: float
    end: float
    style: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class ProjectSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    aspect_label: str = "9:16"
    color_space: str = "bt709"
    # Provenance labelling — Spec §14: never mislabel enhanced output as native.
    resolution_provenance: str = "native"   # native | ai_enhanced | interpolated | mixed
    fps_provenance: str = "native"

    def to_dict(self):
        return asdict(self)


@dataclass
class Timeline:
    schema_version: str = SCHEMA_VERSION
    project: ProjectSettings = field(default_factory=ProjectSettings)
    clips: list = field(default_factory=list)         # list[TimelineClip]
    audio: list = field(default_factory=list)
    mix: object = None          # MixPlan (Phase 4); supersedes `audio` when set         # list[AudioTrack]
    text: list = field(default_factory=list)          # list[TextElement]
    intent: str = ""                                  # the user's creative brief
    style_grammar_id: str = ""
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max([c.timeline_end for c in self.clips], default=0.0)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "project": self.project.to_dict(),
            "duration": self.duration,
            "intent": self.intent,
            "style_grammar_id": self.style_grammar_id,
            "clips": [c.to_dict() for c in self.clips],
            "audio": [a.to_dict() for a in self.audio],
            "text": [t.to_dict() for t in self.text],
            # The mix records what was decided AND what was measured, so a saved
            # project can be audited for what it actually delivered. Omitting it
            # silently discarded every sound decision on save.
            "mix": self.mix.to_dict() if self.mix is not None else None,
            "notes": self.notes,
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
