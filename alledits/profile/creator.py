"""Creator profile and adaptive guidance (Spec §4).

The same shot has to be described very differently to two people. "24-28mm
equivalent, low-angle forward tracking, 60fps, 2s pre/post-roll" is precise and
useless to someone holding a phone for the first time; "put your phone around
knee height, point it slightly up" is clear and patronising to a professional.

So instruction text is generated from the profile, not stored as one fixed
string. Three things drive it:

1. **Instruction level** — how much is spelled out, and whether jargon is used
   or translated.
2. **What the person already knows.** The spec asks the system to stop
   explaining concepts the user understands. Explanations are therefore tracked:
   once a concept has been explained enough times, it stops appearing. This is
   persisted, so it holds across sessions rather than resetting.
3. **What they actually have.** A shot requiring a tripod is not "advice" to
   someone without one — it is an instruction they cannot follow. Those shots
   are REDESIGNED (Spec §5: "if the requested shot is impossible, redesign it"),
   and the substitution is stated so nobody thinks they filmed the original.

Nothing here infers skill from behaviour. The profile holds what the person
told us; guessing that someone is a beginner because a clip was shaky would be
a judgement they never asked for.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


LEVELS = ("teach_me", "normal", "technical", "minimal")

EXPERIENCE = ("none", "beginner", "intermediate", "advanced", "professional")

# Concepts the system may need to explain. Explaining one more than this many
# times is nagging, so it stops.
EXPLAIN_LIMIT = 2

CONCEPT_NOTES = {
    "fps": "frames per second — higher lets the shot be slowed down smoothly later",
    "pre_post_roll": "keep recording a beat before and after the action, so the "
                     "cut has somewhere to land",
    "low_angle": "camera below eye line, which makes the subject look larger",
    "headroom": "space above the subject's head in frame",
    "handheld": "holding the camera rather than fixing it — the small movement "
                "reads as energy",
    "locked_off": "camera completely still, usually on a tripod or a solid surface",
    "tracking": "the camera moves with the subject rather than staying put",
    "white_balance": "how the camera reads the colour of the light",
}

# Jargon -> plain language, used when the level says to avoid it.
PLAIN = {
    "locked off": "completely still, resting on something solid",
    "locked-off": "completely still, resting on something solid",
    "pan": "turn the camera sideways",
    "tracking": "move along with the subject",
    "pre-roll": "start recording early",
    "post-roll": "keep recording after",
    # NOTE: "60fps" is deliberately NOT here. adapt_shot() already rewrites it
    # for the device, and a second pass rewrote the replacement's own wording,
    # producing "the highest frame-rate your phone offers (often the higher
    # frame-rate setting)". One substitution per term, at one stage.
    "headroom": "a little space above the head",
}


@dataclass
class CreatorProfile:
    name: str = "default"
    editing_experience: str = "beginner"
    filming_experience: str = "beginner"
    content_types: list = field(default_factory=list)
    instruction_level: str = "normal"
    known_concepts: list = field(default_factory=list)
    # --- what they actually have to film with ---
    device: str = "phone"              # phone | camera
    has_tripod: bool = False
    has_gimbal: bool = False
    lenses: list = field(default_factory=list)
    lighting: str = "available light"
    crew_size: int = 1                 # 1 = filming alone
    location: str = ""
    # --- adaptive state ---
    explained: dict = field(default_factory=dict)   # concept -> times explained

    # ------------------------------------------------------------- behaviour
    @property
    def wants_jargon(self) -> bool:
        return self.instruction_level in ("technical", "minimal")

    @property
    def wants_why(self) -> bool:
        """Beginners benefit from the reason; professionals asked for minimal."""
        return self.instruction_level in ("teach_me", "normal")

    def knows(self, concept: str) -> bool:
        if concept in self.known_concepts:
            return True
        return self.explained.get(concept, 0) >= EXPLAIN_LIMIT

    def explain(self, concept: str) -> str | None:
        """Return an explanation the FIRST few times, then stop.

        Spec §4: the system should stop repeatedly explaining concepts the user
        already understands. Silence here is the feature.
        """
        if self.instruction_level == "minimal":
            return None
        if concept not in CONCEPT_NOTES or self.knows(concept):
            return None
        self.explained[concept] = self.explained.get(concept, 0) + 1
        return CONCEPT_NOTES[concept]

    def plainify(self, text: str) -> str:
        if self.wants_jargon or not text:
            return text
        out = text
        for term, plain in PLAIN.items():
            out = re.sub(rf"\b{re.escape(term)}\b", plain, out, flags=re.I)
        return out

    # ----------------------------------------------------------- persistence
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CreatorProfile":
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path) -> "CreatorProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------- shot feasibility
@dataclass
class Redesign:
    changed: bool
    reason: str = ""
    substitution: str = ""

    def to_dict(self):
        return asdict(self)


def adapt_shot(spec, profile: CreatorProfile):
    """Make a shot filmable with the gear this person has.

    Returns (adapted_copy, Redesign). The original spec is never mutated, and a
    substitution is always stated: silently swapping a tracking shot for a
    static one would leave the person believing they filmed the shot that was
    planned.
    """
    import copy
    out = copy.deepcopy(spec)
    notes = []

    mv = (spec.movement or "").lower()
    needs_still = "locked" in mv or "still" in mv
    needs_move = ("pan" in mv or "track" in mv or "follow" in mv
                  or "pull back" in mv)

    if needs_still and not profile.has_tripod:
        out.movement = "rest the camera on something solid and do not touch it"
        out.speed = ""      # a still camera has no speed; keeping one contradicts it
        notes.append("you have no tripod, so the locked-off shot becomes a "
                     "braced camera — a wall, a bag or the ground all work")

    if needs_move and profile.crew_size < 2 and "handheld" not in mv:
        if profile.has_gimbal:
            out.movement = "handheld on the gimbal, moving slowly"
            notes.append("filming alone with a gimbal, so the move is handheld "
                         "rather than operated")
        else:
            out.movement = "locked off"
            out.speed = ""
            out.subject_position = (spec.subject_position
                                    or "") + " — move through the frame instead "\
                                    "of the camera moving"
            notes.append("no one else is there to operate a moving camera, so "
                         "the camera stays put and YOU provide the movement; "
                         "this reads differently, and is a real substitution")

    if "handheld" in mv and profile.crew_size < 2 and "follow" in mv:
        notes.append("following yourself is not possible alone — either ask "
                     "someone to hold the camera, or lock it off and walk "
                     "through the frame")

    if profile.device == "phone" and "60fps" in (spec.settings or ""):
        out.settings = (spec.settings or "").replace(
            "60fps", "the highest frame-rate your phone offers (often 60fps)")

    if profile.lighting.startswith("available") and "low" in (spec.name or "").lower():
        notes.append("in available light, keep the subject facing the brightest "
                     "part of the sky or window")

    return out, Redesign(changed=bool(notes), reason="; ".join(notes),
                         substitution=out.movement if notes else "")


def render_instructions(spec, profile: CreatorProfile) -> str:
    """The shot, described the way THIS person needs to hear it."""
    adapted, redesign = adapt_shot(spec, profile)
    lvl = profile.instruction_level
    P = profile.plainify

    if lvl == "minimal":
        bits = [f"{adapted.number}. {adapted.name}"]
        tech = [x for x in (adapted.movement, adapted.height, adapted.distance,
                            adapted.settings) if x]
        bits.append("   " + "; ".join(tech))
        bits.append(f"   {adapted.duration:.0f}s")
        if redesign.changed:
            bits.append(f"   [adapted: {redesign.reason}]")
        return "\n".join(bits)

    lines = [f"Shot {adapted.number} — {adapted.name}"]
    if profile.wants_why and adapted.purpose:
        lines.append(f"  Why: {adapted.purpose}")

    placement = [x for x in (adapted.camera_position, adapted.height,
                             adapted.distance) if x]
    if placement:
        lines.append("  Camera: " + P("; ".join(placement)))
    if adapted.subject_position:
        lines.append("  Where you stand: " + P(adapted.subject_position))
    if adapted.action:
        lines.append(f"  Do: {adapted.action}")
    if adapted.gaze and lvl == "teach_me":
        lines.append(f"  Look: {adapted.gaze}")
    if adapted.movement:
        lines.append("  Camera move: " + P(adapted.movement)
                     + (f", {adapted.speed}" if adapted.speed else ""))
        # Pick the concept from the ORIGINAL direction, not the adapted wording:
        # once "locked off" becomes "rest it on something solid" the word is gone,
        # and matching on the adapted text attached the tracking explanation to a
        # static shot — teaching the wrong thing.
        orig_mv = (spec.movement or "").lower()
        concept = ("locked_off" if ("lock" in orig_mv or "still" in orig_mv)
                   else "tracking" if ("track" in orig_mv or "follow" in orig_mv
                                       or "pan" in orig_mv)
                   else None)
        note = profile.explain(concept) if concept else None
        if note:
            lines.append(f"      ({note})")

    lines.append(f"  Record for about {adapted.duration:.0f} seconds")
    roll = profile.explain("pre_post_roll")
    if roll:
        lines.append(f"      ({roll})")

    if adapted.settings:
        # A beginner still needs to know to use the high frame rate — they just
        # need it in plain words. Omitting it entirely would cost them the shot.
        lines.append("  Settings: " + P(adapted.settings))
        note = profile.explain("fps")
        if note and "fps" in adapted.settings.lower():
            lines.append(f"      ({note})")

    for d in adapted.do_not:
        lines.append(f"  DO NOT: {d}")

    if redesign.changed:
        lines.append(f"  ADAPTED for your setup: {redesign.reason}")
    if profile.wants_why and adapted.edit_use:
        lines.append(f"  This becomes: {adapted.edit_use}")
    return "\n".join(lines)
