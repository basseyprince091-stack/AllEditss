"""Project state and human override (Spec §14, §16, §24).

Two problems this solves.

1. Nothing persisted. Every run started cold, so a user could never come back to
   an edit, and the critique loop's improvements died with the process.

2. The system could explain a decision but the user could not act on it.
   Explainability without override is narration. A user must be able to say
   "not that clip", "always this clip here", "keep this exact cut" — and have
   the system honour it through subsequent re-plans and revisions.

Overrides are stored as DIRECTIVES against slots and shots, not as edits to a
rendered timeline. That matters: the critique loop re-plans and rebuilds, and a
directive expressed as "clip X is banned from slot 3" survives that, whereas a
hand-patched timeline would be silently discarded on the next rebuild.

Precedence is absolute: a human directive outranks the model, the rule-based
planner, and the critic. The critic may not revise away a pinned choice.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from ..core.ids import new_id

PROJECT_VERSION = "1.0.0"


class DirectiveKind(str, Enum):
    REJECT_SHOT = "reject_shot"        # never use this shot anywhere
    REJECT_AT_SLOT = "reject_at_slot"  # don't use this shot in this slot
    PIN_SHOT = "pin_shot"              # this slot MUST use this shot
    LOCK_CLIP = "lock_clip"            # freeze this clip's timing and content
    FORCE_TRANSITION = "force_transition"
    FORCE_EFFECTS = "force_effects"    # exact effect list for a clip
    BAN_EFFECT = "ban_effect"          # never apply this effect type
    SET_DURATION = "set_duration"      # this slot holds for exactly N seconds


@dataclass
class Directive:
    kind: str
    slot_index: int | None = None
    shot_id: str | None = None
    value: object = None
    note: str = ""                      # the user's own words, if given
    # --- content anchor (see OverrideSet.bind) ---
    # A slot index alone is a fragile handle: change the brief or the duration
    # and "slot 7" is a different moment, so a directive silently lands on
    # footage the human never looked at. The anchor records WHAT was on screen
    # when the note was given, so the directive can find that moment again.
    anchor_shot_id: str | None = None
    anchor_t: float | None = None       # timeline position when the note was made
    anchor_status: str = "unanchored"   # unanchored | exact | moved | lost
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: new_id("dir"))

    def to_dict(self):
        return asdict(self)


class OverrideSet:
    """The human's accumulated instructions. Consulted by selection and building."""

    def __init__(self, directives=None):
        self.directives: list[Directive] = list(directives or [])

    # -------------------------------------------------------------- mutation
    def add(self, kind, **kw) -> Directive:
        d = Directive(kind=kind.value if isinstance(kind, DirectiveKind) else kind, **kw)
        # a pin replaces any previous pin for the same slot
        if d.kind == DirectiveKind.PIN_SHOT.value:
            self.directives = [x for x in self.directives
                               if not (x.kind == d.kind and x.slot_index == d.slot_index)]
        self.directives.append(d)
        return d

    SLOT_KINDS = ("lock_clip", "force_transition", "force_effects",
                  "set_duration", "reject_at_slot", "pin_shot")

    def anchor_to(self, directive, clips) -> "Directive":
        """Record what was on screen, so this directive can be re-found later."""
        for c in clips or []:
            try:
                idx = int(str(c.id).lstrip("c"))
            except (ValueError, AttributeError):
                continue
            if idx == directive.slot_index:
                directive.anchor_shot_id = getattr(c, "source_id", None)
                directive.anchor_t = float(getattr(c, "timeline_start", 0.0))
                directive.anchor_status = "exact"
                break
        return directive

    def bind(self, clips) -> dict:
        """Re-resolve anchored directives against a freshly planned timeline.

        Returns {status: [descriptions]} so the caller can TELL THE USER what
        moved and what could not be found. A directive that quietly retargets
        is worse than one that fails: the human sees an unrelated clip change
        and cannot tell why.

        Matching order:
          1. the same shot at the same slot        -> exact
          2. the same shot somewhere else          -> moved (follow it)
          3. the shot is gone from the edit        -> lost (do not apply)
        """
        report = {"exact": [], "moved": [], "lost": []}
        by_slot, by_shot = {}, {}
        for c in clips or []:
            try:
                idx = int(str(c.id).lstrip("c"))
            except (ValueError, AttributeError):
                continue
            by_slot[idx] = c
            by_shot.setdefault(getattr(c, "source_id", None), []).append((idx, c))

        for d in self.directives:
            if d.kind not in self.SLOT_KINDS or not d.anchor_shot_id:
                continue                      # unanchored: legacy slot behaviour
            here = by_slot.get(d.slot_index)
            if here is not None and getattr(here, "source_id", None) == d.anchor_shot_id:
                d.anchor_status = "exact"
                report["exact"].append(f"{d.kind} on slot {d.slot_index}")
                continue
            candidates = by_shot.get(d.anchor_shot_id) or []
            if candidates:
                # follow the shot to wherever it landed, nearest to where it was
                target_t = d.anchor_t if d.anchor_t is not None else 0.0
                idx, _ = min(candidates,
                             key=lambda ic: abs(getattr(ic[1], "timeline_start", 0.0)
                                                - target_t))
                if idx != d.slot_index:
                    report["moved"].append(
                        f"{d.kind}: {d.anchor_shot_id} moved from slot "
                        f"{d.slot_index} to {idx}")
                    d.slot_index = idx
                d.anchor_status = "moved"
            else:
                d.anchor_status = "lost"
                report["lost"].append(
                    f"{d.kind}: {d.anchor_shot_id} is no longer in the edit, "
                    "so this directive was not applied")
        return report

    def _live(self, d) -> bool:
        """A directive whose anchor was lost must not fire on other footage."""
        return d.anchor_status != "lost"

    def remove(self, directive_id: str) -> bool:
        n = len(self.directives)
        self.directives = [d for d in self.directives if d.id != directive_id]
        return len(self.directives) != n

    def clear(self):
        self.directives = []

    # --------------------------------------------------------------- queries
    def banned_shots(self) -> set:
        return {d.shot_id for d in self.directives
                if d.kind == DirectiveKind.REJECT_SHOT.value and d.shot_id}

    def banned_at(self, slot_index: int) -> set:
        return {d.shot_id for d in self.directives
                if d.kind == DirectiveKind.REJECT_AT_SLOT.value
                and d.slot_index == slot_index and d.shot_id}

    def pinned_at(self, slot_index: int) -> str | None:
        for d in reversed(self.directives):
            if (d.kind == DirectiveKind.PIN_SHOT.value
                    and d.slot_index == slot_index and self._live(d)):
                return d.shot_id
        return None

    def locked_slots(self) -> set:
        return {d.slot_index for d in self.directives
                if d.kind == DirectiveKind.LOCK_CLIP.value
                and d.slot_index is not None and self._live(d)}

    def forced_transition(self, slot_index: int):
        for d in reversed(self.directives):
            if (d.kind == DirectiveKind.FORCE_TRANSITION.value
                    and d.slot_index == slot_index and self._live(d)):
                return d.value
        return None

    def forced_effects(self, slot_index: int):
        for d in reversed(self.directives):
            if (d.kind == DirectiveKind.FORCE_EFFECTS.value
                    and d.slot_index == slot_index and self._live(d)):
                return d.value
        return None

    def duration_at(self, slot_index: int):
        """Forced duration for a slot, if the human set one. Last write wins."""
        vals = [d.value for d in self.directives
                if d.kind == DirectiveKind.SET_DURATION.value
                and d.slot_index == slot_index and self._live(d)]
        return float(vals[-1]) if vals else None

    def banned_effects(self) -> set:
        return {d.value for d in self.directives
                if d.kind == DirectiveKind.BAN_EFFECT.value and d.value}

    def allowed(self, shot, slot_index: int) -> bool:
        if shot.id in self.banned_shots():
            return False
        if shot.id in self.banned_at(slot_index):
            return False
        return True

    def describe(self) -> list:
        out = []
        for d in self.directives:
            where = f" at slot {d.slot_index}" if d.slot_index is not None else ""
            what = f" {d.shot_id}" if d.shot_id else (f" {d.value}" if d.value else "")
            out.append(f"{d.kind}{what}{where}" + (f" — \"{d.note}\"" if d.note else ""))
        return out

    def to_dict(self):
        return {"directives": [d.to_dict() for d in self.directives]}

    @classmethod
    def from_dict(cls, data):
        return cls([Directive(**d) for d in (data or {}).get("directives", [])])


@dataclass
class Project:
    """Everything needed to reopen an edit and continue working on it."""
    id: str = field(default_factory=lambda: new_id("proj"))
    name: str = "Untitled"
    version: str = PROJECT_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    brief: str = ""
    target_duration: float = 18.0
    project_settings: dict = field(default_factory=dict)

    clip_paths: list = field(default_factory=list)
    reference_path: str = ""
    music_path: str = ""

    constraints: dict = field(default_factory=dict)
    overrides: OverrideSet = field(default_factory=OverrideSet)
    timeline: dict | None = None
    master: dict | None = None   # delivery master + QC report, for auditability
    style_grammar_id: str = ""
    history: list = field(default_factory=list)     # [{t, event, detail}]

    def record(self, event: str, detail: str = ""):
        self.history.append({"t": time.time(), "event": event, "detail": detail})
        self.updated_at = time.time()

    def to_dict(self):
        d = asdict(self)
        d["overrides"] = self.overrides.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        data = dict(data)
        ov = OverrideSet.from_dict(data.pop("overrides", None))
        known = set(cls.__dataclass_fields__) - {"overrides"}
        return cls(**{k: v for k, v in data.items() if k in known}, overrides=ov)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        tmp.replace(path)               # atomic: never leave a half-written project
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        return cls.from_dict(json.loads(Path(path).read_text()))
