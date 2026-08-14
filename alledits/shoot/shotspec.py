"""Shoot assistant: planning shots that do not exist yet (Spec §5, §24).

The edit modes so far all answer "what can I make from this footage?". This one
answers the question that comes before it: "what should I go and film?"

A ShotSpec is a recording instruction — placement, height, distance, action,
duration, what not to do — plus, crucially, a **machine-checkable requirement**
so the system can later tell whether the footage actually arrived.

HONESTY BOUNDARY, and it runs right through this module.

A shot has two kinds of requirement:

  *measurable*  — camera movement, duration, motion, exposure, speech presence.
                  These reuse FIND's criteria, so coverage is checked against
                  signal that was actually measured.
  *semantic*    — "the ball rolls into frame", "tying boots". Verifying these
                  needs a vision-language model this build does not have.

The system therefore never claims a shot is "covered". It reports that the
measurable part matches and names the semantic part as unverified, leaving the
human to confirm. Claiming coverage on a camera-move match alone would send
someone into an edit believing they had a shot they never filmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ShotSpec:
    """One shot to record. Fields follow Spec §5's checklist."""
    number: int
    name: str
    purpose: str = ""              # why the edit needs it
    edit_use: str = ""             # where it lands in the cut
    location: str = ""
    camera_position: str = ""      # where to put the phone/camera
    height: str = ""
    distance: str = ""
    subject_position: str = ""
    action: str = ""
    gaze: str = ""
    movement: str = ""             # camera movement
    speed: str = ""
    duration: float = 2.0          # seconds to RECORD (longer than the cut)
    settings: str = ""
    do_not: list = field(default_factory=list)
    # --- machine-checkable part ---
    match_query: str = ""          # FIND query describing the measurable shape
    semantic_content: str = ""     # what a model would have to recognise
    optional: bool = False

    def to_dict(self):
        return asdict(self)

    def instructions(self, skill: str = "intermediate") -> str:
        """Human-readable direction, adapted to how much the person knows."""
        beginner = (skill or "").lower().startswith("begin")
        lines = [f"Shot {self.number} — {self.name}"]
        if self.purpose:
            lines.append(f"  Why: {self.purpose}")
        placement = [x for x in (self.camera_position, self.height,
                                 self.distance) if x]
        if placement:
            lines.append("  Camera: " + "; ".join(placement))
        if self.subject_position:
            lines.append(f"  Where you stand: {self.subject_position}")
        if self.action:
            lines.append(f"  Do: {self.action}")
        if self.gaze:
            lines.append(f"  Look: {self.gaze}")
        if self.movement or self.speed:
            lines.append("  Camera move: "
                         + ", ".join(x for x in (self.movement, self.speed) if x))
        lines.append(f"  Record for about {self.duration:.0f} seconds "
                     "(longer than the cut needs, so there is room to trim)")
        if self.settings and not beginner:
            lines.append(f"  Settings: {self.settings}")
        for d in self.do_not:
            lines.append(f"  DO NOT: {d}")
        if self.edit_use:
            lines.append(f"  This becomes: {self.edit_use}")
        return "\n".join(lines)


# --------------------------------------------------------------- sequences
# Named narrative shapes. Each entry is a builder taking the style grammar, so
# the SAME sequence produces different shot lengths and moves for a slow
# cinematic reference than for a fast one — the spec's requirement that the plan
# depends on the reference, not on a fixed template.

def _hold(grammar, intensity: float, floor: float = 1.5) -> float:
    """How long to RECORD a shot whose cut length the style implies.

    Always longer than the edit needs: an in-point has to be chooseable, and a
    clip trimmed to exactly its cut length has no handles.
    """
    try:
        cut = float(grammar.target_shot_duration(intensity))
    except Exception:
        cut = 1.0
    return round(max(floor, cut * 3.0 + 1.0), 1)


def _skill_sequence(grammar) -> list:
    """Spec §24's worked example: the user as subject, a skill demonstration."""
    return [
        ShotSpec(
            number=1, name="Approach", purpose="establishes you and the place",
            edit_use="the opening shot",
            camera_position="on the ground, lens pointing slightly up",
            height="ankle height", distance="about six steps away",
            subject_position="start out of frame, walk in from the left",
            action="walk toward the camera at a normal pace",
            gaze="ahead, not at the lens", movement="locked off", speed="steady",
            duration=_hold(grammar, 0.3),
            settings="highest resolution; 60fps if available so it can be slowed",
            do_not=["do not look at the camera", "do not rush the walk"],
            match_query="static shot",
            semantic_content="a person walking toward camera"),
        ShotSpec(
            number=2, name="Detail", purpose="a quiet beat before the action",
            edit_use="cutaway that lets the edit breathe",
            camera_position="close, just off to one side",
            height="the height of the detail itself",
            distance="an arm's length", subject_position="crouched, hands in frame",
            action="do the small thing slowly — adjust a lace, set the ball",
            gaze="down at your hands", movement="locked off", speed="still",
            duration=_hold(grammar, 0.2),
            do_not=["do not let your hands leave the frame",
                    "do not move the camera while recording"],
            match_query="static sharp shot",
            semantic_content="hands performing a small deliberate action"),
        ShotSpec(
            number=3, name="Object enters", purpose="motion that can carry a cut",
            edit_use="the transition into the main action",
            camera_position="low, on the ground",
            height="just above ground level", distance="two steps back",
            subject_position="out of frame",
            action="roll the ball past the lens, close to it",
            movement="locked off", speed="the object moves, not the camera",
            duration=_hold(grammar, 0.6, floor=2.0),
            do_not=["do not hit the lens", "do not stop recording as it passes"],
            match_query="moving shot",
            semantic_content="an object crossing frame close to the lens"),
        ShotSpec(
            number=4, name="The action", purpose="the reason the video exists",
            edit_use="the peak of the edit, on the drop",
            camera_position="side on, so the movement crosses the frame",
            height="waist height", distance="three or four steps",
            subject_position="centre frame, room on both sides",
            action="perform the move once, committed, at full speed",
            gaze="on what you are doing", movement="locked off, or a slow pan "
                 "following you", speed="whatever the move needs",
            duration=_hold(grammar, 1.0, floor=3.0),
            settings="60fps or higher — this is the shot most likely to be slowed",
            do_not=["do not cut the recording the instant the move ends",
                    "do not drift out of frame"],
            match_query="energetic moving shot",
            semantic_content="the skill or action being performed"),
        ShotSpec(
            number=5, name="Toward the lens", purpose="a whip into the last beat",
            edit_use="transition out of the peak",
            camera_position="low and close",
            height="knee height", distance="two steps",
            subject_position="facing the camera",
            action="send the ball or your hand toward the lens and stop short",
            movement="locked off", speed="fast",
            duration=_hold(grammar, 0.9, floor=2.0),
            do_not=["do not actually hit the camera",
                    "do not slow down before the lens"],
            match_query="energetic moving shot",
            semantic_content="something travelling toward the lens"),
        ShotSpec(
            number=6, name="Release", purpose="somewhere for the edit to land",
            edit_use="the closing shot",
            camera_position="further back, whole body in frame",
            height="chest height", distance="six or seven steps",
            subject_position="centre, with headroom",
            action="react — walk away, celebrate, or simply stop",
            gaze="anywhere but the lens", movement="locked off or a slow pull back",
            speed="unhurried", duration=_hold(grammar, 0.5, floor=2.5),
            do_not=["do not end on a static pose held too long"],
            match_query="static shot",
            semantic_content="a person reacting after the action"),
    ]


def _arrival_sequence(grammar) -> list:
    """A generic place-and-person shape: establish, detail, subject, leave."""
    return [
        ShotSpec(number=1, name="Establish the place", purpose="tells the viewer where they are",
                 edit_use="opening", camera_position="wide, somewhere elevated if possible",
                 height="eye level or above", distance="as far back as the space allows",
                 action="let the place sit still, or pan slowly across it",
                 movement="locked off or a slow pan", speed="slow",
                 duration=_hold(grammar, 0.25, floor=3.0),
                 do_not=["do not pan quickly", "do not include your own shadow"],
                 match_query="static wide shot",
                 semantic_content="an establishing view of a location"),
        ShotSpec(number=2, name="A detail of the place", purpose="texture and specificity",
                 edit_use="cutaway", camera_position="close to the detail",
                 height="the height of the subject", distance="within arm's reach",
                 action="hold on one small thing", movement="locked off", speed="still",
                 duration=_hold(grammar, 0.2), do_not=["do not shake the camera"],
                 match_query="static sharp shot", semantic_content="a close detail"),
        ShotSpec(number=3, name="The subject arrives", purpose="puts a person in the place",
                 edit_use="the turn into the main body",
                 camera_position="side on", height="chest height", distance="four steps",
                 subject_position="enter frame from one side",
                 action="walk in and stop", gaze="into the space, not the lens",
                 movement="locked off", speed="steady", duration=_hold(grammar, 0.5),
                 do_not=["do not enter and exit within the same take"],
                 match_query="static shot", semantic_content="a person entering the frame"),
        ShotSpec(number=4, name="Movement", purpose="energy in the middle of the edit",
                 edit_use="the busiest section",
                 camera_position="handheld, following", height="chest height",
                 distance="close enough to feel involved",
                 action="move with the subject", movement="handheld follow", speed="walking",
                 duration=_hold(grammar, 0.9, floor=3.0),
                 do_not=["do not fight the wobble — let it be handheld"],
                 match_query="handheld moving shot", semantic_content="following a subject"),
        ShotSpec(number=5, name="Leave", purpose="an ending",
                 edit_use="closing", camera_position="wide again",
                 height="eye level", distance="far", action="let the subject exit frame",
                 movement="locked off", speed="still", duration=_hold(grammar, 0.4, floor=2.5),
                 do_not=["do not cut before the frame is empty"],
                 match_query="static wide shot", semantic_content="a subject leaving frame"),
    ]


SEQUENCES = {
    "skill": ("A skill or action demonstration with you as the subject "
              "(Spec §24's worked example)", _skill_sequence),
    "arrival": ("Arriving somewhere: establish, detail, subject, movement, leave",
                _arrival_sequence),
}


def build_sequence(name: str, grammar) -> list:
    """Instantiate a named sequence against a style, so shot LENGTHS follow it."""
    if name not in SEQUENCES:
        raise KeyError(f"unknown sequence {name!r}; "
                       f"available: {', '.join(sorted(SEQUENCES))}")
    return SEQUENCES[name][1](grammar)
