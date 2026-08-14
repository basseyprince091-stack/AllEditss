"""Coverage: which planned shots does the library already have? (Spec §5)

The workflow the spec asks for is EDIT PLAN -> IDENTIFY MISSING SHOTS -> DIRECT
USER -> RECORD -> INSPECT -> APPROVE/RESHOOT. This module is the identify step,
and the inspect step for what comes back.

The central design decision is what "covered" is allowed to mean.

Matching runs against FIND's measured attributes, so it can confirm the shape of
a shot — static or moving, sharp or soft, energetic or calm, the right length.
It cannot confirm the CONTENT: no model here knows whether a ball rolled into
frame. So a shot is never reported as covered. It is reported as:

    MISSING        nothing in the library has the right shape — go and film it
    LIKELY         the shape matches; a human must confirm the content
    UNVERIFIABLE   the requirement is not measurable at all here

Telling someone a shot is covered when only its camera move matched would send
them into an edit believing they have footage they never shot. LIKELY is the
strongest honest claim, and it always names what the human still has to check.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..search.query import parse_query


MISSING = "missing"
LIKELY = "likely"
UNVERIFIABLE = "unverifiable"


@dataclass
class ShotCoverage:
    number: int
    name: str
    status: str
    candidates: list = field(default_factory=list)   # [{shot_id, score, why}]
    checked: list = field(default_factory=list)      # measurable criteria tested
    unchecked: str = ""                              # the semantic part
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CoverageReport:
    sequence: str
    shots: list = field(default_factory=list)

    @property
    def missing(self) -> list:
        return [s for s in self.shots if s.status == MISSING]

    @property
    def likely(self) -> list:
        return [s for s in self.shots if s.status == LIKELY]

    @property
    def unverifiable(self) -> list:
        return [s for s in self.shots if s.status == UNVERIFIABLE]

    @property
    def ready(self) -> bool:
        """Nothing left to film. Says nothing about content being right."""
        return not self.missing

    def summary(self) -> str:
        return (f"{len(self.likely)} likely present, {len(self.missing)} missing, "
                f"{len(self.unverifiable)} not checkable "
                f"({len(self.shots)} shots in the '{self.sequence}' sequence)")

    def to_dict(self):
        return {"sequence": self.sequence,
                "shots": [s.to_dict() for s in self.shots],
                "summary": self.summary(), "ready_to_edit": self.ready}


def _duration_ok(shot, spec) -> bool:
    """The library clip must be long enough to cut the planned shot from."""
    need = float(spec.duration) / 3.0     # the cut length the record time implies
    return float(getattr(shot, "duration", 0.0)) >= max(0.3, need * 0.6)


def assess_coverage(index, specs, sequence_name: str = "custom",
                    min_score: float = 0.999) -> CoverageReport:
    """Compare a shot plan against what is already in the library.

    `min_score` defaults to "every measurable criterion satisfied". A partial
    shape match is not evidence of a shot; it is evidence of a different shot.
    """
    report = CoverageReport(sequence=sequence_name)
    for spec in specs:
        q = parse_query(spec.match_query or "")
        if not q.criteria:
            report.shots.append(ShotCoverage(
                number=spec.number, name=spec.name, status=UNVERIFIABLE,
                unchecked=spec.semantic_content,
                reason=("nothing in this shot's requirement is measurable here, "
                        "so the library cannot be checked against it")))
            continue

        hits = index.search(q, top_k=5)
        inert = list(getattr(index, "last_inert_criteria", []) or [])
        checked = [c.describe() for c in q.criteria
                   if c.describe() not in inert]
        good = [h for h in hits
                if h["score"] >= min_score and _duration_ok(h["shot"], spec)]

        if not checked:
            report.shots.append(ShotCoverage(
                number=spec.number, name=spec.name, status=UNVERIFIABLE,
                unchecked=spec.semantic_content,
                reason=("every requirement for this shot is inert on this "
                        f"library ({'; '.join(inert)})")))
            continue

        if not good:
            near = hits[0]["score"] if hits else 0.0
            report.shots.append(ShotCoverage(
                number=spec.number, name=spec.name, status=MISSING,
                checked=checked, unchecked=spec.semantic_content,
                reason=(f"no clip satisfies {len(checked)} measured "
                        f"requirement(s); closest match met {near:.0%} of them")))
            continue

        report.shots.append(ShotCoverage(
            number=spec.number, name=spec.name, status=LIKELY,
            checked=checked, unchecked=spec.semantic_content,
            candidates=[{"shot_id": h["shot"].id,
                         "score": round(h["score"], 3),
                         "why": "; ".join(h["matched"])} for h in good[:3]],
            reason=("the measurable shape matches; whether it actually shows "
                    f"\"{spec.semantic_content}\" needs a human eye — this build "
                    "has no model that can confirm content")))
    return report


def inspect_recording(index, spec, shot_id: str) -> dict:
    """The INSPECT step: judge a newly recorded clip against its own spec.

    Reports technical problems, which ARE measurable, and abstains on content.
    """
    shot = next((s for s in index.shots if s.id == shot_id), None)
    if shot is None:
        return {"verdict": "unknown", "reason": f"no shot {shot_id!r} in the index"}

    q = parse_query(spec.match_query or "")
    problems, met = [], []
    for c in q.criteria:
        ok = index._passes(index._attr(shot, c.attribute), c)
        if c.negated:
            ok = not ok
        (met if ok else problems).append(c.describe())

    quality = shot.quality or {}
    handling = quality.get("handling", "use")
    defects = [d["defect"] for d in (quality.get("defects") or [])]
    too_short = not _duration_ok(shot, spec)

    verdict = "approve"
    advice = []
    if handling == "reject":
        verdict = "reshoot"
        advice.append("technically unusable: " + "; ".join(quality.get("reasons", [])))
    elif handling == "use_briefly":
        # Salvage-grade footage can only be held for a fraction of a second, so
        # it cannot serve a shot the plan wants on screen for seconds. Approving
        # it would tell someone they had the shot when they have a flash frame.
        cap = quality.get("max_useful_duration")
        verdict = "reshoot"
        advice.append(
            f"usable only briefly (up to {float(cap):.2f}s as a flash or montage "
            f"cut) but this shot is planned to be held — reshoot it if you want "
            "it to carry the moment" if cap else
            "usable only briefly, but this shot is planned to be held")
    elif problems:
        verdict = "reshoot"
        advice.append("does not match the direction: " + "; ".join(problems))
    if too_short:
        verdict = "reshoot"
        advice.append(f"too short — record about {spec.duration:.0f}s so there "
                      "is room to trim")
    if defects and verdict == "approve":
        # Fixable defects are not a reshoot: rescue handles them (Spec §13).
        advice.append("has fixable issues (" + ", ".join(defects)
                      + ") which enhancement will treat — no reshoot needed")

    return {"verdict": verdict, "shot_id": shot_id, "met": met,
            "problems": problems, "advice": advice,
            "content_unverified": spec.semantic_content,
            "note": ("content was not checked — no model here can confirm what "
                     "the footage shows")}
