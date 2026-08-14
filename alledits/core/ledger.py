"""Decision ledger — every editorial decision is recorded with its reasoning
and the evidence it rested on (Principle 13, 17).

This is what lets ALLEDITS say:
  "These three clips match. I recommend #2 because its movement and composition
   match the preceding shot."
and lets a human audit or override it (Principle 14).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict


@dataclass
class Decision:
    stage: str                 # e.g. "clip_selection", "transition_choice"
    subject: str               # what the decision is about, e.g. "slot_03"
    choice: str                # what was chosen
    rationale: str             # why, in plain language
    confidence: float = 0.0
    alternatives: list = field(default_factory=list)   # [{choice, score, why_not}]
    evidence: dict = field(default_factory=dict)       # measured numbers behind it
    actor: str = "rule_based_planner"                  # which component/model decided
    t: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class DecisionLedger:
    def __init__(self):
        self.decisions: list[Decision] = []

    def record(self, **kwargs) -> Decision:
        d = Decision(**kwargs)
        self.decisions.append(d)
        return d

    def by_stage(self, stage: str):
        return [d for d in self.decisions if d.stage == stage]

    def to_dict(self):
        return {"decisions": [d.to_dict() for d in self.decisions]}

    def explain(self, stage: str | None = None) -> str:
        out = []
        for d in self.decisions:
            if stage and d.stage != stage:
                continue
            out.append(f"[{d.stage}/{d.subject}] -> {d.choice} "
                       f"(confidence {d.confidence:.2f}, by {d.actor})\n    {d.rationale}")
            for alt in d.alternatives[:3]:
                out.append(f"    alt: {alt.get('choice')} "
                           f"(score {alt.get('score', 0):.3f}) — {alt.get('why_not','')}")
        return "\n".join(out)
