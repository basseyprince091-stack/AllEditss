"""AUTOPILOT: deciding the treatment when nobody supplies a brief (Spec §16).

Given footage and music and nothing else, the system has to choose how the piece
should feel. It does that the same way an editor does — by trying a few
genuinely different treatments and looking at the results — rather than by
picking a default and asserting it was the right one.

How it works, and why:

- Candidates are explored at PREVIEW scale. Rendering every candidate at full
  raster would cost minutes per file that is about to be discarded, and the
  critic measures the same edit either way.
- Each candidate is scored by the existing critic, which re-measures the
  RENDERED file rather than inspecting the plan. A candidate cannot win by
  having a persuasive plan.
- Analysis is shared through one workdir, so ingest, shot detection, quality
  analysis and music analysis happen once for all candidates.
- The winner is re-rendered at full scale and, optionally, mastered.

The honest part: autopilot reports every candidate's score, not only the
winner's. If the spread is small the choice was close to arbitrary, and the
caller should be able to see that rather than be told a confident story.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .vertical_slice import VerticalSlice


# Deliberately spread across the space rather than clustered: if every candidate
# is a variation on "energetic", exploring them proves nothing. Each is a brief
# in the same language a user would type, so autopilot and manual runs share one
# code path — no hidden second planner.
DEFAULT_CANDIDATES = [
    ("restrained", "cinematic, slow and restrained, smooth, let it breathe"),
    ("balanced", "clean and confident, steady pacing"),
    ("energetic", "energetic and punchy, fast cuts, bold"),
]


@dataclass
class Candidate:
    name: str
    brief: str
    score: float = 0.0
    preview_path: str = ""
    clips: int = 0
    cuts_per_sec: float = 0.0
    critique: object = None
    error: str = ""

    def to_dict(self):
        return {"name": self.name, "brief": self.brief, "score": self.score,
                "clips": self.clips, "cuts_per_sec": round(self.cuts_per_sec, 3),
                "preview_path": self.preview_path, "error": self.error}


@dataclass
class AutopilotResult:
    candidates: list = field(default_factory=list)
    winner: Candidate | None = None
    final_path: str = ""
    master: object = None
    elapsed: float = 0.0

    @property
    def scores(self) -> list:
        return [c.score for c in self.candidates if not c.error]

    @property
    def spread(self) -> float:
        """Full range of candidate scores — how much exploring changed anything."""
        s = self.scores
        return (max(s) - min(s)) if len(s) > 1 else 0.0

    @property
    def margin(self) -> float:
        """Gap between the WINNER and the runner-up.

        This, not the full spread, is what makes a choice defensible. A run
        scoring 7.4 / 7.4 / 4.8 has a spread of 2.56, which looks decisive and
        is not: the top two tied, and picking between them was arbitrary. The
        rejected candidate being poor says nothing about the winner being right.
        """
        s = sorted(self.scores, reverse=True)
        return (s[0] - s[1]) if len(s) > 1 else 0.0

    @property
    def decisive(self) -> bool:
        """Decisive only if the winner actually beat the runner-up."""
        return self.margin >= 0.5

    def to_dict(self):
        return {"candidates": [c.to_dict() for c in self.candidates],
                "winner": self.winner.name if self.winner else None,
                "spread": round(self.spread, 3),
                "margin": round(self.margin, 3), "decisive": self.decisive,
                "final_path": self.final_path,
                "master": self.master.to_dict() if self.master else None,
                "elapsed": round(self.elapsed, 1)}


class Autopilot:
    def __init__(self, workdir: Path, aspect=(1080, 1920), fps=30,
                 candidates=None, deliver_profile: str | None = None,
                 allow_upscale: bool = False):
        self.workdir = Path(workdir)
        self.aspect, self.fps = aspect, fps
        self.candidates = list(candidates or DEFAULT_CANDIDATES)
        self.deliver_profile = deliver_profile
        self.allow_upscale = allow_upscale

    def run(self, clips, reference, music, target_duration: float = 12.0,
            log=print) -> AutopilotResult:
        t0 = time.time()
        res = AutopilotResult()

        log(f"[AUTOPILOT] exploring {len(self.candidates)} treatments "
            f"at preview scale")
        for name, brief in self.candidates:
            cand = Candidate(name=name, brief=brief)
            log(f"\n  -- candidate '{name}': {brief!r}")
            try:
                # One shared workdir, so analysis is computed once and the render
                # cache is reused wherever two candidates agree on a segment.
                vs = VerticalSlice(self.workdir, aspect=self.aspect, fps=self.fps)
                r = vs.run(clips, reference, music, brief,
                           target_duration=target_duration,
                           log=lambda *_: None, stop_after_preview=True)
                if not r.critiques:
                    cand.error = "no critique produced"
                else:
                    crit = r.critiques[-1]
                    cand.critique = crit
                    cand.score = float(crit.score)
                    cand.preview_path = str(r.preview_path or "")
                if r.timeline:
                    cand.clips = len(r.timeline.clips)
                    cand.cuts_per_sec = (len(r.timeline.clips)
                                         / max(r.timeline.duration, 1e-6))
                log(f"     score {cand.score:.1f}/10  "
                    f"({cand.clips} clips, {cand.cuts_per_sec:.2f} cuts/s)")
            except Exception as e:
                # One bad candidate must not sink the run; it is reported, not
                # hidden, and simply cannot win.
                cand.error = f"{type(e).__name__}: {e}"
                log(f"     failed: {cand.error}")
            res.candidates.append(cand)

        viable = [c for c in res.candidates if not c.error]
        if not viable:
            raise RuntimeError("every autopilot candidate failed; "
                               + "; ".join(c.error for c in res.candidates))

        res.winner = max(viable, key=lambda c: c.score)
        log(f"\n[AUTOPILOT] winner: '{res.winner.name}' at "
            f"{res.winner.score:.1f}/10")
        log(f"    scores span {res.spread:.2f}; winner beat the runner-up by "
            f"{res.margin:.2f} "
            + ("(a clear preference)" if res.decisive
               else "(too close to call — the top treatments scored alike, so "
                    "this choice is weakly held)"))

        log("\n[AUTOPILOT] rendering the winner at full scale")
        vs = VerticalSlice(self.workdir, aspect=self.aspect, fps=self.fps,
                           deliver_profile=self.deliver_profile,
                           allow_upscale=self.allow_upscale)
        final = vs.run(clips, reference, music, res.winner.brief,
                       target_duration=target_duration, log=log)
        res.final_path = str(final.final_path)
        res.master = final.master
        res.elapsed = time.time() - t0
        log(f"\n[AUTOPILOT] done in {res.elapsed:.0f}s -> {res.final_path}")
        return res
