"""The long-running operations, exposed as submittable jobs (Spec §28).

Every ALLEDITS operation that takes real time — ingesting a library, running an
edit, exploring treatments, mastering a deliverable — is wrapped here with one
signature: `fn(..., progress=callback)`. That is the signature the job queue
calls, so any of them can run in the background and be polled.

These are thin adapters ON PURPOSE. The pipeline already accepts a `log` or
`progress` callback at every stage, so wrapping is the whole job; reimplementing
any of this inside a task would create a second code path that could drift from
the one the tests cover.

Cancellation works because the queue's progress callback raises inside these
calls. A cancelled render therefore stops at a stage boundary and leaves its
completed segments in the render cache — which is a feature, since resuming
reuses them, but it is recorded on the job rather than left implicit.
"""
from __future__ import annotations

from pathlib import Path


def _fanout(progress, lo: float, hi: float):
    """Map a stage's 0-1 progress onto a slice of the job's overall progress."""
    def report(p: float, msg: str = ""):
        if progress:
            progress(lo + (hi - lo) * max(0.0, min(1.0, p)), msg)
    return report


def ingest_library(clips_dir, workdir, progress=None) -> dict:
    """Index a folder of footage. The slowest first-run step."""
    from ..core.storage import LocalStorage
    from ..media.ingest import Ingestor
    from ..search.index import MediaIndex

    clips = sorted(Path(clips_dir).glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"no .mp4 clips in {clips_dir}")
    ing = Ingestor(LocalStorage(workdir))
    idx = MediaIndex()
    for i, c in enumerate(clips):
        if progress:
            progress(i / len(clips), f"analysing {c.name}")
        idx.add_asset(ing.ingest(c))
    if progress:
        progress(1.0, f"{len(idx.shots)} shots indexed from {len(clips)} clips")
    return {"clips": len(clips), "shots": len(idx.shots), "workdir": str(workdir)}


def run_edit(clips_dir, reference, music, instruction, workdir,
             duration: float = 12.0, deliver: str | None = None,
             style=None, progress=None) -> dict:
    """A full edit. Progress is reported through the pipeline's own log."""
    from .vertical_slice import VerticalSlice

    clips = sorted(Path(clips_dir).glob("*.mp4"))
    report = _fanout(progress, 0.0, 1.0)
    # The pipeline logs prose rather than a fraction, so progress advances by
    # stage. Fabricating a percentage from log volume would be a made-up number.
    stages = {"Ingest": 0.15, "Analyzing music": 0.25, "Analyzing reference": 0.35,
              "Planning": 0.45, "Selecting": 0.55, "Rendering preview": 0.7,
              "Critique": 0.8, "Rendering final": 0.9, "Mastering": 0.97}
    state = {"p": 0.0}

    def log(msg=""):
        text = str(msg)
        for key, frac in stages.items():
            if key.lower() in text.lower() and frac > state["p"]:
                state["p"] = frac
                report(frac, text.strip()[:80])
                return

    vs = VerticalSlice(Path(workdir), deliver_profile=deliver)
    res = vs.run(clips=clips,
                 reference=Path(reference) if reference else None,
                 music=Path(music), instruction=instruction,
                 target_duration=duration, log=log, grammar=style)
    if progress:
        progress(1.0, "complete")
    return {"final_path": str(res.final_path or ""),
            "clips": len(res.timeline.clips) if res.timeline else 0,
            "duration": round(res.timeline.duration, 2) if res.timeline else 0.0,
            "score": round(res.critiques[-1].score, 1) if res.critiques else None,
            "project_path": str(res.project_path or ""),
            "master": res.master.to_dict() if res.master else None}


def run_autopilot(clips_dir, reference, music, workdir, duration: float = 12.0,
                  deliver: str | None = None, progress=None) -> dict:
    from .autopilot import Autopilot

    clips = sorted(Path(clips_dir).glob("*.mp4"))
    ap = Autopilot(Path(workdir), deliver_profile=deliver)
    seen = {"n": 0}
    total = len(ap.candidates) + 1

    def log(msg=""):
        text = str(msg)
        if "candidate '" in text or "rendering the winner" in text:
            seen["n"] += 1
            if progress:
                progress(min(0.95, seen["n"] / total), text.strip()[:80])

    res = ap.run(clips, Path(reference), Path(music),
                 target_duration=duration, log=log)
    if progress:
        progress(1.0, f"winner: {res.winner.name}")
    return res.to_dict()


def run_master(src, out, profile: str = "youtube_shorts",
               allow_upscale: bool = False, progress=None) -> dict:
    from ..master import master

    if progress:
        progress(0.1, f"mastering to {profile}")
    r = master(src, out, profile, allow_upscale=allow_upscale,
               log=lambda *_: None)
    if progress:
        progress(1.0, f"{r.qc.summary()}; conformant={r.conformant}")
    return r.to_dict()


TASKS = {
    "ingest": ingest_library,
    "edit": run_edit,
    "autopilot": run_autopilot,
    "master": run_master,
}
