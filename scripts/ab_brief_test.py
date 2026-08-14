"""Proof that a brief changes the edit.

Runs the SAME footage, reference and music through the full pipeline twice with
opposing briefs, then measures the resulting timelines and rendered files.

If Phase 1 works, these numbers must differ in the predicted directions. If they
don't, the brief is decorative and the phase is not done.

    python3 scripts/ab_brief_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alledits.pipeline.vertical_slice import VerticalSlice   # noqa: E402

MEDIA = Path("/home/claude/testmedia")
WORK = Path("/home/claude/abwork")

BRIEFS = {
    "restrained": "cinematic, slow and restrained, smooth and understated, no shake",
    "chaotic": "chaotic, aggressive and fast, flashy with whip pans and jump cuts",
}


def measure(res):
    tl = res.timeline
    clips = tl.clips
    durs = [c.timeline_duration for c in clips]
    fx = [len(c.effects) for c in clips]
    trans = {}
    for c in clips:
        t = getattr(c.transition_in, "type", None) or "cut"
        trans[t] = trans.get(t, 0) + 1
    fx_types = {}
    for c in clips:
        for e in c.effects:
            fx_types[e.type] = fx_types.get(e.type, 0) + 1
    return {
        "clips": len(clips),
        "duration": round(tl.duration, 2),
        "cuts_per_second": round(len(clips) / max(tl.duration, 1e-6), 3),
        "mean_shot": round(sum(durs) / max(len(durs), 1), 3),
        "min_shot": round(min(durs), 3),
        "max_shot": round(max(durs), 3),
        "effects_per_clip": round(sum(fx) / max(len(fx), 1), 2),
        "total_effects": sum(fx),
        "transitions": trans,
        "hard_cut_share": round(trans.get("cut", 0) / max(len(clips) - 1, 1), 3),
        "dissolve_share": round(trans.get("dissolve", 0) / max(len(clips) - 1, 1), 3),
        "showy_share": round((trans.get("flash", 0) + trans.get("whip", 0))
                             / max(len(clips) - 1, 1), 3),
        "effect_types": fx_types,
        "shake_effects": fx_types.get("shake", 0),
        "grain_effects": fx_types.get("film_grain", 0),
        "critic_score": round(res.critiques[-1].score, 2) if res.critiques else None,
        "pacing_multiplier": round(res.constraints.pacing_multiplier, 3),
        "effect_density": round(res.constraints.effect_density, 3),
        "continuity_weight": round(res.constraints.continuity_weight, 3),
    }


def run(name, brief):
    # Both runs share one workdir so ingestion, reference analysis and any
    # unchanged render segments come from cache — the A/B comparison is about
    # the brief, not about re-doing identical analysis twice.
    print(f"\n{'='*72}\nRUN: {name}\nBRIEF: {brief}\n{'='*72}", flush=True)
    vs = VerticalSlice(WORK, aspect=(1080, 1920), fps=30)
    res = vs.run(clips=sorted((MEDIA / "clips").glob("*.mp4")),
                 reference=MEDIA / "reference.mp4", music=MEDIA / "music.wav",
                 instruction=brief, target_duration=18.0,
                 log=lambda *a: None)   # quiet; we want the numbers
    m = measure(res)
    # preserve each run's output before the next overwrites it
    import shutil
    dest = WORK / f"ALLEDITS_{name}.mp4"
    if res.final_path:
        shutil.copy2(res.final_path, dest)
    m["render"] = str(dest)
    print(json.dumps(m, indent=2))
    return m


CHECKS = [
    ("mean shot length", "mean_shot", "restrained > chaotic", lambda r, c: r > c * 1.25),
    ("clip count", "clips", "restrained < chaotic", lambda r, c: r < c * 0.85),
    ("cuts per second", "cuts_per_second", "restrained < chaotic", lambda r, c: r < c * 0.85),
    ("effects per clip", "effects_per_clip", "restrained < chaotic", lambda r, c: r < c),
    ("dissolve share", "dissolve_share", "restrained > chaotic", lambda r, c: r > c),
    ("showy transitions", "showy_share", "restrained < chaotic", lambda r, c: r < c),
    ("shake effects", "shake_effects", "restrained == 0 < chaotic", lambda r, c: r == 0 and c > 0),
    ("pacing multiplier", "pacing_multiplier", "restrained > chaotic", lambda r, c: r > c),
    ("effect density", "effect_density", "restrained < chaotic", lambda r, c: r < c),
    ("continuity weight", "continuity_weight", "restrained > chaotic", lambda r, c: r > c),
]


def main():
    results = {k: run(k, v) for k, v in BRIEFS.items()}
    r, c = results["restrained"], results["chaotic"]

    print(f"\n{'='*72}\nDIFFERENTIAL RESULTS\n{'='*72}")
    print(f"{'metric':22}{'restrained':>14}{'chaotic':>12}   {'expectation':<28} verdict")
    print("-" * 92)
    passed = failed = 0
    for label, key, expect, fn in CHECKS:
        rv, cv = r[key], c[key]
        ok = fn(rv, cv)
        passed += ok
        failed += not ok
        print(f"{label:22}{rv:>14}{cv:>12}   {expect:<28} {'PASS' if ok else 'FAIL'}")

    print("-" * 92)
    print(f"transitions restrained: {r['transitions']}")
    print(f"transitions chaotic   : {c['transitions']}")
    print(f"effect types restrained: {r['effect_types']}")
    print(f"effect types chaotic   : {c['effect_types']}")
    print(f"\n{passed} passed, {failed} failed")

    ident = all(r[k] == c[k] for k in
                ("clips", "mean_shot", "effects_per_clip", "showy_share"))
    if ident:
        print("\nFAIL: the two briefs produced identical timelines — the brief is inert.")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
