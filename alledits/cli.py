"""ALLEDITS command line interface.

  python -m alledits.cli edit  --clips DIR --reference REF.mp4 --music M.wav \
                               --instruction "..." --duration 18 --out OUT.mp4
  python -m alledits.cli analyze --media FILE
  python -m alledits.cli grammar --reference REF.mp4
  python -m alledits.cli explain --run WORKDIR [--stage clip_selection]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def cmd_edit(a):
    from .pipeline.vertical_slice import VerticalSlice
    clips = sorted(p for p in Path(a.clips).glob("*")
                   if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"})
    if not clips:
        sys.exit(f"no video files found in {a.clips}")
    w, h = (int(x) for x in a.size.split("x"))
    grammar = None
    if getattr(a, "style", None):
        grammar = _library(a).load(a.style)
    elif not a.reference:
        sys.exit("provide --reference, or --style to reuse a saved one")

    vs = VerticalSlice(Path(a.workdir), aspect=(w, h), fps=a.fps,
                       deliver_profile=getattr(a, 'deliver', None),
                       allow_upscale=getattr(a, 'allow_upscale', False))
    res = vs.run(clips=clips,
                 reference=Path(a.reference) if a.reference else None,
                 music=Path(a.music), instruction=a.instruction,
                 target_duration=a.duration, grammar=grammar)
    if a.out and res.final_path:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(res.final_path, a.out)
        print(f"\nwrote {a.out}")
    if res.critiques:
        print(f"final self-assessment: {res.critiques[-1].score:.1f}/10")
    return 0


def cmd_analyze(a):
    from .core.storage import LocalStorage
    from .media.ingest import Ingestor
    ing = Ingestor(LocalStorage(Path(a.workdir) / "store"))
    asset = ing.ingest(Path(a.media), progress=lambda p, m: print(f"  {m}"))
    for s in asset.shots:
        v, q = s.visual, s.quality
        print(f"\n{s.id}  {s.start:.2f}-{s.end:.2f}s")
        print(f"  camera   : {v['camera_movement']} (conf {v['camera_confidence']:.2f})")
        print(f"  energy   : {v['visual_energy']:.2f}   faces: {v['faces']}")
        print(f"  quality  : technical {q['technical_quality']:.2f} / "
              f"creative {q['creative_value']:.2f} -> {q['handling']}")
        print(f"  because  : {'; '.join(q['reasons'])}")
    return 0


def cmd_grammar(a):
    from .reference.analyze_reference import analyze_reference
    g = analyze_reference(Path(a.reference), label=Path(a.reference).stem)
    print(g.to_json())
    return 0


def _load_project(workdir):
    from .core.project import Project
    path = Path(workdir) / "project.json"
    if not path.exists():
        sys.exit(f"no project at {path} — run `alledits edit` first")
    return Project.load(path), path


def cmd_project(a):
    """Inspect a saved project and its human directives."""
    proj, path = _load_project(a.workdir)
    print(f"project : {proj.name}  ({proj.id})")
    print(f"brief   : {proj.brief!r}")
    print(f"duration: {proj.target_duration}s   settings: {proj.project_settings}")
    print(f"sources : {len(proj.clip_paths)} clips, ref={Path(proj.reference_path).name}, "
          f"music={Path(proj.music_path).name}")
    if proj.timeline:
        print(f"timeline: {len(proj.timeline.get('clips', []))} clips, "
              f"{proj.timeline.get('duration', 0):.2f}s")
    print(f"\noverrides ({len(proj.overrides.directives)}):")
    for d, line in zip(proj.overrides.directives, proj.overrides.describe()):
        print(f"  [{d.id}] {line}")
    if proj.history:
        print("\nhistory:")
        for h in proj.history[-8:]:
            print(f"  {h['event']}: {h['detail']}")
    return 0


def cmd_direct(a):
    """Record a human directive against a saved project.

    The directive is stored, not applied immediately: re-running `edit` on the
    same workdir rebuilds the timeline honouring it, reusing cached analysis and
    unchanged render segments.
    """
    from .core.project import DirectiveKind
    proj, path = _load_project(a.workdir)
    kind = DirectiveKind(a.kind)
    kw = {"note": a.note or ""}
    if a.slot is not None:
        kw["slot_index"] = a.slot
    if a.shot:
        kw["shot_id"] = a.shot
    if a.value:
        kw["value"] = a.value

    need_slot = {DirectiveKind.PIN_SHOT, DirectiveKind.REJECT_AT_SLOT,
                 DirectiveKind.LOCK_CLIP, DirectiveKind.FORCE_TRANSITION}
    if kind in need_slot and a.slot is None:
        sys.exit(f"{a.kind} requires --slot")
    if kind in {DirectiveKind.PIN_SHOT, DirectiveKind.REJECT_SHOT,
                DirectiveKind.REJECT_AT_SLOT} and not a.shot:
        sys.exit(f"{a.kind} requires --shot")
    if kind in {DirectiveKind.FORCE_TRANSITION, DirectiveKind.BAN_EFFECT} and not a.value:
        sys.exit(f"{a.kind} requires --value")

    d = proj.overrides.add(kind, **kw)
    proj.record("directive", f"{kind.value} {a.shot or a.value or ''} slot={a.slot}")
    proj.save(path)
    print(f"recorded [{d.id}] {kind.value}")
    print("re-run `alledits edit` on this workdir to rebuild with it applied")
    return 0


def cmd_undirect(a):
    proj, path = _load_project(a.workdir)
    if a.all:
        n = len(proj.overrides.directives)
        proj.overrides.clear()
        proj.record("directive", f"cleared {n}")
        proj.save(path)
        print(f"cleared {n} directive(s)")
        return 0
    if not a.id:
        sys.exit("pass --id <directive_id> or --all")
    ok = proj.overrides.remove(a.id)
    proj.record("directive", f"removed {a.id}")
    proj.save(path)
    print("removed" if ok else f"no directive with id {a.id}")
    return 0 if ok else 1


def cmd_find(a):
    """Search a media library by measured attributes."""
    from .core.storage import LocalStorage
    from .media.ingest import Ingestor
    from .search.index import MediaIndex
    from .search.query import parse_query

    q = parse_query(a.query, limit=a.top)
    print(f"query    : {a.query!r}")
    print(f"criteria : {q.describe()}")
    if q.unmatched_terms:
        print(f"not understood: {', '.join(q.unmatched_terms)}")
        print("  (FIND matches measured attributes, not open-vocabulary meaning)")

    ing = Ingestor(LocalStorage(a.workdir))
    idx, origin = MediaIndex(), {}
    for c in sorted(Path(a.clips).glob("*.mp4")):
        asset = ing.ingest(c)
        origin[asset.id] = c.name
        idx.add_asset(asset)

    results = idx.search(q)
    for c in getattr(idx, "last_inert_criteria", []):
        print(f"note     : '{c}' is not measured on this library — ignored")
    if not results:
        print("\nno shots matched")
        return 0
    print(f"\n{len(results)} result(s):")
    for i, r in enumerate(results, 1):
        sh = r["shot"]
        print(f"  {i}. {origin.get(sh.asset_id, sh.asset_id)}  "
              f"[{sh.start:.2f}-{sh.end:.2f}s]  score {r['score']:.2f}")
        if r["matched"]:
            print(f"       matched: {'; '.join(r['matched'])}")
        if r["unmet"]:
            print(f"       not met: {'; '.join(r['unmet'])}")
    return 0


def _library(a):
    from .reference.style import StyleLibrary
    return StyleLibrary(Path(getattr(a, "styles", None) or
                             (Path.home() / ".alledits" / "styles")))


def _profile_path(a):
    return Path(getattr(a, "profile_path", None)
                or (Path.home() / ".alledits" / "profile.json"))


def _load_profile(a):
    """The saved profile, or a default. Never fails — guidance always works."""
    from .profile import CreatorProfile
    p = _profile_path(a)
    prof = CreatorProfile.load(p) if p.exists() else CreatorProfile()
    if getattr(a, "skill", None):
        # An explicit --skill overrides the stored level for this run only.
        prof.instruction_level = {"beginner": "teach_me",
                                  "intermediate": "normal",
                                  "advanced": "technical"}.get(a.skill,
                                                               prof.instruction_level)
    return prof


def cmd_profile(a):
    """Create, show or update the creator profile (Spec 4)."""
    from .profile import CreatorProfile, LEVELS, EXPERIENCE
    path = _profile_path(a)

    if a.action == "show":
        if not path.exists():
            print("no profile saved yet — run `alledits profile set ...`")
            return 0
        prof = CreatorProfile.load(path)
        d = prof.to_dict()
        for k in ("name", "editing_experience", "filming_experience",
                  "instruction_level", "device", "has_tripod", "has_gimbal",
                  "crew_size", "lighting", "location"):
            print(f"  {k:20} {d[k]}")
        if prof.content_types:
            print(f"  {'content_types':20} {', '.join(prof.content_types)}")
        if prof.explained:
            print(f"  {'already explained':20} "
                  f"{', '.join(sorted(prof.explained))}")
        return 0

    if a.action == "reset":
        if path.exists():
            path.unlink()
        print("profile cleared")
        return 0

    if a.action == "set":
        prof = CreatorProfile.load(path) if path.exists() else CreatorProfile()
        if a.level:
            if a.level not in LEVELS:
                sys.exit(f"level must be one of: {', '.join(LEVELS)}")
            prof.instruction_level = a.level
        for attr, val in (("editing_experience", a.editing),
                          ("filming_experience", a.filming)):
            if val:
                if val not in EXPERIENCE:
                    sys.exit(f"experience must be one of: {', '.join(EXPERIENCE)}")
                setattr(prof, attr, val)
        if a.device:
            prof.device = a.device
        if a.tripod is not None:
            prof.has_tripod = a.tripod
        if a.gimbal is not None:
            prof.has_gimbal = a.gimbal
        if a.crew:
            prof.crew_size = a.crew
        if a.content:
            prof.content_types = [x.strip() for x in a.content.split(",") if x.strip()]
        if a.knows:
            prof.known_concepts = sorted(set(prof.known_concepts)
                                         | {x.strip() for x in a.knows.split(",")})
        prof.save(path)
        print(f"saved -> {path}")
        print(f"  guidance level: {prof.instruction_level}; "
              f"{prof.device}, tripod={prof.has_tripod}, crew={prof.crew_size}")
        return 0

    sys.exit(f"unknown profile action {a.action!r}")


def cmd_shoot(a):
    """Plan shots, and find which the library is missing (Spec 5, 24)."""
    from .shoot import build_sequence, assess_coverage, SEQUENCES, MISSING, LIKELY
    from .profile import render_instructions
    prof = _load_profile(a)
    from .reference.analyze_reference import analyze_reference
    from .reference.style import StyleLibrary

    if a.action == "sequences":
        for name, (desc, _) in sorted(SEQUENCES.items()):
            print(f"  {name:10} {desc}")
        return 0

    if a.style:
        grammar = _library(a).load(a.style)
    elif a.reference:
        grammar = analyze_reference(a.reference, label="reference")
    else:
        sys.exit("provide --reference or --style so shot lengths follow a style")

    try:
        specs = build_sequence(a.sequence, grammar)
    except KeyError as e:
        sys.exit(str(e))

    if a.action == "plan":
        print(f"Shot plan — '{a.sequence}' sequence, lengths from the style "
              f"({grammar.pacing.cuts_per_second:.2f} cuts/s)\n")
        for spec in specs:
            print(render_instructions(spec, prof))
            print()
        return 0

    if a.action == "coverage":
        if not a.clips:
            sys.exit("coverage needs --clips (the footage you already have)")
        from .core.storage import LocalStorage
        from .media.ingest import Ingestor
        from .search.index import MediaIndex
        ing = Ingestor(LocalStorage(a.workdir))
        idx = MediaIndex()
        for c in sorted(Path(a.clips).glob("*.mp4")):
            idx.add_asset(ing.ingest(c))
        rep = assess_coverage(idx, specs, a.sequence)
        print(rep.summary() + "\n")
        for sh in rep.shots:
            print(f"  {sh.status.upper():13} {sh.number}. {sh.name}")
            print(f"      {sh.reason}")
            if sh.unchecked:
                print(f"      not verified: {sh.unchecked}")
        gaps = rep.missing
        if gaps:
            print(f"\n{len(gaps)} shot(s) to film:\n")
            for sh in gaps:
                spec = next(x for x in specs if x.number == sh.number)
                print(render_instructions(spec, prof))
                print()
        else:
            print("\nNothing is missing by measurable shape. Content still needs "
                  "your eye — this build has no model that can confirm what a "
                  "clip shows.")
        return 0

    sys.exit(f"unknown shoot action {a.action!r}")


def cmd_capabilities(a):
    """What this install can do, what it cannot, and what would unlock it."""
    from .intelligence.capabilities import default_registry
    reg = default_registry()
    print(reg.summary())
    print()
    for st in reg.status():
        if st.available:
            print(f"  ON   {st.capability:22} {st.provider}")
        else:
            print(f"  off  {st.capability:22} {st.reason}")
            for u in st.unlocks:
                print(f"         would enable: {u}")
    print("\nFeatures that work today without any model: reference style "
          "extraction,\nbeat-locked editing, footage rescue, sound mixing and "
          "loudness, delivery\nmastering, attribute-based FIND, autopilot, and "
          "director notes.")
    return 0


def cmd_note(a):
    """DIRECTOR: give the edit a note in plain language."""
    from .core.project import DirectiveKind
    from .intelligence.director import parse_note, timeline_from_project

    proj, path = _load_project(a.workdir)
    plan = parse_note(a.text, timeline_from_project(proj))

    print(f"note: {a.text!r}")
    if not plan.understood:
        print("\nnothing in that note could be acted on:")
        for u in plan.unresolved:
            print(f"  - {u}")
        print("\nDIRECTOR matches editorial vocabulary against the existing "
              "edit;\nit does not interpret open-ended description.")
        return 1

    print("\nwill apply:")
    for ch in plan.changes:
        print(f"  - {ch.rationale}")
    if plan.brief_delta:
        print(f"  - re-plan the whole edit with: {plan.brief_delta!r}")
    if plan.unresolved:
        print("\ncould NOT act on:")
        for u in plan.unresolved:
            print(f"  - {u}")

    if a.dry_run:
        print("\n(dry run — nothing saved)")
        return 0

    for ch in plan.changes:
        kw = {"note": a.text}
        if ch.slot_index is not None:
            kw["slot_index"] = ch.slot_index
        if ch.shot_id:
            kw["shot_id"] = ch.shot_id
        if ch.value is not None:
            kw["value"] = ch.value
        d = proj.overrides.add(DirectiveKind(ch.kind), **kw)
        # Anchor to what the human was actually looking at, so the directive
        # survives a re-plan that moves this shot to a different slot.
        proj.overrides.anchor_to(d, timeline_from_project(proj).clips)
    if plan.brief_delta:
        proj.brief = (proj.brief + ", " + plan.brief_delta).strip(", ")
    proj.record("note", a.text)
    proj.save(path)
    print(f"\nrecorded {len(plan.changes)} directive(s)"
          + (" and updated the brief" if plan.brief_delta else ""))
    print("re-run `alledits edit` on this workdir to rebuild with them applied")
    return 0


def cmd_style(a):
    """Save, list, inspect, blend or delete reusable styles."""
    from .reference.analyze_reference import analyze_reference
    from .reference.style import blend_grammars
    lib = _library(a)

    if a.action == "list":
        names = lib.list_names()
        if not names:
            print("no saved styles")
        for n in names:
            print("  " + lib.describe(n))
        return 0

    if a.action == "show":
        if not a.name:
            sys.exit("show requires --name")
        g = lib.load(a.name)
        print(g.to_json())
        return 0

    if a.action == "delete":
        if not a.name:
            sys.exit("delete requires --name")
        print("deleted" if lib.delete(a.name) else f"no style named {a.name!r}")
        return 0

    if a.action == "save":
        if not (a.name and a.reference):
            sys.exit("save requires --name and --reference")
        g = analyze_reference(a.reference, label=a.name)
        print(f"saved -> {lib.save(a.name, g)}")
        print("  " + lib.describe(a.name))
        return 0

    if a.action == "blend":
        if not (a.name and a.sources):
            sys.exit("blend requires --name and --sources "
                     "(e.g. --sources punchy:3,calm:1)")
        weighted = []
        for part in a.sources.split(","):
            nm, _, w = part.partition(":")
            weighted.append((lib.load(nm.strip()), float(w or 1)))
        g = blend_grammars(weighted, label=a.name)
        print(f"saved -> {lib.save(a.name, g)}")
        for n in g.notes:
            print(f"  note: {n}")
        print("  " + lib.describe(a.name))
        return 0

    sys.exit(f"unknown style action {a.action!r}")


def cmd_autopilot(a):
    """Edit with no brief: explore treatments, critique each, deliver the best."""
    from .pipeline.autopilot import Autopilot
    w, h = (int(x) for x in a.aspect.split("x")) if "x" in a.aspect else (1080, 1920)
    clips = sorted(Path(a.clips).glob("*.mp4"))
    if not clips:
        sys.exit(f"no .mp4 clips found in {a.clips}")
    ap = Autopilot(Path(a.workdir), aspect=(w, h), fps=a.fps,
                   deliver_profile=a.deliver, allow_upscale=a.allow_upscale)
    res = ap.run(clips, Path(a.reference), Path(a.music),
                 target_duration=a.duration, log=print)
    print("\ncandidates:")
    for c in res.candidates:
        mark = "*" if res.winner and c.name == res.winner.name else " "
        detail = c.error or (f"{c.score:.1f}/10  {c.clips} clips, "
                             f"{c.cuts_per_sec:.2f} cuts/s")
        print(f"  {mark} {c.name:12} {detail}")
    if not res.decisive:
        print(f"\nnote: the winner beat the runner-up by only {res.margin:.2f} "
              "— a weak preference, not a clear one")
    return 0


def cmd_master(a):
    """Transcode a finished edit to a delivery profile and QC the result."""
    from .master import master, PROFILES
    if a.profile not in PROFILES:
        sys.exit(f"unknown profile {a.profile!r}; "
                 f"available: {', '.join(sorted(PROFILES))}")
    r = master(a.input, a.output, a.profile,
               allow_upscale=a.allow_upscale, log=print)
    print()
    for c in r.qc.checks:
        print(c)
    print(f"\n{r.qc.summary()}")
    print(f"conformant: {r.conformant}")
    if r.qc.resolution_provenance != "native":
        print(f"resolution provenance: {r.qc.resolution_provenance}")
    print(f"-> {r.path}")
    return 0 if r.conformant else 1


def cmd_profiles(a):
    from .master import PROFILES
    for name in sorted(PROFILES):
        p = PROFILES[name]
        print(f"  {name:18} {p.width}x{p.height} @ {p.fps:g}  "
              f"{p.loudness_target:10} {p.notes}")
    return 0


def cmd_explain(a):
    led = Path(a.run) / "store" / "analysis" / "decision_ledger.json"
    if not led.exists():
        sys.exit(f"no decision ledger at {led}")
    data = json.loads(led.read_text())["decisions"]
    for d in data:
        if a.stage and d["stage"] != a.stage:
            continue
        print(f"[{d['stage']}/{d['subject']}] -> {d['choice']} "
              f"(confidence {d['confidence']:.2f}, decided by {d['actor']})")
        print(f"    {d['rationale']}")
        for alt in d.get("alternatives", [])[:2]:
            print(f"    considered {alt['choice']}: {alt['why_not']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser("alledits")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("edit", help="run the full edit pipeline")
    e.add_argument("--clips", required=True)
    e.add_argument("--reference", default=None)
    e.add_argument("--music", required=True)
    e.add_argument("--instruction", default="")
    e.add_argument("--duration", type=float, default=18.0)
    e.add_argument("--size", default="1080x1920")
    e.add_argument("--fps", type=int, default=30)
    e.add_argument("--workdir", default="./alledits_work")
    e.add_argument("--out", default=None)
    e.add_argument("--style", default=None,
                   help="use a saved style instead of analysing a reference")
    e.add_argument("--styles", default=None, help="style library directory")
    e.add_argument("--deliver", default=None,
                   help="also master for a delivery profile (see `alledits profiles`)")
    e.add_argument("--allow-upscale", dest="allow_upscale",
                   action="store_true")
    e.set_defaults(fn=cmd_edit)

    an = sub.add_parser("analyze", help="analyze one media file")
    an.add_argument("--media", required=True)
    an.add_argument("--workdir", default="./alledits_work")
    an.set_defaults(fn=cmd_analyze)

    g = sub.add_parser("grammar", help="extract a style grammar from a reference")
    g.add_argument("--reference", required=True)
    g.set_defaults(fn=cmd_grammar)

    pr = sub.add_parser("project", help="show a saved project and its directives")
    pr.add_argument("--workdir", default="./alledits_work")
    pr.set_defaults(fn=cmd_project)

    di = sub.add_parser("direct", help="record a human directive (pin/reject/lock/...)")
    di.add_argument("--workdir", default="./alledits_work")
    di.add_argument("--kind", required=True,
                    choices=["pin_shot", "reject_shot", "reject_at_slot", "lock_clip",
                             "force_transition", "force_effects", "ban_effect"])
    di.add_argument("--slot", type=int, default=None)
    di.add_argument("--shot", default=None)
    di.add_argument("--value", default=None)
    di.add_argument("--note", default="")
    di.set_defaults(fn=cmd_direct)

    ud = sub.add_parser("undirect", help="remove a directive")
    ud.add_argument("--workdir", default="./alledits_work")
    ud.add_argument("--id", default=None)
    ud.add_argument("--all", action="store_true")
    ud.set_defaults(fn=cmd_undirect)

    fd = sub.add_parser("find", help="search a media library by measured attributes")
    fd.add_argument("--clips", required=True, help="directory of source clips")
    fd.add_argument("--query", required=True)
    fd.add_argument("--top", type=int, default=10)
    fd.add_argument("--workdir", default="./alledits_work")
    fd.set_defaults(fn=cmd_find)

    sh = sub.add_parser("shoot", help="plan shots and find coverage gaps")
    sh.add_argument("action", choices=["plan", "coverage", "sequences"])
    sh.add_argument("--sequence", default="skill")
    sh.add_argument("--reference", default=None)
    sh.add_argument("--style", default=None)
    sh.add_argument("--styles", default=None)
    sh.add_argument("--clips", default=None)
    # No default: a stored profile's guidance level must not be silently
    # overridden by an argument the user never passed.
    sh.add_argument("--skill", default=None,
                    choices=["beginner", "intermediate", "advanced"])
    sh.add_argument("--workdir", default="./alledits_work")
    sh.add_argument("--profile-path", dest="profile_path", default=None)
    sh.set_defaults(fn=cmd_shoot)

    pr = sub.add_parser("profile", help="who you are, what you film with (Spec 4)")
    pr.add_argument("action", choices=["show", "set", "reset"])
    pr.add_argument("--level", default=None,
                    help="teach_me | normal | technical | minimal")
    pr.add_argument("--editing", default=None)
    pr.add_argument("--filming", default=None)
    pr.add_argument("--device", default=None, choices=["phone", "camera"])
    pr.add_argument("--tripod", dest="tripod", action="store_true", default=None)
    pr.add_argument("--no-tripod", dest="tripod", action="store_false")
    pr.add_argument("--gimbal", dest="gimbal", action="store_true", default=None)
    pr.add_argument("--crew", type=int, default=None)
    pr.add_argument("--content", default=None, help="comma-separated")
    pr.add_argument("--knows", default=None,
                    help="concepts to stop explaining, comma-separated")
    pr.add_argument("--profile-path", dest="profile_path", default=None)
    pr.set_defaults(fn=cmd_profile)

    cp = sub.add_parser("capabilities",
                        help="what this install can do, and what is missing")
    cp.set_defaults(fn=cmd_capabilities)

    nt = sub.add_parser("note", help="DIRECTOR: give the edit a note in words")
    nt.add_argument("--workdir", default="./alledits_work")
    nt.add_argument("--text", required=True)
    nt.add_argument("--dry-run", dest="dry_run", action="store_true")
    nt.set_defaults(fn=cmd_note)

    st = sub.add_parser("style", help="save / list / show / blend reusable styles")
    st.add_argument("action",
                    choices=["save", "list", "show", "blend", "delete"])
    st.add_argument("--name", default=None)
    st.add_argument("--reference", default=None)
    st.add_argument("--sources", default=None,
                    help="for blend: name:weight,name:weight")
    st.add_argument("--styles", default=None, help="style library directory")
    st.set_defaults(fn=cmd_style)

    ap = sub.add_parser("autopilot",
                        help="edit with no brief: explore, critique, deliver")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--aspect", default="1080x1920")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--deliver", default=None)
    ap.add_argument("--allow-upscale", dest="allow_upscale", action="store_true")
    ap.add_argument("--workdir", default="./alledits_work")
    ap.set_defaults(fn=cmd_autopilot)

    ms = sub.add_parser("master", help="master an edit to a delivery profile + QC")
    ms.add_argument("--input", required=True)
    ms.add_argument("--output", required=True)
    ms.add_argument("--profile", default="youtube_shorts")
    ms.add_argument("--allow-upscale", dest="allow_upscale",
                    action="store_true",
                    help="permit interpolating pixels the source does not have")
    ms.set_defaults(fn=cmd_master)

    pf = sub.add_parser("profiles", help="list delivery profiles")
    pf.set_defaults(fn=cmd_profiles)

    x = sub.add_parser("explain", help="print the decision ledger for a run")
    x.add_argument("--run", required=True)
    x.add_argument("--stage", default=None)
    x.set_defaults(fn=cmd_explain)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
