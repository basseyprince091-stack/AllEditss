"""The MVP vertical slice (Spec §27).

  1 ingest media      2 analyze media     3 analyze reference
  4 extract editing language              5 select clips
  6 construct timeline                    7 render preview
  8 inspect preview   9 revise           10 render final

Every stage writes to the decision ledger, and the run returns an artifact
bundle that makes the whole chain inspectable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.storage import LocalStorage, ANALYSIS, RENDER, OUTPUT
from ..core.jobs import InlineJobQueue
from ..core.ledger import DecisionLedger
from ..core.project import Project, OverrideSet
from ..core.errors import TimelineValidationError
from ..media.ingest import Ingestor
from ..media.probe import probe
from ..audio.analyze import analyze_audio
from ..audio.mix import plan_mix, validate_mix, diegetic_voice_tracks
from ..master import master as master_file
from ..reference.analyze_reference import analyze_reference
from ..search.index import MediaIndex
from ..intelligence.planner import plan_slots
from ..intelligence.brief import parse_brief
from ..intelligence.critic import critique, apply_revisions, MAX_REVISIONS
from ..intelligence.providers.local_embedder import LocalFeatureEmbedder
from ..intelligence.providers.heuristic_provider import HeuristicProvider
from ..timeline.schema import ProjectSettings, Timeline
from ..timeline.builder import build_timeline
from ..timeline.validator import validate, errors as val_errors, warnings as val_warnings
from ..render.ffmpeg_renderer import FFmpegRenderer


@dataclass
class SliceResult:
    final_path: Path | None = None
    preview_path: Path | None = None
    timeline: Timeline | None = None
    grammar: object = None
    audio: object = None
    mix: object = None
    critiques: list = field(default_factory=list)
    ledger: DecisionLedger = None
    index_stats: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    revisions: int = 0
    constraints: object = None
    overrides: object = None
    project_path: object = None
    master: object = None
    warnings: list = field(default_factory=list)


class VerticalSlice:
    PROJECT_FILE = "project.json"

    def __init__(self, workdir: Path, provider=None, aspect=(1080, 1920), fps=30,
                 project: Project | None = None,
                 deliver_profile: str | None = None,
                 allow_upscale: bool = False):
        self.workdir = Path(workdir)
        self.storage = LocalStorage(self.workdir / "store")
        self.embedder = LocalFeatureEmbedder()
        self.ingestor = Ingestor(self.storage, self.embedder)
        self.renderer = FFmpegRenderer(self.workdir / "render")
        self.provider = provider or HeuristicProvider()
        self.project_config = ProjectSettings(width=aspect[0], height=aspect[1], fps=fps,
                                       aspect_label=f"{aspect[0]}:{aspect[1]}")
        self.ledger = DecisionLedger()
        self.deliver_profile = deliver_profile
        self.allow_upscale = allow_upscale
        self.project = project or self.load_project() or Project()

    # ------------------------------------------------------------- persistence
    @property
    def project_path(self) -> Path:
        return self.workdir / self.PROJECT_FILE

    def load_project(self) -> Project | None:
        p = self.workdir / self.PROJECT_FILE
        return Project.load(p) if p.exists() else None

    def save_project(self) -> Path:
        return self.project.save(self.project_path)

    def run(self, clips: list, reference: Path, music: Path, instruction: str,
            target_duration: float = 18.0, log=print,
            overrides: OverrideSet | None = None,
            stop_after_preview: bool = False,
            grammar: object = None) -> SliceResult:
        r = SliceResult(ledger=self.ledger)
        t0 = time.time()

        # Human directives persist with the project, so they survive re-runs and
        # the critique loop rather than being re-entered every session.
        ov = overrides if overrides is not None else self.project.overrides
        self.project.overrides = ov
        self.project.brief = instruction
        self.project.target_duration = target_duration
        self.project.clip_paths = [str(c) for c in clips]
        self.project.reference_path = str(reference)
        self.project.music_path = str(music)
        self.project.project_settings = self.project_settings_dict()
        r.overrides = ov
        if not ov.directives:
            log("\n[--] No human overrides on this project")
        else:
            log(f"\n[--] Applying {len(ov.directives)} human override(s):")
            for line in ov.describe():
                log(f"    {line}")
            self.ledger.record(stage="human_override", subject="project",
                               choice=f"{len(ov.directives)} directive(s)",
                               rationale="; ".join(ov.describe()), actor="human",
                               confidence=1.0)

        self.ledger.record(
            stage="orchestration", subject="provider",
            choice=self.provider.name,
            rationale=("No language model is configured, so creative planning is done by "
                       "deterministic scoring over measured media features. This is "
                       "labelled rule-based, not AI."
                       if not getattr(self.provider, "available", lambda: False)()
                       or self.provider.name == "rule_based_planner"
                       else "Language model available for creative reasoning."),
            confidence=1.0, actor="orchestrator")

        # ---- 0: interpret the brief into measurable constraints ----
        cons = parse_brief(instruction, provider=self.provider)
        r.constraints = cons
        log(f"\n[0/10] Interpreting brief")
        log(f"    -> {cons.summary()}")
        log(f"    -> interpreted by {cons.actor} (language model: {cons.is_llm})")
        self.ledger.record(
            stage="brief_interpretation", subject="creative_constraints",
            choice=cons.summary(),
            rationale=("The brief was converted into measurable editing constraints "
                       "that govern pacing, selection, effects, transitions and "
                       "grading. " + " ".join(cons.notes)),
            confidence=0.8 if cons.matched_terms else 0.3,
            actor=cons.actor,
            evidence={"pacing_multiplier": cons.pacing_multiplier,
                      "effect_density": cons.effect_density,
                      "continuity_weight": cons.continuity_weight,
                      "intensity_offset": cons.intensity_offset,
                      "motion_preference": cons.motion_preference,
                      "transition_bias": cons.transition_bias,
                      "matched_terms": [m["term"] for m in cons.matched_terms]})

        # ---- 1 & 2: ingest + analyze media ----
        log("\n[1/10] Ingesting and analyzing footage")
        index = MediaIndex(embedder=self.embedder)
        t = time.time()
        for i, c in enumerate(clips):
            asset = self.ingestor.ingest(c, progress=lambda p, m: log(f"    {m}"))
            index.add_asset(asset)
        r.timings["ingest"] = time.time() - t
        r.index_stats = index.stats()
        log(f"    -> {r.index_stats['shots']} shots, "
            f"{r.index_stats['total_duration']:.1f}s of material")
        log(f"    -> handling: {r.index_stats['handling']}")

        # ---- 3: analyze music ----
        log("\n[2/10] Analyzing music")
        t = time.time()
        audio = analyze_audio(music)
        r.audio = audio
        r.timings["audio"] = time.time() - t
        log(f"    -> {audio.bpm:.1f} BPM (confidence {audio.bpm_confidence:.2f}), "
            f"{len(audio.beats)} beats, {len(audio.downbeats)} downbeats, "
            f"{len(audio.drops)} drop(s), {len(audio.sections)} section(s)")
        self.ledger.record(stage="music_analysis", subject="track",
                           choice=f"{audio.bpm:.1f} BPM",
                           rationale=(f"Beat grid recovered from spectral-flux onsets; "
                                      f"{len(audio.beats)} beats detected. Cuts will be "
                                      f"quantized to this grid."),
                           confidence=float(audio.bpm_confidence),
                           evidence={"drops": audio.drops[:5],
                                     "sections": len(audio.sections)})

        # ---- 4 & 5: reference -> editing language ----
        t = time.time()
        if grammar is not None:
            # A saved or blended style is already a measurement — re-analysing a
            # reference we were not given would be inventing one.
            log(f"\n[3/10] Using supplied style "
                f"'{grammar.source_label or grammar.id}'")
        else:
            log("\n[3/10] Analyzing reference edit")
            grammar = analyze_reference(reference, label=Path(reference).stem,
                                        music_beats=audio.beats)
        r.grammar = grammar
        r.timings["reference"] = time.time() - t
        log(f"    -> {grammar.pacing.cuts_per_second:.2f} cuts/s, "
            f"median shot {grammar.pacing.median_shot:.2f}s, "
            f"rhythm '{grammar.pacing.rhythm}'")
        log(f"    -> transitions: cut {grammar.transitions.hard_cut_share:.0%}, "
            f"flash {grammar.transitions.flash_share:.0%}, "
            f"whip {grammar.transitions.whip_share:.0%}, "
            f"dissolve {grammar.transitions.dissolve_share:.0%}")
        log(f"    -> look: contrast {grammar.color.contrast:.2f}, "
            f"saturation {grammar.color.saturation:.2f}, "
            f"warmth {grammar.color.warmth:+.2f}, key '{grammar.color.key}'")
        log(f"    -> arc: {' -> '.join(s['role'] for s in grammar.structure)}")
        self.storage.put_json(ANALYSIS, "style_grammar.json", grammar.to_dict())

        # ---- 6: plan slots ----
        log("\n[4/10] Planning the edit against the beat grid")
        start_offset = 0.0
        if audio.downbeats:
            start_offset = min(d for d in audio.downbeats if d < 4.0) if any(
                d < 4.0 for d in audio.downbeats) else 0.0
        slots = plan_slots(grammar, audio, target_duration,
                           start_offset=start_offset, ledger=self.ledger, cons=cons)
        log(f"    -> {len(slots)} slots over {target_duration:.1f}s "
            f"(from {start_offset:.2f}s in the track)")
        roles = {}
        for s in slots:
            roles[s.role] = roles.get(s.role, 0) + 1
        log(f"    -> arc: {roles}")

        # ---- 7: select + build ----
        diversity, pacing_bias = 0.0, 1.0
        log("\n[5/10] Selecting clips and constructing the timeline")
        t = time.time()
        timeline = build_timeline(slots, index, grammar, audio, self.project_config,
                                  music_path=str(music), music_start=start_offset,
                                  intent=instruction, ledger=self.ledger, cons=cons,
                                  overrides=ov)

        # An explicit duration directive re-flows everything after it, so the
        # piece can end up longer or shorter than asked. That is the human's
        # call — but it must be stated, not left for them to notice in the file.
        drift = timeline.duration - target_duration
        if abs(drift) > 0.05:
            msg = (f"final duration {timeline.duration:.2f}s vs the "
                   f"{target_duration:.2f}s requested ({drift:+.2f}s)")
            if any(d.kind == "set_duration" for d in ov.directives):
                msg += " — caused by an explicit hold/trim directive"
            log(f"    ! {msg}")
            r.warnings.append(msg)
        r.timeline = timeline
        r.timings["build"] = time.time() - t
        log(f"    -> {len(timeline.clips)} clips, {timeline.duration:.2f}s")

        # ---- 8: validate (hard gate) ----
        log("\n[6/10] Validating timeline")
        timeline.mix = self._plan_sound(timeline, music, cons, log)
        r.mix = timeline.mix

        issues = validate(timeline, beat_grid=audio.beats)
        errs, warns = val_errors(issues), val_warnings(issues)
        r.validation = {"errors": [str(i) for i in errs],
                        "warnings": [str(i) for i in warns]}
        for w in warns:
            log(f"    warning: {w}")
        if errs:
            for e in errs:
                log(f"    ERROR: {e}")
            raise TimelineValidationError(errs)
        log(f"    -> passed ({len(warns)} warning(s))")
        self.storage.put_json(ANALYSIS, "timeline_v1.json", timeline.to_dict())

        # ---- 9: render preview -> critique -> revise (bounded) ----
        log("\n[7/10] Rendering preview")
        preview = self.storage.path(RENDER, "preview_v1.mp4")
        res = self.renderer.render(timeline, preview, preview=True,
                                   progress=lambda p, m: log(f"    {m}"))
        r.preview_path = res.path
        r.warnings += res.warnings
        log(f"    -> {res.width}x{res.height} @ {res.fps:.0f}fps, {res.duration:.2f}s")

        for rev in range(MAX_REVISIONS):
            log(f"\n[8/10] Inspecting render (pass {rev+1})")
            crit = critique(res.path, timeline, grammar, audio, ledger=self.ledger,
                            cons=cons)
            r.critiques.append(crit)
            log(f"    -> score {crit.score:.1f}/10")
            log(f"    -> {crit.summary}")
            for i in crit.issues:
                log(f"    [{i['severity']}] {i['code']}: {i['message']}")
            if not crit.issues:
                break
            log(f"\n[9/10] Revising (pass {rev+1})")
            # Structural directives need a re-plan/rebuild — mutating clip
            # durations in place would change the running time, which is locked
            # to the music. Those are handled here, where the planner lives.
            actions = {d.get("action") for d in crit.directives()}
            changed = False
            if actions & {"increase_visual_contrast", "increase_cut_density",
                          "decrease_cut_density", "reshape_intensity"}:
                if "increase_cut_density" in actions:
                    pacing_bias *= 1 / float(max(1e-6, next(
                        (d.get("factor", 1.2) for d in crit.directives()
                         if d.get("action") == "increase_cut_density"), 1.2)))
                if "decrease_cut_density" in actions:
                    pacing_bias *= float(next(
                        (1 / d.get("factor", 0.8) for d in crit.directives()
                         if d.get("action") == "decrease_cut_density"), 1.25))
                if "increase_visual_contrast" in actions:
                    diversity += 1.0
                    log(f"    -> raising shot diversity to {diversity:.1f} so "
                        "neighbouring clips differ enough for cuts to read")
                grammar.pacing_multiplier = pacing_bias
                slots = plan_slots(grammar, audio, target_duration,
                                   start_offset=start_offset, ledger=self.ledger,
                                   cons=cons)
                timeline = build_timeline(slots, index, grammar, audio, self.project_config,
                                          music_path=str(music), music_start=start_offset,
                                          intent=instruction, ledger=self.ledger,
                                          diversity=diversity, cons=cons, overrides=ov)
                changed = True
            timeline, mutated = apply_revisions(timeline, crit, grammar, audio,
                                                slots, ledger=self.ledger,
                                                overrides=ov)
            changed = changed or mutated
            if not changed:
                log("    -> no applicable revision; stopping the loop")
                break
            # A re-planned timeline is a NEW object, so the mix must be re-decided
            # rather than inherited: its duration and fade-out no longer match.
            timeline.mix = self._plan_sound(timeline, music, cons, lambda *_: None)
            r.mix = timeline.mix
            issues = validate(timeline, beat_grid=audio.beats)
            if val_errors(issues):
                log("    -> revision produced an invalid timeline; reverting to previous")
                break
            r.revisions += 1
            preview = self.storage.path(RENDER, f"preview_v{rev+2}.mp4")
            res = self.renderer.render(timeline, preview, preview=True,
                                       progress=lambda p, m: log(f"    {m}"))
            r.preview_path = res.path
        else:
            log("    -> revision budget exhausted (bounded loop)")

        if stop_after_preview:
            # AUTOPILOT explores several treatments; rendering each at full scale
            # would cost minutes per candidate for a file it is going to discard.
            # The critic scores the PREVIEW, which is the same edit at a smaller
            # raster, so the comparison is fair.
            r.timeline = timeline
            r.timings["total"] = time.time() - t0
            self.project.constraints = cons.to_dict()
            self.project.timeline = timeline.to_dict()
            return r

        # ---- 10: final render ----
        log("\n[10/10] Rendering final")
        final = self.storage.path(OUTPUT, "alledits_final.mp4")
        fres = self.renderer.render(timeline, final, preview=False,
                                    progress=lambda p, m: log(f"    {m}"))
        r.final_path = fres.path
        r.warnings += fres.warnings
        r.timeline = timeline
        r.timings["total"] = time.time() - t0
        log(f"    -> {fres.width}x{fres.height} @ {fres.fps:.0f}fps, "
            f"{fres.duration:.2f}s")

        self.storage.put_json(ANALYSIS, "timeline_final.json", timeline.to_dict())
        self.storage.put_json(ANALYSIS, "decision_ledger.json", self.ledger.to_dict())
        self.storage.put_json(ANALYSIS, "critiques.json",
                              [c.to_dict() for c in r.critiques])
        self.storage.put_json(ANALYSIS, "constraints.json", cons.to_dict())

        # persist the project so the edit can be reopened and directed further
        self.project.constraints = cons.to_dict()
        self.project.timeline = timeline.to_dict()
        self.project.style_grammar_id = grammar.id
        self.project.record("render",
                            f"{len(timeline.clips)} clips, {timeline.duration:.2f}s, "
                            f"score {r.critiques[-1].score:.1f}/10"
                            if r.critiques else "rendered")
        if self.deliver_profile:
            log(f"\n[--] Mastering for delivery ({self.deliver_profile})")
            out = self.storage.path(OUTPUT, f"alledits_{self.deliver_profile}.mp4")
            try:
                # MUST be the final render (fres), not `res` — that variable
                # still holds the last PREVIEW from the critique loop, which is
                # rendered at half scale, so mastering it silently delivered an
                # upscaled 540x960 preview instead of the finished edit.
                r.master = master_file(fres.path, out, self.deliver_profile,
                                       allow_upscale=self.allow_upscale, log=log)
                for c in r.master.qc.checks:
                    if c.status != "pass":
                        log(f"    {c}")
                log(f"    -> {r.master.qc.summary()}; "
                    f"conformant={r.master.conformant}")
                self.project.master = r.master.to_dict()
            except Exception as e:
                # A failed master must not destroy the edit that succeeded.
                log(f"    ! mastering failed: {e}")
                r.master = None

        r.project_path = self.save_project()
        log(f"    -> project saved to {r.project_path}")
        return r

    def _plan_sound(self, timeline, music, cons, log):
        """Decide the mix, then gate it the same way the picture is gated.

        An invalid mix plan must stop the render rather than be quietly
        downgraded — a silent fallback to the naive full-volume bed would look
        like success while delivering none of the sound work.
        """
        voices = diegetic_voice_tracks(timeline)
        plan = plan_mix(timeline, str(music) if music else None,
                        target=getattr(cons, "loudness_target", None) or "social",
                        voice_tracks=voices, cons=cons, ledger=self.ledger)
        problems = validate_mix(plan)
        if problems:
            raise ValueError("invalid mix plan: " + "; ".join(problems))
        log(f"\n[07] Sound: {len(plan.tracks)} track(s)"
            + (f", {len(voices)} carrying speech" if voices else ""))
        for reason in plan.reasons:
            log(f"    -> {reason}")
        return plan

    def project_settings_dict(self) -> dict:
        p = self.project_config
        return {"width": p.width, "height": p.height, "fps": p.fps,
                "aspect_label": p.aspect_label}
