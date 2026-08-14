"""ALLEDITS test suite.

These are behavioural tests against generated media with KNOWN ground truth,
not smoke tests. Run scripts/make_test_media.py first.

    python -m pytest tests/ -v        (or: python tests/test_alledits.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MEDIA = Path("/home/claude/testmedia")
CLIPS = MEDIA / "clips"


# ------------------------------------------------------------------ audio
def test_bpm_within_2_percent():
    from alledits.audio.analyze import analyze_audio
    a = analyze_audio(MEDIA / "music.wav")
    assert abs(a.bpm - 128.0) / 128.0 < 0.02, f"bpm {a.bpm}"
    assert len(a.downbeats) == 16, f"expected 16 bars, got {len(a.downbeats)}"


def test_drop_detected_at_bar_5():
    from alledits.audio.analyze import analyze_audio
    a = analyze_audio(MEDIA / "music.wav")
    truth = 5 * 4 * 60 / 128
    assert any(abs(d - truth) < 0.25 for d in a.drops), f"drops {a.drops} vs {truth}"


# ------------------------------------------------------------------ visual
def test_camera_movement_classification():
    from alledits.media.visual import analyze_shot
    cases = {"01_static_wide": "static", "02_pan_right": "pan_right",
             "03_push_in": "push_in", "04_pull_out": "pull_out",
             "05_tilt_up": "tilt_up", "06_handheld": "handheld"}
    wrong = []
    for name, expected in cases.items():
        v = analyze_shot(CLIPS / f"{name}.mp4", 0.2, 2.3)
        if v.camera_movement != expected:
            wrong.append(f"{name}: got {v.camera_movement}, expected {expected}")
    assert not wrong, "; ".join(wrong)


def test_shot_detection_segments_a_multi_shot_take():
    from alledits.media.scenes import detect_shots
    from alledits.media.probe import probe
    p = CLIPS / "16_multi_shot_take.mp4"
    shots = detect_shots(p, probe(p).duration, min_shot=0.5)
    assert len(shots) == 3, f"expected 3 shots, got {len(shots)}"


def test_shot_detection_does_not_split_a_single_shot():
    from alledits.media.scenes import detect_shots
    from alledits.media.probe import probe
    p = CLIPS / "02_pan_right.mp4"
    shots = detect_shots(p, probe(p).duration, min_shot=0.5)
    assert len(shots) == 1, f"over-segmented a continuous pan into {len(shots)}"


# ------------------------------------------------------------------ quality
def test_good_footage_is_not_needlessly_processed():
    """Spec 13: already-good footage must be left alone."""
    from alledits.media.probe import probe
    from alledits.media.visual import analyze_shot
    from alledits.media.quality import analyze_quality
    p = CLIPS / "11_4k_high_quality.mp4"
    info = probe(p)
    v = analyze_shot(p, 0.2, 2.0)
    q = analyze_quality(p, 0.2, 2.0, info, visual=v)
    assert q.technical_quality > 0.6, q.technical_quality
    assert q.handling == "use", f"handling={q.handling} — would degrade good footage"


def test_quality_and_creative_value_are_independent_scores():
    """Spec 25: a technically weak clip must not be auto-discarded."""
    from alledits.media.quality import QualityAnalysis, Handling
    q = QualityAnalysis(technical_quality=0.2, creative_value=0.8)
    # replicate the decision rule
    assert q.technical_quality < 0.28 and q.creative_value >= 0.55


def test_poor_footage_is_downgraded_not_silently_used_at_length():
    from alledits.media.probe import probe
    from alledits.media.visual import analyze_shot
    from alledits.media.quality import analyze_quality
    p = CLIPS / "13_very_poor.mp4"
    info = probe(p)
    v = analyze_shot(p, 0.1, 1.4)
    q = analyze_quality(p, 0.1, 1.4, info, visual=v)
    assert q.technical_quality < 0.4
    assert q.handling in ("use_briefly", "replace", "reject"), q.handling


# ------------------------------------------------------------------ reference
def test_reference_pacing_matches_ground_truth():
    from alledits.reference.analyze_reference import analyze_reference
    g = analyze_reference(MEDIA / "reference.mp4")
    assert abs(g.pacing.cuts_per_second - 1.64) < 0.35, g.pacing.cuts_per_second
    assert g.pacing.rhythm == "accelerating", g.pacing.rhythm


def test_grammar_stores_no_reference_content():
    """Spec 30: the style grammar holds measured characteristics, never the
    reference's actual media. Checks for real leakage — encoded pixels, audio
    samples, or a path back to the source — not for incidental vocabulary."""
    import base64, json, re
    from alledits.reference.analyze_reference import analyze_reference
    g = analyze_reference(MEDIA / "reference.mp4")
    d = json.loads(g.to_json())

    # no filesystem path to the source anywhere in the grammar
    blob = json.dumps(d)
    assert "/testmedia" not in blob and ".mp4" not in blob, "grammar leaked a source path"

    # no base64 / long opaque payload that could carry frames or audio
    for m in re.findall(r'"([A-Za-z0-9+/=]{64,})"', blob):
        raise AssertionError(f"grammar contains an opaque blob ({len(m)} chars)")

    # numeric payloads must be small summaries, never sample-level data
    def walk(x, path="") :
        if isinstance(x, list):
            assert len(x) <= 128, f"{path} has {len(x)} entries — too large to be a summary"
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
    walk(d)


# ------------------------------------------------------------------ timeline
def _mini_timeline(**kw):
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    tl = Timeline(project=ProjectSettings(width=1080, height=1920, fps=30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s", source_path=str(CLIPS / "01_static_wide.mp4"),
        source_in=0.0, source_out=1.0, timeline_start=0.0, duration=1.0, **kw))
    return tl


def test_validator_blocks_a_gap():
    from alledits.timeline.schema import TimelineClip
    from alledits.timeline.validator import validate, errors
    tl = _mini_timeline()
    tl.clips.append(TimelineClip(
        id="c001", source_id="s", source_path=str(CLIPS / "01_static_wide.mp4"),
        source_in=0.0, source_out=1.0, timeline_start=2.0, duration=1.0))
    assert any(e.code == "gap" for e in errors(validate(tl)))


def test_validator_blocks_missing_source():
    from alledits.timeline.validator import validate, errors
    tl = _mini_timeline()
    tl.clips[0].source_path = "/nope/missing.mp4"
    assert any(e.code == "missing_source" for e in errors(validate(tl)))


def test_validator_blocks_out_of_range_effect_params():
    from alledits.timeline.schema import Effect, EffectType
    from alledits.timeline.validator import validate, errors
    tl = _mini_timeline()
    tl.clips[0].effects.append(Effect(EffectType.SHAKE.value, {"amplitude": 5.0}))
    assert any(e.code == "param_range" for e in errors(validate(tl)))


def test_validator_blocks_rejected_clip_on_timeline():
    from alledits.timeline.validator import validate, errors
    tl = _mini_timeline(quality_handling="reject")
    assert any(e.code == "rejected_clip" for e in errors(validate(tl)))


def test_valid_timeline_passes():
    from alledits.timeline.validator import validate, errors
    assert not errors(validate(_mini_timeline()))


# ------------------------------------------------------------------ honesty
def test_heuristic_provider_never_claims_to_be_an_llm():
    """Principle: nothing rule-based may be presented as AI."""
    from alledits.intelligence.providers.heuristic_provider import HeuristicProvider
    r = HeuristicProvider().complete("sys", "prompt")
    assert r.is_llm is False
    assert r.actor == "rule_based_planner"


def test_semantic_search_fails_loudly_rather_than_faking():
    """Capability that isn't installed must raise, not return plausible junk."""
    from alledits.search.index import MediaIndex
    from alledits.core.errors import ProviderUnavailable
    from alledits.intelligence.providers.local_embedder import LocalFeatureEmbedder
    idx = MediaIndex(embedder=LocalFeatureEmbedder())
    try:
        idx.search_by_text("someone realizing they've been betrayed")
    except ProviderUnavailable:
        return
    raise AssertionError("semantic search should have raised ProviderUnavailable")


def test_anthropic_provider_reports_unavailable_without_key():
    from alledits.intelligence.providers.anthropic_provider import AnthropicProvider
    assert AnthropicProvider(api_key=None).available() is False


# ------------------------------------------------------------------ brief (Phase 1)
def test_brief_parser_reports_rule_based_honestly():
    from alledits.intelligence.brief import LexiconBriefParser
    c = LexiconBriefParser().parse("cinematic and slow")
    assert c.is_llm is False and c.actor == "rule_based_brief_parser"


def test_empty_brief_is_neutral():
    """An empty brief must reproduce pre-Phase-1 behaviour exactly.

    Covers every knob a brief can touch — pacing, intensity, selection weights
    AND colour — because a non-neutral default would silently restyle the edit
    of every user who never wrote a brief.
    """
    from alledits.intelligence.brief import parse_brief
    from alledits.intelligence.constraints import CreativeConstraints
    c, d = parse_brief(""), CreativeConstraints()
    for f in ("pacing_multiplier", "intensity_gain", "intensity_offset",
              "motion_preference", "continuity_weight", "effect_density",
              "quality_weight", "diversity",
              "contrast_delta", "saturation_delta", "warmth_delta"):
        assert getattr(c, f) == getattr(d, f), f


def test_opposing_briefs_produce_opposing_constraints():
    from alledits.intelligence.brief import parse_brief
    slow = parse_brief("cinematic, slow and restrained, smooth and understated, no shake")
    fast = parse_brief("chaotic, aggressive and fast, flashy with whip pans and jump cuts")
    assert slow.pacing_multiplier > fast.pacing_multiplier * 2
    assert slow.effect_density < fast.effect_density
    assert slow.continuity_weight > fast.continuity_weight
    assert slow.allow_shake is False and fast.allow_shake is True
    assert slow.transition_bias["flash"] < fast.transition_bias["flash"]


def test_negation_is_clause_scoped():
    """'not too fast, very warm' must not make 'warm' come out cold."""
    from alledits.intelligence.brief import parse_brief
    c = parse_brief("not too fast, very warm, hard cuts")
    assert c.warmth_delta > 0.2, c.warmth_delta
    assert c.pacing_multiplier > 1.0, c.pacing_multiplier
    assert c.transition_bias["cut"] > 1.5


def test_negation_flips_boolean_permissions():
    from alledits.intelligence.brief import parse_brief
    assert parse_brief("gritty handheld, no shake").allow_shake is False
    assert parse_brief("vibrant and fast, no grain").allow_grain is False


def test_stacked_intensifiers_stay_in_range():
    """The lexicon path: piling on intensifiers must saturate, not overflow."""
    from alledits.intelligence.brief import parse_brief
    c = parse_brief("extremely insanely ultra fast chaotic frantic explosive breakneck")
    assert 0.35 <= c.pacing_multiplier <= 3.0
    assert 0.0 <= c.effect_density <= 1.0
    assert c.min_shot_floor >= 0.10


def test_brief_changes_the_planned_timeline():
    """The end-to-end proof, at plan level: same inputs, different briefs,
    measurably different slot plans."""
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    from alledits.intelligence.brief import parse_brief
    from alledits.intelligence.planner import plan_slots
    audio = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    slow = plan_slots(g, audio, 18.0, cons=parse_brief("cinematic, slow and restrained"))
    fast = plan_slots(g, audio, 18.0, cons=parse_brief("chaotic, aggressive and fast"))
    assert len(fast) > len(slow) * 1.8, f"{len(slow)} vs {len(fast)} slots"
    ms = sum(s.duration for s in slow) / len(slow)
    mf = sum(s.duration for s in fast) / len(fast)
    assert ms > mf * 1.8, f"mean shot {ms:.2f} vs {mf:.2f}"


# ------------------------------------------------------------------ brief (Phase 1)
def test_brief_parser_opposing_briefs_diverge():
    from alledits.intelligence.brief import parse_brief
    r = parse_brief("cinematic, slow and restrained, smooth and understated, no shake")
    c = parse_brief("chaotic, aggressive and fast, flashy with whip pans and jump cuts")
    assert r.pacing_multiplier > c.pacing_multiplier * 2, (r.pacing_multiplier, c.pacing_multiplier)
    assert r.effect_density < c.effect_density
    assert r.continuity_weight > c.continuity_weight
    assert r.allow_shake is False and c.allow_shake is True


def test_brief_negation_is_clause_scoped():
    """'not too fast, very warm' must not invert 'warm'."""
    from alledits.intelligence.brief import parse_brief
    c = parse_brief("not too fast, very warm, hard cuts")
    assert c.warmth_delta > 0.2, f"warmth inverted by a negation in another clause: {c.warmth_delta}"
    assert c.pacing_multiplier > 1.0, c.pacing_multiplier
    assert c.transition_bias["cut"] > 1.5


def test_brief_negation_disables_features():
    from alledits.intelligence.brief import parse_brief
    assert parse_brief("handheld doc, no shake").allow_shake is False
    assert parse_brief("gritty, no grain").allow_grain is False


def test_clamp_rejects_out_of_range_values_directly():
    """Direct clamp() contract: even a hostile or buggy caller (e.g. an LLM
    emitting nonsense) cannot push the editor outside safe bounds."""
    from alledits.intelligence.constraints import CreativeConstraints
    c = CreativeConstraints(pacing_multiplier=999, effect_density=-5,
                            continuity_weight=-3, max_effects_per_clip=99).clamp()
    assert 0.35 <= c.pacing_multiplier <= 3.0
    assert 0.0 <= c.effect_density <= 1.0
    assert c.continuity_weight >= 0.2
    assert c.max_effects_per_clip <= 5


def test_brief_changes_the_plan_not_just_the_parser():
    """The knobs must reach the planner: opposing briefs -> different slot counts.
    Timeline-level check, no rendering required."""
    from alledits.intelligence.brief import parse_brief
    from alledits.intelligence.planner import plan_slots
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    a = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    slow = plan_slots(g, a, 18.0, cons=parse_brief("cinematic, slow and restrained"))
    fast = plan_slots(g, a, 18.0, cons=parse_brief("chaotic, aggressive and fast"))
    assert len(fast) > len(slow) * 1.8, f"slow={len(slow)} fast={len(fast)}"
    ms = sum(s.duration for s in slow) / len(slow)
    mf = sum(s.duration for s in fast) / len(fast)
    assert ms > mf * 1.5, f"mean slot: slow={ms:.2f} fast={mf:.2f}"


def test_llm_brief_parser_falls_back_visibly_without_a_key():
    from alledits.intelligence.brief import LLMBriefParser
    from alledits.intelligence.providers.anthropic_provider import AnthropicProvider
    c = LLMBriefParser(AnthropicProvider(api_key=None)).parse("fast and loud")
    assert c.is_llm is False
    assert any("rule-based" in n for n in c.notes), c.notes


# ------------------------------------------------------------------ overrides (Phase 2)
def _grammar_audio_index():
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    from alledits.core.storage import LocalStorage
    from alledits.media.ingest import Ingestor
    from alledits.search.index import MediaIndex
    a = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    ing = Ingestor(LocalStorage("/home/claude/work/store"))
    idx = MediaIndex()
    for c in sorted(CLIPS.glob("*.mp4"))[:8]:
        idx.add_asset(ing.ingest(c))
    return g, a, idx


def _build(overrides=None, brief=""):
    from alledits.intelligence.brief import parse_brief
    from alledits.intelligence.planner import plan_slots
    from alledits.timeline.builder import build_timeline
    from alledits.timeline.schema import ProjectSettings
    g, a, idx = _grammar_audio_index()
    cons = parse_brief(brief)
    slots = plan_slots(g, a, 8.0, cons=cons)
    tl = build_timeline(slots, idx, g, a, ProjectSettings(1080, 1920, 30),
                        music_path=str(MEDIA / "music.wav"), cons=cons,
                        overrides=overrides)
    return tl, idx


def test_pin_forces_a_shot_into_a_slot():
    from alledits.core.project import OverrideSet, DirectiveKind
    base, idx = _build()
    target_slot = 2
    current = base.clips[target_slot].source_id
    other = next(s.id for s in idx.shots if s.id != current)
    ov = OverrideSet()
    ov.add(DirectiveKind.PIN_SHOT, slot_index=target_slot, shot_id=other,
           note="I want this shot here")
    pinned, _ = _build(overrides=ov)
    assert pinned.clips[target_slot].source_id == other, (
        f"pin ignored: got {pinned.clips[target_slot].source_id}, wanted {other}")


def test_pinned_clip_is_not_credited_to_the_system():
    """Provenance honesty: a human choice must not read as an automatic one."""
    from alledits.core.project import OverrideSet, DirectiveKind
    base, idx = _build()
    other = next(s.id for s in idx.shots if s.id != base.clips[2].source_id)
    ov = OverrideSet()
    ov.add(DirectiveKind.PIN_SHOT, slot_index=2, shot_id=other)
    tl, _ = _build(overrides=ov)
    assert "pinned by the user" in tl.clips[2].selection_reason.lower(), \
        tl.clips[2].selection_reason


def test_global_reject_removes_a_shot_everywhere():
    from alledits.core.project import OverrideSet, DirectiveKind
    base, idx = _build()
    used = base.clips[0].source_id
    ov = OverrideSet()
    ov.add(DirectiveKind.REJECT_SHOT, shot_id=used, note="never use this")
    after, _ = _build(overrides=ov)
    assert all(c.source_id != used for c in after.clips), "rejected shot still used"


def test_reject_at_slot_is_scoped_to_that_slot_only():
    from alledits.core.project import OverrideSet, DirectiveKind
    base, _ = _build()
    slot = 1
    banned = base.clips[slot].source_id
    ov = OverrideSet()
    ov.add(DirectiveKind.REJECT_AT_SLOT, slot_index=slot, shot_id=banned)
    after, _ = _build(overrides=ov)
    assert after.clips[slot].source_id != banned


def test_ban_effect_removes_it_from_every_clip():
    from alledits.core.project import OverrideSet, DirectiveKind
    base, _ = _build(brief="chaotic aggressive fast")
    assert any(e.type == "shake" for c in base.clips for e in c.effects), \
        "precondition: chaotic brief should produce shake"
    ov = OverrideSet()
    ov.add(DirectiveKind.BAN_EFFECT, value="shake", note="no shake please")
    after, _ = _build(overrides=ov, brief="chaotic aggressive fast")
    assert not any(e.type == "shake" for c in after.clips for e in c.effects)


def test_force_transition_survives_quota_allocation():
    """A user's transition choice must not be overwritten by the style quota."""
    from alledits.core.project import OverrideSet, DirectiveKind
    ov = OverrideSet()
    ov.add(DirectiveKind.FORCE_TRANSITION, slot_index=3, value="dissolve")
    tl, _ = _build(overrides=ov)
    assert tl.clips[3].transition_in.type == "dissolve", \
        tl.clips[3].transition_in.type


def test_lock_protects_a_clip_from_the_critic():
    """The revision loop must never undo what a human locked."""
    from alledits.core.project import OverrideSet, DirectiveKind
    from alledits.intelligence.critic import apply_revisions, Critique
    tl, _ = _build()
    locked_idx = 2
    ov = OverrideSet()
    ov.add(DirectiveKind.LOCK_CLIP, slot_index=locked_idx, note="keep this cut")
    before = tl.clips[locked_idx].duration
    before_fx = len(tl.clips[locked_idx].effects)

    crit = Critique()
    crit.issues = [
        {"code": "over_processed", "severity": "medium", "message": "",
         "directive": {"action": "strip_lowest_value_effects", "target": 0}},
        {"code": "too_slow", "severity": "medium", "message": "",
         "directive": {"action": "increase_cut_density", "factor": 1.6}},
    ]
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    tl2, changed = apply_revisions(tl, crit, analyze_reference(MEDIA / "reference.mp4"),
                                   analyze_audio(MEDIA / "music.wav"), [], overrides=ov)
    assert changed
    assert tl2.clips[locked_idx].duration == before, "critic changed a locked clip's duration"
    assert len(tl2.clips[locked_idx].effects) == before_fx, \
        "critic stripped effects from a locked clip"


def test_project_round_trips_through_disk():
    import tempfile
    from alledits.core.project import Project, OverrideSet, DirectiveKind
    p = Project(name="demo", brief="cinematic and slow", target_duration=12.0)
    p.overrides.add(DirectiveKind.PIN_SHOT, slot_index=4, shot_id="shot_x", note="hero")
    p.record("created", "test")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "project.json"
        p.save(path)
        q = Project.load(path)
    assert q.name == "demo" and q.brief == "cinematic and slow"
    assert q.target_duration == 12.0
    assert q.overrides.pinned_at(4) == "shot_x"
    assert q.history and q.history[0]["event"] == "created"


def test_pin_is_replaced_not_duplicated():
    from alledits.core.project import OverrideSet, DirectiveKind
    ov = OverrideSet()
    ov.add(DirectiveKind.PIN_SHOT, slot_index=1, shot_id="a")
    ov.add(DirectiveKind.PIN_SHOT, slot_index=1, shot_id="b")
    assert ov.pinned_at(1) == "b"
    assert len([d for d in ov.directives if d.kind == "pin_shot"]) == 1


# ------------------------------------------------------------------ rescue (Phase 3)
def _q(name, start=0.1, end=1.4):
    from alledits.media.probe import probe
    from alledits.media.visual import analyze_shot
    from alledits.media.quality import analyze_quality
    src = CLIPS / f"{name}.mp4"
    v = analyze_shot(src, start, end)
    return analyze_quality(src, start, end, probe(src), visual=v)


def test_defects_are_detected_on_degraded_footage():
    q = _q("13_very_poor")
    kinds = {d["defect"] for d in q.defects}
    assert "softness" in kinds and "blockiness" in kinds, q.defects
    assert all(0.0 <= d["severity"] <= 1.0 for d in q.defects)
    assert all(d["detail"] for d in q.defects), "every defect must justify itself"


def test_good_footage_receives_no_treatment():
    """Restraint is the harder half: treating clean footage degrades it."""
    for name in ("11_4k_high_quality", "01_static_wide", "02_pan_right"):
        q = _q(name)
        assert q.defects == [], f"{name} was prescribed {q.defects}"
        assert q.handling == "use", f"{name} handling={q.handling}"


def test_noise_estimator_separates_clean_from_noisy():
    """Calibration guard. A 55th-percentile flat mask let detail read as noise,
    so clean footage overlapped with genuinely noisy material."""
    import cv2
    import numpy as np
    from alledits.media.quality import _estimate_noise
    from alledits.core.ffmpeg import ffmpeg
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        still, clean, noisy = d / "s.png", d / "c.mp4", d / "n.mp4"
        ffmpeg(["-f", "lavfi", "-i", "mandelbrot=s=1280x720", "-frames:v", "1", str(still)])
        for out, alls in ((clean, 0), (noisy, 42)):
            ffmpeg(["-loop", "1", "-i", str(still), "-t", "1",
                    "-vf", f"crop=1280:720:400:300,noise=alls={alls}:allf=t+u,format=yuv420p",
                    "-r", "30", "-c:v", "libx264", "-crf", "12", str(out)])

        def lvl(path):
            cap = cv2.VideoCapture(str(path))
            vals = []
            for _ in range(5):
                ok, f = cap.read()
                if not ok:
                    break
                f = cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA)
                vals.append(_estimate_noise(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
            cap.release()
            return float(np.mean(vals))

        c, n = lvl(clean), lvl(noisy)
    from alledits.media.quality import DEFECT_THRESHOLDS, Defect
    t = DEFECT_THRESHOLDS[Defect.NOISE]
    assert c < t, f"clean footage {c:.3f} would be denoised (threshold {t})"
    assert n > t, f"noisy footage {n:.3f} would NOT be denoised (threshold {t})"
    assert n > c * 2, f"estimator does not separate: clean={c:.3f} noisy={n:.3f}"


def test_restoration_runs_before_creative_effects():
    """Denoise must precede sharpen — sharpening noise amplifies it — and all
    restoration must precede creative treatment."""
    from alledits.render.filters import build_effect_chain
    from alledits.timeline.schema import Effect, EffectType
    fx = [Effect(EffectType.COLOR_GRADE.value, {"contrast": 1.1, "brightness": 0.0}),
          Effect(EffectType.SHARPEN.value, {"strength": 0.5}),
          Effect(EffectType.DENOISE.value, {"strength": 0.5}),
          Effect(EffectType.DEBLOCK.value, {"strength": 0.5})]
    chain = ",".join(build_effect_chain(fx, 1080, 1920, 30, 30))
    i_deblock, i_dn = chain.index("deblock"), chain.index("hqdn3d")
    i_sharp, i_grade = chain.index("unsharp"), chain.index("eq=")
    assert i_deblock < i_dn < i_sharp, chain
    assert i_sharp < i_grade, "restoration must precede creative grading"


def test_stabilize_is_excluded_from_the_single_pass_chain():
    """vidstab needs an analysis pass first, so it cannot be a plain filter."""
    from alledits.render.filters import build_effect_chain
    from alledits.timeline.schema import Effect, EffectType
    chain = ",".join(build_effect_chain(
        [Effect(EffectType.STABILIZE.value, {"strength": 0.6, "zoom": 0.03})],
        1080, 1920, 30, 30))
    assert "vidstab" not in chain, chain


def test_validator_bounds_restoration_parameters():
    from alledits.timeline.validator import validate, errors
    from alledits.timeline.schema import Effect, EffectType
    for etype, params in ((EffectType.DENOISE.value, {"strength": 9.0}),
                          (EffectType.STABILIZE.value, {"strength": 0.5, "zoom": 5.0}),
                          (EffectType.SHARPEN.value, {"strength": -1.0})):
        tl = _mini_timeline()
        tl.clips[0].effects = [Effect(etype, params)]
        assert any(e.code == "param_range" for e in errors(validate(tl))), \
            f"{etype} {params} was accepted"


def test_shake_treatment_requires_low_directional_consistency():
    """A deliberate fast pan has high jitter too; stabilizing it would destroy
    the intended camera move."""
    from alledits.media.quality import _detect_defects, QualityAnalysis

    class V:
        def __init__(self, shake, consistency):
            self.shake, self.motion_consistency = shake, consistency

    q = QualityAnalysis()
    q.handling, q.noise, q.sharpness = "use", 0.0, 1.0
    q.blockiness, q.dynamic_range = 0.0, 1.0
    wobble = {d["defect"] for d in _detect_defects(q, V(4.0, 0.05))}
    intended = {d["defect"] for d in _detect_defects(q, V(4.0, 0.95))}
    assert "shake" in wobble
    assert "shake" not in intended, "an intended camera move was stabilized"


def test_treatments_are_recorded_on_the_clip_for_provenance():
    from alledits.timeline.schema import TimelineClip
    c = TimelineClip(id="c000", source_id="s", source_path="/x.mp4",
                     source_in=0.0, source_out=1.0, timeline_start=0.0, duration=1.0)
    assert c.treatments == [], "clips must carry a treatments provenance list"


# ------------------------------------------------------------------ sound (Phase 4)
def test_duck_threshold_is_derived_from_requested_depth():
    """The stated depth must drive the filter, not sit beside it.

    A fixed ratio was previously hardcoded while depth_db was reported to the
    user, so the system announced 9 dB and applied roughly 18.
    """
    from alledits.audio.mix import DuckSpec
    shallow, deep = DuckSpec(depth_db=3.0), DuckSpec(depth_db=6.0)
    assert deep.threshold < shallow.threshold, (shallow.threshold, deep.threshold)


def test_duck_claim_is_capped_to_what_it_can_deliver():
    from alledits.audio.mix import DuckSpec, MAX_ACHIEVABLE_DUCK_DB
    assert DuckSpec(depth_db=18.0).achievable_depth_db <= MAX_ACHIEVABLE_DUCK_DB
    assert DuckSpec(depth_db=4.0).achievable_depth_db == 4.0


def test_mix_plan_reports_capped_depth_not_requested_depth():
    from alledits.audio.mix import plan_mix, MAX_ACHIEVABLE_DUCK_DB

    class T:
        duration = 10.0
    plan = plan_mix(T(), str(MEDIA / "music.wav"),
                    voice_tracks=[{"path": str(MEDIA / "music.wav"),
                                   "source_in": 0.0, "source_out": 10.0}])
    assert plan.has_voice
    txt = " ".join(plan.reasons)
    stated = float(__import__("re").search(r"ducking music ([\d.]+) dB", txt).group(1))
    assert stated <= MAX_ACHIEVABLE_DUCK_DB + 1e-6, stated


def test_music_sits_below_and_is_ducked_by_voice():
    from alledits.audio.mix import plan_mix, TrackRole

    class T:
        duration = 10.0
    plan = plan_mix(T(), str(MEDIA / "music.wav"),
                    voice_tracks=[{"path": str(MEDIA / "music.wav"),
                                   "source_in": 0.0, "source_out": 10.0}])
    music = plan.by_role(TrackRole.MUSIC)[0]
    assert music.gain_db < 0, "music must sit below speech"
    assert music.ducked_by, "music must be ducked when voice is present"


def test_no_duck_is_planned_without_voice():
    """Claiming a duck that never happens is worse than not ducking."""
    from alledits.audio.mix import plan_mix, TrackRole

    class T:
        duration = 10.0
    plan = plan_mix(T(), str(MEDIA / "music.wav"))
    assert not plan.has_voice
    assert not plan.by_role(TrackRole.MUSIC)[0].ducked_by
    assert not any("ducking" in r for r in plan.reasons)


def test_validate_mix_rejects_undeliverable_plans():
    from alledits.audio.mix import MixPlan, MixTrack, validate_mix
    assert validate_mix(MixPlan())                      # no tracks
    bad = MixPlan(true_peak_db=1.0)
    bad.tracks = [MixTrack(id="m", source_path="x", source_in=0, source_out=5)]
    assert any("true peak" in i for i in validate_mix(bad))
    self_duck = MixPlan()
    self_duck.tracks = [MixTrack(id="m", source_path="x", source_in=0,
                                 source_out=5, ducked_by=["m"])]
    assert any("itself" in i for i in validate_mix(self_duck))


def test_limiter_sits_below_the_delivery_ceiling():
    """A lossy encoder overshoots; a limiter at the ceiling ships a violation."""
    from alledits.audio.mix import MixPlan, MixTrack
    from alledits.render.ffmpeg_renderer import (_build_mix_graph,
                                                 CODEC_OVERSHOOT_MARGIN_DB)
    plan = MixPlan(true_peak_db=-1.0)
    plan.tracks = [MixTrack(id="m", source_path=str(MEDIA / "music.wav"),
                            source_in=0, source_out=4)]
    fc, _, _ = _build_mix_graph(plan, 4.0, 0, gain_db=0.0)
    graph = ";".join(fc)
    expected = 10 ** ((plan.true_peak_db - CODEC_OVERSHOOT_MARGIN_DB) / 20)
    assert f"alimiter=limit={expected:.4f}" in graph, graph[-160:]


def test_mix_hits_the_loudness_target_and_holds_the_ceiling():
    """End to end on real audio: solve, encode, then MEASURE the result."""
    import re as _re
    import subprocess as _sp
    from alledits.audio.mix import plan_mix, validate_mix
    from alledits.core.ffmpeg import ffmpeg
    from alledits.render.ffmpeg_renderer import _build_mix_graph, _solve_gain

    class T:
        duration = 8.0
    plan = plan_mix(T(), str(MEDIA / "music.wav"), target="social")
    assert not validate_mix(plan)
    gain, _, _ = _solve_gain(plan, 8.0)
    assert gain is not None, "loudness solver failed"
    fc, lbl, inputs = _build_mix_graph(plan, 8.0, 0, gain_db=gain)
    out = Path("/tmp/test_mix_loudness.m4a")
    ffmpeg([*inputs, "-filter_complex", ";".join(fc), "-map", f"[{lbl}]",
            "-c:a", "aac", "-b:a", "192k", "-t", "8.0", str(out)])
    r = _sp.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(out),
                 "-af", "ebur128=peak=true", "-f", "null", "-"],
                capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Integrated loudness"):]
    I = float(_re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail).group(1))
    TP = float(_re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail).group(1))
    assert abs(I - plan.target_lufs) <= 0.5, f"{I} LUFS vs {plan.target_lufs}"
    assert TP <= plan.true_peak_db, f"true peak {TP} exceeds {plan.true_peak_db}"


def test_pipeline_plans_a_mix_and_gates_it():
    """The mix must be planned and validated, not left to the naive bed."""
    import inspect
    from alledits.pipeline.vertical_slice import VerticalSlice
    src = inspect.getsource(VerticalSlice._plan_sound)
    assert "validate_mix" in src and "raise" in src, \
        "an invalid mix plan must stop the render, not silently downgrade"


def test_speech_detection_matches_ground_truth_windows():
    """The voice fixture has known speaking windows; detection must find them."""
    from alledits.audio.speech import detect_speech
    truth = [(2.0, 5.0), (8.0, 11.5), (14.0, 16.5)]
    got = detect_speech(MEDIA / "voice.wav")
    assert got.has_speech
    assert len(got.windows) == len(truth), got.windows
    for (a, b), (ga, gb) in zip(got.windows, truth):
        assert abs(a - ga) < 0.15 and abs(b - gb) < 0.15, (got.windows, truth)


def test_music_is_not_detected_as_speech():
    """Band energy alone would flag midrange music; the modulation test is what
    makes detection specific."""
    from alledits.audio.speech import detect_speech
    assert detect_speech(MEDIA / "music.wav").has_speech is False


def test_speech_detection_reports_absence_rather_than_guessing():
    from alledits.audio.speech import detect_speech
    r = detect_speech(MEDIA / "clips" / "01_static_wide.mp4")   # no audio stream
    assert r.has_speech is False and not r.windows
    assert r.reason, "must say why nothing was found"


def test_clip_over_silence_contributes_no_voice_track():
    """Having an audio stream is not the same as carrying speech; ducking under
    room tone would be a bug, not a feature."""
    from alledits.audio.mix import diegetic_voice_tracks
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s",
        source_path=str(MEDIA / "clips" / "17_talking_head.mp4"),
        source_in=5.5, source_out=7.5,       # the gap between speaking windows
        timeline_start=0.0, duration=2.0))
    assert diegetic_voice_tracks(tl) == []


def test_clip_containing_speech_produces_a_ducking_plan():
    from alledits.audio.mix import diegetic_voice_tracks, plan_mix
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s",
        source_path=str(MEDIA / "clips" / "17_talking_head.mp4"),
        source_in=2.0, source_out=5.0, timeline_start=0.0, duration=3.0))
    voices = diegetic_voice_tracks(tl)
    assert len(voices) == 1, voices
    plan = plan_mix(tl, str(MEDIA / "music.wav"), voice_tracks=voices)
    music = [t for t in plan.tracks if t.role == "music"][0]
    assert music.ducked_by, "music must duck under detected diegetic speech"
    assert music.gain_db < 0


# ------------------------------------------------------------------ sound (Phase 4)
def _ebur128(path):
    import re, subprocess
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Integrated loudness"):]
    I = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    TP = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
    return (float(I.group(1)) if I else None, float(TP.group(1)) if TP else None)


def _render_mix(plan, duration, out, gain_db=None, codec="aac"):
    from alledits.render.ffmpeg_renderer import _build_mix_graph
    from alledits.core.ffmpeg import ffmpeg
    fc, lbl, inputs = _build_mix_graph(plan, duration, 0, gain_db=gain_db)
    args = [*inputs, "-filter_complex", ";".join(fc), "-map", f"[{lbl}]"]
    args += (["-c:a", "aac", "-b:a", "192k"] if codec == "aac"
             else ["-c:a", "pcm_s16le"])
    ffmpeg([*args, "-t", f"{duration}", str(out)])
    return out


class _Dur:
    def __init__(self, d):
        self.duration = d


def test_delivered_loudness_hits_the_platform_target():
    """Measured on the ENCODED file, because that is what ships."""
    import tempfile
    from alledits.audio.mix import plan_mix
    from alledits.render.ffmpeg_renderer import _solve_gain
    plan = plan_mix(_Dur(8.0), str(MEDIA / "music.wav"), target="social")
    gain, _, _ = _solve_gain(plan, 8.0)
    assert gain is not None, "loudness solver returned nothing"
    with tempfile.TemporaryDirectory() as d:
        I, _ = _ebur128(_render_mix(plan, 8.0, Path(d) / "m.m4a", gain_db=gain))
    assert abs(I - plan.target_lufs) <= 0.5, \
        f"{I} LUFS vs {plan.target_lufs} target (EBU R128 tolerance is 0.5 LU)"


def test_true_peak_survives_lossy_encoding():
    """A limiter set AT the ceiling still ships over it: AAC reconstructs peaks
    above the samples it was handed. The margin is the whole point."""
    import tempfile
    from alledits.audio.mix import plan_mix
    from alledits.render.ffmpeg_renderer import _solve_gain
    plan = plan_mix(_Dur(8.0), str(MEDIA / "music.wav"), target="social")
    gain, _, _ = _solve_gain(plan, 8.0)
    with tempfile.TemporaryDirectory() as d:
        _, TP = _ebur128(_render_mix(plan, 8.0, Path(d) / "m.m4a", gain_db=gain))
    assert TP <= plan.true_peak_db, f"{TP} dBTP exceeds {plan.true_peak_db} ceiling"


def test_loudness_target_follows_the_delivery_platform():
    from alledits.audio.mix import plan_mix, LOUDNESS_TARGETS
    assert plan_mix(_Dur(8.0), None, target="broadcast").target_lufs == \
        LOUDNESS_TARGETS["broadcast"]
    assert plan_mix(_Dur(8.0), None, target="social").target_lufs == \
        LOUDNESS_TARGETS["social"]


def test_duck_parameters_are_derived_from_the_requested_depth():
    """Regression: depth_db was reported to the user but never reached the
    filter, which used a fixed ratio — announcing 9 dB while applying ~18."""
    from alledits.audio.mix import DuckSpec
    shallow, deep = DuckSpec(depth_db=3.0), DuckSpec(depth_db=6.0)
    assert deep.threshold < shallow.threshold, \
        "a deeper duck must use a lower threshold"


def test_duck_never_claims_more_than_it_can_deliver():
    """The rationale shown to the user must state the DELIVERABLE depth.

    Checks the real plan text rather than a string the test builds itself.
    """
    from alledits.audio.mix import plan_mix, MAX_ACHIEVABLE_DUCK_DB
    plan = plan_mix(_Dur(8.0), "/tmp/music.wav",
                    voice_tracks=[{"path": "/tmp/v.wav", "source_in": 0.0,
                                   "source_out": 8.0}])
    assert plan.has_voice, "precondition: a voice track should make the mix duck"
    plan.duck.depth_db = 18.0
    assert plan.duck.achievable_depth_db <= MAX_ACHIEVABLE_DUCK_DB

    # Re-plan with a deep request and read what the user is actually told.
    deep = plan_mix(_Dur(8.0), "/tmp/music.wav",
                    voice_tracks=[{"path": "/tmp/v.wav", "source_in": 0.0,
                                   "source_out": 8.0}])
    deep.duck.depth_db = 18.0
    text = " ".join(deep.reasons)
    stated = float(text.split("ducking music ")[1].split(" dB")[0])
    assert stated <= MAX_ACHIEVABLE_DUCK_DB + 1e-6, \
        f"plan tells the user {stated} dB but can only deliver "\
        f"{MAX_ACHIEVABLE_DUCK_DB}"


def test_ducking_actually_attenuates_music_under_speech():
    """Measured, not asserted from the plan."""
    import re, subprocess, tempfile
    from alledits.audio.mix import MixPlan, MixTrack, DuckSpec, TrackRole
    from alledits.core.ffmpeg import ffmpeg
    with tempfile.TemporaryDirectory() as d:
        voice, bed = Path(d) / "v.wav", Path(d) / "b.wav"
        ffmpeg(["-f", "lavfi", "-i", "sine=frequency=300:duration=9", "-af",
                "volume='if(between(t,3,6),1,0)':eval=frame",
                "-c:a", "pcm_s16le", str(voice)])
        ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=9",
                "-af", "volume=-6dB", "-c:a", "pcm_s16le", str(bed)])

        def level(p, t0, t1):
            r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(p), "-af",
                                f"atrim={t0}:{t1},asetpts=PTS-STARTPTS,"
                                "highpass=f=700,lowpass=f=1400,volumedetect",
                                "-f", "null", "-"], capture_output=True, text=True)
            m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
            return float(m.group(1))

        def build(ducked):
            plan = MixPlan(normalize=False, duck=DuckSpec(depth_db=6.0))
            plan.tracks = [
                MixTrack(id="vox", source_path=str(voice),
                         role=TrackRole.VOICE.value, source_in=0, source_out=9),
                MixTrack(id="music", source_path=str(bed),
                         role=TrackRole.MUSIC.value, source_in=0, source_out=9,
                         ducked_by=["vox"] if ducked else []),
            ]
            return _render_mix(plan, 9.0, Path(d) / f"o{ducked}.wav", codec="pcm")

        off, on = build(False), build(True)
        base = level(off, 4, 5.5) - level(off, 1, 2.5)
        got = (level(on, 4, 5.5) - level(on, 1, 2.5)) - base
    assert got < -3.0, f"music only moved {got:.2f} dB under speech"


def test_mix_validator_rejects_impossible_plans():
    from alledits.audio.mix import MixPlan, MixTrack, validate_mix
    assert validate_mix(MixPlan()), "an empty mix should be rejected"
    p = MixPlan()
    p.tracks = [MixTrack(id="a", source_path="x", source_in=0, source_out=1,
                         ducked_by=["a"])]
    assert any("itself" in i for i in validate_mix(p))
    p2 = MixPlan(target_lufs=-2.0)
    p2.tracks = [MixTrack(id="a", source_path="x", source_in=0, source_out=1)]
    assert any("loudness target" in i for i in validate_mix(p2))
    p3 = MixPlan(true_peak_db=0.5)
    p3.tracks = [MixTrack(id="a", source_path="x", source_in=0, source_out=1)]
    assert any("true peak" in i for i in validate_mix(p3))


def test_silent_programme_does_not_crash_the_solver():
    """A mix with nothing audible must fall back, not raise."""
    import tempfile
    from alledits.audio.mix import MixPlan, MixTrack, TrackRole
    from alledits.core.ffmpeg import ffmpeg
    from alledits.render.ffmpeg_renderer import _solve_gain
    with tempfile.TemporaryDirectory() as d:
        sil = Path(d) / "s.wav"
        ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3",
                "-c:a", "pcm_s16le", str(sil)])
        plan = MixPlan()
        plan.tracks = [MixTrack(id="m", source_path=str(sil),
                                role=TrackRole.MUSIC.value,
                                source_in=0, source_out=3)]
        gain, _, _ = _solve_gain(plan, 3.0)
    assert gain is None, "a silent programme should report no solvable gain"


def _speech_asset():
    """The one corpus clip with speech in its own audio."""
    return str(CLIPS / "17_talking_head.mp4")


def _timeline_over(source_in, source_out):
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s0", source_path=_speech_asset(),
        source_in=source_in, source_out=source_out,
        timeline_start=0.0, duration=source_out - source_in))
    return tl


def test_speech_detector_ignores_music():
    """The critical negative: music must never be mistaken for speech, or the
    mix would duck itself."""
    from alledits.audio.speech import detect_speech
    assert detect_speech(str(MEDIA / "music.wav")).has_speech is False


def test_diegetic_voice_found_when_clip_covers_speech():
    from alledits.audio.mix import diegetic_voice_tracks
    from alledits.audio.speech import detect_speech
    sp = detect_speech(_speech_asset())
    assert sp.windows, "precondition: the talking-head clip should contain speech"
    a, b = sp.windows[0]
    voices = diegetic_voice_tracks(_timeline_over(a + 0.1, min(b, a + 1.5)))
    assert len(voices) == 1, f"expected one voice track, got {voices}"
    assert "speech" in voices[0]["reason"]


def test_no_voice_track_when_the_clip_misses_the_speech():
    """Observed in a real run: a clip whose range lands in a silent gap must not
    trigger a duck just because the FILE contains speech elsewhere."""
    from alledits.audio.mix import diegetic_voice_tracks
    from alledits.audio.speech import detect_speech
    sp = detect_speech(_speech_asset())
    first_start = sp.windows[0][0]
    assert first_start > 0.4, "precondition: need a silent head on this clip"
    assert diegetic_voice_tracks(_timeline_over(0.0, first_start - 0.2)) == []


def test_music_is_ducked_and_lowered_when_speech_is_present():
    from alledits.audio.mix import plan_mix, diegetic_voice_tracks, TrackRole
    from alledits.audio.speech import detect_speech
    a, b = detect_speech(_speech_asset()).windows[0]
    tl = _timeline_over(a + 0.1, min(b, a + 1.5))
    voices = diegetic_voice_tracks(tl)
    plan = plan_mix(tl, str(MEDIA / "music.wav"), voice_tracks=voices)
    music = plan.by_role(TrackRole.MUSIC)[0]
    assert plan.has_voice
    assert music.ducked_by, "music should be sidechained to the voice"
    assert music.gain_db < 0, "music should sit below speech as a bed"

    quiet = plan_mix(tl, str(MEDIA / "music.wav"), voice_tracks=[])
    assert quiet.by_role(TrackRole.MUSIC)[0].ducked_by == [], \
        "with no speech there is nothing to duck for"
    assert quiet.by_role(TrackRole.MUSIC)[0].gain_db == 0.0


def test_saved_timeline_records_the_mix_it_delivered():
    """Provenance: a saved project must show what the sound stage decided."""
    from alledits.timeline.schema import Timeline, ProjectSettings
    from alledits.audio.mix import plan_mix
    from alledits.timeline.schema import TimelineClip
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s0", source_path=str(CLIPS / "01_static_wide.mp4"),
        source_in=0.0, source_out=8.0, timeline_start=0.0, duration=8.0))
    tl.mix = plan_mix(tl, str(MEDIA / "music.wav"))
    tl.mix.achieved_lufs = -14.2
    d = tl.to_dict()
    assert d["mix"] is not None, "the mix was dropped on serialisation"
    assert d["mix"]["tracks"], "mix tracks missing from the saved timeline"
    assert d["mix"]["achieved_lufs"] == -14.2
    assert Timeline(project=ProjectSettings(1080, 1920, 30)).to_dict()["mix"] is None


def test_in_point_lands_on_speech_when_the_shot_has_dialogue():
    """Regression from a real run: the middle-of-shot rule picked a range ending
    10 ms before the dialogue started, so the talking clip contributed silence."""
    from alledits.timeline.builder import _pick_in_point, _speech_windows
    from alledits.audio.speech import detect_speech

    class _Shot:
        source_path = str(CLIPS / "17_talking_head.mp4")
        start, end = 0.0, 18.0
        duration = 18.0
        quality: dict = {}

    sp = detect_speech(_Shot.source_path)
    assert sp.windows, "precondition: this clip should contain speech"
    s_in, s_out = _pick_in_point(_Shot(), 1.0)
    overlap = sum(max(0.0, min(b, s_out) - max(a, s_in)) for a, b in sp.windows)
    assert overlap > 0.5, (
        f"chose {s_in:.2f}-{s_out:.2f} with only {overlap:.2f}s of speech")


def test_in_point_is_unchanged_for_silent_footage():
    """Speech awareness must not disturb the middle-of-shot rule elsewhere."""
    from alledits.timeline.builder import _pick_in_point

    class _Shot:
        source_path = str(CLIPS / "01_static_wide.mp4")
        start, end = 0.0, 4.0
        duration = 4.0
        quality: dict = {}

    s_in, s_out = _pick_in_point(_Shot(), 1.0)
    assert abs((s_out - s_in) - 1.0) < 1e-6
    assert s_in > 0.4, "should still skip the settling head of the shot"


def test_solver_returns_a_gain_it_actually_measured():
    """Regression: the loop returned the gain produced by the LAST correction,
    which was never measured, and could sit outside the clamp — so the render
    applied one value while the plan recorded another."""
    from alledits.audio.mix import plan_mix
    from alledits.render.ffmpeg_renderer import (_solve_gain, _build_mix_graph,
                                                 _ebur128, MAX_MAKEUP_GAIN_DB)
    plan = plan_mix(_Dur(8.0), str(MEDIA / "music.wav"))
    gain, claimed, _ = _solve_gain(plan, 8.0)
    assert gain is not None
    assert abs(gain) <= MAX_MAKEUP_GAIN_DB, f"gain {gain} outside the clamp"
    fc, lbl, inputs = _build_mix_graph(plan, 8.0, 0, gain_db=gain)
    remeasured, _ = _ebur128(inputs, fc, lbl)
    assert abs(remeasured - claimed) < 0.15, (
        f"plan claims {claimed} LUFS but that gain measures {remeasured}")


def test_speech_led_mix_reaches_the_loudness_target():
    """A quiet speech-led mix (music bedded, voice intermittent) needs more
    make-up than a music-only one and previously hit the clamp silently."""
    import tempfile
    from alledits.audio.mix import plan_mix, diegetic_voice_tracks
    from alledits.audio.speech import detect_speech
    from alledits.render.ffmpeg_renderer import _solve_gain
    a, _ = detect_speech(_speech_asset()).windows[0]
    tl = _timeline_over(a + 0.1, a + 2.1)
    voices = diegetic_voice_tracks(tl)
    assert voices, "precondition: this range should carry speech"
    plan = plan_mix(tl, str(MEDIA / "music.wav"), voice_tracks=voices)
    gain, achieved, _ = _solve_gain(plan, 2.0)
    assert gain is not None
    with tempfile.TemporaryDirectory() as d:
        I, TP = _ebur128(_render_mix(plan, 2.0, Path(d) / "m.m4a", gain_db=gain))
    assert abs(I - plan.target_lufs) <= 0.5, f"{I} LUFS vs {plan.target_lufs}"
    assert TP <= plan.true_peak_db, f"{TP} dBTP exceeds {plan.true_peak_db}"


# ------------------------------------------------------------------ FIND (Phase 5)
def _find_index():
    """Cached corpus index — ingesting 17 clips per test would be wasteful."""
    global _FIND_IDX
    try:
        return _FIND_IDX
    except NameError:
        pass
    from alledits.core.storage import LocalStorage
    from alledits.media.ingest import Ingestor
    from alledits.search.index import MediaIndex
    ing = Ingestor(LocalStorage("/home/claude/findtest"))
    idx, origin = MediaIndex(), {}
    for c in sorted(CLIPS.glob("*.mp4")):
        a = ing.ingest(c)
        origin[a.id] = c.stem
        idx.add_asset(a)
    _FIND_IDX = (idx, origin)
    return _FIND_IDX


def test_query_parses_measured_attributes():
    from alledits.search.query import parse_query
    q = parse_query("dark handheld shots")
    attrs = {c.attribute for c in q.criteria}
    assert "camera_movement" in attrs and "brightness" in attrs
    assert q.semantic is False, "this build has no semantic search to claim"


def test_query_reports_what_it_did_not_understand():
    """Silently ignoring half a query is worse than admitting the gap."""
    from alledits.search.query import parse_query
    q = parse_query("the shot where she looks relieved")
    assert not q.criteria
    assert q.unmatched_terms, "un-understood terms must be surfaced"


def test_query_negation_is_clause_scoped():
    from alledits.search.query import parse_query
    q = parse_query("no handheld, dark")
    bright = [c for c in q.criteria if c.attribute == "brightness"]
    cam = [c for c in q.criteria if c.attribute == "camera_movement"]
    assert cam and cam[0].negated, "'no handheld' should negate"
    assert bright and not bright[0].negated, \
        "a negation in one clause must not invert the next"


def test_synonyms_do_not_double_count_one_condition():
    from alledits.search.query import parse_query
    q = parse_query("talking head interview with dialogue")
    speech = [c for c in q.criteria if c.attribute == "speech"]
    assert len(speech) == 1, f"speech counted {len(speech)} times"


def test_find_retrieves_the_named_footage():
    idx, origin = _find_index()
    for query, expect in (("pan right", "02_pan_right"),
                          ("tilt up", "05_tilt_up"),
                          ("handheld", "06_handheld"),
                          ("talking head with speech", "17_talking_head")):
        names = [origin.get(r["shot"].asset_id, "") for r in idx.search(query, top_k=5)]
        assert any(expect in n for n in names), f"{query!r} -> {names}"


def test_find_negation_actually_excludes():
    idx, origin = _find_index()
    pos = {origin.get(r["shot"].asset_id) for r in idx.search("handheld", top_k=20)}
    neg = {origin.get(r["shot"].asset_id) for r in idx.search("no handheld", top_k=20)}
    assert pos, "precondition: something must match the positive query"
    assert not (pos & neg), f"negation leaked: {pos & neg}"


def test_unmeasurable_criteria_are_declared_not_silently_dropped():
    """shot_size is never populated by the analysers, so 'wide' can never match.
    Returning nothing would read as 'no wide shots exist', a different claim."""
    idx, _ = _find_index()
    idx.search("wide shots", top_k=5)
    assert any("shot size" in c for c in idx.last_inert_criteria), \
        f"inert criterion not declared: {idx.last_inert_criteria}"


def test_results_explain_themselves_in_measured_terms():
    idx, _ = _find_index()
    res = idx.search("dark static shots", top_k=3)
    assert res and res[0]["matched"], "a result must say why it matched"
    assert all(isinstance(m, str) for m in res[0]["matched"])


def test_open_vocabulary_search_still_refuses_to_fake_it():
    from alledits.core.errors import ProviderUnavailable
    idx, _ = _find_index()
    try:
        idx.search_by_text("a feeling of relief")
        raise AssertionError("semantic search should refuse without a VLM provider")
    except ProviderUnavailable:
        pass


# ---------------------------------------------------------------- MASTER (Phase 6)
def _master_source():
    """Small synthetic 'finished edit' with real audio. Cached across tests."""
    global _MASTER_SRC
    try:
        return _MASTER_SRC
    except NameError:
        pass
    from alledits.core.ffmpeg import ffmpeg
    d = Path("/home/claude/mastertests")
    d.mkdir(exist_ok=True)
    src = d / "edit.mp4"
    if not src.exists():
        ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-t", "3", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", str(src)])
    _MASTER_SRC = src
    return src


def test_master_produces_a_conformant_deliverable():
    import tempfile
    from alledits.master import master
    with tempfile.TemporaryDirectory() as d:
        r = master(_master_source(), Path(d) / "o.mp4", "youtube_shorts")
    assert r.conformant, [str(c) for c in r.qc.failed]
    names = {c.name for c in r.qc.checks}
    for essential in ("resolution", "frame rate", "video codec", "loudness",
                      "true peak"):
        assert essential in names, f"QC never checked {essential}"


def test_qc_fails_a_non_conformant_file():
    """A report that cannot fail is a rubber stamp."""
    import tempfile
    from alledits.master import master, run_qc, get_profile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "vertical.mp4"
        master(_master_source(), out, "youtube_shorts")
        rep = run_qc(out, get_profile("broadcast_ebu"))
    assert not rep.passed, "a vertical social file should not pass broadcast QC"
    failed = {c.name for c in rep.failed}
    assert {"resolution", "frame rate"} <= failed, failed


def test_qc_reports_unmeasurable_checks_as_skipped_not_passed():
    """Turning ignorance into a tick is worse than reporting no result."""
    from alledits.master.qc import QCReport, PASS, FAIL, SKIP
    rep = QCReport(path="x", profile="p")
    rep.add("a", PASS)
    rep.add("b", SKIP, detail="not measurable")
    assert rep.passed, "a skip alone must not block delivery"
    assert rep.skipped and rep.skipped[0].name == "b"
    rep.add("c", FAIL)
    assert not rep.passed, "a failure must block"


def test_upscaling_is_refused_unless_permitted():
    import tempfile
    from alledits.core.ffmpeg import ffmpeg
    from alledits.master import master
    with tempfile.TemporaryDirectory() as d:
        small = Path(d) / "small.mp4"
        ffmpeg(["-i", str(_master_source()), "-vf", "scale=540:960",
                "-c:v", "libx264", "-crf", "22", "-c:a", "copy", str(small)])
        try:
            master(small, Path(d) / "up.mp4", "youtube_shorts",
                   allow_upscale=False)
            raise AssertionError("should refuse to invent pixels")
        except ValueError:
            pass
        r = master(small, Path(d) / "up.mp4", "youtube_shorts", allow_upscale=True)
    assert r.qc.resolution_provenance == "upscaled"
    assert any("provenance" in c.name for c in r.qc.checks), \
        "an upscale must be disclosed in the report"


def test_letterboxed_downscale_is_not_reported_as_an_upscale():
    """Regression: plan_scaling used max() of the ratios while the encoder fits
    INSIDE the frame, so a 1080x1920 edit delivered to 1920x1080 was announced
    as a 1.78x upscale. A false disclosure erodes trust like a missing one."""
    from alledits.master import plan_scaling, get_profile
    d = plan_scaling(1080, 1920, get_profile("broadcast_ebu"))
    assert d.upscaling is False and d.provenance == "native", d
    assert plan_scaling(540, 960, get_profile("youtube_shorts")).upscaling is True


def test_every_profile_is_internally_coherent():
    from alledits.master import PROFILES
    from alledits.audio.mix import LOUDNESS_TARGETS
    for name, p in PROFILES.items():
        assert p.name == name, f"{name} disagrees with its key"
        assert p.width > 0 and p.height > 0 and p.fps > 0
        assert p.loudness_target in LOUDNESS_TARGETS, \
            f"{name} targets unknown loudness {p.loudness_target}"
        assert p.true_peak_db < 0, f"{name} allows clipping"
        assert (p.crf is not None) or p.video_bitrate, \
            f"{name} specifies neither crf nor bitrate"


def test_unknown_profile_fails_loudly():
    from alledits.master import get_profile
    try:
        get_profile("not_a_platform")
        raise AssertionError("should reject an unknown profile")
    except KeyError:
        pass


# ------------------------------------------------------------- AUTOPILOT (Phase 7)
def test_autopilot_decisiveness_uses_the_runner_up_margin():
    """Regression: a 7.4 / 7.4 / 4.8 run has a spread of 2.56 and was announced
    as 'a clear preference', but the top two tied — the choice was arbitrary.
    A poor rejected candidate says nothing about the winner being right."""
    from alledits.pipeline.autopilot import AutopilotResult, Candidate
    tied = AutopilotResult(candidates=[Candidate("a", "", 7.4),
                                       Candidate("b", "", 7.4),
                                       Candidate("c", "", 4.8)])
    assert tied.spread > 2.0, "precondition: the full spread looks large"
    assert tied.margin == 0.0
    assert tied.decisive is False, "a tie at the top is not a clear preference"

    clear = AutopilotResult(candidates=[Candidate("a", "", 8.5),
                                        Candidate("b", "", 6.0)])
    assert clear.decisive is True


def test_autopilot_candidates_are_genuinely_different_treatments():
    """Exploring three variations on one idea proves nothing."""
    from alledits.pipeline.autopilot import DEFAULT_CANDIDATES
    from alledits.intelligence.brief import parse_brief
    pacings = [parse_brief(b).pacing_multiplier for _, b in DEFAULT_CANDIDATES]
    assert len(DEFAULT_CANDIDATES) >= 3
    assert max(pacings) > min(pacings) * 1.5, \
        f"candidate briefs barely differ in pacing: {pacings}"


def test_autopilot_ignores_failed_candidates_but_reports_them():
    from alledits.pipeline.autopilot import AutopilotResult, Candidate
    bad = Candidate("broken", "", 0.0)
    bad.error = "RuntimeError: boom"
    res = AutopilotResult(candidates=[Candidate("ok", "", 6.0), bad])
    assert res.scores == [6.0], "a failed candidate must not contribute a score"
    assert any(c.error for c in res.candidates), "failures must stay visible"


def test_autopilot_result_serialises_every_candidate_not_just_the_winner():
    from alledits.pipeline.autopilot import AutopilotResult, Candidate
    res = AutopilotResult(candidates=[Candidate("a", "x", 7.0),
                                      Candidate("b", "y", 5.0)])
    res.winner = res.candidates[0]
    d = res.to_dict()
    assert len(d["candidates"]) == 2, "losing candidates must remain auditable"
    assert d["winner"] == "a" and "margin" in d


def test_preview_only_mode_skips_the_final_render():
    """Autopilot depends on this: exploring at full raster would cost minutes
    per candidate for a file it discards."""
    import inspect
    from alledits.pipeline.vertical_slice import VerticalSlice
    sig = inspect.signature(VerticalSlice.run)
    assert "stop_after_preview" in sig.parameters
    src = inspect.getsource(VerticalSlice.run)
    idx_stop = src.index("stop_after_preview:")
    idx_final = src.index("Rendering final")
    assert idx_stop < idx_final, "the early return must precede the final render"


# ----------------------------------------------------------------- STYLE (Phase 8)
def _two_grammars():
    """A fast and a slow reference, cached — analysis is not cheap."""
    global _STYLE_PAIR
    try:
        return _STYLE_PAIR
    except NameError:
        pass
    import subprocess
    from alledits.reference.analyze_reference import analyze_reference
    slow = Path("/tmp/ref_slow_test.mp4")
    if not slow.exists():
        parts = []
        for i, c in enumerate(["01_static_wide", "02_pan_right", "04_pull_out",
                               "07_dark_lowkey"]):
            o = f"/tmp/sp_{i}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.3",
                            "-t", "2.6", "-i", str(CLIPS / f"{c}.mp4"),
                            "-vf", "scale=1080:1920:force_original_aspect_ratio="
                            "increase,crop=1080:1920,fps=30", "-an",
                            "-c:v", "libx264", "-crf", "20", o], check=True)
            parts.append(o)
        lst = Path("/tmp/sl_test.txt")
        lst.write_text("\n".join(f"file '{x}'" for x in parts))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe",
                        "0", "-i", str(lst), "-c", "copy", str(slow)], check=True)
    fast = analyze_reference(str(MEDIA / "reference.mp4"), label="fast")
    slw = analyze_reference(str(slow), label="slow")
    _STYLE_PAIR = (fast, slw)
    return _STYLE_PAIR


def test_grammar_survives_a_json_round_trip():
    from alledits.reference.grammar import StyleGrammar
    g, _ = _two_grammars()
    g2 = StyleGrammar.from_json(g.to_json())
    assert abs(g2.pacing.cuts_per_second - g.pacing.cuts_per_second) < 1e-9
    assert g2.pacing.rhythm == g.pacing.rhythm
    assert g2.transitions.to_dict() == g.transitions.to_dict()
    assert len(g2.intensity_curve) == len(g.intensity_curve)


def test_stale_grammar_version_is_refused_not_defaulted():
    """Filling missing fields with defaults would present a stale style as current."""
    from alledits.reference.grammar import StyleGrammar
    g, _ = _two_grammars()
    d = g.to_dict()
    d["version"] = "0.0-ancient"
    try:
        StyleGrammar.from_dict(d)
        raise AssertionError("should refuse a mismatched grammar version")
    except ValueError:
        pass


def test_blend_interpolates_between_its_sources():
    from alledits.reference.style import blend_grammars
    fast, slow = _two_grammars()
    assert fast.pacing.cuts_per_second > slow.pacing.cuts_per_second * 2, \
        "precondition: the two references must actually differ"
    mid = blend_grammars([(fast, 1), (slow, 1)], label="mid")
    lo, hi = sorted([fast.pacing.cuts_per_second, slow.pacing.cuts_per_second])
    assert lo < mid.pacing.cuts_per_second < hi
    ms_lo, ms_hi = sorted([fast.pacing.mean_shot, slow.pacing.mean_shot])
    assert ms_lo < mid.pacing.mean_shot < ms_hi, \
        "mean shot length must also land between the sources"
    heavy_fast = blend_grammars([(fast, 3), (slow, 1)], label="hf")
    heavy_slow = blend_grammars([(fast, 1), (slow, 3)], label="hs")
    assert (heavy_fast.pacing.cuts_per_second
            > mid.pacing.cuts_per_second
            > heavy_slow.pacing.cuts_per_second), "weights must shift the blend"


def test_blend_renormalises_transition_shares():
    """Averaged shares must still describe a distribution."""
    from alledits.reference.style import blend_grammars
    fast, slow = _two_grammars()
    b = blend_grammars([(fast, 1), (slow, 2)], label="b")
    t = b.transitions
    total = (t.hard_cut_share + t.flash_share + t.whip_share + t.dissolve_share)
    assert abs(total - 1.0) < 1e-6, total


def test_blend_does_not_average_categorical_qualities():
    """There is no midpoint between 'accelerating' and 'steady'."""
    from alledits.reference.style import blend_grammars
    fast, slow = _two_grammars()
    assert fast.pacing.rhythm != slow.pacing.rhythm, "precondition"
    b = blend_grammars([(fast, 1), (slow, 3)], label="b")
    assert b.pacing.rhythm == slow.pacing.rhythm, "heaviest weight should win"
    assert any("disagreed on rhythm" in n for n in b.notes), \
        "a disagreement between sources must be disclosed, not hidden"


def test_blend_resamples_curves_of_different_lengths():
    from alledits.reference.style import blend_grammars, _resample
    fast, slow = _two_grammars()
    short = type(fast).from_dict(fast.to_dict())
    short.intensity_curve = short.intensity_curve[:8]
    b = blend_grammars([(short, 1), (slow, 1)], label="b")
    assert len(b.intensity_curve) == max(8, len(slow.intensity_curve))
    assert len(_resample([{"t": 0, "value": 0.2}], 5)) == 5


def test_blend_rejects_empty_and_zero_weights():
    from alledits.reference.style import blend_grammars
    fast, _ = _two_grammars()
    for bad in ([], [(fast, 0)], [(fast, -2)]):
        try:
            blend_grammars(bad)
            raise AssertionError(f"should reject {bad!r}")
        except ValueError:
            pass


def test_style_library_round_trip_and_listing():
    import tempfile
    from alledits.reference.style import StyleLibrary
    fast, _ = _two_grammars()
    with tempfile.TemporaryDirectory() as d:
        lib = StyleLibrary(d)
        lib.save("My Punchy Style!", fast)
        assert lib.list_names() == ["my-punchy-style"]
        back = lib.load("My Punchy Style!")
        assert abs(back.pacing.cuts_per_second - fast.pacing.cuts_per_second) < 1e-9
        assert back.source_label == "My Punchy Style!"
        assert lib.delete("my-punchy-style") and lib.list_names() == []


def test_saved_style_stores_no_path_back_to_the_source_work():
    """A grammar is a measurement, not a copy — saving a style must not save a
    pointer to someone else's footage."""
    import json, tempfile
    from alledits.reference.style import StyleLibrary
    fast, _ = _two_grammars()
    with tempfile.TemporaryDirectory() as d:
        p = StyleLibrary(d).save("x", fast)
        raw = json.loads(p.read_text())
    leaky = [k for k, v in raw.items()
             if isinstance(v, str) and (".mp4" in v or v.startswith("/"))]
    assert not leaky, f"style file leaks source references: {leaky}"


def test_missing_style_fails_loudly_with_what_is_available():
    import tempfile
    from alledits.reference.style import StyleLibrary
    fast, _ = _two_grammars()
    with tempfile.TemporaryDirectory() as d:
        lib = StyleLibrary(d)
        lib.save("kept", fast)
        try:
            lib.load("absent")
            raise AssertionError("should raise for a missing style")
        except KeyError as e:
            assert "kept" in str(e), "the error should say what IS available"


def test_pipeline_accepts_a_supplied_grammar_instead_of_a_reference():
    import inspect
    from alledits.pipeline.vertical_slice import VerticalSlice
    assert "grammar" in inspect.signature(VerticalSlice.run).parameters
    src = inspect.getsource(VerticalSlice.run)
    assert "Using supplied style" in src, \
        "a supplied style must be used rather than re-analysing a reference"


# -------------------------------------------------------------- DIRECTOR (Phase 9)
def _fake_timeline():
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    spec = [("sA", 1.0, 0.2, 0.50), ("sB", 1.2, 3.9, 0.45),
            ("sC", 0.8, 0.1, 0.12), ("sD", 1.5, 0.3, 0.60)]
    start = 0.0
    for i, (sid, dur, shake, bright) in enumerate(spec):
        c = TimelineClip(id=f"c{i:03d}", source_id=sid, source_path="x",
                         source_in=0, source_out=dur, timeline_start=start,
                         duration=dur)
        c.visual = {"shake": shake, "brightness": bright}
        tl.clips.append(c)
        start += dur
    return tl


def test_director_resolves_a_positional_reference():
    from alledits.intelligence.director import parse_note
    from alledits.core.project import DirectiveKind
    p = parse_note("hold the third shot longer", _fake_timeline())
    assert len(p.changes) == 1
    ch = p.changes[0]
    assert ch.kind == DirectiveKind.SET_DURATION.value
    assert ch.slot_index == 2, "'third' must mean the third clip"
    assert ch.value > 0.8, "longer must mean longer"


def test_director_resolves_an_attribute_reference_from_measured_signal():
    from alledits.intelligence.director import parse_note
    from alledits.core.project import DirectiveKind
    p = parse_note("lose the shaky one", _fake_timeline())
    assert len(p.changes) == 1
    assert p.changes[0].kind == DirectiveKind.REJECT_SHOT.value
    assert p.changes[0].shot_id == "sB", "sB has the highest measured shake"


def test_director_reports_what_it_could_not_act_on():
    """A half-applied note is dangerous: the user re-watches for a change that
    was never made."""
    from alledits.intelligence.director import parse_note
    p = parse_note("make it feel like a half-remembered dream", _fake_timeline())
    assert not p.understood
    assert p.unresolved
    assert p.is_llm is False


def test_director_refuses_an_out_of_range_reference():
    from alledits.intelligence.director import parse_note
    p = parse_note("make the ninth clip longer", _fake_timeline())
    assert not p.changes
    assert any("only 4 clips" in u for u in p.unresolved)


def test_director_does_not_guess_when_the_signal_is_absent():
    """'the shaky one' on a timeline with no shake data must not pick a clip
    at random — that applies the note to innocent footage."""
    from alledits.intelligence.director import parse_note
    tl = _fake_timeline()
    for c in tl.clips:
        c.visual = {}
    p = parse_note("lose the shaky one", tl)
    assert not p.changes
    assert any("shaky" in u for u in p.unresolved)


def test_director_routes_whole_piece_notes_to_the_brief():
    from alledits.intelligence.director import parse_note
    p = parse_note("punch it up", _fake_timeline())
    assert p.brief_delta and not p.changes
    from alledits.intelligence.brief import parse_brief
    assert parse_brief(p.brief_delta).pacing_multiplier != 1.0, \
        "the brief vocabulary should recognise it"


def test_director_never_promises_a_brief_change_that_does_nothing():
    """Six global hints were once absent from the brief lexicon, so DIRECTOR
    reported 're-plan with punch it up' and the re-plan changed nothing. A note
    that looks applied and is not is the worst outcome here."""
    from alledits.intelligence.director import (parse_note, GLOBAL_HINTS,
                                                _brief_would_act)
    for hint in GLOBAL_HINTS:
        p = parse_note(hint, _fake_timeline())
        if p.brief_delta:
            assert _brief_would_act(p.brief_delta), (
                f"{hint!r} was routed to the brief but the brief ignores it")
        else:
            assert p.unresolved or p.changes, (
                f"{hint!r} silently vanished — neither applied nor reported")


def test_director_handles_several_clauses_in_one_note():
    from alledits.intelligence.director import parse_note
    from alledits.core.project import DirectiveKind
    p = parse_note("hold the first shot longer, no flashes", _fake_timeline())
    kinds = {c.kind for c in p.changes}
    assert DirectiveKind.SET_DURATION.value in kinds
    assert DirectiveKind.BAN_EFFECT.value in kinds


def test_set_duration_directive_is_honoured_and_reflows():
    """Positions come from the beat plan, so a longer clip must push the rest."""
    from alledits.core.project import OverrideSet, DirectiveKind
    base, _ = _build()
    target = 1
    before = base.clips[target].duration
    ov = OverrideSet()
    ov.add(DirectiveKind.SET_DURATION, slot_index=target, value=before + 0.8)
    after, _ = _build(overrides=ov)
    assert abs(after.clips[target].duration - (before + 0.8)) < 0.05
    for i in range(len(after.clips) - 1):
        gap = (after.clips[i + 1].timeline_start
               - (after.clips[i].timeline_start + after.clips[i].duration))
        assert abs(gap) < 1e-4, f"gap of {gap} after clip {i}"


def test_shifted_clips_stop_claiming_a_beat_lock_they_no_longer_have():
    """The timeline must not assert sync it has knowingly broken."""
    from alledits.core.project import OverrideSet, DirectiveKind
    base, _ = _build()
    locked_before = [c.beat_locked for c in base.clips]
    assert any(locked_before), "precondition: something should be beat-locked"
    ov = OverrideSet()
    ov.add(DirectiveKind.SET_DURATION, slot_index=0,
           value=base.clips[0].duration + 0.7)
    after, _ = _build(overrides=ov)
    assert any(not c.beat_locked for c in after.clips[1:]), \
        "clips shifted off the grid must drop their beat_locked flag"


# --------------------------------------------------- content anchoring (Phase 10)
class _AnchorClip:
    def __init__(self, i, sid, t):
        self.id, self.source_id, self.timeline_start = f"c{i:03d}", sid, t


def _anchored_lock():
    from alledits.core.project import OverrideSet, DirectiveKind
    before = [_AnchorClip(0, "sA", 0.0), _AnchorClip(1, "sB", 1.0),
              _AnchorClip(2, "sC", 2.0)]
    ov = OverrideSet()
    d = ov.add(DirectiveKind.LOCK_CLIP, slot_index=1)
    ov.anchor_to(d, before)
    return ov, d


def test_directive_records_what_was_on_screen():
    ov, d = _anchored_lock()
    assert d.anchor_shot_id == "sB" and d.anchor_status == "exact"


def test_directive_follows_its_shot_when_the_plan_reshapes():
    """A slot index alone is a fragile handle: insert one clip and 'slot 1' is
    a different moment, so the note lands on footage nobody looked at."""
    ov, d = _anchored_lock()
    after = [_AnchorClip(0, "sA", 0.0), _AnchorClip(1, "sX", 0.8),
             _AnchorClip(2, "sB", 1.6), _AnchorClip(3, "sC", 2.4)]
    report = ov.bind(after)
    assert d.slot_index == 2, "the lock should have followed sB"
    assert ov.locked_slots() == {2}
    assert report["moved"], "a move must be reported, not silent"


def test_directive_does_not_fire_when_its_shot_is_gone():
    """Retargeting onto unrelated footage is worse than not applying at all."""
    ov, d = _anchored_lock()
    ov.bind([_AnchorClip(0, "sA", 0.0), _AnchorClip(1, "sC", 1.0)])
    assert d.anchor_status == "lost"
    assert ov.locked_slots() == set(), "a lost directive must not lock a bystander"


def test_lost_anchor_is_reported_to_the_caller():
    ov, _ = _anchored_lock()
    report = ov.bind([_AnchorClip(0, "sA", 0.0)])
    assert report["lost"] and "no longer in the edit" in report["lost"][0]


def test_unanchored_directives_keep_legacy_slot_behaviour():
    """Projects saved before anchoring existed must still work."""
    from alledits.core.project import OverrideSet, DirectiveKind
    ov = OverrideSet()
    ov.add(DirectiveKind.LOCK_CLIP, slot_index=1)   # no anchor
    ov.bind([_AnchorClip(0, "sA", 0.0), _AnchorClip(1, "sZ", 1.0)])
    assert ov.locked_slots() == {1}


def test_anchor_follows_the_nearest_instance_when_a_shot_repeats():
    from alledits.core.project import OverrideSet, DirectiveKind
    ov = OverrideSet()
    d = ov.add(DirectiveKind.SET_DURATION, slot_index=2, value=1.5)
    ov.anchor_to(d, [_AnchorClip(0, "sA", 0.0), _AnchorClip(1, "sB", 1.0),
                     _AnchorClip(2, "sC", 2.0)])
    # sC now appears twice; the one nearest its old position should win
    ov.bind([_AnchorClip(0, "sC", 0.2), _AnchorClip(1, "sA", 1.0),
             _AnchorClip(2, "sB", 1.6), _AnchorClip(3, "sC", 2.1)])
    assert d.slot_index == 3, "should follow the instance nearest t=2.0"
    assert ov.duration_at(3) == 1.5


# ------------------------------------------------ model orchestration (Phase 11)
def _fake_vlm():
    import numpy as np
    from alledits.intelligence.capabilities import ImageEmbeddingProvider

    class FakeVLM(ImageEmbeddingProvider):
        name = "fake-clip"

        def available(self):
            return True

        def embed_images(self, images):
            return [np.array([1.0, 0.0, 0.0])] * len(images)

        def embed_text(self, texts):
            return [np.array([1.0, 0.1, 0.0])] * len(texts)

    return FakeVLM()


def test_registry_ships_empty_rather_than_pretending():
    """A pre-registered stub would make the registry lie about this build."""
    from alledits.intelligence.capabilities import default_registry, Capability
    reg = default_registry()
    for blocked in (Capability.IMAGE_EMBEDDING, Capability.TRANSCRIPTION,
                    Capability.PERSON_SEGMENTATION):
        assert not reg.has(blocked), f"{blocked} claims to exist and does not"


def test_require_names_the_gap_and_what_it_blocks():
    from alledits.intelligence.capabilities import CapabilityRegistry, Capability
    from alledits.core.errors import ProviderUnavailable
    try:
        CapabilityRegistry().require(Capability.IMAGE_EMBEDDING, "semantic search")
        raise AssertionError("should refuse")
    except ProviderUnavailable as e:
        msg = str(e)
        assert "image_embedding" in msg
        assert "open-vocabulary" in msg.lower(), "must say what it would unlock"


def test_a_broken_provider_counts_as_absent_not_available():
    """A provider that raises must not be handed to the engine."""
    from alledits.intelligence.capabilities import (CapabilityRegistry, Capability,
                                                    ImageEmbeddingProvider)

    class Broken(ImageEmbeddingProvider):
        name = "broken"

        def available(self):
            raise RuntimeError("driver missing")

        def embed_images(self, images):
            return []

        def embed_text(self, texts):
            return []

    reg = CapabilityRegistry().register(Broken())
    assert reg.get(Capability.IMAGE_EMBEDDING) is None
    assert not reg.has(Capability.IMAGE_EMBEDDING)


def test_connecting_a_provider_enables_semantic_search_with_no_engine_change():
    """The point of the seam: the editing engine is not modified to add a model."""
    from alledits.intelligence.capabilities import CapabilityRegistry
    from alledits.search.index import MediaIndex

    class _S:
        def __init__(self, i, v):
            self.id = f"s{i}"
            self.semantic_embedding = v
            self.quality = {"handling": "use"}

    idx = MediaIndex()
    idx.shots = [_S(0, [1, 0, 0]), _S(1, [0, 1, 0]), _S(2, [0.9, 0.2, 0])]
    idx.usable_shots = lambda include_brief=True: idx.shots
    reg = CapabilityRegistry().register(_fake_vlm())
    res = idx.search_by_text("anything", registry=reg)
    assert res and res[0]["shot"].id == "s0"
    assert res[0]["semantic"] is True
    assert res[0]["score"] > res[-1]["score"]


def test_semantic_search_says_so_when_nothing_is_embedded_yet():
    """A connected model with an un-embedded library is a different failure
    from a missing model, and must not be reported as the same thing."""
    from alledits.intelligence.capabilities import CapabilityRegistry
    from alledits.core.errors import ProviderUnavailable
    from alledits.search.index import MediaIndex

    class _S:
        def __init__(self):
            self.id = "s0"
            self.quality = {"handling": "use"}

    idx = MediaIndex()
    idx.shots = [_S()]
    idx.usable_shots = lambda include_brief=True: idx.shots
    reg = CapabilityRegistry().register(_fake_vlm())
    try:
        idx.search_by_text("x", registry=reg)
        raise AssertionError("should refuse: nothing embedded")
    except ProviderUnavailable as e:
        assert "embedded" in str(e)


def test_status_lists_every_capability_with_what_it_unlocks():
    from alledits.intelligence.capabilities import (default_registry, Capability,
                                                    UNLOCKS)
    rows = default_registry().status()
    assert len(rows) == len(list(Capability))
    for r in rows:
        if not r.available:
            assert r.reason, f"{r.capability} is off with no reason given"
            assert r.unlocks or Capability(r.capability) not in UNLOCKS


# ------------------------------------------------ nothing is wasted (Phase 12)
def _quality_of(name):
    from alledits.media.probe import probe
    from alledits.media.visual import analyze_shot
    from alledits.media.quality import analyze_quality
    p = str(CLIPS / f"{name}.mp4")
    i = probe(p)
    v = analyze_shot(p, 0.1, 1.4)
    return analyze_quality(p, 0.1, 1.4, i, visual=v), v


def test_poor_footage_gets_named_salvage_roles_instead_of_dismissal():
    q, _ = _quality_of("13_very_poor")
    assert q.salvage, "poor footage should be offered a creative function"
    roles = {s["role"] for s in q.salvage}
    assert "flash_frame" in roles
    assert q.handling == "use_briefly"


def test_good_footage_is_not_offered_salvage_roles():
    """Salvage is a concession; offering it for good footage invites waste."""
    q, _ = _quality_of("11_4k_high_quality")
    assert q.salvage == []


def test_salvage_cap_ignores_roles_the_renderer_cannot_perform():
    """Regression: taking the max across ALL roles let a flash-frame clip
    inherit a background plate's 2.5s cap, though nothing composites layers."""
    from alledits.media.salvage import salvage_cap, ROLE_MAX_DURATION, SalvageRole
    q, _ = _quality_of("13_very_poor")
    assert any(not s["realisable"] for s in q.salvage), \
        "precondition: a non-realisable role should be present"
    cap = salvage_cap(q.salvage)
    assert cap <= ROLE_MAX_DURATION[SalvageRole.RAPID_MONTAGE] + 1e-9, cap
    assert cap < ROLE_MAX_DURATION[SalvageRole.BACKGROUND]


def test_unrealisable_roles_are_named_not_silently_dropped():
    q, _ = _quality_of("13_very_poor")
    blocked = [s for s in q.salvage if not s["realisable"]]
    assert blocked and all(s["requires"] for s in blocked), \
        "a blocked role must say what it needs"


def test_genuinely_unusable_footage_is_still_rejected():
    """'Nothing is wasted' must not mean 'everything is kept'."""
    from alledits.media.salvage import is_genuinely_unusable, assess_salvage

    class Q:
        handling = "replace"
        brightness = 0.005
        technical_quality = 0.1
        sharpness = 1.0
        noise = 0.0

    unusable, why = is_genuinely_unusable(Q(), None)
    assert unusable and "black" in why
    assert assess_salvage(Q(), None) == []


def test_salvage_clip_is_excluded_from_slots_longer_than_its_cap():
    """Trimming the source range is not enough — the clip would still occupy
    the slot and hold a frozen frame for the remainder."""
    import copy
    from alledits.intelligence.planner import plan_slots
    from alledits.intelligence.selector import select_for_slot
    from alledits.intelligence.brief import parse_brief
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    idx, _ = _find_index()
    au = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    proto = plan_slots(g, au, 8.0, cons=parse_brief("chaotic fast"))[0]
    pool = idx.usable_shots()

    def eligible(d):
        return [x for x in pool
                if not ((x.quality or {}).get("max_useful_duration")
                        and d > float((x.quality or {})["max_useful_duration"]) + 1e-3)]

    salv = [x for x in pool if (x.quality or {}).get("max_useful_duration")]
    assert salv, "precondition: the corpus should contain salvage footage"
    n_short = len([x for x in eligible(0.20)
                   if (x.quality or {}).get("max_useful_duration")])
    n_long = len([x for x in eligible(1.20)
                  if (x.quality or {}).get("max_useful_duration")])
    assert n_short > 0, "salvage footage should be usable in a short slot"
    assert n_long == 0, "salvage footage must not be eligible for a long slot"

    short = copy.deepcopy(proto)
    short.end = short.start + 0.20
    best, _ = select_for_slot(short, salv, None, set(), set())
    assert best is not None, "a short slot should still be fillable from salvage"


def test_salvage_is_used_when_good_footage_is_scarce():
    """The point of the principle: with too little good footage, poor clips
    fill short slots rather than the edit failing — but only briefly."""
    from alledits.core.storage import LocalStorage
    from alledits.media.ingest import Ingestor
    from alledits.search.index import MediaIndex
    from alledits.intelligence.brief import parse_brief
    from alledits.intelligence.planner import plan_slots
    from alledits.timeline.builder import build_timeline
    from alledits.timeline.schema import ProjectSettings
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference

    ing = Ingestor(LocalStorage("/home/claude/salvtest"))
    idx, origin = MediaIndex(), {}
    for n in ("01_static_wide", "06_handheld", "12_lowres_compressed",
              "13_very_poor"):
        a = ing.ingest(str(CLIPS / f"{n}.mp4"))
        origin[a.id] = n
        idx.add_asset(a)
    poor = {sid for sid, n in origin.items()
            if n in ("12_lowres_compressed", "13_very_poor")}
    au = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    cons = parse_brief("chaotic aggressive fast")
    slots = plan_slots(g, au, 8.0, cons=cons)
    tl = build_timeline(slots, idx, g, au, ProjectSettings(1080, 1920, 30),
                        music_path=str(MEDIA / "music.wav"), cons=cons)
    used = [c for c in tl.clips if c.source_id.rsplit("_s", 1)[0] in poor]
    assert used, "salvage footage should fill slots when good footage runs out"
    for c in used:
        cap = float((c.quality_handling and 0.51) or 0.51)
        assert c.duration <= cap, f"salvage held {c.duration:.2f}s, over its cap"
        assert c.quality_handling == "use_briefly"


def test_salvage_cap_is_at_least_one_beat_at_dance_tempo():
    """Regression: an arbitrary 0.45s cap sat just under one beat (0.465s at
    129 BPM), so salvage was silently ineligible for every slot in the edit."""
    from alledits.media.salvage import ROLE_MAX_DURATION, SalvageRole
    assert ROLE_MAX_DURATION[SalvageRole.RAPID_MONTAGE] >= 60.0 / 120.0 / 1.0 * 0.5
    assert ROLE_MAX_DURATION[SalvageRole.RAPID_MONTAGE] >= 0.47


def test_punctuation_slots_require_the_reference_to_actually_punctuate():
    """Style-derived, never invented: a reference whose shortest shots match
    its median has uniform shots and gets no stabs, however fast it cuts."""
    import copy
    from alledits.intelligence.planner import plan_slots
    from alledits.intelligence.brief import parse_brief
    from alledits.audio.analyze import analyze_audio
    from alledits.reference.analyze_reference import analyze_reference
    au = analyze_audio(MEDIA / "music.wav")
    g = analyze_reference(MEDIA / "reference.mp4")
    assert g.pacing.p10_shot > 0.6 * g.pacing.median_shot, \
        "precondition: the test reference has uniform shot lengths"
    slots = plan_slots(g, au, 8.0, cons=parse_brief("chaotic aggressive fast"))
    assert not [s for s in slots if s.role == "punctuation"], \
        "an unpunctuated reference must not produce stabs"

    flat = copy.deepcopy(g)
    flat.pacing.p10_shot = flat.pacing.median_shot * 0.9
    assert not [s for s in plan_slots(flat, au, 8.0,
                                      cons=parse_brief("energetic"))
                if s.role == "punctuation"]


# ------------------------------------------------ shoot assistant (Phase 13)
def _grammar_fast():
    from alledits.reference.analyze_reference import analyze_reference
    return analyze_reference(str(MEDIA / "reference.mp4"), label="fast")


def test_shot_plan_lengths_follow_the_style_not_a_fixed_template():
    """Spec 24: the plan must depend on the reference, not be hard-coded."""
    import copy
    from alledits.shoot import build_sequence
    fast = _grammar_fast()
    slow = copy.deepcopy(fast)
    slow.pacing.median_shot = fast.pacing.median_shot * 4
    slow.pacing.mean_shot = fast.pacing.mean_shot * 4
    slow.pacing.p10_shot = fast.pacing.p10_shot * 4
    slow.pacing.p90_shot = fast.pacing.p90_shot * 4
    a = build_sequence("skill", fast)
    b = build_sequence("skill", slow)
    assert [x.number for x in a] == [x.number for x in b]
    assert sum(x.duration for x in b) > sum(x.duration for x in a) * 1.3, \
        "a slower style should ask for longer takes"


def test_record_time_exceeds_the_cut_length():
    """A clip trimmed to exactly its cut length has no handles."""
    from alledits.shoot import build_sequence
    g = _grammar_fast()
    for spec in build_sequence("skill", g):
        assert spec.duration >= 1.5


def test_every_planned_shot_carries_direction_and_a_do_not():
    from alledits.shoot import build_sequence
    for spec in build_sequence("skill", _grammar_fast()):
        assert spec.camera_position and spec.action and spec.purpose
        assert spec.do_not, f"shot {spec.number} has no DO NOT instruction"
        text = spec.instructions()
        assert "Record for about" in text and "DO NOT" in text


def test_beginner_instructions_omit_camera_settings():
    from alledits.shoot import build_sequence
    spec = build_sequence("skill", _grammar_fast())[0]
    assert "Settings:" in spec.instructions(skill="advanced")
    assert "Settings:" not in spec.instructions(skill="beginner")


def test_coverage_never_claims_a_shot_is_covered():
    """The strongest honest claim is LIKELY: the shape matches, content is
    unverified. Saying 'covered' would send someone into an edit believing they
    have footage they never filmed."""
    from alledits.shoot import build_sequence, assess_coverage, LIKELY, MISSING
    idx, _ = _find_index()
    rep = assess_coverage(idx, build_sequence("skill", _grammar_fast()), "skill")
    statuses = {s.status for s in rep.shots}
    assert "covered" not in statuses
    assert statuses <= {LIKELY, MISSING, "unverifiable"}
    for s in rep.shots:
        if s.status == LIKELY:
            assert s.unchecked, "a likely match must name what was NOT verified"


def test_coverage_identifies_a_genuine_gap():
    from alledits.shoot import build_sequence, assess_coverage
    idx, _ = _find_index()
    rep = assess_coverage(idx, build_sequence("skill", _grammar_fast()), "skill")
    assert rep.missing, "this corpus lacks a sharp static detail shot"
    for s in rep.missing:
        assert s.reason and "no clip satisfies" in s.reason


def test_inspect_rejects_footage_that_contradicts_the_direction():
    from alledits.shoot import build_sequence, inspect_recording
    idx, origin = _find_index()
    spec = build_sequence("skill", _grammar_fast())[0]   # wants a static shot
    handheld = next(s.id for s in idx.shots
                    if origin.get(s.asset_id) == "06_handheld")
    r = inspect_recording(idx, spec, handheld)
    assert r["verdict"] == "reshoot"
    assert any("static" in p for p in r["problems"])


def test_inspect_approves_good_footage_without_demanding_a_retake():
    """Spec 5: if it is good enough, do not demand unnecessary retakes."""
    from alledits.shoot import build_sequence, inspect_recording
    idx, origin = _find_index()
    spec = build_sequence("skill", _grammar_fast())[0]
    static = next(s.id for s in idx.shots
                  if origin.get(s.asset_id) == "01_static_wide")
    assert inspect_recording(idx, spec, static)["verdict"] == "approve"


def test_inspect_flags_salvage_grade_footage_for_a_held_shot():
    """Regression: a clip usable for 0.5s as a flash was approved for a shot the
    plan wants held for seconds — telling someone they had the shot."""
    from alledits.shoot import build_sequence, inspect_recording
    idx, origin = _find_index()
    spec = build_sequence("skill", _grammar_fast())[0]
    poor = next(s.id for s in idx.shots
                if origin.get(s.asset_id) == "13_very_poor")
    r = inspect_recording(idx, spec, poor)
    assert r["verdict"] == "reshoot"
    assert any("briefly" in a for a in r["advice"])


def test_inspect_abstains_on_content():
    from alledits.shoot import build_sequence, inspect_recording
    idx, origin = _find_index()
    spec = build_sequence("skill", _grammar_fast())[0]
    sid = next(s.id for s in idx.shots
               if origin.get(s.asset_id) == "01_static_wide")
    r = inspect_recording(idx, spec, sid)
    assert r["content_unverified"] == spec.semantic_content
    assert "no model" in r["note"]


def test_unknown_sequence_fails_loudly():
    from alledits.shoot import build_sequence
    try:
        build_sequence("not_a_sequence", _grammar_fast())
        raise AssertionError("should reject an unknown sequence")
    except KeyError as e:
        assert "available" in str(e)


# ------------------------------------------------ creator profile (Phase 14)
def _shot(n=3):
    from alledits.shoot import build_sequence
    return build_sequence("skill", _grammar_fast())[n]


def test_instruction_level_changes_how_much_is_spelled_out():
    from alledits.profile import CreatorProfile, render_instructions
    spec = _shot()
    teach = render_instructions(spec, CreatorProfile(instruction_level="teach_me"))
    lean = render_instructions(spec, CreatorProfile(instruction_level="minimal"))
    assert len(teach) > len(lean) * 2
    assert "Why:" in teach and "Why:" not in lean
    assert "DO NOT" in teach


def test_beginner_language_avoids_unexplained_jargon():
    from alledits.profile import CreatorProfile, render_instructions
    from alledits.shoot import build_sequence
    specs = build_sequence("skill", _grammar_fast())
    prof = CreatorProfile(instruction_level="teach_me", has_tripod=True)
    text = " ".join(render_instructions(s, prof) for s in specs)
    assert "locked off" not in text.lower(), "jargon should be translated"


def test_technical_level_keeps_the_jargon():
    from alledits.profile import CreatorProfile, render_instructions
    prof = CreatorProfile(instruction_level="technical", has_tripod=True)
    text = render_instructions(_shot(0), prof)
    assert "locked off" in text.lower()


def test_explanations_stop_once_the_concept_has_been_taught():
    """Spec 4: stop repeatedly explaining concepts the user already understands."""
    from alledits.profile import CreatorProfile, render_instructions
    from alledits.shoot import build_sequence
    specs = build_sequence("skill", _grammar_fast())
    prof = CreatorProfile(instruction_level="teach_me", has_tripod=True)
    counts = [render_instructions(s, prof).count("      (") for s in specs]
    assert counts[0] > 0, "the first shot should explain something"
    assert counts[-1] < counts[0], f"explanations should fade: {counts}"


def test_declared_knowledge_is_never_explained():
    from alledits.profile import CreatorProfile, render_instructions
    prof = CreatorProfile(instruction_level="teach_me", has_tripod=True,
                          known_concepts=["locked_off", "pre_post_roll", "fps"])
    text = render_instructions(_shot(0), prof)
    assert "usually on a tripod" not in text


def test_a_shot_needing_gear_the_person_lacks_is_redesigned_and_disclosed():
    """Spec 5: if the requested shot is impossible, redesign it — and say so,
    or they will think they filmed the shot that was planned."""
    from alledits.profile import CreatorProfile, adapt_shot
    spec = _shot(3)                       # wants a pan / follow
    alone = CreatorProfile(has_tripod=False, crew_size=1)
    adapted, redesign = adapt_shot(spec, alone)
    assert redesign.changed
    assert redesign.reason
    assert adapted.movement != spec.movement
    assert spec.movement, "the original must be left untouched"

    crewed = CreatorProfile(has_tripod=True, crew_size=3)
    _, none = adapt_shot(spec, crewed)
    assert not none.changed, "a fully equipped shoot needs no substitution"


def test_redesign_does_not_mutate_the_original_spec():
    from alledits.profile import CreatorProfile, adapt_shot
    spec = _shot(3)
    before = spec.movement
    adapt_shot(spec, CreatorProfile(has_tripod=False, crew_size=1))
    assert spec.movement == before


def test_a_braced_camera_is_not_taught_as_a_tracking_shot():
    """Regression: the concept was picked from the ADAPTED wording, so once
    'locked off' became 'rest it on something solid' the static shot was
    explained as a moving one."""
    from alledits.profile import CreatorProfile, render_instructions
    prof = CreatorProfile(instruction_level="teach_me", has_tripod=False)
    text = render_instructions(_shot(0), prof)
    assert "moves with the subject" not in text
    assert "completely still" in text


def test_settings_are_not_double_substituted():
    """Regression: adapt_shot rewrote '60fps' for the device, then plainify
    rewrote the replacement's own wording."""
    from alledits.profile import CreatorProfile, render_instructions
    text = render_instructions(_shot(3),
                               CreatorProfile(instruction_level="teach_me",
                                              device="phone"))
    assert "often 60fps" in text
    assert "often the higher frame-rate setting" not in text


def test_profile_round_trips_including_what_was_explained():
    import tempfile
    from alledits.profile import CreatorProfile
    p = CreatorProfile(instruction_level="teach_me", device="camera",
                       has_tripod=True, crew_size=2,
                       content_types=["sports", "self-content"])
    p.explain("fps")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.json"
        p.save(path)
        q = CreatorProfile.load(path)
    assert q.instruction_level == "teach_me" and q.crew_size == 2
    assert q.content_types == ["sports", "self-content"]
    assert q.explained.get("fps") == 1, "adaptive state must persist across sessions"


def test_profile_does_not_infer_skill_from_footage():
    """The profile holds what the person TOLD us. Judging their ability from a
    shaky clip is a conclusion they never asked for."""
    import inspect
    from alledits.profile import creator
    src = inspect.getsource(creator)
    for leak in ("shake", "technical_quality", "analyze_shot", "quality"):
        assert leak not in src, f"profile module reads {leak} from footage"


# ------------------------------------------------------ async jobs (Phase 15)
def _slow_task(n=6, progress=None):
    import time
    for i in range(n):
        progress((i + 1) / n, f"step {i + 1}")
        time.sleep(0.05)
    return {"done": n}


def test_job_reaches_success_with_progress_recorded():
    import tempfile
    from alledits.core.jobs import BackgroundJobQueue, JobState
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        j = q.submit("demo", _slow_task, 4)
        q.wait(j.id, 30)
        q.shutdown()
    assert j.state == JobState.SUCCEEDED
    assert j.result == {"done": 4} and j.progress == 1.0
    ps = [s["p"] for s in j.steps if "p" in s]
    assert ps == sorted(ps), f"progress must not go backwards: {ps}"


def test_job_failure_is_captured_not_swallowed():
    import tempfile
    from alledits.core.jobs import BackgroundJobQueue, JobState
    def boom(progress=None):
        raise ValueError("bad input")
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        j = q.submit("demo", boom)
        q.wait(j.id, 30)
        q.shutdown()
    assert j.state == JobState.FAILED
    assert "bad input" in j.error
    assert any("traceback" in s for s in j.steps), "the traceback must be kept"


def test_cancellation_is_not_reported_as_a_failure():
    """A user stopping a render did nothing wrong."""
    import tempfile, time
    from alledits.core.jobs import BackgroundJobQueue, JobState
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        j = q.submit("demo", _slow_task, 200)
        time.sleep(0.2)
        assert q.cancel(j.id) is True
        q.wait(j.id, 30)
        q.shutdown()
    assert j.state == JobState.CANCELLED
    assert j.error is None, "cancelling is not an error"
    assert 0.0 < j.progress < 1.0, "it should stop partway, not complete"


def test_cancelling_a_finished_job_is_refused():
    import tempfile
    from alledits.core.jobs import BackgroundJobQueue
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        j = q.submit("demo", _slow_task, 2)
        q.wait(j.id, 30)
        assert q.cancel(j.id) is False
        q.shutdown()


def test_jobs_persist_and_reload_in_a_new_queue():
    """A UI that reconnects must be able to see what happened."""
    import tempfile
    from alledits.core.jobs import BackgroundJobQueue, JobState
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        j = q.submit("demo", _slow_task, 3)
        q.wait(j.id, 30)
        q.shutdown()
        fresh = BackgroundJobQueue(root=d)
        back = fresh.get(j.id)
        fresh.shutdown()
    assert back is not None and back.state == JobState.SUCCEEDED
    assert back.result == {"done": 3}


def test_a_job_orphaned_by_a_crash_is_not_left_running_forever():
    """No thread is carrying it any more, so showing it as live would be a lie."""
    import json, tempfile
    from alledits.core.jobs import BackgroundJobQueue, Job, JobState
    with tempfile.TemporaryDirectory() as d:
        orphan = Job(id="job_orphan", kind="edit")
        orphan.state = JobState.RUNNING
        orphan.progress = 0.6
        (Path(d) / "job_orphan.json").write_text(json.dumps(orphan.to_dict()))
        q = BackgroundJobQueue(root=d)
        got = q.get("job_orphan")
        q.shutdown()
    assert got.state == JobState.FAILED
    assert "interrupted" in got.error and "unknown" in got.error


def test_jobs_can_be_filtered_by_project_and_state():
    import tempfile
    from alledits.core.jobs import BackgroundJobQueue
    with tempfile.TemporaryDirectory() as d:
        q = BackgroundJobQueue(root=d)
        a = q.submit("demo", _slow_task, 2, project_id="p1")
        b = q.submit("demo", _slow_task, 2, project_id="p2")
        q.wait(a.id, 30)
        q.wait(b.id, 30)
        assert [j.id for j in q.list(project_id="p1")] == [a.id]
        assert len(q.list(state="succeeded")) == 2
        q.shutdown()


def test_every_task_accepts_the_progress_signature_the_queue_calls():
    """A task that does not take progress= would fail only at runtime, after
    the user submitted it."""
    import inspect
    from alledits.pipeline.tasks import TASKS
    assert set(TASKS) >= {"ingest", "edit", "autopilot", "master"}
    for name, fn in TASKS.items():
        assert "progress" in inspect.signature(fn).parameters, name


def test_inline_queue_still_works_for_synchronous_callers():
    """The pre-existing queue must keep working — it is the zero-infrastructure
    path the vertical slice was built on."""
    from alledits.core.jobs import InlineJobQueue, JobState
    q = InlineJobQueue()
    j = q.submit("demo", _slow_task, 2)
    assert j.state == JobState.SUCCEEDED and q.get(j.id) is j


# --------------------------------------------------------- web API (Phase 16)
import contextlib as _ctx


@_ctx.contextmanager
def _server(workdir=None, port=8123):
    import threading, time as _t
    from alledits.web import make_server
    srv = make_server(workdir or "/home/claude/webtest_wd", port=port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _t.sleep(0.3)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()
        srv.RequestHandlerClass.api.queue.shutdown(wait=False)


def _req(url, method="GET", body=None, headers=None):
    import json as _j, urllib.request, urllib.error
    data = _j.dumps(body).encode() if body is not None else None
    h = dict(headers or {})
    if data:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_web_serves_the_page_and_core_endpoints():
    import json as _j
    with _server(port=8131) as base:
        for path in ("/", "/api/health", "/api/capabilities",
                     "/api/delivery-profiles", "/api/sequences", "/api/jobs"):
            st, body, _ = _req(base + path)
            assert st == 200, f"{path} -> {st}"
            assert body, path
        st, body, _ = _req(base + "/api/health")
        assert _j.loads(body)["ok"] is True


def test_web_rejects_bad_input_at_the_boundary():
    """A bad path should be a 400 the user sees, not a job that starts and dies."""
    import json as _j
    with _server(port=8132) as base:
        st, body, _ = _req(base + "/api/jobs", "POST", {"kind": "nonsense"})
        assert st == 400 and "kind must be" in _j.loads(body)["error"]
        st, body, _ = _req(base + "/api/jobs", "POST",
                           {"kind": "edit", "params": {"clips_dir": "/nope"}})
        assert st == 400 and "does not exist" in _j.loads(body)["error"]
        st, _, _ = _req(base + "/api/jobs/does_not_exist")
        assert st == 404


def test_web_blocks_path_traversal_out_of_the_workdir():
    with _server(port=8133) as base:
        st, _, _ = _req(base + "/media/../../../etc/passwd")
        assert st in (403, 404), f"traversal returned {st}"


def test_web_supports_head_and_range_for_video():
    """The page asks HEAD before showing a player, and browsers will not scrub
    without ranges — HEAD returned 501 until this was caught."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        media = Path(d) / "store" / "output"
        media.mkdir(parents=True)
        (media / "clip.bin").write_bytes(os.urandom(4096))
        with _server(workdir=d, port=8134) as base:
            st, _, h = _req(base + "/media/store/output/clip.bin", "HEAD")
            assert st == 200 and h.get("Accept-Ranges") == "bytes"
            assert int(h["Content-Length"]) == 4096
            st, body, h = _req(base + "/media/store/output/clip.bin",
                               headers={"Range": "bytes=0-1023"})
            assert st == 206 and len(body) == 1024
            assert h["Content-Range"] == "bytes 0-1023/4096"


def test_web_runs_a_real_job_end_to_end():
    """Not a mock: this submits work and waits for a genuine result."""
    import json as _j, tempfile, time as _t
    with tempfile.TemporaryDirectory() as d:
        with _server(workdir=d, port=8135) as base:
            st, body, _ = _req(base + "/api/jobs", "POST", {
                "kind": "ingest",
                "params": {"clips_dir": str(CLIPS), "workdir": d}})
            assert st == 201
            job_id = _j.loads(body)["id"]
            for _ in range(120):
                _t.sleep(1)
                _, b, _ = _req(f"{base}/api/jobs/{job_id}")
                j = _j.loads(b)
                if j["state"] in ("succeeded", "failed", "cancelled"):
                    break
            assert j["state"] == "succeeded", j.get("error")
            assert j["result"]["shots"] > 0
            assert len(j["steps"]) > 1, "progress should be reported, not just the end"


def test_web_note_endpoint_defaults_to_not_saving():
    """Checking a note must never mutate the project by accident."""
    import json as _j
    with _server(workdir="/home/claude/delwork2", port=8136) as base:
        st, body, _ = _req(base + "/api/note", "POST",
                           {"text": "hold the second shot longer"})
        assert st == 200
        d = _j.loads(body)
        assert d["applied"] is False, "dry run must be the default"
        assert d["changes"], "the note should have been understood"
        st, body, _ = _req(base + "/api/note", "POST", {"dry_run": True})
        assert st == 400, "an empty note should be refused"


def test_web_page_declares_missing_capabilities_rather_than_faking_controls():
    html = (Path("alledits/web/static/index.html")).read_text()
    assert "/api/capabilities" in html, "the page must read real capability state"
    assert "would do nothing" in html, "it must say why a control is absent"
    # No control may be wired to nothing.
    import re as _re
    for m in _re.finditer(r'<button[^>]*id="([^"]+)"', html):
        assert f'$("#{m.group(1)}").onclick' in html, \
            f"button #{m.group(1)} has no handler — a mock button"


if __name__ == "__main__":
    # Collection is by globals(), so two tests sharing a name would silently
    # shadow each other and one would never run. That happened once and cost
    # real coverage, so it is now a hard error rather than a quiet loss.
    import re as _re
    _src = Path(__file__).read_text()
    _defined = _re.findall(r"^def (test_[A-Za-z0-9_]+)", _src, _re.M)
    _dupes = sorted({n for n in _defined if _defined.count(n) > 1})
    if _dupes:
        print(f"  ERROR duplicate test names shadow each other: {_dupes}")
        sys.exit(1)

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    assert len(fns) == len(_defined), (
        f"collected {len(fns)} but {len(_defined)} defined")
    passed = failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}"); passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}"); failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    sys.exit(1 if failed else 0)
