"""Proof that the sound stage does what the plan claims.

Three claims are checked by measuring the audio, never by trusting the plan:

1. **Loudness** lands on the platform target within EBU R128 delivery tolerance.
2. **True peak** stays under the ceiling AFTER lossy encoding, which is where a
   limiter set exactly at the ceiling fails.
3. **Ducking** actually attenuates music while speech is present, by an amount
   that tracks what was requested — and saturates honestly when it cannot.

    python3 scripts/sound_test.py
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alledits.audio.mix import (MixPlan, MixTrack, DuckSpec, TrackRole,  # noqa: E402
                                plan_mix, validate_mix, MAX_ACHIEVABLE_DUCK_DB)
from alledits.core.ffmpeg import ffmpeg  # noqa: E402
from alledits.render.ffmpeg_renderer import _build_mix_graph, _solve_gain  # noqa: E402

MUSIC = Path("/home/claude/testmedia/music.wav")
TMP = Path("/tmp/soundtest")
TMP.mkdir(exist_ok=True)


def ebur128(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Integrated loudness"):]
    I = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    TP = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
    return (float(I.group(1)) if I else None,
            float(TP.group(1)) if TP else None)


def band_level(path, t0, t1, lo=700, hi=1400):
    """Level of one frequency band, so music can be measured past the voice."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                        f"atrim={t0}:{t1},asetpts=PTS-STARTPTS,"
                        f"highpass=f={lo},lowpass=f={hi},volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def check_loudness(failures):
    print("=" * 74)
    print("LOUDNESS & TRUE PEAK — measured on the encoded deliverable")
    print("=" * 74)

    class T:
        duration = 8.0

    src_i, src_tp = ebur128(MUSIC)
    print(f"\n  source music        I={src_i:>7.1f} LUFS  TP={src_tp:>6.1f} dBFS")

    for target in ("social", "broadcast"):
        plan = plan_mix(T(), str(MUSIC), target=target)
        problems = validate_mix(plan)
        if problems:
            failures.append(f"{target}: invalid plan {problems}")
            continue
        gain, pre_i, _ = _solve_gain(plan, 8.0)
        if gain is None:
            failures.append(f"{target}: could not solve gain")
            continue
        fc, lbl, inputs = _build_mix_graph(plan, 8.0, 0, gain_db=gain)
        out = TMP / f"mix_{target}.m4a"
        ffmpeg([*inputs, "-filter_complex", ";".join(fc), "-map", f"[{lbl}]",
                "-c:a", "aac", "-b:a", "192k", "-t", "8.0", str(out)])
        I, TP = ebur128(out)
        d_ok = abs(I - plan.target_lufs) <= 0.5
        p_ok = TP <= plan.true_peak_db
        print(f"\n  target {target:<10} {plan.target_lufs:>6.1f} LUFS  "
              f"ceiling {plan.true_peak_db:>5.1f} dBTP   (gain {gain:+.1f} dB)")
        print(f"    delivered         I={I:>7.1f} LUFS  TP={TP:>6.1f} dBFS   "
              f"{'PASS' if d_ok else 'FAIL'} / {'PASS' if p_ok else 'FAIL'}")
        if not d_ok:
            failures.append(f"{target}: {I} LUFS vs target {plan.target_lufs}")
        if not p_ok:
            failures.append(f"{target}: TP {TP} exceeds {plan.true_peak_db}")


def check_ducking(failures):
    print("\n" + "=" * 74)
    print("DUCKING — music must move under speech")
    print("=" * 74)
    voice, bed = TMP / "voice.wav", TMP / "bed.wav"
    # Speech present only in the middle, giving a clean before/during comparison.
    ffmpeg(["-f", "lavfi", "-i", "sine=frequency=300:duration=9", "-af",
            "volume='if(between(t,3,6),1,0)':eval=frame",
            "-c:a", "pcm_s16le", str(voice)])
    ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1000:duration=9",
            "-af", "volume=-6dB", "-c:a", "pcm_s16le", str(bed)])

    def render(depth, ducked):
        plan = MixPlan(normalize=False, duck=DuckSpec(depth_db=depth))
        plan.tracks = [
            MixTrack(id="vox", source_path=str(voice), role=TrackRole.VOICE.value,
                     source_in=0, source_out=9),
            MixTrack(id="music", source_path=str(bed), role=TrackRole.MUSIC.value,
                     source_in=0, source_out=9,
                     ducked_by=["vox"] if ducked else []),
        ]
        fc, lbl, inp = _build_mix_graph(plan, 9.0, 0)
        out = TMP / f"duck_{depth}_{ducked}.wav"
        ffmpeg([*inp, "-filter_complex", ";".join(fc), "-map", f"[{lbl}]",
                "-c:a", "pcm_s16le", "-t", "9", str(out)])
        return out

    base = render(6, False)
    b_quiet, b_speech = band_level(base, 1, 2.5), band_level(base, 4, 5.5)
    baseline = b_speech - b_quiet
    print(f"\n  {'asked':>6}{'achieved':>11}   note")
    print(f"  {'none':>6}{0.0:>11.2f}   reference (no sidechain)")

    achieved = []
    for d in (3, 6, 9, 12):
        o = render(d, True)
        got = (band_level(o, 4, 5.5) - band_level(o, 1, 2.5)) - baseline
        achieved.append((d, -got))
        cap = d > MAX_ACHIEVABLE_DUCK_DB
        print(f"  {d:>6}{-got:>11.2f}   "
              f"{'saturated (honest cap)' if cap else 'tracks request'}")

    if not all(a > 2.0 for _, a in achieved):
        failures.append(f"ducking too shallow: {achieved}")
        print("\n  duck engages:                  FAIL")
    else:
        print("\n  duck engages (>2 dB):          PASS")

    # Asking for more must never deliver less; equal is correct at saturation.
    mono = all(achieved[i][1] <= achieved[i + 1][1] + 0.3
               for i in range(len(achieved) - 1))
    print(f"  deeper request never shallower: {'PASS' if mono else 'FAIL'}")
    if not mono:
        failures.append(f"duck not monotonic: {achieved}")

    # The stated depth must be the deliverable one, not the requested one.
    spec = DuckSpec(depth_db=12.0)
    honest = spec.achievable_depth_db <= MAX_ACHIEVABLE_DUCK_DB + 1e-6
    print(f"  claim capped to deliverable:    {'PASS' if honest else 'FAIL'}")
    if not honest:
        failures.append("duck claims more than it can deliver")


def check_diegetic(failures):
    """End to end: a clip carrying speech must duck the music under itself.

    Built explicitly rather than by running the selector, because whether the
    selector happens to choose a talking portion is luck — and a proof that
    depends on luck proves nothing.
    """
    print("\n" + "=" * 74)
    print("DIEGETIC DUCKING — clip audio drives the mix")
    print("=" * 74)
    from alledits.audio.mix import diegetic_voice_tracks, plan_mix
    from alledits.audio.speech import detect_speech
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings

    talking = Path("/home/claude/testmedia/clips/17_talking_head.mp4")
    GROUND_TRUTH = [(2.0, 5.0), (8.0, 11.5), (14.0, 16.5)]

    sp = detect_speech(talking)
    print(f"\n  detected  {[(round(a, 2), round(b, 2)) for a, b in sp.windows]}")
    print(f"  truth     {GROUND_TRUTH}")
    ok = len(sp.windows) == len(GROUND_TRUTH) and all(
        abs(a - g[0]) < 0.15 and abs(b - g[1]) < 0.15
        for (a, b), g in zip(sp.windows, GROUND_TRUTH))
    print(f"  windows match ground truth:     {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"speech windows {sp.windows} != {GROUND_TRUTH}")

    quiet = detect_speech(MUSIC)
    print(f"  music is not mistaken for speech:"
          f" {'PASS' if not quiet.has_speech else 'FAIL'}")
    if quiet.has_speech:
        failures.append("music.wav was detected as speech")

    # A timeline whose clip sits squarely inside a speaking window.
    tl = Timeline(project=ProjectSettings(1080, 1920, 30))
    tl.clips.append(TimelineClip(
        id="c000", source_id="s", source_path=str(talking),
        source_in=2.0, source_out=5.0, timeline_start=0.0, duration=3.0))
    voices = diegetic_voice_tracks(tl)
    print(f"  voice tracks found:             "
          f"{len(voices)} {'PASS' if voices else 'FAIL'}")
    if not voices:
        failures.append("no voice track from a clip that is entirely speech")
        return

    plan = plan_mix(tl, str(MUSIC), voice_tracks=voices)
    music = [t for t in plan.tracks if t.role == "music"][0]
    print(f"  music ducked by voice:          "
          f"{'PASS' if music.ducked_by else 'FAIL'}")
    if not music.ducked_by:
        failures.append("music not ducked despite a voice track")
        return

    # Render with and without the duck and compare the music band during speech.
    def render(ducked):
        import copy
        p2 = copy.deepcopy(plan)
        p2.normalize = False
        for t in p2.tracks:
            if t.role == "music":
                t.ducked_by = list(music.ducked_by) if ducked else []
                t.fade_out = 0.0
        fc, lbl, inp = _build_mix_graph(p2, 3.0, 0)
        out = TMP / f"dieg_{ducked}.wav"
        ffmpeg([*inp, "-filter_complex", ";".join(fc), "-map", f"[{lbl}]",
                "-c:a", "pcm_s16le", "-t", "3", str(out)])
        return out

    # Measure BELOW the speech band. Above it, the voice fixture's out-of-band
    # residual is still louder than the music (-26 dB vs -41 dB), so the duck is
    # invisible there; under 180 Hz the music leads by ~13 dB.
    off = band_level(render(False), 0.5, 2.5, lo=20, hi=180)
    on = band_level(render(True), 0.5, 2.5, lo=20, hi=180)
    delta = on - off
    print(f"  music level {off:.2f} -> {on:.2f} dB during speech "
          f"({delta:+.2f} dB)")
    good = delta < -2.0
    print(f"  duck audibly engages:           {'PASS' if good else 'FAIL'}")
    if not good:
        failures.append(f"diegetic duck only {delta:.2f} dB")


def main():
    failures = []
    check_loudness(failures)
    check_ducking(failures)
    check_diegetic(failures)
    print("\n" + "=" * 74)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        return 1
    print("All sound checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
