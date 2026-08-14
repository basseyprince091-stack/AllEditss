"""Proof that footage restoration actually restores.

Renders a degraded clip twice — once untreated, once with the treatments the
quality analyzer prescribed — then re-measures both with the SAME analyzers used
during ingestion. The claim "we enhanced it" is only allowed if the numbers move
in the right direction.

Also verifies the negative case, which matters just as much: good footage must
come back with no treatments applied at all.

    python3 scripts/rescue_test.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from alledits.core.ffmpeg import ffmpeg  # noqa: E402
from alledits.media.probe import probe  # noqa: E402
from alledits.media.visual import analyze_shot  # noqa: E402
from alledits.media.quality import analyze_quality, _estimate_noise, _estimate_blockiness  # noqa: E402
from alledits.render.filters import build_effect_chain  # noqa: E402
from alledits.timeline.schema import Effect, EffectType  # noqa: E402

MEDIA = Path("/home/claude/testmedia/clips")
TMP = Path("/tmp/rescue")
TMP.mkdir(exist_ok=True)


def measure(path, start=0.1, end=1.4):
    """Measure the same signals the ingest analyzers use."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    sharp, noise, block = [], [], []
    n = 0
    while n < 12:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > 480:
            f = cv2.resize(f, (480, int(f.shape[0] * 480 / f.shape[1])),
                           interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        noise.append(_estimate_noise(g))
        block.append(_estimate_blockiness(g))
        n += 1
    cap.release()
    return {"sharpness": float(np.mean(sharp)) if sharp else 0.0,
            "noise": float(np.mean(noise)) if noise else 0.0,
            "blockiness": float(np.mean(block)) if block else 0.0}


def render(src, effects, out, dur=1.4):
    info = probe(src)
    W = info.width - info.width % 2
    H = info.height - info.height % 2
    chain = [f"fps=30"]
    chain += build_effect_chain(effects, W, H, int(dur * 30), 30)
    chain += [f"scale={W}:{H}", "setsar=1", "format=yuv420p"]
    ffmpeg(["-ss", "0.1", "-t", str(dur), "-i", str(src), "-an",
            "-vf", ",".join(chain), "-c:v", "libx264", "-crf", "12",
            "-preset", "veryfast", str(out)])
    return out


def prescribe(src):
    info = probe(src)
    v = analyze_shot(src, 0.1, 1.4)
    q = analyze_quality(src, 0.1, 1.4, info, visual=v)
    effects = []
    for d in q.defects:
        etype = {"denoise": EffectType.DENOISE, "sharpen": EffectType.SHARPEN,
                 "deblock": EffectType.DEBLOCK,
                 "expand_contrast": EffectType.EXPAND_CONTRAST}.get(d["treatment"])
        if etype is None:
            continue     # stabilize needs the two-pass renderer path
        sev = d["severity"]
        effects.append(Effect(etype.value,
                              {"strength": float(np.clip(0.25 + 0.6 * sev, 0.1, 1.0))},
                              reason=d["detail"]))
    return q, effects


def make_fixtures():
    """Purpose-built defects. The corpus clips exercise softness/blocking; noise
    and shake need footage that actually has them."""
    still = TMP / "still.png"
    if not still.exists():
        ffmpeg(["-f", "lavfi", "-i", "mandelbrot=s=1920x1080", "-frames:v", "1", str(still)])
    noisy, shaky = TMP / "noisy.mp4", TMP / "shaky.mp4"
    if not noisy.exists():
        ffmpeg(["-loop", "1", "-i", str(still), "-t", "2",
                "-vf", "crop=1280:720:400:300,noise=alls=42:allf=t+u,format=yuv420p",
                "-r", "30", "-c:v", "libx264", "-crf", "18", str(noisy)])
    if not shaky.exists():
        ffmpeg(["-loop", "1", "-i", str(still), "-t", "2",
                "-vf", ("crop=1280:720:'400+34*sin(2*PI*7.3*t)+18*sin(2*PI*11*t)'"
                        ":'300+28*sin(2*PI*5.7*t+1)',format=yuv420p"),
                "-r", "30", "-c:v", "libx264", "-crf", "18", str(shaky)])
    return noisy, shaky


def check_stabilize(shaky, failures):
    """Stabilization is two-pass, so it must be exercised through the renderer."""
    from alledits.timeline.schema import Timeline, TimelineClip, ProjectSettings
    from alledits.render.ffmpeg_renderer import FFmpegRenderer
    info = probe(shaky)
    v = analyze_shot(shaky, 0.1, 1.8)
    q = analyze_quality(shaky, 0.1, 1.8, info, visual=v)
    stab = [d for d in q.defects if d["treatment"] == "stabilize"]
    print(f"\n  shaky.mp4  handling={q.handling}  shake={v.shake:.2f} "
          f"consistency={v.motion_consistency:.2f}")
    if not stab:
        failures.append("stabilize not prescribed for obviously shaky footage")
        print("    FAIL: stabilize not prescribed")
        return
    tl = Timeline(project=ProjectSettings(1280, 720, 30))
    c = TimelineClip(id="c000", source_id="s", source_path=str(shaky),
                     source_in=0.1, source_out=1.9, timeline_start=0.0, duration=1.8)
    c.effects = [Effect(EffectType.STABILIZE.value,
                        {"strength": 0.7, "zoom": 0.04}, reason=stab[0]["detail"])]
    tl.clips.append(c)
    out = TMP / "stab_out.mp4"
    FFmpegRenderer(TMP / "stabwork").render(tl, out, preview=False,
                                            progress=lambda p, m: None)
    after = analyze_shot(out, 0.05, 1.7)
    ok = after.shake < v.shake * 0.7
    print(f"    shake {v.shake:.2f} -> {after.shake:.2f}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"stabilize: shake {v.shake:.2f}->{after.shake:.2f}")


def main():
    failures = []
    noisy, shaky = make_fixtures()

    print("=" * 78)
    print("RESTORATION — degraded footage should measurably improve")
    print("=" * 78)
    for name in ("13_very_poor", "12_lowres_compressed"):
        src = MEDIA / f"{name}.mp4"
        q, effects = prescribe(src)
        if not effects:
            print(f"\n{name}: no treatments prescribed — SKIP")
            continue
        before = measure(src)
        out = render(src, effects, TMP / f"{name}_fixed.mp4")
        after = measure(out)
        print(f"\n{name}  handling={q.handling}")
        print(f"  prescribed: {[e.type for e in effects]}")
        print(f"  {'metric':14}{'before':>10}{'after':>10}   direction")
        for k, better_is in (("sharpness", "up"), ("noise", "down"),
                             ("blockiness", "down")):
            b, a = before[k], after[k]
            ok = (a > b * 1.02) if better_is == "up" else (a < b * 0.98)
            treated = any(
                (k == "noise" and e.type == "denoise") or
                (k == "sharpness" and e.type == "sharpen") or
                (k == "blockiness" and e.type == "deblock")
                for e in effects)
            verdict = ("PASS" if ok else "FAIL") if treated else "n/a (untreated)"
            if treated and not ok:
                failures.append(f"{name}:{k} {b:.3f}->{a:.3f}")
            print(f"  {k:14}{b:>10.3f}{a:>10.3f}   want {better_is:<5} {verdict}")

    print("\n" + "=" * 78)
    print("DENOISE / STABILIZE — purpose-built defects")
    print("=" * 78)
    q, effects = prescribe(noisy)
    print(f"\n  noisy.mp4  handling={q.handling}  noise={q.noise:.3f}  "
          f"prescribed={[(e.type, round(e.params['strength'], 2)) for e in effects]}")
    if any(e.type == "denoise" for e in effects):
        b = measure(noisy)
        a = measure(render(noisy, effects, TMP / "noisy_fixed.mp4"))
        ok = a["noise"] < b["noise"] * 0.92
        print(f"    noise {b['noise']:.3f} -> {a['noise']:.3f}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"denoise: {b['noise']:.3f}->{a['noise']:.3f}")
    else:
        failures.append("denoise not prescribed for obviously noisy footage")
        print("    FAIL: denoise not prescribed")
    check_stabilize(shaky, failures)

    print("\n" + "=" * 78)
    print("RESTRAINT — good footage must be left alone")
    print("=" * 78)
    for name in ("11_4k_high_quality", "01_static_wide", "02_pan_right"):
        src = MEDIA / f"{name}.mp4"
        q, effects = prescribe(src)
        ok = q.handling == "use" and not effects
        print(f"  {name:22} handling={q.handling:10} treatments={[e.type for e in effects]} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name}: good footage was treated ({[e.type for e in effects]})")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        return 1
    print("All restoration checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
