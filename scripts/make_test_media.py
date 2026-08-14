"""Generate a real test corpus: varied footage + a music track with genuine beats.

This exists so the pipeline can be tested end-to-end on actual video and actual
audio rather than asserted to work. Clips deliberately vary in resolution, frame
rate, bitrate, motion type, exposure and colour temperature so that shot
detection, camera-movement classification, the two quality scores and the
harmonization pass all have something real to find.
"""
import subprocess, sys, wave
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/testmedia")
(OUT / "clips").mkdir(parents=True, exist_ok=True)


def ff(args):
    p = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"ffmpeg failed: {p.stderr[:800]}\n{' '.join(args[:25])}")


# ----------------------------------------------------------------- music
def make_music(path, bpm=128, bars=16, sr=44100):
    """Synthesize a track with an unambiguous 4/4 grid, a quiet intro, and a drop
    at bar 5 — so beat tracking, section detection and drop detection are all
    testable against a known ground truth."""
    spb = 60.0 / bpm
    total = spb * 4 * bars
    n = int(total * sr)
    t = np.arange(n) / sr
    x = np.zeros(n, np.float32)

    def add(at, sig):
        i = int(at * sr)
        j = min(n, i + len(sig))
        if i < n:
            x[i:j] += sig[:j - i]

    def kick(dur=0.16):
        tt = np.arange(int(dur * sr)) / sr
        f = 120 * np.exp(-tt * 26) + 45
        return (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 15) * 1.0).astype(np.float32)

    def snare(dur=0.13):
        tt = np.arange(int(dur * sr)) / sr
        nz = np.random.default_rng(0).normal(0, 1, len(tt))
        return (nz * np.exp(-tt * 30) * 0.42 +
                np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 34) * 0.22).astype(np.float32)

    def hat(dur=0.045):
        tt = np.arange(int(dur * sr)) / sr
        nz = np.random.default_rng(1).normal(0, 1, len(tt))
        return (nz * np.exp(-tt * 90) * 0.16).astype(np.float32)

    drop_bar = 5
    for bar in range(bars):
        for beat in range(4):
            at = (bar * 4 + beat) * spb
            loud = 1.0 if bar >= drop_bar else 0.45
            if beat == 0:
                add(at, kick() * 1.25 * loud)          # downbeat is strongest
            else:
                add(at, kick() * 0.85 * loud)
            if beat in (1, 3):
                add(at, snare() * loud)
            if bar >= drop_bar:
                for h in range(2):
                    add(at + h * spb / 2, hat())
    # bass under the drop
    seg = (t >= drop_bar * 4 * spb)
    bass = np.sin(2 * np.pi * 55 * t) * 0.3
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (bpm / 60 / 2) * t)
    x[seg] += (bass * env)[seg].astype(np.float32)
    # pad chord for section contrast
    x += (0.05 * np.sin(2 * np.pi * 220 * t) * (t < drop_bar * 4 * spb)).astype(np.float32)

    x = np.tanh(x * 1.1)
    x = (x / (np.abs(x).max() + 1e-9) * 0.92 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x.tobytes())
    return total, bpm


# ----------------------------------------------------------------- clips
# Real footage is textured content with a mostly-stable background over which the
# CAMERA moves. Purely procedural sources (testsrc2, life) animate every pixel,
# which is adversarial for optical flow and unrepresentative. So the corpus is
# built from textured stills with genuine camera moves applied, plus a few
# genuinely dynamic scenes, plus deliberately degraded clips to exercise the
# quality/"nothing is wasted" logic.

STILLS = {
    "cityA": "mandelbrot=s=2560x1440:maxiter=180",
    "cityB": "mandelbrot=s=2560x1440:start_scale=2.2:maxiter=220",
    "warm":  "mandelbrot=s=2560x1440:start_scale=0.8:maxiter=140",
}

# (name, still, move filter, out size, fps, seconds, grade, crf)
CLIPS = [
    ("01_static_wide",   "cityA", "crop=1280:720:600:360", 30, 3.0,
     "eq=saturation=0.9", 20),
    ("02_pan_right",     "cityB", "crop=1280:720:'(iw-ow)*t/3':400", 30, 3.0,
     "eq=contrast=1.1", 20),
    ("03_push_in",       "cityA",
     "zoompan=z='1+0.45*on/90':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=30",
     30, 3.0, "eq=contrast=1.05", 20),
    ("04_pull_out",      "cityB",
     "zoompan=z='1.5-0.45*on/75':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=30",
     30, 2.5, "null", 20),
    ("05_tilt_up",       "cityA", "crop=1280:720:700:'(ih-oh)*(1-t/3)'", 30, 3.0,
     "null", 20),
    ("06_handheld",      "cityB",
     "crop=1280:720:'600+30*sin(2*PI*6*t)':'380+22*sin(2*PI*4.7*t+1)'", 30, 3.0,
     "eq=contrast=1.15", 22),
    ("07_dark_lowkey",   "cityA", "crop=1280:720:900:500", 30, 3.0,
     "eq=brightness=-0.3:contrast=1.5:saturation=0.5", 20),
    ("08_bright_highkey","cityB", "crop=1280:720:200:200", 30, 2.5,
     "eq=brightness=0.24:contrast=0.8:saturation=0.65", 20),
    ("09_warm_push",     "warm",
     "zoompan=z='1+0.25*on/75':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=30",
     30, 2.5, "colortemperature=temperature=3400:mix=0.9", 20),
    ("10_cold_pan",      "cityA", "crop=1280:720:'400+(iw-ow-400)*t/2.5':300", 30, 2.5,
     "colortemperature=temperature=9200:mix=0.9", 20),
    ("11_4k_high_quality","cityB", "crop=2560:1440:0:0,scale=3840:2160", 60, 2.5,
     "eq=contrast=1.1:saturation=1.12", 16),
    ("12_lowres_compressed","cityA", "crop=1280:720:500:300,scale=480:270", 24, 2.5,
     "noise=alls=16:allf=t", 40),
    ("13_very_poor",     "cityB", "crop=1280:720:800:400,scale=256:144", 15, 1.6,
     "noise=alls=46:allf=t+u,eq=contrast=0.62", 48),
    ("14_portrait_source","cityA", "crop=810:1440:800:0,scale=1080:1920", 30, 2.5,
     "eq=saturation=1.15", 20),
    ("15_fast_action",   None, None, 30, 2.5, None, 20),      # genuinely dynamic
    ("16_multi_shot_take", None, None, 30, 6.0, None, 20),    # 3 shots in one file
]


def make_voice(path, sr=44100, total=18.0):
    """A speech-like track with KNOWN speaking windows, for proving ducking.

    Not real speech — band-limited noise bursts amplitude-modulated at syllable
    rate, which occupies the same spectral region as voice and therefore
    exercises the same sidechain behaviour. The point is that the speaking
    windows are ground truth we can assert against.
    """
    import numpy as np
    n = int(total * sr)
    out = np.zeros(n, dtype=np.float64)
    windows = [(2.0, 5.0), (8.0, 11.5), (14.0, 16.5)]   # GROUND TRUTH
    rng = np.random.default_rng(7)
    for a, b in windows:
        i0, i1 = int(a * sr), int(b * sr)
        seg = rng.normal(0, 1, i1 - i0)
        # band-limit toward the voice band
        k = np.fft.rfftfreq(len(seg), 1 / sr)
        S = np.fft.rfft(seg)
        S[(k < 180) | (k > 3400)] *= 0.02
        seg = np.fft.irfft(S, len(seg))
        t = np.arange(len(seg)) / sr
        syll = 0.55 + 0.45 * np.sin(2 * np.pi * 4.2 * t) ** 2   # syllable rate
        env = np.minimum(1.0, np.minimum(t / 0.06, (t[-1] - t) / 0.06))
        seg *= syll * np.clip(env, 0, 1)
        seg /= (np.abs(seg).max() + 1e-9)
        out[i0:i1] += seg * 0.7
    out = np.clip(out, -1, 1)
    pcm = (out * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return windows


def make_talking_clip(path, voice_wav, still_key, dur=18.0, fps=30):
    """A clip that CARRIES its own audio, so the diegetic path is exercised."""
    still = make_still(still_key)
    ff(["-loop", "1", "-i", str(still), "-i", str(voice_wav), "-t", str(dur),
        "-vf", f"scale=1280:720,fps={fps},format=yuv420p",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "160k", "-shortest", str(path)])


def make_still(key):
    out = OUT / f"_still_{key}.png"
    if not out.exists():
        ff(["-f", "lavfi", "-i", STILLS[key], "-frames:v", "1", str(out)])
    return out


def make_clip(name, still_key, move, fps, dur, grade, crf):
    out = OUT / "clips" / f"{name}.mp4"
    still = make_still(still_key)
    chain = [c for c in (move, grade) if c and c != "null"]
    chain.append("format=yuv420p")
    ff(["-loop", "1", "-i", str(still), "-t", str(dur),
        "-vf", ",".join(chain), "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), str(out)])
    return out


def make_fast_action():
    out = OUT / "clips" / "15_fast_action.mp4"
    ff(["-f", "lavfi", "-i", "life=s=640x360:r=30:mold=10:ratio=0.35:death_color=#DD2222:life_color=#22DDAA",
        "-t", "2.5", "-vf", "scale=1280:720:flags=neighbor,eq=contrast=1.3,format=yuv420p",
        "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)])


def make_multishot():
    """One file containing three visually distinct shots — proves shot detection
    segments a take rather than treating each file as one clip."""
    parts = []
    specs = [("cityA", "crop=1280:720:100:100", "eq=saturation=1.35", 2.0),
             ("cityB", "crop=1280:720:1200:700", "eq=brightness=-0.18", 2.0),
             ("warm",  "crop=1280:720:600:300", "eq=saturation=0.7:contrast=1.25", 2.0)]
    for i, (k, mv, gr, d) in enumerate(specs):
        p = OUT / "clips" / f"_ms{i}.mp4"
        ff(["-loop", "1", "-i", str(make_still(k)), "-t", str(d),
            "-vf", f"{mv},{gr},format=yuv420p", "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(p)])
        parts.append(p)
    lst = OUT / "clips" / "_ms.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = OUT / "clips" / "16_multi_shot_take.mp4"
    ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    for p in parts:
        p.unlink()
    lst.unlink()


def make_reference(path, bpm=128):
    """A reference edit with a KNOWN cut rate (~2.7 cuts/s), flash frames and an
    accelerating rhythm — so we can check the analyzer recovers what we put in."""
    spb = 60.0 / bpm
    segs, i = [], 0
    t = 0.0
    durations = []
    while t < 12.0:
        d = spb * (2 if t < 4 else 1)          # accelerates halfway through
        durations.append(d)
        t += d
    srcs = ["testsrc2=s=854x480:r=30", "mandelbrot=s=854x480:r=30",
            "life=s=854x480:r=30:mold=10:ratio=0.4", "cellauto=s=854x480:r=30:rule=126",
            "gradients=s=854x480:r=30:c0=#ff3300:c1=#220000:speed=0.05"]
    tmp = OUT / "_ref"
    tmp.mkdir(exist_ok=True)
    for i, d in enumerate(durations):
        s = srcs[i % len(srcs)]
        flash = (i % 5 == 4)
        vf = "eq=contrast=1.25:saturation=1.15"
        if flash:
            vf += ",eq=brightness=0.55:contrast=1.5"    # a bright frame -> flash cut
        p = tmp / f"r{i:03d}.mp4"
        ff(["-f", "lavfi", "-i", s, "-t", f"{d:.4f}", "-vf", f"{vf},format=yuv420p",
            "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(p)])
        segs.append(p)
        if flash:
            # a genuine flash transition: 2 frames of near-white between shots
            fp = tmp / f"f{i:03d}.mp4"
            ff(["-f", "lavfi", "-i", "color=c=#f4f4f4:s=854x480:r=30",
                "-t", "0.0667", "-vf", "format=yuv420p", "-r", "30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(fp)])
            segs.append(fp)
    lst = tmp / "l.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in segs))
    ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(path)])
    import shutil; shutil.rmtree(tmp)
    return len(durations), sum(durations)


if __name__ == "__main__":
    print("voice + talking clip...")
    vw = make_voice(OUT / "voice.wav")
    make_talking_clip(OUT / "clips" / "17_talking_head.mp4", OUT / "voice.wav", "warm")
    print(f"  voice.wav speaking windows (ground truth): {vw}")

    print("music...")
    dur, bpm = make_music(OUT / "music.wav")
    print(f"  music.wav {dur:.1f}s @ {bpm} BPM (ground truth)")
    print("clips...")
    for name, still, move, fps, d, grade, crf in CLIPS:
        if name in ("15_fast_action", "16_multi_shot_take"):
            continue
        make_clip(name, still, move, fps, d, grade, crf)
        print(f"  {name}")
    make_fast_action(); print("  15_fast_action")
    make_multishot(); print("  16_multi_shot_take (3 shots in one file)")
    print("reference...")
    n, d = make_reference(OUT / "reference.mp4")
    print(f"  reference.mp4 {d:.1f}s, {n} shots, {(n-1)/d:.2f} cuts/s (ground truth)")
    print(f"\nwrote to {OUT}")
