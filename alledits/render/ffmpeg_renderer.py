"""FFmpeg renderer (Spec §12).

Strategy: render each clip to a normalized intermediate segment, then chain the
segments with xfade transitions, then lay in audio.

Why segments rather than one giant filter_complex:
  - incremental rendering + caching: an unchanged clip is not re-rendered on a
    revision pass, which is what makes the critique loop affordable
  - a failure is isolated to one clip and is legible
  - preview and final share one code path at different scales

Transition timing is compensated so that CUT POINTS STAY ON THE BEAT: each
segment is extended by half of each adjoining transition, and the xfade offset
is set so the transition is centred exactly on the planned cut.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ..core.ffmpeg import ffmpeg
from ..core.errors import RenderError
from ..timeline.schema import TransitionType
from .base import Renderer, RenderResult
from .filters import build_effect_chain, XFADE_MAP, stabilize_transform
from ..timeline.schema import EffectType


class FFmpegRenderer(Renderer):
    name = "ffmpeg"

    def __init__(self, workdir: Path, preview_scale: float = 0.5,
                 crf_preview: int = 26, crf_final: int = 18,
                 preset_preview: str = "veryfast", preset_final: str = "slow"):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.preview_scale = preview_scale
        self.crf = {True: crf_preview, False: crf_final}
        self.preset = {True: preset_preview, False: preset_final}

    # ------------------------------------------------------------------ utils
    def _dims(self, tl, preview):
        W, H = tl.project.width, tl.project.height
        if preview:
            W = int(W * self.preview_scale) // 2 * 2
            H = int(H * self.preview_scale) // 2 * 2
        return W, H

    def _seg_key(self, clip, W, H, fps, r_len, head, tail) -> str:
        h = hashlib.sha256()
        h.update(repr((clip.to_dict(), W, H, fps, round(r_len, 4),
                       round(head, 4), round(tail, 4))).encode())
        return h.hexdigest()[:20]

    # -------------------------------------------------------------- segments
    def _render_segment(self, clip, W, H, fps, r_len, head, tail, out: Path):
        """Render one clip, extended by `head`/`tail` for adjoining transitions,
        to exactly r_len seconds."""
        src_in = max(0.0, clip.source_in - head * clip.speed)
        # read only what the source actually has; tpad below holds the last frame
        # for any shortfall (clip.hold_extended), so r_len is always exact.
        src_dur = (r_len) * clip.speed
        chain = []

        # decode-side trim; -ss before -i is fast, accurate enough with -accurate_seek
        chain.append(f"fps={fps}")
        chain.append("scale=iw:ih")     # no-op anchor; real scaling happens in reframe/effects

        nframes = max(2, int(r_len * fps))

        # Stabilization needs a analysis pass over the actual frames before it can
        # transform them, so it cannot live in the single filter chain.
        stab = next((e for e in clip.effects
                     if e.type == EffectType.STABILIZE.value), None)
        stab_frags = []
        if stab is not None:
            trf = out.with_suffix(".trf")
            if not trf.exists():
                ffmpeg(["-accurate_seek", "-ss", f"{src_in:.5f}",
                        "-t", f"{src_dur + 0.5:.5f}", "-i", str(clip.source_path),
                        "-vf", (f"vidstabdetect=shakiness="
                                f"{max(1, min(10, int(1 + 9 * float(stab.params.get('strength', 0.5)))))}"
                                f":accuracy=12:result={trf}"),
                        "-f", "null", "-"])
            stab_frags = stabilize_transform(stab.params, str(trf), W, H)

        fx = build_effect_chain(clip.effects, W, H, nframes, fps)
        chain += stab_frags        # stabilize before any framing decision
        has_reframe = any(f.startswith("scale=") and "force_original_aspect_ratio" in f
                          for f in fx)
        if not has_reframe:
            chain.append(f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic")
            chain.append(f"crop={W}:{H}")
        chain += fx

        if abs(clip.speed - 1.0) > 1e-3:
            chain.append(f"setpts={1.0/clip.speed:.6f}*PTS")

        chain.append(f"scale={W}:{H}:flags=bicubic")
        chain.append("setsar=1")
        chain.append("format=yuv420p")
        # guarantee exact length even if the source runs short
        chain.append("tpad=stop_mode=clone:stop_duration=2")
        chain.append(f"trim=duration={r_len:.5f}")
        chain.append("setpts=PTS-STARTPTS")

        args = ["-accurate_seek", "-ss", f"{src_in:.5f}",
                "-t", f"{src_dur + 2.0:.5f}", "-i", str(clip.source_path),
                "-an", "-vf", ",".join(chain),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                "-video_track_timescale", "90000", str(out)]
        ffmpeg(args)
        return out

    # ----------------------------------------------------------------- render
    def render(self, timeline, out_path, preview: bool = True,
               progress=None) -> RenderResult:
        p = progress or (lambda *a, **k: None)
        tl = timeline
        if not tl.clips:
            raise RenderError("cannot render an empty timeline")

        W, H = self._dims(tl, preview)
        fps = tl.project.fps
        clips = sorted(tl.clips, key=lambda c: c.timeline_start)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        warnings = []

        # transition durations: d_in[i] for clip i
        d_in = []
        for c in clips:
            t = c.transition_in
            d = float(getattr(t, "duration", 0.0) or 0.0) if t else 0.0
            if t and getattr(t, "type", "cut") == TransitionType.CUT.value:
                d = 0.0
            d_in.append(d)

        # segment lengths: L_i + half of each adjoining transition
        seg_specs = []
        for i, c in enumerate(clips):
            head = d_in[i] / 2.0
            tail = (d_in[i + 1] / 2.0) if i + 1 < len(clips) else 0.0
            seg_specs.append((c, head, tail, c.timeline_duration + head + tail))

        seg_paths = []
        cache = self.workdir / ("preview" if preview else "final")
        cache.mkdir(parents=True, exist_ok=True)
        for i, (c, head, tail, r_len) in enumerate(seg_specs):
            key = self._seg_key(c, W, H, fps, r_len, head, tail)
            seg = cache / f"seg_{i:03d}_{key}.mp4"
            if not seg.exists():                      # incremental rendering
                p(0.05 + 0.65 * i / len(seg_specs),
                  f"rendering clip {i+1}/{len(seg_specs)} ({c.role})")
                try:
                    self._render_segment(c, W, H, fps, r_len, head, tail, seg)
                except RenderError as e:
                    raise RenderError(f"clip {c.id} ({c.source_path}): {e}")
            else:
                p(0.05 + 0.65 * i / len(seg_specs), f"clip {i+1} reused from cache")
            seg_paths.append(seg)

        # ---- chain with xfade ----
        p(0.72, "assembling transitions")
        inputs, fc = [], []
        for sp in seg_paths:
            inputs += ["-i", str(sp)]

        # Every branch is forced to one timebase and pixel format. concat emits
        # AVTB (1/1000000) while raw decoded inputs carry 1/90000, and xfade
        # refuses to join links whose timebases differ — so a chain that mixes
        # hard cuts (concat) with treated transitions (xfade) fails without this.
        for i in range(len(seg_paths)):
            # settb must come AFTER fps: the fps filter resets the timebase to 1/fps,
            # so setting it first is silently undone.
            fc.append(f"[{i}:v]fps={fps},format=yuv420p,setsar=1,settb=AVTB[s{i}]")

        cur = "s0"
        for i in range(1, len(seg_paths)):
            d = d_in[i]
            L_prev_planned = sum(c.timeline_duration for c, _, _, _ in seg_specs[:i])
            if d <= 0.001:
                fc.append(f"[{cur}][s{i}]concat=n=2:v=1:a=0,settb=AVTB[v{i}]")
            else:
                xf = XFADE_MAP.get(clips[i].transition_in.type, "dissolve")
                offset = max(0.0, L_prev_planned - d / 2.0)
                fc.append(f"[{cur}][s{i}]xfade=transition={xf}:"
                          f"duration={d:.4f}:offset={offset:.4f},settb=AVTB[v{i}]")
            cur = f"v{i}"

        fc.append(f"[{cur}]format=yuv420p,setsar=1[vout]")

        # ---- audio ----
        amap = []
        if getattr(tl, "mix", None) and tl.mix.tracks:
            gain_db = achieved_i = achieved_tp = None
            if tl.mix.normalize:
                p(0.76, "measuring loudness")
                gain_db, achieved_i, achieved_tp = _solve_gain(tl.mix, tl.duration)
            if gain_db is None:
                warnings.append("loudness measurement unavailable; using "
                                "single-pass normalisation (less accurate)")
            else:
                off = abs(achieved_i - tl.mix.target_lufs)
                p(0.78, f"loudness {achieved_i:.1f} LUFS (gain {gain_db:+.1f} dB)")
                if off > 0.5:
                    warnings.append(
                        f"loudness {achieved_i:.1f} LUFS is {off:.1f} LU from the "
                        f"{tl.mix.target_lufs:.1f} target; the peak ceiling could "
                        "not be held any louder without heavier limiting")
                tl.mix.achieved_lufs = achieved_i
                tl.mix.achieved_tp = achieved_tp
                tl.mix.applied_gain_db = gain_db
            afc, alabel, ainputs = _build_mix_graph(tl.mix, tl.duration,
                                                    base_index=len(seg_paths),
                                                    gain_db=gain_db)
            inputs += ainputs
            fc.extend(afc)
            amap = ["-map", f"[{alabel}]", "-c:a", "aac", "-b:a", "192k"]
        elif tl.audio:
            a = tl.audio[0]
            ai = len(seg_paths)
            inputs += ["-accurate_seek", "-ss", f"{a.source_in:.4f}",
                       "-i", str(a.source_path)]
            afilters = [f"atrim=duration={tl.duration:.4f}", "asetpts=PTS-STARTPTS"]
            if a.fade_in > 0:
                afilters.append(f"afade=t=in:st=0:d={a.fade_in:.3f}")
            if a.fade_out > 0:
                afilters.append(f"afade=t=out:st={max(0.0, tl.duration-a.fade_out):.3f}"
                                f":d={a.fade_out:.3f}")
            if abs(a.gain_db) > 0.01:
                afilters.append(f"volume={a.gain_db:.2f}dB")
            afilters.append("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo")
            fc.append(f"[{ai}:a]{','.join(afilters)}[aout]")
            amap = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]

        args = [*inputs, "-filter_complex", ";".join(fc), "-map", "[vout]", *amap,
                "-c:v", "libx264", "-preset", self.preset[preview],
                "-crf", str(self.crf[preview]), "-pix_fmt", "yuv420p",
                "-r", str(fps), "-movflags", "+faststart",
                "-t", f"{tl.duration:.4f}", str(out_path)]
        p(0.8, "encoding")
        ffmpeg(args)

        from ..media.probe import probe
        info = probe(out_path)
        drift = abs(info.duration - tl.duration)
        if drift > 0.08:
            warnings.append(f"rendered duration {info.duration:.3f}s differs from "
                            f"planned {tl.duration:.3f}s by {drift*1000:.0f}ms")
        p(1.0, "render complete")
        return RenderResult(path=out_path, duration=info.duration, width=info.width,
                            height=info.height, fps=info.fps, preview=preview,
                            segments=[str(s) for s in seg_paths], warnings=warnings)


def _build_mix_graph(plan, duration: float, base_index: int,
                     gain_db: float | None = None):
    """Turn a MixPlan into an ffmpeg audio filter graph.

    Order matters and is not arbitrary:
      per-track trim/gain/fade -> sidechain duck -> amix -> loudnorm -> limiter
    Ducking must happen BEFORE the mix (it acts on one track against another),
    and loudness normalisation must happen AFTER it, because normalising the
    parts individually would undo the level relationships the duck just created.
    """
    fc, inputs = [], []
    labels = {}
    for n, t in enumerate(plan.tracks):
        idx = base_index + n
        inputs += ["-accurate_seek", "-ss", f"{t.source_in:.4f}",
                   "-i", str(t.source_path)]
        f = [f"atrim=duration={max(0.01, min(t.duration, duration)):.4f}",
             "asetpts=PTS-STARTPTS",
             "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"]
        if t.timeline_start > 0.001:
            f.append(f"adelay={int(t.timeline_start * 1000)}:all=1")
        if abs(t.gain_db) > 0.01:
            f.append(f"volume={t.gain_db:.2f}dB")
        if t.fade_in > 0:
            f.append(f"afade=t=in:st=0:d={t.fade_in:.3f}")
        if t.fade_out > 0:
            f.append(f"afade=t=out:st={max(0.0, duration - t.fade_out):.3f}"
                     f":d={t.fade_out:.3f}")
        # Pad so every track spans the full programme; amix otherwise ends at
        # the shortest input and the tail of the edit goes silent.
        f.append(f"apad=whole_dur={duration:.4f}")
        lbl = f"a{n}"
        fc.append(f"[{idx}:a]{','.join(f)}[{lbl}]")
        labels[t.id] = lbl

    # ---- sidechain ducking ----
    for t in plan.tracks:
        keys = [d for d in t.ducked_by if d in labels]
        if not keys:
            continue
        d = plan.duck
        # The sidechain key is a COPY of the voice: it steers the compressor and
        # must still reach the mix in its own right.
        if len(keys) == 1:
            key_lbl = labels[keys[0]]
            fc.append(f"[{key_lbl}]asplit=2[{key_lbl}_m][{key_lbl}_k]")
            labels[keys[0]] = f"{key_lbl}_m"
            sc = f"{key_lbl}_k"
        else:
            parts = []
            for k in keys:
                fc.append(f"[{labels[k]}]asplit=2[{labels[k]}_m][{labels[k]}_k]")
                parts.append(f"[{labels[k]}_k]")
                labels[k] = f"{labels[k]}_m"
            fc.append(f"{''.join(parts)}amix=inputs={len(parts)}:normalize=0[sckey]")
            sc = "sckey"
        # makeup=1 keeps the un-ducked level intact; the ratio and threshold set
        # how far it drops while the key is present.
        fc.append(f"[{labels[t.id]}][{sc}]sidechaincompress="
                  f"threshold={d.threshold}:ratio={d.ratio:.1f}"
                  f":attack={d.attack_ms:.0f}:release={d.release_ms:.0f}"
                  f":makeup=1:level_sc=1[{labels[t.id]}_d]")
        labels[t.id] = f"{labels[t.id]}_d"

    # ---- mix down ----
    order = [labels[t.id] for t in plan.tracks]
    if len(order) == 1:
        cur = order[0]
    else:
        cur = "amixed"
        fc.append("".join(f"[{l}]" for l in order)
                  + f"amix=inputs={len(order)}:normalize=0:dropout_transition=0[{cur}]")

    # ---- loudness + peak ceiling ----
    # Deliberately NOT loudnorm's dynamic mode. Measured against this material it
    # landed 1.4-2.2 LU short of target, because it protects peaks by backing the
    # whole programme off. Explicit gain-to-target plus a limiter reaches the
    # target (measured -13.8 LUFS vs -14.0 asked) and only touches the transients
    # that actually threaten the ceiling.
    ceiling = plan.true_peak_db - CODEC_OVERSHOOT_MARGIN_DB
    if plan.normalize and gain_db is not None:
        fc.append(f"[{cur}]volume={gain_db:.2f}dB[{cur}_g]")
        cur = f"{cur}_g"
    elif plan.normalize:
        # No measurement available: fall back to single-pass loudnorm. Less
        # accurate, but better than shipping an unnormalised mix.
        fc.append(f"[{cur}]loudnorm=I={plan.target_lufs:.1f}:TP={ceiling:.1f}"
                  f":LRA={plan.target_lra:.1f}[{cur}_ln]")
        cur = f"{cur}_ln"

    fc.append(f"[{cur}]alimiter=limit={10 ** (ceiling / 20):.4f}:level=disabled,"
              f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]")
    return fc, "aout", inputs


# Measured across ceilings from -1.0 to -3.0 dBTP: AAC at 192k reconstructs peaks
# 1.2-2.0 dB above the value handed to it. A limiter set at the delivery ceiling
# therefore ships a file that violates it, so the limiter sits this far below.
CODEC_OVERSHOOT_MARGIN_DB = 2.0

# How far the mix may be pushed to reach the loudness target. Beyond this the
# limiter is doing more harm to the material than the loudness gain is worth.
# A speech-led mix sits quiet (music bedded at -10 dB, voice only intermittent),
# so it legitimately needs more make-up than a music-only programme: an 18 dB
# ceiling was reached and silently clamped on a real run.
MAX_MAKEUP_GAIN_DB = 24.0

# EBU R128 delivery tolerance. Converge to inside this, not to zero.
LOUDNESS_TOLERANCE_LU = 0.3

# Each iteration is one audio-only null encode (well under a second).
MAX_GAIN_ITERATIONS = 6


def _ebur128(args_in: list[str], fc: list[str], label: str):
    """Measure integrated loudness and true peak of a filter-graph output."""
    import re
    probe = list(fc) + [f"[{label}]ebur128=peak=true[m]"]
    err = ffmpeg(["-loglevel", "info", *args_in,
                  "-filter_complex", ";".join(probe),
                  "-map", "[m]", "-f", "null", "-"]) or ""
    tail = err[err.rfind("Integrated loudness"):]
    I = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    TP = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
    return (float(I.group(1)) if I else None,
            float(TP.group(1)) if TP else None)


def _solve_gain(plan, duration: float):
    """Find the gain that puts the mix on target, verifying the result.

    Measure, correct, re-measure. The loudness cost of limiting cannot be
    predicted from the input, so the correction has to be checked rather than
    assumed.

    Two bugs this shape avoids, both observed:
      - returning the gain produced by the LAST correction, which was never
        measured. The render then used a different gain than the one verified.
      - returning a gain outside the clamp, so the render silently applied the
        clamped value while the plan recorded the unclamped one.
    Only a (gain, loudness) pair that were measured TOGETHER is ever returned.

    Returns (gain_db, achieved_lufs, achieved_tp) or (None, None, None).
    """
    try:
        base_plan = _no_normalize(plan)
        fc, lbl, inputs = _build_mix_graph(base_plan, duration, base_index=0)
        raw_i, _ = _ebur128(inputs, fc, lbl)
        # ebur128 floors at exactly -70.0 LUFS, so a strict comparison lets
        # digital silence through and the solver then asks for +74 dB of gain.
        if raw_i is None or raw_i <= -70.0:
            return None, None, None      # silence: nothing to normalise

        def clamp(g):
            return max(-MAX_MAKEUP_GAIN_DB, min(MAX_MAKEUP_GAIN_DB, g))

        gain = clamp(plan.target_lufs - raw_i)
        best = None                      # (abs_err, gain, I, TP) — all measured together
        for _ in range(MAX_GAIN_ITERATIONS):
            fc, lbl, inputs = _build_mix_graph(plan, duration, base_index=0,
                                               gain_db=gain)
            got_i, got_tp = _ebur128(inputs, fc, lbl)
            if got_i is None:
                break
            err = plan.target_lufs - got_i
            if best is None or abs(err) < best[0]:
                best = (abs(err), gain, got_i, got_tp)
            if abs(err) <= LOUDNESS_TOLERANCE_LU:
                break
            nxt = clamp(gain + err)
            if abs(nxt - gain) < 0.05:
                break                    # clamped or converged: further steps do nothing
            gain = nxt
        if best is None:
            return None, None, None
        return best[1], best[2], best[3]
    except Exception:
        return None, None, None


def _no_normalize(plan):
    """A copy of the plan with normalisation off, for measuring the raw mix."""
    import copy
    p2 = copy.copy(plan)
    p2.normalize = False
    return p2
