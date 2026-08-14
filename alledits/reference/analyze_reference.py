"""Reference edit analysis (Spec §9).

Measures, from the reference video itself:
  - every cut point (content-aware shot detection)
  - shot-duration distribution and rhythm class
  - an intensity curve built from cut density + motion + luminance dynamics
  - transition types inferred at each cut:
        flash     : luminance spike across the boundary
        whip      : very high optical flow immediately around the boundary
        dissolve  : gradual content change over several frames
        cut       : otherwise
  - grading direction (contrast/saturation/warmth/black level)
  - narrative structure segmentation from the intensity curve
  - beat-sync strength, if the reference has music
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..media.probe import probe
from ..media.scenes import detect_shots, content_curve
from ..media.frames import FrameReader
from ..media.visual import analyze_shot, dominant_colors
from ..core.ids import content_id, file_fingerprint
from .grammar import (StyleGrammar, PacingProfile, MotionProfile,
                      ColorProfile, TransitionProfile)


def _classify_rhythm(durations, times):
    if len(durations) < 4:
        return "steady"
    d = np.asarray(durations)
    cv = d.std() / max(d.mean(), 1e-6)
    half = len(d) // 2
    first, second = d[:half].mean(), d[half:].mean()
    if cv > 0.85:
        return "bursty"
    if second < first * 0.72:
        return "accelerating"
    if second > first * 1.38:
        return "decelerating"
    return "steady"


def _infer_transitions(path, shots, content_t, content_s, lumas):
    """Infer the transition used at each cut from pixel evidence.

    The discriminator is how the change is DISTRIBUTED in time, not how large it
    is:

      hard cut : change concentrated in a single frame -> high peak ratio
      dissolve : comparable total change spread over many frames -> low peak ratio
      flash    : a TRANSIENT luminance spike (up then back down) at the boundary.
                 A merely bright incoming shot is NOT a flash.
      whip     : an extreme single-frame spike — a cut hidden inside motion smear

    Absolute change alone misfires badly on footage that is itself busy, because
    the frames either side of a straight cut are already changing fast.
    """
    kinds = []
    med_s = float(np.median(content_s)) or 1e-6
    hi_s = float(np.percentile(content_s, 99))
    base_l = float(np.median(lumas))

    for sh in shots[1:]:
        t = sh.start
        i = int(np.argmin(np.abs(content_t - t)))
        w = 3
        lo, hi = max(0, i - w), min(len(content_s), i + w + 1)
        local_s = content_s[lo:hi]
        local_l = lumas[lo:hi]
        if len(local_s) < 3:
            kinds.append("cut")
            continue

        peak = float(local_s.max())
        peak_ratio = peak / (float(local_s.sum()) + 1e-9)   # 1.0 = all change in one frame

        # flash: transient brightness spike that comes back down
        pk_i = int(np.argmax(local_l))
        is_transient = (0 < pk_i < len(local_l) - 1 and
                        local_l[pk_i] > max(local_l[0], local_l[-1]) * 1.35 and
                        local_l[pk_i] > base_l * 1.4)
        if is_transient and local_l[pk_i] > 150:
            kinds.append("flash")
            continue

        # whip: one enormous single-frame spike
        if peak > hi_s and peak_ratio > 0.45:
            kinds.append("whip")
            continue

        # dissolve: change genuinely spread across frames
        elevated = int((local_s > med_s * 1.5).sum())
        if peak_ratio < 0.34 and elevated >= 4 and peak < hi_s:
            kinds.append("dissolve")
            continue

        kinds.append("cut")
    return kinds


def analyze_reference(path, label: str = "reference",
                      music_beats: list | None = None) -> StyleGrammar:
    info = probe(path)
    g = StyleGrammar(id=content_id("style", file_fingerprint(path)),
                     source_label=label, duration=info.duration)

    shots = detect_shots(path, info.duration, min_shot=0.12, sensitivity=1.6,
                         analysis_fps=16.0)
    times, scores, lumas = content_curve(path, analysis_fps=16.0)
    durations = [s.duration for s in shots]
    starts = [s.start for s in shots]

    # ---------- pacing ----------
    d = np.asarray(durations)
    p = PacingProfile(
        cuts_per_second=float(max(0, len(shots) - 1) / max(info.duration, 1e-6)),
        mean_shot=float(d.mean()), median_shot=float(np.median(d)),
        p10_shot=float(np.percentile(d, 10)), p90_shot=float(np.percentile(d, 90)),
        shot_duration_std=float(d.std()),
        rhythm=_classify_rhythm(durations, starts),
        duration_histogram=[float(x) for x in np.histogram(d, bins=8, range=(0, max(2.0, d.max())))[0]],
    )
    # fastest 2s window
    if len(starts) > 2:
        best, bt = 0, (0.0, 0.0)
        for s in starts:
            n = sum(1 for x in starts if s <= x < s + 2.0)
            if n > best:
                best, bt = n, (s, s + 2.0)
        p.fastest_window = bt
    g.pacing = p

    # ---------- motion + colour, sampled across shots ----------
    sample = shots[:: max(1, len(shots) // 12)][:12]
    vis = []
    for sh in sample:
        try:
            vis.append(analyze_shot(path, sh.start, min(sh.end, sh.start + 1.5),
                                    analysis_fps=8.0))
        except Exception:
            pass
    if vis:
        flows = [v.flow_magnitude for v in vis]
        moves = {}
        for v in vis:
            moves[v.camera_movement] = moves.get(v.camera_movement, 0) + 1
        g.motion = MotionProfile(
            mean_flow=float(np.mean(flows)),
            motion_variance=float(np.var(flows)),
            dominant_moves=sorted([(k, v / len(vis)) for k, v in moves.items()],
                                  key=lambda x: -x[1])[:4],
            zoom_tendency=float(np.mean([v.zoom_rate for v in vis])),
            shake_level=float(np.mean([v.shake for v in vis])),
        )
        g.color = ColorProfile(
            brightness=float(np.mean([v.brightness for v in vis])),
            contrast=float(np.mean([v.contrast for v in vis])),
            saturation=float(np.mean([v.saturation for v in vis])),
            warmth=float(np.mean([v.warmth for v in vis])),
            colorfulness=float(np.mean([v.colorfulness for v in vis])),
            key=max(set(v.key for v in vis), key=[v.key for v in vis].count),
            black_level=float(np.mean([v.clipped_shadows for v in vis])),
            highlight_level=float(np.mean([v.clipped_highlights for v in vis])),
        )
        mid = FrameReader(path, width=256).frame_at(info.duration / 2)
        if mid is not None:
            g.color.palette = dominant_colors(mid, 4)

    # ---------- transitions ----------
    kinds = _infer_transitions(path, shots, times, scores, lumas)
    n = max(1, len(kinds))
    g.transitions = TransitionProfile(
        hard_cut_share=kinds.count("cut") / n,
        flash_share=kinds.count("flash") / n,
        whip_share=kinds.count("whip") / n,
        dissolve_share=kinds.count("dissolve") / n,
        mean_transition_duration=0.0 if kinds.count("cut") == n else 0.12,
    )

    # ---------- intensity curve ----------
    grid = np.linspace(0, info.duration, 64)
    cut_density = np.zeros_like(grid)
    for s in starts:
        cut_density += np.exp(-((grid - s) ** 2) / (2 * 0.6 ** 2))
    if cut_density.max() > 0:
        cut_density /= cut_density.max()
    motion_i = np.interp(grid, times, ndimage.uniform_filter1d(scores, 8)) if len(times) else np.zeros_like(grid)
    if motion_i.max() > 0:
        motion_i /= motion_i.max()
    inten = np.clip(0.62 * cut_density + 0.38 * motion_i, 0, 1)
    inten = ndimage.uniform_filter1d(inten, 3)
    g.intensity_curve = [{"t": float(t), "value": float(v)} for t, v in zip(grid, inten)]

    # ---------- narrative structure from the intensity curve ----------
    g.structure = _segment_structure(grid, inten, info.duration)

    # ---------- effect density proxy ----------
    g.effect_density = float(np.clip(
        0.5 * (1 - g.transitions.hard_cut_share) +
        0.5 * min(1.0, g.motion.motion_variance / 2.0), 0, 1))

    # ---------- beat sync strength ----------
    if music_beats:
        b = np.asarray(music_beats)
        errs = [float(np.min(np.abs(b - s))) for s in starts[1:]]
        if errs:
            g.beat_sync_strength = float(np.clip(1.0 - np.mean(errs) / 0.25, 0, 1))

    g.notes.append("Grammar stores measured editing characteristics only — "
                   "no frames, audio or content from the reference are retained.")
    return g


def _segment_structure(grid, inten, duration):
    """Label the edit's arc from its intensity curve. Roles are assigned by
    observed shape, not forced onto a fixed template (Spec §23)."""
    if duration <= 0:
        return []
    peak_i = int(np.argmax(inten))
    peak_t = float(grid[peak_i])
    segs = []
    hook_end = min(duration * 0.15, 2.0)
    segs.append({"role": "hook", "start": 0.0, "end": hook_end})
    if peak_t > hook_end + 0.5:
        mid = hook_end + (peak_t - hook_end) * 0.45
        segs.append({"role": "setup", "start": hook_end, "end": mid})
        segs.append({"role": "escalation", "start": mid, "end": peak_t})
    else:
        segs.append({"role": "escalation", "start": hook_end, "end": max(hook_end + 0.5, peak_t)})
    clim_end = min(duration, peak_t + max(1.0, duration * 0.2))
    segs.append({"role": "climax", "start": max(peak_t, segs[-1]["end"]), "end": clim_end})
    if clim_end < duration - 0.2:
        segs.append({"role": "release", "start": clim_end, "end": duration})
    return [s for s in segs if s["end"] > s["start"] + 0.05]
