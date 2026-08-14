"""Content-aware shot boundary detection.

Real algorithm (not a stub): per-frame HSV mean-absolute-difference produces a
content score; boundaries are peaks above an adaptive threshold (median + k*MAD),
with a minimum shot length to suppress flicker. This is the same family of
technique as PySceneDetect's ContentDetector, implemented directly so we carry
no unavailable dependency.

Used for BOTH user footage (segmenting long takes into shots) and reference edits
(recovering the editor's cut points -> pacing analysis).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from .frames import FrameReader


@dataclass
class Shot:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


def content_curve(path: Path | str, analysis_fps: float = 12.0):
    """Return (times, content_score, luma) sampled at analysis_fps."""
    import cv2
    times, scores, lumas = [], [], []
    prev = None
    with FrameReader(path, analysis_fps=analysis_fps, width=224) as fr:
        for t, frame in fr.iter_frames():
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            lumas.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            if prev is None:
                scores.append(0.0)
            else:
                # mean abs diff across H,S,V, normalized to ~0..100
                d = np.abs(hsv - prev)
                scores.append(float(d.mean()))
            times.append(t)
            prev = hsv
    return np.asarray(times), np.asarray(scores), np.asarray(lumas)


def detect_shots(path: Path | str, duration: float, min_shot: float = 0.35,
                 sensitivity: float = 1.0, analysis_fps: float = 12.0) -> list[Shot]:
    times, scores, _ = content_curve(path, analysis_fps=analysis_fps)
    if len(times) < 3:
        return [Shot(0, 0.0, duration)]

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) or 1e-6
    # Adaptive threshold; sensitivity>1 finds more cuts.
    thresh = med + (6.0 / max(0.2, sensitivity)) * mad
    thresh = max(thresh, 3.0)

    # A real cut is a PROMINENT spike, not merely a busy frame. Content that is
    # internally dynamic (crowds, water, fast action, animation) sustains a high
    # content score throughout, so an absolute threshold alone over-segments it.
    # Requiring the peak to stand well above its own local neighbourhood fixes
    # that without making the detector insensitive to genuine cuts.
    cut_times = []
    last = -1e9
    win = 5
    for i, (t, sc) in enumerate(zip(times, scores)):
        if sc <= thresh or (t - last) < min_shot:
            continue
        lo, hi = max(0, i - win), min(len(scores), i + win + 1)
        neigh = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        if len(neigh) and sc < max(1.7 * float(neigh.mean()), float(neigh.max()) * 1.05):
            continue
        cut_times.append(float(t))
        last = t

    bounds = [0.0] + [t for t in cut_times if min_shot < t < duration - min_shot] + [duration]
    shots = []
    for i in range(len(bounds) - 1):
        if bounds[i + 1] - bounds[i] >= min_shot * 0.8:
            shots.append(Shot(len(shots), bounds[i], bounds[i + 1]))
    return shots or [Shot(0, 0.0, duration)]
