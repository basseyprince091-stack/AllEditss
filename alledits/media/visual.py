"""Visual analysis of a shot (Spec §7 Media Brain).

Everything here is measured from real pixels:
  - camera movement  : Farneback dense optical flow, decomposed into
                       translation (pan/tilt), divergence (push/pull) and
                       residual variance (handheld/shake)
  - subject motion   : flow energy after removing the global camera component
  - composition      : salient-region centroid vs rule-of-thirds, headroom
  - colour/lighting  : HSV stats, colourfulness (Hasler-Susstrunk), contrast,
                       temperature proxy, key (high/low), clipping
  - faces            : Haar cascade (ships with OpenCV, no download needed)
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict

from .frames import FrameReader

_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(p)
    return _FACE_CASCADE


def colorfulness(bgr) -> float:
    """Hasler & Susstrunk colourfulness metric, normalized to ~0..1."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    m = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    s = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    return float(min(1.0, (s + 0.3 * m) / 110.0))


def dominant_colors(bgr, k: int = 3):
    small = cv2.resize(bgr, (48, 48), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(float)
    order = np.argsort(-counts)
    return [{"bgr": [int(c) for c in centers[i]],
             "weight": float(counts[i] / counts.sum())} for i in order]


@dataclass
class VisualAnalysis:
    # motion
    camera_movement: str = "static"        # static/pan_left/pan_right/tilt_up/tilt_down/push_in/pull_out/handheld/complex
    camera_confidence: float = 0.0
    flow_magnitude: float = 0.0            # global mean flow, px/frame at 256px width
    flow_direction_deg: float = 0.0        # 0 = screen right, 90 = screen up
    zoom_rate: float = 0.0                 # >0 push in, <0 pull out
    shake: float = 0.0                     # residual flow variance
    subject_motion: float = 0.0            # motion after removing camera component
    motion_consistency: float = 0.0        # 1.0 = steady direction throughout
    flow_coherence: float = 0.0            # 1.0 = whole frame moves together (camera)
    # framing / composition
    faces: int = 0
    face_size_ratio: float = 0.0           # largest face height / frame height
    shot_size: str = "unknown"             # close_up/medium/wide/unknown
    shot_size_basis: str = "none"          # face | none — how shot_size was derived
    shot_size_confidence: float = 0.0
    subject_x: float = 0.5                 # 0..1 salient centroid
    subject_y: float = 0.5
    thirds_alignment: float = 0.0          # 1.0 = on a third line
    # colour / light
    brightness: float = 0.0                # 0..1
    contrast: float = 0.0                  # 0..1 (std of luma)
    saturation: float = 0.0                # 0..1
    colorfulness: float = 0.0
    warmth: float = 0.0                    # -1 cold .. +1 warm
    key: str = "mid"                       # low_key / mid / high_key
    clipped_highlights: float = 0.0
    clipped_shadows: float = 0.0
    dominant_colors: list = field(default_factory=list)
    # derived
    visual_energy: float = 0.0             # 0..1 composite

    def to_dict(self):
        return asdict(self)


def _consistency(dxs, dys) -> float:
    """How aligned the per-frame translation vectors are. ~1.0 for a steady
    camera move, ~0.0 for handheld wobble."""
    vecs = np.stack([np.asarray(dxs), np.asarray(dys)], 1)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    if norms.mean() <= 1e-3:
        return 0.0
    return float(np.linalg.norm((vecs / np.maximum(norms, 1e-6)).mean(0)))


def _classify_camera(dx, dy, div, jitter, mag, half_w, coherence, consistency):
    """Decompose mean flow into a named camera move.

    Three calibration points matter here:

    1. `div` is a per-frame relative scale rate. A 35% push over 3s at 30fps is
       only ~0.004/frame, so thresholds must be small; a threshold of 0.014
       could never fire on real footage. We convert divergence to the pixel
       speed it produces at the frame edge to compare it against panning fairly.

    2. `jitter` is the temporal instability of the GLOBAL translation vector —
       genuine camera shake — not spatial residual, which is dominated by
       subject movement.

    3. `coherence` is |mean flow| / mean|flow|: near 1.0 when the entire frame
       moves as one (the camera moved) and low when only parts of the frame move
       (the subject moved). Translation is only trusted as CAMERA motion in
       proportion to coherence, which stops a locked-off shot of a busy scene
       from being called handheld.
    """
    radial_px = abs(div) * half_w * 2.0
    camera_translation = mag * coherence          # translation attributable to the camera

    if camera_translation < 0.35 and jitter * coherence < 0.30 and radial_px < 0.30:
        return "static", 0.9

    if radial_px > 0.28 and radial_px > camera_translation * 0.75:
        return ("push_in" if div > 0 else "pull_out"), float(min(1.0, radial_px / 1.5))

    # Handheld is defined by jitter RELATIVE to the motion, not absolute jitter:
    # a fast pan carries proportionally large frame-to-frame variation while
    # still travelling in one steady direction. Consistency separates the two
    # cleanly (a steady pan measures ~1.0, real handheld ~0.0).
    relative_jitter = jitter / (mag + 0.3)
    if relative_jitter > 0.5 and consistency < 0.55 and coherence > 0.4:
        return "handheld", float(min(1.0, relative_jitter))

    if abs(dx) > abs(dy) * 1.4 and abs(dx) * coherence > 0.35:
        # image content flows LEFT when the camera pans RIGHT
        return ("pan_right" if dx < 0 else "pan_left"), float(min(1.0, abs(dx) / 3.0))
    if abs(dy) > abs(dx) * 1.4 and abs(dy) * coherence > 0.35:
        return ("tilt_down" if dy < 0 else "tilt_up"), float(min(1.0, abs(dy) / 3.0))
    return "complex", 0.4


def analyze_shot(path, start: float, end: float, analysis_fps: float = 8.0) -> VisualAnalysis:
    va = VisualAnalysis()
    frames, times = [], []
    with FrameReader(path, analysis_fps=analysis_fps, width=256) as fr:
        for t, f in fr.iter_frames(start=start, end=end):
            frames.append(f)
            times.append(t)
            if len(frames) >= 48:
                break
    if not frames:
        return va

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape

    # ---------- motion ----------
    dxs, dys, divs, shakes, mags, cohs = [], [], [], [], [], []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    rx, ry = (xx - cx) / max(cx, 1), (yy - cy) / max(cy, 1)
    subj = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mdx, mdy = float(fx.mean()), float(fy.mean())
        # radial component -> zoom (positive divergence = pushing in)
        denom = float((rx * rx + ry * ry).sum()) or 1.0
        div = float(((fx - mdx) * rx + (fy - mdy) * ry).sum() / denom) / max(cx, 1)
        resid = np.sqrt((fx - mdx - div * rx * cx) ** 2 + (fy - mdy - div * ry * cy) ** 2)
        cohs.append(float(np.hypot(mdx, mdy) /
                           (np.hypot(fx, fy).mean() + 1e-6)))
        dxs.append(mdx); dys.append(mdy); divs.append(div)
        shakes.append(float(resid.std()))
        mags.append(float(np.hypot(mdx, mdy)))
        subj.append(float(np.percentile(resid, 90)))

    if dxs:
        mdx, mdy = float(np.mean(dxs)), float(np.mean(dys))
        va.flow_magnitude = float(np.mean(mags))
        # screen-space direction: invert y because image y grows downward
        va.flow_direction_deg = float(np.degrees(np.arctan2(-mdy, mdx)))
        va.zoom_rate = float(np.mean(divs))
        # camera shake = how much the GLOBAL translation vector wobbles over time.
        # (The spatial residual is subject movement and belongs to subject_motion.)
        dvec = np.stack([np.asarray(dxs), np.asarray(dys)], 1)
        va.shake = float(np.mean(np.linalg.norm(dvec - dvec.mean(0), axis=1)))
        va.subject_motion = float(np.mean(subj))
        va.flow_coherence = float(np.mean(cohs)) if cohs else 0.0
        va.camera_movement, va.camera_confidence = _classify_camera(
            mdx, mdy, va.zoom_rate, va.shake, va.flow_magnitude, cx,
            va.flow_coherence, _consistency(dxs, dys))
        # consistency: how aligned per-frame flow vectors are
        va.motion_consistency = _consistency(dxs, dys)

    # ---------- colour / light (mid frame + averaged) ----------
    mid = frames[len(frames) // 2]
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray_mid = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
    va.brightness = float(gray_mid.mean() / 255.0)
    va.contrast = float(min(1.0, gray_mid.std() / 80.0))
    va.saturation = float(hsv[..., 1].mean() / 255.0)
    va.colorfulness = colorfulness(mid)
    b, g, r = cv2.split(mid.astype(np.float32))
    tot = float(r.mean() + b.mean()) or 1.0
    va.warmth = float(np.clip((r.mean() - b.mean()) / (tot / 2.0), -1, 1))
    va.clipped_highlights = float((gray_mid > 250).mean())
    va.clipped_shadows = float((gray_mid < 5).mean())
    if va.brightness < 0.32 and va.contrast > 0.35:
        va.key = "low_key"
    elif va.brightness > 0.62:
        va.key = "high_key"
    va.dominant_colors = dominant_colors(mid)

    # ---------- composition ----------
    try:
        faces = _face_cascade().detectMultiScale(
            cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY), 1.15, 5, minSize=(18, 18))
        va.faces = int(len(faces))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            va.face_size_ratio = float(fh / h)
            va.subject_x = float((fx + fw / 2) / w)
            va.subject_y = float((fy + fh / 2) / h)
            va.shot_size = ("close_up" if va.face_size_ratio > 0.28
                            else "medium" if va.face_size_ratio > 0.12 else "wide")
            va.shot_size_basis = "face"
            # Confidence follows how far the ratio sits from a class boundary;
            # a face right on 0.28 could reasonably be called either.
            edges = [0.12, 0.28]
            margin = min(abs(va.face_size_ratio - e) for e in edges)
            va.shot_size_confidence = float(min(1.0, 0.5 + margin * 4))
    except Exception:
        pass

    if va.faces == 0:
        # NO SHOT SIZE WITHOUT A FACE — deliberately.
        #
        # Three faceless estimators were built and measured against ground-truth
        # footage before this was left as "unknown":
        #   saliency bbox extent  — tracked synthetic subjects (0.47->0.54,
        #     0.22->0.23, 0.08->0.13) but on real footage returned 1.000 for
        #     scattered texture, because a sprawling merged blob has a tall
        #     bounding box and no subject in it.
        #   blob compactness      — could not separate them: a synthetic close-up
        #     scored 0.296, inside the 0.206-0.316 band of subject-free corpus
        #     frames (spectral-residual saliency fires on edges, not fills).
        #   depth-of-field ratio  — 1.37 close-up vs 1.17 wide, while corpus
        #     frames ranged 1.26-4.53. No separation.
        #
        # A wrong framing label is worse than none: it would silently mis-select
        # shots and make FIND confidently return the wrong footage. Reliable
        # faceless framing needs person/subject segmentation, which is deferred.
        # fall back to gradient saliency centroid
        sob = np.abs(cv2.Sobel(gray_mid, cv2.CV_32F, 1, 0)) + \
              np.abs(cv2.Sobel(gray_mid, cv2.CV_32F, 0, 1))
        s = sob.sum() or 1.0
        va.subject_x = float((sob.sum(0) * np.arange(w)).sum() / s / w)
        va.subject_y = float((sob.sum(1) * np.arange(h)).sum() / s / h)

    thirds = [1 / 3, 2 / 3]
    dx_t = min(abs(va.subject_x - t) for t in thirds)
    dy_t = min(abs(va.subject_y - t) for t in thirds)
    va.thirds_alignment = float(max(0.0, 1.0 - (dx_t + dy_t) / 0.33))

    # ---------- composite energy ----------
    va.visual_energy = float(np.clip(
        0.34 * min(1.0, va.flow_magnitude / 3.0) +
        0.22 * min(1.0, va.subject_motion / 4.0) +
        0.16 * min(1.0, abs(va.zoom_rate) / 0.05) +
        0.14 * va.contrast + 0.14 * va.colorfulness, 0, 1))
    return va
