"""Editing-grammar primitives -> ffmpeg filter fragments (Spec §10, §12).

This is the ONLY place where an editorial idea becomes a rendering instruction.
Filter strings are built from typed, validated parameters — never from model text.

Where a primitive is an approximation of the ideal effect, it says so here and in
the implementation ledger rather than overclaiming.
"""
from __future__ import annotations

import numpy as np

from ..timeline.schema import EffectType, TransitionType


def _f(x, nd=4):
    return f"{float(x):.{nd}f}"


def reframe(params, W, H) -> list[str]:
    """Subject-aware crop to the project aspect (Spec §15).

    Scales the source to cover the target frame, then positions the crop window
    on the detected subject instead of the geometric centre.
    """
    x = float(np.clip(params.get("x", 0.5), 0.0, 1.0))
    y = float(np.clip(params.get("y", 0.5), 0.0, 1.0))
    s = float(params.get("scale", 1.0))
    # scale to cover, then crop
    return [f"scale={int(W*s)}:{int(H*s)}:force_original_aspect_ratio=increase:flags=bicubic",
            f"crop={W}:{H}:'(iw-ow)*{_f(x)}':'(ih-oh)*{_f(y)}'"]


def push_zoom(params, W, H, nframes, fps) -> list[str]:
    z0 = float(params.get("from", 1.0))
    z1 = float(params.get("to", 1.1))
    n = max(nframes, 2)
    # supersample first so zoompan interpolates smoothly rather than stepping
    return [f"scale={W*2}:{H*2}:flags=bicubic",
            (f"zoompan=z='{_f(z0)}+({_f(z1-z0)})*on/{n}'"
             f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
             f":d=1:s={W}x{H}:fps={fps}"),
            "setsar=1"]


def shake(params, W, H) -> list[str]:
    """Handheld/impact shake: oversample then crop with a time-varying offset.
    Two incommensurate frequencies avoid an obviously periodic wobble."""
    a = float(np.clip(params.get("amplitude", 0.006), 0.0, 0.12))
    f = float(np.clip(params.get("frequency", 10.0), 0.1, 30.0))
    pad = max(4, int(min(W, H) * a * 2.2))
    ow, oh = W + pad * 2, H + pad * 2
    ax = pad * 0.9
    return [f"scale={ow}:{oh}:flags=bicubic",
            (f"crop={W}:{H}"
             f":'{pad}+{_f(ax)}*sin(2*PI*{_f(f)}*t)'"
             f":'{pad}+{_f(ax)}*sin(2*PI*{_f(f*1.37)}*t+1.1)'")]


def directional_blur(params, W, H) -> list[str]:
    """True directional blur: rotate the frame so the motion axis is horizontal,
    box-blur on that axis only, rotate back, then recover the original frame."""
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    ang = float(params.get("angle", 0.0))
    r = max(1, int(round(s * 22)))
    rad = np.radians(-ang)          # screen-space angle -> image rotation
    diag = int(np.hypot(W, H)) + 8
    diag += diag % 2
    return [f"rotate={_f(rad)}:ow={diag}:oh={diag}:c=black",
            f"boxblur={r}:1:0:0:0:0",
            f"rotate={_f(-rad)}:ow={diag}:oh={diag}:c=black",
            f"crop={W}:{H}", "setsar=1"]


def gaussian_blur(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.3), 0.0, 1.0))
    return [f"gblur=sigma={_f(s*18)}"]


def radial_blur(params, W, H) -> list[str]:
    """Approximated as a centre-weighted blend of sharp and blurred layers.
    A true rotational/zoom smear needs a per-pixel warp; this reads correctly at
    speed and is documented as an approximation."""
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    return [f"gblur=sigma={_f(s*14)}:steps=2", "vignette=PI/5:mode=backward"]


def color_grade(params, W, H) -> list[str]:
    out = []
    c = float(params.get("contrast", 1.0))
    b = float(params.get("brightness", 0.0))
    sat = float(params.get("saturation", 1.0))
    g = float(params.get("gamma", 1.0))
    if any(abs(v - d) > 1e-3 for v, d in ((c, 1.0), (b, 0.0), (sat, 1.0), (g, 1.0))):
        out.append(f"eq=contrast={_f(c)}:brightness={_f(b)}:saturation={_f(sat)}:gamma={_f(g)}")
    t = float(params.get("temperature", 0.0))
    if abs(t) > 1e-3:
        # -1 (cool) .. +1 (warm) -> 4000K..9000K around a 6500K neutral
        kelvin = int(np.clip(6500 - t * 2500, 1000, 40000))
        out.append(f"colortemperature=temperature={kelvin}:mix=0.85")
    return out


def film_grain(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.2), 0.0, 1.0))
    return [f"noise=alls={int(s*24)}:allf=t+u"]


def vignette(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.3), 0.0, 1.0))
    return [f"vignette=angle=PI/{_f(6 - s*2)}"]


def glow(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.3), 0.0, 1.0))
    return [f"unsharp=5:5:{_f(-s*1.2)}:5:5:0", f"eq=brightness={_f(s*0.03)}"]


def flash(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.5), 0.0, 1.0))
    return [f"eq=brightness={_f(s*0.35)}:contrast={_f(1.0+s*0.15)}"]


def drift(params, W, H) -> list[str]:
    dx = float(np.clip(params.get("dx", 0.0), -0.5, 0.5))
    dy = float(np.clip(params.get("dy", 0.0), -0.5, 0.5))
    pad = int(max(abs(dx), abs(dy)) * min(W, H)) + 4
    return [f"scale={W+2*pad}:{H+2*pad}:flags=bicubic",
            f"crop={W}:{H}:'{pad}+{_f(dx*pad)}*t':'{pad}+{_f(dy*pad)}*t'"]


def denoise(params, W, H) -> list[str]:
    """Temporal+spatial denoise. hqdn3d is used rather than nlmeans because it is
    an order of magnitude faster and the quality gap is small at these strengths;
    over-denoising costs more detail than the noise it removes."""
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    # Range chosen by measurement: at the top end this must actually reduce heavy
    # grain, and the earlier ceiling (6:4.5:9:6.5) was too weak to move the metric.
    ls, cs = 2.0 + 8.0 * s, 1.5 + 6.5 * s          # luma / chroma spatial
    lt, ct = 3.0 + 9.0 * s, 2.0 + 7.0 * s          # luma / chroma temporal
    return [f"hqdn3d={_f(ls,2)}:{_f(cs,2)}:{_f(lt,2)}:{_f(ct,2)}"]


def sharpen(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    return [f"unsharp=5:5:{_f(0.3 + 1.0 * s)}:5:5:0.0"]


def deblock(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    mode = "strong" if s > 0.55 else "weak"
    return [f"deblock=filter={mode}:block=8:alpha={_f(0.03 + 0.06 * s, 3)}"
            f":beta={_f(0.02 + 0.04 * s, 3)}"]


def expand_contrast(params, W, H) -> list[str]:
    s = float(np.clip(params.get("strength", 0.4), 0.0, 1.0))
    return [f"eq=contrast={_f(1.0 + 0.45 * s)}:saturation={_f(1.0 + 0.15 * s)}"]


def stabilize_transform(params, trf_path, W, H) -> list[str]:
    """Second pass of vidstab. The detect pass runs separately in the renderer —
    stabilization cannot be expressed as a single filter."""
    s = float(np.clip(params.get("strength", 0.5), 0.0, 1.0))
    zoom = float(np.clip(params.get("zoom", 0.03), 0.0, 0.2)) * 100
    return [f"vidstabtransform=input={trf_path}:smoothing={int(6 + 24 * s)}"
            f":zoom={_f(zoom,1)}:optzoom=1:interpol=bilinear:crop=black",
            "unsharp=5:5:0.3:3:3:0.0"]


BUILDERS = {
    EffectType.REFRAME.value: reframe,
    EffectType.SHAKE.value: shake,
    EffectType.DIRECTIONAL_BLUR.value: directional_blur,
    EffectType.GAUSSIAN_BLUR.value: gaussian_blur,
    EffectType.RADIAL_BLUR.value: radial_blur,
    EffectType.COLOR_GRADE.value: color_grade,
    EffectType.FILM_GRAIN.value: film_grain,
    EffectType.VIGNETTE.value: vignette,
    EffectType.GLOW.value: glow,
    EffectType.FLASH.value: flash,
    EffectType.DRIFT.value: drift,
    EffectType.DENOISE.value: denoise,
    EffectType.SHARPEN.value: sharpen,
    EffectType.DEBLOCK.value: deblock,
    EffectType.EXPAND_CONTRAST.value: expand_contrast,
}

# Transitions -> xfade transition names verified present in this ffmpeg build.
XFADE_MAP = {
    TransitionType.DISSOLVE.value: "dissolve",
    TransitionType.FLASH.value: "fadewhite",
    TransitionType.WHIP.value: "hblur",
    TransitionType.ZOOM.value: "zoomin",
    TransitionType.SLIDE.value: "slideleft",
    TransitionType.MATCH_MOVEMENT.value: "smoothleft",
}


def build_effect_chain(effects, W, H, nframes, fps) -> list[str]:
    """Ordered filter fragments for one clip. Geometry first, then motion,
    then optics, then colour, then texture — so each stage operates on a
    correctly-framed image."""
    order = {EffectType.DEBLOCK.value: -4, EffectType.DENOISE.value: -3,
             EffectType.SHARPEN.value: -2, EffectType.EXPAND_CONTRAST.value: -1,
             EffectType.STABILIZE.value: -5,
             EffectType.REFRAME.value: 0, EffectType.PUSH_ZOOM.value: 1,
             EffectType.PULL_ZOOM.value: 1, EffectType.DRIFT.value: 2,
             EffectType.SHAKE.value: 3, EffectType.DIRECTIONAL_BLUR.value: 4,
             EffectType.RADIAL_BLUR.value: 4, EffectType.GAUSSIAN_BLUR.value: 4,
             EffectType.COLOR_GRADE.value: 5, EffectType.FLASH.value: 6,
             EffectType.GLOW.value: 7, EffectType.VIGNETTE.value: 8,
             EffectType.FILM_GRAIN.value: 9}
    frags = []
    for eff in sorted(effects, key=lambda e: order.get(e.type, 5)):
        t, p = eff.type, eff.params
        if t == EffectType.STABILIZE.value:
            continue        # handled by the renderer's two-pass path
        if t in (EffectType.PUSH_ZOOM.value, EffectType.PULL_ZOOM.value):
            frags += push_zoom(p, W, H, nframes, fps)
        elif t in BUILDERS:
            frags += BUILDERS[t](p, W, H)
    return frags
