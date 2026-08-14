"""STYLE: blending several references, and reusing a style without its source.

Two capabilities live here.

**Blending.** An editor rarely wants to copy one reference exactly; they want
the pacing of one and the colour of another. `blend_grammars` combines several
measured grammars by weight.

What is blended and what is not matters:
  - Continuous quantities (pacing, colour, motion, transition shares) are
    weighted means. Averaging these is meaningful.
  - Categorical qualities (rhythm, key) are NOT averaged — there is no midpoint
    between "accelerating" and "bursty". The dominant one wins by weight, and
    disagreement is recorded in `notes` rather than hidden, because a blend whose
    sources disagreed is a weaker claim than one where they agreed.
  - Intensity curves are resampled to a common length before averaging, since
    two references of different durations have curves of different lengths and
    zipping them would silently truncate the longer one.

**The library.** A grammar is a small measured description, not the reference
itself — it holds no frames, no audio, and no path back to the source work. That
is what makes it safe to store and reuse: saving a style does not save anyone's
footage. Storing the source path would defeat that, so the library refuses to.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .grammar import (StyleGrammar, PacingProfile, MotionProfile, ColorProfile,
                      TransitionProfile, GRAMMAR_VERSION)


# --------------------------------------------------------------- blending
def _wmean(values, weights) -> float:
    tot = sum(weights)
    if tot <= 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / tot)


def _dominant(labels, weights) -> tuple:
    """Highest-weighted categorical value, plus whether the sources agreed."""
    tally: dict = {}
    for lab, w in zip(labels, weights):
        tally[lab] = tally.get(lab, 0.0) + w
    best = max(tally, key=lambda k: tally[k])
    return best, len(tally) == 1


def _resample(curve, n: int) -> list:
    """Resample an intensity curve to n points by linear interpolation."""
    if not curve:
        return [0.5] * n
    vals = [float(p["value"] if isinstance(p, dict) else p[1]) for p in curve]
    if len(vals) == 1:
        return vals * n
    out = []
    for i in range(n):
        x = i * (len(vals) - 1) / max(n - 1, 1)
        lo = int(x)
        hi = min(lo + 1, len(vals) - 1)
        f = x - lo
        out.append(vals[lo] * (1 - f) + vals[hi] * f)
    return out


def blend_grammars(weighted, label: str = "blend") -> StyleGrammar:
    """Blend [(grammar, weight), ...] into one style.

    Weights are relative; they do not need to sum to 1.
    """
    if not weighted:
        raise ValueError("nothing to blend")
    pairs = [(g, float(w)) for g, w in weighted if w and float(w) > 0]
    if not pairs:
        raise ValueError("every blend weight was zero or negative")
    for g, _ in pairs:
        if g.version != GRAMMAR_VERSION:
            raise ValueError(
                f"cannot blend grammar version {g.version!r} with this build's "
                f"{GRAMMAR_VERSION!r}")
    if len(pairs) == 1:
        only = pairs[0][0]
        out = StyleGrammar.from_dict(only.to_dict())
        out.source_label = label
        out.notes = list(only.notes) + ["blend of a single source (unchanged)"]
        return out

    gs = [g for g, _ in pairs]
    ws = [w for _, w in pairs]
    total = sum(ws)
    shares = [w / total for w in ws]

    out = StyleGrammar(source_label=label, version=GRAMMAR_VERSION)
    out.duration = _wmean([g.duration for g in gs], ws)

    p = PacingProfile()
    for f in ("cuts_per_second", "mean_shot", "median_shot", "p10_shot",
              "p90_shot", "shot_duration_std"):
        setattr(p, f, _wmean([getattr(g.pacing, f) for g in gs], ws))
    p.rhythm, rhythm_agreed = _dominant([g.pacing.rhythm for g in gs], ws)
    out.pacing = p

    m = MotionProfile()
    for f in ("mean_flow", "motion_variance", "zoom_tendency", "shake_level"):
        setattr(m, f, _wmean([getattr(g.motion, f) for g in gs], ws))
    moves: dict = {}
    for g, w in pairs:
        for item in (g.motion.dominant_moves or []):
            try:
                mv, sh = item
            except (TypeError, ValueError):
                continue
            moves[mv] = moves.get(mv, 0.0) + float(sh) * w / total
    m.dominant_moves = sorted(moves.items(), key=lambda kv: -kv[1])[:5]
    out.motion = m

    c = ColorProfile()
    for f in ("brightness", "contrast", "saturation", "warmth", "colorfulness",
              "black_level", "highlight_level"):
        setattr(c, f, _wmean([getattr(g.color, f) for g in gs], ws))
    c.key, key_agreed = _dominant([g.color.key for g in gs], ws)
    c.palette = list(gs[int(max(range(len(ws)), key=lambda i: ws[i]))].color.palette)
    out.color = c

    t = TransitionProfile()
    for f in ("hard_cut_share", "flash_share", "whip_share", "dissolve_share",
              "mean_transition_duration"):
        setattr(t, f, _wmean([getattr(g.transitions, f) for g in gs], ws))
    # Shares must still describe a distribution after averaging.
    tot_share = (t.hard_cut_share + t.flash_share + t.whip_share
                 + t.dissolve_share)
    if tot_share > 0:
        t.hard_cut_share /= tot_share
        t.flash_share /= tot_share
        t.whip_share /= tot_share
        t.dissolve_share /= tot_share
    out.transitions = t

    n = max(len(g.intensity_curve) for g in gs) or 16
    curves = [_resample(g.intensity_curve, n) for g in gs]
    out.intensity_curve = [
        {"t": i / max(n - 1, 1),
         "value": _wmean([cv[i] for cv in curves], ws)} for i in range(n)]

    out.structure = list(gs[int(max(range(len(ws)), key=lambda i: ws[i]))].structure)
    for f in ("effect_density", "beat_sync_strength", "text_presence",
              "pacing_multiplier"):
        setattr(out, f, _wmean([getattr(g, f) for g in gs], ws))

    mix = ", ".join(f"{g.source_label or 'unnamed'} {s:.0%}"
                    for g, s in zip(gs, shares))
    out.notes = [f"blended from {len(gs)} references: {mix}"]
    if not rhythm_agreed:
        out.notes.append(
            f"sources disagreed on rhythm ({', '.join(sorted({g.pacing.rhythm for g in gs}))}); "
            f"took the highest-weighted, {p.rhythm}")
    if not key_agreed:
        out.notes.append(
            f"sources disagreed on key; took the highest-weighted, {c.key}")
    out.id = f"blend_{abs(hash(mix)) & 0xffffffff:08x}"
    return out


# ---------------------------------------------------------------- library
_SAFE = re.compile(r"[^a-z0-9_-]+")


def slug(name: str) -> str:
    s = _SAFE.sub("-", (name or "").strip().lower()).strip("-")
    if not s:
        raise ValueError("style name must contain at least one letter or digit")
    return s


class StyleLibrary:
    """Named, reusable styles on disk.

    A stored style is a measurement, not a copy of the reference: no frames, no
    audio, no source path. Reusing a style therefore never redistributes the
    work it was learned from.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.root / f"{slug(name)}.json"

    def save(self, name: str, grammar: StyleGrammar) -> Path:
        g = StyleGrammar.from_dict(grammar.to_dict())
        g.source_label = name
        payload = g.to_dict()
        # Belt and braces: nothing that could point at the source work.
        for leaky in ("source_path", "path", "reference_path"):
            payload.pop(leaky, None)
        payload["_saved_at"] = time.time()
        p = self.path_for(name)
        p.write_text(json.dumps(payload, indent=2))
        return p

    def load(self, name: str) -> StyleGrammar:
        p = self.path_for(name)
        if not p.exists():
            raise KeyError(f"no saved style named {name!r}; "
                           f"have: {', '.join(self.list_names()) or '(none)'}")
        d = json.loads(p.read_text())
        d.pop("_saved_at", None)
        return StyleGrammar.from_dict(d)

    def list_names(self) -> list:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def describe(self, name: str) -> str:
        g = self.load(name)
        return (f"{name}: {g.pacing.cuts_per_second:.2f} cuts/s, "
                f"mean shot {g.pacing.mean_shot:.2f}s, rhythm {g.pacing.rhythm}, "
                f"cut share {g.transitions.hard_cut_share:.0%}, "
                f"key {g.color.key}")

    def delete(self, name: str) -> bool:
        p = self.path_for(name)
        if p.exists():
            p.unlink()
            return True
        return False
