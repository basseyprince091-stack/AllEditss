"""Sound design: deciding the mix.

Same architectural rule as picture — this module DECIDES and emits a typed plan.
It never builds a command line. The renderer turns the plan into a filter graph,
and the validator can reject a plan before a single sample is processed.

Three things make an edit sound professional rather than amateur, and none of
them are "add music":

1. **Loudness normalisation to a target.** An edit that measures -24 LUFS is
   quiet on every platform; one at -8 LUFS gets turned down by the platform and
   comes back lifeless. Both are fixed by measuring, not guessing.
2. **Ducking.** Music under speech must move out of the way *automatically and
   in time*, which is a sidechain problem, not a keyframing problem.
3. **Headroom.** True peak must stay below 0 dBTP or lossy encoders clip on
   playback even when the file itself looks fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class TrackRole(str, Enum):
    MUSIC = "music"
    VOICE = "voice"          # speech, whether diegetic or narration
    DIEGETIC = "diegetic"    # the clip's own location sound
    SFX = "sfx"
    AMBIENCE = "ambience"


# Platform integrated-loudness targets, in LUFS. These are the levels the
# platforms normalise TO, so delivering at the target means playback matches the
# intent instead of being adjusted after the fact.
LOUDNESS_TARGETS = {
    "social": -14.0,      # YouTube / Instagram / TikTok cluster
    "broadcast": -23.0,   # EBU R128
    "streaming": -16.0,   # Apple Podcasts / spoken word
    "cinema": -27.0,
}

TRUE_PEAK_CEILING = -1.0     # dBTP. Below 0 so lossy encoding cannot clip.


@dataclass
class MixTrack:
    """One element of the mix."""
    id: str
    source_path: str
    role: str = TrackRole.MUSIC.value
    source_in: float = 0.0
    source_out: float = 0.0
    timeline_start: float = 0.0
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    ducked_by: list = field(default_factory=list)   # ids of tracks that duck this
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.source_out - self.source_in)

    def to_dict(self):
        return asdict(self)


# Measured, not modelled. Sweeping ffmpeg's sidechaincompress showed that the
# achieved attenuation is governed almost entirely by THRESHOLD, not ratio
# (ratio 2->20 moved it under 1.5 dB, while threshold 0.045->0.01 moved it 4 dB),
# and that it saturates just under 7 dB. An earlier analytical ratio model
# predicted 9-18 dB and was wrong on both counts.
#   (threshold, achieved_dB) at ratio 8:
DUCK_CALIBRATION = [(0.045, 2.6), (0.010, 6.5), (0.003, 6.9)]
MAX_ACHIEVABLE_DUCK_DB = 6.9
DUCK_RATIO = 8.0        # at/near saturation; higher buys nothing measurable


@dataclass
class DuckSpec:
    """How hard, how fast, and how far the ducked track moves.

    Attack must be fast enough that the first syllable is not buried, and
    release slow enough that the music does not pump between words — the two
    failure modes that make automatic ducking sound automatic.
    """
    depth_db: float = 6.0
    attack_ms: float = 20.0
    release_ms: float = 350.0

    @property
    def ratio(self) -> float:
        return DUCK_RATIO

    @property
    def threshold(self) -> float:
        """Threshold calibrated to produce the requested depth.

        Interpolates the measured table in log-threshold space. Previously a
        fixed threshold and ratio were used while `depth_db` was reported to the
        user, so the stated depth had no connection to what was applied.
        """
        import math
        d = max(0.0, min(self.depth_db, MAX_ACHIEVABLE_DUCK_DB))
        pts = DUCK_CALIBRATION
        if d <= pts[0][1]:
            # shallower than the calibrated range: scale threshold upward
            return min(0.2, pts[0][0] * (pts[0][1] / max(d, 0.5)))
        for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
            if d <= d1:
                f = (d - d0) / (d1 - d0)
                lg = math.log10(t0) + f * (math.log10(t1) - math.log10(t0))
                return round(10 ** lg, 5)
        return pts[-1][0]

    @property
    def achievable_depth_db(self) -> float:
        """What this spec can actually deliver, which may be less than asked."""
        return min(self.depth_db, MAX_ACHIEVABLE_DUCK_DB)

    def to_dict(self):
        d = asdict(self)
        d.update(ratio=self.ratio, threshold=self.threshold,
                 achievable_depth_db=self.achievable_depth_db)
        return d


@dataclass
class MixPlan:
    tracks: list = field(default_factory=list)
    target_lufs: float = LOUDNESS_TARGETS["social"]
    true_peak_db: float = TRUE_PEAK_CEILING
    target_lra: float = 11.0     # loudness range; see note in plan_mix
    duck: DuckSpec = field(default_factory=DuckSpec)
    normalize: bool = True
    reasons: list = field(default_factory=list)
    # Filled in by the renderer once the mix has actually been measured, so the
    # plan records what was DELIVERED and not merely what was intended.
    achieved_lufs: float | None = None
    achieved_tp: float | None = None
    applied_gain_db: float | None = None

    def by_role(self, role) -> list:
        r = role.value if isinstance(role, TrackRole) else role
        return [t for t in self.tracks if t.role == r]

    @property
    def has_voice(self) -> bool:
        return bool(self.by_role(TrackRole.VOICE))

    def to_dict(self):
        d = asdict(self)
        d["duck"] = self.duck.to_dict()
        return d


def diegetic_voice_tracks(timeline, min_overlap: float = 0.25) -> list:
    """Voice tracks taken from the clips' OWN audio.

    A clip only contributes if speech is actually detected inside the portion of
    the source that made it onto the timeline. Adding every clip that merely has
    an audio stream would duck the music under room tone.

    Detection is cached per source file: one clip file commonly supplies several
    shots, and analysing it repeatedly is pure waste.
    """
    from .speech import detect_speech
    cache, out = {}, []
    for clip in sorted(timeline.clips, key=lambda c: c.timeline_start):
        src = str(clip.source_path)
        if src not in cache:
            cache[src] = detect_speech(src)
        sp = cache[src]
        if not sp.has_speech:
            continue
        # How much of THIS clip's source range is speech?
        overlap = sum(max(0.0, min(b, clip.source_out) - max(a, clip.source_in))
                      for a, b in sp.windows)
        if overlap < min_overlap:
            continue
        out.append({
            "path": src,
            "source_in": clip.source_in,
            "source_out": clip.source_out,
            "timeline_start": clip.timeline_start,
            "reason": (f"{overlap:.1f}s of detected speech in this clip's own "
                       f"audio ({sp.reason})"),
        })
    return out


def plan_mix(timeline, music_path: str | None, target: str = "social",
             voice_tracks: list | None = None, cons=None, ledger=None) -> MixPlan:
    """Decide the mix for a finished timeline.

    Ducking is only planned when there is something to duck FOR. Applying a
    sidechain with no voice present costs nothing audible but misrepresents what
    the system did, and a plan that claims a duck it never performs is worse
    than no plan.
    """
    plan = MixPlan(target_lufs=LOUDNESS_TARGETS.get(target, LOUDNESS_TARGETS["social"]))

    # --- voice / diegetic first: it is the reason the mix exists ---
    for i, v in enumerate(voice_tracks or []):
        plan.tracks.append(MixTrack(
            id=f"vox{i:02d}", source_path=str(v["path"]),
            role=TrackRole.VOICE.value,
            source_in=float(v.get("source_in", 0.0)),
            source_out=float(v.get("source_out", timeline.duration)),
            timeline_start=float(v.get("timeline_start", 0.0)),
            gain_db=float(v.get("gain_db", 0.0)),
            reason=v.get("reason") or
            "speech carries the message; the mix is built around it"))

    # --- music ---
    if music_path:
        # Music sits below speech by default. -10 dB is the conventional bed
        # level; the sidechain then takes it further down only while voice is
        # actually present, so the music is full-level between phrases.
        music_gain = -10.0 if plan.has_voice else 0.0
        if cons is not None:
            music_gain += float(getattr(cons, "music_gain_db", 0.0) or 0.0)
        fade_out = min(1.2, max(0.4, timeline.duration * 0.06))
        plan.tracks.append(MixTrack(
            id="music", source_path=str(music_path), role=TrackRole.MUSIC.value,
            source_in=0.0, source_out=timeline.duration, timeline_start=0.0,
            gain_db=music_gain, fade_in=0.0, fade_out=fade_out,
            ducked_by=[t.id for t in plan.by_role(TrackRole.VOICE)],
            reason=("bed under speech, ducked while voice is present"
                    if plan.has_voice else "primary audio bed")))

    if cons is not None and getattr(cons, "loudness_target", None):
        plan.target_lufs = LOUDNESS_TARGETS.get(cons.loudness_target, plan.target_lufs)

    # --- ducking depth follows the brief ---
    if plan.has_voice:
        depth = 9.0
        if cons is not None:
            # A restrained, dialogue-led piece wants the music further back; an
            # aggressive one keeps it forward and lets the voice fight for space.
            depth += 3.0 * float(getattr(cons, "continuity_weight", 1.0) > 1.5)
            depth -= 2.0 * float(getattr(cons, "pacing_multiplier", 1.0) < 0.8)
        plan.duck = DuckSpec(depth_db=max(3.0, min(18.0, depth)))
        got = plan.duck.achievable_depth_db
        note = ("" if got >= plan.duck.depth_db - 0.1 else
                f" (requested {plan.duck.depth_db:.0f} dB; this sidechain "
                f"saturates at {MAX_ACHIEVABLE_DUCK_DB:.1f} dB)")
        plan.reasons.append(
            f"ducking music {got:.1f} dB under speech{note} "
            f"(attack {plan.duck.attack_ms:.0f} ms so the first syllable is not "
            f"buried, release {plan.duck.release_ms:.0f} ms so it does not pump)")
    plan.reasons.append(
        f"normalising to {plan.target_lufs:.0f} LUFS with a "
        f"{plan.true_peak_db:.0f} dBTP ceiling")

    if ledger:
        ledger.record(stage="sound_design", subject="mix",
                      choice=f"{len(plan.tracks)} track(s) @ {plan.target_lufs:.0f} LUFS",
                      rationale="; ".join(plan.reasons),
                      confidence=0.85, actor="rule_based_mixer")
    return plan


def validate_mix(plan: MixPlan) -> list:
    """Reject a plan that cannot produce a deliverable mix."""
    issues = []
    if not plan.tracks:
        issues.append("mix has no tracks")
    ids = [t.id for t in plan.tracks]
    if len(ids) != len(set(ids)):
        issues.append("duplicate track ids")
    for t in plan.tracks:
        if t.source_out <= t.source_in:
            issues.append(f"{t.id}: empty source range")
        if not -60.0 <= t.gain_db <= 24.0:
            issues.append(f"{t.id}: gain {t.gain_db} dB out of range")
        for d in t.ducked_by:
            if d not in ids:
                issues.append(f"{t.id}: ducked by unknown track {d}")
            if d == t.id:
                issues.append(f"{t.id}: cannot duck itself")
    if not 1.0 <= plan.target_lra <= 20.0:
        issues.append(f"implausible loudness range target {plan.target_lra}")
    if not -40.0 <= plan.target_lufs <= -5.0:
        issues.append(f"implausible loudness target {plan.target_lufs}")
    if plan.true_peak_db > 0.0:
        issues.append("true peak ceiling must be below 0 dBTP")
    if not 0.0 < plan.duck.depth_db <= 24.0:
        issues.append(f"duck depth {plan.duck.depth_db} dB out of range")
    return issues
