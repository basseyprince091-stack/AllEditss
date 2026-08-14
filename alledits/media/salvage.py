""""Nothing is wasted": finding a use for imperfect footage (Spec §25).

The rule is not "accept everything". It is: do not reject on technical score
alone, because a clip that fails as a hero shot can still be the right two
tenths of a second.

So a clip that scores badly is asked a second question — *what could this still
do?* — and the answer is a set of SALVAGE ROLES, each with a duration cap and a
reason drawn from what was measured. A clip salvaged as a flash frame is allowed
on screen for 0.2s and not one frame longer; the cap is the whole point, since
the same footage held for two seconds is exactly the failure the quality gate
was protecting against.

Genuinely unusable footage is still rejected. Black frames, frozen frames and
clips with no measurable content serve no creative function, and pretending
otherwise would make the principle meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class SalvageRole(str, Enum):
    FLASH_FRAME = "flash_frame"        # single-beat punctuation
    TRANSITION = "transition"          # motion-carrying wipe between shots
    RAPID_MONTAGE = "rapid_montage"    # one of many fast cuts
    TEXTURE = "texture"                # grain/atmosphere over another image
    BACKGROUND = "background"          # soft plate behind a subject or text


# Longest a clip may be held in each role. These are the contract: a salvaged
# clip is admitted BECAUSE it will be brief.
ROLE_MAX_DURATION = {
    SalvageRole.FLASH_FRAME: 0.20,
    SalvageRole.TRANSITION: 0.35,
    # One beat at 120 BPM. Cuts are quantised to the beat grid, so a cap below
    # one beat makes the role unusable at any normal tempo — at 129 BPM a beat
    # is 0.465s, and an arbitrary 0.45s cap silently excluded salvage footage
    # from every slot in the edit. "Rapid montage" means roughly two cuts a
    # second, which IS one beat at dance tempo, so the cap is now that.
    SalvageRole.RAPID_MONTAGE: 0.50,
    SalvageRole.TEXTURE: 1.20,
    SalvageRole.BACKGROUND: 2.50,
}


# TEXTURE and BACKGROUND require layered composition — placing this clip UNDER
# another image. The renderer composites no layers, so those roles are named
# (they are real editorial uses, and become available the moment compositing
# lands) but marked unrealisable, and they never widen the duration cap. A cap
# derived from a role the renderer cannot perform would let a two-tenths clip be
# held for 2.5 seconds as a "background" it is never actually placed behind.
REALISABLE_ROLES = {SalvageRole.FLASH_FRAME, SalvageRole.TRANSITION,
                    SalvageRole.RAPID_MONTAGE}
ROLE_REQUIRES = {
    SalvageRole.TEXTURE: "layered composition (not implemented)",
    SalvageRole.BACKGROUND: "layered composition (not implemented)",
}


@dataclass
class Salvage:
    role: str
    max_duration: float
    reason: str
    realisable: bool = True
    requires: str = ""

    def to_dict(self):
        return asdict(self)


def is_genuinely_unusable(q, visual) -> tuple:
    """Footage that serves no creative function at all.

    Deliberately narrow: near-black, blown out, or frozen. A merely ugly clip is
    not unusable, which is the whole point of this module.
    """
    b = getattr(q, "brightness", None)
    if b is None:
        b = (getattr(visual, "brightness", 0.5) if visual else 0.5)
    if b < 0.02:
        return True, "essentially black"
    if b > 0.98:
        return True, "completely blown out"
    if visual is not None:
        motion = getattr(visual, "mean_flow", None)
        energy = getattr(visual, "visual_energy", None)
        sharp = getattr(q, "sharpness", 1.0)
        if (motion is not None and motion < 0.01 and energy is not None
                and energy < 0.02 and sharp < 5.0):
            return True, "frozen and featureless"
    return False, ""


def assess_salvage(q, visual) -> list:
    """What creative functions could this clip still serve?

    Returns [] for footage good enough to use normally — salvage roles are a
    concession, and offering them for a clip that does not need them would
    invite the editor to waste good footage on a flash frame.
    """
    handling = getattr(q, "handling", "use")
    if handling in ("use", "enhance"):
        return []

    unusable, why = is_genuinely_unusable(q, visual)
    if unusable:
        return []

    out = []

    def add(role, reason):
        out.append(Salvage(role=role.value,
                           max_duration=ROLE_MAX_DURATION[role], reason=reason,
                           realisable=role in REALISABLE_ROLES,
                           requires=ROLE_REQUIRES.get(role, "")))

    tech = float(getattr(q, "technical_quality", 0.0))
    sharp = float(getattr(q, "sharpness", 0.0))
    noise = float(getattr(q, "noise", 0.0))
    motion = float(getattr(visual, "mean_flow", 0.0) or 0.0) if visual else 0.0
    shake = float(getattr(visual, "shake", 0.0) or 0.0) if visual else 0.0
    energy = float(getattr(visual, "visual_energy", 0.0) or 0.0) if visual else 0.0
    contrast = float(getattr(visual, "contrast", 0.0) or 0.0) if visual else 0.0

    # A flash frame is on screen for ~6 frames. Almost nothing survives long
    # enough to be judged, so the only requirement is that something is visible.
    if contrast > 0.05 or energy > 0.05:
        add(SalvageRole.FLASH_FRAME,
            f"technical quality {tech:.2f} is too low to hold, but at "
            f"{ROLE_MAX_DURATION[SalvageRole.FLASH_FRAME]:.2f}s no one reads detail")

    # Motion hides softness: a blurred frame moving fast reads as energy.
    if motion > 1.5 or shake > 1.0:
        add(SalvageRole.TRANSITION,
            f"movement ({motion:.1f}) carries the eye between shots and masks "
            f"the softness that makes this unusable as a held shot")

    if energy > 0.15 or motion > 0.8:
        add(SalvageRole.RAPID_MONTAGE,
            f"visual energy {energy:.2f} works inside a fast run of cuts where "
            "no single frame is examined")

    # Grain is a defect in a hero shot and a texture over one.
    if noise > 0.15:
        add(SalvageRole.TEXTURE,
            f"the grain that disqualifies this ({noise:.2f}) is usable as "
            "atmosphere over another image")

    # Softness is an asset behind a subject or titles.
    if sharp < 30.0 and shake < 1.5:
        add(SalvageRole.BACKGROUND,
            f"soft ({sharp:.0f}) but steady, so it works as a plate behind a "
            "subject or text rather than as the subject itself")

    return out


def salvage_cap(salvage_list) -> float | None:
    """Longest this clip may be held in a role the renderer can ACTUALLY perform.

    Only realisable roles count. Taking the maximum across all roles let a clip
    salvaged for a 0.2s flash inherit the 2.5s cap of a background plate that
    the renderer never composites — which is precisely the failure the quality
    gate exists to prevent.
    """
    caps = [float(s["max_duration"] if isinstance(s, dict) else s.max_duration)
            for s in (salvage_list or [])
            if (s.get("realisable", True) if isinstance(s, dict)
                else s.realisable)]
    return max(caps) if caps else None


def roles_of(salvage_list) -> set:
    return {(s["role"] if isinstance(s, dict) else s.role) for s in (salvage_list or [])}
