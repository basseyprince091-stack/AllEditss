"""Clip selection and ranking (Spec §6, §8).

Ranks candidates by semantic/feature match AND edit suitability — how well the
shot fits the slot's role, the music's energy, and crucially the shot that comes
before it. That last term is the differentiator: it lets ALLEDITS say

  "These three clips match. I recommend #2 because its movement and composition
   match the preceding shot."

Every score is decomposed and written to the decision ledger, so a human can see
exactly why a clip won and override it (Principles 13, 14).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

WEIGHTS = {
    "energy_fit": 0.22,
    "motion_fit": 0.16,
    "duration_fit": 0.12,
    "quality": 0.14,
    "creative": 0.12,
    "continuity": 0.18,
    "role_fit": 0.06,
}
REUSE_PENALTY = 0.34        # discourage, don't forbid, reusing a shot
SAME_ASSET_PENALTY = 0.07


@dataclass
class Candidate:
    shot: object
    score: float
    parts: dict
    explanation: str


def _continuity(prev, cand) -> tuple[float, str]:
    """Score how well `cand` follows `prev`, and say why in plain language.

    Continuity is a creative tool, not a rule (Spec §23): a deliberate contrast
    cut scores well too. What is penalized is an *accidental* near-match — the
    jump cut that reads as a mistake.
    """
    if prev is None:
        return 0.6, "first shot — nothing to match against"

    pv, cv = prev.visual, cand.visual
    notes = []

    # colour continuity
    dcol = abs(pv["brightness"] - cv["brightness"]) + abs(pv["warmth"] - cv["warmth"]) * 0.5
    col = float(np.clip(1.0 - dcol / 0.9, 0, 1))
    if col > 0.75:
        notes.append("colour and exposure carry over cleanly")
    elif col < 0.35:
        notes.append("strong tonal contrast at the cut")

    # movement relationship
    pm, cm = pv["camera_movement"], cv["camera_movement"]
    pd, cd = pv["flow_direction_deg"], cv["flow_direction_deg"]
    if pm == cm and pm != "static":
        ang = abs(((pd - cd + 180) % 360) - 180)
        if ang < 45:
            move = 1.0
            notes.append(f"movement matches ({pm}, {ang:.0f}° apart) — the cut will "
                         "feel like one continuous gesture")
        elif ang > 135:
            move = 0.72
            notes.append(f"same move reversed ({pm}) — reads as a deliberate whip-back")
        else:
            move = 0.4
            notes.append("similar movement at an awkward angle")
    elif pm == "static" and cm != "static":
        move = 0.85
        notes.append("static into movement — a natural lift in energy")
    elif pm != "static" and cm == "static":
        move = 0.7
        notes.append("movement resolving into stillness")
    elif pm == cm == "static":
        move = 0.45
        notes.append("static to static — risks feeling flat")
    else:
        move = 0.6
        notes.append(f"{pm} into {cm}")

    # framing change: avoid a near-identical framing (jump cut)
    dshot = 1.0 if pv.get("shot_size") != cv.get("shot_size") else 0.45
    dpos = abs(pv["subject_x"] - cv["subject_x"]) + abs(pv["subject_y"] - cv["subject_y"])
    if pv.get("shot_size") == cv.get("shot_size") and dpos < 0.08:
        dshot = 0.18
        notes.append("nearly identical framing — would read as a jump cut")

    score = float(np.clip(0.34 * col + 0.42 * move + 0.24 * dshot, 0, 1))
    return score, "; ".join(notes)


def _similarity(a, b) -> float:
    import numpy as np
    va, vb = np.asarray(a.embedding), np.asarray(b.embedding)
    if not va.size or not vb.size:
        return 0.0
    return float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0))


def _weights_for(cons):
    """The brief re-balances what 'a good clip for this slot' means.
    Cinematic work wants continuity; chaotic work actively wants collision."""
    w = dict(WEIGHTS)
    if cons is not None:
        w["continuity"] *= cons.continuity_weight
        w["quality"] *= cons.quality_weight
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def score_candidate(cand, slot, prev_shot, used_ids, used_assets,
                    diversity: float = 0.0, cons=None) -> Candidate:
    v, q = cand.visual, cand.quality
    w = slot.wants
    parts = {}

    # energy vs. what the slot needs
    e = v["visual_energy"]
    if e < w["min_energy"]:
        parts["energy_fit"] = float(np.clip(1.0 - (w["min_energy"] - e) * 2.2, 0, 1))
    elif e > w["max_energy"]:
        parts["energy_fit"] = float(np.clip(1.0 - (e - w["max_energy"]) * 1.6, 0, 1))
    else:
        parts["energy_fit"] = 1.0

    # motion vs. the slot's intensity target
    want_m = w["prefer_motion"]
    have_m = float(np.clip(v["flow_magnitude"] / 3.0, 0, 1))
    parts["motion_fit"] = float(1.0 - min(1.0, abs(want_m - have_m)))

    # is there enough usable material for this slot length?
    need = slot.duration
    have = cand.duration
    if have >= need:
        parts["duration_fit"] = 1.0
    else:
        parts["duration_fit"] = float(np.clip(have / max(need, 1e-6), 0, 1) ** 1.5)

    parts["quality"] = q["technical_quality"]
    parts["creative"] = q["creative_value"]
    parts["continuity"], cont_note = _continuity(prev_shot, cand)

    role_fit = 1.0
    if w.get("prefer_faces") and v["faces"] == 0:
        role_fit -= 0.35
    if w.get("prefer_move") and v["camera_movement"] != w["prefer_move"]:
        role_fit -= 0.2
    # brief-driven framing and subject preferences
    if cons is not None:
        if cons.prefer_faces > 0:
            role_fit -= cons.prefer_faces * 0.4 * (0.0 if v["faces"] else 1.0)
        elif cons.prefer_faces < 0 and v["faces"]:
            role_fit -= abs(cons.prefer_faces) * 0.3
        if cons.shot_size_preference and v.get("shot_size") not in (
                cons.shot_size_preference, "unknown"):
            role_fit -= 0.25
    parts["role_fit"] = float(max(0.0, role_fit))

    W = _weights_for(cons)
    score = sum(W[k] * parts[k] for k in W)

    # A brief asking for collision cutting inverts the continuity term: shots
    # that would cut smoothly are now the WRONG answer.
    if cons is not None and cons.continuity_weight < 0.7:
        score += W["continuity"] * (1.0 - 2.0 * parts["continuity"]) * 0.8

    # brief-only material may only fill short slots (Spec §25)
    if q["handling"] == "use_briefly":
        if not w.get("allow_brief") or need > 0.5:
            score *= 0.25
        else:
            score *= 1.05   # it's exactly what a flash frame is for
    if q["handling"] == "replace":
        score *= 0.6

    reuse_pen = min(0.85, REUSE_PENALTY * (1.0 + diversity))
    if cand.id in used_ids:
        score *= (1.0 - reuse_pen)
    if cand.asset_id in used_assets:
        score *= (1.0 - min(0.5, SAME_ASSET_PENALTY * (1.0 + diversity)))
    # a cut only reads if the next shot looks different enough from this one
    if prev_shot is not None and diversity > 0:
        sim = _similarity(prev_shot, cand)
        if sim > 0.90:
            score *= max(0.15, 1.0 - diversity * (sim - 0.90) * 8.0)

    expl = (f"energy {parts['energy_fit']:.2f}, motion {parts['motion_fit']:.2f}, "
            f"length {parts['duration_fit']:.2f}, technical {parts['quality']:.2f}, "
            f"creative {parts['creative']:.2f}, continuity {parts['continuity']:.2f} "
            f"({cont_note})")
    return Candidate(shot=cand, score=float(score), parts=parts, explanation=expl)


def select_for_slot(slot, shots, prev_shot, used_ids, used_assets, ledger=None,
                    top_n_alternatives: int = 3, diversity: float = 0.0, cons=None,
                    overrides=None):
    """Rank candidates for a slot.

    Human directives take absolute precedence over scoring: a pinned shot is
    used even if it scores badly, and a banned shot is never used even if it
    scores best. The ledger records that a human made the call, so the decision
    trail stays honest about who decided what.
    """
    if overrides is not None:
        pinned_id = overrides.pinned_at(slot.index)
        if pinned_id:
            pinned = next((s for s in shots if s.id == pinned_id), None)
            if pinned is not None:
                cand = score_candidate(pinned, slot, prev_shot, used_ids,
                                       used_assets, diversity, cons)
                if ledger:
                    ledger.record(
                        stage="clip_selection", subject=f"slot_{slot.index:02d}",
                        choice=pinned.id, actor="human_override", confidence=1.0,
                        rationale=("Pinned by the user, so no ranking was performed. "
                                   f"For reference the automatic score would have been "
                                   f"{cand.score:.3f} ({cand.explanation})."),
                        evidence={"pinned": True, "parts": cand.parts})
                return cand, []
        allowed = [s for s in shots if overrides.allowed(s, slot.index)]
        if allowed:
            shots = allowed

    # Salvaged footage is admitted BECAUSE it will be brief. Trimming its source
    # range is not enough: placed in a long slot it would still occupy that slot,
    # holding a frozen poor-quality frame for the remainder. Exclude it from
    # slots longer than its cap instead — that is what the cap means.
    slot_len = getattr(slot, "duration", None)
    if slot_len is None:
        slot_len = max(0.0, getattr(slot, "end", 0.0) - getattr(slot, "start", 0.0))
    eligible = []
    for sh in shots:
        cap = (sh.quality or {}).get("max_useful_duration")
        if cap and slot_len > float(cap) + 1e-3:
            continue
        eligible.append(sh)
    if eligible:
        shots = eligible
    # If NOTHING fits, fall through with the full list rather than failing the
    # slot: an over-held salvage clip is bad, an empty timeline is worse, and the
    # quality_handling recorded on the clip keeps it visible in the ledger.

    ranked = sorted((score_candidate(s, slot, prev_shot, used_ids, used_assets,
                                     diversity, cons)
                     for s in shots), key=lambda c: -c.score)
    if not ranked:
        return None, []
    best = ranked[0]
    alts = ranked[1:1 + top_n_alternatives]

    if ledger:
        why = (f"Chose {best.shot.id} for the {slot.role} slot "
               f"({slot.start:.2f}-{slot.end:.2f}s). {best.explanation}.")
        if alts and abs(alts[0].score - best.score) < 0.05:
            why += (f" Close call against {alts[0].shot.id} — preferred because "
                    f"{_edge(best, alts[0])}.")
        ledger.record(
            stage="clip_selection", subject=f"slot_{slot.index:02d}",
            choice=best.shot.id, rationale=why, confidence=float(min(1.0, best.score)),
            alternatives=[{"choice": a.shot.id, "score": a.score,
                           "why_not": _edge(best, a)} for a in alts],
            evidence={"parts": best.parts, "slot_wants": slot.wants})
    return best, alts


def _edge(winner: Candidate, loser: Candidate, cons=None) -> str:
    """The single largest weighted advantage the winner had."""
    W = _weights_for(cons)
    diffs = {k: W[k] * (winner.parts[k] - loser.parts[k]) for k in W}
    k = max(diffs, key=diffs.get)
    label = {"energy_fit": "its energy suits this moment better",
             "motion_fit": "its movement matches the target intensity better",
             "duration_fit": "it has enough usable length for this slot",
             "quality": "it is technically cleaner",
             "creative": "it is the stronger shot creatively",
             "continuity": "its movement and composition match the preceding shot",
             "role_fit": "it fits what this part of the edit needs"}[k]
    return f"{label} ({winner.parts[k]:.2f} vs {loser.parts[k]:.2f})"
