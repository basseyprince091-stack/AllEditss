"""Local feature embedder — real, but NOT semantic.

Produces a normalized descriptor per shot from measured colour, motion,
composition and quality features. This genuinely powers "find shots that look
and move like this one" and edit-fit ranking.

It CANNOT do open-vocabulary semantic search ("someone realizing they've been
betrayed") — that needs a vision-language model (CLIP/SigLIP class). The
EmbeddingProvider interface exists precisely so that model drops in later
without changing the index or the search API. Capability is declared honestly
via `semantic = False` so the UI never offers a search it cannot perform.
"""
from __future__ import annotations

import numpy as np

from .base import EmbeddingProvider


class LocalFeatureEmbedder(EmbeddingProvider):
    name = "local_feature_embedder"
    dim = 32
    semantic = False        # declares the capability boundary

    def available(self) -> bool:
        return True

    def embed_analysis(self, visual, quality) -> np.ndarray:
        """Build a descriptor from an analyzed shot."""
        v, q = visual, quality
        move_onehot = [0.0] * 9
        moves = ["static", "pan_left", "pan_right", "tilt_up", "tilt_down",
                 "push_in", "pull_out", "handheld", "complex"]
        if v.camera_movement in moves:
            move_onehot[moves.index(v.camera_movement)] = 1.0
        pal = []
        for c in (v.dominant_colors or [])[:3]:
            b, g, r = c["bgr"]
            pal += [r / 255.0 * c["weight"], g / 255.0 * c["weight"], b / 255.0 * c["weight"]]
        pal += [0.0] * (9 - len(pal))
        vec = np.array([
            v.brightness, v.contrast, v.saturation, v.colorfulness,
            (v.warmth + 1) / 2, v.thirds_alignment, v.subject_x, v.subject_y,
            min(1.0, v.flow_magnitude / 4.0), min(1.0, v.subject_motion / 5.0),
            (np.clip(v.zoom_rate, -0.05, 0.05) + 0.05) / 0.1,
            min(1.0, v.shake / 4.0), v.motion_consistency, v.visual_energy,
            *move_onehot, *pal,
        ], dtype=np.float32)
        vec = np.nan_to_num(vec)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def embed_frames(self, frames):
        raise NotImplementedError("LocalFeatureEmbedder embeds analyses, not raw frames")

    def embed_text(self, texts):
        raise NotImplementedError(
            "Text->vector search requires a vision-language embedding provider. "
            "Not available in this build; see IMPLEMENTATION_LEDGER.md.")
