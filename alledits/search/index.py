"""Media index + vector search (Spec §7, §8).

Backend is brute-force numpy cosine over the in-memory matrix. For an MVP
library (thousands of shots) this is exact and instant. The interface matches
what pgvector/Qdrant/LanceDB need, so scaling is a backend swap.

CAPABILITY HONESTY: search_by_text() raises unless a semantic embedding
provider is installed. ALLEDITS does not fake a semantic match by keyword-
guessing over feature vectors.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import ProviderUnavailable


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a.dot(b) / n)


class MediaIndex:
    def __init__(self, embedder=None):
        self.shots = []          # list[ShotRecord]
        self._matrix = None
        self.embedder = embedder

    def add_asset(self, asset):
        for s in asset.shots:
            self.shots.append(s)
        self._matrix = None

    def _mat(self):
        if self._matrix is None:
            if not self.shots:
                self._matrix = np.zeros((0, 1), np.float32)
            else:
                self._matrix = np.vstack([np.asarray(s.embedding, np.float32)
                                          for s in self.shots])
        return self._matrix

    def search_by_vector(self, vec, top_k=10, exclude=()):
        M = self._mat()
        if M.shape[0] == 0:
            return []
        v = np.asarray(vec, np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        sims = M @ v
        order = np.argsort(-sims)
        out = []
        for i in order:
            s = self.shots[int(i)]
            if s.id in exclude:
                continue
            out.append((s, float(sims[int(i)])))
            if len(out) >= top_k:
                break
        return out

    def find_similar(self, shot, top_k=10):
        return self.search_by_vector(shot.embedding, top_k=top_k, exclude={shot.id})

    def filter(self, predicate):
        return [s for s in self.shots if predicate(s)]

    def search_by_text(self, query: str, top_k=10, registry=None):
        """Open-vocabulary search. Needs a vision-language model.

        Routed through the capability registry so that connecting a CLIP/SigLIP
        provider enables this with no change here: the shot embeddings and the
        query embedding must come from the SAME model, which is exactly what the
        registry guarantees by handing back one provider for both calls.
        """
        from ..intelligence.capabilities import Capability, registry as default_reg
        reg = registry or default_reg
        provider = reg.get(Capability.IMAGE_EMBEDDING)
        if provider is None:
            # Refuse with the concrete gap, and point at what DOES work.
            reg.require(Capability.IMAGE_EMBEDDING,
                        feature="Open-vocabulary semantic search")
        vec = provider.embed_text([query])[0]
        hits = []
        for shot in self.usable_shots():
            emb = getattr(shot, "semantic_embedding", None)
            if emb is None:
                continue
            hits.append((_cosine(vec, emb), shot))
        if not hits:
            raise ProviderUnavailable(
                f"{provider.name} is connected, but no shot in this library has "
                "been embedded yet — re-ingest to build semantic vectors.")
        hits.sort(key=lambda t: -t[0])
        return [{"shot": sh, "score": float(sc), "semantic": True,
                 "matched": [f"semantically similar to {query!r}"], "unmet": []}
                for sc, sh in hits[:top_k]]

    # ------------------------------------------------------------------ FIND
    def _attr(self, shot, name):
        """Read one measured attribute, wherever the analysers put it."""
        if name == "duration":
            return shot.duration
        if name == "speech":
            return self._has_speech(shot)
        if name in shot.visual:
            return shot.visual.get(name)
        if name in shot.quality:
            return shot.quality.get(name)
        return None

    def _has_speech(self, shot):
        """Speech inside THIS shot's range, not merely somewhere in the file."""
        key = str(shot.source_path)
        if not hasattr(self, "_speech_cache"):
            self._speech_cache = {}
        if key not in self._speech_cache:
            try:
                from ..audio.speech import detect_speech
                sp = detect_speech(key)
                self._speech_cache[key] = sp.windows if sp.has_speech else []
            except Exception:
                self._speech_cache[key] = []
        return any(min(b, shot.end) - max(a, shot.start) > 0.25
                   for a, b in self._speech_cache[key])

    @staticmethod
    def _passes(value, crit):
        if value is None:
            return False
        op = crit.op
        if op == "flag":
            return bool(value) if not isinstance(value, (int, float)) else value > 0
        if op == "==":
            return str(value) == str(crit.value)
        if op == "prefix":
            return str(value).startswith(str(crit.value))
        if op == ">":
            return float(value) > float(crit.value)
        if op == "<":
            return float(value) < float(crit.value)
        return False

    def search(self, query, top_k=None):
        """Run a StructuredQuery. Returns [{shot, score, reasons}], best first.

        Scoring is the fraction of criteria satisfied, so a partial match still
        surfaces rather than being silently dropped — with the unmet criteria
        named, which is the useful part.
        """
        from .query import parse_query
        if isinstance(query, str):
            query = parse_query(query)
        pool = self.usable_shots()

        # A criterion whose attribute was never measured on this library can
        # never match. Silently returning zero results for "wide shots" when
        # shot size was never populated looks like "no wide shots exist" — a
        # different and misleading claim. Report it as inert instead.
        inert = []
        for c in query.criteria:
            if c.op == "flag":
                continue
            vals = [self._attr(sh, c.attribute) for sh in pool]
            if all(v is None or v == "unknown" for v in vals):
                inert.append(c)
        effective = [c for c in query.criteria if c not in inert]
        self.last_inert_criteria = [c.describe() for c in inert]

        results = []
        for shot in pool:
            met, missed = [], []
            for c in effective:
                ok = self._passes(self._attr(shot, c.attribute), c)
                if c.negated:
                    ok = not ok
                (met if ok else missed).append(c.describe())
            score = 1.0 if not effective else len(met) / len(effective)
            if effective and not met:
                continue                # nothing asked for was satisfied
            results.append({"shot": shot, "score": score,
                            "matched": met, "unmet": missed})

        if query.sort_by:
            results.sort(key=lambda r: (
                r["score"],
                float(self._attr(r["shot"], query.sort_by) or 0.0)
                * (1 if query.sort_desc else -1)), reverse=True)
        else:
            results.sort(key=lambda r: (r["score"], r["shot"].quality.get(
                "technical_quality", 0.0)), reverse=True)
        return results[:(top_k or query.limit)]

    def usable_shots(self, include_brief=True):
        out = []
        for s in self.shots:
            h = s.quality.get("handling")
            if h == "reject":
                continue
            if h == "use_briefly" and not include_brief:
                continue
            out.append(s)
        return out

    def stats(self):
        from collections import Counter
        return {
            "shots": len(self.shots),
            "handling": dict(Counter(s.quality.get("handling") for s in self.shots)),
            "camera_moves": dict(Counter(s.visual.get("camera_movement") for s in self.shots)),
            "total_duration": sum(s.duration for s in self.shots),
        }
