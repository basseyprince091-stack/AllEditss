"""Media ingestion pipeline (Spec §22).

UPLOAD -> VALIDATE -> PROXY -> SHOT DETECTION -> KEYFRAMES -> VISUAL ANALYSIS
       -> QUALITY ANALYSIS -> EMBEDDING -> INDEX -> READY

Analysis is cached by content fingerprint (Principle 10): re-ingesting the same
file is free.

Deferred, with interfaces in place (see IMPLEMENTATION_LEDGER.md):
  transcription (Whisper-class), semantic vision embeddings, segmentation/tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..core.ids import content_id, file_fingerprint, new_id

# Bump whenever an analyser's OUTPUT changes shape or meaning, so cached
# analysis from an older build is recomputed rather than silently trusted.
ANALYSIS_VERSION = "2026.08-salvage"
from ..core.storage import Storage, RAW, PROXY, ANALYSIS
from ..core.errors import UnsupportedMediaError
from .probe import probe, MediaInfo
from .proxy import make_proxy, needs_proxy
from .scenes import detect_shots
from .visual import analyze_shot, VisualAnalysis
from .quality import analyze_quality, QualityAnalysis
from ..intelligence.providers.local_embedder import LocalFeatureEmbedder


@dataclass
class ShotRecord:
    """A single analyzed shot — the atomic unit ALLEDITS edits with."""
    id: str
    asset_id: str
    source_path: str        # ORIGINAL path — renders always pull from source
    proxy_path: str
    start: float
    end: float
    visual: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    embedding: list = field(default_factory=list)
    transcript: str | None = None       # deferred: needs Whisper-class model
    semantic_tags: list = field(default_factory=list)   # deferred: needs VLM

    @property
    def duration(self):
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


@dataclass
class Asset:
    id: str
    fingerprint: str
    original_path: str
    proxy_path: str
    info: dict
    shots: list = field(default_factory=list)
    kind: str = "video"

    def to_dict(self):
        d = asdict(self)
        d["shots"] = [s.to_dict() if isinstance(s, ShotRecord) else s for s in self.shots]
        return d


class Ingestor:
    def __init__(self, storage: Storage, embedder=None):
        self.storage = storage
        self.embedder = embedder or LocalFeatureEmbedder()

    def ingest(self, path: Path | str, progress=None, min_shot: float = 0.5) -> Asset:
        path = Path(path)
        p = progress or (lambda *a, **k: None)

        # VALIDATE
        info = probe(path)
        if not info.has_video:
            raise UnsupportedMediaError(f"{path.name} has no video stream")
        fp = file_fingerprint(path)
        asset_id = content_id("asset", fp)

        # CACHE — keyed by content fingerprint AND analyser version.
        # Fingerprint alone is not enough: the file is unchanged but the code
        # that measured it may not be. A new defect detector or salvage rule
        # would silently never run on any previously-ingested clip, and the
        # library would hold results no current analyser ever produced.
        cached = self.storage.get_json(ANALYSIS, f"{asset_id}.json")
        if cached and cached.get("analysis_version") != ANALYSIS_VERSION:
            p(0.02, f"{path.name}: analysers changed since this was indexed "
                    f"({cached.get('analysis_version', 'unversioned')} -> "
                    f"{ANALYSIS_VERSION}); re-analysing")
            cached = None
        if cached:
            p(1.0, f"{path.name}: cached analysis reused")
            # to_dict() emits derived fields (duration) that are not constructor
            # args; strip anything not on the dataclass before rehydrating.
            shot_fields = set(ShotRecord.__dataclass_fields__)
            asset_fields = set(Asset.__dataclass_fields__) - {"shots"}
            shots = [ShotRecord(**{k: v for k, v in s.items() if k in shot_fields})
                     for s in cached.pop("shots", [])]
            return Asset(**{k: v for k, v in cached.items() if k in asset_fields},
                         shots=shots)

        # PRESERVE SOURCE (copy in, never move)
        raw = self.storage.put_file(RAW, f"{asset_id}{path.suffix}", path)

        # PROXY
        p(0.1, f"{path.name}: proxy")
        if needs_proxy(info):
            proxy = make_proxy(raw, self.storage.path(PROXY, f"{asset_id}.mp4"), info)
        else:
            proxy = raw

        # SHOT DETECTION
        p(0.25, f"{path.name}: shot detection")
        shots = detect_shots(proxy, info.duration, min_shot=min_shot,
                             sensitivity=1.0, analysis_fps=10.0)

        # PER-SHOT ANALYSIS
        records = []
        for i, sh in enumerate(shots):
            p(0.25 + 0.7 * (i / max(len(shots), 1)),
              f"{path.name}: analyzing shot {i+1}/{len(shots)}")
            end = min(sh.end, sh.start + 4.0)     # cap analysis window for cost
            va = analyze_shot(proxy, sh.start, end)
            qa = analyze_quality(proxy, sh.start, end, info, visual=va)
            vd = va.to_dict()
            # the source aspect must travel with the shot: it is what tells the
            # builder whether a subject-aware reframe is needed at all (Spec 15)
            vd["_src_aspect"] = float(info.aspect)
            vd["_src_size"] = [info.width, info.height]
            rec = ShotRecord(
                id=f"{asset_id}_s{i:03d}", asset_id=asset_id,
                source_path=str(raw), proxy_path=str(proxy),
                start=sh.start, end=sh.end,
                visual=vd, quality=qa.to_dict(),
                embedding=[float(x) for x in self.embedder.embed_analysis(va, qa)],
            )
            records.append(rec)

        asset = Asset(id=asset_id, fingerprint=fp, original_path=str(raw),
                      proxy_path=str(proxy), info=info.to_dict(), shots=records)
        payload = asset.to_dict()
        payload["analysis_version"] = ANALYSIS_VERSION
        self.storage.put_json(ANALYSIS, f"{asset_id}.json", payload)
        p(1.0, f"{path.name}: {len(records)} shot(s) indexed")
        return asset
