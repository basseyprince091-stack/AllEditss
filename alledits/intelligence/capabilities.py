"""Model orchestration: what this install can actually do, and why not (Spec §21).

Several ALLEDITS features need a model this build does not ship: open-vocabulary
semantic search needs a vision-language model, framing detection needs person
segmentation, caption-aware editing needs transcription. The spec's rule is that
models are replaceable components, and the project's rule is that a missing
model is stated rather than faked.

This module is the seam. It does three things:

1. **Declares the interfaces** a provider must implement. Connecting a real
   model later means implementing one small ABC and registering it — no changes
   to the editing engine.
2. **Tracks what is present.** `registry.status()` reports every capability, the
   provider serving it, and for missing ones the reason and what it would unlock.
3. **Gates features honestly.** Code asks `registry.require(Capability.X)` and
   gets a provider or a `ProviderUnavailable` naming the gap. Nothing silently
   degrades into a guess.

The registry never invents a fallback. A heuristic that approximates a model is
only acceptable when it is labelled as such and measured — as FIND's attribute
search is — and that is a deliberate product decision, not something a registry
should do implicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..core.errors import ProviderUnavailable


class Capability(str, Enum):
    LLM_PLANNING = "llm_planning"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"        # CLIP/SigLIP-class
    TRANSCRIPTION = "transcription"            # Whisper-class
    PERSON_SEGMENTATION = "person_segmentation"
    OBJECT_REMOVAL = "object_removal"          # inpainting
    SUPER_RESOLUTION = "super_resolution"
    FRAME_INTERPOLATION = "frame_interpolation"


# What each capability would unlock. Shown to the user so a missing model reads
# as a concrete product gap rather than an abstract dependency.
UNLOCKS = {
    Capability.LLM_PLANNING: [
        "creative planning that reasons about an unusual brief",
        "briefs outside the ~130-term lexicon"],
    Capability.TEXT_EMBEDDING: ["fuzzy matching of style and brief language"],
    Capability.IMAGE_EMBEDDING: [
        "open-vocabulary search ('the shot where she looks relieved')",
        "finding footage by description rather than measured attributes"],
    Capability.TRANSCRIPTION: [
        "cutting to spoken words", "caption generation",
        "finding a moment by what was said"],
    Capability.PERSON_SEGMENTATION: [
        "shot-size detection on footage without detectable faces",
        "subject-aware reframing between aspect ratios (Spec 15)",
        "subject tracking and masking (Spec 16)"],
    Capability.OBJECT_REMOVAL: ["removing distractions from a plate (Spec 18)"],
    Capability.SUPER_RESOLUTION: [
        "delivering a profile larger than the source without interpolation"],
    Capability.FRAME_INTERPOLATION: ["smooth slow motion beyond source frame rate"],
}


# ------------------------------------------------------------------ interfaces
class Provider(ABC):
    """Common contract. `available()` must be cheap and must not raise."""
    capability: Capability = None
    name: str = "unnamed"

    @abstractmethod
    def available(self) -> bool: ...

    def unavailable_reason(self) -> str:
        return "not configured"


class ImageEmbeddingProvider(Provider):
    """CLIP/SigLIP-class. Embeds frames and text into ONE shared space."""
    capability = Capability.IMAGE_EMBEDDING

    @abstractmethod
    def embed_images(self, images) -> list: ...

    @abstractmethod
    def embed_text(self, texts: list) -> list: ...


class TranscriptionProvider(Provider):
    """Whisper-class. Returns [{start, end, text}] in seconds."""
    capability = Capability.TRANSCRIPTION

    @abstractmethod
    def transcribe(self, audio_path, language: str | None = None) -> list: ...


class SegmentationProvider(Provider):
    """Person/subject masks. Returns per-frame masks or bounding boxes."""
    capability = Capability.PERSON_SEGMENTATION

    @abstractmethod
    def segment(self, images) -> list: ...


class InpaintingProvider(Provider):
    capability = Capability.OBJECT_REMOVAL

    @abstractmethod
    def inpaint(self, images, masks) -> list: ...


class UpscaleProvider(Provider):
    capability = Capability.SUPER_RESOLUTION

    @abstractmethod
    def upscale(self, images, factor: float) -> list: ...


@dataclass
class CapabilityStatus:
    capability: str
    available: bool
    provider: str = ""
    reason: str = ""
    unlocks: list = field(default_factory=list)

    def to_dict(self):
        return {"capability": self.capability, "available": self.available,
                "provider": self.provider, "reason": self.reason,
                "unlocks": self.unlocks}


class CapabilityRegistry:
    """Which model serves which capability, and what is missing."""

    def __init__(self):
        self._providers: dict = {}

    def register(self, provider: Provider, capability: Capability | None = None):
        cap = capability or getattr(provider, "capability", None)
        if cap is None:
            raise ValueError(f"{provider!r} declares no capability")
        self._providers[Capability(cap)] = provider
        return self

    def unregister(self, capability: Capability):
        self._providers.pop(Capability(capability), None)

    def get(self, capability: Capability):
        """The provider for a capability, or None. Never raises."""
        p = self._providers.get(Capability(capability))
        try:
            return p if (p is not None and p.available()) else None
        except Exception:
            return None      # a broken provider is an absent one

    def has(self, capability: Capability) -> bool:
        return self.get(capability) is not None

    def require(self, capability: Capability, feature: str = ""):
        """Return the provider or refuse, naming the gap and what it blocks."""
        p = self.get(capability)
        if p is not None:
            return p
        cap = Capability(capability)
        registered = self._providers.get(cap)
        why = (registered.unavailable_reason() if registered is not None
               else "no provider registered")
        unlocks = "; ".join(UNLOCKS.get(cap, []))
        raise ProviderUnavailable(
            f"{feature or cap.value} needs a {cap.value} model ({why}). "
            f"Connect one to enable: {unlocks}."
            if unlocks else
            f"{feature or cap.value} needs a {cap.value} model ({why}).")

    def status(self) -> list:
        out = []
        for cap in Capability:
            p = self._providers.get(cap)
            live = self.get(cap)
            out.append(CapabilityStatus(
                capability=cap.value,
                available=live is not None,
                provider=getattr(live or p, "name", "") if (live or p) else "",
                reason=("" if live is not None else
                        (p.unavailable_reason() if p is not None
                         else "no provider registered")),
                unlocks=list(UNLOCKS.get(cap, []))))
        return out

    def summary(self) -> str:
        rows = self.status()
        on = [s for s in rows if s.available]
        return (f"{len(on)}/{len(rows)} capabilities available"
                + (f" ({', '.join(s.capability for s in on)})" if on else ""))


# Process-wide default. Deliberately EMPTY: this build ships no vision, audio or
# language model, and pre-registering a stub would make the registry lie.
registry = CapabilityRegistry()


def default_registry() -> CapabilityRegistry:
    """Register whatever this environment actually provides.

    Called at startup. Each provider decides for itself whether it is usable, so
    an install with an API key or a local model lights up without code changes.
    """
    from .providers.anthropic_provider import AnthropicProvider
    try:
        llm = AnthropicProvider()
        if llm.available():
            registry.register(llm, Capability.LLM_PLANNING)
    except Exception:
        pass
    try:
        from .providers.local_embedder import LocalEmbedder
        emb = LocalEmbedder()
        if getattr(emb, "available", lambda: False)():
            registry.register(emb, Capability.TEXT_EMBEDDING)
    except Exception:
        pass
    return registry
