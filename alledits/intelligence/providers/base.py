"""Model provider abstraction (Spec §21). Models are replaceable components.

Two tiers, per Spec §20:
  REASONING tier — complex creative direction, reference interpretation, hard
                   clip selection calls, self-critique, complex revision
  ROUTINE   tier — metadata reasoning, ordinary planning, repetitive work

IMPORTANT HONESTY CONTRACT
Every response carries `actor` and `is_llm`. Nothing in ALLEDITS may present a
rule-based result as a model result. If no provider is configured, callers get
a HeuristicProvider response that is explicitly labelled as rule-based, and the
UI/ledger must show it as such (Principle 18: never fake completion).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    REASONING = "reasoning"
    ROUTINE = "routine"


@dataclass
class ModelResponse:
    content: str
    data: dict | None = None
    actor: str = "unknown"
    is_llm: bool = False
    tier: str = Tier.ROUTINE.value
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def complete(self, system: str, prompt: str, tier: Tier = Tier.ROUTINE,
                 json_schema: dict | None = None, max_tokens: int = 2000) -> ModelResponse: ...


class EmbeddingProvider(ABC):
    """Separated from LLMProvider so a vision/embedding model can be swapped
    independently (Spec §7 embeddings, §8 semantic search)."""
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def embed_frames(self, frames) -> "list": ...

    @abstractmethod
    def embed_text(self, texts: list) -> "list": ...
