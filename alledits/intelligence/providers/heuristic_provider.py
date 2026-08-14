"""Rule-based planner. NOT a language model and never presented as one.

is_llm=False on every response, and actor is "rule_based_planner". The decision
ledger and any UI must show this so nothing fake is ever attributed to AI.

Its job is to keep ALLEDITS fully functional with no network or API key: the
editorial decisions it makes are computed from real measured signal (music
analysis, shot analysis, style grammar), not invented.
"""
from __future__ import annotations

from .base import LLMProvider, ModelResponse, Tier


class HeuristicProvider(LLMProvider):
    name = "rule_based_planner"

    def available(self) -> bool:
        return True

    def complete(self, system: str, prompt: str, tier: Tier = Tier.ROUTINE,
                 json_schema: dict | None = None, max_tokens: int = 2000) -> ModelResponse:
        return ModelResponse(
            content=("[rule_based_planner] No language model configured. "
                     "This decision was made by deterministic scoring over measured "
                     "media features."),
            data=None, actor=self.name, is_llm=False, tier=tier.value)
