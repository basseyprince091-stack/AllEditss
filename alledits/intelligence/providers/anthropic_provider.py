"""Anthropic provider — real implementation of the reasoning tier.

Requires ANTHROPIC_API_KEY and network egress. When either is missing,
available() returns False and the orchestrator degrades to the rule-based
planner *with that fact recorded in the decision ledger* — it never pretends.

Model IDs are configuration, not hard-coded assumptions, because the right
model at implementation time is whatever is current (Spec §20).
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from .base import LLMProvider, ModelResponse, Tier
from ...core.errors import ProviderError, ProviderUnavailable

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None,
                 reasoning_model: str | None = None,
                 routine_model: str | None = None,
                 timeout: int = 120):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Configurable so the deployment picks the current best model.
        self.models = {
            Tier.REASONING: reasoning_model or os.environ.get(
                "ALLEDITS_REASONING_MODEL", "claude-opus-4-5"),
            Tier.ROUTINE: routine_model or os.environ.get(
                "ALLEDITS_ROUTINE_MODEL", "claude-sonnet-4-5"),
        }
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, prompt: str, tier: Tier = Tier.ROUTINE,
                 json_schema: dict | None = None, max_tokens: int = 2000) -> ModelResponse:
        if not self.available():
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")

        if json_schema:
            system += ("\n\nRespond with a single valid JSON object and nothing else. "
                       "No prose, no markdown fences. Conform to this schema:\n"
                       + json.dumps(json_schema))

        body = json.dumps({
            "model": self.models[tier],
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ProviderError(f"Anthropic HTTP {e.code}: {e.read()[:400]!r}")
        except Exception as e:
            raise ProviderUnavailable(f"Anthropic unreachable: {e}")

        text = "".join(b.get("text", "") for b in payload.get("content", [])
                       if b.get("type") == "text")
        data = None
        if json_schema:
            cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                raise ProviderError("model did not return parseable JSON")
        return ModelResponse(content=text, data=data,
                             actor=f"anthropic:{self.models[tier]}",
                             is_llm=True, tier=tier.value,
                             usage=payload.get("usage", {}))
