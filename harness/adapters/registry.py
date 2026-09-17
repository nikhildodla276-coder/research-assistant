"""Adapter Registry for Phase A4.4.

Resolves the appropriate BaseProviderAdapter for a given CandidateRecord.
"""

from harness.adapters.base import BaseProviderAdapter
from harness.adapters.duckduckgo import DuckDuckGoAdapter
from harness.adapters.groq import GroqAdapter
from harness.adapters.ollama import OllamaAdapter
from harness.adapters.tavily import TavilyAdapter
from harness.adapters.unsupported import UnsupportedAdapter
from harness.models import AccessMode, CandidateRecord


def get_adapter_for_candidate(candidate: CandidateRecord) -> BaseProviderAdapter:
    """Returns the specialized BaseProviderAdapter instance for the given candidate.

    Falls back cleanly to UnsupportedAdapter if no automated adapter exists.
    """
    provider_id = (candidate.provider_id or "").lower()

    if provider_id == "ollama":
        return OllamaAdapter(candidate)

    if provider_id == "groq":
        return GroqAdapter(candidate)

    if provider_id == "tavily":
        return TavilyAdapter(candidate)

    if provider_id == "duckduckgo":
        return DuckDuckGoAdapter(candidate)

    return UnsupportedAdapter(candidate)
