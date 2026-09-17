"""Provider Adapters Package for A4.4.

Provides unified provider adapter abstractions for LOCAL, API, and OPEN_SOURCE
candidate transports with strict secret redaction and error classification.
"""

from harness.adapters.base import (
    BaseProviderAdapter,
    TransportCallable,
    TransportResponse,
    default_http_transport,
)
from harness.adapters.duckduckgo import DuckDuckGoAdapter
from harness.adapters.groq import GroqAdapter
from harness.adapters.ollama import OllamaAdapter
from harness.adapters.registry import get_adapter_for_candidate
from harness.adapters.tavily import TavilyAdapter
from harness.adapters.unsupported import UnsupportedAdapter

__all__ = [
    "BaseProviderAdapter",
    "TransportCallable",
    "TransportResponse",
    "default_http_transport",
    "OllamaAdapter",
    "GroqAdapter",
    "TavilyAdapter",
    "DuckDuckGoAdapter",
    "UnsupportedAdapter",
    "get_adapter_for_candidate",
]

