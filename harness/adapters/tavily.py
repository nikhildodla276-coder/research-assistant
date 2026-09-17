"""Tavily Provider Adapter for Phase A4.4.

Executes controlled retrieval capability probes against Tavily Search API.
"""

import json
import time
from typing import Any, Dict, List, Optional

from harness.adapters.base import (
    BaseProviderAdapter,
    TransportCallable,
    default_http_transport,
)
from harness.models import (
    CandidateRecord,
    FailureClass,
    FailureClassification,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
)


class TavilyAdapter(BaseProviderAdapter):
    """Adapter for Tavily Search API."""

    TAVILY_API_URL = "https://api.tavily.com/search"

    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        env = environ if environ is not None else {}
        auth_var = self.candidate.auth_env_var or "TAVILY_API_KEY"
        api_key = env.get(auth_var)

        if not api_key:
            return ProbeResult(
                probe_id=request.probe_id,
                candidate_id=self.candidate_id,
                status=ProbeStatus.BLOCKED,
                success=False,
                attempted=False,
                contract_satisfied=False,
                error_classification=FailureClassification(
                    failure_class=FailureClass.AUTHENTICATION_FAILURE,
                    error_code="MISSING_CREDENTIAL",
                    raw_message=f"Environment variable '{auth_var}' is required for Tavily probing.",
                    is_retryable=False,
                    recommended_action="PROVIDE_TAVILY_API_KEY",
                ),
                latency_ms=0.0,
            )

        http_transport = transport or default_http_transport
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": api_key,
            "query": request.prompt,
            "max_results": 2,
        }

        start_time = time.perf_counter()
        attempted = True

        try:
            resp = http_transport(
                self.TAVILY_API_URL,
                method="POST",
                headers=headers,
                json_data=payload,
                timeout=request.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    results = data.get("results", [])
                except Exception as e:
                    return ProbeResult(
                        probe_id=request.probe_id,
                        candidate_id=self.candidate_id,
                        status=ProbeStatus.FAILED,
                        success=False,
                        attempted=attempted,
                        contract_satisfied=False,
                        raw_response_snippet=self._sanitize(resp.text[:500], environ),
                        error_classification=self._classify_adapter_error(
                            http_status=resp.status_code,
                            error_code="MALFORMED_JSON",
                            raw_message=f"Failed to parse Tavily JSON response: {e}",
                            environ=environ,
                        ),
                        latency_ms=elapsed_ms,
                    )

                snippets: List[str] = []
                for idx, r in enumerate(results[:2], start=1):
                    title = r.get("title", "Untitled")
                    url = r.get("url", "")
                    content = r.get("content", "")[:150]
                    snippets.append(f"[{idx}] {title} ({url}): {content}")

                norm_text = "\n".join(snippets) if snippets else "No search results returned."
                contract_satisfied = isinstance(results, list) and len(results) > 0

                return ProbeResult(
                    probe_id=request.probe_id,
                    candidate_id=self.candidate_id,
                    status=ProbeStatus.SUCCESS,
                    success=True,
                    attempted=attempted,
                    contract_satisfied=contract_satisfied,
                    normalized_response=self._sanitize(norm_text, environ),
                    raw_response_snippet=self._sanitize(resp.text[:500], environ),
                    latency_ms=elapsed_ms,
                    telemetry={
                        "query": request.prompt,
                        "results_count": len(results),
                    },
                )
            else:
                return ProbeResult(
                    probe_id=request.probe_id,
                    candidate_id=self.candidate_id,
                    status=ProbeStatus.FAILED,
                    success=False,
                    attempted=attempted,
                    contract_satisfied=False,
                    raw_response_snippet=self._sanitize(resp.text[:500], environ),
                    error_classification=self._classify_adapter_error(
                        http_status=resp.status_code,
                        raw_message=resp.text[:500],
                        environ=environ,
                    ),
                    latency_ms=elapsed_ms,
                )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_type = type(exc).__name__
            err_msg = str(exc)
            return ProbeResult(
                probe_id=request.probe_id,
                candidate_id=self.candidate_id,
                status=ProbeStatus.FAILED,
                success=False,
                attempted=attempted,
                contract_satisfied=False,
                error_classification=self._classify_adapter_error(
                    error_code=err_type,
                    raw_message=f"{err_type}: {err_msg}",
                    environ=environ,
                ),
                latency_ms=elapsed_ms,
            )

