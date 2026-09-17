"""DuckDuckGo Provider Adapter for Phase A4.4.

Executes controlled retrieval capability probes against DuckDuckGo Lite.
"""

import time
import urllib.parse
from typing import Any, Dict, Optional

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


class DuckDuckGoAdapter(BaseProviderAdapter):
    """Adapter for DuckDuckGo open retrieval."""

    DDG_HTML_URL = "https://html.duckduckgo.com/html/"

    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        http_transport = transport or default_http_transport
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "ResearchAssistantVerificationHarness/1.0",
        }
        encoded_query = urllib.parse.urlencode({"q": request.prompt})
        target_url = f"{self.DDG_HTML_URL}?{encoded_query}"

        start_time = time.perf_counter()
        attempted = True

        try:
            resp = http_transport(
                target_url,
                method="GET",
                headers=headers,
                timeout=request.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                raw_text = resp.text
                contract_satisfied = len(raw_text.strip()) > 0
                norm_text = f"DuckDuckGo search probe returned {len(raw_text)} characters."

                return ProbeResult(
                    probe_id=request.probe_id,
                    candidate_id=self.candidate_id,
                    status=ProbeStatus.SUCCESS,
                    success=True,
                    attempted=attempted,
                    contract_satisfied=contract_satisfied,
                    normalized_response=norm_text,
                    raw_response_snippet=self._sanitize(raw_text[:500], environ),
                    latency_ms=elapsed_ms,
                    telemetry={
                        "query": request.prompt,
                        "content_length": len(raw_text),
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

