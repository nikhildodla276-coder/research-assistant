"""Local Ollama Provider Adapter for Phase A4.4.

Executes controlled capability probes against a locally running Ollama daemon.
"""

import json
import time
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


class OllamaAdapter(BaseProviderAdapter):
    """Adapter for LOCAL Ollama daemon."""

    def __init__(self, candidate: CandidateRecord):
        super().__init__(candidate)
        endpoint_override = None
        if candidate.provenance and isinstance(candidate.provenance, dict):
            endpoint_override = candidate.provenance.get("endpoint")
        self.endpoint = endpoint_override or "http://localhost:11434"

    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        http_transport = transport or default_http_transport
        target_url = f"{self.endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self.candidate.technical_identifier,
            "prompt": request.prompt,
            "stream": False,
        }

        start_time = time.perf_counter()
        attempted = True

        try:
            resp = http_transport(
                target_url,
                method="POST",
                headers={"Content-Type": "application/json"},
                json_data=payload,
                timeout=request.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                try:
                    data = resp.json()
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
                            raw_message=f"Failed to parse Ollama JSON response: {e}",
                            environ=environ,
                        ),
                        latency_ms=elapsed_ms,
                    )

                norm_text = str(data.get("response", "")).strip()
                # Check contract
                contract_satisfied = False
                if request.expected_output_format == "json":
                    try:
                        json.loads(norm_text)
                        contract_satisfied = True
                    except Exception:
                        contract_satisfied = False
                else:
                    contract_satisfied = len(norm_text) > 0

                usage: Dict[str, int] = {}
                if "prompt_eval_count" in data and isinstance(data["prompt_eval_count"], int):
                    usage["input_tokens"] = data["prompt_eval_count"]
                if "eval_count" in data and isinstance(data["eval_count"], int):
                    usage["output_tokens"] = data["eval_count"]

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
                    usage_metadata=usage if usage else None,
                    telemetry={
                        "endpoint": self.endpoint,
                        "model": self.candidate.technical_identifier,
                        "total_duration_ns": data.get("total_duration"),
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

