"""Groq Provider Adapter for Phase A4.4.

Executes controlled capability probes against Groq Cloud API.
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


class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq Cloud API."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        env = environ if environ is not None else {}
        auth_var = self.candidate.auth_env_var or "GROQ_API_KEY"
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
                    raw_message=f"Environment variable '{auth_var}' is required for Groq probing.",
                    is_retryable=False,
                    recommended_action="PROVIDE_GROQ_API_KEY",
                ),
                latency_ms=0.0,
            )

        http_transport = transport or default_http_transport
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.candidate.technical_identifier,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": 128,
        }

        start_time = time.perf_counter()
        attempted = True

        try:
            resp = http_transport(
                self.GROQ_API_URL,
                method="POST",
                headers=headers,
                json_data=payload,
                timeout=request.timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("No choices returned in Groq response")
                    content = choices[0].get("message", {}).get("content", "").strip()
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
                            raw_message=f"Failed to parse Groq response: {e}",
                            environ=environ,
                        ),
                        latency_ms=elapsed_ms,
                    )

                contract_satisfied = False
                if request.expected_output_format == "json":
                    try:
                        json.loads(content)
                        contract_satisfied = True
                    except Exception:
                        contract_satisfied = False
                else:
                    contract_satisfied = len(content) > 0

                usage: Dict[str, int] = {}
                raw_usage = data.get("usage")
                if isinstance(raw_usage, dict):
                    if "prompt_tokens" in raw_usage and isinstance(raw_usage["prompt_tokens"], int):
                        usage["input_tokens"] = raw_usage["prompt_tokens"]
                    if "completion_tokens" in raw_usage and isinstance(raw_usage["completion_tokens"], int):
                        usage["output_tokens"] = raw_usage["completion_tokens"]
                    if "total_tokens" in raw_usage and isinstance(raw_usage["total_tokens"], int):
                        usage["total_tokens"] = raw_usage["total_tokens"]

                return ProbeResult(
                    probe_id=request.probe_id,
                    candidate_id=self.candidate_id,
                    status=ProbeStatus.SUCCESS,
                    success=True,
                    attempted=attempted,
                    contract_satisfied=contract_satisfied,
                    normalized_response=self._sanitize(content, environ),
                    raw_response_snippet=self._sanitize(resp.text[:500], environ),
                    latency_ms=elapsed_ms,
                    usage_metadata=usage if usage else None,
                    telemetry={
                        "model": self.candidate.technical_identifier,
                        "finish_reason": choices[0].get("finish_reason") if choices else None,
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

