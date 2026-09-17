"""Fallback adapter for candidates without an automated transport implementation.

Documents what is missing without crashing or inventing interfaces.
"""

from typing import Any, Dict, Optional

from harness.adapters.base import BaseProviderAdapter, TransportCallable
from harness.models import (
    CandidateRecord,
    FailureClass,
    FailureClassification,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
)


class UnsupportedAdapter(BaseProviderAdapter):
    """Adapter for candidates that lack an automated execution interface."""

    def execute_probe(
        self,
        request: ProbeRequest,
        environ: Optional[Dict[str, str]] = None,
        transport: Optional[TransportCallable] = None,
    ) -> ProbeResult:
        reason = (
            f"Candidate '{self.candidate_id}' with provider '{self.candidate.provider_id}' "
            f"and access mode '{self.candidate.access_mode.value}' has no automated adapter."
        )
        return ProbeResult(
            probe_id=request.probe_id,
            candidate_id=self.candidate_id,
            status=ProbeStatus.FAILED,
            success=False,
            attempted=False,
            contract_satisfied=False,
            error_classification=FailureClassification(
                failure_class=FailureClass.UNSUPPORTED_REQUEST,
                error_code="UNSUPPORTED_CANDIDATE_ADAPTER",
                raw_message=reason,
                is_retryable=False,
                recommended_action="IMPLEMENT_DEDICATED_ADAPTER_OR_CHANGE_ACCESS_MODE",
            ),
            latency_ms=0.0,
            telemetry={
                "provider_id": self.candidate.provider_id,
                "access_mode": self.candidate.access_mode.value,
                "status": "NOT_READY",
            },
        )

