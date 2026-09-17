"""Controlled Provider Probing Harness for Phase A4.4.

Orchestrates candidate capability probes through strict A4.3 access/cost gates,
specialized adapters, and standardized output normalization.
Strictly offline unless explicitly instructed with a live transport and valid credentials.
Zero automatic routing, zero failover, zero model ranking.
"""

from typing import Any, Dict, Optional

from harness.adapters.base import TransportCallable
from harness.adapters.registry import get_adapter_for_candidate
from harness.gates import check_candidate_access_and_cost
from harness.models import (
    BenchmarkFixture,
    CandidateRecord,
    FailureClass,
    FailureClassification,
    GateName,
    GateStatus,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
)


def run_candidate_probe(
    candidate: CandidateRecord,
    request: ProbeRequest,
    environ: Optional[Dict[str, str]] = None,
    transport: Optional[TransportCallable] = None,
) -> ProbeResult:
    """Executes a controlled capability probe against a candidate provider.

    Mandatory Gate Enforcement:
    Evaluates candidate against A4.3 pre-probe access and cost gates first.
    If gates fail or candidate is not allowed to proceed, execution aborts immediately
    and an un-attempted, BLOCKED ProbeResult is returned.
    """
    # -------------------------------------------------------------------------
    # Step 1: Pre-Probe Safety Gate Check (Mandatory A4.3 Boundary)
    # -------------------------------------------------------------------------
    gate_decision = check_candidate_access_and_cost(candidate, environ=environ)

    if gate_decision.status != GateStatus.ALLOWED or not gate_decision.may_proceed_to_probe:
        # Determine appropriate failure classification
        if (
            GateName.BILLING_STATUS_GATE in gate_decision.failed_gates
            or GateName.CARD_REQUIRED_GATE in gate_decision.failed_gates
        ):
            fail_class = FailureClass.BILLING_BLOCKED
        elif GateName.CREDENTIAL_GATE in gate_decision.failed_gates:
            fail_class = FailureClass.AUTHENTICATION_FAILURE
        elif GateName.ACCESS_MODE_GATE in gate_decision.failed_gates:
            fail_class = FailureClass.ACCESS_BLOCKED
        elif GateName.USER_APPROVAL_GATE in gate_decision.failed_gates:
            fail_class = FailureClass.ACCESS_BLOCKED
        else:
            fail_class = FailureClass.ACCESS_BLOCKED

        return ProbeResult(
            probe_id=request.probe_id,
            candidate_id=candidate.candidate_id,
            status=ProbeStatus.BLOCKED,
            success=False,
            attempted=False,
            contract_satisfied=False,
            error_classification=FailureClassification(
                failure_class=fail_class,
                error_code="A4_3_PRE_PROBE_GATE_BLOCKED",
                raw_message=f"Pre-probe gate blocked execution: {gate_decision.reason}",
                is_retryable=False,
                recommended_action="RESOLVE_A4_3_GATE_BLOCKERS",
            ),
            latency_ms=0.0,
            telemetry={
                "gate_status": gate_decision.status.value,
                "failed_gates": [g.value for g in gate_decision.failed_gates],
                "requires_user_approval": gate_decision.requires_user_approval,
            },
        )

    # -------------------------------------------------------------------------
    # Step 2: Adapter Selection & Execution
    # -------------------------------------------------------------------------
    adapter = get_adapter_for_candidate(candidate)
    probe_result = adapter.execute_probe(
        request=request,
        environ=environ,
        transport=transport,
    )

    # Attach gate pass telemetry
    probe_result.telemetry["gate_status"] = GateStatus.ALLOWED.value
    probe_result.telemetry["access_mode"] = candidate.access_mode.value

    return probe_result


def run_probe_from_fixture(
    candidate: CandidateRecord,
    fixture: BenchmarkFixture,
    environ: Optional[Dict[str, str]] = None,
    transport: Optional[TransportCallable] = None,
) -> ProbeResult:
    """Convenience helper to create a ProbeRequest from a BenchmarkFixture and execute it."""
    prompt = ""
    if isinstance(fixture.input_payload, dict):
        prompt = fixture.input_payload.get("prompt") or fixture.input_payload.get("query") or ""
    elif isinstance(fixture.input_payload, str):
        prompt = fixture.input_payload

    request = ProbeRequest(
        candidate_id=candidate.candidate_id,
        prompt=prompt,
        expected_output_format="json" if fixture.expected_structure else "text",
        metadata={
            "fixture_id": fixture.fixture_id,
            "fixture_version": fixture.version,
        },
    )

    return run_candidate_probe(
        candidate=candidate,
        request=request,
        environ=environ,
        transport=transport,
    )
