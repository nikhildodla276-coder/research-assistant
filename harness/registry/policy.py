"""Explicit, Deterministic Capability Verification Policy for Phase A4.7.

Encapsulates strict, inspectable rules for translating empirical task observations
into verifiable capability states:
- UNVERIFIED
- VERIFIED_PASS
- VERIFIED_FAIL
- BLOCKED
- NOT_READY

Zero arbitrary percentage thresholds (e.g. no '80% is pass').
Requires 100% contract compliance for VERIFIED_PASS across evaluated tasks.
"""

import statistics
from typing import List, Optional

from harness.models import (
    CandidateType,
    FailureClass,
    ProbeStatus,
    ProvenanceState,
)
from harness.registry.models import (
    CapabilityVerificationRecord,
    RegistryVerificationStatus,
)
from harness.reporting.profiles import LatencyStatistics, TaskObservationResult

POLICY_VERSION: str = "1.0.0"


def evaluate_capability_verification(
    candidate_id: str,
    candidate_type: CandidateType,
    capability: str,
    tasks: List[TaskObservationResult],
    benchmark_id: str,
    benchmark_version: str,
    policy_version: str = POLICY_VERSION,
    notes: Optional[str] = None,
) -> CapabilityVerificationRecord:
    """Evaluates task observations against deterministic contract criteria for a capability.

    Rules:
    1. If no tasks evaluated: UNVERIFIED (observation_count = 0).
    2. If tasks encountered UNSUPPORTED_REQUEST or CONFIGURATION_ERROR without any successes: NOT_READY.
    3. If all evaluated tasks were BLOCKED by safety/cost/access gates: BLOCKED.
    4. If any task failed execution or violated output contract: VERIFIED_FAIL.
    5. If some tasks passed but others were blocked (partial execution): BLOCKED with explicit notes.
       (Passing a subset of tasks does NOT grant capability-wide verification).
    6. If all evaluated tasks succeeded without transport error AND 100% of output contracts
       were satisfied: VERIFIED_PASS.
    """
    clean_id = candidate_id.strip() if candidate_id else ""
    clean_cap = capability.strip() if capability else ""

    if not tasks:
        return CapabilityVerificationRecord(
            candidate_id=clean_id,
            candidate_type=candidate_type,
            capability=clean_cap,
            verification_status=RegistryVerificationStatus.UNVERIFIED,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            observed_task_ids=[],
            passed_task_ids=[],
            failed_task_ids=[],
            blocked_task_ids=[],
            evidence_ids=[],
            observation_count=0,
            contract_satisfied_count=0,
            contract_satisfaction_rate=0.0,
            latency_statistics=None,
            provenance_state=ProvenanceState.NOT_AVAILABLE,
            model_claimed_sources_count=0,
            system_retrieved_sources_count=0,
            policy_version=policy_version,
            notes=notes or "No empirical task observations available for capability.",
        )

    # Deterministic sorting of task results by task_id and evidence_id
    sorted_tasks = sorted(tasks, key=lambda t: (t.task_id, t.evidence_id))
    obs_task_ids = sorted(list({t.task_id for t in sorted_tasks}))
    evidence_ids = sorted(list({t.evidence_id for t in sorted_tasks}))

    passed_task_ids: List[str] = []
    failed_task_ids: List[str] = []
    blocked_task_ids: List[str] = []

    unsupported_or_config_error = False

    for t in sorted_tasks:
        if t.execution_status == ProbeStatus.BLOCKED:
            blocked_task_ids.append(t.task_id)
        elif t.execution_status == ProbeStatus.FAILED:
            failed_task_ids.append(t.task_id)
            if t.failure_classification and t.failure_classification.failure_class in (
                FailureClass.UNSUPPORTED_REQUEST,
                FailureClass.CONFIGURATION_ERROR,
            ):
                unsupported_or_config_error = True
        elif t.execution_status == ProbeStatus.SUCCESS:
            if t.contract_satisfied:
                passed_task_ids.append(t.task_id)
            else:
                failed_task_ids.append(t.task_id)

    passed_unique = sorted(list(set(passed_task_ids)))
    failed_unique = sorted(list(set(failed_task_ids)))
    blocked_unique = sorted(list(set(blocked_task_ids)))

    total_obs = len(sorted_tasks)
    sat_count = len(passed_unique)
    sat_rate = round(sat_count / len(obs_task_ids), 4) if obs_task_ids else 0.0

    # Determine deterministic verification status
    status_notes = notes
    if unsupported_or_config_error and not passed_unique:
        status = RegistryVerificationStatus.NOT_READY
        status_notes = status_notes or "Adapter or candidate execution infrastructure is not ready or unsupported."
    elif len(blocked_unique) == len(obs_task_ids):
        status = RegistryVerificationStatus.BLOCKED
        status_notes = status_notes or "All evaluated tasks were blocked by pre-probe safety/cost gates."
    elif failed_unique:
        status = RegistryVerificationStatus.VERIFIED_FAIL
        status_notes = status_notes or f"Verification failed: {len(failed_unique)} task(s) failed execution or violated contract."
    elif blocked_unique and passed_unique:
        # Partial evidence: some passed, some blocked. Strictly NOT VERIFIED_PASS.
        status = RegistryVerificationStatus.BLOCKED
        status_notes = status_notes or f"Partial evaluation: {len(passed_unique)} passed, {len(blocked_unique)} blocked. Capability not verified."
    elif len(passed_unique) == len(obs_task_ids) and obs_task_ids:
        status = RegistryVerificationStatus.VERIFIED_PASS
        status_notes = status_notes or "All observed benchmark tasks executed successfully and satisfied contract constraints."
    else:
        status = RegistryVerificationStatus.UNVERIFIED
        status_notes = status_notes or "Insufficient empirical observations to determine verification state."

    # Compute Latency Statistics for successful executions
    latencies = [t.latency_ms for t in sorted_tasks if t.execution_status == ProbeStatus.SUCCESS and t.latency_ms > 0]
    if latencies:
        latency_stats = LatencyStatistics(
            sample_count=len(latencies),
            min_ms=round(min(latencies), 2),
            max_ms=round(max(latencies), 2),
            mean_ms=round(statistics.mean(latencies), 2),
            median_ms=round(statistics.median(latencies), 2),
        )
    else:
        latency_stats = None

    # Provenance Observations
    prov_states = {t.provenance_state for t in sorted_tasks}
    if ProvenanceState.RETRIEVED_BY_SYSTEM in prov_states:
        overall_prov = ProvenanceState.RETRIEVED_BY_SYSTEM
    elif ProvenanceState.MODEL_CLAIM_ONLY in prov_states:
        overall_prov = ProvenanceState.MODEL_CLAIM_ONLY
    elif ProvenanceState.VERIFICATION_PENDING in prov_states:
        overall_prov = ProvenanceState.VERIFICATION_PENDING
    else:
        overall_prov = ProvenanceState.NOT_AVAILABLE

    # Trace source counts
    sys_retrieved = sum(1 for t in sorted_tasks if t.provenance_state == ProvenanceState.RETRIEVED_BY_SYSTEM)
    model_claims = sum(1 for t in sorted_tasks if t.provenance_state == ProvenanceState.MODEL_CLAIM_ONLY)

    return CapabilityVerificationRecord(
        candidate_id=clean_id,
        candidate_type=candidate_type,
        capability=clean_cap,
        verification_status=status,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        observed_task_ids=obs_task_ids,
        passed_task_ids=passed_unique,
        failed_task_ids=failed_unique,
        blocked_task_ids=blocked_unique,
        evidence_ids=evidence_ids,
        observation_count=total_obs,
        contract_satisfied_count=sat_count,
        contract_satisfaction_rate=sat_rate,
        latency_statistics=latency_stats,
        provenance_state=overall_prov,
        model_claimed_sources_count=model_claims,
        system_retrieved_sources_count=sys_retrieved,
        policy_version=policy_version,
        notes=status_notes,
    )

