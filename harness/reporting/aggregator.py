"""Deterministic Evidence Aggregation Module for Phase A4.6.

Aggregates raw BenchmarkEvidenceRecord collections into structured,
reproducible CandidateCapabilityProfile artifacts.
Strictly deterministic, zero subjective scoring, zero model rankings.
"""

import statistics
from collections import defaultdict
from typing import Dict, List, Optional

from harness.models import (
    BenchmarkEvidenceRecord,
    ProbeStatus,
    ProvenanceState,
)
from harness.reporting.profiles import (
    CandidateCapabilityProfile,
    CategoryCapabilityObservation,
    ContractSatisfactionSummary,
    LatencyStatistics,
    ProvenanceObservationSummary,
    TaskObservationResult,
)


def aggregate_candidate_evidence(
    candidate_id: str,
    evidence_records: List[BenchmarkEvidenceRecord],
    benchmark_suite_id: Optional[str] = None,
    benchmark_suite_version: Optional[str] = None,
) -> CandidateCapabilityProfile:
    """Aggregates benchmark evidence for a single candidate into a structured profile.

    Guarantees deterministic sorting and bit-for-bit reproducibility for identical inputs.
    """
    clean_id = candidate_id.strip() if candidate_id else ""
    # Filter records matching this candidate_id
    matching = [r for r in evidence_records if r.candidate_id == clean_id]

    # Sort matching records deterministically by task_id then evidence_id
    matching = sorted(matching, key=lambda r: (r.task_id, r.evidence_id))

    suite_id = benchmark_suite_id or (matching[0].benchmark_id if matching else "unknown_suite")
    suite_ver = benchmark_suite_version or (matching[0].task_version if matching else "1.0.0")

    if not matching:
        # Return safe empty baseline profile
        return CandidateCapabilityProfile(
            candidate_id=clean_id or "unknown_candidate",
            benchmark_suite_id=suite_id,
            benchmark_suite_version=suite_ver,
            evaluated_task_count=0,
            successful_task_count=0,
            failed_task_count=0,
            blocked_task_count=0,
            task_results=[],
            capability_observations=[],
            contract_satisfaction_observations=ContractSatisfactionSummary(),
            observed_failure_classes={},
            observed_latency_statistics=LatencyStatistics(),
            provenance_observations=ProvenanceObservationSummary(),
            benchmark_versions=[suite_ver],
            evidence_references=[],
        )

    # 1. Status Counts
    eval_count = len(matching)
    success_count = sum(1 for r in matching if r.execution_status == ProbeStatus.SUCCESS)
    failed_count = sum(1 for r in matching if r.execution_status == ProbeStatus.FAILED)
    blocked_count = sum(1 for r in matching if r.execution_status == ProbeStatus.BLOCKED)

    # 2. Task Results
    task_results: List[TaskObservationResult] = []
    for r in matching:
        task_results.append(
            TaskObservationResult(
                task_id=r.task_id,
                task_version=r.task_version,
                task_category=r.task_category,
                execution_status=r.execution_status,
                contract_satisfied=r.contract_satisfied,
                failure_classification=r.failure_classification,
                latency_ms=round(r.latency_ms, 2),
                provenance_state=r.provenance_metadata.provenance_status,
                evidence_id=r.evidence_id,
            )
        )
    # Ensure stable sorting by task_id
    task_results = sorted(task_results, key=lambda t: (t.task_id, t.evidence_id))

    # 3. Category Capability Observations
    by_category: Dict[str, List[BenchmarkEvidenceRecord]] = defaultdict(list)
    for r in matching:
        by_category[r.task_category].append(r)

    category_observations: List[CategoryCapabilityObservation] = []
    for cat in sorted(by_category.keys()):
        cat_recs = by_category[cat]
        cat_total = len(cat_recs)
        cat_succ = sum(1 for r in cat_recs if r.execution_status == ProbeStatus.SUCCESS)
        cat_sat = sum(1 for r in cat_recs if r.contract_satisfied)
        cat_rate = round(cat_sat / cat_total, 4) if cat_total > 0 else 0.0

        caps: List[str] = []
        if cat_sat > 0:
            caps.append(f"{cat}:contract_satisfied")
        if cat_sat < cat_total:
            caps.append(f"{cat}:contract_unfulfilled")

        category_observations.append(
            CategoryCapabilityObservation(
                task_category=cat,
                total_tasks=cat_total,
                successful_tasks=cat_succ,
                contracts_satisfied=cat_sat,
                contract_satisfaction_rate=cat_rate,
                observed_capabilities=caps,
                evidence_count=cat_total,
            )
        )

    # 4. Overall Contract Satisfaction
    sat_count = sum(1 for r in matching if r.contract_satisfied)
    unsat_count = eval_count - sat_count
    sat_rate = round(sat_count / eval_count, 4) if eval_count > 0 else 0.0
    contract_summary = ContractSatisfactionSummary(
        total_evaluated=eval_count,
        satisfied_count=sat_count,
        unsatisfied_count=unsat_count,
        satisfaction_rate=sat_rate,
    )

    # 5. Observed Failure Classes
    failure_counts: Dict[str, int] = defaultdict(int)
    for r in matching:
        if r.failure_classification:
            fc_val = r.failure_classification.failure_class.value
            failure_counts[fc_val] += 1
    sorted_failures = {k: failure_counts[k] for k in sorted(failure_counts.keys())}

    # 6. Latency Statistics
    latencies = [r.latency_ms for r in matching if r.latency_ms > 0]
    if latencies:
        latency_stats = LatencyStatistics(
            sample_count=len(latencies),
            min_ms=round(min(latencies), 2),
            max_ms=round(max(latencies), 2),
            mean_ms=round(statistics.mean(latencies), 2),
            median_ms=round(statistics.median(latencies), 2),
        )
    else:
        latency_stats = LatencyStatistics()

    # 7. Provenance Observations
    prov_summary = ProvenanceObservationSummary(
        total_observations=eval_count,
        retrieved_by_system_count=sum(
            1 for r in matching if r.provenance_metadata.provenance_status == ProvenanceState.RETRIEVED_BY_SYSTEM
        ),
        model_claim_only_count=sum(
            1 for r in matching if r.provenance_metadata.provenance_status == ProvenanceState.MODEL_CLAIM_ONLY
        ),
        not_available_count=sum(
            1 for r in matching if r.provenance_metadata.provenance_status == ProvenanceState.NOT_AVAILABLE
        ),
        verification_pending_count=sum(
            1 for r in matching if r.provenance_metadata.provenance_status == ProvenanceState.VERIFICATION_PENDING
        ),
        total_model_claimed_sources=sum(
            len(r.provenance_metadata.model_claimed_sources) for r in matching
        ),
        total_system_retrieved_sources=sum(
            len(r.provenance_metadata.system_retrieved_sources) for r in matching
        ),
    )

    # 8. Versions & Evidence References
    versions = sorted(list({r.task_version for r in matching} | {suite_ver}))
    evidence_refs = sorted(list({r.evidence_id for r in matching}))

    return CandidateCapabilityProfile(
        candidate_id=clean_id,
        benchmark_suite_id=suite_id,
        benchmark_suite_version=suite_ver,
        evaluated_task_count=eval_count,
        successful_task_count=success_count,
        failed_task_count=failed_count,
        blocked_task_count=blocked_count,
        task_results=task_results,
        capability_observations=category_observations,
        contract_satisfaction_observations=contract_summary,
        observed_failure_classes=sorted_failures,
        observed_latency_statistics=latency_stats,
        provenance_observations=prov_summary,
        benchmark_versions=versions,
        evidence_references=evidence_refs,
    )


def aggregate_suite_evidence(
    evidence_records: List[BenchmarkEvidenceRecord],
) -> Dict[str, CandidateCapabilityProfile]:
    """Aggregates evidence across all candidates present in evidence records."""
    candidate_ids = sorted(list({r.candidate_id for r in evidence_records if r.candidate_id}))
    profiles: Dict[str, CandidateCapabilityProfile] = {}
    for cid in candidate_ids:
        profiles[cid] = aggregate_candidate_evidence(
            candidate_id=cid,
            evidence_records=evidence_records,
        )
    return profiles

