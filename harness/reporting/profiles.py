"""Candidate Capability Profile Schemas for Phase A4.6.

Defines reproducible, machine-readable data models capturing empirical benchmark
evidence for each evaluated candidate provider/tool.
Contains zero subjective ranking, zero scores, and zero winner designations.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from harness.models import FailureClassification, ProbeStatus, ProvenanceState


class TaskObservationResult(BaseModel):
    """Encapsulates the observed execution outcome of a single benchmark task."""
    task_id: str = Field(description="Unique task identifier")
    task_version: str = Field(description="Version of task fixture executed")
    task_category: str = Field(description="Capability category evaluated")
    execution_status: ProbeStatus = Field(description="Outcome status: SUCCESS, FAILED, or BLOCKED")
    contract_satisfied: bool = Field(description="Whether output contract constraints were satisfied")
    failure_classification: Optional[FailureClassification] = Field(
        default=None,
        description="Structured failure category if execution failed or was blocked",
    )
    latency_ms: float = Field(default=0.0, description="Observed execution latency in milliseconds")
    provenance_state: ProvenanceState = Field(
        description="Source provenance standing (RETRIEVED_BY_SYSTEM, MODEL_CLAIM_ONLY, NOT_AVAILABLE)",
    )
    evidence_id: str = Field(description="Traceable reference to the underlying BenchmarkEvidenceRecord")


class LatencyStatistics(BaseModel):
    """Objective statistical summary of observed execution latencies."""
    sample_count: int = Field(default=0, description="Number of successful latency measurements")
    min_ms: float = Field(default=0.0, description="Minimum observed latency in milliseconds")
    max_ms: float = Field(default=0.0, description="Maximum observed latency in milliseconds")
    mean_ms: float = Field(default=0.0, description="Arithmetic mean observed latency in milliseconds")
    median_ms: float = Field(default=0.0, description="Median observed latency in milliseconds")


class CategoryCapabilityObservation(BaseModel):
    """Empirical observations grouped by capability category."""
    task_category: str = Field(description="Category of capability evaluated")
    total_tasks: int = Field(description="Total number of evaluated tasks in this category")
    successful_tasks: int = Field(description="Number of tasks that executed without transport error")
    contracts_satisfied: int = Field(description="Number of tasks that satisfied output contract")
    contract_satisfaction_rate: float = Field(
        description="Fraction of tasks satisfying contract (0.0 to 1.0)",
    )
    observed_capabilities: List[str] = Field(
        default_factory=list,
        description="List of verified capabilities observed in this category",
    )
    evidence_count: int = Field(description="Number of supporting evidence records")


class ContractSatisfactionSummary(BaseModel):
    """Overall summary of contract satisfaction across all evaluated tasks."""
    total_evaluated: int = Field(default=0, description="Total number of tasks evaluated")
    satisfied_count: int = Field(default=0, description="Number of tasks satisfying contract")
    unsatisfied_count: int = Field(default=0, description="Number of tasks failing contract")
    satisfaction_rate: float = Field(
        default=0.0,
        description="Ratio of satisfied to total evaluated tasks (0.0 to 1.0)",
    )


class ProvenanceObservationSummary(BaseModel):
    """Summary of observed source citations and retrieval provenance."""
    total_observations: int = Field(default=0, description="Total number of task provenance records")
    retrieved_by_system_count: int = Field(
        default=0,
        description="Tasks where sources were actively fetched by system retrieval pipeline",
    )
    model_claim_only_count: int = Field(
        default=0,
        description="Tasks where model asserted citations without system verification",
    )
    not_available_count: int = Field(
        default=0,
        description="Tasks where no citations or retrieval were present",
    )
    verification_pending_count: int = Field(
        default=0,
        description="Tasks where provenance verification is pending",
    )
    total_model_claimed_sources: int = Field(
        default=0,
        description="Total count of individual citations claimed by the model",
    )
    total_system_retrieved_sources: int = Field(
        default=0,
        description="Total count of individual documents retrieved by system",
    )


class CandidateCapabilityProfile(BaseModel):
    """Structured, reproducible capability baseline profile for a candidate provider.

    Contains zero subjective rankings, zero comparative winners, and zero quality scores.
    All metrics represent direct empirical observations.
    """
    candidate_id: str = Field(description="Unique canonical identifier of the evaluated candidate")
    benchmark_suite_id: str = Field(description="Benchmark suite identifier (e.g. sih_cadastral_v1)")
    benchmark_suite_version: str = Field(default="1.0.0", description="Version of the benchmark suite")
    evaluated_task_count: int = Field(description="Total tasks evaluated for this candidate")
    successful_task_count: int = Field(description="Tasks executing with ProbeStatus.SUCCESS")
    failed_task_count: int = Field(description="Tasks executing with ProbeStatus.FAILED")
    blocked_task_count: int = Field(description="Tasks blocked by pre-probe safety gates (ProbeStatus.BLOCKED)")
    task_results: List[TaskObservationResult] = Field(
        default_factory=list,
        description="Deterministic list of individual task results sorted by task_id",
    )
    capability_observations: List[CategoryCapabilityObservation] = Field(
        default_factory=list,
        description="Observations aggregated by category, sorted by task_category",
    )
    contract_satisfaction_observations: ContractSatisfactionSummary = Field(
        default_factory=ContractSatisfactionSummary,
        description="Overall contract compliance summary",
    )
    observed_failure_classes: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of failure classes observed during evaluation, sorted by key",
    )
    observed_latency_statistics: LatencyStatistics = Field(
        default_factory=LatencyStatistics,
        description="Objective latency summary statistics",
    )
    provenance_observations: ProvenanceObservationSummary = Field(
        default_factory=ProvenanceObservationSummary,
        description="Provenance audit summary tracking model claims vs system retrieval",
    )
    benchmark_versions: List[str] = Field(
        default_factory=list,
        description="Sorted list of benchmark/task versions involved in this profile",
    )
    evidence_references: List[str] = Field(
        default_factory=list,
        description="Sorted list of evidence record IDs supporting this profile",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of profile generation",
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must not be empty")
        return v.strip()

