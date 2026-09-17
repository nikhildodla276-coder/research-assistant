"""Machine-readable Registry Capability Verification Schemas for Phase A4.7.

Defines Pydantic models for capability verification records, candidate registry entries,
and versioned immutable registry snapshots.
Contains zero subjective ranking, zero composite scores, and zero 'best model' designations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateType,
    ProvenanceState,
    VerificationState,
)
from harness.reporting.profiles import LatencyStatistics


class RegistryVerificationStatus(str, Enum):
    """Deterministic capability verification status within the registry.
    
    Reflects empirical benchmark evidence standing for a specific capability.
    """
    UNVERIFIED = "UNVERIFIED"
    VERIFIED_PASS = "VERIFIED_PASS"
    VERIFIED_FAIL = "VERIFIED_FAIL"
    BLOCKED = "BLOCKED"
    NOT_READY = "NOT_READY"


class CapabilityVerificationRecord(BaseModel):
    """Machine-readable, empirical capability verification record for a candidate.

    Captures verifiable benchmark evidence for a single capability contract.
    Contains zero subjective scores, zero rankings, and zero 'best model' designations.
    Preserves complete provenance and traceability back to A4.6 evidence records.
    """
    candidate_id: str = Field(description="Unique canonical candidate identifier")
    candidate_type: CandidateType = Field(description="Component type (INFERENCE_MODEL, RETRIEVAL_TOOL, etc.)")
    capability: str = Field(
        description="Evaluated capability name (e.g. structured_json_generation, claim_identification)",
    )
    verification_status: RegistryVerificationStatus = Field(
        description="Deterministic status: UNVERIFIED, VERIFIED_PASS, VERIFIED_FAIL, BLOCKED, NOT_READY",
    )
    benchmark_id: str = Field(description="Benchmark suite identifier (e.g. sih_cadastral_v1)")
    benchmark_version: str = Field(default="1.0.0", description="Version of the benchmark suite evaluated")
    observed_task_ids: List[str] = Field(
        default_factory=list,
        description="Sorted list of task identifiers evaluated for this capability",
    )
    passed_task_ids: List[str] = Field(
        default_factory=list,
        description="Task identifiers where execution succeeded and contracts were satisfied",
    )
    failed_task_ids: List[str] = Field(
        default_factory=list,
        description="Task identifiers where execution failed or contracts were violated",
    )
    blocked_task_ids: List[str] = Field(
        default_factory=list,
        description="Task identifiers where execution was blocked by safety/cost gates",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Traceable references to underlying BenchmarkEvidenceRecords",
    )
    observation_count: int = Field(
        default=0,
        description="Total number of evaluated tasks supporting this capability observation",
    )
    contract_satisfied_count: int = Field(
        default=0,
        description="Number of observed tasks satisfying output contract constraints",
    )
    contract_satisfaction_rate: float = Field(
        default=0.0,
        description="Fraction of observed tasks satisfying contract constraints (0.0 to 1.0)",
    )
    latency_statistics: Optional[LatencyStatistics] = Field(
        default=None,
        description="Empirical latency statistics for successful task executions",
    )
    provenance_state: Optional[ProvenanceState] = Field(
        default=None,
        description="Observed source provenance standing (e.g. RETRIEVED_BY_SYSTEM, MODEL_CLAIM_ONLY)",
    )
    model_claimed_sources_count: int = Field(
        default=0,
        description="Count of citations asserted by model without system retrieval verification",
    )
    system_retrieved_sources_count: int = Field(
        default=0,
        description="Count of sources fetched and verified by system retrieval pipeline",
    )
    verified_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of verification evaluation",
    )
    evidence_source: str = Field(
        default="A4.6_CandidateCapabilityProfile",
        description="Upstream source artifact type for verification evidence",
    )
    policy_version: str = Field(
        default="1.0.0",
        description="Version of deterministic verification policy applied",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Contextual inspection notes on observations, partial evidence, or limits",
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must not be empty")
        return v.strip()

    @field_validator("capability")
    @classmethod
    def validate_capability(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("capability must not be empty")
        return v.strip()

    @field_validator("notes")
    @classmethod
    def validate_no_secrets_in_notes(cls, v: Optional[str]) -> Optional[str]:
        if v:
            v_lower = v.lower()
            if any(s in v_lower for s in ["api_key", "password", "token=", "bearer "]):
                raise ValueError("Potential credential detected in verification record notes")
        return v


class CandidateRegistryEntry(BaseModel):
    """Aggregated registry entry for a candidate, cataloging verified capabilities.

    This is a passive metadata record. It does NOT perform runtime routing,
    dynamic failover, or multi-provider execution.
    """
    candidate_id: str = Field(description="Canonical candidate identifier")
    display_name: str = Field(description="Human-readable title")
    candidate_type: CandidateType = Field(description="Candidate component type")
    provider_id: str = Field(description="Service operator / provider ID")
    access_mode: AccessMode = Field(description="Transport / execution access mode")
    billing_status: BillingStatus = Field(
        default=BillingStatus.UNKNOWN,
        description="Cost / billing liability classification",
    )
    claimed_capabilities: List[str] = Field(
        default_factory=list,
        description="Vendor or manifest claimed capabilities (unverified claims)",
    )
    verified_capabilities: Dict[str, CapabilityVerificationRecord] = Field(
        default_factory=dict,
        description="Empirically verified capability records indexed by capability name",
    )
    overall_verification_state: VerificationState = Field(
        default=VerificationState.UNVERIFIED,
        description="Coarse verification state reflecting aggregate evidence standing",
    )
    last_verified_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of latest capability verification evaluation",
    )

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("candidate_id must not be empty")
        return v.strip()


class RegistrySnapshot(BaseModel):
    """Immutable, versioned snapshot of registry capability verification entries.

    Preserves empirical verification state as a point-in-time baseline.
    Zero database dependencies; serialized directly as deterministic JSON.
    """
    snapshot_id: str = Field(
        description="Unique identifier for this snapshot (e.g. reg_snap_sih_cadastral_v1_20260917)",
    )
    benchmark_suite_id: str = Field(description="Benchmark suite providing empirical evidence")
    benchmark_suite_version: str = Field(default="1.0.0", description="Benchmark suite version")
    policy_version: str = Field(default="1.0.0", description="Verification policy version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of snapshot creation",
    )
    candidate_count: int = Field(default=0, description="Total candidates cataloged in this snapshot")
    entries: Dict[str, CandidateRegistryEntry] = Field(
        default_factory=dict,
        description="Deterministic mapping of candidate entries keyed by candidate_id",
    )

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("snapshot_id must not be empty")
        return v.strip()

