"""Core schemas for the A4 Verification & Benchmark Harness.

Defines Pydantic data models for candidate representations, benchmark fixtures,
mock provider responses, evaluation results, and failure classifications.
All models are strictly typed, serializable, and contain zero secrets.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CandidateType(str, Enum):
    """Categorization of system candidate components."""
    INFERENCE_MODEL = "INFERENCE_MODEL"
    RETRIEVAL_TOOL = "RETRIEVAL_TOOL"
    EXTRACTION_TOOL = "EXTRACTION_TOOL"
    LOCAL_UTILITY = "LOCAL_UTILITY"


class AccessMode(str, Enum):
    """Technical transport or execution mode."""
    WEB = "WEB"
    API = "API"
    SDK = "SDK"
    LOCAL = "LOCAL"
    OPEN_SOURCE = "OPEN_SOURCE"
    MANUAL = "MANUAL"


class BillingStatus(str, Enum):
    """Formal classification of cost liability."""
    LOCAL_FREE = "LOCAL_FREE"
    FREE_SERVICE_ACCESS = "FREE_SERVICE_ACCESS"
    FREE_TIER = "FREE_TIER"
    TRIAL_CREDITS = "TRIAL_CREDITS"
    CARD_REQUIRED = "CARD_REQUIRED"
    METERED = "METERED"
    PAID_ONLY = "PAID_ONLY"
    UNKNOWN = "UNKNOWN"


class VerificationState(str, Enum):
    """Lifecycle verification status of a candidate."""
    UNVERIFIED = "UNVERIFIED"
    VERIFYING = "VERIFYING"
    VERIFIED_PASS = "VERIFIED_PASS"
    VERIFIED_FAIL = "VERIFIED_FAIL"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class FailureClass(str, Enum):
    """Deterministic failure taxonomy matching A4.1 specification."""
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    ACCESS_BLOCKED = "ACCESS_BLOCKED"
    BILLING_UNKNOWN = "BILLING_UNKNOWN"
    BILLING_BLOCKED = "BILLING_BLOCKED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_DEPRECATED = "MODEL_DEPRECATED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"
    TIMEOUT = "TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    SCHEMA_FAILURE = "SCHEMA_FAILURE"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    CAPABILITY_FAILURE = "CAPABILITY_FAILURE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class EvaluationStatus(str, Enum):
    """Overall outcome of evaluating a candidate against a fixture."""
    PASS = "PASS"
    FAIL = "FAIL"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"


class FailureClassification(BaseModel):
    """Encapsulates a categorized failure with raw provider context."""
    failure_class: FailureClass = Field(description="Normalized failure class")
    http_status: Optional[int] = Field(default=None, description="HTTP status code if applicable")
    error_code: Optional[str] = Field(default=None, description="Vendor or system error code string")
    raw_message: Optional[str] = Field(default=None, description="Raw error text or snippet")
    is_retryable: bool = Field(default=False, description="Whether error is transient and retryable")
    recommended_action: Optional[str] = Field(default=None, description="Suggested recovery or mitigation")


class CandidateRecord(BaseModel):
    """Represents a candidate provider, model, or tool under evaluation.
    Contains zero secrets; references environment variable names only."""
    candidate_id: str = Field(description="Unique canonical identifier for the candidate")
    candidate_type: CandidateType = Field(description="Type of candidate component")
    provider_id: str = Field(description="Unique identifier of the operating service provider")
    technical_identifier: str = Field(description="Exact vendor API identifier string or local tag")
    display_name: str = Field(description="Human-readable title")
    access_mode: AccessMode = Field(description="Execution or transport access mode")
    billing_status: BillingStatus = Field(default=BillingStatus.UNKNOWN, description="Cost status classification")
    auth_env_var: Optional[str] = Field(default=None, description="Name of environment variable holding API key")
    card_required: bool = Field(default=False, description="Whether credit card or payment information is required")
    requires_user_approval: bool = Field(default=False, description="Whether explicit human operator approval is required to probe/route")
    claimed_capabilities: List[str] = Field(default_factory=list, description="Capabilities claimed by vendor/report")
    provenance: Optional[Dict[str, Any]] = Field(default=None, description="Reference metadata documenting candidate origin")
    verification_state: VerificationState = Field(default=VerificationState.UNVERIFIED, description="Current verification state")
    freshness_ttl_days: int = Field(default=14, description="Freshness validity window in days")
    last_verified_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of last verification")

    @field_validator("auth_env_var")
    @classmethod
    def validate_no_raw_secrets(cls, v: Optional[str]) -> Optional[str]:
        if v:
            # Enforce that auth_env_var is an uppercase identifier (e.g. GROQ_API_KEY)
            # and NOT an actual secret payload
            if " " in v or len(v) > 64 or not v.replace("_", "").isalnum():
                raise ValueError(f"auth_env_var must be an environment variable name, not a secret: {v}")
        return v


class GateStatus(str, Enum):
    """Deterministic status outcome of pre-probe gate evaluation."""
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


class GateName(str, Enum):
    """Individual gates evaluated during access and cost checks."""
    ACCESS_MODE_GATE = "ACCESS_MODE_GATE"
    BILLING_STATUS_GATE = "BILLING_STATUS_GATE"
    CARD_REQUIRED_GATE = "CARD_REQUIRED_GATE"
    CREDENTIAL_GATE = "CREDENTIAL_GATE"
    USER_APPROVAL_GATE = "USER_APPROVAL_GATE"
    VERIFICATION_STATE_GATE = "VERIFICATION_STATE_GATE"


class GateDecision(BaseModel):
    """Structured, deterministic decision produced by pre-probe safety gates.
    Inviolable invariant: Passing the pre-probe gate NEVER makes a candidate ROUTABLE."""
    candidate_id: str = Field(description="Identifier of candidate being evaluated")
    status: GateStatus = Field(description="ALLOWED if all pre-probe checks pass, BLOCKED otherwise")
    reason: str = Field(description="Human-readable explanation of the gate decision")
    failed_gates: List[GateName] = Field(default_factory=list, description="List of gates that rejected the candidate")
    passed_gates: List[GateName] = Field(default_factory=list, description="List of gates that cleared the candidate")
    may_proceed_to_probe: bool = Field(description="Whether harness is permitted to initiate a future live probe")
    requires_user_approval: bool = Field(description="Whether explicit operator approval is required before probing")
    is_routable: bool = Field(default=False, description="Guaranteed False: Pre-probe gating never grants routing authority")
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp when gate was evaluated"
    )
    gate_details: Dict[str, Any] = Field(default_factory=dict, description="Fine-grained gate evaluation telemetry without secrets")


class ProbeStatus(str, Enum):
    """Normalized status outcome of a provider probe."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ProbeRequest(BaseModel):
    """Structured request for executing a controlled capability probe against a candidate."""
    probe_id: str = Field(
        default_factory=lambda: f"probe_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this probe execution"
    )
    candidate_id: str = Field(description="Identifier of candidate being probed")
    prompt: str = Field(description="Input prompt or query for the probe")
    timeout_seconds: float = Field(default=30.0, description="Execution timeout in seconds")
    expected_output_format: Optional[str] = Field(
        default=None,
        description="Expected format or schema contract (e.g. 'text', 'json', 'search_results')"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary reproduction metadata without secrets")

    @field_validator("metadata")
    @classmethod
    def validate_no_secrets_in_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        for k in v.keys():
            k_lower = str(k).lower()
            if any(s in k_lower for s in ["key", "secret", "token", "password", "auth"]):
                raise ValueError(f"Secret-like key '{k}' forbidden in probe metadata")
        return v


class ProbeResult(BaseModel):
    """Structured, normalized result of a candidate probe execution.
    Contains zero secrets or sensitive credentials."""
    probe_id: str = Field(description="Unique identifier matching the probe request")
    candidate_id: str = Field(description="Identifier of candidate probed")
    status: ProbeStatus = Field(description="Overall probe status: SUCCESS, FAILED, or BLOCKED")
    success: bool = Field(description="True if provider call succeeded and contract was satisfied")
    attempted: bool = Field(description="Whether the provider network/daemon call was attempted")
    contract_satisfied: bool = Field(default=False, description="Whether response satisfied the expected transport/schema contract")
    normalized_response: Optional[str] = Field(default=None, description="Sanitized, normalized response body or text")
    raw_response_snippet: Optional[str] = Field(default=None, description="Sanitized snippet of raw response body for debugging")
    error_classification: Optional[FailureClassification] = Field(default=None, description="Classified failure details if failed or blocked")
    latency_ms: float = Field(default=0.0, description="Measured execution duration in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of execution"
    )
    usage_metadata: Optional[Dict[str, int]] = Field(default=None, description="Non-secret usage tokens/metrics reported by provider")
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Execution reproduction telemetry with zero secrets")


class BenchmarkFixture(BaseModel):
    """A declarative, versioned test fixture defining input and expected behavior."""
    fixture_id: str = Field(description="Unique identifier for the benchmark fixture")
    version: str = Field(default="1.0.0", description="SemVer version of the fixture")
    domain: str = Field(default="generic", description="Target domain (e.g. generic, career_research)")
    objective: str = Field(description="Human-readable objective of this evaluation fixture")
    input_payload: Dict[str, Any] = Field(description="Standardized input data for the probe")
    required_behavior: List[str] = Field(default_factory=list, description="List of expected behavioral invariants")
    expected_structure: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema or structural expectations")
    evaluation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Deterministic scoring and validation rules")
    allowed_variance: Optional[Dict[str, Any]] = Field(default=None, description="Permissible non-semantic variance")
    failure_conditions: List[str] = Field(default_factory=list, description="Explicit failure triggers")


class MockProviderResult(BaseModel):
    """Controlled mock response representing an isolated provider probe output."""
    result_id: str = Field(description="Unique identifier for this mock execution")
    candidate_id: str = Field(description="Candidate identifier being mocked")
    fixture_id: str = Field(description="Fixture identifier executed against")
    http_status: int = Field(default=200, description="Mock HTTP response status code")
    raw_output: Optional[str] = Field(default=None, description="Raw text response body")
    parsed_json: Optional[Dict[str, Any]] = Field(default=None, description="Parsed JSON if available")
    latency_ms: int = Field(default=100, description="Simulated execution latency in milliseconds")
    token_metrics: Optional[Dict[str, int]] = Field(default=None, description="Simulated token consumption")
    error_info: Optional[Dict[str, Any]] = Field(default=None, description="Error details if simulated call failed")
    is_mock: bool = Field(default=True, description="Safety flag confirming this is not a live call")


class EvaluationResult(BaseModel):
    """Structured result of evaluating a provider result against a fixture.
    100% deterministic and contains zero subjective or natural-language scores."""
    evaluation_id: str = Field(description="Unique identifier for this evaluation run")
    fixture_id: str = Field(description="Identifier of the evaluated fixture")
    fixture_version: str = Field(description="Version of the evaluated fixture")
    candidate_id: str = Field(description="Candidate evaluated")
    status: EvaluationStatus = Field(description="Final outcome: PASS, FAIL, or REQUIRES_VERIFICATION")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Quantitative deterministic metrics (0.0 to 1.0, latencies)")
    failure_classification: Optional[FailureClassification] = Field(default=None, description="Categorized failure details if status is FAIL")
    rule_evaluations: List[Dict[str, Any]] = Field(default_factory=list, description="Fine-grained check results for each rule")
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of evaluation"
    )
    is_deterministic: bool = Field(default=True, description="Confirms result was produced without stochastic components")


class ProvenanceState(str, Enum):
    """Explicit provenance classification distinguishing system retrieval from model claims."""
    RETRIEVED_BY_SYSTEM = "RETRIEVED_BY_SYSTEM"
    MODEL_CLAIM_ONLY = "MODEL_CLAIM_ONLY"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"


class ProvenanceRecord(BaseModel):
    """Tracks explicit distinction between model-asserted citations and system-retrieved sources."""
    model_claimed_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources/citations claimed by the model in its response text",
    )
    system_retrieved_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources actually fetched by the system retrieval pipeline",
    )
    retrieval_provider: Optional[str] = Field(default=None, description="Operating retrieval tool/provider ID")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Safe retrieval telemetry without secrets")
    source_identifiers: List[str] = Field(default_factory=list, description="Canonical identifiers or URLs of valid sources")
    evidence_text: Optional[str] = Field(default=None, description="Safe snippet of supporting text if permitted")
    provenance_status: ProvenanceState = Field(
        default=ProvenanceState.NOT_AVAILABLE,
        description="Current verification standing of source provenance",
    )


class BenchmarkTask(BaseModel):
    """A versioned benchmark task specification defining input prompt, capability, and contract."""
    benchmark_id: str = Field(description="Unique benchmark suite identifier")
    task_id: str = Field(description="Unique task identifier within benchmark")
    task_version: str = Field(default="1.0.0", description="SemVer version of task fixture")
    task_category: str = Field(description="Capability category (e.g. structured_extraction, claim_identification)")
    task_description: str = Field(description="Human-readable description of benchmark task")
    input_prompt: str = Field(description="Fixed deterministic input prompt for candidate evaluation")
    expected_capability: str = Field(description="Target capability being tested")
    expected_output_contract: Dict[str, Any] = Field(description="Structural or schema contract required of candidate")
    domain: Dict[str, Any] = Field(description="Domain metadata (name, source_context, version)")
    evaluation_dimensions: List[str] = Field(default_factory=list, description="Explicit dimensions evaluated")
    safety_notes: Optional[str] = Field(default=None, description="Notes on isolation or safety constraints")


class BenchmarkEvidenceRecord(BaseModel):
    """Structured, reproducible evidence record generated from a controlled benchmark task run.
    Contains zero secrets, zero subjective ratings, and zero ranking scores."""
    evidence_id: str = Field(description="Unique identifier for this evidence record")
    benchmark_id: str = Field(description="Benchmark suite identifier")
    task_id: str = Field(description="Task identifier within benchmark suite")
    task_version: str = Field(description="Version of task fixture executed")
    task_category: str = Field(description="Category of capability benchmarked")
    candidate_id: str = Field(description="Identifier of candidate probed")
    probe_id: str = Field(description="Underlying A4.4 probe identifier")
    execution_status: ProbeStatus = Field(description="Outcome status: SUCCESS, FAILED, or BLOCKED")
    contract_satisfied: bool = Field(description="Whether candidate met expected output contract")
    sanitized_response: Optional[str] = Field(default=None, description="Sanitized candidate response text")
    evaluation_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Objective deterministic evaluations (format compliance, rule checks, etc.)",
    )
    failure_classification: Optional[FailureClassification] = Field(
        default=None,
        description="Structured failure categorization if execution failed or was blocked",
    )
    latency_ms: float = Field(default=0.0, description="Transport round-trip duration in milliseconds")
    usage_metadata: Optional[Dict[str, int]] = Field(default=None, description="Non-secret token or call metrics reported by provider")
    provenance_metadata: ProvenanceRecord = Field(
        default_factory=ProvenanceRecord,
        description="Provenance audit tracking system retrieval vs model assertion",
    )
    input_determinism: bool = Field(
        default=True,
        description="Guaranteed True: Input task payload was fixed and deterministic",
    )
    output_determinism: bool = Field(
        default=False,
        description="Provider-dependent: Whether provider output is guaranteed deterministic",
    )
    executed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC execution timestamp",
    )


