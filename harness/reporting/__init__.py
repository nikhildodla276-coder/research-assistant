"""Reporting and Capability Profiling Package for Phase A4.6.

Exports candidate capability profiles, evidence aggregation engines,
and deterministic report generators with strict non-ranking guarantees.
"""

from harness.reporting.aggregator import (
    aggregate_candidate_evidence,
    aggregate_suite_evidence,
)
from harness.reporting.profiles import (
    CandidateCapabilityProfile,
    CategoryCapabilityObservation,
    ContractSatisfactionSummary,
    LatencyStatistics,
    ProvenanceObservationSummary,
    TaskObservationResult,
)
from harness.reporting.report_generator import (
    export_profile_json,
    generate_candidate_markdown_report,
    generate_multi_candidate_comparison_report,
)

__all__ = [
    "TaskObservationResult",
    "LatencyStatistics",
    "CategoryCapabilityObservation",
    "ContractSatisfactionSummary",
    "ProvenanceObservationSummary",
    "CandidateCapabilityProfile",
    "aggregate_candidate_evidence",
    "aggregate_suite_evidence",
    "export_profile_json",
    "generate_candidate_markdown_report",
    "generate_multi_candidate_comparison_report",
]

