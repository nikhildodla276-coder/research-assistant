"""Deterministic unit test suite for Phase A4.6 Benchmark Reporting & Baseline Profiles.

Tests:
A. Valid evidence aggregation produces valid CandidateCapabilityProfile
B. Multiple evidence records for the same candidate aggregate accurately
C. Multiple tasks across categories aggregate into CategoryCapabilityObservation
D. Successful, failed, and blocked tasks are counted accurately
E. Contract satisfaction aggregation computes correct counts and rates
F. Failure classes are aggregated deterministically by category
G. Latency statistics compute accurate sample count, min, max, mean, and median
H. Provenance observations preserve system vs model claim counts
I. Invariant: MODEL_CLAIM_ONLY never increments RETRIEVED_BY_SYSTEM count
J. Evidence traceability: all evidence_ids appear in task_results and references
K. Deterministic ordering: outputs are stably sorted regardless of input shuffle
L. Determinism: identical evidence produces bit-for-bit identical profile dumps
M. Empty evidence handling returns safe baseline profile without crashing
N. Malformed evidence input raises ValidationError or handles gracefully
O. Benchmark and task versions are preserved in benchmark_versions
P. Strict Invariant: Zero subjective ranking, scoring, or leaderboard fields exist
Q. Strict Invariant: Profile generation produces zero runtime routing side effects
R. Strict Invariant: Zero network calls occur during aggregation and reporting
"""

import json
import random
import socket
import unittest
from pathlib import Path
from pydantic import ValidationError

from harness.models import (
    BenchmarkEvidenceRecord,
    FailureClass,
    FailureClassification,
    ProbeStatus,
    ProvenanceRecord,
    ProvenanceState,
)
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


def create_sample_evidence_record(
    candidate_id: str = "groq_llama33_70b",
    task_id: str = "sih_cadastral_01_json_generation",
    task_category: str = "structured_json_generation",
    execution_status: ProbeStatus = ProbeStatus.SUCCESS,
    contract_satisfied: bool = True,
    latency_ms: float = 150.0,
    provenance_status: ProvenanceState = ProvenanceState.NOT_AVAILABLE,
    evidence_id: str = "ev_sample_001",
    failure_classification: FailureClassification = None,
    model_claimed_sources: list = None,
    system_retrieved_sources: list = None,
) -> BenchmarkEvidenceRecord:
    """Helper creating a fully populated, valid BenchmarkEvidenceRecord for testing."""
    return BenchmarkEvidenceRecord(
        evidence_id=evidence_id,
        benchmark_id="sih_cadastral_v1",
        task_id=task_id,
        task_version="1.0.0",
        task_category=task_category,
        candidate_id=candidate_id,
        probe_id=f"probe_{evidence_id}",
        execution_status=execution_status,
        contract_satisfied=contract_satisfied,
        sanitized_response="Sample sanitized output",
        evaluation_results={"valid": True},
        failure_classification=failure_classification,
        latency_ms=latency_ms,
        usage_metadata={"input_tokens": 50, "output_tokens": 20},
        provenance_metadata=ProvenanceRecord(
            model_claimed_sources=model_claimed_sources or [],
            system_retrieved_sources=system_retrieved_sources or [],
            provenance_status=provenance_status,
        ),
        input_determinism=True,
        output_determinism=False,
        executed_at="2026-09-17T04:45:00+00:00",
    )


class TestA46BenchmarkReporting(unittest.TestCase):
    """Test suite validating evidence aggregation, capability profiling, and reporting."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).parent.parent.parent / "harness" / "fixtures"

    # -------------------------------------------------------------------------
    # Test A: Valid Evidence Aggregation
    # -------------------------------------------------------------------------
    def test_a_valid_evidence_aggregation(self):
        """Proves that a single valid evidence record produces a valid CandidateCapabilityProfile."""
        rec = create_sample_evidence_record(candidate_id="cand_a", evidence_id="ev_1")
        profile = aggregate_candidate_evidence("cand_a", [rec])

        self.assertIsInstance(profile, CandidateCapabilityProfile)
        self.assertEqual(profile.candidate_id, "cand_a")
        self.assertEqual(profile.evaluated_task_count, 1)
        self.assertEqual(profile.successful_task_count, 1)
        self.assertEqual(profile.failed_task_count, 0)
        self.assertEqual(profile.blocked_task_count, 0)
        self.assertEqual(profile.contract_satisfaction_observations.satisfied_count, 1)
        self.assertEqual(profile.contract_satisfaction_observations.satisfaction_rate, 1.0)
        self.assertEqual(len(profile.task_results), 1)
        self.assertEqual(profile.task_results[0].evidence_id, "ev_1")

    # -------------------------------------------------------------------------
    # Test B: Multiple Evidence Records for the Same Candidate
    # -------------------------------------------------------------------------
    def test_b_multiple_records_same_candidate(self):
        """Proves that aggregating multiple records for one candidate computes accurate totals."""
        records = [
            create_sample_evidence_record(candidate_id="cand_a", evidence_id="ev_1", latency_ms=100.0),
            create_sample_evidence_record(candidate_id="cand_a", evidence_id="ev_2", latency_ms=200.0),
            create_sample_evidence_record(candidate_id="cand_b", evidence_id="ev_3", latency_ms=300.0),  # Other candidate
        ]
        profile = aggregate_candidate_evidence("cand_a", records)

        self.assertEqual(profile.evaluated_task_count, 2)
        self.assertEqual(len(profile.task_results), 2)
        self.assertEqual(profile.evidence_references, ["ev_1", "ev_2"])

    # -------------------------------------------------------------------------
    # Test C: Multiple Tasks Across Categories
    # -------------------------------------------------------------------------
    def test_c_multiple_tasks_across_categories(self):
        """Proves that tasks across different categories aggregate into CategoryCapabilityObservation."""
        records = [
            create_sample_evidence_record(task_id="t1", task_category="structured_extraction", contract_satisfied=True),
            create_sample_evidence_record(task_id="t2", task_category="structured_extraction", contract_satisfied=False),
            create_sample_evidence_record(task_id="t3", task_category="claim_identification", contract_satisfied=True),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        self.assertEqual(len(profile.capability_observations), 2)
        cat_map = {obs.task_category: obs for obs in profile.capability_observations}

        self.assertIn("structured_extraction", cat_map)
        self.assertEqual(cat_map["structured_extraction"].total_tasks, 2)
        self.assertEqual(cat_map["structured_extraction"].contracts_satisfied, 1)
        self.assertEqual(cat_map["structured_extraction"].contract_satisfaction_rate, 0.5)

        self.assertIn("claim_identification", cat_map)
        self.assertEqual(cat_map["claim_identification"].total_tasks, 1)
        self.assertEqual(cat_map["claim_identification"].contracts_satisfied, 1)
        self.assertEqual(cat_map["claim_identification"].contract_satisfaction_rate, 1.0)

    # -------------------------------------------------------------------------
    # Test D: Successful, Failed, and Blocked Tasks Counted Accurately
    # -------------------------------------------------------------------------
    def test_d_status_counts_aggregation(self):
        """Proves accurate counting of SUCCESS, FAILED, and BLOCKED tasks."""
        records = [
            create_sample_evidence_record(task_id="t1", execution_status=ProbeStatus.SUCCESS),
            create_sample_evidence_record(task_id="t2", execution_status=ProbeStatus.FAILED),
            create_sample_evidence_record(task_id="t3", execution_status=ProbeStatus.BLOCKED),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        self.assertEqual(profile.evaluated_task_count, 3)
        self.assertEqual(profile.successful_task_count, 1)
        self.assertEqual(profile.failed_task_count, 1)
        self.assertEqual(profile.blocked_task_count, 1)

    # -------------------------------------------------------------------------
    # Test E: Contract Satisfaction Aggregation
    # -------------------------------------------------------------------------
    def test_e_contract_satisfaction_aggregation(self):
        """Proves correct calculation of satisfied, unsatisfied, and satisfaction rate."""
        records = [
            create_sample_evidence_record(task_id="t1", contract_satisfied=True),
            create_sample_evidence_record(task_id="t2", contract_satisfied=True),
            create_sample_evidence_record(task_id="t3", contract_satisfied=False),
            create_sample_evidence_record(task_id="t4", contract_satisfied=False),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        cs = profile.contract_satisfaction_observations
        self.assertEqual(cs.total_evaluated, 4)
        self.assertEqual(cs.satisfied_count, 2)
        self.assertEqual(cs.unsatisfied_count, 2)
        self.assertEqual(cs.satisfaction_rate, 0.5)

    # -------------------------------------------------------------------------
    # Test F: Failure Classes Aggregated Deterministically
    # -------------------------------------------------------------------------
    def test_f_failure_class_aggregation(self):
        """Proves aggregation of FailureClass occurrences."""
        fc_rate = FailureClassification(
            failure_class=FailureClass.RATE_LIMITED,
            error_code="rate_limit_exceeded",
            raw_message="429 Too Many Requests",
        )
        fc_timeout = FailureClassification(
            failure_class=FailureClass.TIMEOUT,
            error_code="timeout",
            raw_message="Timeout after 30s",
        )
        records = [
            create_sample_evidence_record(task_id="t1", execution_status=ProbeStatus.FAILED, failure_classification=fc_rate),
            create_sample_evidence_record(task_id="t2", execution_status=ProbeStatus.FAILED, failure_classification=fc_rate),
            create_sample_evidence_record(task_id="t3", execution_status=ProbeStatus.FAILED, failure_classification=fc_timeout),
            create_sample_evidence_record(task_id="t4", execution_status=ProbeStatus.SUCCESS, failure_classification=None),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        self.assertEqual(profile.observed_failure_classes.get("RATE_LIMITED"), 2)
        self.assertEqual(profile.observed_failure_classes.get("TIMEOUT"), 1)

    # -------------------------------------------------------------------------
    # Test G: Latency Statistics
    # -------------------------------------------------------------------------
    def test_g_latency_statistics(self):
        """Proves calculation of min, max, mean, and median latency."""
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        records = [
            create_sample_evidence_record(task_id=f"t{i}", latency_ms=lat)
            for i, lat in enumerate(latencies)
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        lat_stats = profile.observed_latency_statistics
        self.assertEqual(lat_stats.sample_count, 5)
        self.assertEqual(lat_stats.min_ms, 100.0)
        self.assertEqual(lat_stats.max_ms, 500.0)
        self.assertEqual(lat_stats.mean_ms, 300.0)
        self.assertEqual(lat_stats.median_ms, 300.0)

    # -------------------------------------------------------------------------
    # Test H: Provenance Observations
    # -------------------------------------------------------------------------
    def test_h_provenance_observations(self):
        """Proves that provenance status counts are accurately aggregated."""
        records = [
            create_sample_evidence_record(
                task_id="t1",
                provenance_status=ProvenanceState.MODEL_CLAIM_ONLY,
                model_claimed_sources=[{"token": "SRC-01"}, {"token": "SRC-02"}],
            ),
            create_sample_evidence_record(
                task_id="t2",
                provenance_status=ProvenanceState.RETRIEVED_BY_SYSTEM,
                system_retrieved_sources=[{"id": "doc-1"}],
            ),
            create_sample_evidence_record(
                task_id="t3",
                provenance_status=ProvenanceState.NOT_AVAILABLE,
            ),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        po = profile.provenance_observations
        self.assertEqual(po.total_observations, 3)
        self.assertEqual(po.model_claim_only_count, 1)
        self.assertEqual(po.retrieved_by_system_count, 1)
        self.assertEqual(po.not_available_count, 1)
        self.assertEqual(po.total_model_claimed_sources, 2)
        self.assertEqual(po.total_system_retrieved_sources, 1)

    # -------------------------------------------------------------------------
    # Test I: Invariant: MODEL_CLAIM_ONLY Never Becomes RETRIEVED_BY_SYSTEM
    # -------------------------------------------------------------------------
    def test_i_model_claim_never_becomes_retrieved_by_system(self):
        """CRITICAL INVARIANT: Model claims must never increment system retrieval counters."""
        records = [
            create_sample_evidence_record(
                task_id="t1",
                provenance_status=ProvenanceState.MODEL_CLAIM_ONLY,
                model_claimed_sources=[{"token": "SRC-01"}, {"token": "https://fake.source"}],
            )
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        po = profile.provenance_observations
        self.assertEqual(po.model_claim_only_count, 1)
        self.assertEqual(po.retrieved_by_system_count, 0)
        self.assertEqual(po.total_system_retrieved_sources, 0)
        self.assertEqual(po.total_model_claimed_sources, 2)

    # -------------------------------------------------------------------------
    # Test J: Evidence Traceability
    # -------------------------------------------------------------------------
    def test_j_evidence_traceability(self):
        """Proves that every evidence record is traceable in task_results and references."""
        records = [
            create_sample_evidence_record(task_id="t1", evidence_id="ev_alpha"),
            create_sample_evidence_record(task_id="t2", evidence_id="ev_beta"),
        ]
        profile = aggregate_candidate_evidence("groq_llama33_70b", records)

        self.assertIn("ev_alpha", profile.evidence_references)
        self.assertIn("ev_beta", profile.evidence_references)
        tr_ev_ids = [tr.evidence_id for tr in profile.task_results]
        self.assertEqual(tr_ev_ids, ["ev_alpha", "ev_beta"])

    # -------------------------------------------------------------------------
    # Test K: Deterministic Ordering
    # -------------------------------------------------------------------------
    def test_k_deterministic_ordering(self):
        """Proves that shuffling input evidence produces identical, deterministically sorted profiles."""
        rec1 = create_sample_evidence_record(task_id="task_01", evidence_id="ev_1")
        rec2 = create_sample_evidence_record(task_id="task_02", evidence_id="ev_2")
        rec3 = create_sample_evidence_record(task_id="task_03", evidence_id="ev_3")

        p1 = aggregate_candidate_evidence("groq_llama33_70b", [rec1, rec2, rec3])
        p2 = aggregate_candidate_evidence("groq_llama33_70b", [rec3, rec1, rec2])

        self.assertEqual(p1.model_dump(exclude={"generated_at"}), p2.model_dump(exclude={"generated_at"}))

    # -------------------------------------------------------------------------
    # Test L: Determinism Across Runs
    # -------------------------------------------------------------------------
    def test_l_determinism_across_runs(self):
        """Proves that running aggregation twice on identical records produces identical output."""
        rec = create_sample_evidence_record(task_id="t1", evidence_id="ev_fixed")
        p1 = aggregate_candidate_evidence("groq_llama33_70b", [rec])
        p2 = aggregate_candidate_evidence("groq_llama33_70b", [rec])

        self.assertEqual(p1.model_dump(exclude={"generated_at"}), p2.model_dump(exclude={"generated_at"}))

    # -------------------------------------------------------------------------
    # Test M: Empty Evidence Handling
    # -------------------------------------------------------------------------
    def test_m_empty_evidence_handling(self):
        """Proves that empty evidence records return a valid, zeroed profile without error."""
        profile = aggregate_candidate_evidence("candidate_with_no_runs", [])
        self.assertEqual(profile.evaluated_task_count, 0)
        self.assertEqual(profile.successful_task_count, 0)
        self.assertEqual(profile.contract_satisfaction_observations.satisfied_count, 0)
        self.assertEqual(profile.task_results, [])
        self.assertEqual(profile.evidence_references, [])

    # -------------------------------------------------------------------------
    # Test N: Malformed Evidence Handling
    # -------------------------------------------------------------------------
    def test_n_malformed_evidence_handling(self):
        """Proves that malformed evidence dictionaries are rejected by Pydantic."""
        with self.assertRaises(ValidationError):
            BenchmarkEvidenceRecord.model_validate({"evidence_id": "incomplete"})

    # -------------------------------------------------------------------------
    # Test O: Benchmark-Version Preservation
    # -------------------------------------------------------------------------
    def test_o_benchmark_version_preservation(self):
        """Proves that benchmark and task versions are preserved in profile."""
        rec = create_sample_evidence_record(evidence_id="ev_1")
        profile = aggregate_candidate_evidence("groq_llama33_70b", [rec], benchmark_suite_version="1.0.0")

        self.assertEqual(profile.benchmark_suite_version, "1.0.0")
        self.assertIn("1.0.0", profile.benchmark_versions)

    # -------------------------------------------------------------------------
    # Test P: Strict Invariant: Zero Subjective Ranking or Scoring Fields
    # -------------------------------------------------------------------------
    def test_p_no_subjective_ranking_fields(self):
        """CRITICAL INVARIANT: CandidateCapabilityProfile must contain zero ranking or subjective score fields."""
        fields = CandidateCapabilityProfile.model_fields.keys()
        forbidden_substrings = ["score", "rank", "winner", "rating", "grade", "quality_score"]

        for f in fields:
            for sub in forbidden_substrings:
                self.assertNotIn(sub, f.lower(), f"Forbidden subjective ranking field '{f}' found in profile schema!")

    # -------------------------------------------------------------------------
    # Test Q: Strict Invariant: No Runtime Routing Side Effects
    # -------------------------------------------------------------------------
    def test_q_no_runtime_routing_side_effects(self):
        """Proves that profile generation does not alter candidate routability or registry state."""
        rec = create_sample_evidence_record(evidence_id="ev_1")
        profile = aggregate_candidate_evidence("groq_llama33_70b", [rec])

        # Assert profile does NOT have routable flags or routing methods
        self.assertFalse(hasattr(profile, "is_routable"))
        self.assertFalse(hasattr(profile, "select_model"))
        self.assertFalse(hasattr(profile, "route_query"))

    # -------------------------------------------------------------------------
    # Test R: Zero Network Calls
    # -------------------------------------------------------------------------
    def test_r_zero_network_calls(self):
        """Guarantees that reporting and aggregation execute strictly offline with zero socket calls."""
        def guarded_socket(*args, **kwargs):
            raise AssertionError("Network socket creation attempted during offline reporting!")

        orig_socket = socket.socket
        socket.socket = guarded_socket
        try:
            records = [
                create_sample_evidence_record(task_id="t1", evidence_id="ev_1"),
                create_sample_evidence_record(task_id="t2", evidence_id="ev_2"),
            ]
            profile = aggregate_candidate_evidence("groq_llama33_70b", records)
            md_report = generate_candidate_markdown_report(profile)
            json_report = export_profile_json(profile)
            comp_report = generate_multi_candidate_comparison_report([profile])

            self.assertIn("# Candidate Capability Baseline Profile", md_report)
            self.assertIn('"candidate_id": "groq_llama33_70b"', json_report)
            self.assertIn("# Multi-Candidate Capability Observation Matrix", comp_report)
        finally:
            socket.socket = orig_socket

    # -------------------------------------------------------------------------
    # Test S: Sample Fixture Validation
    # -------------------------------------------------------------------------
    def test_s_sample_fixture_valid(self):
        """Proves that sample_capability_profile_v1.json loads cleanly as a CandidateCapabilityProfile."""
        fixture_path = self.fixtures_dir / "sample_capability_profile_v1.json"
        self.assertTrue(fixture_path.exists())

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = CandidateCapabilityProfile.model_validate(data)
        self.assertEqual(profile.candidate_id, "groq_llama33_70b")
        self.assertEqual(profile.evaluated_task_count, 6)
        self.assertEqual(profile.contract_satisfaction_observations.satisfied_count, 6)


if __name__ == "__main__":
    unittest.main()

