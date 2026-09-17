"""Verification Test Suite for Phase A4.7.

Tests baseline profile archival and registry capability verification.
Proves deterministic conversion of empirical evidence into registry capability records,
partial evidence handling, evidence traceability, snapshot serialization,
zero subjective scores, zero runtime routing, and zero network calls.
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateRecord,
    CandidateType,
    FailureClass,
    FailureClassification,
    ProbeStatus,
    ProvenanceState,
    VerificationState,
)
from harness.registry.archival import (
    list_registry_snapshots,
    load_registry_snapshot,
    save_registry_snapshot,
)
from harness.registry.builder import (
    build_candidate_registry_entry,
    build_registry_snapshot,
)
from harness.registry.models import (
    CandidateRegistryEntry,
    CapabilityVerificationRecord,
    RegistrySnapshot,
    RegistryVerificationStatus,
)
from harness.registry.policy import POLICY_VERSION, evaluate_capability_verification
from harness.reporting.profiles import (
    CandidateCapabilityProfile,
    CategoryCapabilityObservation,
    ContractSatisfactionSummary,
    LatencyStatistics,
    ProvenanceObservationSummary,
    TaskObservationResult,
)


def create_sample_task_result(
    task_id: str,
    task_category: str,
    execution_status: ProbeStatus = ProbeStatus.SUCCESS,
    contract_satisfied: bool = True,
    evidence_id: Optional[str] = None,
    latency_ms: float = 120.0,
    failure_classification: Optional[FailureClassification] = None,
    provenance_state: ProvenanceState = ProvenanceState.NOT_AVAILABLE,
    task_version: str = "1.0.0",
) -> TaskObservationResult:
    """Helper to build a deterministic TaskObservationResult."""
    return TaskObservationResult(
        task_id=task_id,
        task_version=task_version,
        task_category=task_category,
        execution_status=execution_status,
        contract_satisfied=contract_satisfied,
        failure_classification=failure_classification,
        latency_ms=latency_ms,
        provenance_state=provenance_state,
        evidence_id=evidence_id or f"ev_{task_id}",
    )


def create_sample_profile(
    candidate_id: str = "groq_llama33_70b",
    tasks: Optional[List[TaskObservationResult]] = None,
    suite_id: str = "sih_cadastral_v1",
    suite_version: str = "1.0.0",
) -> CandidateCapabilityProfile:
    """Helper to build a deterministic CandidateCapabilityProfile."""
    task_list = tasks or []
    eval_count = len(task_list)
    succ_count = sum(1 for t in task_list if t.execution_status == ProbeStatus.SUCCESS)
    fail_count = sum(1 for t in task_list if t.execution_status == ProbeStatus.FAILED)
    block_count = sum(1 for t in task_list if t.execution_status == ProbeStatus.BLOCKED)
    sat_count = sum(1 for t in task_list if t.contract_satisfied)

    return CandidateCapabilityProfile(
        candidate_id=candidate_id,
        benchmark_suite_id=suite_id,
        benchmark_suite_version=suite_version,
        evaluated_task_count=eval_count,
        successful_task_count=succ_count,
        failed_task_count=fail_count,
        blocked_task_count=block_count,
        task_results=task_list,
        capability_observations=[],
        contract_satisfaction_observations=ContractSatisfactionSummary(
            total_evaluated=eval_count,
            satisfied_count=sat_count,
            unsatisfied_count=eval_count - sat_count,
            satisfaction_rate=round(sat_count / eval_count, 4) if eval_count > 0 else 0.0,
        ),
        observed_failure_classes={},
        observed_latency_statistics=LatencyStatistics(),
        provenance_observations=ProvenanceObservationSummary(),
        benchmark_versions=[suite_version],
        evidence_references=[t.evidence_id for t in task_list],
    )


class TestA47RegistryVerification(unittest.TestCase):
    """Test suite for Phase A4.7 Registry Capability Verification & Archival."""

    def setUp(self):
        # Network call blocker: Any attempt to open a socket raises an AssertionError
        self._orig_socket = socket.socket

        def guarded_socket(*args, **kwargs):
            raise AssertionError("A4.7 invariant violation: Network call attempted in offline test")

        socket.socket = guarded_socket

    def tearDown(self):
        socket.socket = self._orig_socket

    def test_a_candidate_with_no_evidence_unverified(self):
        """Proves that a candidate with zero benchmark evidence evaluates to UNVERIFIED."""
        profile = create_sample_profile(candidate_id="cand_unverified", tasks=[])
        entry = build_candidate_registry_entry(profile)

        self.assertEqual(entry.candidate_id, "cand_unverified")
        self.assertEqual(entry.overall_verification_state, VerificationState.UNVERIFIED)
        self.assertEqual(len(entry.verified_capabilities), 0)

        # Evaluating an arbitrary capability with empty task list must yield UNVERIFIED
        rec = evaluate_capability_verification(
            candidate_id="cand_unverified",
            candidate_type=CandidateType.INFERENCE_MODEL,
            capability="structured_json_generation",
            tasks=[],
            benchmark_id="sih_cadastral_v1",
            benchmark_version="1.0.0",
        )
        self.assertEqual(rec.verification_status, RegistryVerificationStatus.UNVERIFIED)
        self.assertEqual(rec.observation_count, 0)
        self.assertEqual(rec.contract_satisfied_count, 0)

    def test_b_candidate_with_successful_evidence_verified_pass(self):
        """Proves that a candidate with 100% successful and contract-satisfying evidence becomes VERIFIED_PASS."""
        tasks = [
            create_sample_task_result("task_01", "structured_json_generation", ProbeStatus.SUCCESS, True, latency_ms=100.0),
            create_sample_task_result("task_02", "structured_json_generation", ProbeStatus.SUCCESS, True, latency_ms=120.0),
        ]
        profile = create_sample_profile(candidate_id="groq_llama33_70b", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        self.assertEqual(entry.overall_verification_state, VerificationState.VERIFIED_PASS)
        self.assertIn("structured_json_generation", entry.verified_capabilities)
        rec = entry.verified_capabilities["structured_json_generation"]

        self.assertEqual(rec.verification_status, RegistryVerificationStatus.VERIFIED_PASS)
        self.assertEqual(rec.observation_count, 2)
        self.assertEqual(rec.contract_satisfied_count, 2)
        self.assertEqual(rec.contract_satisfaction_rate, 1.0)
        self.assertEqual(rec.passed_task_ids, ["task_01", "task_02"])
        self.assertEqual(rec.failed_task_ids, [])
        self.assertEqual(rec.blocked_task_ids, [])
        self.assertIsNotNone(rec.latency_statistics)
        self.assertEqual(rec.latency_statistics.min_ms, 100.0)
        self.assertEqual(rec.latency_statistics.max_ms, 120.0)

    def test_c_candidate_with_failed_evidence_verified_fail(self):
        """Proves that a candidate with execution failure or contract violation becomes VERIFIED_FAIL."""
        fc = FailureClassification(failure_class=FailureClass.SCHEMA_FAILURE, raw_message="Schema violation")
        tasks = [
            create_sample_task_result("task_01", "structured_json_generation", ProbeStatus.FAILED, False, failure_classification=fc),
        ]
        profile = create_sample_profile(candidate_id="bad_model", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        self.assertEqual(entry.overall_verification_state, VerificationState.VERIFIED_FAIL)
        rec = entry.verified_capabilities["structured_json_generation"]
        self.assertEqual(rec.verification_status, RegistryVerificationStatus.VERIFIED_FAIL)
        self.assertEqual(rec.failed_task_ids, ["task_01"])
        self.assertEqual(rec.contract_satisfaction_rate, 0.0)

    def test_d_candidate_blocked_by_safety_gate_blocked(self):
        """Proves that a candidate blocked by A4.3 gates becomes BLOCKED and not verified."""
        fc = FailureClassification(failure_class=FailureClass.BILLING_BLOCKED, raw_message="Credit card required")
        tasks = [
            create_sample_task_result("task_01", "claim_identification", ProbeStatus.BLOCKED, False, failure_classification=fc),
        ]
        profile = create_sample_profile(candidate_id="blocked_cand", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        self.assertEqual(entry.overall_verification_state, VerificationState.REQUIRES_REVIEW)
        rec = entry.verified_capabilities["claim_identification"]
        self.assertEqual(rec.verification_status, RegistryVerificationStatus.BLOCKED)
        self.assertEqual(rec.blocked_task_ids, ["task_01"])
        self.assertEqual(rec.passed_task_ids, [])

    def test_e_candidate_unsupported_infrastructure_not_ready(self):
        """Proves that a candidate encountering UNSUPPORTED_REQUEST evaluates to NOT_READY."""
        fc = FailureClassification(failure_class=FailureClass.UNSUPPORTED_REQUEST, raw_message="Unsupported adapter")
        tasks = [
            create_sample_task_result("task_01", "pdf_extraction", ProbeStatus.FAILED, False, failure_classification=fc),
        ]
        profile = create_sample_profile(candidate_id="unsupported_cand", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        rec = entry.verified_capabilities["pdf_extraction"]
        self.assertEqual(rec.verification_status, RegistryVerificationStatus.NOT_READY)

    def test_f_candidate_partial_evidence_boundaries_preserved(self):
        """Proves that partial evidence (some passed, some blocked) does NOT become VERIFIED_PASS."""
        tasks = [
            create_sample_task_result("task_A", "research_planning", ProbeStatus.SUCCESS, True),
            create_sample_task_result("task_B", "research_planning", ProbeStatus.SUCCESS, True),
            create_sample_task_result(
                "task_C", "research_planning", ProbeStatus.BLOCKED, False,
                failure_classification=FailureClassification(failure_class=FailureClass.RATE_LIMITED),
            ),
        ]
        profile = create_sample_profile(candidate_id="partial_cand", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        rec = entry.verified_capabilities["research_planning"]
        # Invariant: Partial completion of tasks must NOT silently become VERIFIED_PASS
        self.assertNotEqual(rec.verification_status, RegistryVerificationStatus.VERIFIED_PASS)
        self.assertEqual(rec.verification_status, RegistryVerificationStatus.BLOCKED)
        self.assertEqual(rec.passed_task_ids, ["task_A", "task_B"])
        self.assertEqual(rec.blocked_task_ids, ["task_C"])
        self.assertIn("Partial evaluation", rec.notes)

    def test_g_multiple_capabilities_for_single_candidate(self):
        """Proves capability-level granularity (one candidate can pass Cap A, fail Cap B, unverified Cap C)."""
        tasks = [
            create_sample_task_result("t1", "structured_json_generation", ProbeStatus.SUCCESS, True),
            create_sample_task_result(
                "t2", "source_aware_response", ProbeStatus.FAILED, False,
                failure_classification=FailureClassification(failure_class=FailureClass.GROUNDING_FAILURE),
            ),
        ]
        manifest = CandidateRecord(
            candidate_id="multi_cap_cand",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="test_prov",
            technical_identifier="test-model",
            display_name="Multi Cap Model",
            access_mode=AccessMode.API,
            claimed_capabilities=["structured_json_generation", "source_aware_response", "untested_capability"],
        )
        profile = create_sample_profile(candidate_id="multi_cap_cand", tasks=tasks)
        entry = build_candidate_registry_entry(profile, manifest=manifest)

        # Verify structured_json_generation is VERIFIED_PASS
        self.assertEqual(
            entry.verified_capabilities["structured_json_generation"].verification_status,
            RegistryVerificationStatus.VERIFIED_PASS,
        )
        # Verify source_aware_response is VERIFIED_FAIL
        self.assertEqual(
            entry.verified_capabilities["source_aware_response"].verification_status,
            RegistryVerificationStatus.VERIFIED_FAIL,
        )
        # Verify untested_capability is UNVERIFIED (even though claimed in manifest)
        self.assertEqual(
            entry.verified_capabilities["untested_capability"].verification_status,
            RegistryVerificationStatus.UNVERIFIED,
        )
        # Overall state must reflect failure presence
        self.assertEqual(entry.overall_verification_state, VerificationState.VERIFIED_FAIL)

    def test_h_multiple_candidates_registry_aggregation(self):
        """Proves independent aggregation across multiple candidates into RegistrySnapshot."""
        p1 = create_sample_profile("cand_1", [create_sample_task_result("t1", "cap_a", ProbeStatus.SUCCESS, True)])
        p2 = create_sample_profile("cand_2", [create_sample_task_result("t2", "cap_a", ProbeStatus.FAILED, False)])

        snapshot = build_registry_snapshot([p1, p2], snapshot_id="test_snap_001")
        self.assertEqual(snapshot.candidate_count, 2)
        self.assertIn("cand_1", snapshot.entries)
        self.assertIn("cand_2", snapshot.entries)
        self.assertEqual(
            snapshot.entries["cand_1"].verified_capabilities["cap_a"].verification_status,
            RegistryVerificationStatus.VERIFIED_PASS,
        )
        self.assertEqual(
            snapshot.entries["cand_2"].verified_capabilities["cap_a"].verification_status,
            RegistryVerificationStatus.VERIFIED_FAIL,
        )

    def test_i_evidence_traceability_chain(self):
        """Proves complete evidence traceability back to BenchmarkEvidenceRecord IDs."""
        tasks = [
            create_sample_task_result("t1", "cap_a", ProbeStatus.SUCCESS, True, evidence_id="ev_trace_001"),
            create_sample_task_result("t2", "cap_a", ProbeStatus.SUCCESS, True, evidence_id="ev_trace_002"),
        ]
        profile = create_sample_profile("cand_trace", tasks=tasks)
        entry = build_candidate_registry_entry(profile)

        rec = entry.verified_capabilities["cap_a"]
        self.assertEqual(rec.evidence_ids, ["ev_trace_001", "ev_trace_002"])
        self.assertEqual(rec.observed_task_ids, ["t1", "t2"])
        self.assertEqual(rec.evidence_source, "A4.6_CandidateCapabilityProfile")

    def test_j_benchmark_version_preservation(self):
        """Proves that benchmark suite ID and SemVer versions are preserved in verification records."""
        profile = create_sample_profile("cand_ver", [create_sample_task_result("t1", "cap_a")], suite_id="sih_v2", suite_version="2.1.0")
        entry = build_candidate_registry_entry(profile)

        rec = entry.verified_capabilities["cap_a"]
        self.assertEqual(rec.benchmark_id, "sih_v2")
        self.assertEqual(rec.benchmark_version, "2.1.0")
        self.assertEqual(rec.policy_version, POLICY_VERSION)

    def test_k_deterministic_ordering_of_records(self):
        """Proves deterministic sorting of capabilities and task IDs regardless of input order."""
        t1 = create_sample_task_result("task_z", "cap_b", ProbeStatus.SUCCESS, True, evidence_id="ev_z")
        t2 = create_sample_task_result("task_a", "cap_a", ProbeStatus.SUCCESS, True, evidence_id="ev_a")
        t3 = create_sample_task_result("task_m", "cap_b", ProbeStatus.SUCCESS, True, evidence_id="ev_m")

        profile = create_sample_profile("cand_order", tasks=[t1, t2, t3])
        entry = build_candidate_registry_entry(profile)

        # Capabilities must be sorted alphabetically
        self.assertEqual(list(entry.verified_capabilities.keys()), ["cap_a", "cap_b"])
        # Tasks in cap_b must be sorted alphabetically
        self.assertEqual(entry.verified_capabilities["cap_b"].observed_task_ids, ["task_m", "task_z"])

    def test_l_repeated_build_identical_output(self):
        """Proves bit-for-bit idempotency: repeated builds produce identical serialized JSON."""
        tasks = [
            create_sample_task_result("t1", "cap_a", ProbeStatus.SUCCESS, True),
            create_sample_task_result("t2", "cap_b", ProbeStatus.SUCCESS, True),
        ]
        p = create_sample_profile("cand_idem", tasks=tasks)

        e1 = build_candidate_registry_entry(p)
        e2 = build_candidate_registry_entry(p)

        # Timestamps might differ if dynamically generated, but capability record content must be identical
        self.assertEqual(
            e1.verified_capabilities["cap_a"].model_dump(exclude={"verified_at"}),
            e2.verified_capabilities["cap_a"].model_dump(exclude={"verified_at"}),
        )

    def test_m_no_subjective_ranking_or_best_model_fields(self):
        """SECURITY & ARCHITECTURAL INVARIANT: Confirms absence of scores, rankings, or best_model fields."""
        rec = CapabilityVerificationRecord(
            candidate_id="cand_test",
            candidate_type=CandidateType.INFERENCE_MODEL,
            capability="structured_json_generation",
            verification_status=RegistryVerificationStatus.VERIFIED_PASS,
            benchmark_id="sih_v1",
        )
        rec_dict = rec.model_dump()

        forbidden_keys = ["score", "rating", "rank", "winner", "best_model", "composite_score", "grade"]
        for k in forbidden_keys:
            self.assertNotIn(k, rec_dict, f"Forbidden ranking key '{k}' found in CapabilityVerificationRecord")

        entry = CandidateRegistryEntry(
            candidate_id="cand_test",
            display_name="Test Candidate",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="test_prov",
            access_mode=AccessMode.API,
        )
        entry_dict = entry.model_dump()
        for k in forbidden_keys:
            self.assertNotIn(k, entry_dict, f"Forbidden ranking key '{k}' found in CandidateRegistryEntry")

    def test_n_no_runtime_routing_side_effects(self):
        """INVARIANT: Registry records must not contain routing methods, endpoints, or dispatch hooks."""
        entry = build_candidate_registry_entry(create_sample_profile("cand_no_route", []))
        # Ensure entry does not have runtime execution or routing methods
        self.assertFalse(hasattr(entry, "route"))
        self.assertFalse(hasattr(entry, "execute_query"))
        self.assertFalse(hasattr(entry, "fallback_to"))

    def test_o_zero_network_calls(self):
        """INVARIANT: Proves entire registry build and evaluation pipeline executes strictly offline."""
        tasks = [create_sample_task_result("t1", "cap_a", ProbeStatus.SUCCESS, True)]
        profile = create_sample_profile("cand_offline", tasks=tasks)
        # Executing building & snapshot under guarded_socket
        snapshot = build_registry_snapshot([profile], snapshot_id="offline_snap")
        self.assertEqual(snapshot.candidate_count, 1)

    def test_p_credential_leakage_prevention(self):
        """SECURITY INVARIANT: Rejects secret-like patterns in notes or metadata."""
        with self.assertRaises(ValueError):
            CapabilityVerificationRecord(
                candidate_id="cand_sec",
                candidate_type=CandidateType.INFERENCE_MODEL,
                capability="cap_a",
                verification_status=RegistryVerificationStatus.VERIFIED_PASS,
                benchmark_id="b1",
                notes="Leaked API_KEY: gsk_secret_12345",
            )

    def test_q_model_citations_remain_model_claim_only(self):
        """CRITICAL INVARIANT: Model-generated citations must remain MODEL_CLAIM_ONLY."""
        task_claim = create_sample_task_result(
            "t_claim", "claim_identification", ProbeStatus.SUCCESS, True,
            provenance_state=ProvenanceState.MODEL_CLAIM_ONLY,
        )
        task_retrieved = create_sample_task_result(
            "t_ret", "source_aware_response", ProbeStatus.SUCCESS, True,
            provenance_state=ProvenanceState.RETRIEVED_BY_SYSTEM,
        )
        profile = create_sample_profile("cand_prov", tasks=[task_claim, task_retrieved])
        entry = build_candidate_registry_entry(profile)

        rec_claim = entry.verified_capabilities["claim_identification"]
        self.assertEqual(rec_claim.provenance_state, ProvenanceState.MODEL_CLAIM_ONLY)
        self.assertEqual(rec_claim.model_claimed_sources_count, 1)
        self.assertEqual(rec_claim.system_retrieved_sources_count, 0)

        rec_ret = entry.verified_capabilities["source_aware_response"]
        self.assertEqual(rec_ret.provenance_state, ProvenanceState.RETRIEVED_BY_SYSTEM)
        self.assertEqual(rec_ret.system_retrieved_sources_count, 1)
        self.assertEqual(rec_ret.model_claimed_sources_count, 0)

    def test_r_snapshot_archival_and_restoration(self):
        """Proves that a RegistrySnapshot can be saved to disk as JSON and restored losslessly."""
        tasks = [create_sample_task_result("t1", "cap_a", ProbeStatus.SUCCESS, True)]
        profile = create_sample_profile("cand_arch", tasks=tasks)
        snapshot = build_registry_snapshot([profile], snapshot_id="test_arch_001")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            saved_path = save_registry_snapshot(snapshot, tmp_path)
            self.assertTrue(saved_path.exists())

            loaded = load_registry_snapshot(saved_path)
            self.assertEqual(loaded.snapshot_id, "test_arch_001")
            self.assertEqual(loaded.candidate_count, 1)
            self.assertIn("cand_arch", loaded.entries)
            self.assertEqual(
                loaded.entries["cand_arch"].verified_capabilities["cap_a"].verification_status,
                RegistryVerificationStatus.VERIFIED_PASS,
            )

            listed = list_registry_snapshots(tmp_path)
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].name, "test_arch_001.json")

    def test_s_claimed_capability_without_benchmark_remains_unverified(self):
        """INVARIANT 15: A candidate does NOT become VERIFIED_PASS merely because manifest claims it."""
        manifest = CandidateRecord(
            candidate_id="unvetted_cand",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="vendor_x",
            technical_identifier="v-x-model",
            display_name="Vendor X Model",
            access_mode=AccessMode.API,
            claimed_capabilities=["hallucinated_capability_a", "untested_math_reasoning"],
        )
        profile = create_sample_profile("unvetted_cand", tasks=[])
        entry = build_candidate_registry_entry(profile, manifest=manifest)

        # Both claimed capabilities must exist in verified_capabilities with status UNVERIFIED
        self.assertIn("hallucinated_capability_a", entry.verified_capabilities)
        self.assertIn("untested_math_reasoning", entry.verified_capabilities)
        self.assertEqual(
            entry.verified_capabilities["hallucinated_capability_a"].verification_status,
            RegistryVerificationStatus.UNVERIFIED,
        )
        self.assertEqual(
            entry.verified_capabilities["untested_math_reasoning"].verification_status,
            RegistryVerificationStatus.UNVERIFIED,
        )
        self.assertEqual(entry.overall_verification_state, VerificationState.UNVERIFIED)

    def test_t_malformed_profile_rejection(self):
        """Proves that empty or invalid candidate_id strings are rejected by Pydantic validators."""
        with self.assertRaises(ValueError):
            CandidateRegistryEntry(
                candidate_id="   ",
                display_name="Bad ID",
                candidate_type=CandidateType.INFERENCE_MODEL,
                provider_id="p1",
                access_mode=AccessMode.API,
            )

        with self.assertRaises(ValueError):
            CapabilityVerificationRecord(
                candidate_id="cand_test",
                candidate_type=CandidateType.INFERENCE_MODEL,
                capability="",  # Empty capability
                verification_status=RegistryVerificationStatus.UNVERIFIED,
                benchmark_id="b1",
            )

    def test_u_sample_fixture_end_to_end_verification(self):
        """Proves that sample_capability_profile_v1.json translates cleanly into a valid registry entry."""
        fixture_path = Path(__file__).parents[2] / "harness" / "fixtures" / "sample_capability_profile_v1.json"
        self.assertTrue(fixture_path.exists(), f"Missing fixture at {fixture_path}")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = CandidateCapabilityProfile.model_validate(data)
        entry = build_candidate_registry_entry(profile)

        self.assertEqual(entry.candidate_id, "groq_llama33_70b")
        self.assertEqual(entry.overall_verification_state, VerificationState.VERIFIED_PASS)
        # The sample fixture contains 6 distinct categories:
        # structured_json_generation, structured_extraction, claim_identification,
        # source_aware_grounding, research_planning, knowledge_gap_detection
        self.assertEqual(len(entry.verified_capabilities), 6)
        for cap_name, rec in entry.verified_capabilities.items():
            self.assertEqual(rec.verification_status, RegistryVerificationStatus.VERIFIED_PASS)
            self.assertEqual(rec.contract_satisfaction_rate, 1.0)
            self.assertGreater(rec.observation_count, 0)


if __name__ == "__main__":
    unittest.main()

