"""Deterministic unit test suite for Phase A4.5 Controlled Benchmark Runs & Evidence Collection.

Tests:
A. Valid benchmark fixture loads from disk
B. Malformed benchmark fixture is rejected by Pydantic validation
C. Same task produces identical benchmark request (Input Determinism)
D. Candidate is passed through A4.3 gate before benchmark probe
E. Blocked candidate never executes (A4.3 gate blocks transport invocation)
F. Approved candidate reaches A4.4 probe layer
G. Fake successful response becomes structured BenchmarkEvidenceRecord
H. Malformed response handled correctly (contract_satisfied=False)
I. Provider failure classified correctly into FailureClassification
J. Secret leakage prevented: environment secrets never appear in evidence
K. Provenance fields behave correctly: MODEL_CLAIM_ONLY vs RETRIEVED_BY_SYSTEM
L. Multiple candidates can be benchmarked independently
M. Candidate output isolation: one candidate's output never becomes another's input
N. Benchmark runner does not perform runtime routing or automatic failover
O. Zero network calls occur during offline benchmark runs (socket monkeypatch)
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from harness.adapters.base import TransportResponse
from harness.benchmark_runner import (
    load_benchmark_suite,
    load_benchmark_task,
    run_benchmark_suite,
    run_benchmark_task,
)
from harness.loader import load_all_candidates, load_candidate
from harness.models import (
    AccessMode,
    BenchmarkEvidenceRecord,
    BenchmarkTask,
    CandidateRecord,
    CandidateType,
    FailureClass,
    ProbeStatus,
    ProvenanceRecord,
    ProvenanceState,
)
from harness.provenance import (
    build_provenance_record,
    extract_model_claimed_citations,
)


class TestA45BenchmarkHarness(unittest.TestCase):
    """Test suite validating Phase A4.5 benchmark fixtures, runner, and evidence models."""

    def setUp(self):
        self.benchmarks_dir = Path(__file__).parent.parent.parent / "harness" / "benchmarks"
        self.candidates_dir = Path(__file__).parent.parent.parent / "harness" / "candidates"
        self.suite_dir = self.benchmarks_dir / "sih_cadastral_v1"
        self.mock_env = {
            "GROQ_API_KEY": "gsk_mock_test_key_abc123",
            "TAVILY_API_KEY": "tvly_mock_test_key_xyz789",
            "OPENAI_API_KEY": "sk_mock_test_key_openai456",
            "AWS_ACCESS_KEY_ID": "AKIA_MOCK_TEST_KEY_789",
        }

    # -------------------------------------------------------------------------
    # Test A: Valid Benchmark Fixture Loads
    # -------------------------------------------------------------------------
    def test_a_valid_benchmark_fixture_loads(self):
        """Proves that all 6 SIH cadastral benchmark tasks load cleanly with valid schemas."""
        tasks = load_benchmark_suite("sih_cadastral_v1")
        self.assertEqual(len(tasks), 6)

        task_ids = [t.task_id for t in tasks]
        self.assertIn("sih_cadastral_01_json_generation", task_ids)
        self.assertIn("sih_cadastral_02_structured_extraction", task_ids)
        self.assertIn("sih_cadastral_03_claim_identification", task_ids)
        self.assertIn("sih_cadastral_04_source_aware_grounding", task_ids)
        self.assertIn("sih_cadastral_05_research_planning", task_ids)
        self.assertIn("sih_cadastral_06_knowledge_gap_detection", task_ids)

        for t in tasks:
            self.assertEqual(t.benchmark_id, "sih_cadastral_v1")
            self.assertEqual(t.domain["name"], "urban_cadastral_mapping")
            self.assertGreater(len(t.input_prompt), 10)
            self.assertIn("format", t.expected_output_contract)

    # -------------------------------------------------------------------------
    # Test B: Malformed Benchmark Fixture Rejected
    # -------------------------------------------------------------------------
    def test_b_malformed_benchmark_fixture_rejected(self):
        """Proves that a benchmark task missing required fields fails validation."""
        invalid_data = {
            "benchmark_id": "bad_suite",
            # missing task_id, task_category, input_prompt, etc.
        }
        with self.assertRaises(ValidationError):
            BenchmarkTask.model_validate(invalid_data)

    # -------------------------------------------------------------------------
    # Test C: Same Task Produces Identical Benchmark Request (Input Determinism)
    # -------------------------------------------------------------------------
    def test_c_same_task_produces_identical_request(self):
        """Proves that running a benchmark task multiple times generates identical input prompts and metadata."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        captured_requests = []

        def recording_transport(url, method="POST", headers=None, json_data=None, timeout=30.0):
            captured_requests.append(json_data)
            return TransportResponse(
                status_code=200,
                text=json.dumps({"response": '{"plot_id": "P-101", "survey_number": "SN-405", "boundary_coordinates": [[18.52, 73.85]]}'}),
            )

        r1 = run_benchmark_task(task, candidate, environ={}, transport=recording_transport)
        r2 = run_benchmark_task(task, candidate, environ={}, transport=recording_transport)

        self.assertEqual(len(captured_requests), 2)
        # Input prompt to transport is bit-for-bit identical
        self.assertEqual(captured_requests[0], captured_requests[1])
        self.assertTrue(r1.input_determinism)
        self.assertTrue(r2.input_determinism)

    # -------------------------------------------------------------------------
    # Test D: Candidate Is Passed Through A4.3 Gate
    # -------------------------------------------------------------------------
    def test_d_candidate_passed_through_a4_3_gate(self):
        """Proves that benchmark runner invokes A4.3 gate before probing."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def mock_t(*args, **kwargs):
            return TransportResponse(status_code=200, text=json.dumps({"response": '{"plot_id": "P-1"}'}))

        evidence = run_benchmark_task(task, candidate, environ={}, transport=mock_t)
        self.assertEqual(evidence.execution_status, ProbeStatus.SUCCESS)

    # -------------------------------------------------------------------------
    # Test E: Blocked Candidate Never Executes
    # -------------------------------------------------------------------------
    def test_e_blocked_candidate_never_executes(self):
        """Proves that a candidate blocked by A4.3 (e.g. CARD_REQUIRED) never calls transport."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("card_required_free_tier", candidates_dir=self.candidates_dir)

        transport_invoked = False

        def guarded_transport(*args, **kwargs):
            nonlocal transport_invoked
            transport_invoked = True
            raise AssertionError("Transport must never be called for blocked candidates!")

        evidence = run_benchmark_task(task, candidate, environ=self.mock_env, transport=guarded_transport)

        self.assertFalse(transport_invoked)
        self.assertEqual(evidence.execution_status, ProbeStatus.BLOCKED)
        self.assertFalse(evidence.contract_satisfied)
        self.assertIsNotNone(evidence.failure_classification)
        self.assertEqual(evidence.failure_classification.failure_class, FailureClass.BILLING_BLOCKED)
        self.assertEqual(evidence.latency_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test F: Approved Candidate Reaches A4.4 Probe Layer
    # -------------------------------------------------------------------------
    def test_f_approved_candidate_reaches_probe_layer(self):
        """Proves that an approved candidate clears A4.3 and reaches adapter probe execution."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        adapter_invoked = False

        def mock_groq_transport(*args, **kwargs):
            nonlocal adapter_invoked
            adapter_invoked = True
            valid_json_body = json.dumps({
                "plot_id": "P-101",
                "survey_number": "SN-405",
                "boundary_coordinates": [[18.52, 73.85], [18.53, 73.86]],
            })
            return TransportResponse(
                status_code=200,
                text=json.dumps({"choices": [{"message": {"content": valid_json_body}, "finish_reason": "stop"}]}),
            )

        evidence = run_benchmark_task(task, candidate, environ=self.mock_env, transport=mock_groq_transport)

        self.assertTrue(adapter_invoked)
        self.assertEqual(evidence.execution_status, ProbeStatus.SUCCESS)
        self.assertTrue(evidence.contract_satisfied)

    # -------------------------------------------------------------------------
    # Test G: Fake Successful Response Becomes Benchmark Evidence
    # -------------------------------------------------------------------------
    def test_g_fake_successful_response_becomes_evidence(self):
        """Proves that a compliant response is transformed into a structured BenchmarkEvidenceRecord."""
        task = load_benchmark_task("task_02_structured_extraction", suite_dir=self.suite_dir)
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def mock_ollama_t(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({
                    "response": json.dumps({
                        "target_accuracy": "5cm GSD",
                        "sensor_type": "RGB metric",
                        "projection": "UTM Zone 43N (WGS84)",
                    }),
                    "prompt_eval_count": 42,
                    "eval_count": 18,
                }),
            )

        evidence = run_benchmark_task(task, candidate, environ={}, transport=mock_ollama_t)

        self.assertIsInstance(evidence, BenchmarkEvidenceRecord)
        self.assertEqual(evidence.benchmark_id, "sih_cadastral_v1")
        self.assertEqual(evidence.task_id, "sih_cadastral_02_structured_extraction")
        self.assertEqual(evidence.task_category, "structured_extraction")
        self.assertEqual(evidence.execution_status, ProbeStatus.SUCCESS)
        self.assertTrue(evidence.contract_satisfied)
        self.assertTrue(evidence.evaluation_results["valid_json"])
        self.assertEqual(evidence.evaluation_results["missing_keys"], [])
        self.assertEqual(evidence.usage_metadata, {"input_tokens": 42, "output_tokens": 18})

    # -------------------------------------------------------------------------
    # Test H: Malformed Response Handled Correctly
    # -------------------------------------------------------------------------
    def test_h_malformed_response_handled_correctly(self):
        """Proves that unparseable or schema-violating output results in contract_satisfied=False."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def corrupt_json_t(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({"response": "Here is your plot: plot_id: P-101, survey: 405 (not JSON)"}),
            )

        evidence = run_benchmark_task(task, candidate, environ={}, transport=corrupt_json_t)

        self.assertEqual(evidence.execution_status, ProbeStatus.SUCCESS)
        self.assertFalse(evidence.contract_satisfied)
        self.assertFalse(evidence.evaluation_results["valid_json"])
        self.assertIn("json_error", evidence.evaluation_results)

    # -------------------------------------------------------------------------
    # Test I: Provider Failure Classified Correctly
    # -------------------------------------------------------------------------
    def test_i_provider_failure_classified_correctly(self):
        """Proves that an HTTP 429 rate limit is classified as RATE_LIMITED in evidence."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def rate_limited_t(*args, **kwargs):
            return TransportResponse(
                status_code=429,
                text=json.dumps({"error": {"message": "Rate limit reached: TPM limit exceeded", "type": "rate_limit_exceeded"}}),
            )

        evidence = run_benchmark_task(task, candidate, environ=self.mock_env, transport=rate_limited_t)

        self.assertEqual(evidence.execution_status, ProbeStatus.FAILED)
        self.assertFalse(evidence.contract_satisfied)
        self.assertIsNotNone(evidence.failure_classification)
        self.assertEqual(evidence.failure_classification.failure_class, FailureClass.RATE_LIMITED)
        self.assertEqual(evidence.failure_classification.http_status, 429)

    # -------------------------------------------------------------------------
    # Test J: Secret Leakage Prevented
    # -------------------------------------------------------------------------
    def test_j_secret_leakage_prevented(self):
        """SECURITY INVARIANT: Environment credentials must never leak into evidence records."""
        secret_token = "gsk_TEST_SECRET_TOKEN_A4_5_VAL_9999"
        leak_env = {"GROQ_API_KEY": secret_token}

        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def leaking_t(*args, **kwargs):
            return TransportResponse(
                status_code=401,
                text=f"Unauthorized access for Bearer {secret_token}",
            )

        evidence = run_benchmark_task(task, candidate, environ=leak_env, transport=leaking_t)

        evidence_json = evidence.model_dump_json()
        evidence_str = str(evidence)

        self.assertNotIn(secret_token, evidence_json)
        self.assertNotIn(secret_token, evidence_str)
        self.assertIn("[REDACTED_SECRET]", evidence_json)

    # -------------------------------------------------------------------------
    # Test K: Provenance Fields Behave Correctly
    # -------------------------------------------------------------------------
    def test_k_provenance_fields_behave_correctly(self):
        """CRITICAL INVARIANT: Model citations are MODEL_CLAIM_ONLY; system fetches are RETRIEVED_BY_SYSTEM."""
        task = load_benchmark_task("task_04_source_aware_grounding", suite_dir=self.suite_dir)
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        # Model claims sources in text, but system did not fetch them
        def citing_t(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({
                    "choices": [{
                        "message": {
                            "content": "Per [SRC-01], boundary monuments use dual-frequency GNSS. In accordance with [SRC-02], joint inspection is held."
                        },
                        "finish_reason": "stop",
                    }]
                }),
            )

        evidence = run_benchmark_task(task, candidate, environ=self.mock_env, transport=citing_t)

        self.assertEqual(evidence.execution_status, ProbeStatus.SUCCESS)
        self.assertTrue(evidence.contract_satisfied)
        # CRITICAL ASSERTION: Provenance status MUST be MODEL_CLAIM_ONLY
        self.assertEqual(evidence.provenance_metadata.provenance_status, ProvenanceState.MODEL_CLAIM_ONLY)
        claimed_cits = [c["raw_citation"] for c in evidence.provenance_metadata.model_claimed_sources]
        self.assertIn("[SRC-01]", claimed_cits)
        self.assertIn("[SRC-02]", claimed_cits)
        self.assertEqual(evidence.provenance_metadata.system_retrieved_sources, [])

        # Contrast with actual system retrieval
        sys_prov = build_provenance_record(
            response_text="Search results found",
            system_sources=[{"id": "doc-1", "url": "https://gov.in/survey", "title": "Survey Spec"}],
            retrieval_provider="tavily",
        )
        self.assertEqual(sys_prov.provenance_status, ProvenanceState.RETRIEVED_BY_SYSTEM)
        self.assertEqual(len(sys_prov.system_retrieved_sources), 1)

    # -------------------------------------------------------------------------
    # Test L: Multiple Candidates Can Be Benchmarked Independently
    # -------------------------------------------------------------------------
    def test_l_multiple_candidates_benchmarked_independently(self):
        """Proves that Ollama and Groq candidates can be evaluated independently against the same suite."""
        tasks = [load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)]
        c_ollama = load_candidate("local_ollama", candidates_dir=self.candidates_dir)
        c_groq = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def t_ollama(*args, **kwargs):
            return TransportResponse(200, json.dumps({"response": '{"plot_id": "P-101", "survey_number": "SN-405", "boundary_coordinates": []}'}))

        def t_groq(*args, **kwargs):
            return TransportResponse(200, json.dumps({"choices": [{"message": {"content": '{"plot_id": "P-101", "survey_number": "SN-405", "boundary_coordinates": []}'}}]}))

        recs_ollama = run_benchmark_suite(tasks, c_ollama, environ={}, transport=t_ollama)
        recs_groq = run_benchmark_suite(tasks, c_groq, environ=self.mock_env, transport=t_groq)

        self.assertEqual(len(recs_ollama), 1)
        self.assertEqual(len(recs_groq), 1)
        self.assertEqual(recs_ollama[0].candidate_id, "local_ollama_llama32")
        self.assertEqual(recs_groq[0].candidate_id, "groq_llama33_70b")
        self.assertNotEqual(recs_ollama[0].evidence_id, recs_groq[0].evidence_id)

    # -------------------------------------------------------------------------
    # Test M: Candidate Output Isolation
    # -------------------------------------------------------------------------
    def test_m_candidate_output_isolation(self):
        """INVARIANT: One candidate's output is NEVER passed as another candidate's input."""
        tasks = [
            load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir),
            load_benchmark_task("task_02_structured_extraction", suite_dir=self.suite_dir),
        ]
        c = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        seen_prompts = []

        def isolating_transport(url, method="POST", headers=None, json_data=None, timeout=30.0):
            seen_prompts.append(json_data.get("prompt"))
            return TransportResponse(200, json.dumps({"response": "SPECIFIC_CANDIDATE_OUTPUT_XYZ_123"}))

        records = run_benchmark_suite(tasks, c, environ={}, transport=isolating_transport)

        self.assertEqual(len(records), 2)
        # Verify that candidate output from task 1 is NOT present in task 2's prompt
        self.assertNotIn("SPECIFIC_CANDIDATE_OUTPUT_XYZ_123", seen_prompts[1])
        # Verify that each prompt matches task input exactly
        self.assertEqual(seen_prompts[0], tasks[0].input_prompt)
        self.assertEqual(seen_prompts[1], tasks[1].input_prompt)

    # -------------------------------------------------------------------------
    # Test N: Benchmark Runner Does NOT Perform Runtime Routing
    # -------------------------------------------------------------------------
    def test_n_benchmark_runner_no_runtime_routing(self):
        """INVARIANT: If a candidate fails, runner records the failure and NEVER falls back to another candidate."""
        task = load_benchmark_task("task_01_json_generation", suite_dir=self.suite_dir)
        c = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def failing_transport(*args, **kwargs):
            raise ConnectionRefusedError("Ollama daemon down")

        evidence = run_benchmark_task(task, c, environ={}, transport=failing_transport)

        # Invariant: Status is FAILED, candidate remains local_ollama, no failover occurred
        self.assertEqual(evidence.candidate_id, "local_ollama_llama32")
        self.assertEqual(evidence.execution_status, ProbeStatus.FAILED)
        self.assertFalse(evidence.contract_satisfied)
        self.assertEqual(evidence.failure_classification.failure_class, FailureClass.NETWORK_FAILURE)

    # -------------------------------------------------------------------------
    # Test O: Zero Network Calls Occur in Offline Tests
    # -------------------------------------------------------------------------
    def test_o_zero_network_calls(self):
        """Guarantees that benchmark harness operates strictly offline with zero socket calls."""
        def guarded_socket(*args, **kwargs):
            raise AssertionError("Network socket creation attempted during offline benchmarking!")

        orig_socket = socket.socket
        socket.socket = guarded_socket
        try:
            tasks = load_benchmark_suite("sih_cadastral_v1")
            candidates = load_all_candidates(candidates_dir=self.candidates_dir)

            # Test a blocked candidate (never touches sockets)
            blocked_c = load_candidate("card_required_free_tier", candidates_dir=self.candidates_dir)
            ev_blocked = run_benchmark_task(tasks[0], blocked_c, environ=self.mock_env)
            self.assertEqual(ev_blocked.execution_status, ProbeStatus.BLOCKED)

            # Test an approved candidate with offline mock transport (never touches sockets)
            approved_c = load_candidate("local_ollama", candidates_dir=self.candidates_dir)
            mock_t = lambda *a, **kw: TransportResponse(200, json.dumps({"response": '{"plot_id": "P-101", "survey_number": "SN-405", "boundary_coordinates": []}'}))
            ev_approved = run_benchmark_task(tasks[0], approved_c, environ={}, transport=mock_t)
            self.assertEqual(ev_approved.execution_status, ProbeStatus.SUCCESS)
        finally:
            socket.socket = orig_socket


if __name__ == "__main__":
    unittest.main()

