"""Deterministic unit test suite for Phase A4.2 Fixture Framework & Evaluation Schemas.

Tests:
A. Valid fixture loads from storage
B. Invalid fixture is rejected by Pydantic validation
C. Valid mock output passes with status=PASS and conformance=1.0
D. Malformed JSON output fails deterministically with INVALID_RESPONSE / SCHEMA_FAILURE
E. Missing required field fails deterministically with SCHEMA_FAILURE
F. Invalid schema type fails deterministically with SCHEMA_FAILURE
G. Provider-style errors (HTTP 429, 413, 500, 401, timeout) are classified accurately
H. Determinism: identical inputs produce bit-for-bit identical results across runs
I. Zero network calls occur during evaluation
J. Zero secrets are written or stored in candidate or evaluation models
K. Grounding evaluator validates true citations and flags fabricated citations
"""

import json
import socket
import unittest
from pathlib import Path
from pydantic import ValidationError

from harness.models import (
    AccessMode,
    BenchmarkFixture,
    BillingStatus,
    CandidateRecord,
    CandidateType,
    EvaluationResult,
    EvaluationStatus,
    FailureClass,
    MockProviderResult,
    VerificationState,
)
from harness.evaluators import evaluate_result, load_fixture
from harness.mocks import get_mock_result, MOCK_RESULTS
from harness.taxonomy import classify_failure


class TestA42DeterministicHarness(unittest.TestCase):
    """Test suite establishing deterministic foundation for provider verification."""

    def setUp(self):
        self.fixtures_dir = Path(__file__).parent.parent.parent / "harness" / "fixtures"

    # -------------------------------------------------------------------------
    # Test A: Valid Fixture Loads
    # -------------------------------------------------------------------------
    def test_a_valid_fixture_loads(self):
        """Proves that structured_json_v1 and basic_generation_v1 load cleanly."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        self.assertEqual(fixture.fixture_id, "structured_json_v1")
        self.assertEqual(fixture.version, "1.0.0")
        self.assertIn("role_title", fixture.expected_structure.get("required", []))

        fixture_gen = load_fixture("basic_generation_v1", fixtures_dir=self.fixtures_dir)
        self.assertEqual(fixture_gen.fixture_id, "basic_generation_v1")
        self.assertIn("prompt", fixture_gen.input_payload)

    # -------------------------------------------------------------------------
    # Test B: Invalid Fixture Is Rejected
    # -------------------------------------------------------------------------
    def test_b_invalid_fixture_rejected(self):
        """Proves that a fixture missing required fields fails Pydantic validation."""
        invalid_data = {
            "version": "1.0.0",
            # missing fixture_id, objective, input_payload
        }
        with self.assertRaises(ValidationError):
            BenchmarkFixture.model_validate(invalid_data)

    # -------------------------------------------------------------------------
    # Test C: Valid Mock Output Passes
    # -------------------------------------------------------------------------
    def test_c_valid_mock_output_passes(self):
        """Proves that a valid structured JSON mock achieves status=PASS and score=1.0."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("VALID_STRUCTURED_JSON")

        eval_result = evaluate_result(fixture, mock_result)

        self.assertEqual(eval_result.status, EvaluationStatus.PASS)
        self.assertIsNone(eval_result.failure_classification)
        self.assertEqual(eval_result.metrics.get("schema_conformance"), 1.0)
        self.assertEqual(eval_result.metrics.get("required_field_score"), 1.0)

    # -------------------------------------------------------------------------
    # Test D: Malformed Output Fails Deterministically
    # -------------------------------------------------------------------------
    def test_d_malformed_output_fails_deterministically(self):
        """Proves that unparseable/truncated JSON fails with INVALID_RESPONSE."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("MALFORMED_JSON")

        eval_result = evaluate_result(fixture, mock_result)

        self.assertEqual(eval_result.status, EvaluationStatus.FAIL)
        self.assertIsNotNone(eval_result.failure_classification)
        self.assertIn(
            eval_result.failure_classification.failure_class,
            [FailureClass.INVALID_RESPONSE, FailureClass.SCHEMA_FAILURE],
        )
        self.assertEqual(eval_result.metrics.get("schema_conformance"), 0.0)

    # -------------------------------------------------------------------------
    # Test E: Missing Required Field Fails Deterministically
    # -------------------------------------------------------------------------
    def test_e_missing_required_field_fails_deterministically(self):
        """Proves that JSON missing 'min_years_experience' fails with SCHEMA_FAILURE."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("MISSING_REQUIRED_FIELD")

        eval_result = evaluate_result(fixture, mock_result)

        self.assertEqual(eval_result.status, EvaluationStatus.FAIL)
        self.assertIsNotNone(eval_result.failure_classification)
        self.assertEqual(eval_result.failure_classification.failure_class, FailureClass.SCHEMA_FAILURE)
        self.assertLess(eval_result.metrics.get("required_field_score", 1.0), 1.0)

    # -------------------------------------------------------------------------
    # Test F: Invalid Schema Type Fails Deterministically
    # -------------------------------------------------------------------------
    def test_f_invalid_schema_type_fails_deterministically(self):
        """Proves that string 'five years' instead of integer fails with SCHEMA_FAILURE."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("INVALID_SCHEMA_TYPE")

        eval_result = evaluate_result(fixture, mock_result)

        self.assertEqual(eval_result.status, EvaluationStatus.FAIL)
        self.assertIsNotNone(eval_result.failure_classification)
        self.assertEqual(eval_result.failure_classification.failure_class, FailureClass.SCHEMA_FAILURE)
        self.assertEqual(eval_result.metrics.get("schema_conformance"), 0.0)

    # -------------------------------------------------------------------------
    # Test G: Provider-Style Errors Are Classified Accurately
    # -------------------------------------------------------------------------
    def test_g_provider_errors_classified_accurately(self):
        """Proves that HTTP 429, 413, 500, and Timeout map to exact failure classes."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)

        # 1. HTTP 429 -> RATE_LIMITED (retryable)
        res_429 = evaluate_result(fixture, get_mock_result("PROVIDER_ERROR_HTTP_429"))
        self.assertEqual(res_429.status, EvaluationStatus.FAIL)
        self.assertEqual(res_429.failure_classification.failure_class, FailureClass.RATE_LIMITED)
        self.assertTrue(res_429.failure_classification.is_retryable)

        # 2. HTTP 413 -> REQUEST_TOO_LARGE (non-retryable)
        res_413 = evaluate_result(fixture, get_mock_result("PROVIDER_ERROR_HTTP_413"))
        self.assertEqual(res_413.status, EvaluationStatus.FAIL)
        self.assertEqual(res_413.failure_classification.failure_class, FailureClass.REQUEST_TOO_LARGE)
        self.assertFalse(res_413.failure_classification.is_retryable)

        # 3. Timeout -> TIMEOUT
        res_timeout = evaluate_result(fixture, get_mock_result("PROVIDER_ERROR_TIMEOUT"))
        self.assertEqual(res_timeout.status, EvaluationStatus.FAIL)
        self.assertEqual(res_timeout.failure_classification.failure_class, FailureClass.TIMEOUT)

        # 4. HTTP 500 -> PROVIDER_ERROR
        res_500 = evaluate_result(fixture, get_mock_result("PROVIDER_ERROR_HTTP_500"))
        self.assertEqual(res_500.status, EvaluationStatus.FAIL)
        self.assertEqual(res_500.failure_classification.failure_class, FailureClass.PROVIDER_ERROR)

        # 5. Direct classifier test for HTTP 401
        res_401 = classify_failure(http_status=401, error_code="invalid_api_key", raw_message="Unauthorized access")
        self.assertEqual(res_401.failure_class, FailureClass.AUTHENTICATION_FAILURE)

    # -------------------------------------------------------------------------
    # Test H: Determinism (Same Input -> Same Output)
    # -------------------------------------------------------------------------
    def test_h_same_input_same_output_determinism(self):
        """Proves that evaluating the same mock against the same fixture 10 times yields identical metrics."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("VALID_STRUCTURED_JSON")

        runs = [evaluate_result(fixture, mock_result) for _ in range(10)]

        first_status = runs[0].status
        first_metrics = runs[0].metrics
        first_rules = runs[0].rule_evaluations

        for idx, r in enumerate(runs[1:], start=1):
            self.assertEqual(r.status, first_status, f"Run {idx} status differed")
            self.assertEqual(r.metrics, first_metrics, f"Run {idx} metrics differed")
            self.assertEqual(r.rule_evaluations, first_rules, f"Run {idx} rules differed")

    # -------------------------------------------------------------------------
    # Test I: Zero Network Calls Occur
    # -------------------------------------------------------------------------
    def test_i_zero_network_calls_occur(self):
        """Guarantees that evaluation operates strictly offline without opening sockets."""
        fixture = load_fixture("structured_json_v1", fixtures_dir=self.fixtures_dir)
        mock_result = get_mock_result("VALID_STRUCTURED_JSON")

        original_socket = socket.socket

        def guarded_socket(*args, **kwargs):
            raise AssertionError("A4.2 evaluation attempted to open a network socket!")

        socket.socket = guarded_socket
        try:
            res = evaluate_result(fixture, mock_result)
            self.assertEqual(res.status, EvaluationStatus.PASS)
        finally:
            socket.socket = original_socket

    # -------------------------------------------------------------------------
    # Test J: Zero Secrets Written or Stored
    # -------------------------------------------------------------------------
    def test_j_zero_secrets_stored_in_candidate(self):
        """Proves that CandidateRecord only allows env var names and rejects raw secrets."""
        # 1. Valid env var name passes
        cand = CandidateRecord(
            candidate_id="cand_test_001",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="groq",
            technical_identifier="llama-3.3-70b-versatile",
            display_name="Test Llama Model",
            access_mode=AccessMode.API,
            billing_status=BillingStatus.FREE_TIER,
            auth_env_var="GROQ_API_KEY",
            claimed_capabilities=["STRUCTURED_SYNTHESIS"],
        )
        self.assertEqual(cand.auth_env_var, "GROQ_API_KEY")

        # 2. Raw secret string (with spaces or long token shape) is rejected
        with self.assertRaises(ValueError):
            CandidateRecord(
                candidate_id="cand_test_bad",
                candidate_type=CandidateType.INFERENCE_MODEL,
                provider_id="groq",
                technical_identifier="test",
                display_name="Bad Secret Candidate",
                access_mode=AccessMode.API,
                auth_env_var="gsk_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", # > 64 chars
            )

    # -------------------------------------------------------------------------
    # Test K: Grounding Evaluator Validates True & Fabricated Citations
    # -------------------------------------------------------------------------
    def test_k_grounding_evaluator(self):
        """Proves that valid evidence citations achieve 1.0, while fabricated IDs fail."""
        fixture = load_fixture("claim_grounding_v1", fixtures_dir=self.fixtures_dir)

        # 1. Valid Grounding: cites [1] and [2] -> PASS
        res_valid = evaluate_result(fixture, get_mock_result("VALID_CLAIM_GROUNDING"))
        self.assertEqual(res_valid.status, EvaluationStatus.PASS)
        self.assertEqual(res_valid.metrics.get("citation_precision"), 1.0)
        self.assertEqual(res_valid.metrics.get("grounding_score"), 1.0)

        # 2. Fabricated Citation: cites [99] -> FAIL with GROUNDING_FAILURE
        res_fab = evaluate_result(fixture, get_mock_result("FABRICATED_CITATION_GROUNDING"))
        self.assertEqual(res_fab.status, EvaluationStatus.FAIL)
        self.assertEqual(res_fab.failure_classification.failure_class, FailureClass.GROUNDING_FAILURE)
        self.assertEqual(res_fab.metrics.get("citation_precision"), 0.0)

    # -------------------------------------------------------------------------
    # Test L: Basic Generation Evaluator
    # -------------------------------------------------------------------------
    def test_l_basic_generation_evaluator(self):
        """Proves basic generation evaluator passes non-empty and fails empty output."""
        fixture = load_fixture("basic_generation_v1", fixtures_dir=self.fixtures_dir)

        # 1. Valid text -> PASS
        res_valid = evaluate_result(fixture, get_mock_result("VALID_BASIC_GENERATION"))
        self.assertEqual(res_valid.status, EvaluationStatus.PASS)
        self.assertEqual(res_valid.metrics.get("conformance_score"), 1.0)

        # 2. Empty text -> FAIL
        res_empty = evaluate_result(fixture, get_mock_result("EMPTY_BASIC_GENERATION"))
        self.assertEqual(res_empty.status, EvaluationStatus.FAIL)
        self.assertEqual(res_empty.metrics.get("char_count"), 0.0)


if __name__ == "__main__":
    unittest.main()

