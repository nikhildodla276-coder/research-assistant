"""Deterministic unit test suite for Phase A4.4 Live Provider Adapters & Probing Harness.

Tests:
A. Valid local candidate reaches adapter and normalizes response
B. Blocked candidate never reaches adapter (gated by A4.3)
C. Missing credential never reaches provider (gated by A4.3)
D. Provider authentication failure (HTTP 401) is classified correctly
E. Timeout is classified correctly
F. Connection failure is classified correctly
G. Malformed provider response is classified correctly
H. Valid provider response normalizes correctly
I. Provider-specific response becomes common ProbeResult
J. Secrets never appear in ProbeResult, telemetry, or error messages
K. Repeated deterministic fixture produces identical normalized result
L. Unsupported candidate adapter fails safely with UNSUPPORTED_REQUEST
M. ProbeRequest rejects secret-like keys in metadata
N. Offline deterministic execution produces zero network calls (socket monkeypatch)
"""

import json
import socket
import unittest
from pathlib import Path
from pydantic import ValidationError

from harness.adapters.base import TransportResponse
from harness.evaluators import load_fixture
from harness.loader import load_all_candidates, load_candidate
from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateRecord,
    CandidateType,
    FailureClass,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
    VerificationState,
)
from harness.probing import run_candidate_probe, run_probe_from_fixture


class TestA44ProbingHarness(unittest.TestCase):
    """Test suite validating Phase A4.4 adapter abstraction and probing harness."""

    def setUp(self):
        self.candidates_dir = Path(__file__).parent.parent.parent / "harness" / "candidates"
        self.fixtures_dir = Path(__file__).parent.parent.parent / "harness" / "fixtures"
        self.mock_env = {
            "GROQ_API_KEY": "gsk_mock_test_key_abc123",
            "TAVILY_API_KEY": "tvly_mock_test_key_xyz789",
            "OPENAI_API_KEY": "sk_mock_test_key_openai456",
            "AWS_ACCESS_KEY_ID": "AKIA_MOCK_TEST_KEY_789",
        }

    # -------------------------------------------------------------------------
    # Test A: Valid Local Candidate Reaches Adapter
    # -------------------------------------------------------------------------
    def test_a_valid_local_candidate_reaches_adapter(self):
        """Proves that a valid local candidate clears A4.3 and executes through OllamaAdapter."""
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def mock_ollama_transport(url, method="POST", headers=None, json_data=None, timeout=30.0):
            self.assertIn("11434/api/generate", url)
            self.assertEqual(json_data.get("model"), "llama3.2:3b")
            return TransportResponse(
                status_code=200,
                text=json.dumps({
                    "response": "PONG",
                    "total_duration": 12000000,
                    "prompt_eval_count": 5,
                    "eval_count": 1,
                }),
            )

        req = ProbeRequest(
            probe_id="test_probe_a",
            candidate_id=candidate.candidate_id,
            prompt="Respond with PONG",
            expected_output_format="text",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ={},
            transport=mock_ollama_transport,
        )

        self.assertEqual(result.status, ProbeStatus.SUCCESS)
        self.assertTrue(result.success)
        self.assertTrue(result.attempted)
        self.assertTrue(result.contract_satisfied)
        self.assertEqual(result.normalized_response, "PONG")
        self.assertEqual(result.usage_metadata, {"input_tokens": 5, "output_tokens": 1})
        self.assertGreaterEqual(result.latency_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test B: Blocked Candidate Never Reaches Adapter
    # -------------------------------------------------------------------------
    def test_b_blocked_candidate_never_reaches_adapter(self):
        """Proves that a candidate blocked by A4.3 (card required/metered) never calls adapter."""
        candidate = load_candidate("card_required_free_tier", candidates_dir=self.candidates_dir)
        transport_called = False

        def guarded_transport(*args, **kwargs):
            nonlocal transport_called
            transport_called = True
            raise AssertionError("Transport should never be called for blocked candidates!")

        req = ProbeRequest(
            probe_id="test_probe_b",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=self.mock_env,
            transport=guarded_transport,
        )

        self.assertFalse(transport_called)
        self.assertEqual(result.status, ProbeStatus.BLOCKED)
        self.assertFalse(result.success)
        self.assertFalse(result.attempted)
        self.assertFalse(result.contract_satisfied)
        self.assertIsNotNone(result.error_classification)
        self.assertEqual(result.error_classification.failure_class, FailureClass.BILLING_BLOCKED)
        self.assertEqual(result.latency_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test C: Missing Credential Never Reaches Provider
    # -------------------------------------------------------------------------
    def test_c_missing_credential_never_reaches_provider(self):
        """Proves that a candidate with missing API key is blocked by A4.3 and never calls transport."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)
        transport_called = False

        def guarded_transport(*args, **kwargs):
            nonlocal transport_called
            transport_called = True
            raise AssertionError("Transport should never be called when credentials are missing!")

        req = ProbeRequest(
            probe_id="test_probe_c",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ={},  # empty environ -> missing GROQ_API_KEY
            transport=guarded_transport,
        )

        self.assertFalse(transport_called)
        self.assertEqual(result.status, ProbeStatus.BLOCKED)
        self.assertFalse(result.attempted)
        self.assertEqual(result.error_classification.failure_class, FailureClass.AUTHENTICATION_FAILURE)

    # -------------------------------------------------------------------------
    # Test D: Provider Authentication Failure Classified Correctly
    # -------------------------------------------------------------------------
    def test_d_provider_auth_failure_classified_correctly(self):
        """Proves that an HTTP 401 from provider is correctly classified as AUTHENTICATION_FAILURE."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def mock_401_transport(*args, **kwargs):
            return TransportResponse(
                status_code=401,
                text=json.dumps({"error": {"message": "Invalid API Key provided", "type": "invalid_request_error"}}),
            )

        req = ProbeRequest(
            probe_id="test_probe_d",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=self.mock_env,
            transport=mock_401_transport,
        )

        self.assertEqual(result.status, ProbeStatus.FAILED)
        self.assertFalse(result.success)
        self.assertTrue(result.attempted)
        self.assertIsNotNone(result.error_classification)
        self.assertEqual(result.error_classification.failure_class, FailureClass.AUTHENTICATION_FAILURE)
        self.assertEqual(result.error_classification.http_status, 401)

    # -------------------------------------------------------------------------
    # Test E: Timeout Classified Correctly
    # -------------------------------------------------------------------------
    def test_e_timeout_classified_correctly(self):
        """Proves that transport TimeoutError is classified as FailureClass.TIMEOUT."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def mock_timeout_transport(*args, **kwargs):
            raise TimeoutError("Client timed out waiting for provider response after 30s")

        req = ProbeRequest(
            probe_id="test_probe_e",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=self.mock_env,
            transport=mock_timeout_transport,
        )

        self.assertEqual(result.status, ProbeStatus.FAILED)
        self.assertFalse(result.success)
        self.assertTrue(result.attempted)
        self.assertEqual(result.error_classification.failure_class, FailureClass.TIMEOUT)

    # -------------------------------------------------------------------------
    # Test F: Connection Failure Classified Correctly
    # -------------------------------------------------------------------------
    def test_f_connection_failure_classified_correctly(self):
        """Proves that connection refusal/network error is classified as NETWORK_FAILURE."""
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def mock_conn_transport(*args, **kwargs):
            raise ConnectionRefusedError("Connection refused to http://localhost:11434")

        req = ProbeRequest(
            probe_id="test_probe_f",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ={},
            transport=mock_conn_transport,
        )

        self.assertEqual(result.status, ProbeStatus.FAILED)
        self.assertFalse(result.success)
        self.assertTrue(result.attempted)
        self.assertEqual(result.error_classification.failure_class, FailureClass.NETWORK_FAILURE)

    # -------------------------------------------------------------------------
    # Test G: Malformed Provider Response Classified Correctly
    # -------------------------------------------------------------------------
    def test_g_malformed_provider_response_classified_correctly(self):
        """Proves that unparseable or corrupted JSON response is classified as INVALID_RESPONSE."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        def mock_corrupt_transport(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text="<html><body>502 Bad Gateway - unparseable non-json body</body></html>",
            )

        req = ProbeRequest(
            probe_id="test_probe_g",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=self.mock_env,
            transport=mock_corrupt_transport,
        )

        self.assertEqual(result.status, ProbeStatus.FAILED)
        self.assertFalse(result.success)
        self.assertTrue(result.attempted)
        self.assertEqual(result.error_classification.failure_class, FailureClass.INVALID_RESPONSE)

    # -------------------------------------------------------------------------
    # Test H: Valid Provider Response Normalizes Correctly
    # -------------------------------------------------------------------------
    def test_h_valid_provider_response_normalizes_correctly(self):
        """Proves that a structured retrieval response normalizes cleanly."""
        candidate = load_candidate("tavily_search", candidates_dir=self.candidates_dir)

        def mock_tavily_transport(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({
                    "results": [
                        {"title": "Doc A", "url": "https://example.com/a", "content": "Information A"},
                        {"title": "Doc B", "url": "https://example.com/b", "content": "Information B"},
                    ]
                }),
            )

        req = ProbeRequest(
            probe_id="test_probe_h",
            candidate_id=candidate.candidate_id,
            prompt="Research query",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=self.mock_env,
            transport=mock_tavily_transport,
        )

        self.assertEqual(result.status, ProbeStatus.SUCCESS)
        self.assertTrue(result.success)
        self.assertTrue(result.contract_satisfied)
        self.assertIn("[1] Doc A", result.normalized_response)
        self.assertIn("[2] Doc B", result.normalized_response)
        self.assertEqual(result.telemetry["results_count"], 2)

    # -------------------------------------------------------------------------
    # Test I: Provider-Specific Response Becomes Common ProbeResult
    # -------------------------------------------------------------------------
    def test_i_provider_specific_response_becomes_common_probe_result(self):
        """Proves that Ollama, Groq, and DuckDuckGo adapters all output conformant ProbeResult."""
        c_ollama = load_candidate("local_ollama", candidates_dir=self.candidates_dir)
        c_groq = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)
        c_ddg = load_candidate("duckduckgo_lite", candidates_dir=self.candidates_dir)

        def t_ollama(*args, **kwargs):
            return TransportResponse(status_code=200, text=json.dumps({"response": "Ollama OK"}))

        def t_groq(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({"choices": [{"message": {"content": "Groq OK"}, "finish_reason": "stop"}]}),
            )

        def t_ddg(*args, **kwargs):
            return TransportResponse(status_code=200, text="<html><body>DuckDuckGo Search Results</body></html>")

        r1 = run_candidate_probe(c_ollama, ProbeRequest(candidate_id=c_ollama.candidate_id, prompt="Hi"), environ={}, transport=t_ollama)
        r2 = run_candidate_probe(c_groq, ProbeRequest(candidate_id=c_groq.candidate_id, prompt="Hi"), environ=self.mock_env, transport=t_groq)
        r3 = run_candidate_probe(c_ddg, ProbeRequest(candidate_id=c_ddg.candidate_id, prompt="Hi"), environ={}, transport=t_ddg)

        for res in [r1, r2, r3]:
            self.assertIsInstance(res, ProbeResult)
            self.assertEqual(res.status, ProbeStatus.SUCCESS)
            self.assertTrue(res.success)
            self.assertTrue(res.contract_satisfied)
            self.assertIsNotNone(res.normalized_response)
            self.assertGreaterEqual(res.latency_ms, 0.0)
            self.assertEqual(res.telemetry["gate_status"], "ALLOWED")

    # -------------------------------------------------------------------------
    # Test J: Secrets Never Appear in ProbeResult or Telemetry
    # -------------------------------------------------------------------------
    def test_j_secrets_never_appear_in_probe_result(self):
        """SECURITY INVARIANT: Environment credentials must never leak into ProbeResult."""
        sensitive_token = "gsk_SUPER_CONFIDENTIAL_KEY_XYZ_98765"
        leakage_env = {"GROQ_API_KEY": sensitive_token}

        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        # Provider maliciously or accidentally echoes the secret back in body
        def echoing_transport(*args, **kwargs):
            return TransportResponse(
                status_code=401,
                text=f"Authentication failed: token {sensitive_token} is invalid or expired",
            )

        req = ProbeRequest(
            probe_id="test_probe_j",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(
            candidate=candidate,
            request=req,
            environ=leakage_env,
            transport=echoing_transport,
        )

        res_json = result.model_dump_json()
        res_str = str(result)

        self.assertNotIn(sensitive_token, res_json)
        self.assertNotIn(sensitive_token, res_str)
        self.assertIn("[REDACTED_SECRET]", res_json)

    # -------------------------------------------------------------------------
    # Test K: Deterministic Fixture Execution
    # -------------------------------------------------------------------------
    def test_k_deterministic_fixture_probe(self):
        """Proves that executing a probe against capability_probe_v1 produces deterministic output."""
        fixture = load_fixture("capability_probe_v1", fixtures_dir=self.fixtures_dir)
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)

        def mock_deterministic_transport(*args, **kwargs):
            return TransportResponse(
                status_code=200,
                text=json.dumps({"response": "PONG", "total_duration": 1000}),
            )

        r1 = run_probe_from_fixture(candidate, fixture, environ={}, transport=mock_deterministic_transport)
        r2 = run_probe_from_fixture(candidate, fixture, environ={}, transport=mock_deterministic_transport)

        d1 = r1.model_dump(exclude={"probe_id", "timestamp", "latency_ms"})
        d2 = r2.model_dump(exclude={"probe_id", "timestamp", "latency_ms"})

        self.assertEqual(d1, d2)
        self.assertEqual(r1.normalized_response, "PONG")

    # -------------------------------------------------------------------------
    # Test L: Unsupported Candidate Adapter
    # -------------------------------------------------------------------------
    def test_l_unsupported_candidate_adapter_fails_safely(self):
        """Proves that a candidate with unsupported access mode/provider fails safely."""
        candidate = CandidateRecord(
            candidate_id="unsupported_candidate",
            candidate_type=CandidateType.LOCAL_UTILITY,
            provider_id="custom_unknown_device",
            technical_identifier="device-v1",
            display_name="Unknown Device",
            access_mode=AccessMode.LOCAL,
            billing_status=BillingStatus.LOCAL_FREE,
        )

        req = ProbeRequest(
            probe_id="test_probe_l",
            candidate_id=candidate.candidate_id,
            prompt="Ping",
        )

        result = run_candidate_probe(candidate, req, environ={})
        self.assertEqual(result.status, ProbeStatus.FAILED)
        self.assertFalse(result.success)
        self.assertFalse(result.attempted)
        self.assertEqual(result.error_classification.failure_class, FailureClass.UNSUPPORTED_REQUEST)

    # -------------------------------------------------------------------------
    # Test M: ProbeRequest Rejects Secret-Like Metadata Keys
    # -------------------------------------------------------------------------
    def test_m_metadata_secret_injection_rejected(self):
        """Proves that ProbeRequest rejects secret-like metadata keys."""
        with self.assertRaises(ValidationError):
            ProbeRequest(
                candidate_id="test_c",
                prompt="Ping",
                metadata={"api_key": "secret_value_attempt"},
            )

        with self.assertRaises(ValidationError):
            ProbeRequest(
                candidate_id="test_c",
                prompt="Ping",
                metadata={"user_token": "token_val"},
            )

    # -------------------------------------------------------------------------
    # Test N: Zero Network Calls Occur During Deterministic Tests
    # -------------------------------------------------------------------------
    def test_n_zero_network_calls(self):
        """Guarantees that probing harness operates strictly offline with zero socket calls."""
        def guarded_socket(*args, **kwargs):
            raise AssertionError("Network socket creation attempted during offline probing!")

        orig_socket = socket.socket
        socket.socket = guarded_socket
        try:
            candidates = load_all_candidates(candidates_dir=self.candidates_dir)
            for c in candidates:
                req = ProbeRequest(probe_id="net_test", candidate_id=c.candidate_id, prompt="Offline ping")
                # When transport is not supplied, blocked candidates abort at A4.3 without touching sockets
                if c.candidate_id in ["local_ollama_llama32", "duckduckgo_lite"]:
                    # Provide an offline mock transport so local/open candidates also don't open sockets
                    mock_t = lambda *a, **kw: TransportResponse(200, "{\"response\": \"offline_mock\"}")
                    res = run_candidate_probe(c, req, environ=self.mock_env, transport=mock_t)
                else:
                    res = run_candidate_probe(c, req, environ=self.mock_env)
                self.assertIsInstance(res, ProbeResult)
        finally:
            socket.socket = orig_socket


if __name__ == "__main__":
    unittest.main()
