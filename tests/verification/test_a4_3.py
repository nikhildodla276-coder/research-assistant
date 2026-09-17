"""Deterministic unit test suite for Phase A4.3 Candidate Model & Access/Cost Gates.

Covers all A4.3 pre-probe safety gates:
A. Valid local/free candidate clears all gates -> ALLOWED (may_proceed_to_probe=True)
B. Valid free-tier API with card_required=False & env var present -> ALLOWED (may_proceed_to_probe=True)
C. Missing required credential env var -> BLOCKED (CREDENTIAL_GATE)
D. Raw secret supplied to auth_env_var -> Rejected with ValidationError
E. UNKNOWN billing status -> BLOCKED (BILLING_STATUS_GATE fails closed)
F. CARD_REQUIRED -> BLOCKED (CARD_REQUIRED_GATE fails closed)
G. METERED billing status -> BLOCKED (BILLING_STATUS_GATE fails closed)
H. PAID_ONLY billing status -> BLOCKED (BILLING_STATUS_GATE fails closed)
I. Missing or null billing status -> BLOCKED (BILLING_STATUS_GATE fails closed)
J. Unsupported access mode (MANUAL) -> BLOCKED (ACCESS_MODE_GATE fails closed)
K. Missing or null access mode -> BLOCKED (ACCESS_MODE_GATE fails closed)
L. Inviolable Invariant: Passing gate NEVER grants routing authority (is_routable == False)
M. requires_user_approval=True blocks probe -> BLOCKED (USER_APPROVAL_GATE)
N. Zero network calls occur during gate checks (socket monkeypatch)
O. Determinism: identical inputs produce identical GateDecision outputs
P. Zero secret leakage: raw secret values never appear in GateDecision output
Q. Candidate loader correctly parses all manifests and handles error conditions
R. Candidate in VERIFIED_FAIL state is BLOCKED by VERIFICATION_STATE_GATE
"""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from harness.gates import check_candidate_access_and_cost
from harness.loader import load_all_candidates, load_candidate
from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateRecord,
    CandidateType,
    GateDecision,
    GateName,
    GateStatus,
    VerificationState,
)


class TestA43AccessAndCostGates(unittest.TestCase):
    """Test suite verifying access and cost gates for Phase A4.3."""

    def setUp(self):
        self.candidates_dir = Path(__file__).parent.parent.parent / "harness" / "candidates"
        self.mock_env = {
            "GROQ_API_KEY": "gsk_mock_test_key_abc123",
            "TAVILY_API_KEY": "tvly_mock_test_key_xyz789",
            "OPENAI_API_KEY": "sk_mock_test_key_openai456",
            "AWS_ACCESS_KEY_ID": "AKIA_MOCK_TEST_KEY_789",
            "SKETCHY_API_KEY": "sketchy_mock_test_key_000",
            "ANTHROPIC_API_KEY": "sk-ant-api_mock_test_key_111",
        }

    # -------------------------------------------------------------------------
    # Test A: Valid Local / Free Candidate
    # -------------------------------------------------------------------------
    def test_a_valid_local_free_allowed(self):
        """Proves that a local, free candidate clears all gates without credentials."""
        candidate = load_candidate("local_ollama", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ={})

        self.assertEqual(decision.status, GateStatus.ALLOWED)
        self.assertTrue(decision.may_proceed_to_probe)
        self.assertFalse(decision.requires_user_approval)
        self.assertFalse(decision.is_routable)
        self.assertEqual(decision.failed_gates, [])
        self.assertIn(GateName.ACCESS_MODE_GATE, decision.passed_gates)
        self.assertIn(GateName.BILLING_STATUS_GATE, decision.passed_gates)
        self.assertIn(GateName.CARD_REQUIRED_GATE, decision.passed_gates)
        self.assertIn(GateName.CREDENTIAL_GATE, decision.passed_gates)

    # -------------------------------------------------------------------------
    # Test B: Valid Free-Tier API Candidate
    # -------------------------------------------------------------------------
    def test_b_valid_free_tier_api_allowed(self):
        """Proves that a free-tier API candidate with env var present clears all gates."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.ALLOWED)
        self.assertTrue(decision.may_proceed_to_probe)
        self.assertFalse(decision.requires_user_approval)
        self.assertFalse(decision.is_routable)
        self.assertEqual(decision.failed_gates, [])
        self.assertIn(GateName.CREDENTIAL_GATE, decision.passed_gates)
        self.assertTrue(decision.gate_details["credential"]["credential_present"])

    # -------------------------------------------------------------------------
    # Test C: Missing Required Credential Environment Variable
    # -------------------------------------------------------------------------
    def test_c_missing_credential_env_var_blocked(self):
        """Proves that an API candidate with missing env var fails CREDENTIAL_GATE."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)
        # Pass empty environ to simulate missing environment variable
        decision = check_candidate_access_and_cost(candidate, environ={})

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.CREDENTIAL_GATE, decision.failed_gates)
        self.assertFalse(decision.gate_details["credential"]["credential_present"])
        self.assertIn("GROQ_API_KEY", decision.reason)

    # -------------------------------------------------------------------------
    # Test D: Raw Secret Rejection in CandidateRecord
    # -------------------------------------------------------------------------
    def test_d_raw_secret_rejected(self):
        """Proves that raw API keys supplied as auth_env_var are rejected by validation."""
        # Secrets with spaces
        with self.assertRaises(ValidationError):
            CandidateRecord(
                candidate_id="bad_auth_var_spaces",
                candidate_type=CandidateType.INFERENCE_MODEL,
                provider_id="bad_provider",
                technical_identifier="model-x",
                display_name="Bad Auth Model",
                access_mode=AccessMode.API,
                billing_status=BillingStatus.FREE_TIER,
                auth_env_var="Bearer secret_token_value_here",
            )

        # Raw keys exceeding 64 chars
        with self.assertRaises(ValidationError):
            CandidateRecord(
                candidate_id="bad_auth_var_length",
                candidate_type=CandidateType.INFERENCE_MODEL,
                provider_id="bad_provider",
                technical_identifier="model-x",
                display_name="Bad Auth Model",
                access_mode=AccessMode.API,
                billing_status=BillingStatus.FREE_TIER,
                auth_env_var="A" * 65,
            )

        # Raw keys containing non-alphanumeric/underscore characters (e.g. hyphens, colons)
        with self.assertRaises(ValidationError):
            CandidateRecord(
                candidate_id="bad_auth_var_chars",
                candidate_type=CandidateType.INFERENCE_MODEL,
                provider_id="bad_provider",
                technical_identifier="model-x",
                display_name="Bad Auth Model",
                access_mode=AccessMode.API,
                billing_status=BillingStatus.FREE_TIER,
                auth_env_var="sk-proj-1234567890abcdef",
            )

    # -------------------------------------------------------------------------
    # Test E: UNKNOWN Billing Status Fails Closed
    # -------------------------------------------------------------------------
    def test_e_unknown_billing_blocked(self):
        """Proves that UNKNOWN billing status is blocked under default-deny."""
        candidate = load_candidate("unknown_billing_provider", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.BILLING_STATUS_GATE, decision.failed_gates)
        self.assertIn("default-deny", decision.reason)

    # -------------------------------------------------------------------------
    # Test F: CARD_REQUIRED Fails Closed
    # -------------------------------------------------------------------------
    def test_f_card_required_blocked(self):
        """Proves that card_required=True is blocked by CARD_REQUIRED_GATE."""
        candidate = load_candidate("card_required_free_tier", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.CARD_REQUIRED_GATE, decision.failed_gates)
        self.assertTrue(decision.gate_details["card_required"]["card_required"])

    # -------------------------------------------------------------------------
    # Test G: METERED Billing Status Fails Closed
    # -------------------------------------------------------------------------
    def test_g_metered_billing_blocked(self):
        """Proves that METERED pay-as-you-go billing is blocked by BILLING_STATUS_GATE."""
        candidate = load_candidate("metered_provider", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.BILLING_STATUS_GATE, decision.failed_gates)

    # -------------------------------------------------------------------------
    # Test H: PAID_ONLY Billing Status Fails Closed
    # -------------------------------------------------------------------------
    def test_h_paid_only_blocked(self):
        """Proves that PAID_ONLY billing is blocked by BILLING_STATUS_GATE."""
        candidate = load_candidate("paid_only_provider", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.BILLING_STATUS_GATE, decision.failed_gates)

    # -------------------------------------------------------------------------
    # Test I: Missing Billing Info Fails Closed
    # -------------------------------------------------------------------------
    def test_i_missing_billing_info_blocked(self):
        """Proves that missing billing status fails closed."""
        candidate = CandidateRecord(
            candidate_id="missing_billing",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="test",
            technical_identifier="m",
            display_name="Missing Billing",
            access_mode=AccessMode.API,
            billing_status=BillingStatus.UNKNOWN,
        )
        decision = check_candidate_access_and_cost(candidate, environ={})
        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertIn(GateName.BILLING_STATUS_GATE, decision.failed_gates)

    # -------------------------------------------------------------------------
    # Test J: Unsupported Access Mode Fails Closed
    # -------------------------------------------------------------------------
    def test_j_unsupported_access_mode_blocked(self):
        """Proves that MANUAL access mode is blocked for automated probing."""
        candidate = load_candidate("unsupported_manual_provider", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.ACCESS_MODE_GATE, decision.failed_gates)

    # -------------------------------------------------------------------------
    # Test K: Missing Access Mode Fails Closed
    # -------------------------------------------------------------------------
    def test_k_missing_access_mode_blocked(self):
        """Proves that missing access mode is blocked."""
        # Using model_construct to bypass Pydantic construction validation to test gate robustness
        candidate = CandidateRecord.model_construct(
            candidate_id="null_access_mode",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="test",
            technical_identifier="m",
            display_name="Null Access Mode",
            access_mode=None,
            billing_status=BillingStatus.LOCAL_FREE,
            card_required=False,
            requires_user_approval=False,
            verification_state=VerificationState.UNVERIFIED,
        )
        decision = check_candidate_access_and_cost(candidate, environ={})
        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertIn(GateName.ACCESS_MODE_GATE, decision.failed_gates)

    # -------------------------------------------------------------------------
    # Test L: Passing Gate NEVER Grants Routing Authority
    # -------------------------------------------------------------------------
    def test_l_passing_gate_never_makes_candidate_routable(self):
        """INVARIANT: Pre-probe gating must NEVER set is_routable=True.
        Passing a safety gate only allows a candidate to be probed; only a full
        verification suite run can recommend routing eligibility."""
        candidates = load_all_candidates(candidates_dir=self.candidates_dir)
        for candidate in candidates:
            decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)
            # Invariant: regardless of decision status, is_routable is unconditionally False
            self.assertFalse(decision.is_routable, f"Violation for candidate {candidate.candidate_id}")

    # -------------------------------------------------------------------------
    # Test M: User Approval Required Blocks Probe
    # -------------------------------------------------------------------------
    def test_m_user_approval_required_blocks_probe(self):
        """Proves that candidate requiring user approval cannot proceed to probe."""
        candidate = load_candidate("approval_required_candidate", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertTrue(decision.requires_user_approval)
        self.assertIn(GateName.USER_APPROVAL_GATE, decision.failed_gates)
        self.assertEqual(decision.gate_details["user_approval"]["status"], "PENDING_APPROVAL")

    # -------------------------------------------------------------------------
    # Test N: Zero Network Calls Occur During Gate Checks
    # -------------------------------------------------------------------------
    def test_n_zero_network_calls(self):
        """Proves that gate evaluation executes with zero network socket operations."""
        def guarded_socket(*args, **kwargs):
            raise AssertionError("Network socket creation attempted during offline gate checks!")

        orig_socket = socket.socket
        socket.socket = guarded_socket
        try:
            candidates = load_all_candidates(candidates_dir=self.candidates_dir)
            self.assertGreaterEqual(len(candidates), 8)
            for candidate in candidates:
                _ = check_candidate_access_and_cost(candidate, environ=self.mock_env)
        finally:
            socket.socket = orig_socket

    # -------------------------------------------------------------------------
    # Test O: Determinism of Gate Decisions
    # -------------------------------------------------------------------------
    def test_o_determinism_across_runs(self):
        """Proves that identical candidate input yields bit-for-bit identical decisions."""
        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)

        decision1 = check_candidate_access_and_cost(candidate, environ=self.mock_env)
        decision2 = check_candidate_access_and_cost(candidate, environ=self.mock_env)

        dump1 = decision1.model_dump(exclude={"checked_at"})
        dump2 = decision2.model_dump(exclude={"checked_at"})
        self.assertEqual(dump1, dump2)

    # -------------------------------------------------------------------------
    # Test P: Zero Secret Leakage
    # -------------------------------------------------------------------------
    def test_p_zero_secret_leakage(self):
        """SECURITY INVARIANT: Environment variable values must never leak into GateDecision."""
        sensitive_token = "TOP_SECRET_REDACTED_TOKEN_ABC_999999"
        leakage_env = {"GROQ_API_KEY": sensitive_token}

        candidate = load_candidate("groq_llama33", candidates_dir=self.candidates_dir)
        decision = check_candidate_access_and_cost(candidate, environ=leakage_env)

        decision_str = str(decision)
        decision_json = decision.model_dump_json()

        self.assertNotIn(sensitive_token, decision_str)
        self.assertNotIn(sensitive_token, decision_json)
        self.assertNotIn(sensitive_token, str(decision.gate_details))

    # -------------------------------------------------------------------------
    # Test Q: Candidate Loader Functionality and Error Handling
    # -------------------------------------------------------------------------
    def test_q_candidate_loader(self):
        """Tests loader functions for candidate discovery, parsing, and error handling."""
        # Load by ID without extension
        c1 = load_candidate("duckduckgo_lite", candidates_dir=self.candidates_dir)
        self.assertEqual(c1.candidate_id, "duckduckgo_lite")

        # Load by filename with extension
        c2 = load_candidate("duckduckgo_lite.json", candidates_dir=self.candidates_dir)
        self.assertEqual(c2.candidate_id, "duckduckgo_lite")

        # Nonexistent candidate raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_candidate("nonexistent_candidate_xyz", candidates_dir=self.candidates_dir)

        # Corrupted JSON raises ValueError
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write("{bad_json: true,")
            tf_path = tf.name

        try:
            with self.assertRaises(ValueError):
                load_candidate(tf_path)
        finally:
            Path(tf_path).unlink()

    # -------------------------------------------------------------------------
    # Test R: Verification State Gate
    # -------------------------------------------------------------------------
    def test_r_verified_fail_state_blocked(self):
        """Proves that a candidate in VERIFIED_FAIL state fails VERIFICATION_STATE_GATE."""
        candidate = CandidateRecord(
            candidate_id="failing_candidate",
            candidate_type=CandidateType.INFERENCE_MODEL,
            provider_id="groq",
            technical_identifier="llama-3.3-70b-versatile",
            display_name="Failing Groq Candidate",
            access_mode=AccessMode.API,
            billing_status=BillingStatus.FREE_TIER,
            auth_env_var="GROQ_API_KEY",
            verification_state=VerificationState.VERIFIED_FAIL,
        )
        decision = check_candidate_access_and_cost(candidate, environ=self.mock_env)
        self.assertEqual(decision.status, GateStatus.BLOCKED)
        self.assertFalse(decision.may_proceed_to_probe)
        self.assertIn(GateName.VERIFICATION_STATE_GATE, decision.failed_gates)


if __name__ == "__main__":
    unittest.main()

