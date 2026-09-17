"""Deterministic Pre-Probe Access & Cost Gates for Phase A4.3.

Evaluates candidate metadata against strict, fail-closed access, billing,
card requirement, and credential existence gates before any live probe is permitted.
Contains zero network operations, zero LLM dependencies, and zero secret leakage.
"""

import os
from typing import Any, Dict, List, Optional

from harness.models import (
    AccessMode,
    BillingStatus,
    CandidateRecord,
    GateDecision,
    GateName,
    GateStatus,
    VerificationState,
)

# Supported access modes for automated test probing
SUPPORTED_ACCESS_MODES = {
    AccessMode.API,
    AccessMode.LOCAL,
    AccessMode.OPEN_SOURCE,
}

# Explicitly allowed safe zero-cost billing classifications
SAFE_BILLING_STATUSES = {
    BillingStatus.LOCAL_FREE,
    BillingStatus.FREE_SERVICE_ACCESS,
    BillingStatus.FREE_TIER,
}


def check_candidate_access_and_cost(
    candidate: CandidateRecord,
    environ: Optional[Dict[str, str]] = None,
) -> GateDecision:
    """Evaluates whether a candidate clears all access, billing, and credential gates.

    Enforces fail-closed semantics: any missing, unsupported, or ambiguous
    condition immediately produces a BLOCKED decision.
    Never inspects or stores secret values.
    Never sets is_routable=True (pre-probe gating never grants production routing authority).
    """
    env = environ if environ is not None else os.environ
    failed_gates: List[GateName] = []
    passed_gates: List[GateName] = []
    gate_details: Dict[str, Any] = {}
    failure_reasons: List[str] = []

    # -------------------------------------------------------------------------
    # Gate 1: Access Mode Gate
    # -------------------------------------------------------------------------
    if not candidate.access_mode:
        failed_gates.append(GateName.ACCESS_MODE_GATE)
        failure_reasons.append("Access mode is missing or unspecified.")
        gate_details["access_mode"] = {"status": "FAILED", "mode": None}
    elif candidate.access_mode not in SUPPORTED_ACCESS_MODES:
        failed_gates.append(GateName.ACCESS_MODE_GATE)
        failure_reasons.append(
            f"Access mode '{candidate.access_mode.value}' is unsupported for automated probing."
        )
        gate_details["access_mode"] = {
            "status": "FAILED",
            "mode": candidate.access_mode.value,
            "supported": [m.value for m in SUPPORTED_ACCESS_MODES],
        }
    else:
        passed_gates.append(GateName.ACCESS_MODE_GATE)
        gate_details["access_mode"] = {"status": "PASSED", "mode": candidate.access_mode.value}

    # -------------------------------------------------------------------------
    # Gate 2: Billing Status Gate (Default-Deny)
    # -------------------------------------------------------------------------
    if not candidate.billing_status or candidate.billing_status == BillingStatus.UNKNOWN:
        failed_gates.append(GateName.BILLING_STATUS_GATE)
        failure_reasons.append(
            f"Billing status '{candidate.billing_status.value if candidate.billing_status else 'MISSING'}' "
            "fails closed under default-deny cost policy."
        )
        gate_details["billing_status"] = {"status": "FAILED", "classification": "UNKNOWN_OR_MISSING"}
    elif candidate.billing_status not in SAFE_BILLING_STATUSES:
        failed_gates.append(GateName.BILLING_STATUS_GATE)
        failure_reasons.append(
            f"Billing status '{candidate.billing_status.value}' is potentially chargeable or metered and is blocked."
        )
        gate_details["billing_status"] = {
            "status": "FAILED",
            "classification": candidate.billing_status.value,
            "safe_allowed": [s.value for s in SAFE_BILLING_STATUSES],
        }
    else:
        passed_gates.append(GateName.BILLING_STATUS_GATE)
        gate_details["billing_status"] = {"status": "PASSED", "classification": candidate.billing_status.value}

    # -------------------------------------------------------------------------
    # Gate 3: Card Required Gate
    # -------------------------------------------------------------------------
    if candidate.card_required:
        failed_gates.append(GateName.CARD_REQUIRED_GATE)
        failure_reasons.append("Candidate requires a payment card on file, risking unexpected charges.")
        gate_details["card_required"] = {"status": "FAILED", "card_required": True}
    elif candidate.billing_status == BillingStatus.LOCAL_FREE and candidate.card_required:
        # Contradictory metadata
        failed_gates.append(GateName.CARD_REQUIRED_GATE)
        failure_reasons.append("Contradictory metadata: LOCAL_FREE candidate declares card_required=True.")
        gate_details["card_required"] = {"status": "FAILED", "contradiction": True}
    else:
        passed_gates.append(GateName.CARD_REQUIRED_GATE)
        gate_details["card_required"] = {"status": "PASSED", "card_required": False}

    # -------------------------------------------------------------------------
    # Gate 4: Credential Gate
    # -------------------------------------------------------------------------
    if candidate.auth_env_var:
        # Authentication variable specified: verify existence in environment
        # SECURITY INVARIANT: Check presence only; NEVER read, log, or persist the secret value
        secret_val = env.get(candidate.auth_env_var)
        if not secret_val or not str(secret_val).strip():
            failed_gates.append(GateName.CREDENTIAL_GATE)
            failure_reasons.append(
                f"Required credential environment variable '{candidate.auth_env_var}' is missing or empty."
            )
            gate_details["credential"] = {
                "status": "FAILED",
                "auth_env_var": candidate.auth_env_var,
                "credential_present": False,
            }
        else:
            passed_gates.append(GateName.CREDENTIAL_GATE)
            gate_details["credential"] = {
                "status": "PASSED",
                "auth_env_var": candidate.auth_env_var,
                "credential_present": True,
            }
    else:
        # No auth_env_var specified: allowed only if local or public zero-auth service
        if candidate.access_mode == AccessMode.API and candidate.billing_status not in (
            BillingStatus.FREE_SERVICE_ACCESS,
            BillingStatus.LOCAL_FREE,
        ):
            failed_gates.append(GateName.CREDENTIAL_GATE)
            failure_reasons.append(
                f"API candidate '{candidate.candidate_id}' requires authentication environment variable specification."
            )
            gate_details["credential"] = {
                "status": "FAILED",
                "reason": "Missing auth_env_var specification for commercial API",
            }
        else:
            passed_gates.append(GateName.CREDENTIAL_GATE)
            gate_details["credential"] = {
                "status": "PASSED",
                "reason": "No credentials required for local or public open service",
                "credential_present": True,
            }

    # -------------------------------------------------------------------------
    # Gate 5: Verification State Gate
    # -------------------------------------------------------------------------
    # Ensure failed candidates cannot proceed without explicit re-verification
    if candidate.verification_state == VerificationState.VERIFIED_FAIL:
        failed_gates.append(GateName.VERIFICATION_STATE_GATE)
        failure_reasons.append(
            "Candidate is in VERIFIED_FAIL state and requires remediation before re-probing."
        )
        gate_details["verification_state"] = {"status": "FAILED", "state": candidate.verification_state.value}
    else:
        passed_gates.append(GateName.VERIFICATION_STATE_GATE)
        gate_details["verification_state"] = {"status": "PASSED", "state": candidate.verification_state.value}

    # -------------------------------------------------------------------------
    # Gate 6: User Approval Gate
    # -------------------------------------------------------------------------
    requires_user_approval = bool(candidate.requires_user_approval)
    if requires_user_approval:
        gate_details["user_approval"] = {
            "status": "PENDING_APPROVAL",
            "requires_user_approval": True,
            "note": "Candidate requires explicit operator authorization prior to probing.",
        }
    else:
        gate_details["user_approval"] = {"status": "PASSED", "requires_user_approval": False}

    # -------------------------------------------------------------------------
    # Synthesize Final Decision
    # -------------------------------------------------------------------------
    is_blocked = len(failed_gates) > 0

    if is_blocked:
        overall_status = GateStatus.BLOCKED
        may_proceed = False
        summary_reason = "; ".join(failure_reasons)
    elif requires_user_approval:
        # Pre-probe checks cleared, but blocked pending explicit human sign-off
        overall_status = GateStatus.BLOCKED
        may_proceed = False
        failed_gates.append(GateName.USER_APPROVAL_GATE)
        summary_reason = "All access and cost checks passed, but candidate requires explicit human operator approval before live probing."
    else:
        overall_status = GateStatus.ALLOWED
        may_proceed = True
        summary_reason = "Candidate cleared all pre-probe access, billing, and credential gates."

    return GateDecision(
        candidate_id=candidate.candidate_id,
        status=overall_status,
        reason=summary_reason,
        failed_gates=failed_gates,
        passed_gates=passed_gates,
        may_proceed_to_probe=may_proceed,
        requires_user_approval=requires_user_approval,
        is_routable=False,  # Inviolable: pre-probe gating NEVER grants routing authority
        gate_details=gate_details,
    )

