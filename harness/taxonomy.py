"""Deterministic failure classifier for provider, probe, and schema errors.

Translates raw HTTP status codes, exception types, and error strings into the
canonical 17-class failure taxonomy defined in A4.1.
"""

from typing import Optional
from harness.models import FailureClass, FailureClassification


def classify_failure(
    http_status: Optional[int] = None,
    error_code: Optional[str] = None,
    raw_message: Optional[str] = None,
) -> FailureClassification:
    """Deterministically categorizes an execution failure into a standard FailureClassification."""
    code_str = (error_code or "").lower()
    msg_str = (raw_message or "").lower()
    combined_err = f"{code_str} {msg_str}".strip()

    # 1. Authentication failures (HTTP 401)
    if http_status == 401 or any(k in combined_err for k in ["invalid_api_key", "unauthorized", "auth_failure", "invalid api key"]):
        return FailureClassification(
            failure_class=FailureClass.AUTHENTICATION_FAILURE,
            http_status=http_status or 401,
            error_code=error_code,
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="VERIFY_API_CREDENTIALS_IN_ENVIRONMENT",
        )

    # 2. Access / Sandbox blocked (HTTP 403)
    if http_status == 403 or any(k in combined_err for k in ["forbidden", "permission_denied", "access_blocked", "access denied"]):
        return FailureClassification(
            failure_class=FailureClass.ACCESS_BLOCKED,
            http_status=http_status or 403,
            error_code=error_code,
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="INSPECT_FIREWALL_SANDBOX_POLICY",
        )

    # 3. Model Not Found (HTTP 404)
    if http_status == 404 or any(k in combined_err for k in ["model_not_found", "endpoint_not_found", "does not exist"]):
        return FailureClassification(
            failure_class=FailureClass.MODEL_NOT_FOUND,
            http_status=http_status or 404,
            error_code=error_code,
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="DISCOVER_CURRENT_ACTIVE_MODEL_IDENTIFIER",
        )

    # 4. Rate Limited (HTTP 429)
    if http_status == 429 or any(k in combined_err for k in ["rate_limit", "too many requests", "quota_exceeded", "resource_exhausted"]):
        return FailureClassification(
            failure_class=FailureClass.RATE_LIMITED,
            http_status=http_status or 429,
            error_code=error_code or "rate_limit_exceeded",
            raw_message=raw_message,
            is_retryable=True,
            recommended_action="APPLY_EXPONENTIAL_BACKOFF_OR_PAUSE",
        )

    # 5. Request Too Large / Input ceiling exceeded (HTTP 413)
    if http_status == 413 or any(k in combined_err for k in ["request_too_large", "payload too large", "tpm limit", "token limit exceeded", "max_tokens_exceeded"]):
        return FailureClassification(
            failure_class=FailureClass.REQUEST_TOO_LARGE,
            http_status=http_status or 413,
            error_code=error_code or "request_too_large",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="ENFORCE_LOWER_SYNTHESIS_INPUT_BUDGET",
        )

    # 6. Timeout
    if any(k in combined_err for k in ["timeout", "timed out", "deadline_exceeded", "connecttimeout", "readtimeout"]):
        return FailureClassification(
            failure_class=FailureClass.TIMEOUT,
            http_status=http_status or 408,
            error_code=error_code or "timeout",
            raw_message=raw_message,
            is_retryable=True,
            recommended_action="RETRY_ONCE_WITH_ISOLATED_DEADLINE",
        )

    # 7. Provider Errors (HTTP 500, 502, 503, 504)
    if (http_status and 500 <= http_status <= 599) or any(k in combined_err for k in ["internal_server_error", "bad_gateway", "service_unavailable", "upstream_error"]):
        return FailureClassification(
            failure_class=FailureClass.PROVIDER_ERROR,
            http_status=http_status or 500,
            error_code=error_code or "provider_error",
            raw_message=raw_message,
            is_retryable=True,
            recommended_action="RECORD_PROVIDER_HEALTH_INCIDENT",
        )

    # 8. Schema Failure (Validation error on output)
    if any(k in combined_err for k in ["schema_failure", "jsonschema", "validation_error", "schema_mismatch", "type_error"]):
        return FailureClassification(
            failure_class=FailureClass.SCHEMA_FAILURE,
            http_status=http_status,
            error_code=error_code or "schema_validation_failure",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="CHECK_PROMPT_INSTRUCTIONS_AND_SCHEMA_CONSTRAINTS",
        )

    # 9. Invalid Response (Malformed JSON or syntax error)
    if any(k in combined_err for k in ["jsondecodeerror", "invalid_json", "malformed_json", "empty_body", "unparseable"]):
        return FailureClassification(
            failure_class=FailureClass.INVALID_RESPONSE,
            http_status=http_status,
            error_code=error_code or "invalid_response",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="CHECK_JSON_EXTRACTION_BOUNDARY",
        )

    # 10. Capability Failure (Refusal)
    if any(k in combined_err for k in ["refusal", "as an ai", "cannot fulfill", "safety_violation"]):
        return FailureClassification(
            failure_class=FailureClass.CAPABILITY_FAILURE,
            http_status=http_status,
            error_code=error_code or "capability_refusal",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="REFORMULATE_PROMPT_OR_EXCLUDE_MODEL",
        )

    # 11. Grounding Failure
    if any(k in combined_err for k in ["grounding_failure", "unsupported_claim", "fabricated_citation"]):
        return FailureClassification(
            failure_class=FailureClass.GROUNDING_FAILURE,
            http_status=http_status,
            error_code=error_code or "grounding_failure",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="REJECT_UNVERIFIED_ASSERTIONS",
        )

    # 12. Network Failure
    if any(k in combined_err for k in ["connection_refused", "connectionrefused", "refused", "connection_reset", "connectionreset", "dns_lookup_failed", "network_unreachable", "connectionerror", "connecterror", "urlerror", "remotedisconnected", "newconnectionerror", "socket.error"]):
        return FailureClassification(
            failure_class=FailureClass.NETWORK_FAILURE,
            http_status=http_status,
            error_code=error_code or "network_failure",
            raw_message=raw_message,
            is_retryable=True,
            recommended_action="CHECK_LOCAL_HOST_NETWORK_CONFIG",
        )

    # 13. Configuration Error
    if any(k in combined_err for k in ["configuration_error", "config_error", "missing_configuration", "invalid_config"]):
        return FailureClassification(
            failure_class=FailureClass.CONFIGURATION_ERROR,
            http_status=http_status,
            error_code=error_code or "configuration_error",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="VERIFY_PROVIDER_CANDIDATE_CONFIGURATION",
        )

    # 14. Unsupported Request
    if any(k in combined_err for k in ["unsupported_request", "not_implemented", "unsupported_provider", "unsupported_access_mode", "not_ready"]):
        return FailureClassification(
            failure_class=FailureClass.UNSUPPORTED_REQUEST,
            http_status=http_status,
            error_code=error_code or "unsupported_request",
            raw_message=raw_message,
            is_retryable=False,
            recommended_action="VERIFY_ADAPTER_SUPPORT_FOR_CANDIDATE",
        )

    # Default fallback
    return FailureClassification(
        failure_class=FailureClass.UNKNOWN_FAILURE,
        http_status=http_status,
        error_code=error_code or "unknown",
        raw_message=raw_message,
        is_retryable=False,
        recommended_action="MANUAL_INVESTIGATION_REQUIRED",
    )


