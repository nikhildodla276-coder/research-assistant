"""Predefined, controlled mock provider results for deterministic testing.

Contains valid, malformed, schema-violating, and provider-error mock results.
All mocks are 100% deterministic, offline, and require zero real API access.
"""

from typing import Dict, Any
from harness.models import MockProviderResult


def get_mock_result(mock_key: str, candidate_id: str = "mock_provider/test_model") -> MockProviderResult:
    """Factory returning a copy of a controlled MockProviderResult by key."""
    if mock_key not in MOCK_RESULTS:
        raise KeyError(f"Unknown mock result key: '{mock_key}'. Available: {list(MOCK_RESULTS.keys())}")
    base = MOCK_RESULTS[mock_key]
    return MockProviderResult(
        result_id=f"{base['result_id']}_{candidate_id.replace('/', '_')}",
        candidate_id=candidate_id,
        fixture_id=base["fixture_id"],
        http_status=base.get("http_status", 200),
        raw_output=base.get("raw_output"),
        parsed_json=base.get("parsed_json"),
        latency_ms=base.get("latency_ms", 120),
        token_metrics=base.get("token_metrics"),
        error_info=base.get("error_info"),
        is_mock=True,
    )


MOCK_RESULTS: Dict[str, Dict[str, Any]] = {
    # 1. Structured JSON: Perfect valid output
    "VALID_STRUCTURED_JSON": {
        "result_id": "mock_valid_structured_001",
        "fixture_id": "structured_json_v1",
        "http_status": 200,
        "raw_output": '{\n  "role_title": "Senior Backend Engineer",\n  "min_years_experience": 5,\n  "programming_languages": ["Python", "Go"]\n}',
        "parsed_json": {
            "role_title": "Senior Backend Engineer",
            "min_years_experience": 5,
            "programming_languages": ["Python", "Go"],
        },
        "latency_ms": 230,
        "token_metrics": {"prompt_tokens": 85, "completion_tokens": 42, "total_tokens": 127},
    },

    # 2. Structured JSON: Malformed JSON syntax error
    "MALFORMED_JSON": {
        "result_id": "mock_malformed_json_002",
        "fixture_id": "structured_json_v1",
        "http_status": 200,
        "raw_output": '{\n  "role_title": "Senior Backend Engineer",\n  "min_years_experience": 5,\n  "programming_languages": ["Python", "Go"',  # Unclosed bracket
        "parsed_json": None,
        "latency_ms": 190,
        "token_metrics": {"prompt_tokens": 85, "completion_tokens": 38, "total_tokens": 123},
    },

    # 3. Structured JSON: Missing required field ("min_years_experience")
    "MISSING_REQUIRED_FIELD": {
        "result_id": "mock_missing_field_003",
        "fixture_id": "structured_json_v1",
        "http_status": 200,
        "raw_output": '{\n  "role_title": "Senior Backend Engineer",\n  "programming_languages": ["Python", "Go"]\n}',
        "parsed_json": {
            "role_title": "Senior Backend Engineer",
            "programming_languages": ["Python", "Go"],
        },
        "latency_ms": 210,
        "token_metrics": {"prompt_tokens": 85, "completion_tokens": 35, "total_tokens": 120},
    },

    # 4. Structured JSON: Type mismatch ("min_years_experience" is string "5 years", not integer 5)
    "INVALID_SCHEMA_TYPE": {
        "result_id": "mock_invalid_type_004",
        "fixture_id": "structured_json_v1",
        "http_status": 200,
        "raw_output": '{\n  "role_title": "Senior Backend Engineer",\n  "min_years_experience": "five years",\n  "programming_languages": ["Python", "Go"]\n}',
        "parsed_json": {
            "role_title": "Senior Backend Engineer",
            "min_years_experience": "five years",
            "programming_languages": ["Python", "Go"],
        },
        "latency_ms": 225,
        "token_metrics": {"prompt_tokens": 85, "completion_tokens": 40, "total_tokens": 125},
    },

    # 5. Provider-style error: HTTP 429 Rate Limit
    "PROVIDER_ERROR_HTTP_429": {
        "result_id": "mock_err_429_005",
        "fixture_id": "structured_json_v1",
        "http_status": 429,
        "raw_output": '{"error": {"message": "Rate limit exceeded: 30 requests per minute", "code": "rate_limit_exceeded"}}',
        "parsed_json": None,
        "error_info": {
            "error_code": "rate_limit_exceeded",
            "message": "Rate limit exceeded: 30 requests per minute",
        },
        "latency_ms": 45,
    },

    # 6. Provider-style error: HTTP 413 Request Too Large
    "PROVIDER_ERROR_HTTP_413": {
        "result_id": "mock_err_413_006",
        "fixture_id": "structured_json_v1",
        "http_status": 413,
        "raw_output": '{"error": {"message": "Requested 9352 tokens, TPM limit 8000", "code": "request_too_large"}}',
        "parsed_json": None,
        "error_info": {
            "error_code": "request_too_large",
            "message": "Requested 9352 tokens, TPM limit 8000",
        },
        "latency_ms": 60,
    },

    # 7. Provider-style error: Timeout
    "PROVIDER_ERROR_TIMEOUT": {
        "result_id": "mock_err_timeout_007",
        "fixture_id": "structured_json_v1",
        "http_status": 408,
        "raw_output": None,
        "error_info": {
            "error_code": "timeout",
            "message": "Probe execution exceeded deadline of 15.0 seconds",
        },
        "latency_ms": 15020,
    },

    # 8. Provider-style error: HTTP 500 Internal Error
    "PROVIDER_ERROR_HTTP_500": {
        "result_id": "mock_err_500_008",
        "fixture_id": "structured_json_v1",
        "http_status": 500,
        "raw_output": '{"error": {"message": "Internal server error occurred in inference cluster", "code": "internal_server_error"}}',
        "parsed_json": None,
        "error_info": {
            "error_code": "internal_server_error",
            "message": "Internal server error occurred in inference cluster",
        },
        "latency_ms": 520,
    },

    # 9. Basic Generation: Valid concise text response
    "VALID_BASIC_GENERATION": {
        "result_id": "mock_valid_gen_009",
        "fixture_id": "basic_generation_v1",
        "http_status": 200,
        "raw_output": "An application programming interface (API) is a computing interface that defines interactions between multiple software intermediaries.",
        "parsed_json": None,
        "latency_ms": 180,
        "token_metrics": {"prompt_tokens": 22, "completion_tokens": 24, "total_tokens": 46},
    },

    # 10. Basic Generation: Empty text response
    "EMPTY_BASIC_GENERATION": {
        "result_id": "mock_empty_gen_010",
        "fixture_id": "basic_generation_v1",
        "http_status": 200,
        "raw_output": "   ",
        "parsed_json": None,
        "latency_ms": 110,
        "token_metrics": {"prompt_tokens": 22, "completion_tokens": 0, "total_tokens": 22},
    },

    # 11. Claim Grounding: Valid claims citing true evidence IDs [1, 2]
    "VALID_CLAIM_GROUNDING": {
        "result_id": "mock_valid_grounding_011",
        "fixture_id": "claim_grounding_v1",
        "http_status": 200,
        "raw_output": '{\n  "claims": [\n    {"claim_text": "ACME Corp requires 3+ years experience in Python and Kubernetes", "evidence_ids": [1]},\n    {"claim_text": "The position is 100% remote within North America", "evidence_ids": [2]}\n  ]\n}',
        "parsed_json": {
            "claims": [
                {
                    "claim_text": "ACME Corp requires 3+ years experience in Python and Kubernetes",
                    "evidence_ids": [1],
                },
                {
                    "claim_text": "The position is 100% remote within North America",
                    "evidence_ids": [2],
                },
            ]
        },
        "latency_ms": 280,
        "token_metrics": {"prompt_tokens": 110, "completion_tokens": 62, "total_tokens": 172},
    },

    # 12. Claim Grounding: Fabricated citation citing nonexistent ID [99]
    "FABRICATED_CITATION_GROUNDING": {
        "result_id": "mock_fabricated_cit_012",
        "fixture_id": "claim_grounding_v1",
        "http_status": 200,
        "raw_output": '{\n  "claims": [\n    {"claim_text": "Company provides free catered lunch daily", "evidence_ids": [99]}\n  ]\n}',
        "parsed_json": {
            "claims": [
                {
                    "claim_text": "Company provides free catered lunch daily",
                    "evidence_ids": [99],
                }
            ]
        },
        "latency_ms": 260,
        "token_metrics": {"prompt_tokens": 110, "completion_tokens": 30, "total_tokens": 140},
    },
}

