"""Deterministic evaluation engine for the A4 Verification & Benchmark Harness.

Performs schema validation, required-field checks, grounding validation,
and failure classification. Strictly deterministic: zero LLMs, zero natural language
heuristics, and zero network calls.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.models import (
    BenchmarkFixture,
    EvaluationResult,
    EvaluationStatus,
    FailureClass,
    FailureClassification,
    MockProviderResult,
)
from harness.taxonomy import classify_failure

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(fixture_id: str, fixtures_dir: Optional[Path] = None) -> BenchmarkFixture:
    """Loads and validates a versioned BenchmarkFixture from JSON storage."""
    target_dir = fixtures_dir or FIXTURES_DIR
    fixture_path = target_dir / f"{fixture_id}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Benchmark fixture not found: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate strictly against BenchmarkFixture model
    return BenchmarkFixture.model_validate(data)


def validate_schema(instance: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates an instance against a JSON Schema deterministically.
    Uses jsonschema if available; otherwise falls back to a structural Python validator."""
    if not isinstance(instance, dict):
        return False, f"Expected object root, got {type(instance).__name__}"

    if HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=instance, schema=schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, f"Schema validation error at path '{list(e.path)}': {e.message}"
        except Exception as e:
            return False, f"Schema engine error: {str(e)}"

    # Pure Python structural fallback
    required_fields = schema.get("required", [])
    for rf in required_fields:
        if rf not in instance:
            return False, f"Missing required property: '{rf}'"

    props = schema.get("properties", {})
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, spec in props.items():
        if key in instance:
            val = instance[key]
            expected_type_name = spec.get("type")
            if expected_type_name and expected_type_name in type_map:
                expected_py_type = type_map[expected_type_name]
                # In Python bool is a subclass of int; guard against bool masquerading as int
                if expected_type_name == "integer" and isinstance(val, bool):
                    return False, f"Property '{key}' must be integer, got bool"
                if not isinstance(val, expected_py_type):
                    return False, f"Property '{key}' must be {expected_type_name}, got {type(val).__name__}"

            # Array item check
            if expected_type_name == "array" and "items" in spec and isinstance(val, list):
                item_type_name = spec["items"].get("type")
                if item_type_name and item_type_name in type_map:
                    item_py_type = type_map[item_type_name]
                    for idx, item_val in enumerate(val):
                        if not isinstance(item_val, item_py_type):
                            return False, f"Item at '{key}[{idx}]' must be {item_type_name}, got {type(item_val).__name__}"

    return True, None


def evaluate_result(
    fixture: BenchmarkFixture,
    mock_result: MockProviderResult,
) -> EvaluationResult:
    """Evaluates a mock provider result against a fixture deterministically.

    Dispatches to fixture-specific rules, computes quantitative metrics,
    classifies failures, and returns a fully serializable EvaluationResult.
    """
    evaluation_id = f"eval_{fixture.fixture_id}_{mock_result.result_id}"
    rule_evaluations: List[Dict[str, Any]] = []
    metrics: Dict[str, float] = {"latency_ms": float(mock_result.latency_ms)}

    # 1. Check for provider-level errors (HTTP != 200 or error_info present)
    if mock_result.http_status != 200 or mock_result.error_info:
        err = mock_result.error_info or {}
        classification = classify_failure(
            http_status=mock_result.http_status,
            error_code=err.get("error_code"),
            raw_message=err.get("message") or mock_result.raw_output,
        )
        return EvaluationResult(
            evaluation_id=evaluation_id,
            fixture_id=fixture.fixture_id,
            fixture_version=fixture.version,
            candidate_id=mock_result.candidate_id,
            status=EvaluationStatus.FAIL,
            metrics={"latency_ms": float(mock_result.latency_ms), "success_rate": 0.0},
            failure_classification=classification,
            rule_evaluations=[{
                "rule": "http_success",
                "passed": False,
                "detail": f"HTTP {mock_result.http_status}: {classification.failure_class.value}",
            }],
        )

    # 2. Dispatch to specific fixture family evaluators
    if fixture.fixture_id.startswith("basic_generation"):
        return _evaluate_basic_generation(fixture, mock_result, evaluation_id)

    if fixture.fixture_id.startswith("structured_json"):
        return _evaluate_structured_json(fixture, mock_result, evaluation_id)

    if fixture.fixture_id.startswith("claim_grounding"):
        return _evaluate_claim_grounding(fixture, mock_result, evaluation_id)

    # Generic rule-based evaluation fallback
    return _evaluate_generic_fixture(fixture, mock_result, evaluation_id)


def _evaluate_basic_generation(
    fixture: BenchmarkFixture,
    mock_result: MockProviderResult,
    evaluation_id: str,
) -> EvaluationResult:
    """Evaluates basic text completion for non-emptiness and length bounds."""
    text = (mock_result.raw_output or "").strip()
    rule_evals = []
    metrics = {
        "latency_ms": float(mock_result.latency_ms),
        "char_count": float(len(text)),
    }

    # Rule: non_empty
    is_non_empty = len(text) > 0
    rule_evals.append({
        "rule": "non_empty",
        "passed": is_non_empty,
        "detail": f"Length: {len(text)} chars",
    })

    if not is_non_empty:
        return EvaluationResult(
            evaluation_id=evaluation_id,
            fixture_id=fixture.fixture_id,
            fixture_version=fixture.version,
            candidate_id=mock_result.candidate_id,
            status=EvaluationStatus.FAIL,
            metrics=metrics,
            failure_classification=classify_failure(
                error_code="empty_response",
                raw_message="Model returned empty or whitespace-only response",
            ),
            rule_evaluations=rule_evals,
        )

    # Rule: min_length / max_length
    min_chars = 20
    max_chars = 500
    for r in fixture.evaluation_rules:
        if r.get("rule") == "min_length":
            min_chars = r.get("min_chars", min_chars)
        elif r.get("rule") == "max_length":
            max_chars = r.get("max_chars", max_chars)

    min_pass = len(text) >= min_chars
    max_pass = len(text) <= max_chars
    rule_evals.append({"rule": "min_length", "passed": min_pass, "detail": f"{len(text)} >= {min_chars}"})
    rule_evals.append({"rule": "max_length", "passed": max_pass, "detail": f"{len(text)} <= {max_chars}"})

    all_passed = is_non_empty and min_pass and max_pass
    metrics["conformance_score"] = 1.0 if all_passed else 0.0

    return EvaluationResult(
        evaluation_id=evaluation_id,
        fixture_id=fixture.fixture_id,
        fixture_version=fixture.version,
        candidate_id=mock_result.candidate_id,
        status=EvaluationStatus.PASS if all_passed else EvaluationStatus.FAIL,
        metrics=metrics,
        failure_classification=None if all_passed else classify_failure(
            error_code="length_bound_violation",
            raw_message=f"Output length {len(text)} outside allowed bounds [{min_chars}, {max_chars}]",
        ),
        rule_evaluations=rule_evals,
    )


def _evaluate_structured_json(
    fixture: BenchmarkFixture,
    mock_result: MockProviderResult,
    evaluation_id: str,
) -> EvaluationResult:
    """Evaluates structured JSON output against schema and required fields."""
    rule_evals = []
    metrics = {"latency_ms": float(mock_result.latency_ms)}

    # Parse JSON
    parsed = mock_result.parsed_json
    if parsed is None:
        if mock_result.raw_output:
            try:
                parsed = json.loads(mock_result.raw_output)
            except Exception as e:
                return EvaluationResult(
                    evaluation_id=evaluation_id,
                    fixture_id=fixture.fixture_id,
                    fixture_version=fixture.version,
                    candidate_id=mock_result.candidate_id,
                    status=EvaluationStatus.FAIL,
                    metrics={"schema_conformance": 0.0, "latency_ms": float(mock_result.latency_ms)},
                    failure_classification=classify_failure(
                        error_code="invalid_json",
                        raw_message=f"JSON decode failed: {str(e)}",
                    ),
                    rule_evaluations=[{"rule": "valid_json", "passed": False, "detail": str(e)}],
                )
        else:
            return EvaluationResult(
                evaluation_id=evaluation_id,
                fixture_id=fixture.fixture_id,
                fixture_version=fixture.version,
                candidate_id=mock_result.candidate_id,
                status=EvaluationStatus.FAIL,
                metrics={"schema_conformance": 0.0, "latency_ms": float(mock_result.latency_ms)},
                failure_classification=classify_failure(
                    error_code="empty_body",
                    raw_message="Mock result provided neither parsed_json nor raw_output",
                ),
                rule_evaluations=[{"rule": "valid_json", "passed": False, "detail": "Missing output"}],
            )

    rule_evals.append({"rule": "valid_json", "passed": True, "detail": "JSON parsed successfully"})

    # Check against expected structure schema
    if fixture.expected_structure:
        valid_schema, err_detail = validate_schema(parsed, fixture.expected_structure)
        rule_evals.append({
            "rule": "schema_conformance",
            "passed": valid_schema,
            "detail": err_detail or "Schema matches expected structure",
        })

        if not valid_schema:
            metrics["schema_conformance"] = 0.0
            metrics["required_field_score"] = 0.0
            return EvaluationResult(
                evaluation_id=evaluation_id,
                fixture_id=fixture.fixture_id,
                fixture_version=fixture.version,
                candidate_id=mock_result.candidate_id,
                status=EvaluationStatus.FAIL,
                metrics=metrics,
                failure_classification=classify_failure(
                    error_code="schema_failure",
                    raw_message=err_detail,
                ),
                rule_evaluations=rule_evals,
            )

    # Required field verification
    required_fields = []
    if fixture.expected_structure:
        required_fields = fixture.expected_structure.get("required", [])

    for r in fixture.evaluation_rules:
        if r.get("rule") == "required_fields":
            required_fields = list(set(required_fields + r.get("fields", [])))

    if required_fields:
        present = [f for f in required_fields if f in parsed]
        field_score = len(present) / len(required_fields)
        all_req_present = len(present) == len(required_fields)
        metrics["required_field_score"] = float(field_score)
        rule_evals.append({
            "rule": "required_fields",
            "passed": all_req_present,
            "detail": f"{len(present)}/{len(required_fields)} fields present",
        })
        if not all_req_present:
            metrics["schema_conformance"] = 0.0
            return EvaluationResult(
                evaluation_id=evaluation_id,
                fixture_id=fixture.fixture_id,
                fixture_version=fixture.version,
                candidate_id=mock_result.candidate_id,
                status=EvaluationStatus.FAIL,
                metrics=metrics,
                failure_classification=classify_failure(
                    error_code="missing_required_fields",
                    raw_message=f"Missing fields: {set(required_fields) - set(present)}",
                ),
                rule_evaluations=rule_evals,
            )
    else:
        metrics["required_field_score"] = 1.0

    metrics["schema_conformance"] = 1.0

    return EvaluationResult(
        evaluation_id=evaluation_id,
        fixture_id=fixture.fixture_id,
        fixture_version=fixture.version,
        candidate_id=mock_result.candidate_id,
        status=EvaluationStatus.PASS,
        metrics=metrics,
        failure_classification=None,
        rule_evaluations=rule_evals,
    )


def _evaluate_claim_grounding(
    fixture: BenchmarkFixture,
    mock_result: MockProviderResult,
    evaluation_id: str,
) -> EvaluationResult:
    """Evaluates factual claims and asserts evidence IDs exist in fixture context."""
    rule_evals = []
    metrics = {"latency_ms": float(mock_result.latency_ms)}

    parsed = mock_result.parsed_json
    if parsed is None and mock_result.raw_output:
        try:
            parsed = json.loads(mock_result.raw_output)
        except Exception as e:
            return EvaluationResult(
                evaluation_id=evaluation_id,
                fixture_id=fixture.fixture_id,
                fixture_version=fixture.version,
                candidate_id=mock_result.candidate_id,
                status=EvaluationStatus.FAIL,
                metrics={"grounding_score": 0.0, "citation_precision": 0.0},
                failure_classification=classify_failure(error_code="invalid_json", raw_message=str(e)),
                rule_evaluations=[{"rule": "valid_json", "passed": False, "detail": str(e)}],
            )

    if not isinstance(parsed, dict) or "claims" not in parsed:
        return EvaluationResult(
            evaluation_id=evaluation_id,
            fixture_id=fixture.fixture_id,
            fixture_version=fixture.version,
            candidate_id=mock_result.candidate_id,
            status=EvaluationStatus.FAIL,
            metrics={"grounding_score": 0.0, "citation_precision": 0.0},
            failure_classification=classify_failure(
                error_code="schema_failure",
                raw_message="Missing top-level 'claims' array",
            ),
            rule_evaluations=[{"rule": "claims_key_present", "passed": False, "detail": "Missing claims key"}],
        )

    claims = parsed.get("claims", [])
    if not isinstance(claims, list) or not claims:
        return EvaluationResult(
            evaluation_id=evaluation_id,
            fixture_id=fixture.fixture_id,
            fixture_version=fixture.version,
            candidate_id=mock_result.candidate_id,
            status=EvaluationStatus.FAIL,
            metrics={"grounding_score": 0.0, "citation_precision": 0.0},
            failure_classification=classify_failure(
                error_code="empty_claims",
                raw_message="'claims' array is empty",
            ),
            rule_evaluations=[{"rule": "non_empty_claims", "passed": False, "detail": "0 claims"}],
        )

    # Retrieve valid evidence IDs from fixture rule
    valid_ids = {1, 2}
    for r in fixture.evaluation_rules:
        if r.get("rule") == "valid_evidence_citations":
            valid_ids = set(r.get("valid_ids", [1, 2]))

    total_citations = 0
    valid_citations = 0
    grounded_claims = 0

    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        c_ids = claim.get("evidence_ids", [])
        if not isinstance(c_ids, list):
            c_ids = [c_ids]
        
        claim_has_invalid = False
        for cid in c_ids:
            total_citations += 1
            if cid in valid_ids:
                valid_citations += 1
            else:
                claim_has_invalid = True

        if c_ids and not claim_has_invalid:
            grounded_claims += 1

    precision = (valid_citations / total_citations) if total_citations > 0 else 0.0
    grounding_score = (grounded_claims / len(claims)) if claims else 0.0

    metrics["citation_precision"] = float(precision)
    metrics["grounding_score"] = float(grounding_score)
    metrics["claim_count"] = float(len(claims))

    all_grounded = precision == 1.0 and grounding_score == 1.0
    rule_evals.append({
        "rule": "valid_evidence_citations",
        "passed": all_grounded,
        "detail": f"Precision: {precision:.2f}, Grounding: {grounding_score:.2f} ({valid_citations}/{total_citations} valid)",
    })

    return EvaluationResult(
        evaluation_id=evaluation_id,
        fixture_id=fixture.fixture_id,
        fixture_version=fixture.version,
        candidate_id=mock_result.candidate_id,
        status=EvaluationStatus.PASS if all_grounded else EvaluationStatus.FAIL,
        metrics=metrics,
        failure_classification=None if all_grounded else classify_failure(
            error_code="grounding_failure",
            raw_message=f"Fabricated citation IDs detected. Valid IDs: {valid_ids}",
        ),
        rule_evaluations=rule_evals,
    )


def _evaluate_generic_fixture(
    fixture: BenchmarkFixture,
    mock_result: MockProviderResult,
    evaluation_id: str,
) -> EvaluationResult:
    """Fallback evaluator for arbitrary declarative fixtures."""
    has_output = bool(mock_result.raw_output or mock_result.parsed_json)
    return EvaluationResult(
        evaluation_id=evaluation_id,
        fixture_id=fixture.fixture_id,
        fixture_version=fixture.version,
        candidate_id=mock_result.candidate_id,
        status=EvaluationStatus.PASS if has_output else EvaluationStatus.FAIL,
        metrics={"latency_ms": float(mock_result.latency_ms), "success": 1.0 if has_output else 0.0},
        failure_classification=None if has_output else classify_failure(raw_message="Empty generic output"),
        rule_evaluations=[{"rule": "has_output", "passed": has_output, "detail": "Output present"}],
    )

