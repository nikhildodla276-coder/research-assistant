"""Benchmark Runner and Evidence Collection Engine for Phase A4.5.

Executes fixed, versioned benchmark task fixtures against approved candidate providers
through A4.3 pre-probe safety gates and A4.4 provider adapters.
Collects objective, deterministic evidence with strict source provenance tracking.
Zero subjective ranking, zero runtime routing, zero secret persistence.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from harness.adapters.base import TransportCallable
from harness.models import (
    BenchmarkEvidenceRecord,
    BenchmarkTask,
    CandidateRecord,
    FailureClass,
    FailureClassification,
    ProbeRequest,
    ProbeResult,
    ProbeStatus,
    ProvenanceRecord,
    ProvenanceState,
)
from harness.probing import run_candidate_probe
from harness.provenance import (
    build_provenance_record,
    extract_model_claimed_citations,
)

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def load_benchmark_task(
    task_path_or_id: Union[str, Path],
    suite_dir: Optional[Path] = None,
) -> BenchmarkTask:
    """Loads and validates a single BenchmarkTask fixture from JSON storage."""
    target_dir = suite_dir or (BENCHMARKS_DIR / "sih_cadastral_v1")
    task_str = str(task_path_or_id)

    direct_path = Path(task_path_or_id)
    if direct_path.is_file():
        file_path = direct_path
    else:
        if not task_str.endswith(".json"):
            task_str += ".json"
        file_path = target_dir / task_str

    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark task fixture not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in benchmark task '{file_path}': {e}") from e

    return BenchmarkTask.model_validate(data)


def load_benchmark_suite(
    suite_dir_or_name: Union[str, Path] = "sih_cadastral_v1",
) -> List[BenchmarkTask]:
    """Loads all BenchmarkTask fixtures in a benchmark suite directory in sorted order."""
    if isinstance(suite_dir_or_name, Path):
        target_dir = suite_dir_or_name
    else:
        target_dir = BENCHMARKS_DIR / str(suite_dir_or_name)

    if not target_dir.exists():
        raise FileNotFoundError(f"Benchmark suite directory not found: {target_dir}")

    task_files = sorted(target_dir.glob("*.json"))
    tasks: List[BenchmarkTask] = []
    for tf in task_files:
        tasks.append(load_benchmark_task(tf))

    return tasks


def evaluate_task_contract(
    task: BenchmarkTask,
    probe_result: ProbeResult,
) -> Tuple[bool, Dict[str, Any], ProvenanceRecord]:
    """Deterministically validates whether a probe result satisfied the task contract.

    Returns:
        (contract_satisfied, evaluation_results_dict, provenance_record)
    """
    if probe_result.status != ProbeStatus.SUCCESS or not probe_result.normalized_response:
        eval_details = {
            "status": probe_result.status.value,
            "contract_passed": False,
            "reason": "Probe did not succeed or returned empty response",
        }
        prov = ProvenanceRecord(provenance_status=ProvenanceState.NOT_AVAILABLE)
        return False, eval_details, prov

    text = probe_result.normalized_response.strip()
    contract = task.expected_output_contract
    expected_format = contract.get("format", "text").lower()
    eval_details: Dict[str, Any] = {
        "expected_format": expected_format,
        "char_count": len(text),
    }

    # -------------------------------------------------------------------------
    # Format 1: Structured JSON Contract
    # -------------------------------------------------------------------------
    if expected_format == "json":
        parsed_data = None
        # Check if text is valid JSON or contained inside markdown block
        clean_text = text
        if clean_text.startswith("```json") and clean_text.endswith("```"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```") and clean_text.endswith("```"):
            clean_text = clean_text[3:-3].strip()

        try:
            parsed_data = json.loads(clean_text)
            eval_details["valid_json"] = True
        except Exception as e:
            eval_details["valid_json"] = False
            eval_details["json_error"] = str(e)
            eval_details["contract_passed"] = False
            prov = build_provenance_record(response_text=text)
            return False, eval_details, prov

        # Validate required keys
        required_keys = contract.get("required_keys", [])
        missing_keys = []
        if isinstance(parsed_data, dict):
            for k in required_keys:
                if k not in parsed_data:
                    missing_keys.append(k)
        else:
            missing_keys = required_keys

        eval_details["required_keys_checked"] = required_keys
        eval_details["missing_keys"] = missing_keys
        keys_satisfied = len(missing_keys) == 0

        # Validate min_items for list fields if specified
        min_items = contract.get("min_items")
        min_items_satisfied = True
        if min_items is not None:
            # Check if any list in parsed_data satisfies min_items
            list_lens = [len(v) for v in parsed_data.values() if isinstance(v, list)]
            if not list_lens or max(list_lens) < min_items:
                min_items_satisfied = False
            eval_details["min_items_checked"] = min_items
            eval_details["min_items_satisfied"] = min_items_satisfied

        contract_passed = keys_satisfied and min_items_satisfied
        eval_details["contract_passed"] = contract_passed

        prov = build_provenance_record(response_text=text)
        return contract_passed, eval_details, prov

    # -------------------------------------------------------------------------
    # Format 2: Text with Citation / Source Grounding Contract
    # -------------------------------------------------------------------------
    required_citations = contract.get("required_citations", [])
    citations_present = []
    missing_citations = []
    for cit in required_citations:
        if cit.lower() in text.lower():
            citations_present.append(cit)
        else:
            missing_citations.append(cit)

    eval_details["required_citations"] = required_citations
    eval_details["citations_present"] = citations_present
    eval_details["missing_citations"] = missing_citations

    contract_passed = len(missing_citations) == 0 if required_citations else len(text) > 0
    eval_details["contract_passed"] = contract_passed

    prov = build_provenance_record(response_text=text)
    return contract_passed, eval_details, prov


def run_benchmark_task(
    task: BenchmarkTask,
    candidate: CandidateRecord,
    environ: Optional[Dict[str, str]] = None,
    transport: Optional[TransportCallable] = None,
) -> BenchmarkEvidenceRecord:
    """Executes a single benchmark task against a candidate and returns structured evidence.

    Follows strict pipeline:
    Task Fixture → A4.3 Gate → A4.4 Probing Harness → Contract Evaluation → Evidence Record.
    """
    probe_id = f"bm_{task.task_id}_{uuid.uuid4().hex[:8]}"

    probe_req = ProbeRequest(
        probe_id=probe_id,
        candidate_id=candidate.candidate_id,
        prompt=task.input_prompt,
        expected_output_format=task.expected_output_contract.get("format", "text"),
        metadata={
            "benchmark_id": task.benchmark_id,
            "task_id": task.task_id,
            "task_version": task.task_version,
        },
    )

    # Invoke probing harness (which unconditionally enforces A4.3 pre-probe gates)
    probe_result = run_candidate_probe(
        candidate=candidate,
        request=probe_req,
        environ=environ,
        transport=transport,
    )

    # Evaluate task contract deterministically
    contract_satisfied, eval_results, prov_record = evaluate_task_contract(task, probe_result)

    evidence_id = f"ev_{task.task_id}_{candidate.candidate_id}_{uuid.uuid4().hex[:6]}"

    return BenchmarkEvidenceRecord(
        evidence_id=evidence_id,
        benchmark_id=task.benchmark_id,
        task_id=task.task_id,
        task_version=task.task_version,
        task_category=task.task_category,
        candidate_id=candidate.candidate_id,
        probe_id=probe_result.probe_id,
        execution_status=probe_result.status,
        contract_satisfied=contract_satisfied,
        sanitized_response=probe_result.normalized_response,
        evaluation_results=eval_results,
        failure_classification=probe_result.error_classification,
        latency_ms=probe_result.latency_ms,
        usage_metadata=probe_result.usage_metadata,
        provenance_metadata=prov_record,
        input_determinism=True,
        output_determinism=False,
    )


def run_benchmark_suite(
    benchmark_tasks: List[BenchmarkTask],
    candidate: CandidateRecord,
    environ: Optional[Dict[str, str]] = None,
    transport: Optional[TransportCallable] = None,
) -> List[BenchmarkEvidenceRecord]:
    """Runs a series of benchmark tasks sequentially against a single candidate.

    Guarantees isolation:
    Each task execution is independent. One task's output is NEVER fed into another.
    """
    records: List[BenchmarkEvidenceRecord] = []
    for task in benchmark_tasks:
        rec = run_benchmark_task(
            task=task,
            candidate=candidate,
            environ=environ,
            transport=transport,
        )
        records.append(rec)
    return records

