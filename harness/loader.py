"""Candidate Loader Module for the A4 Verification & Benchmark Harness.

Loads, parses, and validates CandidateRecord manifests from disk.
Strictly offline, deterministic, and contains zero network access or secret leakage.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from harness.models import CandidateRecord

CANDIDATES_DIR = Path(__file__).parent / "candidates"


def load_candidate(
    candidate_id_or_path: Union[str, Path],
    candidates_dir: Optional[Path] = None,
) -> CandidateRecord:
    """Loads and validates a CandidateRecord from a JSON manifest.

    Args:
        candidate_id_or_path: The filename/ID (e.g. 'groq_llama33' or 'groq_llama33.json')
                              or an absolute/relative Path to a manifest file.
        candidates_dir: Optional directory override. Defaults to harness/candidates/.

    Returns:
        Validated CandidateRecord instance.

    Raises:
        FileNotFoundError: If the manifest file cannot be found.
        ValueError: If JSON is invalid or fails CandidateRecord validation.
    """
    target_dir = candidates_dir or CANDIDATES_DIR
    candidate_str = str(candidate_id_or_path)

    # Resolve direct file path if provided
    direct_path = Path(candidate_id_or_path)
    if direct_path.is_file():
        file_path = direct_path
    else:
        if not candidate_str.endswith(".json"):
            candidate_str += ".json"
        file_path = target_dir / candidate_str

    if not file_path.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in candidate manifest '{file_path}': {e}") from e

    return CandidateRecord.model_validate(data)


def load_all_candidates(candidates_dir: Optional[Path] = None) -> List[CandidateRecord]:
    """Loads and validates all CandidateRecord manifests in the candidate directory.

    Manifests are loaded in deterministic sorted alphabetical order.

    Args:
        candidates_dir: Optional directory override. Defaults to harness/candidates/.

    Returns:
        List of validated CandidateRecord instances.
    """
    target_dir = candidates_dir or CANDIDATES_DIR
    if not target_dir.exists():
        return []

    manifest_files = sorted(target_dir.glob("*.json"))
    candidates: List[CandidateRecord] = []
    for file_path in manifest_files:
        candidates.append(load_candidate(file_path))

    return candidates

