"""Provenance tracking and citation audit module for Phase A4.5.

Strictly distinguishes between:
1. Sources claimed by a model in response text (MODEL_CLAIM_ONLY)
2. Sources actually retrieved by the system pipeline (RETRIEVED_BY_SYSTEM)

Guarantees that a model-generated citation or URL is never confused with
system-verified retrieval evidence.
"""

import re
from typing import Any, Dict, List, Optional

from harness.models import ProvenanceRecord, ProvenanceState


CITATION_PATTERNS = [
    re.compile(r"\[(SRC-[A-Za-z0-9_\-]+)\]", re.IGNORECASE),
    re.compile(r"\[([0-9]+)\]"),
    re.compile(r"https?://[^\s)\]\"'>]+", re.IGNORECASE),
]


def extract_model_claimed_citations(text: Optional[str]) -> List[Dict[str, Any]]:
    """Extracts citation tokens, tags, and URLs asserted by a model in its text output."""
    if not text:
        return []

    claimed: List[Dict[str, Any]] = []
    seen: set = set()

    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0)
            if raw not in seen:
                seen.add(raw)
                cit_type = "url" if raw.startswith("http") else "tag"
                token = match.group(1) if match.groups() else raw
                claimed.append({
                    "raw_citation": raw,
                    "token": token,
                    "citation_type": cit_type,
                    "verification_status": "UNVERIFIED_MODEL_CLAIM",
                })

    return claimed


def build_provenance_record(
    response_text: Optional[str] = None,
    system_sources: Optional[List[Dict[str, Any]]] = None,
    retrieval_provider: Optional[str] = None,
    retrieval_metadata: Optional[Dict[str, Any]] = None,
    evidence_text: Optional[str] = None,
) -> ProvenanceRecord:
    """Builds a structured ProvenanceRecord enforcing the retrieval vs model assertion boundary."""
    model_claims = extract_model_claimed_citations(response_text)
    sys_sources = system_sources or []
    ret_meta = retrieval_metadata or {}

    source_ids: List[str] = []
    for s in sys_sources:
        sid = s.get("url") or s.get("id") or s.get("source_id")
        if sid:
            source_ids.append(str(sid))

    # Determine explicit provenance standing
    if sys_sources and len(sys_sources) > 0:
        status = ProvenanceState.RETRIEVED_BY_SYSTEM
    elif model_claims and len(model_claims) > 0:
        status = ProvenanceState.MODEL_CLAIM_ONLY
    else:
        status = ProvenanceState.NOT_AVAILABLE

    return ProvenanceRecord(
        model_claimed_sources=model_claims,
        system_retrieved_sources=sys_sources,
        retrieval_provider=retrieval_provider,
        retrieval_metadata=ret_meta,
        source_identifiers=source_ids,
        evidence_text=evidence_text,
        provenance_status=status,
    )

