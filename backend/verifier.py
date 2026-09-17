import json
import re
from typing import Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    from models import Claim, EvidenceItem, Source, VerificationStatus
except ImportError:
    from backend.models import Claim, EvidenceItem, Source, VerificationStatus

EXTRACTION_SYSTEM_PROMPT = """You are a meticulous technical analyst extracting discrete factual claims from a research report.

GUIDELINES:
1. Extract atomic, verifiable factual assertions made in the report.
2. Avoid compound statements; separate multiple claims into independent items.
3. For each claim, identify the 1-based evidence ID numbers (e.g., [1], [2]) cited or linked to it in the text.
4. If a claim has no citations or supporting evidence in the text, leave evidence_ids as an empty list [].

Respond ONLY with a JSON array in the following format:
[
  {
    "claim_id": 1,
    "claim_text": "Atomic factual assertion made in the report.",
    "evidence_ids": [1]
  }
]
"""

VERIFICATION_SYSTEM_PROMPT = """You are an evidence-bounded verification judge.

Your task is to determine whether the provided Evidence Excerpt(s) strictly support the Claim.

RULES:
1. Base your evaluation ONLY on the provided evidence text. Do NOT use outside knowledge or extrapolate.
2. Select exactly one verification status:
   - "supported": The evidence directly and explicitly confirms the factual assertion.
   - "partially_supported": The evidence confirms part of the assertion or related concept, but does not substantiate all specific claims, metrics, or scope.
   - "not_supported": The evidence contradicts, does not mention, or fails to substantiate the claim.
   - "uncertain": The evidence is ambiguous, vague, or inconclusive.
3. Provide a concise, objective reason explaining how the evidence supports or fails to support the claim.
4. Assign a confidence score between 0.0 and 1.0.

Respond ONLY with a JSON object in this schema:
{
  "status": "supported" | "partially_supported" | "not_supported" | "uncertain",
  "reason": "Concise factual reason citing evidence contents.",
  "confidence": 0.95
}
"""


def _extract_json_block(text: str) -> str:
    """Extracts JSON block from potential markdown code fences or surrounding text.
    Correctly supports both top-level JSON objects ({...}) and arrays ([...]) by
    inspecting whichever delimiter appears earlier in the text."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return text.strip()

    if start_brace != -1 and start_bracket != -1:
        if start_bracket < start_brace:
            end_bracket = text.rfind("]")
            if end_bracket > start_bracket:
                return text[start_bracket : end_bracket + 1].strip()
        else:
            end_brace = text.rfind("}")
            if end_brace > start_brace:
                return text[start_brace : end_brace + 1].strip()
    elif start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket > start_bracket:
            return text[start_bracket : end_bracket + 1].strip()
    elif start_brace != -1:
        end_brace = text.rfind("}")
        if end_brace > start_brace:
            return text[start_brace : end_brace + 1].strip()

    return text.strip()


def extract_claims(
    llm: ChatGroq,
    content: str,
    valid_evidence_ids: List[int],
) -> List[Claim]:
    """Extracts atomic claims from report content and validates evidence references."""
    human_prompt = f"RESEARCH REPORT CONTENT:\n{content}\n\nExtract 2 to 5 atomic factual claims and their cited evidence IDs."
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_json = _extract_json_block(response.content)
        data = json.loads(raw_json)
    except Exception as err:
        # Edge Case A: Malformed extraction output -> Controlled result, does not crash, preserves error
        return [
            Claim(
                claim_id=1,
                claim_text="Report synthesis generated factual assertions.",
                evidence_ids=[],
                invalid_evidence_ids=[],
                verification_status=VerificationStatus.UNCERTAIN,
                verification_reason=f"Claim extraction output malformed: {type(err).__name__} ({str(err)})",
            )
        ]

    if not isinstance(data, list) or len(data) == 0:
        return [
            Claim(
                claim_id=1,
                claim_text="Report synthesis generated factual assertions.",
                evidence_ids=[],
                invalid_evidence_ids=[],
                verification_status=VerificationStatus.UNCERTAIN,
                verification_reason="No structured claims parsed from LLM extraction response.",
            )
        ]

    valid_id_set = set(valid_evidence_ids)
    claims: List[Claim] = []

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        raw_ids = item.get("evidence_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = [raw_ids] if isinstance(raw_ids, int) else []

        valid_ids: List[int] = []
        invalid_ids: List[int] = []

        for eid in raw_ids:
            if isinstance(eid, int) and eid in valid_id_set:
                valid_ids.append(eid)
            elif isinstance(eid, int):
                # Edge Case B: Claim references nonexistent evidence
                invalid_ids.append(eid)

        claim_text = str(item.get("claim_text", f"Claim {idx}")).strip()

        claim = Claim(
            claim_id=item.get("claim_id", idx),
            claim_text=claim_text,
            evidence_ids=valid_ids,
            invalid_evidence_ids=invalid_ids,
            verification_status=VerificationStatus.UNCERTAIN,
            verification_reason="",
        )

        # Edge Case B & C: If no valid evidence was linked, explicitly flag as NOT_SUPPORTED
        if not valid_ids:
            claim.verification_status = VerificationStatus.NOT_SUPPORTED
            if invalid_ids:
                claim.verification_reason = f"Claim references nonexistent evidence IDs {invalid_ids}; no valid evidence supplied."
            else:
                claim.verification_reason = "No supporting evidence items were linked to this claim."

        claims.append(claim)

    return claims


def verify_single_claim(
    llm: ChatGroq,
    claim: Claim,
    evidence_map: Dict[int, EvidenceItem],
    valid_source_ids: Optional[Set[int]] = None,
) -> Claim:
    """Compares a single claim against its linked evidence items, validating source provenance."""
    # Edge Cases B & C: No valid evidence linked -> already flagged, cannot become SUPPORTED
    if not claim.evidence_ids:
        if claim.invalid_evidence_ids:
            claim.verification_status = VerificationStatus.NOT_SUPPORTED
            claim.verification_reason = (
                f"Claim references invalid evidence IDs {claim.invalid_evidence_ids}; rejected without support."
            )
        else:
            claim.verification_status = VerificationStatus.NOT_SUPPORTED
            claim.verification_reason = "No supporting evidence provided or retrieved for this claim."
        return claim

    # Assemble linked evidence snippets, checking for detached evidence (Failure E / Test G)
    evidence_snippets = []
    detached_eids = []
    for eid in claim.evidence_ids:
        ev = evidence_map.get(eid)
        if ev:
            if valid_source_ids is not None and ev.source_id not in valid_source_ids:
                detached_eids.append(eid)
                continue
            evidence_snippets.append(f"[{ev.id}] (Source: {ev.domain}) {ev.title}\nExcerpt: {ev.content}")

    if detached_eids:
        for deid in detached_eids:
            if deid not in claim.invalid_evidence_ids:
                claim.invalid_evidence_ids.append(deid)

    if not evidence_snippets:
        claim.verification_status = VerificationStatus.NOT_SUPPORTED
        if detached_eids:
            claim.verification_reason = (
                f"Linked evidence item(s) {detached_eids} reference nonexistent sources; detached evidence rejected."
            )
        else:
            claim.verification_reason = "Linked evidence items could not be found in active state."
        return claim

    prompt_text = (
        f"CLAIM TO VERIFY:\n\"{claim.claim_text}\"\n\n"
        f"LINKED EVIDENCE EXCERPTS:\n" + "\n\n".join(evidence_snippets) + "\n\n"
        "Evaluate strictly if the evidence supports this specific claim."
    )

    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        raw_json = _extract_json_block(response.content)
        data = json.loads(raw_json)

        status_str = str(data.get("status", "uncertain")).lower().strip()
        status_map = {
            "supported": VerificationStatus.SUPPORTED,
            "partially_supported": VerificationStatus.PARTIALLY_SUPPORTED,
            "not_supported": VerificationStatus.NOT_SUPPORTED,
            "uncertain": VerificationStatus.UNCERTAIN,
        }
        claim.verification_status = status_map.get(status_str, VerificationStatus.UNCERTAIN)
        claim.verification_reason = str(data.get("reason", "Verification evaluated against evidence.")).strip()
        confidence_val = data.get("confidence")
        if isinstance(confidence_val, (int, float)):
            claim.confidence = float(confidence_val)

    except Exception as err:
        # Edge Case D: Verification LLM/API failure -> Does not crash research process
        claim.verification_status = VerificationStatus.UNCERTAIN
        claim.verification_reason = f"Verification LLM call failed: {type(err).__name__} ({str(err)})"

    return claim


def verify_claims_batch(
    llm: ChatGroq,
    claims: List[Claim],
    evidence_items: List[EvidenceItem],
    sources: Optional[List[Source]] = None,
) -> List[Claim]:
    """Verifies a list of claims against available evidence items and validates source provenance."""
    evidence_map = {e.id: e for e in evidence_items}
    valid_source_ids = {s.id for s in sources} if sources is not None else None
    verified_results = []
    for claim in claims:
        verified = verify_single_claim(llm, claim, evidence_map, valid_source_ids=valid_source_ids)
        verified_results.append(verified)
    return verified_results

