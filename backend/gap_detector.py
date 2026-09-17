import json
import re
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    from models import (
        EvidenceItem,
        GapPriority,
        KnowledgeGap,
        ResearchState,
        ResearchSufficiency,
        SufficiencyStatus,
        VerificationStatus,
    )
except ImportError:
    from backend.models import (
        EvidenceItem,
        GapPriority,
        KnowledgeGap,
        ResearchState,
        ResearchSufficiency,
        SufficiencyStatus,
        VerificationStatus,
    )

GAP_DETECTION_SYSTEM_PROMPT = """You are a rigorous research auditor assessing whether current research evidence adequately satisfies the research plan.

Your role is to identify critical, actionable Knowledge Gaps.

GUIDELINES:
1. Examine:
   - The Research Plan questions
   - The gathered Evidence excerpts
   - The verified Claims and their statuses (especially 'not_supported' or 'uncertain' claims)
2. Identify Knowledge Gaps:
   - A research question lacking specific, verifiable evidence.
   - An essential claim that is NOT_SUPPORTED or UNCERTAIN.
   - Missing technical metrics, architectures, or comparative data.
3. Assign Priority:
   - "high": Missing evidence completely blocks answering an essential research question.
   - "medium": Research question is partially addressed, but vital operational or technical information is missing.
   - "low": Minor uncertainty or secondary detail that does not materially impair the research objective.
4. Do NOT create gaps for every minor ambiguity. Focus only on gaps that materially affect the research objective.

Respond ONLY with a valid JSON array in this schema:
[
  {
    "gap_id": 1,
    "description": "Specific statement of missing technical information",
    "related_question_id": 1,
    "reason": "Why this gap exists based on current claims/evidence",
    "priority": "high" | "medium" | "low"
  }
]
"""

QUERY_FORMULATION_SYSTEM_PROMPT = """You are a search query engineer.

Your task is to formulate ONE highly focused, targeted search query to find missing evidence for the identified knowledge gaps.

RULES:
1. Formulate a search query that directly targets the missing information.
2. Do NOT simply repeat the broad original topic.
3. Combine domain keywords, technical terminology, and specific questions from the gap.
4. Return ONLY the search query string, nothing else.
"""

RESOLUTION_EVALUATION_SYSTEM_PROMPT = """You are an objective research validator.

Evaluate whether the newly retrieved Follow-up Evidence resolves the specified Knowledge Gaps.

For each gap, determine if the new evidence provides the missing factual information.

Respond ONLY with a JSON array:
[
  {
    "gap_id": 1,
    "resolved": true | false,
    "resolution_notes": "Concise explanation of how new evidence resolved or failed to resolve the gap."
  }
]
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


def detect_knowledge_gaps(llm: ChatGroq, state: ResearchState) -> List[KnowledgeGap]:
    """Analyzes current research state to detect and prioritize knowledge gaps."""
    # Build context for the gap detector
    lines = [f"RESEARCH OBJECTIVE:\n{state.objective}\n"]

    if state.plan and state.plan.research_questions:
        lines.append("PLANNED RESEARCH QUESTIONS:")
        for q in state.plan.research_questions:
            lines.append(f"[{q.id}] {q.question} (Rationale: {q.rationale})")
        lines.append("")

    if state.claims:
        lines.append("CURRENT CLAIMS & VERIFICATION STATUS:")
        for c in state.claims:
            ev_ids = f" [Evidence: {c.evidence_ids}]" if c.evidence_ids else " [No Evidence]"
            lines.append(f"- Claim {c.claim_id}: \"{c.claim_text}\" — Status: {c.verification_status.value.upper()}{ev_ids}")
            if c.verification_reason:
                lines.append(f"  Reason: {c.verification_reason}")
        lines.append("")

    if state.evidence:
        lines.append("AVAILABLE EVIDENCE EXCERPTS:")
        for e in state.evidence:
            lines.append(f"[{e.id}] ({e.domain}) {e.title}: {e.content[:160]}...")
    else:
        lines.append("AVAILABLE EVIDENCE EXCERPTS:\nNo evidence retrieved.")

    lines.append("\nTASK:\nIdentify critical knowledge gaps. Respond strictly with a JSON array.")
    prompt_text = "\n".join(lines)

    messages = [
        SystemMessage(content=GAP_DETECTION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        raw_json = _extract_json_block(response.content)
        data = json.loads(raw_json)
    except Exception as err:
        # Failure A: Malformed output -> Controlled fallback, does not crash, does NOT claim SUFFICIENT
        return [
            KnowledgeGap(
                gap_id=1,
                description="Unable to parse fine-grained gaps from detector output; conservative gap tracked.",
                related_question_id=None,
                reason=f"Gap detector output malformed: {type(err).__name__} ({str(err)})",
                priority=GapPriority.MEDIUM,
                resolved=False,
            )
        ]

    if not isinstance(data, list):
        return []

    # Valid question IDs from plan
    valid_qids = {q.id for q in state.plan.research_questions} if (state.plan and state.plan.research_questions) else set()
    priority_map = {
        "high": GapPriority.HIGH,
        "medium": GapPriority.MEDIUM,
        "low": GapPriority.LOW,
    }

    gaps: List[KnowledgeGap] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue

        raw_qid = item.get("related_question_id")
        valid_qid = None
        invalid_qid = None

        if isinstance(raw_qid, int):
            if valid_qids and raw_qid in valid_qids:
                valid_qid = raw_qid
            elif valid_qids:
                # Failure B: Gap references nonexistent research question
                invalid_qid = raw_qid

        raw_priority = str(item.get("priority", "medium")).lower().strip()
        priority = priority_map.get(raw_priority, GapPriority.MEDIUM)

        description = str(item.get("description", f"Unresolved knowledge gap {idx}")).strip()
        reason = str(item.get("reason", "Identified gap in current research evidence.")).strip()
        if invalid_qid is not None:
            reason += f" (Note: referenced nonexistent question ID {invalid_qid} was rejected)"

        gap = KnowledgeGap(
            gap_id=item.get("gap_id", idx),
            description=description,
            related_question_id=valid_qid,
            invalid_question_id=invalid_qid,
            reason=reason,
            priority=priority,
            resolved=False,
        )
        gaps.append(gap)

    return gaps


def formulate_follow_up_query(
    llm: ChatGroq,
    state: ResearchState,
    actionable_gaps: List[KnowledgeGap],
) -> str:
    """Generates a targeted search query addressing the most important gaps."""
    if not actionable_gaps:
        return state.objective

    gap_descriptions = []
    for g in actionable_gaps[:2]:
        gap_descriptions.append(f"- [{g.priority.value.upper()}] {g.description} (Reason: {g.reason})")

    prompt_text = (
        f"RESEARCH OBJECTIVE: {state.objective}\n\n"
        f"KEY KNOWLEDGE GAPS TO RESOLVE:\n" + "\n".join(gap_descriptions) + "\n\n"
        "Generate ONE targeted, concise search query to find the missing evidence."
    )

    messages = [
        SystemMessage(content=QUERY_FORMULATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        query = response.content.strip().strip('"').strip("'")
        if query:
            return query
    except Exception:
        pass

    # Safe deterministic fallback query
    return f"{state.objective} {actionable_gaps[0].description[:60]}"


def evaluate_gap_resolution(
    llm: ChatGroq,
    gaps: List[KnowledgeGap],
    new_evidence: List[EvidenceItem],
) -> List[KnowledgeGap]:
    """Checks whether newly gathered follow-up evidence resolves the knowledge gaps."""
    if not new_evidence:
        # Failure D: Empty follow-up evidence -> No false resolution, retain gaps
        for g in gaps:
            g.resolved = False
            g.resolution_notes = "Follow-up search returned no new evidence; gap remains unresolved."
        return gaps

    evidence_text = "\n\n".join(
        f"[{e.id}] ({e.domain}) {e.title}:\n{e.content}" for e in new_evidence
    )
    gaps_text = "\n".join(
        f"Gap {g.gap_id} ({g.priority.value.upper()}): {g.description}" for g in gaps
    )

    prompt_text = (
        f"KNOWLEDGE GAPS:\n{gaps_text}\n\n"
        f"NEW FOLLOW-UP EVIDENCE:\n{evidence_text}\n\n"
        "Evaluate whether the new evidence resolves each gap. Respond with a JSON array."
    )

    messages = [
        SystemMessage(content=RESOLUTION_EVALUATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        raw_json = _extract_json_block(response.content)
        data = json.loads(raw_json)

        res_map = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "gap_id" in item:
                    res_map[item["gap_id"]] = item

        for g in gaps:
            if g.gap_id in res_map:
                res_info = res_map[g.gap_id]
                g.resolved = bool(res_info.get("resolved", False))
                g.resolution_notes = str(res_info.get("resolution_notes", "Evaluated against follow-up evidence."))
            else:
                g.resolution_notes = "Not specifically addressed in follow-up validation."

    except Exception as err:
        # Failure E: Re-verification / validation failure -> Preserve previous state, do NOT mark resolved
        for g in gaps:
            g.resolution_notes = f"Follow-up resolution evaluation failed: {type(err).__name__}; retained as unresolved."

    return gaps


def evaluate_sufficiency(
    state: ResearchState,
    follow_up_performed: bool,
    iteration_count: int,
) -> ResearchSufficiency:
    """Computes explicit sufficiency status and grounds the stopping decision."""
    unresolved_actionable = [
        g for g in state.gaps
        if not g.resolved and g.priority in (GapPriority.HIGH, GapPriority.MEDIUM)
    ]
    unresolved_ids = [g.gap_id for g in unresolved_actionable]

    if not unresolved_actionable:
        # Condition A / C: Research is sufficient
        if follow_up_performed:
            reason = "Follow-up research resolved initial high/medium knowledge gaps; research is sufficient."
        else:
            reason = "Initial research provided adequate evidence; no critical knowledge gaps detected."
        return ResearchSufficiency(
            status=SufficiencyStatus.SUFFICIENT,
            reason=reason,
            remaining_gap_ids=[],
            follow_up_performed=follow_up_performed,
            iteration_count=iteration_count,
        )

    # Condition B / Failure F: Maximum iteration reached (hard-bound = 1)
    if iteration_count >= 1:
        reason = (
            f"Maximum follow-up iteration (1) reached. {len(unresolved_actionable)} important gap(s) "
            "remain unresolved and are documented as research limitations."
        )
    else:
        reason = f"Research is insufficient: {len(unresolved_actionable)} high/medium knowledge gap(s) remain."

    return ResearchSufficiency(
        status=SufficiencyStatus.INSUFFICIENT,
        reason=reason,
        remaining_gap_ids=unresolved_ids,
        follow_up_performed=follow_up_performed,
        iteration_count=iteration_count,
    )

