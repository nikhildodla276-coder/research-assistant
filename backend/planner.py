import json
import re
from typing import List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    from models import (
        EvidenceItem,
        ResearchComplexity,
        ResearchPlan,
        ResearchQuestion,
        ResearchStatus,
        ResearchContext,
    )
except ImportError:
    from backend.models import (
        EvidenceItem,
        ResearchComplexity,
        ResearchPlan,
        ResearchQuestion,
        ResearchStatus,
        ResearchContext,
    )

PLANNER_SYSTEM_PROMPT = """You are a senior technical research architect and research planner.

Your role is to analyze a research objective, review preliminary source evidence, and produce a grounded research plan.

GUIDELINES:
1. Distinguish between:
   - User Objective / Context (what the user asked for)
   - Preliminary Source Evidence (facts grounded in web sources)
   - Your Technical Interpretation (analytical decomposition)
2. Assess Complexity:
   - "simple": Factual, narrow, single-faceted topic. Needs exactly 1 focused question. Requires approval = false.
   - "complex": Multi-faceted technical challenge, architecture, comparison, or problem statement (e.g., SIH challenges). Needs 2 to 4 distinct sub-questions covering feasibility, methodologies, data/tooling, and evaluation. Requires approval = true.
3. Preliminary Findings:
   - Summarize 2-4 concrete preliminary facts found in the preliminary evidence snippets. Do not hallucinate facts not in the evidence.
4. Research Questions:
   - Decompose into actionable, specific research questions. For each, provide a brief rationale explaining why answering it is necessary.
5. Planned Areas:
   - List key technical/domain areas to investigate.

Respond ONLY with a valid JSON object in the following schema:
{
  "interpreted_objective": "Clear, precise technical articulation of the objective",
  "complexity": "simple" or "complex",
  "requires_approval": true or false,
  "preliminary_findings": ["Finding 1 grounded in evidence", "Finding 2 grounded in evidence"],
  "planned_areas": ["Area 1", "Area 2"],
  "research_questions": [
    {
      "id": 1,
      "question": "Specific question to research?",
      "rationale": "Why this question is essential"
    }
  ]
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


def build_planner_prompt(
    objective: str,
    preliminary_evidence: List[EvidenceItem],
    context: Optional[Union[ResearchContext, str]] = None,
    purpose: Optional[str] = None,
) -> str:
    sections = [f"USER RESEARCH OBJECTIVE:\n{objective}"]

    effective_purpose = purpose
    if isinstance(context, ResearchContext):
        if context.purpose:
            effective_purpose = context.purpose
        if effective_purpose:
            sections.append(f"STATED PURPOSE:\n{effective_purpose}")

        ctx_block = context.format_prompt_block()
        if ctx_block:
            sections.append(f"RESEARCH CONTEXT & REQUIREMENTS:\n{ctx_block}")
    else:
        if effective_purpose:
            sections.append(f"STATED PURPOSE:\n{effective_purpose}")
        if context:
            sections.append(f"ADDITIONAL USER CONTEXT:\n{context}")

    if preliminary_evidence:
        evidence_lines = []
        for e in preliminary_evidence:
            evidence_lines.append(f"[{e.id}] ({e.domain}) {e.title}\nExcerpt: {e.content}")
        sections.append("PRELIMINARY SOURCE EVIDENCE:\n" + "\n\n".join(evidence_lines))
    else:
        sections.append("PRELIMINARY SOURCE EVIDENCE:\nNo initial sources retrieved.")

    sections.append("TASK:\nAnalyze the above inputs and generate the structured research plan JSON.")
    return "\n\n".join(sections)


def generate_research_plan(
    llm: ChatGroq,
    objective: str,
    preliminary_evidence: List[EvidenceItem],
    context: Optional[Union[ResearchContext, str]] = None,
    purpose: Optional[str] = None,
) -> ResearchPlan:
    """Invokes the LLM to generate a grounded ResearchPlan from preliminary evidence."""
    prompt_text = build_planner_prompt(
        objective=objective,
        preliminary_evidence=preliminary_evidence,
        context=context,
        purpose=purpose,
    )

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    response = llm.invoke(messages)
    raw_content = response.content
    cleaned_json = _extract_json_block(raw_content)

    try:
        data = json.loads(cleaned_json)
    except Exception as err:
        # Fallback to a safe simple plan if JSON decode fails
        return ResearchPlan(
            original_objective=objective,
            interpreted_objective=objective,
            context=context,
            purpose=purpose,
            complexity=ResearchComplexity.SIMPLE,
            preliminary_findings=[e.content[:120] for e in preliminary_evidence[:2]],
            research_questions=[
                ResearchQuestion(
                    id=1,
                    question=f"What are the key technical findings for: {objective}?",
                    rationale="Direct investigation of user objective",
                )
            ],
            planned_areas=["General Research"],
            requires_approval=False,
            status=ResearchStatus.APPROVED,
        )

    # Parse complexity
    raw_complexity = str(data.get("complexity", "simple")).lower().strip()
    complexity = (
        ResearchComplexity.COMPLEX
        if "complex" in raw_complexity
        else ResearchComplexity.SIMPLE
    )

    requires_approval = bool(data.get("requires_approval", complexity == ResearchComplexity.COMPLEX))

    # Parse questions
    raw_questions = data.get("research_questions", [])
    questions: List[ResearchQuestion] = []
    if isinstance(raw_questions, list):
        for idx, q in enumerate(raw_questions, start=1):
            if isinstance(q, dict):
                questions.append(
                    ResearchQuestion(
                        id=q.get("id", idx),
                        question=q.get("question", f"Research angle {idx}"),
                        rationale=q.get("rationale", "Necessary for objective fulfillment"),
                    )
                )
            elif isinstance(q, str):
                questions.append(
                    ResearchQuestion(
                        id=idx,
                        question=q,
                        rationale="Necessary for objective fulfillment",
                    )
                )

    if not questions:
        questions.append(
            ResearchQuestion(
                id=1,
                question=f"Investigate core requirements for: {objective}",
                rationale="Primary objective question",
            )
        )

    # State transition: complex tasks with approval required enter AWAITING_APPROVAL
    initial_status = (
        ResearchStatus.AWAITING_APPROVAL
        if (complexity == ResearchComplexity.COMPLEX and requires_approval)
        else ResearchStatus.APPROVED
    )

    return ResearchPlan(
        original_objective=objective,
        interpreted_objective=data.get("interpreted_objective", objective),
        context=context,
        purpose=purpose,
        complexity=complexity,
        preliminary_findings=data.get("preliminary_findings", []),
        research_questions=questions,
        planned_areas=data.get("planned_areas", []),
        requires_approval=requires_approval,
        status=initial_status,
    )

