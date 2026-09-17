from datetime import datetime, timezone
import os
import re
from typing import Optional, Dict, List, Union

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

try:
    from models import (
        Source,
        EvidenceItem,
        ResearchState,
        ResearchPlan,
        ResearchStatus,
        ResearchComplexity,
        ResearchContext,
        ResearchDepth,
        FreshnessRequirement,
        Claim,
        VerificationStatus,
        KnowledgeGap,
        GapPriority,
        ResearchSufficiency,
        SufficiencyStatus,
        parse_tavily_results,
        SourceTier,
        SearchQuery,
        RetrievalTelemetry,
    )
    from planner import generate_research_plan
    from verifier import extract_claims, verify_claims_batch, verify_single_claim
    from gap_detector import (
        detect_knowledge_gaps,
        formulate_follow_up_query,
        evaluate_gap_resolution,
        evaluate_sufficiency,
    )
    from retrieval import (
        generate_search_queries,
        CandidatePool,
        canonicalize_url,
        classify_source_tier,
    )
    from budget import (
        prepare_synthesis_prompt_within_budget,
        DEFAULT_SYNTHESIS_BUDGET_TOKENS,
    )
except ImportError:
    from backend.models import (
        Source,
        EvidenceItem,
        ResearchState,
        ResearchPlan,
        ResearchStatus,
        ResearchComplexity,
        ResearchContext,
        ResearchDepth,
        FreshnessRequirement,
        Claim,
        VerificationStatus,
        KnowledgeGap,
        GapPriority,
        ResearchSufficiency,
        SufficiencyStatus,
        parse_tavily_results,
        SourceTier,
        SearchQuery,
        RetrievalTelemetry,
    )
    from backend.planner import generate_research_plan
    from backend.verifier import extract_claims, verify_claims_batch, verify_single_claim
    from backend.gap_detector import (
        detect_knowledge_gaps,
        formulate_follow_up_query,
        evaluate_gap_resolution,
        evaluate_sufficiency,
    )
    from backend.retrieval import (
        generate_search_queries,
        CandidatePool,
        canonicalize_url,
        classify_source_tier,
    )
    from backend.budget import (
        prepare_synthesis_prompt_within_budget,
        DEFAULT_SYNTHESIS_BUDGET_TOKENS,
    )

load_dotenv()

# In-memory storage
store: Dict[str, InMemoryChatMessageHistory] = {}
research_states: Dict[str, ResearchState] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def get_research_state(session_id: str) -> Optional[ResearchState]:
    return research_states.get(session_id)


def get_session_claims(session_id: str) -> List[Claim]:
    state = research_states.get(session_id)
    return state.claims if state else []


def get_session_gaps(session_id: str) -> List[KnowledgeGap]:
    state = research_states.get(session_id)
    return state.gaps if state else []


def get_session_sufficiency(session_id: str) -> Optional[ResearchSufficiency]:
    state = research_states.get(session_id)
    return state.sufficiency if state else []


def approve_plan(session_id: str) -> Optional[ResearchPlan]:
    """Transitions a session's research plan status from AWAITING_APPROVAL to APPROVED."""
    state = research_states.get(session_id)
    if state and state.plan:
        state.plan.status = ResearchStatus.APPROVED
        state.status = ResearchStatus.APPROVED
        return state.plan
    return None


llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
chain = RunnableWithMessageHistory(llm, get_session_history)
tool = TavilySearch(max_results=5)

RESEARCH_SYSTEM_PROMPT = (
    "You are a rigorous, evidence-grounded research assistant producing an objective technical report.\n\n"
    "STRICT CITATION & FACTUALITY RULES:\n"
    "1. Base all findings, claims, and analysis strictly and exclusively on the provided Evidence items.\n"
    "2. Address the primary objective and the planned research questions thoroughly.\n"
    "3. Do NOT invent, assume, or extrapolate unsupported facts, statistics, or dates.\n"
    "4. Use inline numbered citations such as [1], [2], [3] immediately following any statement or finding derived from that evidence.\n"
    "5. If an evidence item only partially addresses a point or details are uncertain, explicitly state the limitation.\n"
    "6. Format the report using clear Markdown headings (e.g., Executive Summary, Research Questions & Key Findings, Technical Details/Analysis).\n"
    "7. Do NOT write a 'Sources', 'References', or 'Claim Verification Summary' section at the end — the system appends verified sections automatically."
)


def _sanitize_report_body(report_text: str) -> str:
    """Removes any model-generated Sources, References, or Gaps sections to guarantee application ownership."""
    cleaned = re.sub(
        r"\n##+\s*(?:Sources|References|Citations|Claim Verification Summary|Research Gaps & Limitations)\b.*$",
        "",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def normalize_research_context(
    objective: str,
    context: Optional[Union[ResearchContext, dict, str]] = None,
    purpose: Optional[str] = None,
) -> ResearchContext:
    """Normalizes optional context representations into a unified, validated ResearchContext."""
    if isinstance(context, ResearchContext):
        if not context.objective and objective:
            context.objective = objective
        if not context.purpose and purpose:
            context.purpose = purpose
        return context

    if isinstance(context, dict):
        ctx_data = dict(context)
        if "objective" not in ctx_data or not ctx_data["objective"]:
            ctx_data["objective"] = objective
        if "purpose" not in ctx_data and purpose:
            ctx_data["purpose"] = purpose
        return ResearchContext(**ctx_data)

    if isinstance(context, str):
        return ResearchContext(
            objective=objective,
            purpose=purpose,
            user_context=context,
        )

    # context is None
    return ResearchContext(
        objective=objective,
        purpose=purpose,
    )


def build_synthesis_prompt(
    objective: str,
    plan_context: str,
    evidence_context: str,
    context: Optional[ResearchContext] = None,
    stage_header: str = "Retrieved Evidence",
) -> str:
    """Constructs the LLM synthesis prompt integrating objective, research context requirements,
    decomposed research plan, and structured evidence pool."""
    sections = [f"Research Objective: {objective}"]

    if context:
        ctx_block = context.format_prompt_block()
        if ctx_block:
            sections.append(f"Research Context & Requirements:\n{ctx_block}")

    sections.append(f"Research Plan Scope:\n{plan_context}")
    sections.append(f"{stage_header}:\n{evidence_context}")
    sections.append(
        "Synthesize a comprehensive, structured research report for this objective, "
        "systematically addressing the planned research questions. "
        "Cite specific evidence items using bracketed numbers (e.g. [1], [2]) based strictly on the provided evidence."
    )
    return "\n\n".join(sections)


async def plan_research(
    objective: str,
    session_id: str = "default",
    context: Optional[Union[ResearchContext, dict, str]] = None,
    purpose: Optional[str] = None,
) -> ResearchPlan:
    """Performs preliminary research and generates a grounded ResearchPlan."""
    # 1. Preliminary search to scope the topic and ground the planner
    retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    raw_results = tool.invoke(objective)
    sources, evidence_items = parse_tavily_results(
        raw_results,
        start_idx=1,
        stage="initial",
        iteration=0,
        retrieval_timestamp=retrieval_timestamp,
    )

    # Normalize research context
    norm_context = normalize_research_context(objective, context=context, purpose=purpose)

    # 2. Decompose into structured research plan
    plan = generate_research_plan(
        llm=llm,
        objective=objective,
        preliminary_evidence=evidence_items,
        context=norm_context,
        purpose=norm_context.purpose or purpose,
    )

    state = ResearchState(
        objective=objective,
        session_id=session_id,
        context=norm_context,
        purpose=norm_context.purpose or purpose,
        status=plan.status,
        plan=plan,
        sources=sources,
        evidence=evidence_items,
        follow_up_count=0,
    )
    research_states[session_id] = state
    return plan


async def run_research(
    topic: str,
    session_id: str = "default",
    context: Optional[Union[ResearchContext, dict, str]] = None,
    purpose: Optional[str] = None,
) -> str:
    """Runs preliminary scoping, plan generation, initial synthesis, claim verification,
    bounded knowledge gap detection, optional 1-iteration follow-up, and cited report assembly."""
    # 1. Preliminary scoping and research planning
    plan = await plan_research(
        objective=topic,
        session_id=session_id,
        context=context,
        purpose=purpose,
    )

    state = research_states[session_id]
    state.status = ResearchStatus.EXECUTING

    # 2. Multi-Query Generation
    queries = generate_search_queries(
        llm=llm,
        objective=topic,
        context=state.context,
        plan=plan,
    )

    # 3. Multi-Query Retrieval & Candidate Pool Execution
    pool = CandidatePool(provider="tavily")
    retrieval_start = datetime.now(timezone.utc)

    for q in queries:
        try:
            raw_res = tool.invoke(q.query_text)
            ts = datetime.now(timezone.utc).isoformat()
            pool.add_query_results(
                query=q,
                results=raw_res,
                stage="initial",
                iteration=0,
                retrieval_timestamp=ts,
            )
        except Exception as err:
            pool.record_provider_error(query=q, error_msg=str(err))

    retrieval_end = datetime.now(timezone.utc)
    pool.duration_ms = (retrieval_end - retrieval_start).total_seconds() * 1000.0

    sources, evidence_items, telemetry = pool.build_sources_and_evidence(start_idx=1)
    state.sources = sources
    state.evidence = evidence_items
    state.telemetry = telemetry

    if not state.evidence:
        fallback_report = (
            f"## Research Objective: {topic}\n\n"
            "No verified web evidence could be retrieved for this objective during preliminary scoping. "
            "Unable to synthesize an evidence-grounded report without reliable sources."
        )
        state.report = fallback_report
        state.status = ResearchStatus.COMPLETED
        state.sufficiency = ResearchSufficiency(
            status=SufficiencyStatus.INSUFFICIENT,
            reason="No initial web evidence could be retrieved.",
            remaining_gap_ids=[],
            follow_up_performed=False,
            iteration_count=0,
        )
        history = get_session_history(session_id)
        history.add_user_message(f"Research objective: {topic}")
        history.add_ai_message(fallback_report)
        return fallback_report

    # 4. Build synthesis prompt bounded by token budget
    plan_context = plan.format_plan_markdown() if hasattr(plan, "format_plan_markdown") else str(plan)
    human_prompt, reduction_info = prepare_synthesis_prompt_within_budget(
        objective=topic,
        plan_context=plan_context,
        evidence_items=state.evidence,
        sources=state.sources,
        context=state.context,
        stage_header="Retrieved Evidence",
        budget_tokens=DEFAULT_SYNTHESIS_BUDGET_TOKENS,
    )
    if state.telemetry and reduction_info.get("reduction_occurred"):
        state.telemetry.input_reduction = reduction_info

    if not human_prompt:
        # Test C: Budget cannot safely be satisfied even after reduction
        fallback_report = (
            f"## Research Objective: {topic}\n\n"
            "The synthesis request could not be safely executed within the configured LLM token budget. "
            "All retrieved evidence items and sources have been preserved in the research state."
        )
        state.report = fallback_report
        state.status = ResearchStatus.COMPLETED
        state.sufficiency = ResearchSufficiency(
            status=SufficiencyStatus.INSUFFICIENT,
            reason="Input token budget exceeded by fixed prompt requirements.",
            remaining_gap_ids=[],
            follow_up_performed=False,
            iteration_count=0,
        )
        history = get_session_history(session_id)
        history.add_user_message(f"Research objective: {topic}")
        history.add_ai_message(fallback_report)
        return fallback_report

    # 5. LLM synthesis strictly conditioned on structured evidence with error boundary
    try:
        response = llm.invoke([
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        report_body = _sanitize_report_body(response.content)
    except Exception as synth_err:
        report_body = (
            f"## Research Objective: {topic}\n\n"
            f"LLM synthesis encountered a provider error: {type(synth_err).__name__} ({str(synth_err)}). "
            "Retrieved evidence and source provenance are preserved."
        )
        if state.telemetry:
            state.telemetry.provider_errors["synthesis_initial"] = str(synth_err)


    # 6. Extract atomic claims and perform initial verification
    valid_evidence_ids = [e.id for e in state.evidence]
    raw_claims = extract_claims(
        llm=llm,
        content=report_body,
        valid_evidence_ids=valid_evidence_ids,
    )
    verified_claims = verify_claims_batch(
        llm=llm,
        claims=raw_claims,
        evidence_items=state.evidence,
        sources=state.sources,
    )
    state.claims = verified_claims

    # 7. Detect Knowledge Gaps
    gaps = detect_knowledge_gaps(llm=llm, state=state)
    state.gaps = gaps

    actionable_gaps = [
        g for g in gaps
        if not g.resolved and g.priority in (GapPriority.HIGH, GapPriority.MEDIUM)
    ]

    follow_up_performed = False

    # 8. Bounded Adaptive Research Loop (MAXIMUM 1 ITERATION)
    if actionable_gaps and state.follow_up_count < 1:
        state.follow_up_count += 1
        follow_up_performed = True

        # Formulate targeted search query
        target_query = formulate_follow_up_query(llm, state, actionable_gaps)
        fu_query = SearchQuery(
            query_id=100 + state.follow_up_count,
            query_text=target_query,
            target_question_ids=[g.related_question_id for g in actionable_gaps if g.related_question_id] or [1],
        )

        fu_pool = CandidatePool(provider="tavily")
        fu_start = datetime.now(timezone.utc)
        try:
            raw_followup = tool.invoke(target_query)
            followup_ts = datetime.now(timezone.utc).isoformat()
            fu_pool.add_query_results(
                query=fu_query,
                results=raw_followup,
                stage="follow_up",
                iteration=1,
                retrieval_timestamp=followup_ts,
            )
        except Exception as search_err:
            fu_pool.record_provider_error(query=fu_query, error_msg=str(search_err))
            for g in actionable_gaps:
                g.resolution_notes = f"Follow-up search failed: {type(search_err).__name__} ({str(search_err)})"

        fu_end = datetime.now(timezone.utc)
        fu_pool.duration_ms = (fu_end - fu_start).total_seconds() * 1000.0
        followup_sources, followup_evidence, fu_telemetry = fu_pool.build_sources_and_evidence(
            start_idx=len(state.sources) + 1
        )

        if state.telemetry:
            dump_fu = getattr(fu_telemetry, "model_dump", None) or getattr(fu_telemetry, "dict")
            state.telemetry.follow_up_telemetry = dump_fu()

        if followup_evidence:
            state.sources.extend(followup_sources)
            state.evidence.extend(followup_evidence)

            # Re-verify non-supported or uncertain claims against expanded evidence pool
            evidence_map = {e.id: e for e in state.evidence}
            claims_to_reverify = [
                c for c in state.claims
                if c.verification_status != VerificationStatus.SUPPORTED
            ]
            if claims_to_reverify:
                # Add newly available follow-up evidence IDs to claims if relevant
                followup_ids = [e.id for e in followup_evidence]
                for c in claims_to_reverify:
                    # Let the verifier re-check against the expanded evidence
                    c.evidence_ids = list(set(c.evidence_ids + followup_ids))
                    verify_single_claim(
                        llm,
                        c,
                        evidence_map,
                        valid_source_ids={s.id for s in state.sources},
                    )

            # Evaluate gap resolution
            evaluate_gap_resolution(llm, actionable_gaps, followup_evidence)

            # Re-synthesize final report integrating new evidence within token budget
            updated_prompt, resynth_reduction_info = prepare_synthesis_prompt_within_budget(
                objective=topic,
                plan_context=plan_context,
                evidence_items=state.evidence,
                sources=state.sources,
                context=state.context,
                stage_header="Expanded Evidence Pool (Initial + Follow-up Search)",
                budget_tokens=DEFAULT_SYNTHESIS_BUDGET_TOKENS,
            )
            if state.telemetry and resynth_reduction_info.get("reduction_occurred"):
                state.telemetry.input_reduction = resynth_reduction_info

            if updated_prompt:
                try:
                    updated_response = llm.invoke([
                        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
                        HumanMessage(content=updated_prompt),
                    ])
                    report_body = _sanitize_report_body(updated_response.content)
                except Exception as resynth_err:
                    if state.telemetry:
                        state.telemetry.provider_errors["synthesis_followup"] = str(resynth_err)

        else:
            # Failure D: Follow-up returned no useful evidence -> retain gaps
            for g in actionable_gaps:
                if not g.resolution_notes:
                    g.resolution_notes = "Follow-up search returned no new evidence; gap remains unresolved."

    # 7. Evaluate Research Sufficiency & Stopping Decision
    sufficiency = evaluate_sufficiency(
        state=state,
        follow_up_performed=follow_up_performed,
        iteration_count=state.follow_up_count,
    )
    state.sufficiency = sufficiency

    # 8. Assemble Transparent Final Report
    report_parts = [report_body]

    claims_markdown = state.format_claims_markdown()
    if claims_markdown:
        report_parts.append(claims_markdown)

    gaps_markdown = state.format_gaps_markdown()
    if gaps_markdown:
        report_parts.append(gaps_markdown)

    sources_section = state.format_sources_markdown()
    if sources_section:
        report_parts.append(sources_section)

    final_report = "\n\n".join(report_parts)
    state.report = final_report
    state.status = ResearchStatus.COMPLETED

    # 9. Update session history cleanly
    history = get_session_history(session_id)
    history.add_user_message(f"Research objective: {topic}")
    history.add_ai_message(final_report)

    return final_report


async def run_chat(message: str, session_id: str = "default") -> str:
    response = chain.invoke(
        HumanMessage(content=message),
        config={"configurable": {"session_id": session_id}},
    )
    return response.content
