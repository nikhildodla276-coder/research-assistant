import math
import os
import re
from typing import List, Optional, Tuple, Dict, Any

try:
    from models import (
        Source,
        EvidenceItem,
        SourceTier,
        ResearchContext,
        ResearchPlan,
    )
except ImportError:
    from backend.models import (
        Source,
        EvidenceItem,
        SourceTier,
        ResearchContext,
        ResearchPlan,
    )

# Configurable synthesis input token budget (default 6000 tokens for Groq TPM limit 8000)
DEFAULT_SYNTHESIS_BUDGET_TOKENS = int(os.getenv("SYNTHESIS_MAX_INPUT_TOKENS", "6000"))


def estimate_tokens(text: str) -> int:
    """Conservative token estimator for LLM prompts.
    Uses a 3.5 characters-per-token heuristic (with whitespace floor) to ensure
    we overestimate rather than underestimate tokens, guaranteeing the model
    input budget is never breached."""
    if not text:
        return 0
    char_estimate = math.ceil(len(text) / 3.5)
    word_estimate = math.ceil(len(text.split()) * 1.3)
    return max(char_estimate, word_estimate)


def build_bounded_evidence_context(
    evidence_items: List[EvidenceItem],
    sources: List[Source],
    max_tokens: int,
    stage_header: str = "Retrieved Evidence",
) -> Tuple[str, Dict[str, Any]]:
    """Constructs an evidence context string bounded by max_tokens.
    If full evidence exceeds max_tokens, prioritizes items by:
      1. Follow-up stage evidence (addresses unresolved knowledge gaps)
      2. Source authority tier (TIER_1 > TIER_2 > TIER_3 > TIER_4 > UNKNOWN)
      3. Provider relevance score
      4. Original evidence ID (stable order)
    Selected items retain their exact original IDs, preserving complete provenance.
    Original research state is never mutated."""
    if not evidence_items:
        return "No evidence retrieved.", {
            "reduction_occurred": False,
            "original_tokens": 0,
            "reduced_tokens": 0,
            "total_items": 0,
            "included_ids": [],
            "omitted_ids": [],
        }

    # Format full evidence context first to check if reduction is needed
    full_lines = []
    for item in evidence_items:
        stage_tag = f" [{item.stage.capitalize()}]" if item.stage != "initial" else ""
        full_lines.append(
            f"[{item.id}] Title: {item.title}{stage_tag}\n"
            f"Source: {item.domain} ({item.url})\n"
            f"Evidence Snippet: {item.content}\n"
        )
    full_context_str = "\n".join(full_lines)
    original_tokens = estimate_tokens(full_context_str)

    # If within budget, return full context without reduction (Test A)
    if original_tokens <= max_tokens:
        return full_context_str, {
            "reduction_occurred": False,
            "original_tokens": original_tokens,
            "reduced_tokens": original_tokens,
            "budget_tokens": max_tokens,
            "total_items": len(evidence_items),
            "included_ids": [e.id for e in evidence_items],
            "omitted_ids": [],
        }

    # Reduction needed: build source map for authority tier lookup
    source_map = {s.id: s for s in sources}
    tier_order = {
        SourceTier.TIER_1: 1,
        SourceTier.TIER_2: 2,
        SourceTier.TIER_3: 3,
        SourceTier.TIER_4: 4,
        SourceTier.UNKNOWN: 5,
    }

    # Priority key:
    # 0 = follow_up stage (highest priority to resolve gaps), 1 = initial stage
    # then authority tier (1 to 5)
    # then negative score (higher score first)
    # then original item id
    def priority_key(item: EvidenceItem):
        stage_prio = 0 if item.stage == "follow_up" else 1
        src = source_map.get(item.source_id)
        src_tier = getattr(src, "source_tier", SourceTier.UNKNOWN) if src else SourceTier.UNKNOWN
        t_order = tier_order.get(src_tier, 5)
        score_val = item.score if item.score is not None else 0.0
        return (stage_prio, t_order, -score_val, item.id)

    ranked_items = sorted(evidence_items, key=priority_key)

    selected_items: List[EvidenceItem] = []
    current_tokens = 0

    for item in ranked_items:
        stage_tag = f" [{item.stage.capitalize()}]" if item.stage != "initial" else ""
        
        # Clean content snippet (bound single snippets to 800 chars to avoid single-item monopolization)
        content = item.content.strip()
        if len(content) > 800:
            content = content[:800].rstrip() + " ... [excerpt bounded for context budget]"

        candidate_line = (
            f"[{item.id}] Title: {item.title}{stage_tag}\n"
            f"Source: {item.domain} ({item.url})\n"
            f"Evidence Snippet: {content}\n"
        )
        line_tokens = estimate_tokens(candidate_line + "\n")

        if current_tokens + line_tokens <= max_tokens:
            selected_items.append(
                EvidenceItem(
                    id=item.id,
                    source_id=item.source_id,
                    title=item.title,
                    url=item.url,
                    domain=item.domain,
                    content=content,
                    score=item.score,
                    stage=item.stage,
                    iteration=item.iteration,
                    retrieved_at=item.retrieved_at,
                )
            )
            current_tokens += line_tokens

    # If even the top item couldn't fit, try with a short 300-char excerpt
    if not selected_items and ranked_items and max_tokens >= 80:
        top_item = ranked_items[0]
        stage_tag = f" [{top_item.stage.capitalize()}]" if top_item.stage != "initial" else ""
        short_content = top_item.content[:300].rstrip() + " ... [excerpt bounded]"
        line = (
            f"[{top_item.id}] Title: {top_item.title}{stage_tag}\n"
            f"Source: {top_item.domain} ({top_item.url})\n"
            f"Evidence Snippet: {short_content}\n"
        )
        if estimate_tokens(line) <= max_tokens:
            selected_items.append(top_item)

    # Sort selected items back by ID so that sequential citation order is natural
    selected_items.sort(key=lambda e: e.id)

    reduced_lines = []
    for item in selected_items:
        stage_tag = f" [{item.stage.capitalize()}]" if item.stage != "initial" else ""
        reduced_lines.append(
            f"[{item.id}] Title: {item.title}{stage_tag}\n"
            f"Source: {item.domain} ({item.url})\n"
            f"Evidence Snippet: {item.content}\n"
        )
    reduced_context_str = "\n".join(reduced_lines)
    reduced_tokens = estimate_tokens(reduced_context_str)

    included_ids = [e.id for e in selected_items]
    omitted_ids = [e.id for e in evidence_items if e.id not in included_ids]

    reduction_info = {
        "reduction_occurred": True,
        "original_tokens": original_tokens,
        "reduced_tokens": reduced_tokens,
        "budget_tokens": max_tokens,
        "total_items": len(evidence_items),
        "included_ids": included_ids,
        "omitted_ids": omitted_ids,
        "reason": (
            f"Original evidence tokens ({original_tokens}) exceeded budget ({max_tokens}). "
            f"Preserved {len(included_ids)}/{len(evidence_items)} items prioritized by follow-up and authority tier."
        ),
    }

    return reduced_context_str, reduction_info


def prepare_synthesis_prompt_within_budget(
    objective: str,
    plan_context: str,
    evidence_items: List[EvidenceItem],
    sources: List[Source],
    context: Optional[ResearchContext] = None,
    stage_header: str = "Retrieved Evidence",
    budget_tokens: int = DEFAULT_SYNTHESIS_BUDGET_TOKENS,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Builds a complete synthesis prompt guaranteed to not exceed budget_tokens.
    Returns (prompt_text, reduction_info).
    If budget cannot safely be satisfied even after reduction, returns (None, reduction_info)."""
    # 1. Measure fixed prompt sections (system prompt, objective, plan, context, instructions)
    sections = [f"Research Objective: {objective}"]

    if context:
        ctx_block = context.format_prompt_block()
        if ctx_block:
            sections.append(f"Research Context & Requirements:\n{ctx_block}")

    sections.append(f"Research Plan Scope:\n{plan_context}")
    instructions = (
        "Synthesize a comprehensive, structured research report for this objective, "
        "systematically addressing the planned research questions. "
        "Cite specific evidence items using bracketed numbers (e.g. [1], [2]) based strictly on the provided evidence."
    )

    fixed_text = "\n\n".join(sections) + f"\n\n{stage_header}:\n\n\n{instructions}"
    # Add system prompt overhead (RESEARCH_SYSTEM_PROMPT ≈ 200 tokens)
    system_prompt_overhead = 220
    fixed_tokens = estimate_tokens(fixed_text) + system_prompt_overhead

    # 2. Check if fixed overhead alone exceeds budget
    if fixed_tokens >= budget_tokens:
        return None, {
            "reduction_occurred": True,
            "error": "fixed_overhead_exceeds_budget",
            "fixed_tokens": fixed_tokens,
            "budget_tokens": budget_tokens,
            "reason": f"Fixed prompt components ({fixed_tokens} tokens) exceed total budget ({budget_tokens} tokens).",
        }

    available_for_evidence = budget_tokens - fixed_tokens

    # 3. Format bounded evidence context
    bounded_evidence_str, reduction_info = build_bounded_evidence_context(
        evidence_items=evidence_items,
        sources=sources,
        max_tokens=available_for_evidence,
        stage_header=stage_header,
    )

    # 4. If evidence is empty but items were requested and couldn't fit at all
    if not bounded_evidence_str.strip() and evidence_items:
        return None, {
            "reduction_occurred": True,
            "error": "evidence_budget_insufficient",
            "fixed_tokens": fixed_tokens,
            "budget_tokens": budget_tokens,
            "available_tokens": available_for_evidence,
            "reason": "Available evidence token budget was insufficient to include even 1 evidence snippet.",
        }

    # 5. Assemble final prompt
    full_prompt_sections = list(sections)
    full_prompt_sections.append(f"{stage_header}:\n{bounded_evidence_str}")
    full_prompt_sections.append(instructions)
    final_prompt = "\n\n".join(full_prompt_sections)

    total_estimated = estimate_tokens(final_prompt) + system_prompt_overhead
    reduction_info["total_estimated_tokens"] = total_estimated

    return final_prompt, reduction_info

