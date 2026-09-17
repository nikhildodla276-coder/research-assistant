from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Any, Union, Set
from urllib.parse import urlparse
from pydantic import BaseModel, Field, field_validator


class ResearchComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class ResearchStatus(str, Enum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    UNCERTAIN = "uncertain"


class GapPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SufficiencyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ResearchDepth(str, Enum):
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"


class FreshnessRequirement(str, Enum):
    ANY = "any"
    CURRENT = "current"
    HISTORICAL = "historical"


class SourceTier(str, Enum):
    """Conservative classification of source publisher authority (independent of topical relevance)."""
    TIER_1 = "tier_1"  # Government, official institutions, standards bodies, primary documentation, preprint archives
    TIER_2 = "tier_2"  # Universities, recognized academic and research institutions
    TIER_3 = "tier_3"  # Reputable journalism, established trade publications, industry sources
    TIER_4 = "tier_4"  # Commercial marketing, generic blogs, forums, weak aggregators
    UNKNOWN = "unknown"  # Insufficient information to classify confidently


class SearchQuery(BaseModel):
    """Represents a targeted, keyword-dense search query derived from research questions."""
    query_id: int = Field(description="1-based numeric identifier for the query")
    query_text: str = Field(description="Concise, keyword-dense search query string")
    target_question_ids: List[int] = Field(default_factory=list, description="IDs of planned research questions this query addresses")


class RetrievalTelemetry(BaseModel):
    """Structured telemetry recording retrieval execution details."""
    provider: str = Field(default="tavily", description="Primary search provider")
    queries_executed: List[SearchQuery] = Field(default_factory=list, description="Structured queries executed")
    candidates_discovered: int = Field(default=0, description="Total candidate items discovered across all queries")
    duplicates_removed: int = Field(default=0, description="Duplicate candidate URLs pruned during deduplication")
    unique_candidates_retained: int = Field(default=0, description="Unique candidate sources retained in candidate pool")
    candidates_rejected: int = Field(default=0, description="Candidates rejected due to invalid URLs or unusable state")
    rejection_reasons: dict[str, int] = Field(default_factory=dict, description="Counts of rejected candidates by reason code")
    provider_errors: dict[str, str] = Field(default_factory=dict, description="Errors encountered during query execution keyed by query_id")
    duration_ms: Optional[float] = Field(default=None, description="Retrieval duration in milliseconds")
    follow_up_telemetry: Optional[dict[str, Any]] = Field(default=None, description="Telemetry from follow-up search if executed")
    input_reduction: Optional[dict[str, Any]] = Field(default=None, description="Telemetry recording synthesis input token budget reduction")



class ResearchContext(BaseModel):
    """Structured representation of why and how research is conducted."""
    objective: Optional[str] = Field(default=None, description="What the user wants to research")
    purpose: Optional[str] = Field(default=None, description="Why the user needs this research")
    user_context: Optional[str] = Field(default=None, description="Situation or background influencing the research")
    target_audience: Optional[str] = Field(default=None, description="Who will use the resulting information")
    constraints: Optional[str] = Field(default=None, description="Restrictions affecting the research")
    required_depth: ResearchDepth = Field(default=ResearchDepth.STANDARD, description="Depth of investigation")
    freshness_requirement: FreshnessRequirement = Field(default=FreshnessRequirement.ANY, description="Freshness requirement")
    source_requirements: List[str] = Field(default_factory=list, description="Preferred source quality/types")
    expected_output: Optional[str] = Field(default=None, description="Expected form of result")

    @field_validator("required_depth", mode="before")
    @classmethod
    def _normalize_depth(cls, v):
        if isinstance(v, ResearchDepth):
            return v
        if isinstance(v, str):
            s = v.lower().strip()
            if "deep" in s:
                return ResearchDepth.DEEP
            if "shallow" in s:
                return ResearchDepth.SHALLOW
            if "standard" in s:
                return ResearchDepth.STANDARD
            raise ValueError(f"Invalid required_depth: '{v}'. Must be one of {[e.value for e in ResearchDepth]}")
        return v

    @field_validator("freshness_requirement", mode="before")
    @classmethod
    def _normalize_freshness(cls, v):
        if isinstance(v, FreshnessRequirement):
            return v
        if isinstance(v, str):
            s = v.lower().strip()
            if "curr" in s or "recent" in s:
                return FreshnessRequirement.CURRENT
            if "hist" in s:
                return FreshnessRequirement.HISTORICAL
            if "any" in s:
                return FreshnessRequirement.ANY
            raise ValueError(f"Invalid freshness_requirement: '{v}'. Must be one of {[e.value for e in FreshnessRequirement]}")
        return v

    def format_prompt_block(self) -> str:
        """Formats the context fields into an instructive block for LLM prompts."""
        parts = []
        if self.purpose:
            parts.append(f"- Purpose: {self.purpose}")
        if self.user_context:
            parts.append(f"- Background / Situation: {self.user_context}")
        if self.target_audience:
            parts.append(f"- Target Audience: {self.target_audience}")
        if self.constraints:
            parts.append(f"- Constraints: {self.constraints}")
        if self.required_depth:
            parts.append(f"- Depth: {self.required_depth.value}")
        if self.freshness_requirement:
            parts.append(f"- Freshness: {self.freshness_requirement.value}")
        if self.source_requirements:
            reqs = ", ".join(self.source_requirements)
            parts.append(f"- Source Requirements: {reqs}")
        if self.expected_output:
            parts.append(f"- Expected Output: {self.expected_output}")
        return "\n".join(parts)


class Source(BaseModel):
    """Represents an external web source retrieved during research."""
    id: int = Field(description="1-based numeric identifier corresponding to citations [1], [2], etc.")
    title: str = Field(description="Page or document title")
    url: str = Field(default="", description="Direct URL to source")
    domain: str = Field(default="", description="Netloc domain (e.g. arxiv.org)")
    source_type: str = Field(default="web", description="Type of source (web, academic, government, etc.)")
    retrieved_at: Optional[str] = Field(default=None, description="Timezone-aware ISO 8601 UTC timestamp when retrieved")
    published_date: Optional[str] = Field(default=None, description="Publication date if available")
    score: Optional[float] = Field(default=None, description="Tavily relevance score")
    source_tier: SourceTier = Field(default=SourceTier.UNKNOWN, description="Classified source authority tier")
    query_ids: List[int] = Field(default_factory=list, description="IDs of queries that discovered this source")


class EvidenceItem(BaseModel):
    """Represents an atomic piece of evidence extracted from a source."""
    id: int = Field(description="1-based identifier matching the citation number")
    source_id: int = Field(description="ID of parent source")
    title: str = Field(default="", description="Title of parent source")
    url: str = Field(default="", description="URL of parent source")
    domain: str = Field(default="", description="Domain of parent source")
    content: str = Field(description="Verbatim snippet or relevant text content")
    score: Optional[float] = Field(default=None, description="Relevance score")
    stage: str = Field(default="initial", description="'initial' or 'follow_up'")
    iteration: int = Field(default=0, description="0 for initial search, 1 for follow-up")
    retrieved_at: Optional[str] = Field(default=None, description="Timezone-aware ISO 8601 UTC timestamp matching parent source")


class ResearchQuestion(BaseModel):
    """Represents a structured sub-question within a research plan."""
    id: int = Field(description="1-based identifier for the sub-question")
    question: str = Field(description="Focused research question to investigate")
    rationale: str = Field(description="Why this question is required to answer the objective")
    status: str = Field(default="planned", description="Status: planned, executing, completed")


class ResearchPlan(BaseModel):
    """Encapsulates the structured research plan and scoping decomposition."""
    original_objective: str = Field(description="Original user-provided objective")
    interpreted_objective: str = Field(description="Normalized/clarified technical objective")
    context: Optional[Union[ResearchContext, str]] = Field(default=None, description="Optional user context or structured ResearchContext")
    purpose: Optional[str] = Field(default=None, description="Optional purpose/goals for research")
    complexity: ResearchComplexity = Field(default=ResearchComplexity.SIMPLE, description="Simple or complex")
    preliminary_findings: List[str] = Field(default_factory=list, description="Findings from preliminary scoping search")
    research_questions: List[ResearchQuestion] = Field(default_factory=list, description="Decomposed research questions")
    planned_areas: List[str] = Field(default_factory=list, description="Target technical areas to investigate")
    requires_approval: bool = Field(default=False, description="Whether approval is required before execution")
    status: ResearchStatus = Field(default=ResearchStatus.PLANNING, description="Current workflow state")

    def format_plan_markdown(self) -> str:
        """Formats the research plan into a clean Markdown summary."""
        lines = [
            f"### Research Plan: {self.interpreted_objective}",
            f"- **Complexity**: {self.complexity.value.capitalize()}",
            f"- **Approval Required**: {'Yes' if self.requires_approval else 'No'} (Status: `{self.status.value}`)",
        ]
        if self.purpose:
            lines.append(f"- **Purpose**: {self.purpose}")
        if self.context:
            if isinstance(self.context, ResearchContext):
                ctx_summary = self.context.user_context or self.context.format_prompt_block()
                lines.append(f"- **Context**: {ctx_summary}")
            else:
                lines.append(f"- **Context**: {self.context}")

        if self.preliminary_findings:
            lines.append("\n**Preliminary Findings:**")
            for finding in self.preliminary_findings:
                lines.append(f"- {finding}")

        if self.research_questions:
            lines.append("\n**Planned Research Questions:**")
            for q in self.research_questions:
                lines.append(f"{q.id}. **{q.question}** — *{q.rationale}*")

        return "\n".join(lines)


class Claim(BaseModel):
    """Represents a discrete factual claim extracted from research content."""
    claim_id: int = Field(description="1-based numeric identifier for the claim")
    claim_text: str = Field(description="The atomic factual assertion")
    evidence_ids: List[int] = Field(default_factory=list, description="Valid 1-based IDs referencing EvidenceItem.id")
    invalid_evidence_ids: List[int] = Field(default_factory=list, description="Nonexistent or rejected evidence IDs")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNCERTAIN,
        description="Verification outcome against linked evidence"
    )
    verification_reason: str = Field(default="", description="Reasoning for verification outcome")
    confidence: Optional[float] = Field(default=None, description="Optional confidence score (0.0 to 1.0)")


class KnowledgeGap(BaseModel):
    """Represents an unresolved or insufficiently supported knowledge gap."""
    gap_id: int = Field(description="1-based numeric identifier for the gap")
    description: str = Field(description="Concrete statement of missing or incomplete information")
    related_question_id: Optional[int] = Field(default=None, description="Related ResearchQuestion.id if valid")
    invalid_question_id: Optional[int] = Field(default=None, description="Flagged invalid question reference")
    reason: str = Field(description="Why this gap exists (unsupported claim, missing evidence, partial data)")
    priority: GapPriority = Field(default=GapPriority.MEDIUM, description="HIGH, MEDIUM, or LOW priority")
    resolved: bool = Field(default=False, description="Whether gap was resolved by follow-up research")
    resolution_notes: Optional[str] = Field(default=None, description="Details on resolution or remaining limitations")


class ResearchSufficiency(BaseModel):
    """Encapsulates the explicit research sufficiency outcome and stopping decision."""
    status: SufficiencyStatus = Field(default=SufficiencyStatus.INSUFFICIENT, description="SUFFICIENT or INSUFFICIENT")
    reason: str = Field(default="", description="Grounded explanation of sufficiency decision")
    remaining_gap_ids: List[int] = Field(default_factory=list, description="IDs of gaps remaining unresolved")
    follow_up_performed: bool = Field(default=False, description="Whether targeted follow-up research was executed")
    iteration_count: int = Field(default=0, description="Follow-up iteration count (hard maximum: 1)")


class ResearchState(BaseModel):
    """Encapsulates the state of a research session for an objective."""
    objective: str = Field(description="User's research goal or query")
    session_id: str = Field(default="default", description="Conversation session ID")
    context: Optional[ResearchContext] = Field(default=None, description="Structured research context")
    purpose: Optional[str] = Field(default=None, description="User-provided purpose")
    status: ResearchStatus = Field(default=ResearchStatus.PLANNING, description="Workflow execution status")
    plan: Optional[ResearchPlan] = Field(default=None, description="Structured research plan")
    sources: List[Source] = Field(default_factory=list, description="Retrieved sources")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Structured evidence items")
    claims: List[Claim] = Field(default_factory=list, description="Extracted factual claims with verification status")
    gaps: List[KnowledgeGap] = Field(default_factory=list, description="Identified knowledge gaps")
    sufficiency: Optional[ResearchSufficiency] = Field(default=None, description="Sufficiency evaluation and stopping decision")
    follow_up_count: int = Field(default=0, description="Follow-up search iteration count (max 1)")
    telemetry: Optional[RetrievalTelemetry] = Field(default=None, description="Retrieval execution telemetry")
    report: Optional[str] = Field(default=None, description="Synthesized cited research report")

    @field_validator("context", mode="before")
    @classmethod
    def _coerce_context(cls, v):
        if v is None:
            return None
        if isinstance(v, ResearchContext):
            return v
        if isinstance(v, dict):
            return ResearchContext(**v)
        if isinstance(v, str):
            return ResearchContext(user_context=v)
        return v

    def format_evidence_context(self) -> str:
        """Formats evidence items into indexed context for the LLM."""
        if not self.evidence:
            return "No evidence retrieved."

        lines = []
        for item in self.evidence:
            stage_tag = f" [{item.stage.capitalize()}]" if item.stage != "initial" else ""
            lines.append(
                f"[{item.id}] Title: {item.title}{stage_tag}\n"
                f"Source: {item.domain} ({item.url})\n"
                f"Evidence Snippet: {item.content}\n"
            )
        return "\n".join(lines)

    def format_sources_markdown(self) -> str:
        """Deterministically generates the verified markdown Sources section."""
        if not self.sources:
            return ""

        lines = ["## Sources\n"]
        for s in self.sources:
            display_title = s.title if s.title else (s.url if s.url else f"Source {s.id}")
            retrieved_info = f" *(Retrieved: {s.retrieved_at})*" if s.retrieved_at else ""
            if s.url:
                if s.domain:
                    lines.append(f"[{s.id}] [{display_title}]({s.url}) — *{s.domain}*{retrieved_info}")
                else:
                    lines.append(f"[{s.id}] [{display_title}]({s.url}){retrieved_info}")
            else:
                # Source has no usable URL: report cleanly without a fabricated link or invented URL
                if s.domain:
                    lines.append(f"[{s.id}] {display_title} — *{s.domain}*{retrieved_info}")
                else:
                    lines.append(f"[{s.id}] {display_title}{retrieved_info}")
        return "\n".join(lines)

    def format_claims_markdown(self) -> str:
        """Formats verification results into a clean, transparent summary."""
        if not self.claims:
            return ""

        lines = ["## Claim Verification Summary\n"]
        status_badges = {
            VerificationStatus.SUPPORTED: "Verified",
            VerificationStatus.PARTIALLY_SUPPORTED: "⚠️ Partially Supported",
            VerificationStatus.NOT_SUPPORTED: "❌ Not Supported",
            VerificationStatus.UNCERTAIN: "❓ Uncertain",
        }

        for c in self.claims:
            badge = status_badges.get(c.verification_status, c.verification_status.value)
            ev_str = f" [Evidence: {', '.join(f'[{i}]' for i in c.evidence_ids)}]" if c.evidence_ids else ""
            if c.invalid_evidence_ids:
                ev_str += f" *(Invalid references rejected: {c.invalid_evidence_ids})*"
            lines.append(f"- **Claim {c.claim_id}**: \"{c.claim_text}\" — **{badge}**{ev_str}")
            if c.verification_reason:
                lines.append(f"  *Reason*: {c.verification_reason}")

        return "\n".join(lines)

    def format_gaps_markdown(self) -> str:
        """Formats unresolved knowledge gaps into a transparent report section."""
        unresolved_gaps = [g for g in self.gaps if not g.resolved]
        if not unresolved_gaps:
            return ""

        lines = ["## Research Gaps & Limitations\n"]
        for g in unresolved_gaps:
            priority_tag = f"[{g.priority.value.upper()} PRIORITY]"
            qid_str = f" (Related Question {g.related_question_id})" if g.related_question_id else ""
            lines.append(f"- **{priority_tag}**: {g.description}{qid_str}")
            lines.append(f"  *Reason*: {g.reason}")
            if g.resolution_notes:
                lines.append(f"  *Follow-up Outcome*: {g.resolution_notes}")

        if self.sufficiency:
            lines.append(f"\n*Sufficiency Evaluation*: **{self.sufficiency.status.value.upper()}** — {self.sufficiency.reason}")

        return "\n".join(lines)

    def get_claim_provenance(self, claim_id: int) -> Optional[dict]:
        """Traces a claim back to its linked evidence items, their parent sources, and URLs."""
        claim = next((c for c in self.claims if c.claim_id == claim_id), None)
        if not claim:
            return None

        evidence_map = {e.id: e for e in self.evidence}
        sources_map = {s.id: s for s in self.sources}

        provenance_chain = []
        for eid in claim.evidence_ids:
            ev = evidence_map.get(eid)
            if not ev:
                provenance_chain.append({
                    "evidence_id": eid,
                    "error": "Evidence item not found in state",
                    "valid": False,
                })
                continue

            src = sources_map.get(ev.source_id)
            provenance_chain.append({
                "evidence_id": ev.id,
                "evidence_content": ev.content,
                "stage": ev.stage,
                "iteration": ev.iteration,
                "retrieved_at": ev.retrieved_at,
                "source_id": ev.source_id,
                "source_title": src.title if src else None,
                "source_url": src.url if src else None,
                "domain": src.domain if src else None,
                "source_type": src.source_type if src else None,
                "source_tier": src.source_tier.value if (src and hasattr(src, "source_tier") and src.source_tier) else None,
                "query_ids": src.query_ids if (src and hasattr(src, "query_ids")) else [],
                "source_retrieved_at": src.retrieved_at if src else None,
                "valid_source": src is not None,
            })

        return {
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "verification_status": claim.verification_status.value,
            "verification_reason": claim.verification_reason,
            "confidence": claim.confidence,
            "provenance": provenance_chain,
            "invalid_evidence_ids": claim.invalid_evidence_ids,
        }

    def validate_evidence(self) -> List[int]:
        """Returns list of evidence IDs that reference nonexistent sources (detached evidence)."""
        source_ids = {s.id for s in self.sources}
        return [e.id for e in self.evidence if e.source_id not in source_ids]


def parse_tavily_results(
    raw_results: Any,
    start_idx: int = 1,
    stage: str = "initial",
    iteration: int = 0,
    retrieval_timestamp: Optional[str] = None,
    query_ids: Optional[List[int]] = None,
    source_tier: SourceTier = SourceTier.UNKNOWN,
) -> tuple[List[Source], List[EvidenceItem]]:
    """Converts raw Tavily output into structured Source and EvidenceItem instances with sequential IDs."""
    items = []
    if isinstance(raw_results, dict):
        items = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        items = raw_results

    sources: List[Source] = []
    evidence_items: List[EvidenceItem] = []

    for offset, item in enumerate(items):
        idx = start_idx + offset
        if not isinstance(item, dict):
            continue

        raw_url = item.get("url")
        url = str(raw_url).strip() if raw_url else ""
        title = (item.get("title") or url or f"Source {idx}").strip()
        content = (item.get("content") or "").strip()
        score = item.get("score")
        pub_date = item.get("published_date")
        source_type = item.get("source_type", "web")
        # Use explicit retrieved_at from item if provided, otherwise the passed retrieval_timestamp, or None (never fabricate)
        retrieved_at = item.get("retrieved_at") or retrieval_timestamp

        domain = ""
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc or ""
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = ""

        source = Source(
            id=idx,
            title=title,
            url=url,
            domain=domain,
            source_type=source_type,
            source_tier=item.get("source_tier", source_tier),
            query_ids=item.get("query_ids", query_ids or []),
            retrieved_at=retrieved_at,
            published_date=pub_date,
            score=score,
        )
        evidence = EvidenceItem(
            id=idx,
            source_id=idx,
            title=title,
            url=url,
            domain=domain,
            content=content,
            score=score,
            stage=stage,
            iteration=iteration,
            retrieved_at=retrieved_at,
        )

        sources.append(source)
        evidence_items.append(evidence)

    return sources, evidence_items
