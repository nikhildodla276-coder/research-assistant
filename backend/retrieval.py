import json
import re
from typing import List, Optional, Tuple, Dict, Any, Union
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    from models import (
        Source,
        EvidenceItem,
        SearchQuery,
        RetrievalTelemetry,
        SourceTier,
        ResearchPlan,
        ResearchComplexity,
        ResearchContext,
        ResearchDepth,
    )
    from planner import _extract_json_block
except ImportError:
    from backend.models import (
        Source,
        EvidenceItem,
        SearchQuery,
        RetrievalTelemetry,
        SourceTier,
        ResearchPlan,
        ResearchComplexity,
        ResearchContext,
        ResearchDepth,
    )
    from backend.planner import _extract_json_block

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "ref", "fbclid", "gclid", "msclkid", "twclid", "igshid", "mc_cid", "mc_eid",
    "_ga", "_gl", "ysclid", "yclid",
}

TIER_1_DOMAINS = {
    "opengeospatial.org", "w3.org", "ietf.org", "iso.org", "ieee.org",
    "nist.gov", "iana.org", "arxiv.org", "biorxiv.org", "medrxiv.org",
    "ncbi.nlm.nih.gov", "docs.python.org", "developer.nvidia.com",
    "docs.kernel.org", "bhuvan.nrsc.gov.in", "dilrmp.gov.in",
}

TIER_3_DOMAINS = {
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "thehindu.com",
    "indianexpress.com", "bloomberg.com", "techcrunch.com", "wired.com",
    "arstechnica.com", "ieee-spectrum.org", "nature.com", "science.org",
}

TIER_4_DOMAINS = {
    "medium.com", "substack.com", "reddit.com", "quora.com", "dev.to",
    "hashnode.com", "blogspot.com", "wordpress.com", "linkedin.com",
    "twitter.com", "x.com", "geeksforgeeks.org",
}


def canonicalize_url(url: str) -> str:
    """Deterministically canonicalizes a URL for deduplication.
    - Strips leading/trailing whitespace
    - Normalizes scheme and netloc to lowercase
    - Strips default ports (:80 for http, :443 for https)
    - Strips trailing slash on path (except root / or empty)
    - Strips tracking query parameters (utm_*, ref, fbclid, etc.)
    - Sorts remaining query parameters deterministically
    - Strips fragment identifiers (#...)
    - Rejects non-http/https schemes or malformed URLs
    """
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except Exception:
        return ""

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    netloc = parts.netloc.lower()
    if not netloc:
        return ""

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize path: strip trailing slash, but preserve empty
    path = parts.path
    if path:
        stripped_path = path.rstrip("/")
        path = stripped_path
    else:
        path = ""

    # Filter and sort query parameters
    query = ""
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        filtered = [
            (k, v) for k, v in pairs
            if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        if filtered:
            filtered.sort(key=lambda item: (item[0], item[1]))
            query = urlencode(filtered)

    # Fragment is always omitted
    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))


def classify_source_tier(domain: str, url: str = "") -> SourceTier:
    """Classifies source authority tier based on publisher/domain characteristics.
    Authority is strictly decoupled from topical relevance.
    - TIER_1: Government (.gov, .nic.in, .mil, .europa.eu), official standards bodies (w3.org, ietf.org, opengeospatial.org), official preprint/scientific archives (arxiv.org, ncbi.nlm.nih.gov), official platform docs.
    - TIER_2: Academic & research institutions (.edu, .ac.in, .edu.in, .ac.uk, recognized corporate research labs).
    - TIER_3: Reputable journalism, established trade and technical publications.
    - TIER_4: Commercial marketing, generic blogs, forums, user-generated content aggregators.
    - UNKNOWN: All domains with insufficient publisher authority evidence. Never falsely elevated.
    """
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]

    u = (url or "").lower().strip()

    if not d:
        return SourceTier.UNKNOWN

    # Exact TIER_1 domain or subdomain
    if d in TIER_1_DOMAINS or any(d.endswith("." + td) for td in TIER_1_DOMAINS):
        return SourceTier.TIER_1

    # Government suffixes
    if (
        d.endswith(".gov.in")
        or d.endswith(".nic.in")
        or d.endswith(".gov")
        or d.endswith(".mil")
        or d.endswith(".europa.eu")
        or d.endswith(".gov.uk")
        or d.endswith(".gov.au")
    ):
        return SourceTier.TIER_1

    # Academic institutions (TIER_2)
    if d.endswith(".edu") or d.endswith(".ac.in") or d.endswith(".edu.in") or d.endswith(".ac.uk"):
        return SourceTier.TIER_2

    if d in ("research.google", "openai.com") and "/research" in u:
        return SourceTier.TIER_2

    # Established trade / journalism (TIER_3)
    if d in TIER_3_DOMAINS or any(d.endswith("." + td) for td in TIER_3_DOMAINS):
        return SourceTier.TIER_3

    # Blogs, forums, UGC (TIER_4)
    if d in TIER_4_DOMAINS or any(d.endswith("." + td) for td in TIER_4_DOMAINS):
        return SourceTier.TIER_4

    # Default to UNKNOWN - conservative, no false elevation
    return SourceTier.UNKNOWN


QUERY_GEN_SYSTEM_PROMPT = """You are a senior search retrieval engineer specializing in targeted search query formulation.

Your task is to decompose a research objective and its planned research questions into 2 to 3 concise, keyword-dense search queries for web retrieval.

STRICT QUERY DESIGN RULES:
1. Generate 2 to 3 search queries for complex research.
2. Each query must be concise and keyword-dense (3 to 8 words).
3. Do NOT make queries full conversational sentences. Use retrieval keywords.
4. Each query must directly address one or more of the provided Research Questions.
5. Do NOT duplicate queries. Each query must target a distinct aspect of the problem.
6. Return ONLY a valid JSON array of query objects matching this schema:
[
  {
    "query_id": 1,
    "query_text": "keyword dense query text",
    "target_question_ids": [1]
  }
]
"""


def generate_search_queries(
    llm: Optional[ChatGroq],
    objective: str,
    context: Optional[ResearchContext] = None,
    plan: Optional[ResearchPlan] = None,
) -> List[SearchQuery]:
    """Generates targeted, keyword-dense search queries mapped to planned research questions.
    - Simple research: produces exactly 1 focused query.
    - Complex research: produces 2-3 distinct queries mapped to target question IDs.
    - Deterministically deduplicates queries, strips quotes/punctuation, and filters empty queries.
    - Provides safe, controlled fallback if LLM output is malformed or unavailable.
    """
    is_simple = False
    if plan and plan.complexity == ResearchComplexity.SIMPLE:
        is_simple = True
    elif not plan and (not context or context.required_depth == ResearchDepth.SHALLOW):
        is_simple = True

    if is_simple:
        clean_text = re.sub(r'["\']', '', objective).strip()
        return [
            SearchQuery(
                query_id=1,
                query_text=clean_text or objective,
                target_question_ids=[1],
            )
        ]

    # For complex research: prompt LLM to generate 2-3 queries
    prompt_lines = [f"USER RESEARCH OBJECTIVE:\n{objective}"]
    if context:
        ctx_str = context.format_prompt_block()
        if ctx_str:
            prompt_lines.append(f"RESEARCH CONTEXT:\n{ctx_str}")

    questions = plan.research_questions if (plan and plan.research_questions) else []
    if questions:
        q_lines = [f"{q.id}. {q.question} (Rationale: {q.rationale})" for q in questions]
        prompt_lines.append("PLANNED RESEARCH QUESTIONS:\n" + "\n".join(q_lines))
    else:
        prompt_lines.append("PLANNED RESEARCH QUESTIONS:\n1. Core technical investigation and requirements.")

    prompt_lines.append("TASK:\nGenerate 2 to 3 keyword-dense search queries as a JSON array.")
    human_content = "\n\n".join(prompt_lines)

    raw_response = None
    if llm:
        try:
            response = llm.invoke([
                SystemMessage(content=QUERY_GEN_SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ])
            raw_response = response.content
        except Exception:
            raw_response = None

    parsed_queries: List[SearchQuery] = []
    if raw_response:
        cleaned_json = _extract_json_block(raw_response)
        try:
            data = json.loads(cleaned_json)
            if isinstance(data, dict) and "queries" in data:
                data = data["queries"]
            if isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    if isinstance(item, dict):
                        q_text = str(item.get("query_text", "")).strip()
                        q_text = re.sub(r'["\']', '', q_text).strip()
                        t_ids = item.get("target_question_ids", [])
                        if not isinstance(t_ids, list):
                            t_ids = [t_ids] if isinstance(t_ids, int) else []
                        if q_text:
                            parsed_queries.append(
                                SearchQuery(
                                    query_id=idx,
                                    query_text=q_text,
                                    target_question_ids=t_ids or [idx],
                                )
                            )
                    elif isinstance(item, str) and item.strip():
                        q_text = re.sub(r'["\']', '', item).strip()
                        if q_text:
                            parsed_queries.append(
                                SearchQuery(
                                    query_id=idx,
                                    query_text=q_text,
                                    target_question_ids=[idx],
                                )
                            )
        except Exception:
            parsed_queries = []

    # Safe fallback if LLM output was malformed or empty (Failure Mode A)
    if not parsed_queries:
        if questions:
            for idx, q in enumerate(questions[:3], start=1):
                clean_q = re.sub(r'["\']', '', q.question).strip()
                keywords = re.sub(
                    r'\b(what|which|how|why|when|where|is|are|the|a|an|for|in|of|and|or)\b',
                    '',
                    clean_q,
                    flags=re.IGNORECASE,
                )
                keywords = " ".join(keywords.split())
                fallback_text = keywords if len(keywords) >= 10 else clean_q
                parsed_queries.append(
                    SearchQuery(
                        query_id=idx,
                        query_text=fallback_text,
                        target_question_ids=[q.id],
                    )
                )
        else:
            clean_obj = re.sub(r'["\']', '', objective).strip()
            parsed_queries.append(
                SearchQuery(
                    query_id=1,
                    query_text=clean_obj or objective,
                    target_question_ids=[1],
                )
            )

    # Post-processing:
    # 1. Deduplicate queries deterministically (Failure Mode B)
    # 2. Reject empty queries (Failure Mode C)
    seen_texts = set()
    unique_queries: List[SearchQuery] = []
    for q in parsed_queries:
        norm_text = " ".join(q.query_text.lower().split())
        if not norm_text:
            continue
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)
        unique_queries.append(
            SearchQuery(
                query_id=len(unique_queries) + 1,
                query_text=q.query_text,
                target_question_ids=q.target_question_ids or [len(unique_queries) + 1],
            )
        )

    if not unique_queries:
        unique_queries.append(
            SearchQuery(
                query_id=1,
                query_text=re.sub(r'["\']', '', objective).strip() or objective,
                target_question_ids=[1],
            )
        )

    return unique_queries[:3]


class CandidatePool:
    """Lightweight candidate aggregator handling URL canonicalization, deduplication,
    source authority classification, and retrieval telemetry accounting."""

    def __init__(self, provider: str = "tavily"):
        self.provider = provider
        self.candidates_discovered = 0
        self.duplicates_removed = 0
        self.candidates_rejected = 0
        self.rejection_reasons: Dict[str, int] = {}
        self.unique_candidates: Dict[str, dict] = {}
        self.queries_executed: List[SearchQuery] = []
        self.provider_errors: Dict[str, str] = {}
        self.duration_ms: Optional[float] = None

    def add_query_results(
        self,
        query: SearchQuery,
        results: Any,
        stage: str = "initial",
        iteration: int = 0,
        retrieval_timestamp: Optional[str] = None,
    ) -> None:
        """Ingests raw results from a single query execution."""
        if query not in self.queries_executed:
            self.queries_executed.append(query)
        items = []
        if isinstance(results, dict):
            items = results.get("results", [])
        elif isinstance(results, list):
            items = results

        for item in items:
            self.candidates_discovered += 1
            if not isinstance(item, dict):
                self.candidates_rejected += 1
                self.rejection_reasons["unusable_candidate"] = self.rejection_reasons.get("unusable_candidate", 0) + 1
                continue

            raw_url = item.get("url")
            if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
                self.candidates_rejected += 1
                self.rejection_reasons["invalid_url"] = self.rejection_reasons.get("invalid_url", 0) + 1
                continue

            canonical = canonicalize_url(raw_url)
            if not canonical:
                self.candidates_rejected += 1
                self.rejection_reasons["invalid_url"] = self.rejection_reasons.get("invalid_url", 0) + 1
                continue

            if canonical in self.unique_candidates:
                # Exact canonical duplicate detected across or within queries
                self.duplicates_removed += 1
                self.unique_candidates[canonical]["query_ids"].add(query.query_id)
            else:
                title = (item.get("title") or canonical).strip()
                content = (item.get("content") or "").strip()
                score = item.get("score")
                pub_date = item.get("published_date")
                parsed_domain = urlparse(canonical).netloc
                if parsed_domain.startswith("www."):
                    parsed_domain = parsed_domain[4:]

                tier = classify_source_tier(parsed_domain, canonical)
                ts = item.get("retrieved_at") or retrieval_timestamp

                self.unique_candidates[canonical] = {
                    "url": canonical,
                    "original_url": raw_url,
                    "title": title,
                    "content": content,
                    "score": score,
                    "domain": parsed_domain,
                    "published_date": pub_date,
                    "retrieved_at": ts,
                    "source_tier": tier,
                    "query_ids": {query.query_id},
                    "stage": stage,
                    "iteration": iteration,
                }

    def record_provider_error(self, query: SearchQuery, error_msg: str) -> None:
        """Records a query execution failure without dropping other query results (Failure Mode G).
        Note: A query failure does not discover any candidates; therefore it does NOT increment
        candidate-level counters (candidates_rejected or candidates_discovered). It is tracked
        exclusively in provider_errors and queries_executed."""
        if query not in self.queries_executed:
            self.queries_executed.append(query)
        self.provider_errors[str(query.query_id)] = error_msg

    def build_sources_and_evidence(
        self, start_idx: int = 1
    ) -> Tuple[List[Source], List[EvidenceItem], RetrievalTelemetry]:
        """Finalizes the pool into structured Source and EvidenceItem lists, and constructs telemetry.
        Preserves arithmetic consistency:
        candidates_discovered - duplicates_removed - candidates_rejected == unique_candidates_retained
        """
        unique_retained = len(self.unique_candidates)
        # Arithmetic consistency guarantee
        assert (
            self.candidates_discovered - self.duplicates_removed - self.candidates_rejected
            == unique_retained
        ), f"Telemetry arithmetic inconsistency: {self.candidates_discovered} - {self.duplicates_removed} - {self.candidates_rejected} != {unique_retained}"

        tier_order = {
            SourceTier.TIER_1: 1,
            SourceTier.TIER_2: 2,
            SourceTier.TIER_3: 3,
            SourceTier.TIER_4: 4,
            SourceTier.UNKNOWN: 5,
        }

        # Stable sort: prioritize higher authority tiers, retaining lower tiers
        sorted_candidates = sorted(
            self.unique_candidates.values(),
            key=lambda c: tier_order.get(c["source_tier"], 5)
        )

        sources: List[Source] = []
        evidence_items: List[EvidenceItem] = []

        for offset, c in enumerate(sorted_candidates):
            idx = start_idx + offset
            source = Source(
                id=idx,
                title=c["title"],
                url=c["url"],
                domain=c["domain"],
                source_type="web",
                source_tier=c["source_tier"],
                query_ids=sorted(list(c["query_ids"])),
                retrieved_at=c["retrieved_at"],
                published_date=c["published_date"],
                score=c["score"],
            )
            evidence = EvidenceItem(
                id=idx,
                source_id=idx,
                title=c["title"],
                url=c["url"],
                domain=c["domain"],
                content=c["content"],
                score=c["score"],
                stage=c["stage"],
                iteration=c["iteration"],
                retrieved_at=c["retrieved_at"],
            )
            sources.append(source)
            evidence_items.append(evidence)

        telemetry = RetrievalTelemetry(
            provider=self.provider,
            queries_executed=self.queries_executed,
            candidates_discovered=self.candidates_discovered,
            duplicates_removed=self.duplicates_removed,
            unique_candidates_retained=unique_retained,
            candidates_rejected=self.candidates_rejected,
            rejection_reasons=dict(self.rejection_reasons),
            provider_errors=dict(self.provider_errors),
            duration_ms=self.duration_ms,
        )

        return sources, evidence_items, telemetry

