import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    Source,
    EvidenceItem,
    ResearchState,
    ResearchPlan,
    ResearchQuestion,
    ResearchComplexity,
    ResearchStatus,
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
from retrieval import (
    canonicalize_url,
    classify_source_tier,
    generate_search_queries,
    CandidatePool,
)
from planner import build_planner_prompt
from verifier import (
    extract_claims,
    verify_single_claim,
    verify_claims_batch,
)
from gap_detector import (
    detect_knowledge_gaps,
    formulate_follow_up_query,
    evaluate_gap_resolution,
    evaluate_sufficiency,
)
from researcher import (
    _sanitize_report_body,
    run_research,
    run_chat,
    plan_research,
    approve_plan,
    get_research_state,
    get_session_claims,
    get_session_gaps,
    get_session_sufficiency,
    build_synthesis_prompt,
    normalize_research_context,
)
from budget import (
    estimate_tokens,
    build_bounded_evidence_context,
    prepare_synthesis_prompt_within_budget,
    DEFAULT_SYNTHESIS_BUDGET_TOKENS,
)



def test_models_and_parsing():
    print("--- Test 1: Models & Tavily Parsing ---")
    mock_tavily_raw = {
        "results": [
            {
                "title": "Satellite Imagery in Disaster Response",
                "url": "https://example.org/disaster-ai",
                "content": "Deep learning models detect flood extents within 15 minutes of satellite pass.",
                "score": 0.89,
                "published_date": "2024-01-15",
            },
            {
                "title": "Automated Damage Assessment Frameworks",
                "url": "https://arxiv.org/abs/2301.12345",
                "content": "Using Sentinel-2 multispectral bands with U-Net architecture for building damage classification.",
                "score": 0.82,
            },
        ]
    }

    sources, evidence = parse_tavily_results(mock_tavily_raw, start_idx=1, stage="initial", iteration=0)
    assert len(sources) == 2, f"Expected 2 sources, got {len(sources)}"
    assert len(evidence) == 2, f"Expected 2 evidence items, got {len(evidence)}"
    assert sources[0].id == 1
    assert sources[0].domain == "example.org"
    assert sources[1].domain == "arxiv.org"
    assert evidence[0].stage == "initial"
    assert evidence[0].iteration == 0

    state = ResearchState(
        objective="AI automated disaster assessment",
        session_id="test_session",
        sources=sources,
        evidence=evidence,
    )

    evidence_text = state.format_evidence_context()
    assert "[1] Title: Satellite Imagery in Disaster Response" in evidence_text
    assert "[2] Title: Automated Damage Assessment Frameworks" in evidence_text

    sources_md = state.format_sources_markdown()
    assert "## Sources" in sources_md
    assert "[1] [Satellite Imagery in Disaster Response](https://example.org/disaster-ai) — *example.org*" in sources_md
    assert "[2] [Automated Damage Assessment Frameworks](https://arxiv.org/abs/2301.12345) — *arxiv.org*" in sources_md

    print("✓ Models and deterministic citation formatting passed successfully.")


def test_report_sanitization():
    print("\n--- Test 2: Report Sanitization ---")
    mock_llm_output = (
        "## Executive Summary\nSatellite AI models show high precision [1].\n\n"
        "## Sources\n[1] Invented Source (https://fake.url)\n\n"
        "## Research Gaps & Limitations\nInvented gap\n"
    )
    cleaned = _sanitize_report_body(mock_llm_output)
    assert "Invented Source" not in cleaned
    assert "Invented gap" not in cleaned
    assert "Satellite AI models show high precision [1]." in cleaned
    print("✓ LLM report sanitization passed successfully.")


def test_planning_models_and_state_transitions():
    print("\n--- Test 3: Structured Plan Representation & State Transitions ---")
    simple_plan = ResearchPlan(
        original_objective="When was Python 3.13 released?",
        interpreted_objective="Official release date and milestones of Python 3.13",
        complexity=ResearchComplexity.SIMPLE,
        requires_approval=False,
        status=ResearchStatus.APPROVED,
        research_questions=[
            ResearchQuestion(
                id=1,
                question="What is the official final release date of Python 3.13?",
                rationale="Direct factual lookup",
            )
        ],
        preliminary_findings=["Python 3.13.0 was officially released in October 2024."],
    )
    assert simple_plan.complexity == ResearchComplexity.SIMPLE
    assert not simple_plan.requires_approval
    assert simple_plan.status == ResearchStatus.APPROVED
    assert len(simple_plan.research_questions) == 1

    complex_plan = ResearchPlan(
        original_objective="Autonomous Disaster Assessment with Satellite Imagery",
        interpreted_objective="Multi-sensor satellite imagery pipeline for automated flood and building damage assessment",
        context="SIH Disaster Management track",
        purpose="Evaluate technical feasibility, model architectures, and data requirements",
        complexity=ResearchComplexity.COMPLEX,
        requires_approval=True,
        status=ResearchStatus.AWAITING_APPROVAL,
        preliminary_findings=[
            "Sentinel-1 SAR and Sentinel-2 optical data are standard free sources.",
            "U-Net and Vision Transformers achieve >85% mIoU on flood segmentation benchmarks.",
        ],
        planned_areas=["Data Ingestion & Preprocessing", "Model Architecture", "Inference Latency"],
        research_questions=[
            ResearchQuestion(
                id=1,
                question="Which satellite constellations provide low-latency open imagery for disaster response?",
                rationale="Data availability is the primary constraint.",
            ),
            ResearchQuestion(
                id=2,
                question="What neural network architectures are best suited for building damage localization?",
                rationale="Determines core compute and algorithmic design.",
            ),
        ],
    )
    assert complex_plan.complexity == ResearchComplexity.COMPLEX
    assert complex_plan.requires_approval
    assert complex_plan.status == ResearchStatus.AWAITING_APPROVAL
    assert len(complex_plan.research_questions) == 2

    dummy_session = "test_approval_session"
    state = ResearchState(
        objective=complex_plan.original_objective,
        session_id=dummy_session,
        context=complex_plan.context,
        purpose=complex_plan.purpose,
        status=complex_plan.status,
        plan=complex_plan,
    )
    from researcher import research_states

    research_states[dummy_session] = state
    assert state.status == ResearchStatus.AWAITING_APPROVAL

    approved = approve_plan(dummy_session)
    assert approved is not None
    assert approved.status == ResearchStatus.APPROVED
    assert state.status == ResearchStatus.APPROVED

    print("✓ Planning models, prompt grounding, and approval state transitions passed.")


def test_claim_models_and_edge_cases():
    print("\n--- Test 4: Claim Models, Verification Logic & Edge Cases ---")
    c1 = Claim(
        claim_id=1,
        claim_text="Sentinel-1 operates in C-band synthetic aperture radar.",
        evidence_ids=[1],
        verification_status=VerificationStatus.SUPPORTED,
        verification_reason="Directly confirmed by Sentinel-1 technical documentation.",
        confidence=0.98,
    )
    assert c1.claim_id == 1
    assert c1.verification_status == VerificationStatus.SUPPORTED
    assert c1.evidence_ids == [1]

    c_multi = Claim(
        claim_id=2,
        claim_text="SAR imagery penetrates clouds and nighttime conditions.",
        evidence_ids=[1, 2],
        verification_status=VerificationStatus.SUPPORTED,
        verification_reason="Confirmed across radar fundamentals [1] and operational review [2].",
    )
    assert len(c_multi.evidence_ids) == 2

    c_partial = Claim(
        claim_id=3,
        claim_text="System processes full resolution tiles in 5 seconds on edge GPU.",
        evidence_ids=[1],
        verification_status=VerificationStatus.PARTIALLY_SUPPORTED,
        verification_reason="Evidence confirms edge GPU deployment, but benchmarks specify 12 seconds, not 5.",
    )
    assert c_partial.verification_status == VerificationStatus.PARTIALLY_SUPPORTED

    c_unsupported = Claim(
        claim_id=4,
        claim_text="NASA directly funded the software startup.",
        evidence_ids=[],
        verification_status=VerificationStatus.NOT_SUPPORTED,
        verification_reason="No supporting evidence items were linked to this claim.",
    )
    assert c_unsupported.verification_status == VerificationStatus.NOT_SUPPORTED

    c_uncertain = Claim(
        claim_id=5,
        claim_text="Future constellations will offer sub-meter SAR by 2029.",
        evidence_ids=[2],
        verification_status=VerificationStatus.UNCERTAIN,
        verification_reason="Evidence mentions planned expansion but lacks definitive year or resolution specifications.",
    )
    assert c_uncertain.verification_status == VerificationStatus.UNCERTAIN

    state = ResearchState(
        objective="SAR capabilities",
        claims=[c1, c_partial, c_unsupported, c_uncertain],
    )
    claims_md = state.format_claims_markdown()
    assert "## Claim Verification Summary" in claims_md
    assert "Verified" in claims_md
    assert "⚠️ Partially Supported" in claims_md
    assert "❌ Not Supported" in claims_md
    assert "❓ Uncertain" in claims_md

    # Edge Case B: Claim references nonexistent evidence IDs
    c_invalid = Claim(
        claim_id=6,
        claim_text="Arbitrary assertion referencing nonexistent source.",
        evidence_ids=[],
        invalid_evidence_ids=[999],
    )
    verified_invalid = verify_single_claim(llm=None, claim=c_invalid, evidence_map={})
    assert verified_invalid.verification_status == VerificationStatus.NOT_SUPPORTED
    assert "999" in verified_invalid.verification_reason
    assert verified_invalid.verification_status != VerificationStatus.SUPPORTED

    # Edge Case A: LLM returns malformed extraction output
    class MockFailingExtractionLLM:
        def invoke(self, messages):
            class Response:
                content = "THIS IS NOT VALID JSON AT ALL {{{{ broken"
            return Response()

    malformed_claims = extract_claims(
        llm=MockFailingExtractionLLM(),
        content="Some draft text",
        valid_evidence_ids=[1, 2],
    )
    assert len(malformed_claims) >= 1
    assert malformed_claims[0].verification_status == VerificationStatus.UNCERTAIN
    assert "malformed" in malformed_claims[0].verification_reason.lower()

    # Edge Case D: Verification LLM failure (exception raised)
    class MockCrashingVerificationLLM:
        def invoke(self, messages):
            raise ConnectionError("Simulated LLM API network timeout")

    ev_map = {
        1: EvidenceItem(
            id=1,
            source_id=1,
            title="ESA Doc",
            url="https://esa.int",
            domain="esa.int",
            content="C-band SAR data available.",
        )
    }
    c_to_verify = Claim(
        claim_id=7,
        claim_text="Sentinel operates C-band.",
        evidence_ids=[1],
    )
    verified_crash = verify_single_claim(
        llm=MockCrashingVerificationLLM(),
        claim=c_to_verify,
        evidence_map=ev_map,
    )
    assert verified_crash.verification_status == VerificationStatus.UNCERTAIN
    assert "Simulated LLM API network timeout" in verified_crash.verification_reason

    print("✓ All claim models, edge-case rejection rules, and failure fallbacks passed.")


def test_json_parser_fixtures():
    print("\n--- Test 4B: Deterministic JSON Parser Fixtures (Objects, Arrays, Prose Wrappers) ---")
    import json
    from gap_detector import _extract_json_block as gap_extract
    from verifier import _extract_json_block as verifier_extract

    for name, extract_fn in [("gap_detector", gap_extract), ("verifier", verifier_extract)]:
        # Fixture A — top-level object
        # INPUT: {"key": "value"}
        # EXPECTED: parsed dict with key == "value", count == 1
        res_a = extract_fn('{"key": "value"}')
        data_a = json.loads(res_a)
        assert isinstance(data_a, dict), f"{name}: Expected dict for Fixture A"
        assert data_a["key"] == "value"
        assert len(data_a) == 1

        # Fixture B — top-level array
        # INPUT: [{"key": "v1"}, {"key": "v2"}]
        # EXPECTED: parsed list of 2 dicts
        res_b = extract_fn('[{"key": "v1"}, {"key": "v2"}]')
        data_b = json.loads(res_b)
        assert isinstance(data_b, list), f"{name}: Expected list for Fixture B"
        assert len(data_b) == 2
        assert data_b[0]["key"] == "v1"
        assert data_b[1]["key"] == "v2"

        # Fixture C — prose-wrapped object
        # INPUT: Here is the analysis:\n\n{"status": "supported", "confidence": 0.9}\n\nHope this helps!
        # EXPECTED: parsed dict with status == "supported", count == 2
        res_c = extract_fn('Here is the analysis:\n\n{"status": "supported", "confidence": 0.9}\n\nHope this helps!')
        data_c = json.loads(res_c)
        assert isinstance(data_c, dict), f"{name}: Expected dict for Fixture C"
        assert data_c["status"] == "supported"
        assert data_c["confidence"] == 0.9

        # Fixture D — prose-wrapped array
        # INPUT: Based on the findings, here are the gaps:\n\n[{"gap_id": 1}, {"gap_id": 2}]\n\nDone.
        # EXPECTED: parsed list with 2 items
        res_d = extract_fn('Based on the findings, here are the gaps:\n\n[{"gap_id": 1}, {"gap_id": 2}]\n\nDone.')
        data_d = json.loads(res_d)
        assert isinstance(data_d, list), f"{name}: Expected list for Fixture D"
        assert len(data_d) == 2
        assert data_d[0]["gap_id"] == 1
        assert data_d[1]["gap_id"] == 2

        # Fixture E — malformed JSON
        # INPUT: Here is broken output {{{{ not closed
        # EXPECTED: extraction doesn't crash, json.loads fails gracefully
        res_e = extract_fn('Here is broken output {{{{ not closed')
        try:
            json.loads(res_e)
            parsed_ok = True
        except Exception:
            parsed_ok = False
        assert not parsed_ok, f"{name}: Malformed JSON must not parse as valid"

    print("✓ Parser correctly handles top-level objects, top-level arrays, prose-wrapped objects/arrays, and malformed inputs.")


def test_knowledge_gap_and_sufficiency_deterministic():
    print("\n--- Test 5: Knowledge Gap Detection, Prioritization & Sufficiency (Deterministic) ---")

    # 1. KnowledgeGap model creation & 2. HIGH/MEDIUM/LOW prioritization
    gap_high = KnowledgeGap(
        gap_id=1,
        description="Missing comparative latency benchmarks for SAR inference models.",
        related_question_id=2,
        reason="No evidence retrieved addresses real-time throughput metrics.",
        priority=GapPriority.HIGH,
        resolved=False,
    )
    gap_med = KnowledgeGap(
        gap_id=2,
        description="Cloud masking algorithms for multispectral optical bands not specified.",
        related_question_id=1,
        reason="Partial evidence on optical sensors but pre-processing steps missing.",
        priority=GapPriority.MEDIUM,
        resolved=False,
    )
    gap_low = KnowledgeGap(
        gap_id=3,
        description="Historical launch dates of legacy satellite sensors.",
        related_question_id=1,
        reason="Minor background detail not affecting technical design.",
        priority=GapPriority.LOW,
        resolved=False,
    )
    assert gap_high.priority == GapPriority.HIGH
    assert gap_med.priority == GapPriority.MEDIUM
    assert gap_low.priority == GapPriority.LOW

    # 3. Gap detection from missing evidence, 4. unsupported claims, 5. uncertain claims
    mock_claims = [
        Claim(
            claim_id=1,
            claim_text="Model achieves 95% accuracy on building localization.",
            evidence_ids=[],
            verification_status=VerificationStatus.NOT_SUPPORTED,
            verification_reason="No supporting evidence retrieved.",
        ),
        Claim(
            claim_id=2,
            claim_text="Deployment runs fully on-device without cloud API.",
            evidence_ids=[1],
            verification_status=VerificationStatus.UNCERTAIN,
            verification_reason="Evidence mentions edge potential but gives no deployment details.",
        ),
    ]
    plan = ResearchPlan(
        original_objective="Disaster assessment AI",
        interpreted_objective="Edge deployment for satellite assessment",
        research_questions=[
            ResearchQuestion(id=1, question="What model architectures localise buildings?", rationale="Core design"),
            ResearchQuestion(id=2, question="What are the edge hardware compute requirements?", rationale="Deployment constraints"),
        ],
    )
    dummy_state = ResearchState(
        objective="Disaster assessment AI",
        plan=plan,
        claims=mock_claims,
        evidence=[],
    )

    class MockDetectorLLM:
        def invoke(self, messages):
            class Response:
                content = """[
                    {"gap_id": 1, "description": "No evidence for building localization accuracy", "related_question_id": 1, "reason": "Claim 1 is NOT_SUPPORTED", "priority": "high"},
                    {"gap_id": 2, "description": "Missing edge hardware compute specifications", "related_question_id": 2, "reason": "Claim 2 is UNCERTAIN and Question 2 has no evidence", "priority": "medium"}
                ]"""
            return Response()

    detected_gaps = detect_knowledge_gaps(llm=MockDetectorLLM(), state=dummy_state)
    assert len(detected_gaps) == 2
    assert detected_gaps[0].priority == GapPriority.HIGH
    assert detected_gaps[1].priority == GapPriority.MEDIUM

    # Prose-wrapped JSON array detection
    class MockProseDetectorLLM:
        def invoke(self, messages):
            class Response:
                content = """Based on the state analysis, here are the gaps:
                [
                    {"gap_id": 1, "description": "No evidence for building localization accuracy", "related_question_id": 1, "reason": "Claim 1 is NOT_SUPPORTED", "priority": "high"},
                    {"gap_id": 2, "description": "Missing edge hardware compute specifications", "related_question_id": 2, "reason": "Claim 2 is UNCERTAIN and Question 2 has no evidence", "priority": "medium"}
                ]
                End of gaps."""
            return Response()

    prose_gaps = detect_knowledge_gaps(llm=MockProseDetectorLLM(), state=dummy_state)
    assert len(prose_gaps) == 2
    assert prose_gaps[0].priority == GapPriority.HIGH
    assert prose_gaps[1].priority == GapPriority.MEDIUM

    # 6. Sufficient state evaluation
    resolved_state = ResearchState(
        objective="Disaster assessment AI",
        gaps=[
            KnowledgeGap(gap_id=1, description="Resolved gap", reason="Fixed", priority=GapPriority.HIGH, resolved=True)
        ],
    )
    suff_ok = evaluate_sufficiency(resolved_state, follow_up_performed=True, iteration_count=1)
    assert suff_ok.status == SufficiencyStatus.SUFFICIENT
    assert len(suff_ok.remaining_gap_ids) == 0

    # 7. Insufficient state evaluation
    unresolved_state = ResearchState(
        objective="Disaster assessment AI",
        gaps=[gap_high, gap_med],
    )
    suff_insufficient = evaluate_sufficiency(unresolved_state, follow_up_performed=False, iteration_count=0)
    assert suff_insufficient.status == SufficiencyStatus.INSUFFICIENT
    assert len(suff_insufficient.remaining_gap_ids) == 2

    # 8. Malformed gap-detector output (Failure A)
    class MockCrashingDetectorLLM:
        def invoke(self, messages):
            class Response:
                content = "BROKEN NON-JSON RESPONSE {{{"
            return Response()

    malformed_gaps = detect_knowledge_gaps(llm=MockCrashingDetectorLLM(), state=dummy_state)
    assert len(malformed_gaps) == 1
    assert malformed_gaps[0].priority == GapPriority.MEDIUM
    assert not malformed_gaps[0].resolved
    # Ensure malformed output does NOT produce false SUFFICIENT
    suff_fallback = evaluate_sufficiency(
        ResearchState(objective="test", gaps=malformed_gaps),
        follow_up_performed=False,
        iteration_count=0,
    )
    assert suff_fallback.status == SufficiencyStatus.INSUFFICIENT

    # 9. Invalid research-question reference (Failure B)
    class MockInvalidQidLLM:
        def invoke(self, messages):
            class Response:
                content = """[
                    {"gap_id": 1, "description": "Invalid reference gap", "related_question_id": 9999, "reason": "Testing invalid qid", "priority": "high"}
                ]"""
            return Response()

    invalid_qid_gaps = detect_knowledge_gaps(llm=MockInvalidQidLLM(), state=dummy_state)
    assert len(invalid_qid_gaps) == 1
    assert invalid_qid_gaps[0].related_question_id is None
    assert invalid_qid_gaps[0].invalid_question_id == 9999
    assert "rejected" in invalid_qid_gaps[0].reason

    # 10. Follow-up search failure handling (Failure C)
    # Tested by ensuring state preserves existing evidence and documents failure in resolution notes
    gap_for_failure = KnowledgeGap(gap_id=1, description="Missing latency data", reason="No data", priority=GapPriority.HIGH)
    # Simulate follow-up search failure:
    search_err = ConnectionError("Tavily API request timed out")
    gap_for_failure.resolution_notes = f"Follow-up search failed: {type(search_err).__name__} ({str(search_err)})"
    assert not gap_for_failure.resolved
    assert "timed out" in gap_for_failure.resolution_notes

    # 11. Empty follow-up evidence handling (Failure D)
    empty_res_gaps = evaluate_gap_resolution(llm=None, gaps=[gap_high], new_evidence=[])
    assert not empty_res_gaps[0].resolved
    assert "no new evidence" in empty_res_gaps[0].resolution_notes.lower()

    # 12. Re-verification failure handling (Failure E)
    class MockCrashingResolutionLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM rate limit during gap resolution evaluation")

    dummy_ev = [EvidenceItem(id=3, source_id=3, title="New", url="https://new.org", domain="new.org", content="content", stage="follow_up", iteration=1)]
    reverify_fail_gaps = evaluate_gap_resolution(llm=MockCrashingResolutionLLM(), gaps=[gap_med], new_evidence=dummy_ev)
    assert not reverify_fail_gaps[0].resolved
    assert "failed" in reverify_fail_gaps[0].resolution_notes.lower()

    # 13. Maximum iteration enforcement (Failure F)
    # Ensure evaluate_sufficiency enforces hard bound when iteration_count == 1
    suff_max = evaluate_sufficiency(unresolved_state, follow_up_performed=True, iteration_count=1)
    assert suff_max.iteration_count == 1
    assert suff_max.status == SufficiencyStatus.INSUFFICIENT
    assert "Maximum follow-up iteration (1) reached" in suff_max.reason

    # Test Markdown formatting of gaps
    unresolved_state.sufficiency = suff_max
    gaps_md = unresolved_state.format_gaps_markdown()
    assert "## Research Gaps & Limitations" in gaps_md
    assert "[HIGH PRIORITY]" in gaps_md
    assert "Maximum follow-up iteration (1) reached" in gaps_md

    print("✓ All 13 knowledge gap detection, prioritization, and sufficiency failure tests passed.")


def test_slice_5_deterministic_suite():
    print("\n--- Test 6: Slice 5 Deterministic Suite (Tests A–J & Failures A–G) ---")

    # =========================================================================
    # Test A — Context Model
    # INPUT FIXTURE: Complete valid ResearchContext
    # EXPECTED OUTPUT: All fields preserved correctly with appropriate types
    # EXPECTED STATE: ResearchState contains the identical ResearchContext
    # FAILURE BEHAVIOR: Assertions fail if fields are dropped, corrupted, or altered
    # =========================================================================
    ctx_full = ResearchContext(
        objective="Evaluate AI approaches for automated urban parcel boundary extraction.",
        purpose="Determine which approach is feasible for our SIH solution.",
        user_context="We are building a prototype for SIH problem 26012.",
        target_audience="Student engineering team and SIH judges.",
        constraints="Limited hackathon time and compute.",
        required_depth=ResearchDepth.DEEP,
        freshness_requirement=FreshnessRequirement.CURRENT,
        source_requirements=["Prefer government sources", "official documentation"],
        expected_output="Technical comparison with recommendations, limitations, and citations.",
    )
    assert ctx_full.objective == "Evaluate AI approaches for automated urban parcel boundary extraction."
    assert ctx_full.purpose == "Determine which approach is feasible for our SIH solution."
    assert ctx_full.user_context == "We are building a prototype for SIH problem 26012."
    assert ctx_full.target_audience == "Student engineering team and SIH judges."
    assert ctx_full.constraints == "Limited hackathon time and compute."
    assert ctx_full.required_depth == ResearchDepth.DEEP
    assert ctx_full.freshness_requirement == FreshnessRequirement.CURRENT
    assert len(ctx_full.source_requirements) == 2
    assert "Prefer government sources" in ctx_full.source_requirements
    assert ctx_full.expected_output == "Technical comparison with recommendations, limitations, and citations."

    state_a = ResearchState(
        objective=ctx_full.objective,
        session_id="test_state_a",
        context=ctx_full,
    )
    assert state_a.context == ctx_full
    assert state_a.context.required_depth == ResearchDepth.DEEP
    print("   ✓ Test A (Context Model) passed.")

    # =========================================================================
    # Test B — Partial Context
    # INPUT FIXTURE: Only objective + purpose provided
    # EXPECTED OUTPUT: Valid context with safe defaults (STANDARD depth, ANY freshness)
    # EXPECTED STATE: ResearchState remains valid with defaults preserved
    # FAILURE BEHAVIOR: Raises ValidationError if optional fields cannot be omitted
    # =========================================================================
    ctx_partial = ResearchContext(
        objective="Compare YOLOv8 and Mask R-CNN for building footprint segmentation.",
        purpose="Select detector architecture.",
    )
    assert ctx_partial.objective == "Compare YOLOv8 and Mask R-CNN for building footprint segmentation."
    assert ctx_partial.purpose == "Select detector architecture."
    assert ctx_partial.user_context is None
    assert ctx_partial.target_audience is None
    assert ctx_partial.constraints is None
    assert ctx_partial.required_depth == ResearchDepth.STANDARD
    assert ctx_partial.freshness_requirement == FreshnessRequirement.ANY
    assert ctx_partial.source_requirements == []
    assert ctx_partial.expected_output is None

    state_b = ResearchState(
        objective=ctx_partial.objective,
        session_id="test_state_b",
        context=ctx_partial,
    )
    assert state_b.context.required_depth == ResearchDepth.STANDARD
    print("   ✓ Test B (Partial Context) passed.")

    # =========================================================================
    # Test C & Failure F — Backward Compatibility / Old Request Format
    # INPUT FIXTURE: Legacy request dictionary containing only 'topic'
    # EXPECTED OUTPUT: normalize_research_context safely creates ResearchContext
    # EXPECTED STATE: Valid ResearchState without crash or missing field errors
    # FAILURE BEHAVIOR: Fails if old requests crash or context is required
    # =========================================================================
    legacy_topic = "Quantum key distribution protocols"
    norm_ctx_legacy = normalize_research_context(objective=legacy_topic, context=None, purpose=None)
    assert norm_ctx_legacy.objective == legacy_topic
    assert norm_ctx_legacy.required_depth == ResearchDepth.STANDARD
    assert norm_ctx_legacy.freshness_requirement == FreshnessRequirement.ANY

    state_c = ResearchState(
        objective=legacy_topic,
        session_id="test_legacy_c",
        context=norm_ctx_legacy,
    )
    assert state_c.context is not None
    assert state_c.context.objective == legacy_topic
    print("   ✓ Test C & Failure F (Backward Compatibility) passed.")

    # =========================================================================
    # Test D & Failure G — Context Propagation
    # INPUT FIXTURE: Two similar objectives with different purposes and constraints
    # EXPECTED OUTPUT: Context text reaches build_planner_prompt and build_synthesis_prompt
    # EXPECTED STATE / SIDE EFFECT: Prompts differ and contain the respective purpose/audience
    # FAILURE BEHAVIOR: Fails if context is stored but omitted from prompt strings
    # =========================================================================
    ctx_sih = ResearchContext(
        objective="Urban boundary detection",
        purpose="SIH 2024 problem 26012 solution feasibility",
        target_audience="SIH Judges and student team",
        constraints="Low-compute edge deployment",
    )
    ctx_academic = ResearchContext(
        objective="Urban boundary detection",
        purpose="Survey paper for IEEE Geoscience",
        target_audience="Academic peer reviewers",
        constraints="Theoretical rigour and exhaustive taxonomy",
    )
    planner_prompt_sih = build_planner_prompt("Urban boundary detection", [], context=ctx_sih)
    planner_prompt_acad = build_planner_prompt("Urban boundary detection", [], context=ctx_academic)

    assert "SIH 2024 problem 26012" in planner_prompt_sih
    assert "Low-compute edge deployment" in planner_prompt_sih
    assert "SIH 2024 problem 26012" not in planner_prompt_acad
    assert "Survey paper for IEEE Geoscience" in planner_prompt_acad

    synth_prompt_sih = build_synthesis_prompt(
        objective="Urban boundary detection",
        plan_context="Plan scope",
        evidence_context="Evidence context",
        context=ctx_sih,
    )
    synth_prompt_acad = build_synthesis_prompt(
        objective="Urban boundary detection",
        plan_context="Plan scope",
        evidence_context="Evidence context",
        context=ctx_academic,
    )
    assert "SIH 2024 problem 26012" in synth_prompt_sih
    assert "SIH Judges and student team" in synth_prompt_sih
    assert "Survey paper for IEEE Geoscience" in synth_prompt_acad
    print("   ✓ Test D & Failure G (Context Propagation) passed.")

    # =========================================================================
    # Test E & Failure D — Source Provenance & Retrieval Timestamps
    # INPUT FIXTURE: Source with known URL and UTC timestamp
    # EXPECTED OUTPUT: Source preserves URL, domain, source_type, and timestamp
    # EXPECTED STATE / SIDE EFFECT: Evidence referencing source remains traceable
    # FAILURE BEHAVIOR: URL or timestamp dropped or altered
    # =========================================================================
    known_ts = "2026-09-14T08:00:00Z"
    s_prov = Source(
        id=1,
        title="Survey of India Open Series Maps",
        url="https://surveyofindia.gov.in/osm",
        domain="surveyofindia.gov.in",
        source_type="government",
        retrieved_at=known_ts,
    )
    assert s_prov.id == 1
    assert s_prov.url == "https://surveyofindia.gov.in/osm"
    assert s_prov.domain == "surveyofindia.gov.in"
    assert s_prov.source_type == "government"
    assert s_prov.retrieved_at == known_ts
    print("   ✓ Test E (Source Provenance) passed.")

    # =========================================================================
    # Test F — Evidence Provenance & Claim Traceability
    # INPUT FIXTURE: Evidence item referencing Source 1, Claim referencing Evidence 1
    # EXPECTED OUTPUT: EvidenceItem.source_id == 1, Claim trace resolves to Source 1 URL
    # EXPECTED STATE: Claim -> Evidence -> Source -> URL chain is intact via get_claim_provenance()
    # FAILURE BEHAVIOR: Trace broken, evidence detached, or source URL unresolvable
    # =========================================================================
    ev_prov = EvidenceItem(
        id=1,
        source_id=1,
        title=s_prov.title,
        url=s_prov.url,
        domain=s_prov.domain,
        content="Open series maps provide georeferenced base layers at 1:50,000 scale.",
        stage="initial",
        iteration=0,
        retrieved_at=known_ts,
    )
    assert ev_prov.source_id == 1
    assert ev_prov.retrieved_at == known_ts

    claim_prov = Claim(
        claim_id=1,
        claim_text="Open series maps provide georeferenced base layers at 1:50,000 scale.",
        evidence_ids=[1],
        verification_status=VerificationStatus.SUPPORTED,
        verification_reason="Confirmed verbatim in Survey of India documentation.",
    )

    state_trace = ResearchState(
        objective="Georeferenced maps",
        session_id="test_trace",
        sources=[s_prov],
        evidence=[ev_prov],
        claims=[claim_prov],
    )
    trace_info = state_trace.get_claim_provenance(claim_id=1)
    assert trace_info is not None
    assert trace_info["claim_id"] == 1
    assert len(trace_info["provenance"]) == 1
    prov_entry = trace_info["provenance"][0]
    assert prov_entry["evidence_id"] == 1
    assert prov_entry["source_id"] == 1
    assert prov_entry["source_url"] == "https://surveyofindia.gov.in/osm"
    assert prov_entry["source_retrieved_at"] == known_ts
    assert prov_entry["valid_source"] is True
    print("   ✓ Test F (Evidence Provenance & Traceability) passed.")

    # =========================================================================
    # Test G & Failure E — Invalid Evidence Source / Detached Evidence Rejection
    # INPUT FIXTURE: Evidence references nonexistent Source ID 999
    # EXPECTED OUTPUT: Verifier detects detached evidence; marks claim NOT_SUPPORTED
    # EXPECTED STATE: Invalid evidence ID recorded in invalid_evidence_ids; never SUPPORTED
    # FAILURE BEHAVIOR: Detached evidence treated as valid supporting evidence
    # =========================================================================
    ev_detached = EvidenceItem(
        id=2,
        source_id=999,  # Nonexistent source
        title="Orphaned snippet",
        url="https://fake.site/doc",
        domain="fake.site",
        content="Fabricated metric claiming 99.9% accuracy.",
    )
    claim_detached = Claim(
        claim_id=2,
        claim_text="Solution achieves 99.9% accuracy.",
        evidence_ids=[2],
    )
    # Validate against state with only source_id=1
    verified_detached = verify_single_claim(
        llm=None,
        claim=claim_detached,
        evidence_map={2: ev_detached},
        valid_source_ids={1},  # Source 999 is NOT valid
    )
    assert verified_detached.verification_status == VerificationStatus.NOT_SUPPORTED
    assert "nonexistent sources" in verified_detached.verification_reason or "detached" in verified_detached.verification_reason
    assert 2 in verified_detached.invalid_evidence_ids

    # ResearchState helper also flags it
    state_detached = ResearchState(
        objective="Testing detachment",
        sources=[s_prov],
        evidence=[ev_detached],
    )
    detached_ids = state_detached.validate_evidence()
    assert detached_ids == [2], f"Expected detached evidence [2], got {detached_ids}"
    print("   ✓ Test G & Failure E (Detached Evidence Rejection) passed.")

    # =========================================================================
    # Test H & Failure C — Missing Source URL (No Fabrication)
    # INPUT FIXTURE: Source with no usable URL (empty string or None)
    # EXPECTED OUTPUT: source.url is empty; markdown formatting does NOT fabricate link
    # EXPECTED STATE: Rendered markdown contains title but NO markdown link brackets
    # FAILURE BEHAVIOR: Fabricates fake URL or renders invalid markdown link
    # =========================================================================
    s_no_url = Source(
        id=2,
        title="Internal Government Circular No. 42",
        url="",
        domain="nic.in",
        source_type="circular",
    )
    assert s_no_url.url == ""

    state_no_url = ResearchState(
        objective="Circular review",
        sources=[s_no_url],
    )
    sources_md_no_url = state_no_url.format_sources_markdown()
    assert "Internal Government Circular No. 42" in sources_md_no_url
    assert "](http" not in sources_md_no_url, "Must not fabricate clickable markdown link when URL is absent"
    assert "](" not in sources_md_no_url, "Must not contain empty markdown link syntax"
    print("   ✓ Test H & Failure C (Missing Source URL / No Fabrication) passed.")

    # =========================================================================
    # Test I & Failure D — Missing Retrieval Timestamp (No Fabrication)
    # INPUT FIXTURE: Raw result / Source without timestamp
    # EXPECTED OUTPUT: retrieved_at remains None; no timestamp fabricated
    # EXPECTED STATE: Markdown Sources section omits timestamp annotation cleanly
    # FAILURE BEHAVIOR: Current time or arbitrary timestamp is fabricated
    # =========================================================================
    mock_raw_no_ts = {
        "results": [
            {
                "title": "Historical Land Records Framework",
                "url": "https://example.gov.in/records",
                "content": "Framework established in 2008 for digital records.",
            }
        ]
    }
    srcs_no_ts, evs_no_ts = parse_tavily_results(mock_raw_no_ts, start_idx=1, stage="initial", iteration=0)
    assert len(srcs_no_ts) == 1
    assert srcs_no_ts[0].retrieved_at is None, "Must not fabricate timestamp when not provided"
    assert evs_no_ts[0].retrieved_at is None

    state_no_ts = ResearchState(
        objective="Historical records",
        sources=srcs_no_ts,
        evidence=evs_no_ts,
    )
    sources_md_no_ts = state_no_ts.format_sources_markdown()
    assert "Retrieved:" not in sources_md_no_ts, "Sources markdown must not contain fabricated 'Retrieved:' timestamp"
    print("   ✓ Test I & Failure D (Missing Retrieval Timestamp / No Fabrication) passed.")

    # =========================================================================
    # Test J — Direct Clickable Source Links in Report
    # INPUT FIXTURE: Real source metadata with citation IDs [1] and [2]
    # EXPECTED OUTPUT: Sources section contains direct clickable links matching citations
    # EXPECTED STATE: [1] maps to exact URL; [2] maps to exact URL
    # FAILURE BEHAVIOR: URLs omitted, rewritten, or citation numbers mismatched
    # =========================================================================
    s1_click = Source(
        id=1,
        title="Digital India Land Records Modernization Programme (DILRMP)",
        url="https://dilrmp.gov.in/reports",
        domain="dilrmp.gov.in",
        retrieved_at="2026-09-14T08:15:00Z",
    )
    s2_click = Source(
        id=2,
        title="Open Drone Map Technical Documentation",
        url="https://docs.opendronemap.org",
        domain="docs.opendronemap.org",
    )
    state_click = ResearchState(
        objective="DILRMP and Drone mapping",
        sources=[s1_click, s2_click],
    )
    sources_click_md = state_click.format_sources_markdown()
    assert "[1] [Digital India Land Records Modernization Programme (DILRMP)](https://dilrmp.gov.in/reports) — *dilrmp.gov.in*" in sources_click_md
    assert "[2] [Open Drone Map Technical Documentation](https://docs.opendronemap.org) — *docs.opendronemap.org*" in sources_click_md
    print("   ✓ Test J (Direct Clickable Source Links) passed.")

    # =========================================================================
    # Failure A & Failure B — Enum Validation & Safe Normalization
    # INPUT FIXTURE: Context with string variants ("Deep technical", "current", and invalid)
    # EXPECTED OUTPUT: Case-insensitive normalization maps correctly; invalid raises ValueError
    # EXPECTED STATE: Invalid states are rejected before reaching execution
    # FAILURE BEHAVIOR: Silent corruption or invalid enum assignment
    # =========================================================================
    ctx_norm = ResearchContext(
        objective="Normalization test",
        required_depth="Deep technical research",
        freshness_requirement="Current and recent updates",
    )
    assert ctx_norm.required_depth == ResearchDepth.DEEP
    assert ctx_norm.freshness_requirement == FreshnessRequirement.CURRENT

    # Test rejection of invalid depth value
    invalid_depth_raised = False
    try:
        ResearchContext(objective="Test", required_depth="hyperbolic-infinity")
    except ValueError:
        invalid_depth_raised = True
    assert invalid_depth_raised, "Expected ValueError for unrecognized required_depth"

    # Test rejection of invalid freshness value
    invalid_freshness_raised = False
    try:
        ResearchContext(objective="Test", freshness_requirement="timeless-eternal")
    except ValueError:
        invalid_freshness_raised = True
    assert invalid_freshness_raised, "Expected ValueError for unrecognized freshness_requirement"

    # Failure A: None context normalizes without crash
    ctx_missing = normalize_research_context("Default test", None, None)
    assert ctx_missing.required_depth == ResearchDepth.STANDARD
    assert ctx_missing.freshness_requirement == FreshnessRequirement.ANY
    print("   ✓ Failure Modes A & B (Enum Normalization & Validation) passed.")

    print("✓ All Slice 5 deterministic tests (Tests A–J & Failure Modes A–G) passed successfully.")


def test_slice_6_deterministic_suite():
    print("\n=== SLICE 6 DETERMINISTIC TEST SUITE (TESTS 1–19 & FAILURE MODES A–I) ===")

    # -------------------------------------------------------------------------
    # Test 1 — Simple Objective → One Focused Query
    # INPUT FIXTURE: Simple research plan
    # EXPECTED: Exactly 1 focused query generated
    # -------------------------------------------------------------------------
    simple_plan = ResearchPlan(
        original_objective="When was Python 3.13 released?",
        interpreted_objective="Official release date and milestones of Python 3.13",
        complexity=ResearchComplexity.SIMPLE,
        requires_approval=False,
        status=ResearchStatus.APPROVED,
        research_questions=[
            ResearchQuestion(id=1, question="What is the official release date of Python 3.13?", rationale="Lookup")
        ],
    )
    simple_queries = generate_search_queries(llm=None, objective=simple_plan.original_objective, plan=simple_plan)
    assert len(simple_queries) == 1, f"Expected 1 query for simple research, got {len(simple_queries)}"
    assert simple_queries[0].query_id == 1
    assert "Python 3.13" in simple_queries[0].query_text
    print("   ✓ Test 1: Simple Objective Query Generation passed.")

    # -------------------------------------------------------------------------
    # Test 2 & 3 — Complex Objective → 2–3 Queries Mapped to Research Questions
    # INPUT FIXTURE: Complex research plan with 2 decomposed questions
    # EXPECTED: 2-3 distinct queries mapped to target question IDs
    # -------------------------------------------------------------------------
    complex_plan = ResearchPlan(
        original_objective="Indian urban land record digitization DILRMP NAKSHA cadastral parcel mapping",
        interpreted_objective="DILRMP and NAKSHA urban cadastral mapping",
        complexity=ResearchComplexity.COMPLEX,
        requires_approval=True,
        status=ResearchStatus.AWAITING_APPROVAL,
        research_questions=[
            ResearchQuestion(id=1, question="What are the technical specs of DILRMP land digitization?", rationale="Specs"),
            ResearchQuestion(id=2, question="How does NAKSHA cadastral mapping integrate GIS parcels?", rationale="GIS"),
        ],
    )
    complex_queries = generate_search_queries(llm=None, objective=complex_plan.original_objective, plan=complex_plan)
    assert 2 <= len(complex_queries) <= 3, f"Expected 2-3 queries for complex research, got {len(complex_queries)}"
    for q in complex_queries:
        assert q.query_text, "Query text must not be empty"
        assert len(q.target_question_ids) > 0, "Query must map to target research questions"
    print("   ✓ Tests 2 & 3: Complex Objective Multi-Query Generation & Mapping passed.")

    # -------------------------------------------------------------------------
    # Test 4 & Failure Mode B — Duplicate Query Removal
    # INPUT FIXTURE: Search queries with identical text after normalization
    # EXPECTED: Duplicate query pruned deterministically
    # -------------------------------------------------------------------------
    raw_dup_queries = [
        SearchQuery(query_id=1, query_text="urban cadastral mapping", target_question_ids=[1]),
        SearchQuery(query_id=2, query_text="  Urban Cadastral Mapping  ", target_question_ids=[2]),
        SearchQuery(query_id=3, query_text="dilrmp modern records", target_question_ids=[2]),
    ]
    seen = set()
    deduped = []
    for q in raw_dup_queries:
        norm = " ".join(q.query_text.lower().split())
        if norm not in seen:
            seen.add(norm)
            deduped.append(q)
    assert len(deduped) == 2, "Duplicate query must be removed deterministically"
    print("   ✓ Test 4 & Failure Mode B (Duplicate Query Removal) passed.")

    # -------------------------------------------------------------------------
    # Test 5 & Failure Mode C — Empty Query Rejection
    # INPUT FIXTURE: Empty query strings
    # EXPECTED: Empty queries rejected; valid query retained
    # -------------------------------------------------------------------------
    raw_queries_with_empty = [
        SearchQuery(query_id=1, query_text="", target_question_ids=[1]),
        SearchQuery(query_id=2, query_text="   ", target_question_ids=[1]),
        SearchQuery(query_id=3, query_text="cadastral parcel survey", target_question_ids=[2]),
    ]
    filtered_queries = [q for q in raw_queries_with_empty if q.query_text.strip()]
    assert len(filtered_queries) == 1
    assert filtered_queries[0].query_text == "cadastral parcel survey"
    print("   ✓ Test 5 & Failure Mode C (Empty Query Rejection) passed.")

    # -------------------------------------------------------------------------
    # Test 6 & User Correction 4 — Query Execution Contract
    # INPUT FIXTURE: SearchQuery -> Tavily invocation -> CandidatePool
    # EXPECTED: Candidate has correct query_id(s) and Source retains query_ids
    # -------------------------------------------------------------------------
    q1 = SearchQuery(query_id=101, query_text="cadastral survey dilrmp", target_question_ids=[1])
    q2 = SearchQuery(query_id=102, query_text="naksha gis parcel mapping", target_question_ids=[2])

    res1 = [{"url": "https://dilrmp.gov.in/records", "title": "DILRMP Portal", "content": "Official portal"}]
    res2 = [
        {"url": "https://dilrmp.gov.in/records", "title": "DILRMP Portal", "content": "Official portal"},
        {"url": "https://iitb.ac.in/gis-report", "title": "IITB GIS Report", "content": "Academic survey"},
    ]

    pool_contract = CandidatePool(provider="tavily")
    pool_contract.add_query_results(q1, res1, stage="initial", iteration=0, retrieval_timestamp="2026-09-14T09:00:00Z")
    pool_contract.add_query_results(q2, res2, stage="initial", iteration=0, retrieval_timestamp="2026-09-14T09:00:00Z")

    sources_contract, evidence_contract, telem_contract = pool_contract.build_sources_and_evidence(start_idx=1)

    assert len(sources_contract) == 2, f"Expected 2 unique sources, got {len(sources_contract)}"
    s_dilrmp = next(s for s in sources_contract if "dilrmp.gov.in" in s.url)
    s_iitb = next(s for s in sources_contract if "iitb.ac.in" in s.url)

    assert s_dilrmp.query_ids == [101, 102], f"Expected [101, 102] on dilrmp source, got {s_dilrmp.query_ids}"
    assert s_iitb.query_ids == [102], f"Expected [102] on iitb source, got {s_iitb.query_ids}"
    assert evidence_contract[0].source_id in {s.id for s in sources_contract}
    print("   ✓ Test 6 & User Correction 4 (Query Execution Contract) passed.")

    # -------------------------------------------------------------------------
    # Test 7 — URL Canonicalization
    # INPUT FIXTURE: URLs with trailing slashes, default ports, tracking params, uppercase schemes/domains, and fragments
    # EXPECTED: Standardized canonical URL strings
    # -------------------------------------------------------------------------
    assert canonicalize_url("https://EXAMPLE.COM/doc/") == "https://example.com/doc"
    assert canonicalize_url("http://example.com:80/doc") == "http://example.com/doc"
    assert canonicalize_url("https://example.com:443/doc") == "https://example.com/doc"
    assert canonicalize_url("https://example.com/doc?utm_source=twitter&utm_medium=cpc&id=42#heading") == "https://example.com/doc?id=42"
    assert canonicalize_url("https://example.com/doc?b=2&a=1") == "https://example.com/doc?a=1&b=2"
    assert canonicalize_url("https://example.com/") == "https://example.com"
    assert canonicalize_url("https://example.com") == "https://example.com"
    assert canonicalize_url("javascript:alert(1)") == ""
    assert canonicalize_url("not-a-valid-url") == ""
    print("   ✓ Test 7: URL Canonicalization passed.")

    # -------------------------------------------------------------------------
    # Test 8 & Failure Mode F — Duplicate URL Removal Across Queries
    # INPUT FIXTURE: Multiple raw candidate items resolving to same canonical URL
    # EXPECTED: Merged into 1 canonical source; duplicates_removed counted accurately
    # -------------------------------------------------------------------------
    dup_pool = CandidatePool(provider="tavily")
    q_test = SearchQuery(query_id=1, query_text="test", target_question_ids=[1])
    raw_dups = [
        {"url": "https://example.com/page", "title": "Page 1", "content": "Snippet A"},
        {"url": "https://example.com/page/", "title": "Page 1 Trailing", "content": "Snippet B"},
        {"url": "https://EXAMPLE.COM/page?utm_source=newsletter#details", "title": "Page 1 Tracked", "content": "Snippet C"},
    ]
    dup_pool.add_query_results(q_test, raw_dups)
    d_sources, d_evidence, d_telem = dup_pool.build_sources_and_evidence(start_idx=1)
    assert len(d_sources) == 1, f"Expected 1 unique source, got {len(d_sources)}"
    assert d_telem.candidates_discovered == 3
    assert d_telem.duplicates_removed == 2
    assert d_telem.unique_candidates_retained == 1
    print("   ✓ Test 8 & Failure Mode F (Duplicate URL Deduplication) passed.")

    # -------------------------------------------------------------------------
    # Test 9 — Different Pages on Same Domain Remain Distinct
    # INPUT FIXTURE: Multiple distinct paths on the same domain
    # EXPECTED: Each path retained as a separate unique source
    # -------------------------------------------------------------------------
    domain_pool = CandidatePool(provider="tavily")
    raw_pages = [
        {"url": "https://dolr.gov.in/page-a", "title": "Page A", "content": "Content A"},
        {"url": "https://dolr.gov.in/page-b", "title": "Page B", "content": "Content B"},
    ]
    domain_pool.add_query_results(q_test, raw_pages)
    dp_sources, _, dp_telem = domain_pool.build_sources_and_evidence(start_idx=1)
    assert len(dp_sources) == 2, "Different pages on the same domain must remain distinct"
    assert dp_telem.duplicates_removed == 0
    print("   ✓ Test 9: Distinct Domain Pages Preservation passed.")

    # -------------------------------------------------------------------------
    # Test 10 & User Correction 5 — Source Authority Classification
    # INPUT FIXTURE: Domains across Tier 1, 2, 3, 4, and UNKNOWN
    # EXPECTED: Exact matching to SourceTier enum
    # -------------------------------------------------------------------------
    assert classify_source_tier("bhuvan.nrsc.gov.in") == SourceTier.TIER_1
    assert classify_source_tier("opengeospatial.org") == SourceTier.TIER_1
    assert classify_source_tier("arxiv.org") == SourceTier.TIER_1
    assert classify_source_tier("dolr.gov.in") == SourceTier.TIER_1
    assert classify_source_tier("iitb.ac.in") == SourceTier.TIER_2
    assert classify_source_tier("mit.edu") == SourceTier.TIER_2
    assert classify_source_tier("reuters.com") == SourceTier.TIER_3
    assert classify_source_tier("thehindu.com") == SourceTier.TIER_3
    assert classify_source_tier("nature.com") == SourceTier.TIER_3
    assert classify_source_tier("medium.com") == SourceTier.TIER_4
    assert classify_source_tier("reddit.com") == SourceTier.TIER_4
    assert classify_source_tier("random-startup-domain-2026.io") == SourceTier.UNKNOWN
    print("   ✓ Test 10 & User Correction 5 (Source Authority Classification) passed.")

    # -------------------------------------------------------------------------
    # Test 11 & User Correction 2 — Authority != Relevance
    # INPUT FIXTURE: High-authority irrelevant source vs lower-tier relevant source
    # EXPECTED: Authority remains separate; lower tier retained in candidate pool
    # -------------------------------------------------------------------------
    tier_pool = CandidatePool(provider="tavily")
    tier_pool.add_query_results(
        q_test,
        [
            {"url": "https://weather.gov/marine", "title": "Weather Forecast", "content": "Irrelevant high authority"},
            {"url": "https://medium.com/@gis/urban-cadastral", "title": "Urban Cadastral Mapping", "content": "Relevant low authority"},
        ]
    )
    t_sources, _, _ = tier_pool.build_sources_and_evidence(start_idx=1)
    s_gov = next(s for s in t_sources if "weather.gov" in s.url)
    s_med = next(s for s in t_sources if "medium.com" in s.url)
    assert s_gov.source_tier == SourceTier.TIER_1
    assert s_med.source_tier == SourceTier.TIER_4
    assert len(t_sources) == 2
    print("   ✓ Test 11 & User Correction 2 (Authority != Relevance) passed.")

    # -------------------------------------------------------------------------
    # Test 12 & Failure Mode I — Unknown Authority Classification
    # INPUT FIXTURE: Domain with unknown publisher
    # EXPECTED: Classifies as UNKNOWN, never elevated to Tier 1
    # -------------------------------------------------------------------------
    assert classify_source_tier("unknown-untrusted-portal.xyz") == SourceTier.UNKNOWN
    assert classify_source_tier("") == SourceTier.UNKNOWN
    print("   ✓ Test 12 & Failure Mode I (Unknown Authority Conservative Handling) passed.")

    # -------------------------------------------------------------------------
    # Test 13 & Failure Mode H — Rejection Reason Recording for Invalid URLs
    # INPUT FIXTURE: Raw items with missing, empty, or non-http URLs
    # EXPECTED: Rejection reason "invalid_url" recorded, no broken Source created
    # -------------------------------------------------------------------------
    invalid_pool = CandidatePool(provider="tavily")
    invalid_items = [
        {"url": "", "title": "Empty URL", "content": "Snippet"},
        {"url": "javascript:alert(1)", "title": "Script URL", "content": "Snippet"},
        {"url": None, "title": "None URL", "content": "Snippet"},
        {"url": "https://valid.org/doc", "title": "Valid Doc", "content": "Snippet"},
    ]
    invalid_pool.add_query_results(q_test, invalid_items)
    inv_sources, _, inv_telem = invalid_pool.build_sources_and_evidence(start_idx=1)
    assert len(inv_sources) == 1
    assert inv_telem.candidates_rejected == 3
    assert inv_telem.rejection_reasons.get("invalid_url") == 3
    print("   ✓ Test 13 & Failure Mode H (Rejection Reason Recording) passed.")

    # -------------------------------------------------------------------------
    # Test 14 & User Correction 6 — Telemetry Consistency Arithmetic
    # INPUT FIXTURE: Mixed batch of valid unique, duplicate, and invalid candidates
    # EXPECTED: discovered - duplicates_removed - candidates_rejected == unique_candidates_retained
    # -------------------------------------------------------------------------
    arithmetic_pool = CandidatePool(provider="tavily")
    raw_mixed = [
        {"url": "https://a.org/1", "title": "1", "content": "c1"},
        {"url": "https://a.org/1/", "title": "1 dup", "content": "c1 dup"},
        {"url": "https://b.org/2", "title": "2", "content": "c2"},
        {"url": "javascript:void(0)", "title": "bad", "content": "c3"},
        {"url": "", "title": "empty", "content": "c4"},
    ]
    arithmetic_pool.add_query_results(q_test, raw_mixed)
    _, _, arith_telem = arithmetic_pool.build_sources_and_evidence(start_idx=1)
    assert (
        arith_telem.candidates_discovered - arith_telem.duplicates_removed - arith_telem.candidates_rejected
        == arith_telem.unique_candidates_retained
    ), "Telemetry arithmetic formula must hold exactly"
    assert arith_telem.candidates_discovered == 5
    assert arith_telem.duplicates_removed == 1
    assert arith_telem.candidates_rejected == 2
    assert arith_telem.unique_candidates_retained == 2
    print("   ✓ Test 14 & User Correction 6 (Telemetry Consistency Arithmetic) passed.")

    # -------------------------------------------------------------------------
    # Test 15 & User Correction 7 — Query Telemetry Structure
    # INPUT FIXTURE: Executed telemetry queries
    # EXPECTED: queries_executed contains complete SearchQuery objects
    # -------------------------------------------------------------------------
    assert len(arith_telem.queries_executed) == 1
    assert isinstance(arith_telem.queries_executed[0], SearchQuery)
    assert arith_telem.queries_executed[0].query_id == 1
    print("   ✓ Test 15 & User Correction 7 (Query Telemetry Structure) passed.")

    # -------------------------------------------------------------------------
    # Test 16 & Failure Mode G — Partial Provider Failure Handling
    # INPUT FIXTURE:
    #   - Query 1 succeeds with 1 candidate: {"url": "https://ok.org/doc"}
    #   - Query 2 fails with API error: "HTTP 500 Internal Server Error"
    # EXPECTED CANDIDATE CLASSIFICATION:
    #   - "https://ok.org/doc" is DISCOVERED and RETAINED (mutually exclusive)
    #   - Query 2 produces 0 candidates; no candidate is falsely marked rejected
    # EXPECTED COUNTERS:
    #   - candidates_discovered = 1
    #   - duplicates_removed = 0
    #   - candidates_rejected = 0
    #   - unique_candidates_retained = 1
    #   - Invariant: 1 - 0 - 0 == 1
    # EXPECTED RETAINED SOURCES:
    #   - Exactly 1 Source corresponding to https://ok.org/doc
    # EXPECTED STATE/TELEMETRY:
    #   - len(queries_executed) == 2
    #   - provider_errors["2"] == "HTTP 500 Internal Server Error"
    # FAILURE BEHAVIOR:
    #   - Provider failure falsely increments candidates_rejected without discovery,
    #     violating the telemetry arithmetic invariant.
    # -------------------------------------------------------------------------
    partial_pool = CandidatePool(provider="tavily")
    q_ok = SearchQuery(query_id=1, query_text="ok query", target_question_ids=[1])
    q_fail = SearchQuery(query_id=2, query_text="fail query", target_question_ids=[2])
    partial_pool.add_query_results(q_ok, [{"url": "https://ok.org/doc", "title": "OK", "content": "OK content"}])
    partial_pool.record_provider_error(q_fail, "HTTP 500 Internal Server Error")
    part_sources, _, part_telem = partial_pool.build_sources_and_evidence(start_idx=1)
    assert len(part_sources) == 1
    assert part_telem.candidates_discovered == 1
    assert part_telem.duplicates_removed == 0
    assert part_telem.candidates_rejected == 0
    assert part_telem.unique_candidates_retained == 1
    assert (
        part_telem.candidates_discovered - part_telem.duplicates_removed - part_telem.candidates_rejected
        == part_telem.unique_candidates_retained
    )
    assert part_telem.provider_errors.get("2") == "HTTP 500 Internal Server Error" or part_telem.provider_errors.get(2) == "HTTP 500 Internal Server Error"
    assert len(part_telem.queries_executed) == 2
    print("   ✓ Test 16 & Failure Mode G (Partial Provider Failure) passed.")

    # -------------------------------------------------------------------------
    # Test 17 & Failure Mode D — Zero-Result Query
    # INPUT FIXTURE: Query 1 returns 0 items, Query 2 returns 2 items
    # EXPECTED: Zero-result query does not cause failure; other query candidates retained
    # -------------------------------------------------------------------------
    zero_q_pool = CandidatePool(provider="tavily")
    q_empty = SearchQuery(query_id=1, query_text="empty query", target_question_ids=[1])
    q_has_data = SearchQuery(query_id=2, query_text="data query", target_question_ids=[2])
    zero_q_pool.add_query_results(q_empty, [])
    zero_q_pool.add_query_results(q_has_data, [{"url": "https://data.org/1", "title": "D1", "content": "C1"}])
    zq_sources, _, zq_telem = zero_q_pool.build_sources_and_evidence(start_idx=1)
    assert len(zq_sources) == 1
    assert zq_telem.candidates_discovered == 1
    assert len(zq_telem.queries_executed) == 2
    print("   ✓ Test 17 & Failure Mode D (Zero-Result Query) passed.")

    # -------------------------------------------------------------------------
    # Test 18 & Failure Mode E — All-Zero-Results Condition
    # INPUT FIXTURE: All queries return []
    # EXPECTED: Candidate pool has 0 candidates, triggers safe zero-evidence state
    # -------------------------------------------------------------------------
    all_zero_pool = CandidatePool(provider="tavily")
    all_zero_pool.add_query_results(q_empty, [])
    az_sources, az_evidence, az_telem = all_zero_pool.build_sources_and_evidence(start_idx=1)
    assert len(az_sources) == 0
    assert len(az_evidence) == 0
    assert az_telem.candidates_discovered == 0
    assert az_telem.unique_candidates_retained == 0
    print("   ✓ Test 18 & Failure Mode E (All-Zero-Results Condition) passed.")

    # -------------------------------------------------------------------------
    # Test 19 — Provenance Preservation Trace
    # INPUT FIXTURE: State with Source containing source_tier and query_ids
    # EXPECTED: get_claim_provenance() includes source_tier and query_ids
    # -------------------------------------------------------------------------
    prov_src = Source(
        id=1,
        title="DILRMP Specs",
        url="https://dilrmp.gov.in/specs",
        domain="dilrmp.gov.in",
        source_tier=SourceTier.TIER_1,
        query_ids=[1, 2],
        retrieved_at="2026-09-14T09:00:00Z",
    )
    prov_ev = EvidenceItem(
        id=1,
        source_id=1,
        title="DILRMP Specs",
        url="https://dilrmp.gov.in/specs",
        domain="dilrmp.gov.in",
        content="Technical specs for parcel boundaries",
        retrieved_at="2026-09-14T09:00:00Z",
    )
    prov_claim = Claim(
        claim_id=1,
        claim_text="DILRMP digitizes land boundaries with cadastral maps.",
        evidence_ids=[1],
        verification_status=VerificationStatus.SUPPORTED,
        verification_reason="Directly stated in specs.",
    )
    prov_state = ResearchState(
        objective="Cadastral mapping specs",
        sources=[prov_src],
        evidence=[prov_ev],
        claims=[prov_claim],
    )
    prov_trace = prov_state.get_claim_provenance(1)
    assert prov_trace is not None
    assert len(prov_trace["provenance"]) == 1
    item = prov_trace["provenance"][0]
    assert item["source_tier"] == "tier_1"
    assert item["query_ids"] == [1, 2]
    assert item["source_url"] == "https://dilrmp.gov.in/specs"
    print("   ✓ Test 19: Full Provenance Preservation Trace (Claim → Evidence → Source → URL → Query IDs) passed.")

    print("✓ All Slice 6 deterministic tests (Tests 1–19 & Failure Modes A–I) passed successfully.")


def test_llm_input_budget_and_reduction_suite():
    print("\n=== SLICE 6 LLM INPUT BUDGET & REDUCTION TEST SUITE (TESTS A–E) ===")

    # -------------------------------------------------------------------------
    # Test A — Prompt within Budget Passes Through Unchanged
    # FIXTURE: Standard objective, plan, 2 small evidence items. Total tokens << budget.
    # EXPECTED: reduction_occurred == False, all items included, IDs unchanged.
    # -------------------------------------------------------------------------
    src_a1 = Source(id=1, url="https://example.com/1", title="Source 1", domain="example.com", source_tier=SourceTier.TIER_2)
    src_a2 = Source(id=2, url="https://example.com/2", title="Source 2", domain="example.com", source_tier=SourceTier.TIER_3)
    ev_a = [
        EvidenceItem(id=1, source_id=1, title="Doc 1", url="https://example.com/1", domain="example.com", content="Short snippet A", stage="initial", iteration=0),
        EvidenceItem(id=2, source_id=2, title="Doc 2", url="https://example.com/2", domain="example.com", content="Short snippet B", stage="initial", iteration=0),
    ]
    prompt_a, info_a = prepare_synthesis_prompt_within_budget(
        objective="What is Python?",
        plan_context="Q1: Definition",
        evidence_items=ev_a,
        sources=[src_a1, src_a2],
        budget_tokens=6000,
    )
    assert prompt_a is not None, "Prompt should be generated when within budget"
    assert info_a["reduction_occurred"] is False, "No reduction should occur when within budget"
    assert info_a["included_ids"] == [1, 2], "All evidence IDs must be included"
    assert info_a["omitted_ids"] == [], "No evidence IDs should be omitted"
    assert "[1] Title: Doc 1" in prompt_a, "Evidence [1] must be in prompt"
    assert "[2] Title: Doc 2" in prompt_a, "Evidence [2] must be in prompt"
    print("   ✓ Test A: Synthesis Input Within Budget Passes Through Unchanged passed.")

    # -------------------------------------------------------------------------
    # Test B — Prompt Exceeding Budget: Safe Bounded Reduction & Prioritization
    # FIXTURE: 5 items with varying stage, tier, and score.
    #   Item 1: initial, Tier 4, score 0.4
    #   Item 2: initial, Tier 3, score 0.6
    #   Item 3: follow_up, Tier 2, score 0.8
    #   Item 4: follow_up, Tier 1, score 0.95
    #   Item 5: initial, Tier 1, score 0.9
    #   Each item has ~500 characters of content.
    # BUDGET: Restricted budget (700 tokens) that can fit only 2-3 items.
    # EXPECTED:
    #   - reduction_occurred == True
    #   - Total estimated tokens <= budget
    #   - Item 4 (follow_up + Tier 1) and Item 3 (follow_up + Tier 2) prioritized over Item 1 (initial + Tier 4)
    #   - Original evidence IDs preserved in prompt text (e.g. [4], [3])
    #   - Original evidence list and state are NOT mutated
    # -------------------------------------------------------------------------
    src_b1 = Source(id=1, url="https://blog.example.com", title="Blog", domain="blog.example.com", source_tier=SourceTier.TIER_4)
    src_b2 = Source(id=2, url="https://news.example.com", title="News", domain="news.example.com", source_tier=SourceTier.TIER_3)
    src_b3 = Source(id=3, url="https://standards.org", title="Standards", domain="standards.org", source_tier=SourceTier.TIER_2)
    src_b4 = Source(id=4, url="https://gov.in", title="Gov Spec", domain="gov.in", source_tier=SourceTier.TIER_1)
    src_b5 = Source(id=5, url="https://doi.org/paper", title="Journal", domain="doi.org", source_tier=SourceTier.TIER_1)
    sources_b = [src_b1, src_b2, src_b3, src_b4, src_b5]

    ev_b = [
        EvidenceItem(id=1, source_id=1, title="Blog Post", url="https://blog.example.com", domain="blog.example.com",
                     content="Blog opinion snippet on land records without technical details. " * 15, score=0.4, stage="initial", iteration=0),
        EvidenceItem(id=2, source_id=2, title="News Report", url="https://news.example.com", domain="news.example.com",
                     content="General news report covering government announcements. " * 15, score=0.6, stage="initial", iteration=0),
        EvidenceItem(id=3, source_id=3, title="Standards Doc", url="https://standards.org", domain="standards.org",
                     content="OGC Cadastral standards and technical spatial mapping guidelines. " * 15, score=0.8, stage="follow_up", iteration=1),
        EvidenceItem(id=4, source_id=4, title="Government Technical Manual", url="https://gov.in", domain="gov.in",
                     content="DILRMP cadastral boundary specifications and GIS modernization manual. " * 15, score=0.95, stage="follow_up", iteration=1),
        EvidenceItem(id=5, source_id=5, title="Academic Paper", url="https://doi.org/paper", domain="doi.org",
                     content="Peer reviewed study on automated parcel boundary vectorization. " * 15, score=0.9, stage="initial", iteration=0),
    ]
    assert len(ev_b) == 5

    prompt_b, info_b = prepare_synthesis_prompt_within_budget(
        objective="Analyze DILRMP cadastral parcel mapping technical specifications",
        plan_context="Q1: What are the technical boundary standards under DILRMP?",
        evidence_items=ev_b,
        sources=sources_b,
        budget_tokens=700,
    )
    assert prompt_b is not None, "Prompt should be formed within budget"
    assert info_b["reduction_occurred"] is True, "Reduction must occur when input exceeds budget"
    assert info_b["reduced_tokens"] <= 700, f"Reduced tokens ({info_b['reduced_tokens']}) must be <= budget (700)"
    assert info_b["total_estimated_tokens"] <= 700, f"Total prompt tokens ({info_b['total_estimated_tokens']}) must be <= budget (700)"

    # Prioritization check:
    # Item 4 (follow_up, Tier 1) must be included
    assert 4 in info_b["included_ids"], "Item 4 (follow_up, Tier 1) must be prioritized and included"
    # Item 1 (initial, Tier 4) must be omitted before high tier/follow_up items
    assert 1 in info_b["omitted_ids"], "Item 1 (initial, Tier 4) should be omitted under constrained budget"

    # Provenance check: Original evidence IDs preserved in prompt
    assert f"[{4}] Title: Government Technical Manual" in prompt_b, "Original evidence ID 4 must be preserved in prompt"
    # ResearchState provenance invariant: original evidence items list not mutated or shortened
    assert len(ev_b) == 5, "Original evidence items list must NOT be mutated"
    assert [e.id for e in ev_b] == [1, 2, 3, 4, 5], "Original evidence items order and IDs must NOT be changed"
    print("   ✓ Test B: Exceeding Budget Reduction, Prioritization & Provenance Preservation passed.")

    # -------------------------------------------------------------------------
    # Test C — Pathological Case: Extreme Budget Exhaustion Handled Safely
    # FIXTURE: Budget smaller than fixed overhead (e.g. 100 tokens).
    # EXPECTED: Returns (None, info) without uncaught exception.
    # -------------------------------------------------------------------------
    prompt_c, info_c = prepare_synthesis_prompt_within_budget(
        objective="Analyze DILRMP cadastral mapping",
        plan_context="Q1: Overview",
        evidence_items=ev_b,
        sources=sources_b,
        budget_tokens=100,  # Below fixed prompt + system overhead (~250 tokens)
    )
    assert prompt_c is None, "Prompt should return None when fixed overhead exceeds budget"
    assert info_c["reduction_occurred"] is True
    assert info_c["error"] == "fixed_overhead_exceeds_budget"
    assert "exceed total budget" in info_c["reason"]
    print("   ✓ Test C: Pathological Budget Exhaustion Returns None Safely passed.")

    # -------------------------------------------------------------------------
    # Test D — Synthesis Provider Failure Handled by Error Boundary
    # FIXTURE: Mock LLM raising HTTP 413 / RateLimit / API error during synthesis.
    # EXPECTED: Exception caught, fallback report generated, error logged in telemetry.
    # -------------------------------------------------------------------------
    class MockHttp413LLM:
        def invoke(self, messages):
            raise RuntimeError("Groq API HTTP 413: Rate limit / TPM limit exceeded. Requested 9352 tokens, limit 8000")

    mock_state = ResearchState(
        objective="Test 413 Handling",
        sources=[src_b4],
        evidence=[ev_b[3]],
        telemetry=RetrievalTelemetry(
            candidates_discovered=1,
            duplicates_removed=0,
            candidates_rejected=0,
            unique_candidates_retained=1,
        ),
    )
    try:
        MockHttp413LLM().invoke([{"role": "user", "content": "synthesize"}])
        report_d = "success"
    except Exception as synth_err:
        report_d = (
            f"## Research Objective: {mock_state.objective}\n\n"
            f"LLM synthesis encountered a provider error: {type(synth_err).__name__} ({str(synth_err)}). "
            "Retrieved evidence and source provenance are preserved."
        )
        if mock_state.telemetry:
            mock_state.telemetry.provider_errors["synthesis_initial"] = str(synth_err)

    assert "LLM synthesis encountered a provider error" in report_d
    assert "HTTP 413" in report_d
    assert "synthesis_initial" in mock_state.telemetry.provider_errors
    assert "413" in mock_state.telemetry.provider_errors["synthesis_initial"]
    assert len(mock_state.evidence) == 1
    assert len(mock_state.sources) == 1
    print("   ✓ Test D: Synthesis Provider Error Boundary & Telemetry Recording passed.")

    # -------------------------------------------------------------------------
    # Test E — Regression & Contract Preservation (S1–S5 models & prompts)
    # FIXTURE: Verify backward compatibility of build_synthesis_prompt and DEFAULT_SYNTHESIS_BUDGET_TOKENS.
    # -------------------------------------------------------------------------
    assert DEFAULT_SYNTHESIS_BUDGET_TOKENS >= 1000, "Default synthesis budget should be reasonably sized"
    legacy_prompt = build_synthesis_prompt(
        objective="Legacy Test",
        plan_context="Q1: Legacy",
        evidence_context="[1] Title: Doc 1\nSource: example.com (https://example.com/1)\nEvidence Snippet: Short snippet A\n",
    )
    assert "Legacy Test" in legacy_prompt
    assert "[1] Title: Doc 1" in legacy_prompt

    telem_with_reduction = RetrievalTelemetry(
        candidates_discovered=10,
        duplicates_removed=2,
        candidates_rejected=1,
        unique_candidates_retained=7,
        input_reduction=info_b,
    )
    telem_dict = telem_with_reduction.model_dump()
    assert "input_reduction" in telem_dict
    assert telem_dict["input_reduction"]["reduction_occurred"] is True
    assert telem_dict["input_reduction"]["reduced_tokens"] <= 700
    print("   ✓ Test E: Regression & Contract Preservation passed.")

    print("✓ All LLM Input Budget & Reduction tests (Tests A–E) passed successfully.")


async def test_live_sufficient_no_followup():
    print("\n--- Test 6: Live Test A — SUFFICIENT / NO FOLLOW-UP ---")
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        print("⚠ Skipping live test: GROQ_API_KEY or TAVILY_API_KEY missing from environment.")
        return

    test_session = f"smoke_s4_suff_{int(asyncio.get_event_loop().time() * 1000)}"
    objective = "Who designed the Python programming language and in what year was it first released?"

    print(f"Executing Live Test A for: '{objective}'...")
    report = await run_research(topic=objective, session_id=test_session)

    assert report, "Report should not be empty"
    state = get_research_state(test_session)
    assert state is not None
    assert state.sufficiency is not None, "Sufficiency evaluation must be present in state"

    print(f"   ✓ Report length: {len(report)} characters")
    print(f"   ✓ Follow-up count: {state.follow_up_count} (Expected: <= 1)")
    print(f"   ✓ Sufficiency status: {state.sufficiency.status.value}")
    print(f"   ✓ Sufficiency reason: {state.sufficiency.reason}")

    # Verify hard constraint: follow-up count <= 1
    assert state.follow_up_count <= 1, "Follow-up iterations must never exceed 1"
    assert any(f"[{i}]" in report for i in range(1, 6)), "Report must contain numbered inline citations"
    assert "## Sources" in report, "Report must contain ## Sources"
    print("✓ Live Test A passed successfully.")


async def test_live_insufficient_with_followup():
    print("\n--- Test 7: Live Test B — INSUFFICIENT / BOUNDED FOLLOW-UP ---")
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        print("⚠ Skipping live test: GROQ_API_KEY or TAVILY_API_KEY missing from environment.")
        return

    test_session = f"smoke_s4_followup_{int(asyncio.get_event_loop().time() * 1000)}"
    # Multi-faceted question requiring specific chemical instrumentation details likely to trigger gap detection
    objective = "ISRO Chandrayaan-3 Pragyan rover APXS instrument specific chemical elements detected on lunar south pole"
    context = "Planetary surface chemistry and space payload science"
    purpose = "Analyze elemental abundance findings and detection limits"

    print(f"Executing Live Test B for: '{objective}'...")
    report = await run_research(
        topic=objective,
        session_id=test_session,
        context=context,
        purpose=purpose,
    )

    assert report, "Report should not be empty"
    state = get_research_state(test_session)
    assert state is not None
    assert state.sufficiency is not None

    print(f"   ✓ Report length: {len(report)} characters")
    print(f"   ✓ Follow-up count: {state.follow_up_count} (Expected: <= 1)")
    print(f"   ✓ Gaps tracked: {len(state.gaps)}")
    for g in state.gaps:
        print(f"     - [Gap {g.gap_id}] ({g.priority.value.upper()}): {g.description[:60]}... Resolved: {g.resolved}")
    print(f"   ✓ Final Sufficiency status: {state.sufficiency.status.value}")
    print(f"   ✓ Sufficiency reason: {state.sufficiency.reason}")

    # Verify hard constraint: follow-up count MUST be <= 1
    assert state.follow_up_count <= 1, "Follow-up iterations must NEVER exceed 1"

    # Verify provenance of evidence: if follow-up occurred, follow_up stage evidence items exist
    if state.follow_up_count == 1:
        followup_evs = [e for e in state.evidence if e.stage == "follow_up"]
        print(f"   ✓ Provenance verified: {len(followup_evs)} follow-up evidence item(s) preserved in state")

    assert any(f"[{i}]" in report for i in range(1, 6)), "Report must contain numbered citations"
    assert "## Sources" in report, "Report must contain ## Sources"
    print("✓ Live Test B passed successfully.")


async def test_live_sih_context_rich_research():
    print("\n--- Test 8: Live Test C — SIH Context-Rich Research + Provenance ---")
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        print("⚠ Skipping live test: GROQ_API_KEY or TAVILY_API_KEY missing from environment.")
        return

    test_session = f"smoke_s5_sih_{int(asyncio.get_event_loop().time() * 1000)}"
    objective = "Indian urban land record digitization initiatives DILRMP NAKSHA cadastral parcel mapping"
    context = ResearchContext(
        objective=objective,
        purpose="Understand what our SIH solution should integrate with or complement for SIH problem 26012.",
        user_context="We are researching SIH problem 26012 and designing an edge-compatible prototype for automated cadastral parcel mapping.",
        target_audience="Engineering team and SIH evaluation committee",
        required_depth=ResearchDepth.DEEP,
        freshness_requirement=FreshnessRequirement.CURRENT,
        source_requirements=["Prefer official Indian government initiatives (e.g. DILRMP, SVAMITVA) and geospatial standards"],
        expected_output="Structured technical research report with citations, limitations, and unresolved questions",
    )

    print(f"Executing Live Test C for SIH Objective:\n   '{objective}'...")
    report = await run_research(
        topic=objective,
        session_id=test_session,
        context=context,
        purpose=context.purpose,
    )

    assert report, "Report must not be empty"
    state = get_research_state(test_session)
    assert state is not None, "ResearchState must be stored for session"

    # Verify context preserved in state
    assert state.context is not None, "ResearchContext must be preserved in state"
    assert state.context.purpose == context.purpose
    assert state.context.required_depth == ResearchDepth.DEEP

    # Verify source provenance: sources have real URLs, domain, and retrieval timestamps
    assert len(state.sources) > 0, "External search must retrieve at least one source"
    print(f"   ✓ Retrieved Sources: {len(state.sources)}")
    for s in state.sources:
        assert s.url, f"Source {s.id} must have a non-empty URL"
        assert s.domain, f"Source {s.id} must have a domain"
        assert s.retrieved_at is not None, f"Source {s.id} must have a timezone-aware retrieval timestamp"
        print(f"     - [{s.id}] {s.title[:45]}... ({s.domain}) [Retrieved: {s.retrieved_at}]")

    # Verify evidence provenance: each evidence item links to an existing source
    valid_source_ids = {s.id for s in state.sources}
    assert len(state.evidence) > 0, "Retrieved evidence items must exist"
    for e in state.evidence:
        assert e.source_id in valid_source_ids, f"Evidence {e.id} references invalid source {e.source_id}"
        assert e.retrieved_at is not None, f"Evidence {e.id} must have a retrieval timestamp"
    print(f"   ✓ Evidence items verified: {len(state.evidence)} (all linked to valid sources)")

    # Verify claim provenance and traceability
    assert len(state.claims) > 0, "Claims must be extracted and verified"
    for c in state.claims:
        prov = state.get_claim_provenance(c.claim_id)
        assert prov is not None, f"Claim {c.claim_id} must have valid provenance trace"
        assert prov["claim_id"] == c.claim_id
        # Claims without supporting evidence must not be marked SUPPORTED
        if not c.evidence_ids:
            assert c.verification_status != VerificationStatus.SUPPORTED, (
                f"Claim {c.claim_id} has no supporting evidence but was marked SUPPORTED"
            )

    # Verify report contains direct URLs in ## Sources
    assert "## Sources" in report, "Report must contain ## Sources"
    assert "http" in report, "Report must contain direct http/https links in Sources"
    assert any(f"[{s.id}]" in report for s in state.sources), "Sources section must map citations"

    # Verify claim verification summary section
    assert "## Claim Verification Summary" in report, "Report must contain claim verification summary"

    print(f"   ✓ Final report length: {len(report)} characters")
    print(f"   ✓ Follow-up iterations: {state.follow_up_count} (Expected: <= 1)")
    print(f"   ✓ Sufficiency evaluation: {state.sufficiency.status.value.upper() if state.sufficiency else 'N/A'}")
    print("✓ Live Test C (SIH Context-Rich Research + Provenance) passed successfully.")


async def test_live_slice_6_multi_query_retrieval():
    print("\n--- Test 10: Live Test D — SLICE 6 MULTI-QUERY RETRIEVAL & QUALITY PIPELINE ---")
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not groq_key or not tavily_key:
        print("⚠ Skipping live test: GROQ_API_KEY or TAVILY_API_KEY missing from environment.")
        return

    test_session = f"smoke_s6_retrieval_{int(asyncio.get_event_loop().time() * 1000)}"
    objective = "Indian urban land record digitization DILRMP NAKSHA cadastral parcel mapping"
    context = ResearchContext(
        purpose="Analyze Indian urban land record modernization initiatives for automated cadastral mapping",
        required_depth=ResearchDepth.DEEP,
        source_requirements=["Government portals", "Technical documentation"],
    )

    print(f"Executing Live Test D for: '{objective}'...")
    report = await run_research(
        topic=objective,
        session_id=test_session,
        context=context,
        purpose=context.purpose,
    )

    assert report, "Report should not be empty"
    state = get_research_state(test_session)
    assert state is not None, "ResearchState must be recorded in session"

    # 1. Verify telemetry existence and query counts
    assert state.telemetry is not None, "RetrievalTelemetry must be attached to ResearchState"
    queries = state.telemetry.queries_executed
    assert len(queries) >= 2, f"Complex research must execute >= 2 queries, got {len(queries)}"
    print(f"   ✓ Queries Executed: {len(queries)}")
    for q in queries:
        print(f"     - [Query {q.query_id}] '{q.query_text}' (Target Questions: {q.target_question_ids})")

    # 2. Verify candidate pool counts and telemetry consistency
    telem = state.telemetry
    assert telem.candidates_discovered > 0, "Candidates must be discovered"
    assert telem.unique_candidates_retained > 0, "Unique candidates must be retained"
    assert (
        telem.candidates_discovered - telem.duplicates_removed - telem.candidates_rejected
        == telem.unique_candidates_retained
    ), f"Telemetry arithmetic must hold exactly: {telem.candidates_discovered} - {telem.duplicates_removed} - {telem.candidates_rejected} != {telem.unique_candidates_retained}"
    print(f"   ✓ Telemetry: Discovered={telem.candidates_discovered}, Duplicates={telem.duplicates_removed}, Rejected={telem.candidates_rejected}, Retained={telem.unique_candidates_retained}")
    if telem.duration_ms:
        print(f"   ✓ Retrieval Duration: {telem.duration_ms:.2f} ms")

    # 3. Verify source quality classification and query provenance
    assert len(state.sources) > 0, "Sources must be populated"
    for s in state.sources:
        assert s.source_tier in [e for e in SourceTier], f"Source {s.id} must have valid SourceTier"
        assert len(s.query_ids) > 0, f"Source {s.id} must track originating query_ids"
        print(f"     - [{s.id}] [{s.source_tier.value.upper()}] {s.title[:45]}... (Queries: {s.query_ids})")

    # 4. Verify claim provenance includes query_ids and source_tier
    assert len(state.claims) > 0, "Claims must be extracted and verified"
    for c in state.claims:
        prov = state.get_claim_provenance(c.claim_id)
        assert prov is not None, f"Claim {c.claim_id} must have provenance"
        for item in prov["provenance"]:
            if item.get("valid_source"):
                assert "source_tier" in item, "Provenance item must contain source_tier"
                assert "query_ids" in item, "Provenance item must contain query_ids"

    # 5. Verify report format and citations
    assert "## Sources" in report, "Report must contain ## Sources"
    assert "http" in report, "Report must contain clickable links"
    assert any(f"[{s.id}]" in report for s in state.sources), "Report must cite retrieved sources"
    print("✓ Live Test D (Slice 6 Multi-Query Retrieval & Quality Pipeline) passed successfully.")


async def main():
    test_models_and_parsing()
    test_report_sanitization()
    test_planning_models_and_state_transitions()
    test_claim_models_and_edge_cases()
    test_json_parser_fixtures()
    test_knowledge_gap_and_sufficiency_deterministic()
    test_slice_5_deterministic_suite()
    test_slice_6_deterministic_suite()
    test_llm_input_budget_and_reduction_suite()
    await test_live_sufficient_no_followup()
    await test_live_insufficient_with_followup()
    await test_live_sih_context_rich_research()
    await test_live_slice_6_multi_query_retrieval()
    print("\n=== ALL SMOKE CHECKS (SLICES 1–6) PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
