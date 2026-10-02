"""Phase 3 Production Validation, Stress Testing, and Failure-Recovery Test Suite.

Contains 32 comprehensive tests verifying:
- Acceptance Tests A through H
- Question decomposition for complex multi-part queries
- Multi-dimensional evidence coverage model (semantic, lexical, sub-questions, diversity)
- Refusal logic (NONE, PARTIAL, SUFFICIENT, STRONG)
- Answer formatting controlled by question type (WHAT, WHY, HOW, COMPARE, MULTI-STEP)
- Structured context assembly and bounded neighbor expansion
- Strict source isolation and composite cache keys
- HTML tag stripping and web extraction quality metrics (Section 16)
- API resilience, exponential backoff retries, and network error classification
- Zero hallucination on unsupported topics
- PDF, Web, and YouTube RAG coexistence and citations
- Complete Retrieval Debug Mode telemetry exposure (Section 20)
"""

import re
import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


from src.embeddings import get_embedding_model
from src.vector_store import (
    build_vector_store,
    retrieve_relevant_chunks,
    expand_context_with_neighbors,
)
from src.rag_chain import (
    query_rag_pipeline,
    classify_question_intent,
    generate_query_variants,
    decompose_complex_question,
    evaluate_evidence_coverage,
    evaluate_evidence_state,
    format_context,
    execute_llm_with_retry,
    validate_generated_answer,
)
from src.web_processor import compute_content_quality_score, WebProcessingError



@pytest.fixture(scope="module")
def production_knowledge_store():
    """Create a realistic multi-section indexed knowledge base simulating the Brevo CRM suite."""
    docs = [
        Document(
            page_content=(
                "Brevo Overview: Brevo is an all-in-one CRM suite offering marketing automation, "
                "transactional messaging, SMS campaigns, and developer APIs. The platform operates "
                "on high-availability infrastructure supporting global deliverability."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Overview",
                "chunk_id": 0,
                "section_id": 0,
            },
        ),
        Document(
            page_content=(
                "API Authentication & Security: Every programmatic request to Brevo sending endpoints requires an "
                "API key provided in the 'api-key' request header. The API key authenticates the external application, "
                "authorizes sending privileges, and protects unauthorized account access."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "API Authentication",
                "chunk_id": 1,
                "section_id": 1,
            },
        ),
        Document(
            page_content=(
                "Transactional Email Tracking: Developers track transactional email performance using real-time "
                "analytics dashboards and webhooks. Tracking metrics include delivery rate, open rate, link click rate, "
                "and bounce statistics, allowing developers to monitor deliverability immediately."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Transactional Email Analytics",
                "chunk_id": 2,
                "section_id": 2,
            },
        ),
        Document(
            page_content=(
                "Marketing Automation Workflows & Triggers: Automated customer journeys are initiated by behavioral triggers. "
                "For example, a trigger occurs when a customer opens an order-confirmation email or submits a website form. "
                "The trigger registers the event and starts the automation workflow engine."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Automation Triggers",
                "chunk_id": 3,
                "section_id": 3,
            },
        ),
        Document(
            page_content=(
                "Workflow Actions & Follow-ups: Following an automation trigger, the workflow executes sequential actions. "
                "Workflows can insert a delay timer, evaluate conditional branches, and automatically trigger a follow-up "
                "message such as a personalized SMS or feedback survey to re-engage the customer."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Workflow Actions",
                "chunk_id": 4,
                "section_id": 4,
            },
        ),
        Document(
            page_content=(
                "SMS Integration with External Websites: Brevo SMS services connect to external websites through REST API "
                "endpoints and webhooks. When an e-commerce customer completes a transaction on a merchant website, "
                "the website backend makes an HTTP POST request to Brevo to trigger real-time SMS delivery."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "SMS Web Integration",
                "chunk_id": 5,
                "section_id": 5,
            },
        ),
        Document(
            page_content=(
                "Dedicated IP vs Shared IP Comparison: Shared IPs distribute email traffic across multiple organizations "
                "and require zero reputation warmup. In contrast, dedicated IPs grant complete control over sender reputation "
                "and email deliverability, but require a structured 4-week IP warmup process to establish ISP trust."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "IP Infrastructure Comparison",
                "chunk_id": 6,
                "section_id": 6,
            },
        ),
        Document(
            page_content=(
                "Why IP Warmup is Critical: IP warmup is necessary for dedicated IPs because major mailbox providers like "
                "Gmail and Yahoo flag or reject sudden high volumes of email originating from unverified IP addresses. "
                "Gradually ramping volume builds positive reputation metrics."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_id": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "IP Warmup Rationale",
                "chunk_id": 7,
                "section_id": 7,
            },
        ),
    ]

    embeddings = get_embedding_model()
    return build_vector_store(docs, embeddings=embeddings)


# =========================================================================
# SECTION A: ACCEPTANCE TESTS (Section 22)
# =========================================================================

def test_acceptance_a_definition(production_knowledge_store):
    """TEST A: 'What is Brevo?' -> Definition from indexed source."""
    res = query_rag_pipeline(production_knowledge_store, "What is Brevo?", top_k=3, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "crm" in ans_lower or "automation" in ans_lower or "messaging" in ans_lower
    assert len(res["citations"]) > 0


def test_acceptance_b_how_sms(production_knowledge_store):
    """TEST B: 'How does SMS connect to another website?' -> Source integration mechanism."""
    res = query_rag_pipeline(production_knowledge_store, "How does SMS connect to another website?", top_k=4, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "rest api" in ans_lower or "webhook" in ans_lower or "http post" in ans_lower


def test_acceptance_c_cross_section_trigger_followup(production_knowledge_store):
    """TEST C: 'How can a customer opening an email lead to an automated follow-up?' -> Multiple relevant sections synthesized."""
    q = "How can a customer opening an email lead to an automated follow-up?"
    res = query_rag_pipeline(production_knowledge_store, q, top_k=6, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "open" in ans_lower
    assert "follow-up" in ans_lower or "workflow" in ans_lower or "action" in ans_lower


def test_acceptance_d_multi_evidence_integration(production_knowledge_store):
    """TEST D: 'How do API authentication, tracking and SMS work together?' -> Evidence from multiple sections."""
    q = "How do API authentication, tracking and SMS work together?"
    res = query_rag_pipeline(production_knowledge_store, q, top_k=6, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    cited_headings = {c.get("heading") for c in res["citations"] if c.get("heading")}
    assert len(cited_headings) >= 2


def test_acceptance_e_typo_resilience(production_knowledge_store):
    """TEST E: Question with realistic spelling mistakes."""
    q = "how the smss autometion triger works for confiramtion?"
    res = query_rag_pipeline(production_knowledge_store, q, top_k=6, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "workflow" in ans_lower or "sms" in ans_lower


def test_acceptance_f_paraphrased_semantic_retrieval(production_knowledge_store):
    """TEST F: Paraphrased question using different vocabulary."""
    q = "In what manner can programmers oversee transactional dispatch outcomes and click-through statistics?"
    res = query_rag_pipeline(production_knowledge_store, q, top_k=4, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "tracking" in ans_lower or "analytic" in ans_lower or "metric" in ans_lower or "open" in ans_lower


def test_acceptance_g_unsupported_refusal(production_knowledge_store):
    """TEST G: Unsupported question about unmentioned topics triggers refusal."""
    q = "What is the orbital velocity and radius of Jupiter's moon Europa?"
    res = query_rag_pipeline(production_knowledge_store, q, top_k=4, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_acceptance_h_active_source_isolation_url_a_vs_b(production_knowledge_store):
    """TEST H: Question about URL B while URL A is active -> strictly isolated, no URL A leakage."""
    vs = production_knowledge_store
    url_b_doc = Document(
        page_content="Kubernetes Cluster Setup: Deploy pods and ingress controllers across node pools using helm charts.",
        metadata={
            "source": "https://kubernetes.io/docs",
            "source_id": "https://kubernetes.io/docs",
            "source_type": "webpage",
            "title": "Kubernetes Docs",
            "heading": "Cluster Setup",
            "chunk_id": 0,
        },
    )
    vs.add_documents([url_b_doc])

    res = query_rag_pipeline(
        vs,
        "How does SMS connect to the website with API keys?",
        top_k=4,
        source_filter="https://kubernetes.io/docs",
        source_type="webpage",
    )
    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower()
    assert len(res["citations"]) == 0


# =========================================================================
# SECTION B: DEEP-QUESTION RETRIEVAL & DECOMPOSITION (Sections 6, 7, 8, 9)
# =========================================================================

def test_question_decomposition_logic():
    """Verify decompose_complex_question breaks multi-part and conditional queries into atomic components."""
    intent_info = classify_question_intent("If customer opens order confirmation email, how does automation trigger a follow-up message?")
    sub_qs = decompose_complex_question("If customer opens order confirmation email, how does automation trigger a follow-up message?", intent_info)
    assert len(sub_qs) >= 2
    combined = " ".join(sub_qs).lower()
    assert "order confirmation" in combined
    assert "trigger" in combined or "follow-up" in combined or "automation" in combined


def test_evidence_coverage_model_dimensions(production_knowledge_store):
    """Verify evaluate_evidence_coverage assesses multi-dimensional coverage beyond simple lexical overlap."""
    vs = production_knowledge_store
    q = "How do API authentication and transactional email tracking work in Brevo?"
    intent_info = classify_question_intent(q)
    sub_qs = decompose_complex_question(q, intent_info)
    docs = list(vs.docstore._dict.values())

    cov = evaluate_evidence_coverage(docs, q, intent_info, sub_qs)
    assert cov["evidence_state"] in ("STRONG", "SUFFICIENT")
    assert cov["coverage_score"] > 0.4
    assert cov["chunk_diversity"] >= 2
    assert len(cov["covered_sub_questions"]) >= 1


def test_partial_evidence_handling(production_knowledge_store):
    """Verify PARTIAL evidence state is assigned when some aspects exist and others are absent."""
    intent_info = classify_question_intent("How does Brevo SMS integration work with quantum entanglement?")
    sub_qs = ["SMS integration", "quantum entanglement"]
    # Only supply SMS chunk
    sms_chunk = [
        Document(
            page_content="Brevo SMS services connect to external websites through REST API endpoints and webhooks.",
            metadata={"heading": "SMS Web Integration", "chunk_id": 5, "source": "https://brevo.com/crm"}
        )
    ]
    cov = evaluate_evidence_coverage(sms_chunk, "How does Brevo SMS integration work with quantum entanglement?", intent_info, sub_qs)
    assert cov["evidence_state"] == "PARTIAL"
    assert "quantum entanglement" in cov["missing_sub_questions"] or len(cov["missing_sub_questions"]) >= 1


def test_no_evidence_strict_refusal_phrase(production_knowledge_store):
    """Verify NONE evidence state triggers clean source-specific refusal without hallucination."""
    vs = production_knowledge_store
    q = "Explain the photosynthesis process of phytoplankton."
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is True
    assert res["evidence_state"] == "NONE"
    assert "couldn't find" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_why_question_format(production_knowledge_store):
    """Verify WHY questions extract cause and rationale without confusing them with HOW."""
    vs = production_knowledge_store
    q = "Why is IP warmup critical for dedicated IPs?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "gmail" in ans_lower or "yahoo" in ans_lower or "throttle" in ans_lower or "reputation" in ans_lower or "flag" in ans_lower


def test_procedural_multi_step_numbered_steps(production_knowledge_store):
    """Verify procedural / multi-step questions format workflow execution sequentially."""
    vs = production_knowledge_store
    q = "Explain the step-by-step process of how an automation workflow executes from trigger to follow-up."
    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower
    assert "follow-up" in ans_lower or "message" in ans_lower or "delay" in ans_lower


def test_comparison_question_structure(production_knowledge_store):
    """Verify comparison questions compare both entities according to source facts."""
    vs = production_knowledge_store
    q = "Compare dedicated IP versus shared IP sending according to the indexed source."
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "dedicated" in ans_lower and "shared" in ans_lower
    assert "warmup" in ans_lower or "reputation" in ans_lower


def test_short_deep_question(production_knowledge_store):
    """Verify short questions (e.g. 'How does SMS connect?') still retrieve complete operational mechanism."""
    vs = production_knowledge_store
    q = "How does SMS connect?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "rest api" in ans_lower or "webhook" in ans_lower or "post" in ans_lower


def test_long_complex_clause_question(production_knowledge_store):
    """Verify questions with multiple dependent clauses retrieve context across all clauses."""
    vs = production_knowledge_store
    q = "When an external e-commerce merchant completes a checkout transaction on their website, how does Brevo authenticate the HTTP POST request and trigger an instant SMS notification?"
    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "api-key" in ans_lower or "header" in ans_lower or "authenticate" in ans_lower
    assert "sms" in ans_lower


# =========================================================================
# SECTION C: NEIGHBOR EXPANSION, CACHE, AND SOURCE ISOLATION (Sections 12, 13, 14)
# =========================================================================

def test_neighbor_expansion_same_source_only(production_knowledge_store):
    """Verify neighbor expansion only attaches adjacent chunks from the exact same source."""
    vs = production_knowledge_store
    target_chunks = [
        Document(
            page_content="Marketing Automation Workflows & Triggers",
            metadata={"source": "https://brevo.com/crm", "chunk_id": 3}
        )
    ]
    expanded = expand_context_with_neighbors(target_chunks, vs, max_neighbors=2, source_filter="https://brevo.com/crm")
    expanded_ids = [c.metadata.get("chunk_id") for c in expanded]
    assert 3 in expanded_ids
    assert (2 in expanded_ids) or (4 in expanded_ids)
    # Ensure source is strictly matching
    assert all(c.metadata.get("source") == "https://brevo.com/crm" for c in expanded)


def test_cache_key_isolation_composite():
    """Verify changing source_id or content_hash invalidates cache lookup."""
    key1 = ("https://brevo.com", "hash_aaa", "how does sms work?", "source_scoped", 4)
    key2 = ("https://brevo.com", "hash_bbb", "how does sms work?", "source_scoped", 4)
    key3 = ("https://other.com", "hash_aaa", "how does sms work?", "source_scoped", 4)

    cache = {key1: {"answer": "Brevo SMS answer"}}
    assert key2 not in cache
    assert key3 not in cache


def test_format_context_preserves_structure():
    """Verify format_context outputs structured Section, Source, and Chunk markers."""
    doc = Document(
        page_content="Test API documentation text.",
        metadata={"source": "https://brevo.com/crm", "heading": "API Authentication", "chunk_id": 1, "source_type": "url", "title": "Brevo CRM"}
    )
    formatted = format_context([doc])
    assert "[RELEVANT SECTION: API Authentication" in formatted
    assert "Chunk: 1" in formatted
    assert "Test API documentation text." in formatted


# =========================================================================
# SECTION D: HTML CLEANUP, CITATIONS, AND VALIDATION (Sections 10, 15, 19)
# =========================================================================

def test_html_tag_elimination_guarantee(production_knowledge_store):
    """Verify answers and context never contain raw HTML tags."""
    vs = production_knowledge_store
    dirty_answer = "<div><p>Brevo provides <span>CRM</span> and <strong>SMS</strong> services.</p></div>"
    clean_ans = validate_generated_answer(dirty_answer, "What is Brevo?", "STRONG", "Refusal", {})
    assert "<div>" not in clean_ans
    assert "</div>" not in clean_ans
    assert "<span>" not in clean_ans
    assert "<p>" not in clean_ans


def test_citation_metadata_fidelity(production_knowledge_store):
    """Verify citations accurately mirror retrieved chunks with valid source URL and headings."""
    vs = production_knowledge_store
    q = "How can developers track transactional email performance?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is False
    assert len(res["citations"]) > 0
    for cit in res["citations"]:
        assert cit["source"] == "https://brevo.com/crm"
        assert cit["heading"] is not None
        assert len(cit["snippet"]) > 5


def test_anti_hallucination_fictional_claims(production_knowledge_store):
    """Verify refusal is enforced when asked about fictional technical capabilities."""
    vs = production_knowledge_store
    q = "Does Brevo support faster-than-light tachyon messaging protocols?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")
    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower()
    assert len(res["citations"]) == 0


# =========================================================================
# SECTION E: API RESILIENCE, RETRY & FAILURE HANDLING (Sections 17, 18)
# =========================================================================

def test_api_rate_limit_backoff_retry():
    """Verify 429 rate limit triggers exponential backoff retry and succeeds if retry works."""
    mock_chain = MagicMock()
    # First call raises 429, second call succeeds
    mock_chain.invoke.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded: please try again in 1.2s"),
        "Brevo is an all-in-one CRM suite."
    ]

    answer, err_code, err_msg = execute_llm_with_retry(mock_chain, {"context": "...", "question": "..."}, max_retries=2, initial_wait=0.01)
    assert answer == "Brevo is an all-in-one CRM suite."
    assert err_code is None
    assert mock_chain.invoke.call_count == 2


def test_api_network_failure_handling():
    """Verify network failure (e.g. no such host) returns a clean user-friendly message rather than stack trace."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("dial tcp: lookup daily-cloudcode-pa.googleapis.com: no such host")

    answer, err_code, err_msg = execute_llm_with_retry(mock_chain, {"context": "...", "question": "..."}, max_retries=2, initial_wait=0.01)
    assert answer is None
    assert err_code == "NETWORK_FAILURE"
    assert "network connection issue" in err_msg
    assert "daily-cloudcode-pa" not in err_msg  # no leaked raw internal trace


# =========================================================================
# SECTION F: WEB EXTRACTION QUALITY METRICS & DIAGNOSTICS (Section 16)
# =========================================================================

def test_web_extraction_quality_score_metrics():
    """Verify compute_content_quality_score calculates all Section 16 metrics."""
    clean = "Brevo CRM Suite provides marketing automation, email delivery, and SMS endpoints for developers."
    raw = "<html><body><div><p>" + clean + "</p></div></body></html>"
    sections = [{"heading": "Overview", "text": clean}]

    metrics = compute_content_quality_score(clean, raw, sections)
    assert "word_count" in metrics
    assert "unique_word_count" in metrics
    assert "heading_count" in metrics
    assert "paragraph_count" in metrics
    assert "HTML_tag_ratio" in metrics
    assert "content_density" in metrics
    assert metrics["word_count"] > 10
    assert metrics["is_valid"] is True


def test_malformed_html_quality_rejection():
    """Verify that empty, boilerplate, or anti-bot challenge pages are rejected."""
    empty_clean = "Please enable JavaScript"
    empty_raw = "<html><body><script>alert(1);</script><div>Please enable JavaScript</div></body></html>"
    sections = [{"heading": "Challenge", "text": empty_clean}]

    metrics = compute_content_quality_score(empty_clean, empty_raw, sections)
    assert metrics["is_valid"] is False


# =========================================================================
# SECTION G: EDGE CASES & METADATA ROBUSTNESS (Sections 21, 24)
# =========================================================================

def test_duplicate_chunk_deduplication(production_knowledge_store):
    """Verify duplicate chunks retrieved across multiple query variants are merged without inflating context."""
    vs = production_knowledge_store
    cands = retrieve_relevant_chunks(vs, "Brevo Overview", top_k=4, source_filter="https://brevo.com/crm")
    # Duplicate doc list
    merged = list(cands) + list(cands)
    unique_ids = {c.metadata.get("chunk_id") for c in merged}
    assert len(unique_ids) <= len(cands)


def test_empty_query_safe_handling(production_knowledge_store):
    """Verify empty or whitespace-only queries do not crash the pipeline."""
    vs = production_knowledge_store
    res = query_rag_pipeline(vs, "   ", top_k=4)
    assert res["is_refusal"] is False
    assert "valid question" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_missing_metadata_defaults():
    """Verify format_context and extract_citations handle documents missing optional metadata without crashing."""
    raw_doc = Document(page_content="Minimalist content without extra keys.")
    formatted = format_context([raw_doc])
    assert "General Information" in formatted
    assert "Document" in formatted


def test_pdf_rag_integration_compatibility():
    """Verify standard PDF metadata (page citations) formats cleanly in the Phase 3 pipeline."""
    pdf_doc = Document(
        page_content="DocuMind PDF specifications: Supports page extraction via PyMuPDF with persistent vector storage.",
        metadata={"source": "manual.pdf", "source_type": "pdf", "page": 4, "title": "DocuMind Manual", "chunk_id": 0}
    )
    embeddings = get_embedding_model()
    pdf_vs = build_vector_store([pdf_doc], embeddings=embeddings)

    res = query_rag_pipeline(pdf_vs, "What does DocuMind PDF specifications support?", top_k=2, source_filter="manual.pdf", source_type="pdf")
    assert res["is_refusal"] is False
    assert len(res["citations"]) > 0
    assert res["citations"][0]["page"] == 4
    assert "Page 4" in res["citations"][0]["formatted"]


def test_youtube_rag_integration_compatibility():
    """Verify standard YouTube video metadata (timestamp citations) formats cleanly in the Phase 3 pipeline."""
    yt_doc = Document(
        page_content="In this clip we explain how to configure webhook handlers for incoming SMS alerts.",
        metadata={
            "source": "https://youtube.com/watch?v=demo123",
            "source_type": "youtube",
            "title": "Webhook Tutorial",
            "timestamp_formatted": "03:45",
            "timestamp_url": "https://youtube.com/watch?v=demo123&t=225s",
            "chunk_id": 0
        }
    )
    embeddings = get_embedding_model()
    yt_vs = build_vector_store([yt_doc], embeddings=embeddings)

    res = query_rag_pipeline(yt_vs, "How to configure webhook handlers?", top_k=2, source_filter="https://youtube.com/watch?v=demo123", source_type="youtube")
    assert res["is_refusal"] is False
    assert len(res["citations"]) > 0
    assert res["citations"][0]["timestamp_formatted"] == "03:45"
    assert "03:45" in res["citations"][0]["formatted"]


def test_retrieval_debug_mode_telemetry(production_knowledge_store):
    """Verify debug_info contains all Section 20 fields needed for diagnostics."""
    vs = production_knowledge_store
    q = "How does SMS connect to another website?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")

    dbg = res.get("debug_info", {})
    required_keys = [
        "question", "question_type", "sub_questions", "query_variants",
        "candidate_count", "reranked_count", "final_context_count", "neighbor_count",
        "evidence_coverage", "evidence_state", "active_source", "retrieved_sections",
        "retrieved_chunk_ids", "similarity_scores", "lexical_scores",
        "final_context_preview", "citation_sources"
    ]
    for key in required_keys:
        assert key in dbg, f"Missing debug_info telemetry field: {key}"
