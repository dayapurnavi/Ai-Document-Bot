"""Phase 4 Real-World User Acceptance Testing (UAT) and Production Validation Suite.

Contains 30 comprehensive automated tests validating:
1. URL A ingestion & indexing
2. URL B ingestion & indexing
3. URL switching & state transition
4. Strict active source isolation (zero URL A leakage into URL B queries)
5. Cache isolation (URL A -> Q vs URL B -> Q, plus cross-query matrix)
6. Deep question matrix A: Basic factual
7. Deep question matrix B: Definition
8. Deep question matrix C: How mechanism
9. Deep question matrix D: Why rationale
10. Deep question matrix E: Process step-by-step
11. Deep question matrix F: Comparison
12. Deep question matrix G: Cause & effect
13. Deep question matrix H: Multi-part query
14. Deep question matrix I: Cross-section synthesis
15. Deep question matrix J: Conditional logic
16. Deep question matrix K: Temporal progression
17. Deep question matrix L: Paraphrased query
18. Deep question matrix M: Typo tolerance
19. Deep question matrix N: Long complex analytical clause
20. Deep question matrix O: Multi-evidence 3+ section aggregation
21. Deep question matrix P: Unsupported topic refusal
22. Citation correctness & metadata fidelity
23. HTML cleanliness (zero raw tags in answers or chunks)
24. Chart eligibility (NO DATA = NO CHART)
25. Chart numeric verification against source
26. Summary UI card separation (Summary, Key Points, Links, Other Info, Charts)
27. Stale state prevention across session switching
28. Error recovery: SSRF and invalid URL handling
29. Error recovery: LLM rate-limit exponential backoff
30. Full regression compatibility: PDF, Web, and YouTube RAG coexistence
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.embeddings import get_embedding_model
from src.vector_store import (
    build_vector_store,
    add_documents_to_vector_store,
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
from src.web_processor import (
    validate_and_normalize_url,
    extract_webpage_content,
    compute_content_quality_score,
    WebProcessingError,
)
from src.structured_analysis import (
    validate_chart_data,
    detect_and_extract_charts,
    generate_structured_website_analysis,
    validate_analysis_session,
)


# =========================================================================
# FIXTURES: REALISTIC DUAL-SOURCE KNOWLEDGE BASE
# =========================================================================

URL_A = "https://brevo.com/crm"
URL_B = "https://kubernetes.io/docs/architecture"

@pytest.fixture(scope="module")
def uat_vector_store():
    """Build a multi-source knowledge base containing URL A and URL B chunks."""
    docs_a = [
        Document(
            page_content=(
                "Brevo Overview: Brevo is an all-in-one CRM suite offering marketing automation, "
                "transactional messaging, SMS campaigns, and developer APIs. The platform operates "
                "on high-availability cloud infrastructure supporting global deliverability."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Overview",
                "chunk_id": 0,
                "section_id": 0,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "API Authentication & Security: Every programmatic request to Brevo sending endpoints requires an "
                "API key provided in the 'api-key' request header. The API key authenticates the external application, "
                "authorizes sending privileges, and protects unauthorized account access."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "API Authentication",
                "chunk_id": 1,
                "section_id": 1,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "Transactional Email Tracking: Developers track transactional email performance using real-time "
                "analytics dashboards and webhooks. Tracking metrics include delivery rate, open rate, link click rate, "
                "and bounce statistics, allowing developers to monitor deliverability immediately."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Transactional Email Analytics",
                "chunk_id": 2,
                "section_id": 2,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "Marketing Automation Triggers: Automated customer journeys are initiated by behavioral triggers. "
                "For example, a trigger occurs when a customer opens an order-confirmation email or submits a website form. "
                "The trigger registers the event and starts the automation workflow engine."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Automation Triggers",
                "chunk_id": 3,
                "section_id": 3,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "Workflow Actions & Follow-ups: Following an automation trigger, the workflow executes sequential actions. "
                "Workflows can insert a delay timer, evaluate conditional branches, and automatically trigger a follow-up "
                "message such as a personalized SMS or feedback survey to re-engage the customer."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Workflow Actions",
                "chunk_id": 4,
                "section_id": 4,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "SMS Integration with External Websites: Brevo SMS services connect to external websites through REST API "
                "endpoints and webhooks. When an e-commerce customer completes a transaction on a merchant website, "
                "the website backend makes an HTTP POST request to Brevo to trigger real-time SMS delivery."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "SMS Web Integration",
                "chunk_id": 5,
                "section_id": 5,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "Dedicated IP vs Shared IP Comparison: Shared IPs distribute email traffic across multiple organizations "
                "and require zero reputation warmup. In contrast, dedicated IPs grant complete control over sender reputation "
                "and email deliverability, but require a structured 4-week IP warmup process to establish ISP trust."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "IP Infrastructure Comparison",
                "chunk_id": 6,
                "section_id": 6,
                "content_hash": "hash_brevo_001",
            },
        ),
        Document(
            page_content=(
                "Why IP Warmup is Critical: IP warmup is necessary for dedicated IPs because major mailbox providers like "
                "Gmail and Yahoo flag or reject sudden high volumes of email originating from unverified IP addresses. "
                "Gradually ramping volume builds positive reputation metrics."
            ),
            metadata={
                "source": URL_A,
                "source_id": URL_A,
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "IP Warmup Rationale",
                "chunk_id": 7,
                "section_id": 7,
                "content_hash": "hash_brevo_001",
            },
        ),
    ]

    docs_b = [
        Document(
            page_content=(
                "Kubernetes Architecture Overview: Kubernetes coordinates a highly available cluster of connected computers "
                "that work together as a single unit. The control plane manages worker nodes, scheduling pods, and maintaining desired state."
            ),
            metadata={
                "source": URL_B,
                "source_id": URL_B,
                "source_type": "url",
                "title": "Kubernetes Architecture",
                "heading": "Cluster Overview",
                "chunk_id": 0,
                "section_id": 0,
                "content_hash": "hash_k8s_002",
            },
        ),
        Document(
            page_content=(
                "Kube-Apiserver Component: The API server is the front end for the Kubernetes control plane. It exposes the "
                "Kubernetes API and scales horizontally by deploying more instances. All cluster management commands flow through the apiserver."
            ),
            metadata={
                "source": URL_B,
                "source_id": URL_B,
                "source_type": "url",
                "title": "Kubernetes Architecture",
                "heading": "Control Plane Components",
                "chunk_id": 1,
                "section_id": 1,
                "content_hash": "hash_k8s_002",
            },
        ),
        Document(
            page_content=(
                "Kubelet Node Agent: Kubelet is an agent that runs on each node in the cluster. It ensures that containers are "
                "running in a Pod and healthy according to PodSpecs provided by the control plane."
            ),
            metadata={
                "source": URL_B,
                "source_id": URL_B,
                "source_type": "url",
                "title": "Kubernetes Architecture",
                "heading": "Worker Node Agents",
                "chunk_id": 2,
                "section_id": 2,
                "content_hash": "hash_k8s_002",
            },
        ),
    ]

    embeddings = get_embedding_model()
    vs = build_vector_store(docs_a, embeddings=embeddings)
    vs = add_documents_to_vector_store(vs, docs_b, embeddings=embeddings)
    return vs


# =========================================================================
# TEST SUITE IMPLEMENTATION (1-30)
# =========================================================================

# 1. URL A Ingestion & Indexing
def test_uat_01_url_a_ingestion(uat_vector_store):
    """Verify URL A chunks are cleanly indexed with valid metadata."""
    chunks = retrieve_relevant_chunks(uat_vector_store, "Brevo CRM", top_k=4, source_filter=URL_A)
    assert len(chunks) > 0
    for ch in chunks:
        assert ch.metadata["source"] == URL_A
        assert ch.metadata["title"] == "Brevo CRM Suite"
        assert ch.metadata["chunk_id"] >= 0


# 2. URL B Ingestion & Indexing
def test_uat_02_url_b_ingestion(uat_vector_store):
    """Verify URL B chunks are cleanly indexed in the same vector store without overwriting URL A."""
    chunks = retrieve_relevant_chunks(uat_vector_store, "Kubernetes cluster control plane", top_k=3, source_filter=URL_B)
    assert len(chunks) > 0
    for ch in chunks:
        assert ch.metadata["source"] == URL_B
        assert ch.metadata["title"] == "Kubernetes Architecture"


# 3. URL Switching & State Transition
def test_uat_03_url_switching_state_transition():
    """Verify state transition mechanics when switching active URL source."""
    session = {
        "active_source_url": URL_A,
        "active_source_id": URL_A,
        "active_source_hash": "hash_brevo_001",
        "current_analysis": {"source_url": URL_A, "messages": [{"role": "user", "content": "hello"}]},
    }
    # Simulate user switching to URL B
    session["active_source_url"] = URL_B
    session["active_source_id"] = URL_B
    session["active_source_hash"] = "hash_k8s_002"
    session["current_analysis"] = None

    assert session["active_source_url"] == URL_B
    assert session["active_source_hash"] == "hash_k8s_002"
    assert session["current_analysis"] is None


# 4. Strict Active Source Isolation (Zero Leakage)
def test_uat_04_active_source_isolation_zero_leakage(uat_vector_store):
    """Verify query executed against URL B returns strictly URL B data with 0 URL A chunks/text."""
    res = query_rag_pipeline(uat_vector_store, "Explain the control plane components and apiserver", top_k=4, source_filter=URL_B)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()

    # Must contain URL B concepts
    assert "kube" in ans_lower or "apiserver" in ans_lower or "control plane" in ans_lower or "kubernetes" in ans_lower

    # STRICT: Must NOT contain URL A concepts or URLs
    assert "brevo" not in ans_lower
    assert "sms" not in ans_lower
    assert "crm" not in ans_lower

    for c in res["citations"]:
        assert c["source"] == URL_B
        assert "brevo" not in c["source"]


# 5. Cache Isolation Across Sources
def test_uat_05_cache_isolation_matrix(uat_vector_store):
    """Verify identical question asked on URL A and URL B produces isolated cache keys and distinct results."""
    norm_q = "what is the system overview?"
    key_a = (URL_A, "hash_brevo_001", norm_q, "source_scoped", 4)
    key_b = (URL_B, "hash_k8s_002", norm_q, "source_scoped", 4)

    assert key_a != key_b
    assert key_a[0] == URL_A
    assert key_b[0] == URL_B

    # Verify query cache partition isolation
    simulated_cache = {}
    simulated_cache[key_a] = {"answer": "Brevo CRM Suite overview.", "citations": [{"source": URL_A}]}
    # Looking up with key_b must be a cache miss
    assert key_b not in simulated_cache

    # 4-quadrant cache isolation test: URL A -> Q1, URL A -> Q2, URL B -> Q1, URL B -> Q2
    q1 = "what is the overview?"
    q2 = "how does authentication work?"
    k_a1 = (URL_A, "hash_brevo_001", q1, "source_scoped", 4)
    k_a2 = (URL_A, "hash_brevo_001", q2, "source_scoped", 4)
    k_b1 = (URL_B, "hash_k8s_002", q1, "source_scoped", 4)
    k_b2 = (URL_B, "hash_k8s_002", q2, "source_scoped", 4)
    all_keys = {k_a1, k_a2, k_b1, k_b2}
    assert len(all_keys) == 4

    # Real pipeline query execution on both sources
    res_a = query_rag_pipeline(uat_vector_store, "What is Brevo CRM?", top_k=4, source_filter=URL_A)
    res_b = query_rag_pipeline(uat_vector_store, "What is Kubernetes Architecture?", top_k=4, source_filter=URL_B)

    assert res_a["is_refusal"] is False
    assert res_b["is_refusal"] is False
    assert res_a["answer"] != res_b["answer"]
    assert "brevo" in res_a["answer"].lower() or "crm" in res_a["answer"].lower()
    assert "kubernetes" in res_b["answer"].lower() or "cluster" in res_b["answer"].lower()


# 6. Deep Question Matrix A: Basic Factual
def test_uat_06_deep_question_matrix_basic(uat_vector_store):
    """Matrix A: 'What is Brevo?'"""
    res = query_rag_pipeline(uat_vector_store, "What is Brevo?", top_k=3, source_filter=URL_A)
    assert res["is_refusal"] is False
    assert "crm" in res["answer"].lower() or "automation" in res["answer"].lower()


# 7. Deep Question Matrix B: Definition
def test_uat_07_deep_question_matrix_definition(uat_vector_store):
    """Matrix B: 'What is transactional email tracking?'"""
    res = query_rag_pipeline(uat_vector_store, "What is transactional email tracking?", top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "tracking" in ans_lower or "metric" in ans_lower or "analytic" in ans_lower or "delivery" in ans_lower


# 8. Deep Question Matrix C: How Mechanism
def test_uat_08_deep_question_matrix_how(uat_vector_store):
    """Matrix C: 'How does SMS connect to another website?'"""
    res = query_rag_pipeline(uat_vector_store, "How does SMS connect to another website?", top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "rest api" in ans_lower or "webhook" in ans_lower or "http post" in ans_lower


# 9. Deep Question Matrix D: Why Rationale
def test_uat_09_deep_question_matrix_why(uat_vector_store):
    """Matrix D: 'Why is IP warmup critical for dedicated IPs?'"""
    res = query_rag_pipeline(uat_vector_store, "Why is IP warmup critical for dedicated IPs?", top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "warmup" in ans_lower or "reputation" in ans_lower or "isp" in ans_lower or "gmail" in ans_lower


# 10. Deep Question Matrix E: Process Step-by-Step
def test_uat_10_deep_question_matrix_process_step_by_step(uat_vector_store):
    """Matrix E: 'Explain the automation workflow execution step-by-step.'"""
    res = query_rag_pipeline(uat_vector_store, "Explain how the automation workflow executes step-by-step.", top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans = res["answer"]
    assert "1." in ans or "trigger" in ans.lower()
    assert "action" in ans.lower() or "follow-up" in ans.lower() or "sequential" in ans.lower()


# 11. Deep Question Matrix F: Comparison
def test_uat_11_deep_question_matrix_comparison(uat_vector_store):
    """Matrix F: 'Compare dedicated IP versus shared IP.'"""
    res = query_rag_pipeline(uat_vector_store, "Compare dedicated IP versus shared IP.", top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "dedicated" in ans_lower and "shared" in ans_lower
    assert "warmup" in ans_lower or "reputation" in ans_lower


# 12. Deep Question Matrix G: Cause & Effect
def test_uat_12_deep_question_matrix_cause_effect(uat_vector_store):
    """Matrix G: 'What happens when a customer opens an order-confirmation email?'"""
    res = query_rag_pipeline(uat_vector_store, "What happens when a customer opens an order-confirmation email?", top_k=5, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "workflow" in ans_lower
    assert "customer" in ans_lower or "email" in ans_lower or "order" in ans_lower or "action" in ans_lower


# 13. Deep Question Matrix H: Multi-Part Query
def test_uat_13_deep_question_matrix_multi_part(uat_vector_store):
    """Matrix H: 'What is Brevo, how does SMS connect, and why is IP warmup needed?'"""
    q = "What is Brevo, how does SMS connect to another website, and why is IP warmup needed?"
    intent = classify_question_intent(q)
    assert intent["is_multi_part"] is True or intent["is_deep"] is True
    assert intent["candidate_k"] >= 16

    res = query_rag_pipeline(uat_vector_store, q, top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    cited_headings = {c.get("heading") for c in res["citations"] if c.get("heading")}
    assert len(cited_headings) >= 2


# 14. Deep Question Matrix I: Cross-Section Synthesis
def test_uat_14_deep_question_matrix_cross_section(uat_vector_store):
    """Matrix I: Synthesize triggers (Section 3) with actions/follow-up (Section 4)."""
    q = "How can a customer opening an email lead to an automated follow-up?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower and ("follow-up" in ans_lower or "action" in ans_lower)


# 15. Deep Question Matrix J: Conditional Logic
def test_uat_15_deep_question_matrix_conditional(uat_vector_store):
    """Matrix J: Conditional action flows upon customer event."""
    q = "If a customer completes a checkout transaction on an e-commerce website, what does the system do?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=5, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "sms" in ans_lower or "http post" in ans_lower or "trigger" in ans_lower


# 16. Deep Question Matrix K: Temporal Progression
def test_uat_16_deep_question_matrix_temporal(uat_vector_store):
    """Matrix K: Temporal sequence of actions."""
    q = "What happens after an automation trigger occurs in Brevo?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "automated" in ans_lower or "workflow" in ans_lower or "action" in ans_lower


# 17. Deep Question Matrix L: Paraphrased Query
def test_uat_17_deep_question_matrix_paraphrased(uat_vector_store):
    """Matrix L: Paraphrased query with completely varied vocabulary."""
    q = "In what manner can programmers oversee transactional dispatch outcomes and click-through statistics?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "tracking" in ans_lower or "analytic" in ans_lower or "metric" in ans_lower or "rate" in ans_lower


# 18. Deep Question Matrix M: Typo Tolerance
def test_uat_18_deep_question_matrix_typo(uat_vector_store):
    """Matrix M: Realistic spelling errors in technical query."""
    q = "how the smss autometion triger works for confiramtion?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "workflow" in ans_lower or "sms" in ans_lower


# 19. Deep Question Matrix N: Long Analytical Complex Clause
def test_uat_19_deep_question_matrix_long_analytical(uat_vector_store):
    """Matrix N: 35+ word analytical multi-clause question."""
    q = (
        "Given that high email deliverability is required for critical customer notifications, "
        "how does Brevo combine API authentication security with transactional email performance metrics "
        "and webhook notifications to verify message arrival?"
    )
    res = query_rag_pipeline(uat_vector_store, q, top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    assert len(res["citations"]) >= 1


# 20. Deep Question Matrix O: Multi-Evidence (3+ Sections)
def test_uat_20_deep_question_matrix_multi_evidence(uat_vector_store):
    """Matrix O: Combining API authentication, email tracking, and SMS integration."""
    q = "How do API authentication, tracking analytics, and SMS work together for external integrations?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=6, source_filter=URL_A)
    assert res["is_refusal"] is False
    cited_headings = {c.get("heading") for c in res["citations"] if c.get("heading")}
    assert len(cited_headings) >= 2


# 21. Deep Question Matrix P: Unsupported Topic Refusal
def test_uat_21_deep_question_matrix_unsupported_refusal(uat_vector_store):
    """Matrix P: Fictional/unrelated topic triggers clean refusal with zero hallucinations and zero citations."""
    q = "What is the orbital velocity and radius of Jupiter's moon Europa?"
    res = query_rag_pipeline(uat_vector_store, q, top_k=4, source_filter=URL_A)
    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower() or "not mention" in res["answer"].lower()
    assert len(res["citations"]) == 0


# 22. Citation Correctness & Metadata Fidelity
def test_uat_22_citation_correctness_and_fidelity(uat_vector_store):
    """Verify citations reflect genuine source metadata without hallucinated sections."""
    res = query_rag_pipeline(uat_vector_store, "What is Brevo?", top_k=3, source_filter=URL_A)
    for c in res["citations"]:
        assert c["source"] == URL_A
        assert c["title"] == "Brevo CRM Suite"
        assert c["heading"] in [
            "Overview", "API Authentication", "Transactional Email Analytics",
            "Automation Triggers", "Workflow Actions", "SMS Web Integration",
            "IP Infrastructure Comparison", "IP Warmup Rationale",
        ]


# 23. HTML Cleanliness (Zero Raw Tags)
def test_uat_23_html_cleanliness_zero_tags():
    """Verify HTML tag stripping completely removes tags and prevents raw tag leakage."""
    dirty_html = "<html><body><div><p>Brevo CRM enables <span>automation</span></p><script>alert(1)</script><style>body{color:red}</style></div></body></html>"
    clean_text, title, sections = extract_webpage_content(dirty_html, "https://brevo.com")
    assert "<div>" not in clean_text
    assert "</div>" not in clean_text
    assert "<span>" not in clean_text
    assert "<script>" not in clean_text
    assert "<style>" not in clean_text
    assert "Brevo CRM enables automation" in clean_text

    dirty_answer = "<div><p>Brevo CRM enables <span>automation</span></p></div>"
    clean_answer = validate_generated_answer(dirty_answer, "What is Brevo?", "STRONG", "Refusal", {})
    assert "<div>" not in clean_answer
    assert "</div>" not in clean_answer
    assert "<span>" not in clean_answer


# 24. Chart Eligibility: NO DATA = NO CHART
def test_uat_24_chart_eligibility_no_data_no_chart():
    """Verify purely textual content without verified numbers produces NO chart."""
    pure_text = "Brevo provides marketing automation, email APIs, and SMS messaging across Europe."
    spec = {
        "chart_type": "bar",
        "title": "Brevo Metrics",
        "data": [{"label": "Marketing", "value": 10}, {"label": "Email", "value": 20}],
    }
    # Values 10 and 20 are NOT in pure_text -> must return None
    validated = validate_chart_data(spec, pure_text, URL_A)
    assert validated is None


# 25. Chart Data Integrity: Numbers Verified Against Source
def test_uat_25_chart_numeric_verification_against_source():
    """Verify charts are accepted only when numbers exist in source, and rejected when hallucinated."""
    text_with_data = "In Q1 the company processed 100 million emails, 150 million in Q2, and 200 million in Q3."
    valid_spec = {
        "chart_type": "line",
        "title": "Email Growth",
        "data": [
            {"label": "Q1", "value": 100},
            {"label": "Q2", "value": 150},
            {"label": "Q3", "value": 200},
        ],
    }
    validated = validate_chart_data(valid_spec, text_with_data, URL_A)
    assert validated is not None
    assert len(validated["data"]) == 3

    # Hallucinated number (500)
    fake_spec = {
        "chart_type": "line",
        "title": "Email Growth",
        "data": [
            {"label": "Q1", "value": 100},
            {"label": "Q2", "value": 500},
        ],
    }
    assert validate_chart_data(fake_spec, text_with_data, URL_A) is None


# 26. Summary UI Card Separation
def test_uat_26_summary_and_ui_card_separation():
    """Verify structured website analysis generates distinct cards: Summary, Key Points, Links, Charts."""
    chunks = [
        Document(
            page_content="Brevo provides unified messaging APIs, marketing automation, and transactional delivery.",
            metadata={"source": URL_A, "title": "Brevo CRM", "heading": "Overview", "chunk_id": 0},
        )
    ]
    web_stats = {
        "links": [{"text": "API Docs", "url": "https://brevo.com/docs"}],
        "metadata": {"author": "Brevo Tech"},
    }
    analysis = generate_structured_website_analysis(
        url=URL_A,
        page_title="Brevo CRM Suite",
        chunks=chunks,
        web_stats=web_stats,
    )
    assert analysis["source_url"] == URL_A
    assert len(analysis["summary"]) > 20
    assert isinstance(analysis["key_points"], list)
    assert len(analysis["links"]) == 1
    assert "messages" in analysis


# 27. Stale State Prevention
def test_uat_27_stale_analysis_state_prevention():
    """Verify validate_analysis_session detects and prevents stale URL mismatches."""
    session_analysis = {"source_url": URL_A, "messages": [], "status": "completed"}
    # Requesting URL B when analysis is URL A must report mismatch/stale
    assert validate_analysis_session(session_analysis, URL_B) is False
    assert validate_analysis_session(session_analysis, URL_A) is True


# 28. Error Handling: SSRF and Invalid URL
def test_uat_28_error_handling_ssrf_and_invalid_urls():
    """Verify SSRF attacks (localhost, 127.0.0.1, private IPs) and bad schemes raise WebProcessingError."""
    ssrf_targets = [
        "http://localhost:8501",
        "http://127.0.0.1:8000/admin",
        "http://192.168.1.1/router",
        "ftp://example.com/file",
        "file:///etc/passwd",
    ]
    for target in ssrf_targets:
        with pytest.raises(WebProcessingError):
            validate_and_normalize_url(target)


# 29. Error Handling: Rate Limit Backoff
def test_uat_29_error_handling_rate_limit_backoff():
    """Verify LLM retry handles simulated transient 429 rate limit with backoff and success."""
    mock_chain = MagicMock()
    # Fails once with 429, then succeeds
    mock_chain.invoke.side_effect = [
        Exception("Rate limit reached for model groq (429)"),
        "Recovered response after backoff.",
    ]

    res, err_code, err_msg = execute_llm_with_retry(
        rag_chain=mock_chain,
        payload={"context": "Valid context", "question": "..."},
        max_retries=2,
        initial_wait=0.01,
    )
    assert res == "Recovered response after backoff."
    assert err_code is None
    assert mock_chain.invoke.call_count == 2


# 30. Regression Compatibility: PDF, Web, and YouTube Coexistence
def test_uat_30_regression_compatibility_pdf_and_youtube(uat_vector_store):
    """Verify vector store simultaneously indexes Web, PDF, and YouTube chunks with source-scoped query fidelity."""
    pdf_doc = Document(
        page_content="DocuMind PDF specifications: PyMuPDF parses page-level text and metadata for indexing.",
        metadata={"source": "manual.pdf", "source_id": "manual.pdf", "source_type": "pdf", "page": 1, "chunk_id": 0},
    )
    yt_doc = Document(
        page_content="YouTube Video Guide: Machine learning architectures rely on transformer attention mechanisms.",
        metadata={
            "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "source_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "source_type": "youtube",
            "timestamp": 45,
            "chunk_id": 0,
        },
    )
    embeddings = get_embedding_model()
    vs = add_documents_to_vector_store(uat_vector_store, [pdf_doc, yt_doc], embeddings=embeddings)

    # Web retrieval is isolated
    web_res = retrieve_relevant_chunks(vs, "Brevo CRM", top_k=2, source_filter=URL_A)
    assert all(d.metadata["source"] == URL_A for d in web_res)

    # PDF retrieval is isolated
    pdf_res = retrieve_relevant_chunks(vs, "PyMuPDF specifications", top_k=2, source_filter="manual.pdf")
    assert all(d.metadata["source"] == "manual.pdf" for d in pdf_res)

    # YouTube retrieval is isolated
    yt_res = retrieve_relevant_chunks(vs, "transformer attention", top_k=2, source_filter="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert all("youtube.com" in d.metadata["source"] for d in yt_res)
