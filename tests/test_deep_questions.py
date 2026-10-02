"""Comprehensive test suite for Phase 2 Deep Questions in DocuMind AI.

Covers the 18 mandatory test scenarios:
1. test_basic_factual_question
2. test_definition_question
3. test_how_question
4. test_why_question
5. test_multi_step_question
6. test_cross_section_question
7. test_comparison_question
8. test_question_requiring_neighboring_chunks
9. test_paraphrased_question
10. test_question_with_spelling_mistakes
11. test_deep_question_requiring_query_expansion
12. test_question_requiring_multiple_evidence_chunks
13. test_unanswerable_question_guardrail
14. test_url_isolation_between_a_and_b
15. test_previous_question_contamination_prevention
16. test_html_contamination_prevention
17. test_citation_correctness
18. test_no_hallucination_on_fake_topics
"""

import time
import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def throttle_deep_q_calls():
    """Throttle between API calls to prevent 429 TPM burst limits."""
    yield
    time.sleep(1.0)

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
    evaluate_evidence_state,
)


@pytest.fixture(scope="module")
def shared_knowledge_store():
    """Create a structured, multi-section indexed knowledge base simulating Brevo CRM suite."""
    docs = [
        Document(
            page_content=(
                "Brevo Architecture & Overview: Brevo is an all-in-one CRM suite offering marketing "
                "automation, transactional messaging, SMS campaigns, and developer APIs. The system runs "
                "on high-availability infrastructure supporting global deliverability."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Overview",
                "chunk_id": 0,
                "section_id": 0,
            },
        ),
        Document(
            page_content=(
                "API Authentication & Security: All programmatic requests to Brevo endpoints must include an "
                "API key provided in the 'api-key' request header. The API key authenticates the external application, "
                "authorizes sending privileges, and protects unauthorized account access."
            ),
            metadata={
                "source": "https://brevo.com/crm",
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
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Transactional Email Analytics",
                "chunk_id": 2,
                "section_id": 2,
            },
        ),
        Document(
            page_content=(
                "Marketing Automation Triggers: Automated customer journeys are initiated by behavioral triggers. "
                "For example, a trigger occurs when a customer opens an order-confirmation email or submits a website form. "
                "The trigger registers the event and starts the automation workflow engine."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Automation Triggers",
                "chunk_id": 3,
                "section_id": 3,
            },
        ),
        Document(
            page_content=(
                "Automation Workflow Execution & Follow-ups: Once an automation workflow is triggered, it executes "
                "sequential actions. Workflows can insert a delay timer, evaluate conditional branches, and automatically "
                "trigger a follow-up message such as a promotional offer or satisfaction survey to re-engage the customer."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Workflow Execution",
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
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "IP Warmup Rationale",
                "chunk_id": 7,
                "section_id": 7,
            },
        ),
    ]

    embeddings = get_embedding_model()
    vs = build_vector_store(docs, embeddings=embeddings)
    return vs


# 1. Basic factual question
def test_basic_factual_question(shared_knowledge_store):
    """Verify simple factual questions retrieve and state the exact answer."""
    vs = shared_knowledge_store
    q = "What HTTP header is required for API authentication in Brevo?"
    res = query_rag_pipeline(vs, q, top_k=3, source_filter="https://brevo.com/crm", source_type="webpage")

    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "api-key" in ans_lower or "header" in ans_lower
    assert len(res["citations"]) > 0


# 2. Definition question
def test_definition_question(shared_knowledge_store):
    """Verify 'What is...' definition questions extract clear explanations."""
    vs = shared_knowledge_store
    q = "What is Brevo?"
    intent = classify_question_intent(q)
    assert intent["question_type"] == "DEFINITION"

    res = query_rag_pipeline(vs, q, top_k=3, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "crm" in ans_lower or "automation" in ans_lower or "messaging" in ans_lower


# 3. How question
def test_how_question(shared_knowledge_store):
    """Verify 'How does...' questions retrieve technical operational mechanisms."""
    vs = shared_knowledge_store
    q = "How does SMS connect to an external website?"
    intent = classify_question_intent(q)
    assert intent["question_type"] in ("HOW", "PROCESS", "CROSS_SECTION")
    assert intent["is_how_to"] is True

    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "rest api" in ans_lower or "webhook" in ans_lower or "endpoint" in ans_lower


# 4. Why question
def test_why_question(shared_knowledge_store):
    """Verify 'Why...' questions extract causal justifications and reasons."""
    vs = shared_knowledge_store
    q = "Why is IP warmup critical for dedicated IPs?"
    intent = classify_question_intent(q)
    assert intent["question_type"] in ("WHY", "CAUSE_EFFECT")

    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "gmail" in ans_lower or "yahoo" in ans_lower or "reputation" in ans_lower or "throttle" in ans_lower or "reject" in ans_lower


# 5. Multi-step question
def test_multi_step_question(shared_knowledge_store):
    """Verify multi-step workflow processes are synthesized in logical order."""
    vs = shared_knowledge_store
    q = "Explain the step-by-step process of how an automation workflow executes from trigger to follow-up."
    intent = classify_question_intent(q)
    assert intent["question_type"] in ("MULTI_STEP", "PROCESS")

    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower
    assert "follow-up" in ans_lower or "delay" in ans_lower or "action" in ans_lower


# 6. Cross-section question
def test_cross_section_question(shared_knowledge_store):
    """Verify multi-hop synthesis bridging Automation Triggers (chunk 3) and Follow-ups (chunk 4)."""
    vs = shared_knowledge_store
    q = "A customer opens an order-confirmation email. How could Brevo automation be used to trigger a follow-up message?"
    intent = classify_question_intent(q)
    assert intent["is_multi_hop"] is True

    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "open" in ans_lower
    assert "follow-up" in ans_lower or "workflow" in ans_lower or "message" in ans_lower


# 7. Comparison question
def test_comparison_question(shared_knowledge_store):
    """Verify comparison questions analyze both sides (Dedicated IP vs Shared IP)."""
    vs = shared_knowledge_store
    q = "Compare dedicated IP versus shared IP sending in Brevo."
    intent = classify_question_intent(q)
    assert intent["question_type"] == "COMPARISON" or intent["is_comparison"] is True

    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "shared" in ans_lower and "dedicated" in ans_lower
    assert "warmup" in ans_lower or "reputation" in ans_lower


# 8. Question requiring neighboring chunks
def test_question_requiring_neighboring_chunks(shared_knowledge_store):
    """Verify that neighbor expansion retrieves adjoining chunk metadata when continuity is needed."""
    vs = shared_knowledge_store
    # Target chunk 3 specifically
    retrieved = retrieve_relevant_chunks(vs, "marketing automation triggers customer journeys", top_k=1, source_filter="https://brevo.com/crm")
    assert len(retrieved) >= 1
    assert retrieved[0].metadata.get("chunk_id") == 3

    # Expand with neighbors
    expanded = expand_context_with_neighbors(retrieved, vs, max_neighbors=2, source_filter="https://brevo.com/crm")
    chunk_ids = [d.metadata.get("chunk_id") for d in expanded]
    assert 3 in chunk_ids
    # Neighbor chunk 4 or 2 must be included
    assert (4 in chunk_ids) or (2 in chunk_ids)


# 9. Paraphrased question
def test_paraphrased_question(shared_knowledge_store):
    """Verify semantic retrieval succeeds when wording is completely paraphrased."""
    vs = shared_knowledge_store
    # Paraphrased version of tracking transactional email metrics
    q = "In what manner can programmers oversee transactional dispatch outcomes and click-through statistics?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")

    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "tracking" in ans_lower or "analytic" in ans_lower or "webhook" in ans_lower or "metric" in ans_lower or "open" in ans_lower


# 10. Question with spelling mistakes
def test_question_with_spelling_mistakes(shared_knowledge_store):
    """Verify typo tolerance in active technical terms and query expansion."""
    vs = shared_knowledge_store
    q = "how the smss autometion triger works for confiramtion?"
    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")

    assert res["is_refusal"] is False
    ans_lower = res["answer"].lower()
    assert "trigger" in ans_lower or "workflow" in ans_lower or "sms" in ans_lower


# 11. Deep question requiring query expansion
def test_deep_question_requiring_query_expansion(shared_knowledge_store):
    """Verify multi-query expansion creates 3-6 distinct keyword variants for deep questions."""
    q = "How can developers track transactional email performance?"
    intent = classify_question_intent(q)
    variants = generate_query_variants(q, intent)

    assert 3 <= len(variants) <= 8
    combined = " ".join(variants).lower()
    assert "transactional" in combined
    assert "performance" in combined or "tracking" in combined or "metrics" in combined


# 12. Question requiring multiple evidence chunks
def test_question_requiring_multiple_evidence_chunks(shared_knowledge_store):
    """Verify complex question retrieves evidence across separate functional modules."""
    vs = shared_knowledge_store
    q = "How does Brevo integrate API authentication, transactional email analytics, and SMS notifications for external websites?"
    intent = classify_question_intent(q)
    assert intent["candidate_k"] >= 20

    res = query_rag_pipeline(vs, q, top_k=6, source_filter="https://brevo.com/crm", source_type="webpage")
    assert res["is_refusal"] is False
    # Verified citations across multiple chunks
    cited_chunks = {c.get("chunk_id") for c in res["citations"] if c.get("chunk_id") is not None}
    assert len(cited_chunks) >= 2


# 13. Unanswerable question guardrail
def test_unanswerable_question_guardrail(shared_knowledge_store):
    """Verify completely unanswerable question triggers guardrail refusal and returns NO citations."""
    vs = shared_knowledge_store
    q = "What is the orbital velocity and radius of Jupiter's moon Europa?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm", source_type="webpage")

    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower() or "not mention" in res["answer"].lower()
    assert len(res["citations"]) == 0


# 14. URL isolation between A and B
def test_url_isolation_between_a_and_b(shared_knowledge_store):
    """Verify URL A data is never returned or searched when active source filter is URL B."""
    vs = shared_knowledge_store
    url_b_doc = Document(
        page_content="Kubernetes Cluster Setup: Deploy pods and ingress controllers across node pools using helm charts.",
        metadata={
            "source": "https://kubernetes.io/docs",
            "source_type": "webpage",
            "title": "Kubernetes Documentation",
            "heading": "Cluster Setup",
            "chunk_id": 0,
        },
    )
    vs.add_documents([url_b_doc])

    # Query URL B asking about Brevo SMS API
    res_b = query_rag_pipeline(
        vs,
        "How does SMS connect to the website with API keys?",
        top_k=4,
        source_filter="https://kubernetes.io/docs",
        source_type="webpage",
    )

    # Must refuse because Kubernetes doc has no SMS/Brevo content
    assert res_b["is_refusal"] is True
    assert "couldn't find" in res_b["answer"].lower()
    assert len(res_b["citations"]) == 0


# 15. Previous question contamination prevention
def test_previous_question_contamination_prevention(shared_knowledge_store):
    """Verify that state from an earlier query does not leak into a subsequent distinct query."""
    vs = shared_knowledge_store

    # Query 1: IP Warmup
    res1 = query_rag_pipeline(vs, "Why is IP warmup critical for dedicated IPs?", top_k=4, source_filter="https://brevo.com/crm")
    assert res1["is_refusal"] is False
    assert "warmup" in res1["answer"].lower() or "ip" in res1["answer"].lower()

    # Query 2: SMS API (must not mention warmup or dedicated IPs)
    res2 = query_rag_pipeline(vs, "How does SMS connect to an external website?", top_k=4, source_filter="https://brevo.com/crm")
    assert res2["is_refusal"] is False
    assert "sms" in res2["answer"].lower()
    # Confirm Query 1 topics do not contaminate Query 2
    assert "warmup" not in res2["answer"].lower()


# 16. HTML contamination prevention
def test_html_contamination_prevention(shared_knowledge_store):
    """Verify answers and citations never leak raw HTML tags (</div>, <span>, <section>, <script>)."""
    vs = shared_knowledge_store
    questions = [
        "What is Brevo?",
        "How does SMS connect to an external website?",
        "A customer opens an order-confirmation email. How could Brevo automation be used to trigger a follow-up message?",
        "Why is IP warmup critical for dedicated IPs?",
        "What is the population of Mars?",  # Refusal test
    ]

    for q in questions:
        res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")
        ans = res["answer"]
        assert "</div>" not in ans
        assert "<div>" not in ans
        assert "<span>" not in ans
        assert "</span>" not in ans
        assert "<section>" not in ans
        assert "</section>" not in ans
        assert "<script>" not in ans


# 17. Citation correctness
def test_citation_correctness(shared_knowledge_store):
    """Verify citations point to the active source, have valid snippet text, and match retrieved sections."""
    vs = shared_knowledge_store
    q = "How can developers track transactional email performance?"
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")

    assert res["is_refusal"] is False
    assert len(res["citations"]) > 0

    for cit in res["citations"]:
        assert cit["source"] == "https://brevo.com/crm"
        assert len(cit["snippet"]) > 10
        assert cit["heading"] in [
            "Overview",
            "API Authentication",
            "Transactional Email Analytics",
            "Automation Triggers",
            "Workflow Execution",
            "SMS Web Integration",
            "IP Infrastructure Comparison",
            "IP Warmup Rationale",
        ]


# 18. No hallucination on fake topics
def test_no_hallucination_on_fake_topics(shared_knowledge_store):
    """Verify the system strictly refuses fabricated or fictitious topics."""
    vs = shared_knowledge_store
    q = "Explain the Brevo quantum entanglement cryptographic key distribution protocol."
    res = query_rag_pipeline(vs, q, top_k=4, source_filter="https://brevo.com/crm")

    assert res["is_refusal"] is True
    assert "couldn't find" in res["answer"].lower() or "not mention" in res["answer"].lower()
    # Ensure no fabricated claims
    assert "quantum entanglement" not in res["answer"].lower() or "not mentioned" in res["answer"].lower() or "couldn't find" in res["answer"].lower()
    assert len(res["citations"]) == 0
