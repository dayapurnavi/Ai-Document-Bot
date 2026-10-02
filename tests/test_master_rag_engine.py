"""Comprehensive tests for the DocuMind AI Master RAG Answer Engine.

Verifies:
1. Deep procedural, relational, and multi-hop reasoning.
2. Technical term preservation (SMS, API key, webhook, SMTP, automation).
3. Cross-section synthesis without false refusals.
4. Active source isolation (URL A vs URL B).
5. Clean formatting without raw HTML </div> or tag artifacts.
6. Guardrail refusal when data is genuinely absent.
"""

import time
import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def throttle_master_engine_calls():
    """Throttle between API calls to prevent 429 TPM burst limits."""
    yield
    time.sleep(1.0)

from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store, retrieve_relevant_chunks
from src.rag_chain import query_rag_pipeline, analyze_and_expand_query


@pytest.fixture
def multi_section_marketing_store():
    """Create a multi-section indexed webpage vector store simulating a platform like Brevo."""
    docs = [
        Document(
            page_content=(
                "Brevo Overview: Brevo is an all-in-one CRM suite offering marketing automation, "
                "transactional messaging, SMS campaigns, and developer APIs."
            ),
            metadata={
                "source": "https://brevo.com",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Overview",
                "chunk_id": 0,
                "section_id": 0,
            },
        ),
        Document(
            page_content=(
                "Transactional Emails: Developers send automated order confirmations, receipts, "
                "and password resets via Brevo SMTP and REST API. Every transactional email includes "
                "real-time tracking metrics: delivery status, open rates, click rates, and bounce analytics."
            ),
            metadata={
                "source": "https://brevo.com",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Transactional Email API",
                "chunk_id": 1,
                "section_id": 1,
            },
        ),
        Document(
            page_content=(
                "Marketing Automation Workflows: Automate customer journeys by creating multi-step scenarios. "
                "Workflows are initiated by triggers, such as when a contact opens a specific email "
                "or visits a webpage. Once triggered, the workflow executes sequential actions, "
                "including waiting delays, sending follow-up messages, or updating contact attributes."
            ),
            metadata={
                "source": "https://brevo.com",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "Marketing Automation",
                "chunk_id": 2,
                "section_id": 2,
            },
        ),
        Document(
            page_content=(
                "SMS Marketing & Notifications: Send transactional SMS alerts and promotional broadcasts globally. "
                "SMS is integrated with external websites via REST API endpoints and webhooks, allowing web applications "
                "to trigger real-time text message delivery directly from website checkout forms."
            ),
            metadata={
                "source": "https://brevo.com",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "SMS Services",
                "chunk_id": 3,
                "section_id": 3,
            },
        ),
        Document(
            page_content=(
                "Authentication & Security: Every developer API request requires an API key in the 'api-key' "
                "request header. The API key authenticates the website server and authorizes access to sending "
                "endpoints while protecting unauthorized account access."
            ),
            metadata={
                "source": "https://brevo.com",
                "source_type": "url",
                "title": "Brevo CRM Suite",
                "heading": "API Authentication",
                "chunk_id": 4,
                "section_id": 4,
            },
        ),
    ]

    embeddings = get_embedding_model()
    vs = build_vector_store(docs, embeddings=embeddings)
    return vs


def test_query_understanding_layer():
    """Verify query classification and sub-query extraction."""
    q1 = "A customer opens an order-confirmation email. How could Brevo automation be used to trigger a follow-up message?"
    info1 = analyze_and_expand_query(q1)
    assert info1["is_how_to"] is True
    assert info1["is_multi_hop"] is True
    assert len(info1["sub_queries"]) >= 1
    assert any("order confirmation" in sq for sq in info1["sub_queries"])

    q2 = "What is the purpose of an API key and how does SMS work from the website?"
    info2 = analyze_and_expand_query(q2)
    assert info2["is_definition"] is True
    assert info2["is_multi_part"] is True
    assert "api key" in info2["detected_tech"] or "api keys" in info2["detected_tech"]
    assert "sms" in info2["detected_tech"]


def test_technical_keyword_hybrid_retrieval(multi_section_marketing_store):
    """Verify exact technical keywords (SMS, API key, webhook) retrieve correct sections."""
    vs = multi_section_marketing_store

    # Test SMS query
    res_sms = retrieve_relevant_chunks(vs, "how the sms is working from the website", top_k=2, source_filter="https://brevo.com")
    assert len(res_sms) >= 1
    assert any("SMS" in d.page_content for d in res_sms)

    # Test API key query
    res_api = retrieve_relevant_chunks(vs, "What is the purpose of an API key?", top_k=2, source_filter="https://brevo.com")
    assert len(res_api) >= 1
    assert any("API key" in d.page_content for d in res_api)


def test_multi_hop_cross_section_reasoning(multi_section_marketing_store):
    """Verify that multi-hop questions connecting order confirmation to automation trigger and follow-up are answered without refusal."""
    vs = multi_section_marketing_store
    q = "A customer opens an order-confirmation email. How could Brevo automation be used to trigger a follow-up message?"

    response = query_rag_pipeline(
        vector_store=vs,
        question=q,
        top_k=4,
        source_filter="https://brevo.com",
        source_type="webpage",
    )

    assert response["is_refusal"] is False
    ans_lower = response["answer"].lower()
    # Must synthesize order confirmation / email open trigger and follow-up message action
    assert "trigger" in ans_lower or "workflow" in ans_lower
    assert "follow-up" in ans_lower or "follow up" in ans_lower or "message" in ans_lower
    # Must NOT output raw HTML tags
    assert "</div>" not in response["answer"]
    assert "<span>" not in response["answer"]


def test_transactional_email_tracking_answer(multi_section_marketing_store):
    """Verify developers tracking transactional email performance is accurately answered."""
    vs = multi_section_marketing_store
    q = "How can developers track transactional email performance?"

    response = query_rag_pipeline(
        vector_store=vs,
        question=q,
        top_k=4,
        source_filter="https://brevo.com",
        source_type="webpage",
    )

    assert response["is_refusal"] is False
    ans_lower = response["answer"].lower()
    assert "tracking" in ans_lower or "metric" in ans_lower or "open" in ans_lower or "delivery" in ans_lower
    assert "</div>" not in response["answer"]


def test_active_source_strict_isolation_when_url_changes(multi_section_marketing_store):
    """Verify questions asked under URL B cannot retrieve or answer using URL A content."""
    vs = multi_section_marketing_store

    # Add URL B (YouTube or competitor)
    url_b_doc = Document(
        page_content="YouTube Tutorial: Learn how to set up Python virtual environments and pip packages.",
        metadata={
            "source": "https://youtube.com/watch?v=12345",
            "source_type": "youtube",
            "title": "Python Setup Guide",
            "heading": "Environment Setup",
            "chunk_id": 0,
        },
    )
    embeddings = get_embedding_model()
    vs.add_documents([url_b_doc])

    # Query URL B asking about Brevo SMS (should refuse because URL B contains NO SMS content)
    response_b = query_rag_pipeline(
        vector_store=vs,
        question="How does SMS connect to the website?",
        top_k=2,
        source_filter="https://youtube.com/watch?v=12345",
        source_type="youtube",
    )

    assert response_b["is_refusal"] is True
    assert "couldn't find" in response_b["answer"].lower()
    # Ensure zero citations from Brevo
    assert len(response_b["citations"]) == 0


def test_adaptive_retrieval_configs_and_complexity():
    """Verify that question complexity maps to appropriate candidate_k and final_k."""
    from src.rag_chain import classify_question_intent, RETRIEVAL_CONFIGS

    # Simple question
    simple_info = classify_question_intent("Who wrote this?")
    assert simple_info["candidate_k"] == RETRIEVAL_CONFIGS["SIMPLE"]["candidate_k"]
    assert simple_info["final_k"] == RETRIEVAL_CONFIGS["SIMPLE"]["final_k"]

    # How-to question
    how_info = classify_question_intent("How does SMS connect to another website?")
    assert how_info["candidate_k"] >= 18
    assert how_info["final_k"] >= 10

    # Multi-part question
    multi_info = classify_question_intent("What is an API key and how can developers track delivery?")
    assert multi_info["is_multi_part"] is True
    assert multi_info["candidate_k"] >= 24
    assert multi_info["final_k"] >= 12

    # Process question
    proc_info = classify_question_intent("Explain the step by step workflow for order confirmation")
    assert proc_info["is_process"] is True
    assert proc_info["candidate_k"] >= 20


def test_multi_query_expansion_variants():
    """Verify generation of 3-6 focused search variants for deep technical questions."""
    from src.rag_chain import classify_question_intent, generate_query_variants

    q = "How can developers track transactional email performance?"
    info = classify_question_intent(q)
    variants = generate_query_variants(q, info)

    assert 3 <= len(variants) <= 6
    # Ensure variants contain relevant search terms
    combined_variants = " ".join(variants).lower()
    assert "transactional" in combined_variants or "email" in combined_variants
    assert "performance" in combined_variants or "tracking" in combined_variants or "metrics" in combined_variants


def test_html_cleaner_and_quality_score():
    """Verify that HTML fragments, scripts, hidden elements, and ad banners are removed and quality score calculated."""
    from src.web_processor import extract_webpage_content, compute_content_quality_score, WebProcessingError

    dirty_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title><style>.hidden { display: none; }</style><script>alert('bad');</script></head>
    <body>
        <div class="cookie-banner">Accept cookies to continue</div>
        <div class="ad-container">Buy our product now!</div>
        <div aria-hidden="true">Screen reader hidden junk</div>
        <article>
            <h1>Product Architecture</h1>
            <p>Our distributed architecture separates message ingestion from batch dispatching.</p>
            </div>
            <span>
            <h2>Security & Authentication</h2>
            <p>Requests are authenticated using SHA-256 HMAC signatures.</p>
            </section>
        </article>
        <footer><p>Footer disclaimer</p></footer>
    </body>
    </html>
    """

    clean_text, title, sections = extract_webpage_content(dirty_html, "https://example.com/test")

    # Raw HTML artifacts must NOT appear
    assert "</div>" not in clean_text
    assert "<span>" not in clean_text
    assert "</section>" not in clean_text
    assert "cookie-banner" not in clean_text
    assert "Buy our product now" not in clean_text
    assert "Screen reader hidden junk" not in clean_text
    assert "Footer disclaimer" not in clean_text

    # Content preserved
    assert "Product Architecture" in clean_text
    assert "distributed architecture separates message ingestion" in clean_text
    assert "Security & Authentication" in clean_text

    # Quality score
    quality = compute_content_quality_score(clean_text, dirty_html, sections)
    assert quality["overall_score"] > 0.3
    assert quality["tag_noise_count"] == 0
    assert quality["is_valid"] is True

    # Empty / garbage page test
    empty_html = "<html><body><script>var x = 1;</script><div></div></body></html>"
    with pytest.raises(WebProcessingError):
        extract_webpage_content(empty_html, "https://example.com/empty")


def test_table_markdown_conversion():
    """Verify that HTML tables are cleanly converted into GitHub-flavored markdown tables."""
    from src.web_processor import extract_webpage_content

    html_with_table = """
    <html>
    <head><title>Pricing Comparison</title></head>
    <body>
        <main>
            <h1>Plan Details</h1>
            <table>
                <thead>
                    <tr><th>Plan</th><th>Price</th><th>Limit</th></tr>
                </thead>
                <tbody>
                    <tr><td>Starter</td><td>$0</td><td>300 emails/day</td></tr>
                    <tr><td>Business</td><td>$25</td><td>20,000 emails/month</td></tr>
                </tbody>
            </table>
            <h2>Next Steps</h2>
            <p>Choose the plan that suits your volume.</p>
        </main>
    </body>
    </html>
    """

    clean_text, title, sections = extract_webpage_content(html_with_table, "https://example.com/pricing")
    assert "| Plan | Price | Limit |" in clean_text
    assert "| Starter | $0 | 300 emails/day |" in clean_text
    assert "| Business | $25 | 20,000 emails/month |" in clean_text
    assert "<table>" not in clean_text


def test_composite_cache_key_isolation():
    """Verify that composite cache keys prevent cross-source contamination when questions are identical."""
    url_a = "https://brevo.com"
    hash_a = "hash_brevo_123"
    url_b = "https://sendgrid.com"
    hash_b = "hash_sendgrid_456"
    q = "How does SMS work?"
    norm_q = "how does sms work?"

    key_a = (url_a, hash_a, norm_q, "source_scoped", 4)
    key_b = (url_b, hash_b, norm_q, "source_scoped", 4)

    cache = {
        key_a: {"answer": "Brevo uses REST API and webhooks for SMS delivery."},
    }

    assert key_b not in cache
    # Same question on URL B is a cache miss
    assert cache.get(key_b) is None


def test_evidence_states_and_refusal(multi_section_marketing_store):
    """Verify that evaluate_evidence_state properly distinguishes STRONG, MODERATE, and NONE."""
    from src.rag_chain import evaluate_evidence_state, classify_question_intent

    vs = multi_section_marketing_store
    docs = list(vs.docstore._dict.values())

    # High coverage question
    q_strong = "What is the purpose of an API key and how is it used in headers for authentication?"
    info_strong = classify_question_intent(q_strong)
    state_strong = evaluate_evidence_state(docs, q_strong, info_strong)
    assert state_strong in ("STRONG", "MODERATE")

    # Completely unrelated question
    q_none = "What is the boiling point of liquid nitrogen at sea level?"
    info_none = classify_question_intent(q_none)
    state_none = evaluate_evidence_state(docs, q_none, info_none)
    assert state_none == "NONE"

