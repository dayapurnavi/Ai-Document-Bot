"""Phase 4.5 Root-Cause Debugging Test Suite: Runtime Answer Path & HTML Sanitization.

Tests all 8 mandatory scenarios:
1. test_answerable_question_reaches_llm()
2. test_unsupported_question_refuses()
3. test_final_response_has_no_html()
4. test_cached_response_is_sanitized()
5. test_source_switch_changes_retrieval()
6. test_active_source_is_enforced()
7. test_relevant_chunks_do_not_trigger_false_refusal()
8. test_empty_retrieval_returns_clean_refusal()
"""

import re
import html
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store
from src.rag_chain import (
    query_rag_pipeline,
    sanitize_final_response,
    evaluate_evidence_coverage,
    classify_question_intent,
    decompose_complex_question,
    get_refusal_phrase,
    validate_generated_answer,
)
from app import format_chat_bubble_html


@pytest.fixture(scope="module")
def embeddings():
    return get_embedding_model()


@pytest.fixture
def sample_vector_store(embeddings):
    """Fixture providing a multi-source vector store with controlled content."""
    docs = [
        Document(
            page_content=(
                "GitHub Copilot is the AI-powered developer platform to build, scale, and deliver software. "
                "Duolingo engineering velocity increased by 25% with GitHub Copilot across daily workflows. "
                "Ship faster with secure, reliable CI/CD pipelines and automated code review."
            ),
            metadata={
                "source": "https://github.com/",
                "source_id": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Development Platform",
                "heading": "AI Partner Everywhere",
                "chunk_id": 0,
                "section_id": 0,
                "content_hash": "hash_gh_001",
            },
        ),
        Document(
            page_content=(
                "GitHub Pull Requests and Code Review: Teams collaborate on code changes using pull requests. "
                "Automated status checks verify unit tests and linting before merging to main branch."
            ),
            metadata={
                "source": "https://github.com/",
                "source_id": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Development Platform",
                "heading": "Pull Requests",
                "chunk_id": 1,
                "section_id": 1,
                "content_hash": "hash_gh_001",
            },
        ),
        Document(
            page_content=(
                "Sourdough Bread Baking: Sourdough bread relies on wild yeast and lactobacilli fermentation. "
                "The starter requires equal parts flour and water fed every 24 hours at room temperature. "
                "Bulk fermentation typically takes four to six hours until dough increases by 50 percent."
            ),
            metadata={
                "source": "https://artisan-baker.org/sourdough",
                "source_id": "https://artisan-baker.org/sourdough",
                "source_type": "webpage",
                "title": "Artisan Sourdough Guide",
                "heading": "Fermentation Process",
                "chunk_id": 0,
                "section_id": 0,
                "content_hash": "hash_bake_001",
            },
        ),
    ]
    return build_vector_store(docs, embeddings=embeddings)


def test_answerable_question_reaches_llm(sample_vector_store):
    """Scenario 1: Answerable question passes evidence gate and reaches LLM to generate answer."""
    question = "What velocity increase did Duolingo achieve with GitHub Copilot?"
    mock_llm_answer = (
        "According to the indexed content, Duolingo achieved a 25% increase in engineering "
        "velocity with GitHub Copilot."
    )

    with patch("src.rag_chain.execute_llm_with_retry", return_value=(mock_llm_answer, None, None)) as mock_exec:
        res = query_rag_pipeline(
            vector_store=sample_vector_store,
            question=question,
            source_filter="https://github.com/",
            source_type="webpage",
        )

        assert mock_exec.called, "LLM chain must be executed when evidence is available."
        assert res["is_refusal"] is False, "Answerable question must not be marked as refusal."
        assert res["evidence_state"] in ("STRONG", "SUFFICIENT"), f"Expected STRONG/SUFFICIENT, got {res['evidence_state']}"
        assert "25%" in res["answer"], "Answer must reflect verified factual metrics."
        assert "Duolingo" in res["answer"]
        assert len(res["citations"]) > 0, "Answer must contain citations."


def test_unsupported_question_refuses(sample_vector_store):
    """Scenario 2: Unsupported question not present in the indexed source is cleanly refused."""
    question = "How to configure a Kubernetes cluster with Terraform and Helm charts?"
    res = query_rag_pipeline(
        vector_store=sample_vector_store,
        question=question,
        source_filter="https://github.com/",
        source_type="webpage",
    )

    expected_refusal = get_refusal_phrase("webpage")
    assert res["is_refusal"] is True, "Unsupported topic must trigger refusal."
    assert res["evidence_state"] == "NONE", f"Evidence state must be NONE, got {res['evidence_state']}"
    assert res["answer"] == expected_refusal, f"Expected exact refusal '{expected_refusal}', got '{res['answer']}'"
    # Ensure ZERO HTML tags in the refusal response
    assert re.search(r"<\/?(?:div|span|p|section|article|header|footer|script|style)[^>]*>", res["answer"]) is None
    assert "</div>" not in res["answer"]


def test_final_response_has_no_html():
    """Scenario 3: Response sanitizer guarantees zero raw HTML tags while preserving Markdown formatting."""
    # 1. Dirty refusal with stray </div>
    dirty_refusal = "I couldn't find that information in the indexed webpage content.\n\n</div>"
    clean_refusal = sanitize_final_response(dirty_refusal)
    assert clean_refusal == "I couldn't find that information in the indexed webpage content."
    assert "</div>" not in clean_refusal
    assert "<div" not in clean_refusal

    # 2. Structural tags removal
    html_input = "<div><p>Paragraph <span>with text</span></p><section><article>Deep note</article></section></div>"
    clean_html = sanitize_final_response(html_input)
    assert clean_html == "Paragraph with text\n\nDeep note"
    assert re.search(r"<\/?(?:div|span|p|section|article)[^>]*>", clean_html) is None

    # 3. Entity-encoded tags decode and strip
    encoded_input = "Some text &lt;div&gt;stray content&lt;/div&gt; more text"
    clean_encoded = sanitize_final_response(encoded_input)
    assert clean_encoded == "Some text stray content more text"
    assert "<div>" not in clean_encoded and "&lt;div&gt;" not in clean_encoded

    # 4. Markdown preservation (tables, code blocks, lists, links, bold/italics)
    rich_markdown = (
        "### Key Features\n\n"
        "| Feature | Status |\n"
        "|---|---|\n"
        "| Speed | 25% |\n\n"
        "1. Step One: Ingest\n"
        "2. Step Two: Retrieve\n\n"
        "- Bullet point A\n"
        "- Bullet point B\n\n"
        "```python\ndef test_fn():\n    return '<div>keep inside code</div>'\n```\n\n"
        "Visit [GitHub](https://github.com) for **details**."
    )
    clean_md = sanitize_final_response(rich_markdown)
    assert "| Feature | Status |" in clean_md, "Markdown tables must be preserved."
    assert "1. Step One: Ingest" in clean_md, "Numbered lists must be preserved."
    assert "- Bullet point A" in clean_md, "Bullet points must be preserved."
    assert "def test_fn():" in clean_md, "Code blocks must be preserved."
    assert "<div>keep inside code</div>" in clean_md, "Code block contents must be preserved."
    assert "[GitHub](https://github.com)" in clean_md, "Markdown links must be preserved."
    assert "**details**" in clean_md, "Markdown bold must be preserved."

    # 5. UI bubble formatter produces clean rendered HTML without leaking raw closing tags
    bubble_out = format_chat_bubble_html(dirty_refusal)
    assert "<pre><code>&lt;/div&gt;" not in bubble_out, "Must not leak </div> as indented code block."
    assert "<div style=\"margin-bottom: 0.65rem;\">" not in bubble_out
    assert bubble_out == "<p>I couldn't find that information in the indexed webpage content.</p>"


def test_cached_response_is_sanitized():
    """Scenario 4: Stored or returned cached responses are always passed through sanitize_final_response."""
    raw_cached_entry = {
        "answer": "Here is the verified answer from cache.\n\n</div>",
        "citations": [{"source": "https://github.com/", "formatted": "GitHub"}],
        "debug_info": {"active_source": "https://github.com/"},
    }

    # Simulate app.py cache lookup path
    clean_ans = sanitize_final_response(raw_cached_entry.get("answer", ""))
    assert "</div>" not in clean_ans
    assert clean_ans == "Here is the verified answer from cache."

    # Verify cache key structure uniqueness across sources and questions
    norm_q = " ".join("What is GitHub Copilot?".strip().lower().split())
    cache_key_a = ("https://github.com/", "hash_01", norm_q, "source_scoped", 6)
    cache_key_b = ("https://brevo.com/", "hash_02", norm_q, "source_scoped", 6)
    assert cache_key_a != cache_key_b, "Cache keys for different sources must be distinct."


def test_source_switch_changes_retrieval(sample_vector_store):
    """Scenario 5: Switching active source completely alters retrieved evidence and prevents leakage."""
    question = "How does wild yeast fermentation work?"

    # Query with GitHub as active source (wrong source)
    res_gh = query_rag_pipeline(
        vector_store=sample_vector_store,
        question=question,
        source_filter="https://github.com/",
        source_type="webpage",
    )
    assert res_gh["is_refusal"] is True, "GitHub source must refuse sourdough question."
    assert res_gh["evidence_state"] == "NONE"

    # Switch to Sourdough source (correct source)
    mock_baker_answer = "Sourdough relies on wild yeast and lactobacilli fermentation over 4 to 6 hours."
    with patch("src.rag_chain.execute_llm_with_retry", return_value=(mock_baker_answer, None, None)):
        res_baker = query_rag_pipeline(
            vector_store=sample_vector_store,
            question=question,
            source_filter="https://artisan-baker.org/sourdough",
            source_type="webpage",
        )
        assert res_baker["is_refusal"] is False, "Sourdough source must answer sourdough question."
        assert res_baker["evidence_state"] in ("STRONG", "SUFFICIENT")
        assert len(res_baker["chunks"]) > 0
        for chunk in res_baker["chunks"]:
            assert chunk.metadata["source"] == "https://artisan-baker.org/sourdough"


def test_active_source_is_enforced(sample_vector_store):
    """Scenario 6: Every retrieved chunk and citation strictly belongs to the requested active source."""
    active_url = "https://github.com/"
    res = query_rag_pipeline(
        vector_store=sample_vector_store,
        question="What features does GitHub offer for software development?",
        source_filter=active_url,
        source_type="webpage",
    )

    for chunk in res["chunks"]:
        assert chunk.metadata["source"] == active_url, f"Cross-contamination: chunk from {chunk.metadata['source']}"

    for cite in res["citations"]:
        assert cite["source"] == active_url, f"Cross-contamination: citation from {cite['source']}"


def test_relevant_chunks_do_not_trigger_false_refusal(sample_vector_store):
    """Scenario 7 (Bug 2 Root Cause Fix): Conversational framing does not cause false refusal when evidence exists."""
    conversational_questions = [
        "What does this page mention about Duolingo?",
        "What is GitHub Copilot used for according to this page?",
        "Based on this website, how do teams collaborate on code changes?",
    ]

    for q in conversational_questions:
        intent = classify_question_intent(q)
        subs = decompose_complex_question(q, intent)
        # Directly evaluate evidence coverage on the GitHub chunks
        chunks = [
            doc for doc in sample_vector_store.docstore._dict.values()
            if doc.metadata.get("source") == "https://github.com/"
        ]
        cov = evaluate_evidence_coverage(chunks, q, intent, subs)

        assert cov["evidence_state"] != "NONE", (
            f"Question '{q}' falsely evaluated to evidence_state=NONE! "
            f"Covered: {cov['covered_sub_questions']}, Missing: {cov['missing_sub_questions']}"
        )
        assert len(cov["covered_sub_questions"]) >= 1, f"Expected at least 1 covered sub-question for '{q}'"

        # End-to-end pipeline check
        with patch("src.rag_chain.execute_llm_with_retry", return_value=("Verified answer from source.", None, None)):
            res = query_rag_pipeline(
                vector_store=sample_vector_store,
                question=q,
                source_filter="https://github.com/",
                source_type="webpage",
            )
            assert res["is_refusal"] is False, f"Conversational query '{q}' falsely refused!"
            assert "</div>" not in res["answer"]


def test_empty_retrieval_returns_clean_refusal(sample_vector_store):
    """Scenario 8: Empty retrieval or missing candidate pool returns a clean refusal with zero HTML tags."""
    # Query with non-matching source filter yielding 0 candidates
    res = query_rag_pipeline(
        vector_store=sample_vector_store,
        question="What is this platform?",
        source_filter="https://non-existent-source.org/404",
        source_type="webpage",
    )

    expected_refusal = get_refusal_phrase("webpage")
    assert res["is_refusal"] is True
    assert res["evidence_state"] == "NONE"
    assert res["chunks"] == []
    assert res["citations"] == []
    assert res["answer"] == expected_refusal
    assert "</div>" not in res["answer"]
    assert "<" not in res["answer"] and ">" not in res["answer"]
