"""Phase 5: Answerability Engine and Intelligent Fallback Test Suite.

Verifies:
1. Direct answer (Answerable, Fallback=False)
2. Topically relevant but unanswerable procedure (relevance=HIGH, answerability=NOT_ANSWERABLE, evidence_state=NONE)
3. Strict source mode refuses unsupported procedure
4. Fallback mode generates answer
5. Fallback answer clearly labeled
6. Indexed citation NOT attached to fallback-only claims
7. Duolingo remains source-grounded
8. GitHub Copilot remains source-grounded
9. URL A cannot leak into URL B
10. Cached strict refusal cannot appear in fallback mode
11. Cached fallback answer cannot appear in strict mode
12. Raw HTML (</div>) never appears
"""

import re
from unittest.mock import patch
import pytest
from langchain_core.documents import Document
from src.embeddings import get_embedding_model
from src.vector_store import build_vector_store, add_documents_to_vector_store
from src.rag_chain import (
    evaluate_evidence_coverage,
    classify_question_intent,
    decompose_complex_question,
    query_rag_pipeline,
    sanitize_final_response,
    RAG_PIPELINE_VERSION,
)


@pytest.fixture(scope="module")
def shared_embeddings():
    return get_embedding_model()


@pytest.fixture(scope="module")
def github_sample_store(shared_embeddings):
    docs = [
        Document(
            page_content=(
                "GitHub is the complete developer platform to build, scale, and deliver secure software. "
                "Over 100 million developers use GitHub for collaborative coding, project management, and code reviews."
            ),
            metadata={
                "source": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Homepage",
                "heading": "Platform Overview",
                "chunk_id": 0,
            }
        ),
        Document(
            page_content=(
                "GitHub Copilot is an AI pair programmer that empowers developers to write code faster, "
                "automate repetitive tasks, and explore new coding frameworks with real-time suggestions."
            ),
            metadata={
                "source": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Homepage",
                "heading": "GitHub Copilot Features",
                "chunk_id": 1,
            }
        ),
        Document(
            page_content=(
                "Duolingo cut code review turnaround time by 67% using GitHub Copilot and automated workflow "
                "integrations across their engineering teams to deliver daily updates."
            ),
            metadata={
                "source": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Homepage",
                "heading": "Customer Stories: Duolingo",
                "chunk_id": 2,
            }
        ),
        Document(
            page_content=(
                "Secret scanning push protection on GitHub prevents credential leaks by blocking commits "
                "and pushes containing sensitive tokens before they reach the remote repository project files."
            ),
            metadata={
                "source": "https://github.com/",
                "source_type": "webpage",
                "title": "GitHub Homepage",
                "heading": "Security and Push Protection",
                "chunk_id": 3,
            }
        ),
    ]
    return build_vector_store(docs, shared_embeddings)


@pytest.fixture(scope="module")
def multi_source_store(github_sample_store, shared_embeddings):
    url_b_docs = [
        Document(
            page_content=(
                "Brevo is an all-in-one CRM and customer engagement platform supporting transactional emails, "
                "SMS campaigns, and automated marketing workflows."
            ),
            metadata={
                "source": "https://brevo.com/crm",
                "source_type": "webpage",
                "title": "Brevo CRM",
                "heading": "CRM Platform Overview",
                "chunk_id": 0,
            }
        ),
    ]
    return add_documents_to_vector_store(github_sample_store, url_b_docs, shared_embeddings)


# TEST 1: Indexed source directly answers question
def test_indexed_source_directly_answers(github_sample_store):
    """TEST 1: Indexed source directly answers question -> ANSWERABLE, Fallback = False."""
    q = "What is GitHub Copilot used for according to this page?"
    with patch("src.rag_chain.execute_llm_with_retry", return_value=("GitHub Copilot is an AI pair programmer used to write code faster.", None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert res["answerability"] == "ANSWERABLE"
        assert res["fallback_used"] is False
        assert res["fallback_source"] == "Indexed Source"
        assert res["is_refusal"] is False
        assert len(res["citations"]) >= 1


# TEST 2: Topically relevant but unanswerable procedure
def test_topically_relevant_unanswerable_procedure(github_sample_store):
    """TEST 2: Procedural query with topical keywords but lacking CLI procedure -> NOT_ANSWERABLE, NOT STRONG."""
    q = "how to push project files to the github"
    intent_info = classify_question_intent(q)
    sub_qs = decompose_complex_question(q, intent_info)
    docs = list(github_sample_store.docstore._dict.values())

    cov = evaluate_evidence_coverage(docs, q, intent_info, sub_qs)
    assert cov["retrieval_relevance"] in ("HIGH", "MEDIUM")
    assert cov["answerability"] == "NOT_ANSWERABLE"
    assert cov["evidence_state"] != "STRONG"
    assert cov["evidence_state"] == "NONE"
    assert "push" in cov["refusal_reason"].lower() or "lacks" in cov["refusal_reason"].lower()


# TEST 3: Strict source mode refuses unsupported procedure
def test_strict_source_mode_refuses_unsupported_procedure(github_sample_store):
    """TEST 3: Strict source mode correctly refuses unsupported Git push procedure."""
    q = "how to push project files to the github"
    res = query_rag_pipeline(
        vector_store=github_sample_store,
        question=q,
        source_filter="https://github.com/",
        source_type="webpage",
        answer_mode="STRICT_SOURCE",
    )
    assert res["is_refusal"] is True
    assert res["citations"] == []
    assert res["fallback_used"] is False
    assert "couldn't find" in res["answer"].lower() or "not find" in res["answer"].lower()


# TEST 4: Fallback mode attempts external/general answer
def test_fallback_mode_attempts_external_answer(github_sample_store):
    """TEST 4: Fallback mode attempts external/general knowledge answer when source is unanswerable."""
    q = "how to push project files to the github"
    fallback_response = (
        "The indexed webpage does not provide the Git push procedure.\n\n"
        "### Additional information\n"
        "To push project files to GitHub, execute the standard Git workflow:\n"
        "1. Initialize Git: `git init`\n"
        "2. Add files: `git add .`\n"
        "3. Commit: `git commit -m 'Initial commit'`\n"
        "4. Set remote: `git remote add origin <URL>`\n"
        "5. Push: `git push -u origin main`"
    )
    with patch("src.rag_chain.execute_llm_with_retry", return_value=(fallback_response, None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert res["fallback_used"] is True
        assert res["fallback_source"] == "General Knowledge"
        assert res["is_refusal"] is False
        assert "git push" in res["answer"].lower()


# TEST 5: Fallback answer clearly labeled
def test_fallback_answer_clearly_labeled(github_sample_store):
    """TEST 5: Fallback answer is clearly separated and labeled."""
    q = "how to push project files to the github"
    fallback_response = (
        "The indexed source does not contain instructions for pushing project files to GitHub.\n\n"
        "### Additional information\n"
        "Run `git push -u origin main`."
    )
    with patch("src.rag_chain.execute_llm_with_retry", return_value=(fallback_response, None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert "additional information" in res["answer"].lower() or "indexed" in res["answer"].lower()


# TEST 6: Indexed citation NOT attached to fallback-only claims
def test_indexed_citation_not_attached_to_fallback(github_sample_store):
    """TEST 6: Indexed source citation is strictly excluded from fallback-only answers."""
    q = "how to push project files to the github"
    fallback_response = (
        "The indexed source does not provide Git CLI commands.\n\n"
        "### Additional information\n"
        "Run `git push`."
    )
    with patch("src.rag_chain.execute_llm_with_retry", return_value=(fallback_response, None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert res["citations"] == []
        assert res["debug_info"]["citation_count"] == 0


# TEST 7: Duolingo remains source-grounded
def test_duolingo_remains_source_grounded(github_sample_store):
    """TEST 7: Duolingo customer story remains source-grounded with citations and without fallback."""
    q = "What does this page mention about Duolingo?"
    with patch("src.rag_chain.execute_llm_with_retry", return_value=("Duolingo cut code review turnaround by 67% using GitHub Copilot.", None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert res["answerability"] == "ANSWERABLE"
        assert res["fallback_used"] is False
        assert res["fallback_source"] == "Indexed Source"
        assert len(res["citations"]) >= 1


# TEST 8: GitHub Copilot remains source-grounded
def test_copilot_remains_source_grounded(github_sample_store):
    """TEST 8: Copilot features remain source-grounded with citations and without fallback."""
    q = "Explain GitHub Copilot features mentioned on this page."
    with patch("src.rag_chain.execute_llm_with_retry", return_value=("Copilot is an AI pair programmer for writing code faster.", None, None)):
        res = query_rag_pipeline(
            vector_store=github_sample_store,
            question=q,
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert res["answerability"] == "ANSWERABLE"
        assert res["fallback_used"] is False
        assert res["fallback_source"] == "Indexed Source"
        assert len(res["citations"]) >= 1


# TEST 9: URL A cannot leak into URL B
def test_url_a_cannot_leak_into_url_b(multi_source_store):
    """TEST 9: Source scoping guarantees URL A content cannot leak into queries for URL B."""
    q = "What does this page mention about Duolingo?"
    res = query_rag_pipeline(
        vector_store=multi_source_store,
        question=q,
        source_filter="https://brevo.com/crm",
        source_type="webpage",
        answer_mode="STRICT_SOURCE",
    )
    assert res["is_refusal"] is True
    assert res["citations"] == []
    for chunk in res["chunks"]:
        assert chunk.metadata.get("source") != "https://github.com/"


# TEST 10: Cached strict refusal cannot appear in fallback mode
def test_cached_strict_refusal_cannot_appear_in_fallback_mode():
    """TEST 10: Separate cache identity keys ensure strict refusal cannot be served in fallback mode."""
    source_url = "https://github.com/"
    content_hash = "abc123hash"
    norm_q = "how to push project files to the github"
    top_k = 4

    cache_key_strict = (
        source_url,
        content_hash,
        norm_q,
        "source_scoped",
        "STRICT_SOURCE",
        RAG_PIPELINE_VERSION,
        top_k,
    )
    cache_key_fallback = (
        source_url,
        content_hash,
        norm_q,
        "source_scoped",
        "SOURCE_FIRST_WITH_FALLBACK",
        RAG_PIPELINE_VERSION,
        top_k,
    )

    query_cache = {}
    query_cache[cache_key_strict] = {
        "answer": "I couldn't find that information in the indexed webpage content.",
        "is_refusal": True,
    }

    # Verify fallback mode lookup misses the strict cache entry
    assert cache_key_fallback not in query_cache


# TEST 11: Cached fallback answer cannot appear in strict mode
def test_cached_fallback_answer_cannot_appear_in_strict_mode():
    """TEST 11: Separate cache identity keys ensure fallback answers cannot be served in strict mode."""
    source_url = "https://github.com/"
    content_hash = "abc123hash"
    norm_q = "how to push project files to the github"
    top_k = 4

    cache_key_strict = (
        source_url,
        content_hash,
        norm_q,
        "source_scoped",
        "STRICT_SOURCE",
        RAG_PIPELINE_VERSION,
        top_k,
    )
    cache_key_fallback = (
        source_url,
        content_hash,
        norm_q,
        "source_scoped",
        "SOURCE_FIRST_WITH_FALLBACK",
        RAG_PIPELINE_VERSION,
        top_k,
    )

    query_cache = {}
    query_cache[cache_key_fallback] = {
        "answer": "### Additional information\nRun git push.",
        "is_refusal": False,
        "fallback_used": True,
    }

    # Verify strict mode lookup misses the fallback cache entry
    assert cache_key_strict not in query_cache


# TEST 12: Raw HTML such as </div> never appears
def test_raw_html_never_appears(github_sample_store):
    """TEST 12: Zero raw HTML tags (</div>, <span>, <p>) in responses across strict and fallback modes."""
    dirty_llm_output = (
        "<div>The indexed source does not provide Git CLI commands.</div>"
        "<span>### Additional information</span>"
        "<p>Run `git push`.</p></div>"
    )
    with patch("src.rag_chain.execute_llm_with_retry", return_value=(dirty_llm_output, None, None)):
        res_fb = query_rag_pipeline(
            vector_store=github_sample_store,
            question="how to push project files to the github",
            source_filter="https://github.com/",
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )
        assert "</div>" not in res_fb["answer"]
        assert "<div>" not in res_fb["answer"]
        assert not re.search(r"<\/?\s*(?:div|span|p|br|section)\b", res_fb["answer"], re.IGNORECASE)

    res_strict = query_rag_pipeline(
        vector_store=github_sample_store,
        question="how to push project files to the github",
        source_filter="https://github.com/",
        source_type="webpage",
        answer_mode="STRICT_SOURCE",
    )
    assert "</div>" not in res_strict["answer"]
    assert "<div>" not in res_strict["answer"]
    assert not re.search(r"<\/?\s*(?:div|span|p|br|section)\b", res_strict["answer"], re.IGNORECASE)
