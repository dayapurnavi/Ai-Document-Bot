"""Automated test suite for URL Analysis Isolation, Structured Output, and AI Chart Validation."""

import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from src.web_processor import (
    validate_and_normalize_url,
    extract_page_links,
    extract_page_metadata,
)
from src.structured_analysis import (
    validate_chart_data,
    detect_and_extract_charts,
    generate_structured_website_analysis,
    validate_analysis_session,
)
from src.vector_store import (
    build_vector_store,
    add_documents_to_vector_store,
    retrieve_relevant_chunks,
)
from src.embeddings import get_embedding_model
from src.rag_chain import query_rag_pipeline


SAMPLE_URL_A = "https://website-a.com/docs"
SAMPLE_URL_B = "https://website-b.com/analytics"

SAMPLE_CHUNKS_A = [
    Document(
        page_content="### Overview\nWebsite A provides automated customer onboarding workflows.",
        metadata={"source": SAMPLE_URL_A, "source_type": "url", "title": "Website A Docs", "heading": "Overview", "chunk_id": 0},
    ),
    Document(
        page_content="### Authentication\nUse API Token Bearer header for all Website A calls.",
        metadata={"source": SAMPLE_URL_A, "source_type": "url", "title": "Website A Docs", "heading": "Authentication", "chunk_id": 1},
    ),
]

SAMPLE_CHUNKS_B = [
    Document(
        page_content="### Revenue Statistics\nWebsite B generated 100 in 2023, 150 in 2024, and 200 in 2025.",
        metadata={"source": SAMPLE_URL_B, "source_type": "url", "title": "Website B Analytics", "heading": "Revenue Statistics", "chunk_id": 0},
    ),
    Document(
        page_content="### Infrastructure\nWebsite B runs on global low-latency edge servers.",
        metadata={"source": SAMPLE_URL_B, "source_type": "url", "title": "Website B Analytics", "heading": "Infrastructure", "chunk_id": 1},
    ),
]


# TEST 1 & 2: Analyze URL A and URL B - verify source_url identity
def test_url_analysis_source_identity():
    stats_a = {
        "links": [{"text": "Guide A", "url": "https://website-a.com/guide"}],
        "metadata": {"author": "Author A"},
    }
    analysis_a = generate_structured_website_analysis(
        url=SAMPLE_URL_A,
        page_title="Website A Docs",
        chunks=SAMPLE_CHUNKS_A,
        web_stats=stats_a,
    )
    assert analysis_a["source_url"] == SAMPLE_URL_A
    assert analysis_a["status"] == "completed"

    stats_b = {
        "links": [{"text": "Metrics B", "url": "https://website-b.com/metrics"}],
        "metadata": {"author": "Author B"},
    }
    analysis_b = generate_structured_website_analysis(
        url=SAMPLE_URL_B,
        page_title="Website B Analytics",
        chunks=SAMPLE_CHUNKS_B,
        web_stats=stats_b,
    )
    assert analysis_b["source_url"] == SAMPLE_URL_B
    assert analysis_b["status"] == "completed"


# TEST 3, 4, 5, 6: Verify URL A data (summary, links, citations, charts) is NOT in URL B analysis
def test_url_data_isolation_between_a_and_b():
    stats_a = {
        "links": [{"text": "Link Alpha", "url": "https://website-a.com/alpha"}],
        "metadata": {"author": "Alpha Author"},
    }
    stats_b = {
        "links": [{"text": "Link Beta", "url": "https://website-b.com/beta"}],
        "metadata": {"author": "Beta Author"},
    }

    analysis_a = generate_structured_website_analysis(SAMPLE_URL_A, "Title A", SAMPLE_CHUNKS_A, stats_a)
    analysis_b = generate_structured_website_analysis(SAMPLE_URL_B, "Title B", SAMPLE_CHUNKS_B, stats_b)

    # 3. Summary isolation
    assert analysis_a["summary"] != analysis_b["summary"]
    assert "Website A" not in analysis_b["summary"]

    # 4. Links isolation
    link_urls_b = [l["url"] for l in analysis_b["links"]]
    assert "https://website-a.com/alpha" not in link_urls_b
    assert "https://website-b.com/beta" in link_urls_b

    # 5. Citations isolation
    for c in analysis_b["citations"]:
        assert c["source"] == SAMPLE_URL_B
        assert c["source"] != SAMPLE_URL_A

    # 6. Charts isolation
    for ch in analysis_b["charts"]:
        assert ch["source_url"] == SAMPLE_URL_B
        assert ch["source_url"] != SAMPLE_URL_A


# TEST 7: When URL B fails, URL A result is NOT displayed / rejected by session validator
def test_url_failure_isolation():
    failed_analysis = {
        "analysis_id": "failed-id-123",
        "source_url": SAMPLE_URL_B,
        "source_title": SAMPLE_URL_B,
        "status": "failed",
        "error": "DNS resolution failed",
        "summary": "Website analysis failed.",
        "key_points": [],
        "links": [],
        "charts": [],
        "citations": [],
    }

    # Stale result validator rejects failed analysis
    assert validate_analysis_session(failed_analysis, SAMPLE_URL_B) is False
    # If URL A was previously active, it must not match URL B
    assert validate_analysis_session({"source_url": SAMPLE_URL_A, "status": "completed"}, SAMPLE_URL_B) is False


# TEST 8: Source-scoped retrieval: When URL A and URL B coexist, scoping to URL B returns ONLY URL B
def test_source_scoped_retrieval_isolation():
    embeddings = get_embedding_model()
    # Build combined FAISS index
    all_chunks = SAMPLE_CHUNKS_A + SAMPLE_CHUNKS_B
    vs = build_vector_store(all_chunks, embeddings=embeddings)

    # Scoped to URL B
    scoped_b = retrieve_relevant_chunks(
        vector_store=vs,
        query="What is this about and what are the statistics?",
        top_k=5,
        source_filter=SAMPLE_URL_B,
    )

    assert len(scoped_b) > 0
    for chunk in scoped_b:
        assert chunk.metadata.get("source") == SAMPLE_URL_B
        assert chunk.metadata.get("source") != SAMPLE_URL_A


# TEST 9: Global cross-source retrieval still retrieves both URL A and URL B when no filter is passed
def test_global_cross_source_retrieval():
    embeddings = get_embedding_model()
    all_chunks = SAMPLE_CHUNKS_A + SAMPLE_CHUNKS_B
    vs = build_vector_store(all_chunks, embeddings=embeddings)

    # Global mode: source_filter is None
    global_results = retrieve_relevant_chunks(
        vector_store=vs,
        query="Tell me about onboarding workflows and revenue statistics",
        top_k=10,
        source_filter=None,
    )

    sources_found = {doc.metadata.get("source") for doc in global_results}
    assert SAMPLE_URL_A in sources_found
    assert SAMPLE_URL_B in sources_found


# TEST 10: PDF + URL coexistence and source-scoped isolation
def test_pdf_and_url_coexistence_and_isolation():
    embeddings = get_embedding_model()
    pdf_doc = Document(
        page_content="Quarterly performance review for Q3 shows 95% satisfaction.",
        metadata={"source": "review.pdf", "source_type": "pdf", "page": 1, "chunk_id": 0},
    )
    combined_docs = [pdf_doc] + SAMPLE_CHUNKS_B
    vs = build_vector_store(combined_docs, embeddings=embeddings)

    # Scoped retrieval for URL B must not return PDF chunk
    scoped_res = retrieve_relevant_chunks(
        vector_store=vs,
        query="performance review and statistics",
        top_k=5,
        source_filter=SAMPLE_URL_B,
    )
    for c in scoped_res:
        assert c.metadata.get("source") == SAMPLE_URL_B
        assert c.metadata.get("source") != "review.pdf"


# TEST 11: Valid numerical time-series generates verified chart
def test_chart_generation_on_numerical_data():
    charts = detect_and_extract_charts(SAMPLE_CHUNKS_B, SAMPLE_URL_B)
    assert len(charts) >= 1
    ch = charts[0]
    assert ch["source_url"] == SAMPLE_URL_B
    assert ch["chart_type"] in ("line", "bar")
    assert len(ch["data"]) >= 3
    # Check that labels are years and values are numeric
    labels = [p["label"] for p in ch["data"]]
    values = [p["value"] for p in ch["data"]]
    assert "2023" in labels
    assert 100 in values or 100.0 in values


# TEST 12: Source contains no numerical data -> NO chart generated (NO DATA = NO CHART)
def test_no_chart_when_no_numerical_data():
    non_numeric_chunks = [
        Document(
            page_content="### Architecture\nDocuMind AI uses Python, Streamlit, and Hugging Face embeddings.",
            metadata={"source": SAMPLE_URL_A, "source_type": "url", "title": "Tech Stack"},
        )
    ]
    charts = detect_and_extract_charts(non_numeric_chunks, SAMPLE_URL_A)
    assert charts == []


# TEST 13: Strict validation rejects LLM hallucinated / unverified numbers
def test_chart_validation_rejects_hallucinated_numbers():
    source_text = "Website stats: 2023 revenue was 500 million."
    # Candidate chart where 9999 was invented/hallucinated by LLM
    candidate_chart = {
        "chart_type": "bar",
        "title": "Revenue",
        "x_label": "Year",
        "y_label": "USD",
        "data": [
            {"label": "2023", "value": 500},
            {"label": "2024", "value": 9999},  # 9999 is NOT in source_text!
        ],
    }
    validated = validate_chart_data(candidate_chart, source_text=source_text, source_url=SAMPLE_URL_A)
    # Must be rejected because 9999 does not appear in source_text
    assert validated is None


# TEST 14: Chart citation points to current URL
def test_chart_citation_matches_source_url():
    charts = detect_and_extract_charts(SAMPLE_CHUNKS_B, SAMPLE_URL_B)
    assert len(charts) >= 1
    assert SAMPLE_URL_B in charts[0]["source_citation"]


# TEST 15: Link extraction extracts valid links and discards anchors/noise
def test_extract_page_links():
    sample_html = """
    <html>
    <body>
        <nav><a href="#content">Skip</a><a href="javascript:void(0)">JS</a></nav>
        <main>
            <p>Read our documentation:</p>
            <a href="/docs/start">Quick Start Guide</a>
            <a href="https://partner.com/sdk">Partner SDK</a>
            <a href="/login">Sign In</a>
            <a href="/docs/start">Quick Start Guide</a> <!-- duplicate -->
        </main>
    </body>
    </html>
    """
    links = extract_page_links(sample_html, base_url="https://example.com/app")
    extracted_urls = [l["url"] for l in links]
    # Anchors and javascript discarded
    assert not any("#" in u for u in extracted_urls)
    assert not any("javascript" in u for u in extracted_urls)
    # Valid absolute links present
    assert "https://example.com/docs/start" in extracted_urls
    assert "https://partner.com/sdk" in extracted_urls
    # Duplicates removed
    assert len([u for u in extracted_urls if u == "https://example.com/docs/start"]) == 1
