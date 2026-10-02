"""Comprehensive test suite for YouTube RAG integration, transcript processing,
active source isolation, navigation filtering, and timestamp citations.
"""

from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from src.youtube_processor import (
    detect_source_type,
    extract_youtube_video_id,
    normalize_youtube_url,
    is_youtube_navigation_content,
    format_timestamp,
    chunk_transcript_segments,
    process_youtube_url,
    fetch_youtube_metadata,
)
from src.rag_chain import (
    query_rag_pipeline,
    format_context,
    extract_citations,
    get_refusal_phrase,
    REFUSAL_PHRASES,
)
from src.vector_store import (
    build_vector_store,
    add_documents_to_vector_store,
    retrieve_relevant_chunks,
)
from src.embeddings import get_embedding_model
from src.structured_analysis import (
    generate_structured_website_analysis,
    detect_and_extract_charts,
)


YT_URL_1 = "https://www.youtube.com/watch?v=MWQfz4i8OWw"
YT_URL_2 = "https://youtu.be/dQw4w9WgXcQ"
WEB_URL = "https://example.com/company"


# TEST 1: Source type detection
def test_detect_source_type():
    assert detect_source_type("https://www.youtube.com/watch?v=MWQfz4i8OWw") == "youtube"
    assert detect_source_type("https://youtu.be/MWQfz4i8OWw") == "youtube"
    assert detect_source_type("https://m.youtube.com/watch?v=MWQfz4i8OWw") == "youtube"
    assert detect_source_type("https://www.youtube.com/shorts/MWQfz4i8OWw") == "youtube"
    assert detect_source_type("https://www.youtube.com/embed/MWQfz4i8OWw") == "youtube"
    assert detect_source_type("https://example.com/research") == "webpage"
    assert detect_source_type("sample_report.pdf") == "pdf"
    assert detect_source_type("pdf:document.pdf") == "pdf"


# TEST 2: Video ID extraction
def test_extract_youtube_video_id():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=MWQfz4i8OWw") == "MWQfz4i8OWw"
    assert extract_youtube_video_id("https://youtu.be/MWQfz4i8OWw?t=120s") == "MWQfz4i8OWw"
    assert extract_youtube_video_id("https://www.youtube.com/shorts/MWQfz4i8OWw") == "MWQfz4i8OWw"
    assert extract_youtube_video_id("https://www.youtube.com/embed/MWQfz4i8OWw") == "MWQfz4i8OWw"
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=MWQfz4i8OWw&feature=share") == "MWQfz4i8OWw"
    assert extract_youtube_video_id("https://example.com/video") is None


# TEST 3: URL Normalization
def test_normalize_youtube_url():
    norm = normalize_youtube_url("https://youtu.be/MWQfz4i8OWw?t=30s&si=abc")
    assert norm == "https://www.youtube.com/watch?v=MWQfz4i8OWw"


# TEST 4: Timestamp formatting
def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(134) == "02:14"
    assert format_timestamp(3665) == "01:01:05"


# TEST 5 & 6: Transcript converted into Documents with source_type=youtube
def test_transcript_chunking_and_metadata():
    entries = [
        {"text": "Once upon a time in a lovely village,", "start": 0.0, "duration": 4.0},
        {"text": "there lived a little girl named Karen.", "start": 4.5, "duration": 3.5},
        {"text": "She loved red shoes more than anything in the world.", "start": 8.5, "duration": 5.0},
    ]

    docs = chunk_transcript_segments(
        transcript_entries=entries,
        video_title="The Red Shoes",
        video_id="MWQfz4i8OWw",
        canonical_url=YT_URL_1,
        target_words=10,
        overlap_words=2,
    )

    assert len(docs) >= 1
    for d in docs:
        assert d.metadata["source_type"] == "youtube"
        assert d.metadata["source"] == YT_URL_1
        assert d.metadata["video_id"] == "MWQfz4i8OWw"
        assert "timestamp_start" in d.metadata
        assert "timestamp_end" in d.metadata
        assert "timestamp_formatted" in d.metadata
        assert "timestamp_url" in d.metadata
        assert "&t=" in d.metadata["timestamp_url"]


# TEST 7 & 8: YouTube navigation text is never indexed as content
def test_youtube_navigation_exclusion():
    nav_text_1 = "About Press Copyright Contact us Creators Advertise Developers Terms Privacy Policy & Safety"
    nav_text_2 = "© 2026 Google LLC How YouTube works Test new features"
    real_text = "Tia and Tofu present the story of the magic red shoes and Karen."

    assert is_youtube_navigation_content(nav_text_1) is True
    assert is_youtube_navigation_content(nav_text_2) is True
    assert is_youtube_navigation_content(real_text) is False


# TEST 9: YouTube Fallback when transcript is unavailable
def test_youtube_fallback_without_transcript():
    with patch("src.youtube_processor.fetch_youtube_metadata") as mock_meta, \
         patch("src.youtube_processor.fetch_youtube_transcript") as mock_trans:
        mock_meta.return_value = {
            "title": "Documentary Video",
            "author": "Science Channel",
            "author_url": "https://www.youtube.com/@science",
            "description": "An in-depth look at quantum computing.",
            "video_id": "MWQfz4i8OWw",
        }
        mock_trans.return_value = ([], False, None)

        docs, stats = process_youtube_url(YT_URL_1)

        assert len(docs) == 1
        assert stats["has_transcript"] is False
        assert docs[0].metadata["has_transcript"] is False
        assert "quantum computing" in docs[0].page_content.lower()
        # Links should have Watch Video and Channel, but zero platform navigation
        link_texts = [lnk["text"] for lnk in stats["links"]]
        assert any("Watch Video" in t for t in link_texts)
        assert any("Channel" in t for t in link_texts)
        assert not any("Terms" in t or "About" in t or "Press" in t for t in link_texts)


# TEST 10 & 11: YouTube source embedding and source-scoped retrieval
def test_youtube_embedding_and_retrieval():
    docs_yt = [
        Document(
            page_content="The main story is about Karen and her magic red shoes given by Tia and Tofu.",
            metadata={
                "source": YT_URL_1,
                "source_type": "youtube",
                "title": "The Red Shoes",
                "video_id": "MWQfz4i8OWw",
                "timestamp_start": 10.0,
                "timestamp_end": 25.0,
                "timestamp_formatted": "00:10-00:25",
                "timestamp_url": f"{YT_URL_1}&t=10s",
                "chunk_id": 0,
            },
        ),
        Document(
            page_content="At the end of the story, Karen learns humility and finds peace.",
            metadata={
                "source": YT_URL_1,
                "source_type": "youtube",
                "title": "The Red Shoes",
                "video_id": "MWQfz4i8OWw",
                "timestamp_start": 300.0,
                "timestamp_end": 320.0,
                "timestamp_formatted": "05:00-05:20",
                "timestamp_url": f"{YT_URL_1}&t=300s",
                "chunk_id": 1,
            },
        ),
    ]

    docs_web = [
        Document(
            page_content="Example Corp provides cloud enterprise database solutions.",
            metadata={
                "source": WEB_URL,
                "source_type": "url",
                "title": "Example Corp",
                "heading": "Cloud Services",
                "chunk_id": 0,
            },
        )
    ]

    embeddings = get_embedding_model()
    vs = build_vector_store(docs_yt + docs_web, embeddings)

    # 1. Scoped retrieval for YouTube
    retrieved_yt = retrieve_relevant_chunks(
        vector_store=vs,
        query="Who are the main characters in the story?",
        top_k=2,
        source_filter=YT_URL_1,
    )
    assert len(retrieved_yt) >= 1
    assert all(d.metadata["source"] == YT_URL_1 for d in retrieved_yt)
    assert "Karen" in retrieved_yt[0].page_content

    # 2. Scoped retrieval for Webpage
    retrieved_web = retrieve_relevant_chunks(
        vector_store=vs,
        query="What cloud services are provided?",
        top_k=2,
        source_filter=WEB_URL,
    )
    assert len(retrieved_web) >= 1
    assert all(d.metadata["source"] == WEB_URL for d in retrieved_web)


# TEST 12: Question about URL A cannot retrieve URL B when URL B is active
def test_active_source_isolation_between_youtube_and_web():
    docs_yt = [
        Document(
            page_content="Karen dances uncontrollably in the red shoes.",
            metadata={"source": YT_URL_1, "source_type": "youtube", "title": "The Red Shoes", "chunk_id": 0},
        )
    ]
    docs_web = [
        Document(
            page_content="Quarterly revenue reached 50 million dollars.",
            metadata={"source": WEB_URL, "source_type": "url", "title": "Financial Report", "chunk_id": 0},
        )
    ]

    embeddings = get_embedding_model()
    vs = build_vector_store(docs_yt + docs_web, embeddings)

    # Ask about Karen while WEB_URL is active
    retrieved_while_web_active = retrieve_relevant_chunks(
        vector_store=vs,
        query="What happened to Karen in the story?",
        top_k=5,
        source_filter=WEB_URL,
    )
    # MUST NOT retrieve any chunks from YT_URL_1
    assert all(d.metadata["source"] == WEB_URL for d in retrieved_while_web_active)
    assert not any("Karen" in d.page_content for d in retrieved_while_web_active)


# TEST 13 & 14: Source-specific refusal messages
def test_source_specific_refusal():
    assert get_refusal_phrase("youtube") == "I couldn't find that information in the indexed video transcript or available video metadata."
    assert get_refusal_phrase("webpage") == "I couldn't find that information in the indexed webpage content."
    assert get_refusal_phrase("url") == "I couldn't find that information in the indexed webpage content."
    assert get_refusal_phrase("pdf") == "I couldn't find that information in the indexed PDF content."
    assert get_refusal_phrase(None) == "I couldn't find this information in the uploaded documents."


# TEST 15: Timestamp citation formatting and clickable URLs
def test_youtube_timestamp_citations():
    chunks = [
        Document(
            page_content="Karen put on the red shoes and began to dance.",
            metadata={
                "source": YT_URL_1,
                "source_type": "youtube",
                "title": "The Red Shoes",
                "video_id": "MWQfz4i8OWw",
                "timestamp_formatted": "02:14-02:38",
                "timestamp_url": "https://www.youtube.com/watch?v=MWQfz4i8OWw&t=134s",
            },
        )
    ]

    citations = extract_citations(chunks)
    assert len(citations) == 1
    c = citations[0]
    assert c["source_type"] == "youtube"
    assert c["timestamp_formatted"] == "02:14-02:38"
    assert c["timestamp_url"] == "https://www.youtube.com/watch?v=MWQfz4i8OWw&t=134s"
    assert c["formatted"] == "The Red Shoes — 02:14-02:38"


# TEST 16 & 17: Zero fake charts for narrative story
def test_zero_fake_charts_for_fairy_tale():
    story_chunks = [
        Document(
            page_content="Once upon a time, a young girl named Karen lived with her mother in a quiet forest village.",
            metadata={"source": YT_URL_1, "source_type": "youtube", "title": "The Red Shoes"},
        ),
        Document(
            page_content="She was gifted red shoes by an old shoemaker and danced through the green meadows.",
            metadata={"source": YT_URL_1, "source_type": "youtube", "title": "The Red Shoes"},
        ),
    ]

    charts = detect_and_extract_charts(chunks=story_chunks, source_url=YT_URL_1)
    # Story has NO numerical data tables or stats; must return empty list
    assert charts == []
