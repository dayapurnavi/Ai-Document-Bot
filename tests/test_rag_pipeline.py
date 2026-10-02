"""Automated integration tests for DocuMind AI RAG pipeline."""

import os
from pathlib import Path
import pytest
import pymupdf
from langchain_core.documents import Document

from src.pdf_processor import (
    extract_pages_from_pdf,
    chunk_documents,
    process_pdf_files,
    PDFProcessingError,
)
from src.embeddings import get_embedding_model
from src.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    clear_persisted_vector_store,
    retrieve_relevant_chunks,
    add_documents_to_vector_store,
    remove_source_from_vector_store,
    validate_vector_store,
)
from src.source_registry import (
    compute_content_hash,
    is_source_indexed,
    register_source,
    unregister_source_by_name_or_hash,
    load_source_registry,
    clear_source_registry,
)
from src.rag_chain import query_rag_pipeline, REFUSAL_PHRASE, extract_citations, format_context
from src.config import VECTORSTORE_DIR


@pytest.fixture(scope="session")
def sample_pdf_files(tmp_path_factory):
    """Create two synthetic multi-page PDF documents for testing."""
    test_dir = tmp_path_factory.mktemp("test_docs")

    # Document 1: Company Policy (2 pages)
    pdf1_path = test_dir / "Company_Policy.pdf"
    doc1 = pymupdf.open()
    p1 = doc1.new_page()
    p1.insert_text(
        pymupdf.Point(50, 72),
        "DocuMind AI Project Specification:\n"
        "DocuMind AI is an enterprise document question answering system. "
        "It utilizes Hugging Face sentence-transformers for vector embeddings "
        "and FAISS for fast similarity search."
    )
    p2 = doc1.new_page()
    p2.insert_text(
        pymupdf.Point(50, 72),
        "Remote Work Policy Guidelines:\n"
        "Employees are permitted to work remotely up to 3 days per week "
        "subject to prior approval from their department manager. "
        "Core collaboration hours are 10:00 AM to 4:00 PM."
    )
    doc1.save(str(pdf1_path))
    doc1.close()

    # Document 2: Health Benefits (1 page)
    pdf2_path = test_dir / "Health_Benefits.pdf"
    doc2 = pymupdf.open()
    p_health = doc2.new_page()
    p_health.insert_text(
        pymupdf.Point(50, 72),
        "Annual Wellness Stipend:\n"
        "All full-time staff receive a wellness stipend of 750 dollars annually "
        "for gym memberships, ergonomic equipment, and fitness classes."
    )
    doc2.save(str(pdf2_path))
    doc2.close()

    return [pdf1_path, pdf2_path]


# TEST 1: Existing PDF ingestion still passes
def test_pdf_text_extraction(sample_pdf_files):
    """Verify PyMuPDF extracts text page by page with correct metadata."""
    pdf1_path = sample_pdf_files[0]
    pages = extract_pages_from_pdf(pdf1_path, filename="Company_Policy.pdf")

    assert len(pages) == 2
    assert pages[0].metadata["source"] == "Company_Policy.pdf"
    assert pages[0].metadata["source_type"] == "pdf"
    assert pages[0].metadata["page"] == 1
    assert "DocuMind AI" in pages[0].page_content

    assert pages[1].metadata["source"] == "Company_Policy.pdf"
    assert pages[1].metadata["page"] == 2
    assert "Remote Work Policy" in pages[1].page_content


def test_chunking_preserves_metadata(sample_pdf_files):
    """Verify recursive text splitter preserves page and source metadata."""
    pdf1_path = sample_pdf_files[0]
    pages = extract_pages_from_pdf(pdf1_path, filename="Company_Policy.pdf")
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=40)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "Company_Policy.pdf"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["page"] in (1, 2)
        assert "chunk_id" in chunk.metadata


# TEST 2: Existing PDF retrieval still passes
def test_embedding_and_faiss_vector_store(sample_pdf_files, tmp_path):
    """Verify embedding generation and FAISS similarity retrieval."""
    files_to_process = [(f, f.name) for f in sample_pdf_files]
    all_chunks, stats = process_pdf_files(files_to_process, chunk_size=500, chunk_overlap=50)

    assert stats["total_chunks"] >= 3
    assert len(stats["processed_files"]) == 2

    embeddings = get_embedding_model()
    vector_store = build_vector_store(all_chunks, embeddings)
    assert vector_store is not None

    # Test retrieval for Remote Work
    results = retrieve_relevant_chunks(vector_store, "remote work days", top_k=2)
    assert len(results) > 0
    assert any("3 days" in r.page_content for r in results)
    assert results[0].metadata["source"] == "Company_Policy.pdf"
    assert results[0].metadata["page"] == 2

    # Test retrieval across multiple documents: Wellness stipend
    results_health = retrieve_relevant_chunks(vector_store, "wellness stipend amount", top_k=2)
    assert len(results_health) > 0
    assert any("750 dollars" in r.page_content for r in results_health)
    assert any(r.metadata["source"] == "Health_Benefits.pdf" for r in results_health)


# TEST 3: Existing Groq grounded answer still passes
def test_rag_grounded_answer_and_citations(sample_pdf_files):
    """Verify grounded generation and precise citations via Groq LLM."""
    files_to_process = [(f, f.name) for f in sample_pdf_files]
    chunks, _ = process_pdf_files(files_to_process)
    vector_store = build_vector_store(chunks)

    # Question 1: In-domain question from Document 1 Page 2
    response = query_rag_pipeline(
        vector_store=vector_store,
        question="How many days of remote work are employees permitted per week?",
        top_k=3,
    )
    assert response["is_refusal"] is False
    assert "3" in response["answer"]
    assert len(response["citations"]) >= 1
    top_citation = response["citations"][0]
    assert top_citation["source"] == "Company_Policy.pdf"
    assert top_citation["page"] == 2

    # Question 2: In-domain question from Document 2 Page 1
    response2 = query_rag_pipeline(
        vector_store=vector_store,
        question="What is the annual wellness stipend amount?",
        top_k=3,
    )
    assert response2["is_refusal"] is False
    assert "750" in response2["answer"]
    assert any(c["source"] == "Health_Benefits.pdf" for c in response2["citations"])


# TEST 4: Existing hallucination guard still passes
def test_rag_guardrail_refusal_on_unknown_information(sample_pdf_files):
    """Verify that questions not in the documents trigger strict refusal without hallucination."""
    files_to_process = [(f, f.name) for f in sample_pdf_files]
    chunks, _ = process_pdf_files(files_to_process)
    vector_store = build_vector_store(chunks)

    out_of_domain_query = "What is the capital city of Japan and who is the prime minister?"
    response = query_rag_pipeline(
        vector_store=vector_store,
        question=out_of_domain_query,
        top_k=3,
    )

    # Must refuse and not hallucinate facts
    assert response["is_refusal"] is True
    assert "couldn't find" in response["answer"].lower() or "could not find" in response["answer"].lower()
    assert len(response["citations"]) == 0


# TEST 5, 6, 7: Persistence save, load, and survival across restart
def test_persistence_save_load_and_restart(sample_pdf_files, tmp_path):
    """Verify saving vector store, loading vector store, and survival across reload."""
    test_vs_dir = tmp_path / "persistence_test"
    files = [(sample_pdf_files[0], "Company_Policy.pdf")]
    chunks, _ = process_pdf_files(files)

    embeddings = get_embedding_model()
    vs = build_vector_store(chunks, embeddings)
    assert validate_vector_store(vs) is True

    # TEST 5: Save
    assert save_vector_store(vs, folder_path=test_vs_dir) is True
    assert (test_vs_dir / "index.faiss").exists()
    assert (test_vs_dir / "index.pkl").exists()

    # TEST 6: Load
    loaded_vs = load_vector_store(folder_path=test_vs_dir, embeddings=embeddings)
    assert loaded_vs is not None
    assert validate_vector_store(loaded_vs) is True
    assert loaded_vs.index.ntotal == vs.index.ntotal

    # TEST 7: Query loaded store after restart simulation
    results = retrieve_relevant_chunks(loaded_vs, "remote work guidelines", top_k=1)
    assert len(results) > 0
    assert "remote" in results[0].page_content.lower()


# TEST 8, 9: Duplicate PDF detected and no vector count increase
def test_duplicate_pdf_detection_and_no_vector_increase(sample_pdf_files, tmp_path):
    """Verify duplicate PDF detection via content hash and preventing duplicate vectors."""
    content_bytes = sample_pdf_files[0].read_bytes()
    content_hash = compute_content_hash(content_bytes)

    test_vs_dir = tmp_path / "dup_test"
    register_source(
        source="Company_Policy.pdf",
        source_type="pdf",
        content_hash=content_hash,
        chunk_count=2,
        total_pages=2,
        folder_path=test_vs_dir,
    )

    # TEST 8: Duplicate detected
    assert is_source_indexed(content_hash, folder_path=test_vs_dir) is True

    # TEST 9: Build initial store, verify duplicate chunk submission is guarded
    chunks, _ = process_pdf_files([(content_bytes, "Company_Policy.pdf", content_hash)])
    vs = build_vector_store(chunks)
    initial_count = vs.index.ntotal

    # If duplicate is filtered out, count remains identical
    duplicate_candidates = [(content_bytes, "Company_Policy.pdf", content_hash)]
    filtered_new = [
        item for item in duplicate_candidates
        if not is_source_indexed(item[2], folder_path=test_vs_dir)
    ]
    assert len(filtered_new) == 0  # Successfully identified as duplicate

    # Adding nothing ensures vector count does not increase
    updated_vs = add_documents_to_vector_store(vs, [])
    assert updated_vs.index.ntotal == initial_count


# TEST 10, 11: Re-indexing replaces old source safely and leaves unrelated sources intact
def test_pdf_reindexing_and_isolation(sample_pdf_files):
    """Verify re-indexing a PDF cleans old chunks and preserves unrelated documents."""
    chunks_doc1, _ = process_pdf_files([(sample_pdf_files[0], "Company_Policy.pdf")])
    chunks_doc2, _ = process_pdf_files([(sample_pdf_files[1], "Health_Benefits.pdf")])

    embeddings = get_embedding_model()
    # Build combined index with both documents
    combined_vs = build_vector_store(chunks_doc1 + chunks_doc2, embeddings)
    total_chunks = len(chunks_doc1) + len(chunks_doc2)
    assert combined_vs.index.ntotal == total_chunks

    # TEST 10 & 11: Remove Document 1 to simulate re-index replacement
    isolated_vs = remove_source_from_vector_store(combined_vs, "Company_Policy.pdf", embeddings)
    assert isolated_vs is not None
    assert isolated_vs.index.ntotal == len(chunks_doc2)

    # Document 2 chunks remain 100% intact
    doc2_chunks_in_store = [
        d for d in isolated_vs.docstore._dict.values()
        if d.metadata.get("source") == "Health_Benefits.pdf"
    ]
    assert len(doc2_chunks_in_store) == len(chunks_doc2)

    # Document 1 chunks are completely removed
    doc1_chunks_in_store = [
        d for d in isolated_vs.docstore._dict.values()
        if d.metadata.get("source") == "Company_Policy.pdf"
    ]
    assert len(doc1_chunks_in_store) == 0


# TEST 12: 0-byte PDF handled gracefully
def test_zero_byte_pdf_handling(tmp_path):
    """Verify 0-byte PDF is skipped gracefully without exceptions."""
    zero_file = tmp_path / "zero_byte.pdf"
    zero_file.write_bytes(b"")

    # From file path
    pages = extract_pages_from_pdf(zero_file)
    assert pages == []

    # From raw bytes
    pages_bytes = extract_pages_from_pdf(b"")
    assert pages_bytes == []

    # Batch process
    chunks, stats = process_pdf_files([(b"", "empty.pdf")])
    assert len(chunks) == 0
    assert "empty.pdf" in stats["empty_files"]


# TEST 13, 14: Citation extraction for PDF and future URL compatibility
def test_citation_extraction_pdf_and_url_compatibility():
    """Verify citation extraction supports PDF with page and URL without page."""
    pdf_chunk = Document(
        page_content="Policy guideline content",
        metadata={
            "source": "Handbook.pdf",
            "source_type": "pdf",
            "page": 5,
            "chunk_id": 0,
        },
    )

    url_chunk = Document(
        page_content="API authentication guidelines",
        metadata={
            "source": "https://api.example.com/docs",
            "source_type": "url",
            "title": "API Documentation",
            "heading": "Authentication",
            "chunk_id": 1,
        },
    )

    # TEST 13: PDF Citation
    pdf_cits = extract_citations([pdf_chunk])
    assert len(pdf_cits) == 1
    assert pdf_cits[0]["source"] == "Handbook.pdf"
    assert pdf_cits[0]["source_type"] == "pdf"
    assert pdf_cits[0]["page"] == 5
    assert pdf_cits[0]["url"] is None
    assert pdf_cits[0]["formatted"] == "Handbook.pdf — Page 5"

    # TEST 14: URL Citation (handled without web fetching)
    url_cits = extract_citations([url_chunk])
    assert len(url_cits) == 1
    assert url_cits[0]["source_type"] == "url"
    assert url_cits[0]["page"] is None
    assert url_cits[0]["url"] == "https://api.example.com/docs"
    assert "Authentication" in url_cits[0]["formatted"]


# TEST 15: Backward compatibility for old PDF metadata without source_type
def test_backward_compatibility_old_metadata_without_source_type():
    """Verify legacy chunk metadata missing source_type defaults gracefully to PDF."""
    legacy_chunk = Document(
        page_content="Legacy indexed content",
        metadata={
            "source": "legacy_doc.pdf",
            "page": 3,
            "chunk_id": 0,
            # source_type is omitted
        },
    )

    # Chunk splitter backward compatibility
    standardized = chunk_documents([legacy_chunk], chunk_size=500, chunk_overlap=50)
    assert standardized[0].metadata["source_type"] == "pdf"

    # Citation extraction backward compatibility
    cits = extract_citations([legacy_chunk])
    assert len(cits) == 1
    assert cits[0]["source_type"] == "pdf"
    assert cits[0]["page"] == 3
    assert cits[0]["formatted"] == "legacy_doc.pdf — Page 3"

    # Context formatting backward compatibility
    context = format_context([legacy_chunk])
    assert "legacy_doc.pdf | Page: 3" in context
