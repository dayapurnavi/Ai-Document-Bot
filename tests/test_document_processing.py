"""Tests for PDF and multi-format document processing (.pdf, .txt, .md, .docx)."""

import io
import zipfile
import pytest
import pymupdf
from langchain_core.documents import Document

from src.pdf_processor import (
    extract_pages_from_pdf,
    process_pdf_files,
    chunk_documents,
    PDFProcessingError,
)
from src.vector_store import (
    build_vector_store,
    retrieve_relevant_chunks,
)
from src.rag_chain import (
    get_refusal_phrase,
    format_context,
    extract_citations,
)


def create_sample_pdf(text: str = "DocuMind enterprise document intelligence workspace.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def create_sample_docx(text: str = "Word documents contain structured paragraphs and sections.") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '<w:body>\n'
            '<w:p><w:r><w:t>' + text + '</w:t></w:r></w:p>\n'
            '</w:body>\n'
            '</w:document>'
        )
        z.writestr("word/document.xml", xml_content)
    return buf.getvalue()


def test_pdf_extraction():
    pdf_bytes = create_sample_pdf("This is a verified PDF document.")
    docs = extract_pages_from_pdf(pdf_bytes, filename="test.pdf")
    assert len(docs) == 1
    assert "verified PDF document" in docs[0].page_content
    assert docs[0].metadata["source"] == "test.pdf"
    assert docs[0].metadata["source_type"] == "pdf"
    assert docs[0].metadata["page"] == 1


def test_txt_extraction():
    txt_bytes = b"Python is used for machine learning and natural language processing."
    docs = extract_pages_from_pdf(txt_bytes, filename="notes.txt")
    assert len(docs) == 1
    assert "machine learning" in docs[0].page_content
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["source_type"] == "document"
    assert docs[0].metadata["page"] == 1


def test_md_extraction():
    md_bytes = b"# Release Notes\nDocuMind AI supports document Q&A exclusively."
    docs = extract_pages_from_pdf(md_bytes, filename="readme.md")
    assert len(docs) == 1
    assert "Release Notes" in docs[0].page_content
    assert docs[0].metadata["source"] == "readme.md"
    assert docs[0].metadata["source_type"] == "document"


def test_docx_extraction():
    docx_bytes = create_sample_docx("DocuMind processes Microsoft Word documents without dependencies.")
    docs = extract_pages_from_pdf(docx_bytes, filename="report.docx")
    assert len(docs) == 1
    assert "Microsoft Word" in docs[0].page_content
    assert docs[0].metadata["source"] == "report.docx"
    assert docs[0].metadata["source_type"] == "document"


def test_batch_process_documents():
    sources = [
        (create_sample_pdf("PDF content section."), "doc1.pdf"),
        (b"Plain text document content.", "doc2.txt"),
        (b"# Markdown document content.", "doc3.md"),
        (create_sample_docx("Docx word content."), "doc4.docx"),
    ]
    chunks, stats = process_pdf_files(sources)
    assert len(stats["processed_files"]) == 4
    assert len(stats["empty_files"]) == 0
    assert len(stats["errors"]) == 0
    assert len(chunks) == 4

    vs = build_vector_store(chunks)
    assert vs.index.ntotal == 4

    retrieved = retrieve_relevant_chunks(vs, "PDF content section", top_k=2)
    assert len(retrieved) > 0
    assert retrieved[0].metadata["source"] == "doc1.pdf"


def test_document_refusal_phrase():
    assert get_refusal_phrase("pdf") == "I couldn't find that information in the indexed PDF content."
    assert get_refusal_phrase("document") == "I couldn't find that information in the indexed document content."
    assert get_refusal_phrase(None) == "I couldn't find this information in the uploaded documents."
    assert get_refusal_phrase("default") == "I couldn't find this information in the uploaded documents."


def test_citations_document_formatting():
    chunks = [
        Document(page_content="Content from PDF", metadata={"source": "annual_report.pdf", "source_type": "pdf", "page": 3, "chunk_id": 0}),
        Document(page_content="Content from TXT", metadata={"source": "notes.txt", "source_type": "document", "page": 1, "chunk_id": 1}),
    ]
    citations = extract_citations(chunks)
    assert len(citations) == 2
    assert citations[0]["formatted"] == "annual_report.pdf — Page 3"
    assert citations[1]["formatted"] == "notes.txt — Page 1"
