"""PDF processing module using PyMuPDF and LangChain text splitters."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, BinaryIO, Optional
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.source_registry import compute_content_hash


class PDFProcessingError(Exception):
    """Custom exception raised for PDF processing failures."""
    pass


def extract_pages_from_pdf(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: str = "document.pdf",
    content_hash: Optional[str] = None,
) -> List[Document]:
    """
    Extract text page by page from a PDF file using PyMuPDF.
    Gracefully handles empty and 0-byte files without low-level exceptions.

    Args:
        file_source: File path, file-like object, or raw bytes.
        filename: Name of the source file for metadata citations.
        content_hash: Optional pre-computed SHA-256 content hash.

    Returns:
        List of LangChain Document objects with page text and metadata.
    """
    documents: List[Document] = []
    computed_hash = content_hash

    try:
        raw_bytes: bytes = b""
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                raise PDFProcessingError(f"File not found: {path}")
            if path.stat().st_size == 0:
                return []
            if not filename or filename == "document.pdf":
                filename = path.name
            raw_bytes = path.read_bytes()
        elif isinstance(file_source, bytes):
            if len(file_source) == 0:
                return []
            raw_bytes = file_source
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            if len(content) == 0:
                return []
            if hasattr(file_source, "name") and file_source.name:
                filename = file_source.name
            raw_bytes = content
        else:
            raise PDFProcessingError(f"Unsupported file source type: {type(file_source)}")

        if not computed_hash:
            computed_hash = compute_content_hash(raw_bytes)

        fname_lower = filename.lower()

        # Support Plain Text & Markdown Documents (.txt, .md, .text)
        if fname_lower.endswith((".txt", ".md", ".text")):
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            if not text:
                return []
            doc_metadata: Dict[str, Any] = {
                "source": filename,
                "source_type": "document",
                "title": filename,
                "page": 1,
                "total_pages": 1,
            }
            if computed_hash:
                doc_metadata["content_hash"] = computed_hash
            return [Document(page_content=text, metadata=doc_metadata)]

        # Support Microsoft Word Documents (.docx)
        if fname_lower.endswith(".docx"):
            import zipfile
            import xml.etree.ElementTree as ET
            import io
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                    xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in tree.iter(f"{{{ns['w']}}}p"):
                    texts = [node.text for node in p.iter(f"{{{ns['w']}}}t") if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                docx_text = "\n\n".join(paragraphs).strip()
                if not docx_text:
                    return []
                doc_metadata = {
                    "source": filename,
                    "source_type": "document",
                    "title": filename,
                    "page": 1,
                    "total_pages": 1,
                }
                if computed_hash:
                    doc_metadata["content_hash"] = computed_hash
                return [Document(page_content=docx_text, metadata=doc_metadata)]
            except Exception as docx_err:
                raise PDFProcessingError(f"Failed to extract DOCX text from {filename}: {str(docx_err)}") from docx_err

        # Standard PDF Processing via PyMuPDF
        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        total_pages = doc.page_count
        if total_pages == 0:
            doc.close()
            return []

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            text = page.get_text("text") or ""
            cleaned_text = text.strip()

            if cleaned_text:
                doc_metadata = {
                    "source": filename,
                    "source_type": "pdf",
                    "title": filename,
                    "page": page_idx + 1,  # 1-indexed page number
                    "total_pages": total_pages,
                }
                if computed_hash:
                    doc_metadata["content_hash"] = computed_hash

                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata=doc_metadata,
                    )
                )

        doc.close()
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Failed to extract text from {filename}: {str(exc)}") from exc

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split extracted page documents into overlapping chunks while preserving and standardizing metadata.

    Args:
        documents: List of page-level Documents.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Enrich metadata with chunk identifiers and enforce contract backward compatibility
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown.pdf"
        if "source_type" not in chunk.metadata:
            chunk.metadata["source_type"] = "pdf"
        if "title" not in chunk.metadata:
            chunk.metadata["title"] = chunk.metadata.get("source", "Untitled Document")
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1 if chunk.metadata.get("source_type") in ("pdf", "document") else None

    return chunks


def process_pdf_files(
    file_sources: List[Union[Tuple[Any, str], Tuple[Any, str, str]]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Batch process multiple uploaded PDF files into chunked documents.

    Args:
        file_sources: List of tuples (file_source, filename) or (file_source, filename, content_hash).
        chunk_size: Character chunk size.
        chunk_overlap: Character overlap.

    Returns:
        Tuple of (list_of_chunked_documents, statistics_dictionary).
    """
    all_chunks: List[Document] = []
    stats: Dict[str, Any] = {
        "processed_files": [],
        "empty_files": [],
        "errors": [],
        "total_pages": 0,
        "total_chunks": 0,
    }

    for item in file_sources:
        if len(item) == 3:
            file_source, fname, content_hash = item
        else:
            file_source, fname = item
            content_hash = None

        try:
            pages = extract_pages_from_pdf(file_source, filename=fname, content_hash=content_hash)
            if not pages:
                stats["empty_files"].append(fname)
                continue

            file_pages_count = len(pages)
            stats["total_pages"] += file_pages_count

            chunks = chunk_documents(
                pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_chunks.extend(chunks)

            stats["processed_files"].append(
                {
                    "filename": fname,
                    "pages": file_pages_count,
                    "chunks": len(chunks),
                    "content_hash": content_hash or (pages[0].metadata.get("content_hash") if pages else ""),
                }
            )
        except Exception as e:
            stats["errors"].append({"filename": fname, "error": str(e)})

    stats["total_chunks"] = len(all_chunks)
    return all_chunks, stats
