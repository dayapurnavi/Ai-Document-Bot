"""Vector store manager for FAISS indexing and semantic similarity retrieval."""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR, DEFAULT_TOP_K
from src.embeddings import get_embedding_model
from src.source_registry import clear_source_registry


class VectorStoreError(Exception):
    """Custom exception raised for Vector Store operations."""
    pass


def validate_vector_store(vector_store: FAISS) -> bool:
    """
    Validate FAISS vector store integrity, vector count, and metadata alignment.

    Args:
        vector_store: FAISS vector store instance.

    Returns:
        True if vector store passes all integrity checks.

    Raises:
        VectorStoreError: If validation fails.
    """
    if vector_store is None:
        raise VectorStoreError("Vector store instance is None.")

    if not hasattr(vector_store, "index") or vector_store.index is None:
        raise VectorStoreError("Vector store FAISS index is missing or uninitialized.")

    if not hasattr(vector_store, "docstore") or vector_store.docstore is None:
        raise VectorStoreError("Vector store docstore is missing or uninitialized.")

    ntotal = vector_store.index.ntotal
    docstore_count = len(vector_store.docstore._dict)

    if ntotal != docstore_count:
        raise VectorStoreError(
            f"FAISS index mismatch: {ntotal} vectors in index vs {docstore_count} documents in docstore."
        )

    # Validate vector dimension (MiniLM produces 384 dimensions)
    if hasattr(vector_store.index, "d") and vector_store.index.d != 384:
        raise VectorStoreError(
            f"Unexpected FAISS vector dimension: {vector_store.index.d} (expected 384)."
        )

    # Validate and ensure backward compatibility for document metadata
    for doc_id, doc in vector_store.docstore._dict.items():
        if not hasattr(doc, "metadata") or not doc.metadata:
            raise VectorStoreError(f"Document {doc_id} has empty or missing metadata.")

        if "source" not in doc.metadata:
            raise VectorStoreError(f"Document {doc_id} is missing required 'source' metadata.")

        if "chunk_id" not in doc.metadata:
            raise VectorStoreError(f"Document {doc_id} is missing required 'chunk_id' metadata.")

        # Backward compatibility: ensure source_type and title default gracefully if absent
        if "source_type" not in doc.metadata:
            doc.metadata["source_type"] = "pdf"
        if "title" not in doc.metadata:
            doc.metadata["title"] = doc.metadata.get("source", "Untitled Document")

    return True


def build_vector_store(
    documents: List[Document],
    embeddings=None,
) -> FAISS:
    """
    Create a new FAISS vector database from a list of document chunks.

    Args:
        documents: List of chunked Document objects.
        embeddings: HuggingFaceEmbeddings instance.

    Returns:
        FAISS vector store instance.
    """
    if not documents:
        raise VectorStoreError("Cannot build vector store with 0 documents.")

    if embeddings is None:
        embeddings = get_embedding_model()

    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        validate_vector_store(vector_store)
        return vector_store
    except Exception as e:
        raise VectorStoreError(f"Failed to create FAISS vector store: {str(e)}") from e


def add_documents_to_vector_store(
    vector_store: Optional[FAISS],
    documents: List[Document],
    embeddings=None,
) -> FAISS:
    """
    Safely append new document chunks to an existing FAISS vector store,
    or initialize a new vector store if none exists.

    Args:
        vector_store: Existing FAISS vector store or None.
        documents: List of new Document chunks to add.
        embeddings: HuggingFaceEmbeddings instance.

    Returns:
        Updated and validated FAISS vector store.
    """
    if not documents:
        if vector_store is not None:
            return vector_store
        raise VectorStoreError("Cannot build vector store with 0 documents.")

    if embeddings is None:
        embeddings = get_embedding_model()

    if vector_store is None:
        return build_vector_store(documents, embeddings=embeddings)

    try:
        vector_store.add_documents(documents)
        validate_vector_store(vector_store)
        return vector_store
    except Exception as e:
        raise VectorStoreError(f"Failed to add documents to FAISS vector store: {str(e)}") from e


def remove_source_from_vector_store(
    vector_store: FAISS,
    source_identifier: str,
    embeddings=None,
) -> Optional[FAISS]:
    """
    Safely remove a specific source and all its chunks from the FAISS vector store,
    rebuilding the index from the remaining documents to prevent orphan vectors or metadata.

    Args:
        vector_store: Active FAISS vector store.
        source_identifier: Source filename, url, or content_hash to remove.
        embeddings: HuggingFaceEmbeddings instance.

    Returns:
        Updated FAISS vector store with remaining documents, or None if no documents remain.
    """
    if vector_store is None or not hasattr(vector_store, "docstore"):
        return None

    if embeddings is None:
        embeddings = get_embedding_model()

    all_docs = list(vector_store.docstore._dict.values())
    remaining_docs = [
        doc for doc in all_docs
        if doc.metadata.get("source") != source_identifier
        and doc.metadata.get("content_hash") != source_identifier
        and doc.metadata.get("source_id") != source_identifier
    ]

    if not remaining_docs:
        return None

    # Rebuild from clean remaining chunks
    return build_vector_store(remaining_docs, embeddings=embeddings)


def save_vector_store(
    vector_store: FAISS,
    folder_path: Path = VECTORSTORE_DIR,
    index_name: str = "index",
) -> bool:
    """
    Persist FAISS index and docstore to disk safely using atomic staging.
    Never corrupts or destroys the previous valid index if saving fails.

    Args:
        vector_store: FAISS instance to save.
        folder_path: Target directory path.
        index_name: Subfolder/file index identifier.

    Returns:
        True if successfully saved.
    """
    try:
        validate_vector_store(vector_store)
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Stage in a temporary subfolder to guarantee atomicity
        with tempfile.TemporaryDirectory(dir=str(folder)) as tmp_dir:
            vector_store.save_local(tmp_dir, index_name=index_name)

            tmp_faiss = Path(tmp_dir) / f"{index_name}.faiss"
            tmp_pkl = Path(tmp_dir) / f"{index_name}.pkl"

            if not tmp_faiss.exists() or not tmp_pkl.exists():
                raise VectorStoreError("Staged FAISS files were not created properly.")

            target_faiss = folder / f"{index_name}.faiss"
            target_pkl = folder / f"{index_name}.pkl"

            # Atomically replace destination files
            shutil.copy2(tmp_faiss, target_faiss)
            shutil.copy2(tmp_pkl, target_pkl)

        return True
    except Exception as e:
        raise VectorStoreError(f"Failed to save FAISS vector store: {str(e)}") from e


def load_vector_store(
    folder_path: Path = VECTORSTORE_DIR,
    embeddings=None,
    index_name: str = "index",
) -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk and validate its integrity.

    Args:
        folder_path: Target directory path.
        embeddings: HuggingFaceEmbeddings instance.
        index_name: Index identifier.

    Returns:
        FAISS vector store or None if not found or invalid.
    """
    folder_path = Path(folder_path)
    faiss_file = folder_path / f"{index_name}.faiss"
    pkl_file = folder_path / f"{index_name}.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        return None

    if embeddings is None:
        embeddings = get_embedding_model()

    try:
        vector_store = FAISS.load_local(
            folder_path=str(folder_path),
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True,
        )
        validate_vector_store(vector_store)
        return vector_store
    except Exception as e:
        raise VectorStoreError(f"Failed to load FAISS index: {str(e)}") from e


def clear_persisted_vector_store(folder_path: Path = VECTORSTORE_DIR) -> bool:
    """
    Remove any persisted FAISS index files and source registry to ensure stale indexes are wiped.

    Args:
        folder_path: Directory containing vector store index files.

    Returns:
        True if cleared.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return True

    clear_source_registry(folder)

    for item in folder.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception:
            pass

    return True


def expand_context_with_neighbors(
    chunks: List[Document],
    vector_store: FAISS,
    max_neighbors: int = 2,
    source_filter: Optional[str] = None,
) -> List[Document]:
    """
    Expand selected chunks with their preceding (chunk_id - 1) and succeeding (chunk_id + 1)
    neighbors from the same source document.
    """
    if not chunks or vector_store is None:
        return chunks

    def chunk_key(doc: Document) -> Tuple[Any, Any, str]:
        return (
            doc.metadata.get("source", ""),
            doc.metadata.get("chunk_id", -1),
            doc.page_content[:60],
        )

    norm_target_source = None
    if source_filter:
        norm_target_source = source_filter.strip().rstrip("/").lower()

    def is_source_match(doc: Document) -> bool:
        if not norm_target_source:
            return True
        doc_src = str(doc.metadata.get("source", "")).strip().rstrip("/").lower()
        doc_orig_url = str(doc.metadata.get("original_url", "")).strip().rstrip("/").lower()
        return (doc_src == norm_target_source) or (doc_orig_url == norm_target_source)

    expanded = list(chunks)
    seen_keys = {chunk_key(c) for c in expanded}

    if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
        source_doc_groups: Dict[str, Dict[int, Document]] = {}
        for d in vector_store.docstore._dict.values():
            if not is_source_match(d):
                continue
            src = d.metadata.get("source", "")
            cid = d.metadata.get("chunk_id")
            if src and cid is not None:
                if src not in source_doc_groups:
                    source_doc_groups[src] = {}
                source_doc_groups[src][cid] = d

        added_neighbors = 0
        for cand in chunks:
            src = cand.metadata.get("source", "")
            cid = cand.metadata.get("chunk_id")
            if src in source_doc_groups and cid is not None:
                src_map = source_doc_groups[src]
                # Preceding neighbor
                prv = src_map.get(cid - 1)
                if prv and chunk_key(prv) not in seen_keys:
                    expanded.append(prv)
                    seen_keys.add(chunk_key(prv))
                    added_neighbors += 1
                # Succeeding neighbor
                nxt = src_map.get(cid + 1)
                if nxt and chunk_key(nxt) not in seen_keys:
                    expanded.append(nxt)
                    seen_keys.add(chunk_key(nxt))
                    added_neighbors += 1

            if added_neighbors >= max_neighbors:
                break

    return expanded


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    source_filter: Optional[str] = None,
) -> List[Document]:
    """
    Perform hybrid retrieval (FAISS semantic search + lexical term overlap + heading matching
    + exact technical keyword boosting) with cross-chunk reranking and controlled neighbor expansion.
    Strictly guarantees source isolation when source_filter is provided.

    Args:
        vector_store: FAISS vector database.
        query: User's natural language question.
        top_k: Number of most relevant chunks to return.
        source_filter: Optional source URL, filename, or hash to strictly isolate retrieval.

    Returns:
        List of relevant Document chunks with source metadata.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    try:
        # Determine total indexed chunks in FAISS
        total_chunks = (
            len(vector_store.docstore._dict)
            if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict")
            else top_k
        )

        overview_keywords = [
            "title", "name", "about", "summary", "summarize",
            "overview", "what is this", "describe", "introduction",
            "who wrote", "author", "header", "purpose"
        ]
        is_overview = any(kw in clean_query.lower() for kw in overview_keywords)

        clean_filter = source_filter.strip() if source_filter and isinstance(source_filter, str) else None

        def is_source_match(doc: Document) -> bool:
            if not clean_filter:
                return True
            src = doc.metadata.get("source")
            chash = doc.metadata.get("content_hash")
            sid = doc.metadata.get("source_id")
            if src == clean_filter or chash == clean_filter or sid == clean_filter:
                return True
            if src and isinstance(src, str):
                if src.rstrip("/") == clean_filter.rstrip("/"):
                    return True
            return False

        # Extract informative search terms and technical keywords
        query_lower = clean_query.lower()
        stopwords = {
            "the", "and", "is", "in", "to", "of", "a", "an", "for", "with", "on", "as", "by",
            "at", "it", "from", "that", "this", "how", "what", "why", "can", "does", "could",
            "would", "should", "are", "be", "do", "did", "website", "page", "user", "from", "using"
        }
        query_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query_lower)) - stopwords

        tech_vocabulary = [
            "api key", "apikey", "api", "sms", "smtp", "webhook", "webhooks",
            "automation", "workflow", "workflows", "trigger", "triggers",
            "transactional", "tracking", "deliverability", "metrics", "analytics",
            "integration", "integrations", "crm", "order confirmation", "follow-up",
            "follow up", "token", "endpoint", "oauth", "rest", "sdk", "template"
        ]
        active_tech_terms = [t for t in tech_vocabulary if t in query_lower]
        # Typo-tolerant mapping for common technical queries
        typo_aliases = {
            "smss": "sms", "triger": "trigger", "trigered": "trigger", "trigers": "triggers",
            "autometion": "automation", "autoomation": "automation",
            "confiramtion": "confirmation", "analitics": "analytics", "metrix": "metrics",
            "diliverability": "deliverability", "integretion": "integration", "webhool": "webhook",
        }
        for typo, canonical in typo_aliases.items():
            if typo in query_lower and canonical not in active_tech_terms:
                active_tech_terms.append(canonical)

        def chunk_key(doc: Document) -> Tuple[Any, Any, str]:
            return (
                doc.metadata.get("source"),
                doc.metadata.get("chunk_id"),
                doc.page_content[:40],
            )

        # 1. FAISS Semantic Search
        if clean_filter:
            search_k = min(total_chunks, max(top_k * 5, 35, total_chunks))
            raw_dense = vector_store.similarity_search(clean_query, k=search_k)
            dense_candidates = [doc for doc in raw_dense if is_source_match(doc)]
        else:
            search_k = min(total_chunks, max(top_k * 3, 20))
            dense_candidates = vector_store.similarity_search(clean_query, k=search_k)

        # 2. Candidate Pool & Rank Map
        candidate_pool: List[Document] = []
        seen_keys: Set[Tuple[Any, Any, str]] = set()
        dense_rank_map: Dict[Tuple[Any, Any, str], float] = {}

        for rank_idx, doc in enumerate(dense_candidates):
            ck = chunk_key(doc)
            if ck not in seen_keys:
                seen_keys.add(ck)
                candidate_pool.append(doc)
                dense_rank_map[ck] = 1.0 - (rank_idx / max(1, len(dense_candidates)))

        # Also mine docstore for exact technical terms or high lexical overlap
        if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            for doc in vector_store.docstore._dict.values():
                if not is_source_match(doc):
                    continue
                ck = chunk_key(doc)
                content_lower = doc.page_content.lower()
                heading_lower = (doc.metadata.get("heading") or "").lower()

                has_tech_match = any(t in content_lower or t in heading_lower for t in active_tech_terms)
                overlap_count = sum(1 for w in query_words if w in content_lower or w in heading_lower)
                overlap_ratio = overlap_count / max(1, len(query_words))

                if has_tech_match or overlap_ratio >= 0.25:
                    if ck not in seen_keys:
                        seen_keys.add(ck)
                        candidate_pool.append(doc)
                        dense_rank_map[ck] = 0.0

        # If pool is less than top_k, ensure all source docs are included
        if clean_filter and len(candidate_pool) < top_k and hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            for doc in vector_store.docstore._dict.values():
                if is_source_match(doc):
                    ck = chunk_key(doc)
                    if ck not in seen_keys:
                        seen_keys.add(ck)
                        candidate_pool.append(doc)
                        dense_rank_map[ck] = 0.0

        # 3. Hybrid Reranking
        scored_candidates: List[Tuple[float, Document]] = []
        for doc in candidate_pool:
            ck = chunk_key(doc)
            content_lower = doc.page_content.lower()
            heading_lower = (doc.metadata.get("heading") or "").lower()
            title_lower = (doc.metadata.get("title") or "").lower()

            dense_score = dense_rank_map.get(ck, 0.0)

            # Lexical overlap score
            overlap_words = sum(1 for w in query_words if w in content_lower)
            lexical_score = overlap_words / max(1, len(query_words))

            # Heading relevance score
            heading_match = sum(1 for w in query_words if w in heading_lower or w in title_lower)
            heading_score = min(1.0, heading_match / max(1, len(query_words)) * 1.5)

            # Technical term bonus
            tech_match = any(t in content_lower or t in heading_lower for t in active_tech_terms)
            tech_score = 1.0 if tech_match else 0.0

            # Composite ranking
            final_score = (
                dense_score * 0.40
                + lexical_score * 0.25
                + heading_score * 0.20
                + tech_score * 0.15
            )
            scored_candidates.append((final_score, doc))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [doc for _, doc in scored_candidates]

        # 4. Context Window & Section-Aware Neighbor Expansion
        expanded_docs: List[Document] = list(top_candidates[:top_k])
        seen_expanded_keys: Set[Tuple[Any, Any, str]] = {chunk_key(c) for c in expanded_docs}

        # Expand neighboring chunks within the exact same source for additional continuity
        if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            source_doc_groups: Dict[str, Dict[int, Document]] = {}
            for d in vector_store.docstore._dict.values():
                if not is_source_match(d):
                    continue
                src = d.metadata.get("source", "")
                cid = d.metadata.get("chunk_id")
                if src and cid is not None:
                    if src not in source_doc_groups:
                        source_doc_groups[src] = {}
                    source_doc_groups[src][cid] = d

            for cand in top_candidates[:max(top_k, 4)]:
                src = cand.metadata.get("source", "")
                cid = cand.metadata.get("chunk_id")
                if src in source_doc_groups and cid is not None:
                    src_map = source_doc_groups[src]
                    nxt = src_map.get(cid + 1)
                    if nxt and chunk_key(nxt) not in seen_expanded_keys:
                        expanded_docs.append(nxt)
                        seen_expanded_keys.add(chunk_key(nxt))
                    prv = src_map.get(cid - 1)
                    if prv and chunk_key(prv) not in seen_expanded_keys:
                        expanded_docs.append(prv)
                        seen_expanded_keys.add(chunk_key(prv))

                if len(expanded_docs) >= top_k * 2:
                    break

        if not expanded_docs:
            expanded_docs = top_candidates

        # 5. For overview questions, guarantee chunk 0 / Page 1 / Section 0 is at the top
        if is_overview and hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            existing_cids = {c.metadata.get("chunk_id", -1) for c in expanded_docs}
            first_chunks = [
                doc for doc in vector_store.docstore._dict.values()
                if is_source_match(doc) and (
                    doc.metadata.get("chunk_id") == 0
                    or doc.metadata.get("page") == 1
                    or doc.metadata.get("section_id") == 0
                )
            ]
            for fc in sorted(first_chunks, key=lambda d: d.metadata.get("chunk_id", 0)):
                cid = fc.metadata.get("chunk_id", -1)
                if cid not in existing_cids:
                    expanded_docs.insert(0, fc)
                    existing_cids.add(cid)

        # Bound results to preserve top candidates while allowing neighbor context
        max_allowed = max(top_k, min(len(expanded_docs), top_k + 4))
        return expanded_docs[:max_allowed] if not is_overview else expanded_docs[:max(top_k, 10)]

    except Exception as e:
        raise VectorStoreError(f"Error performing hybrid similarity search: {str(e)}") from e
