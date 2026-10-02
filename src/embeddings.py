import streamlit as st
from functools import lru_cache
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


@st.cache_resource(show_spinner=False)
def _cached_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and cache embedding model in Streamlit memory across sessions."""
    encode_kwargs = {"normalize_embeddings": True}
    try:
        model_kwargs = {"device": "cpu", "local_files_only": True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception:
        model_kwargs = {"device": "cpu"}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


@lru_cache(maxsize=1)
def _lru_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Fallback LRU cached loader when called outside Streamlit runtime."""
    encode_kwargs = {"normalize_embeddings": True}
    try:
        model_kwargs = {"device": "cpu", "local_files_only": True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception:
        model_kwargs = {"device": "cpu"}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


def get_embedding_model(
    model_name: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """
    Get or initialize a cached HuggingFaceEmbeddings instance.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    Normalizes embeddings for optimal cosine/inner-product similarity search in FAISS.
    """
    selected_model = model_name or EMBEDDING_MODEL_NAME
    try:
        return _cached_embedding_model(selected_model)
    except Exception:
        return _lru_embedding_model(selected_model)

