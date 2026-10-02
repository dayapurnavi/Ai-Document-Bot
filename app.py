"""DocuMind AI - Enterprise Document & Web Intelligence Workspace.

Features:
- Global Knowledge Base (PDFs + Webpages in unified FAISS)
- Complete URL Analysis Session Isolation (Zero cross-contamination between URLs)
- New Result Page with Dedicated Source Header
- Structured Output Cards: Summary, Key Insights, Links, Other Info, AI Charts, Citations
- Strict AI Chart Validation: NO DATA = NO CHART (source-verified numeric data only)
- Clean, User-Focused Home Page (no developer jargon or technical internals)
- Source-Scoped Q&A for Webpages and Cross-Source Global Chat
"""

import html
import os
import re
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import streamlit as st

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    is_groq_configured,
    update_groq_api_key,
)
from src.pdf_processor import process_pdf_files, chunk_documents
from src.web_processor import (
    process_web_url,
    validate_and_normalize_url,
    WebProcessingError,
    SSRFSecurityError,
)
from src.youtube_processor import (
    process_youtube_url,
    detect_source_type,
    normalize_youtube_url,
    extract_youtube_video_id,
    YouTubeProcessingError,
)
from src.embeddings import get_embedding_model
from src.vector_store import (
    build_vector_store,
    add_documents_to_vector_store,
    remove_source_from_vector_store,
    save_vector_store,
    load_vector_store,
    clear_persisted_vector_store,
    validate_vector_store,
    retrieve_relevant_chunks,
)
from src.source_registry import (
    compute_content_hash,
    is_source_indexed,
    register_source,
    unregister_source_by_name_or_hash,
    list_registered_sources,
    get_source_by_url,
    is_url_indexed,
    has_url_content_changed,
)
from src.rag_chain import query_rag_pipeline, REFUSAL_PHRASE, sanitize_final_response, RAG_PIPELINE_VERSION
from src.structured_analysis import (
    generate_structured_website_analysis,
    validate_analysis_session,
    detect_and_extract_charts,
)

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI — Workspace",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- COMPLETE DESIGN SYSTEM CSS ---
STITCH_COMPLETE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* 1. HIDE ALL DEFAULT STREAMLIT CHROME COMPLETELY */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stToolbar"] { visibility: hidden !important; }
    
    /* 2. BASE THEME & TYPOGRAPHY */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #131b2e !important;
        background-color: #faf8ff !important;
    }

    
    /* 3. SIDEBAR COMPLETE STITCH RESTYLING */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eaedff !important;
        box-shadow: 0 1px 12px rgba(15, 23, 42, 0.04) !important;
        width: 330px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding: 1.25rem 1.1rem 2rem 1.1rem !important;
    }
    
    /* 4. MAIN CANVAS CONTAINER RESET */
    .main {
        background-color: #faf8ff !important;
    }
    .main .block-container {
        max-width: 1560px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* 5. BRAND HEADER & LOGO */
    .stitch-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.25rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #eaedff;
    }
    .stitch-logo-badge {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4338ca, #712ae2);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.25rem;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.3);
    }
    .stitch-title-block {
        display: flex;
        flex-direction: column;
    }
    .stitch-brand-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        color: #131b2e;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .stitch-version-chip {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        background: #eaedff;
        color: #4338ca;
        border-radius: 6px;
        font-family: 'Geist', sans-serif;
    }
    .stitch-brand-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        font-family: 'Geist', sans-serif;
        font-weight: 500;
    }
    
    /* 6. TOP WORKSPACE HEADER & BREADCRUMBS */
    .stitch-top-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(16px);
        border: 1px solid #eaedff;
        border-radius: 16px;
        padding: 0.85rem 1.5rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.03);
    }
    .stitch-breadcrumbs {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Geist', sans-serif;
        font-size: 0.88rem;
        color: #64748b;
    }
    .stitch-breadcrumbs-active {
        color: #4338ca;
        font-weight: 600;
    }
    
    /* 7. CLEAN HOME PAGE STYLING */
    .home-hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
        border: 1px solid #eaedff;
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 20px rgba(67, 56, 202, 0.05);
    }
    .home-hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #131b2e;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .home-hero-tagline {
        font-size: 1.05rem;
        color: #4338ca;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .home-hero-desc {
        font-size: 0.92rem;
        color: #64748b;
        max-width: 800px;
        line-height: 1.55;
    }
    
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 1.75rem;
    }
    .metric-pill {
        background: #ffffff;
        border: 1px solid #eaedff;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-pill:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(67, 56, 202, 0.08);
        border-color: #cbd5e1;
    }
    .metric-icon-box {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
    }
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #131b2e;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-family: 'Geist', sans-serif;
    }
    
    /* 8. STRUCTURED ANALYSIS RESULT PAGE CARDS */
    .page-source-header {
        background: #ffffff;
        border: 1px solid #eaedff;
        border-left: 5px solid #4338ca;
        border-radius: 16px;
        padding: 1.3rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    }
    .source-header-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        font-family: 'Geist', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #4338ca;
        background: #eef2ff;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .status-badge-ready {
        font-family: 'Geist', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        color: #0d9488;
        background: #f0fdf4;
        padding: 3px 8px;
        border-radius: 9999px;
        border: 1px solid #ccfbf1;
    }
    .source-header-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        color: #131b2e;
        margin: 0.2rem 0 0.5rem 0;
    }
    .source-header-meta {
        font-size: 0.85rem;
        color: #64748b;
        display: flex;
        gap: 12px;
        align-items: center;
    }
    .source-header-meta a {
        color: #4338ca;
        text-decoration: underline;
        font-weight: 600;
    }
    
    .structured-card {
        background: #ffffff;
        border: 1px solid #eaedff;
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.03);
    }
    .card-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .card-title-group {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #131b2e;
    }
    .card-footer-scope {
        margin-top: 1rem;
        padding-top: 0.65rem;
        border-top: 1px solid #f8fafc;
        font-size: 0.78rem;
        color: #64748b;
        font-family: 'Geist', sans-serif;
    }
    .card-footer-scope a {
        color: #4338ca;
        font-weight: 600;
        text-decoration: none;
    }
    
    .insight-bullet-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 0.65rem;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
    }
    .insight-dot {
        color: #4338ca;
        font-weight: 800;
        font-size: 1.1rem;
        line-height: 1.2;
    }
    
    .link-chip-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 10px;
    }
    .link-chip {
        background: #f8fafc;
        border: 1px solid #eaedff;
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        display: flex;
        flex-direction: column;
        gap: 2px;
        text-decoration: none !important;
        transition: all 0.15s ease;
    }
    .link-chip:hover {
        background: #ffffff;
        border-color: #cbd5e1;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.08);
    }
    .link-chip-text {
        font-size: 0.88rem;
        font-weight: 600;
        color: #4338ca;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .link-chip-url {
        font-size: 0.74rem;
        color: #64748b;
        font-family: 'Geist', sans-serif;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .meta-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .meta-tag-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #475569;
        font-family: 'Geist', sans-serif;
    }
    
    .source-citation-card {
        background: #f8fafc;
        border: 1px solid #eaedff;
        border-left: 4px solid #4338ca;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* 9. CHAT STYLING */
    .user-msg-row {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 1.25rem;
    }
    .user-msg-bubble {
        background: #4338ca;
        color: #ffffff;
        border-radius: 16px 16px 2px 16px;
        padding: 0.9rem 1.2rem;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.25);
    }
    .ai-msg-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 1.25rem;
    }
    .ai-msg-bubble {
        background: #ffffff;
        border: 1px solid #eaedff;
        border-radius: 16px 16px 16px 2px;
        padding: 1.1rem 1.35rem;
        font-size: 0.92rem;
        line-height: 1.55;
        color: #1e293b;
        box-shadow: 0 1px 6px rgba(15, 23, 42, 0.03);
    }
    
    /* Button customizations */
    div.stButton > button[kind="primary"] {
        background: #4338ca !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 2px 6px rgba(67, 56, 202, 0.25) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #3730a3 !important;
        box-shadow: 0 4px 12px rgba(67, 56, 202, 0.35) !important;
    }
    div.stButton > button:not([kind="primary"]) {
        border-radius: 10px !important;
        border: 1px solid #eaedff !important;
        background: #ffffff !important;
        color: #131b2e !important;
        font-weight: 600 !important;
    }
    div.stButton > button:not([kind="primary"]):hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-ring 2s infinite cubic-bezier(0.66, 0, 0, 1);
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
"""

st.markdown(STITCH_COMPLETE_CSS, unsafe_allow_html=True)


def format_chat_bubble_html(raw_content: str) -> str:
    """Format sanitized markdown text safely into HTML without breaking markdown syntax or producing stray tags."""
    if not raw_content:
        return ""
    clean = sanitize_final_response(raw_content)
    try:
        from markdown_it import MarkdownIt
        md = MarkdownIt("commonmark", {"breaks": True}).enable("table")
        rendered = md.render(clean)
        return rendered.strip()
    except Exception:
        escaped = html.escape(clean).replace("\n", "<br/>")
        return f"<p>{escaped}</p>"


def render_retrieval_debug_expander(debug_info: dict):
    """Render retrieval diagnostics and evidence inspect panel when debug mode is enabled."""
    if not debug_info or not isinstance(debug_info, dict):
        return
    with st.expander("🛠️ Retrieval Diagnostics & Evidence Inspector", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Question Type", debug_info.get("question_type", "N/A"))
        c2.metric("Retrieval Relevance", debug_info.get("retrieval_relevance", "N/A"))
        c3.metric("Answerability", debug_info.get("answerability", "N/A"))
        c4.metric("Evidence State", debug_info.get("evidence_state", "N/A"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Candidates", debug_info.get("candidate_count", 0))
        c6.metric("Context Chunks", debug_info.get("final_context_count", debug_info.get("total_context_chunks", 0)))
        c7.metric("Fallback Used", str(debug_info.get("fallback_used", False)))
        c8.metric("Fallback Source", str(debug_info.get("fallback_source", "None")))

        st.markdown(f"**Active Source:** `{debug_info.get('active_source', 'None')}`")
        st.markdown(
            f"**Source Answerable:** `{debug_info.get('source_answerable', False)}` | "
            f"**LLM Called:** `{debug_info.get('llm_called', False)}` | "
            f"**Refusal:** `{debug_info.get('is_refusal', False)}`"
        )
        if debug_info.get("refusal_reason"):
            st.markdown(f"**Reason / Notes:** *{debug_info.get('refusal_reason')}*")

        if debug_info.get("neighbor_count", 0) > 0:
            st.markdown(f"**Neighbor Chunks Expanded:** `{debug_info.get('neighbor_count', 0)}`")

        if debug_info.get("final_context_preview"):
            st.markdown("**Top Context Snippets Sent to LLM:**")
            for idx, snip in enumerate(debug_info.get("final_context_preview", [])[:3], 1):
                st.code(snip, language="markdown")


# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    """Ensure all required session state variables are initialized and restore persistent vector store."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "kb_stats" not in st.session_state:
        st.session_state.kb_stats = {
            "processed": False,
            "document_names": [],
            "total_pages": 0,
            "total_chunks": 0,
            "file_size_kb": 0.0,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "groq_model": GROQ_MODEL,
        }
    if "uploaded_file_fingerprint" not in st.session_state:
        st.session_state.uploaded_file_fingerprint = ""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = GROQ_API_KEY if is_groq_configured(GROQ_API_KEY) else ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = (
            GROQ_MODEL
            if GROQ_MODEL in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
            else "openai/gpt-oss-20b"
        )
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # URL Analysis Isolation and Navigation
    if "current_view" not in st.session_state:
        st.session_state.current_view = "home"  # "home", "website_analysis", "knowledge_chat"
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "analyses" not in st.session_state:
        st.session_state.analyses = {}  # { normalized_url: analysis_dict }
    if "active_url" not in st.session_state:
        st.session_state.active_url = None
    if "active_source_id" not in st.session_state:
        st.session_state.active_source_id = None
    if "active_source_url" not in st.session_state:
        st.session_state.active_source_url = None
    if "active_source_type" not in st.session_state:
        st.session_state.active_source_type = None
    if "active_source_hash" not in st.session_state:
        st.session_state.active_source_hash = None
    if "active_analysis_id" not in st.session_state:
        st.session_state.active_analysis_id = None
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}  # { (source_id, content_hash, normalized_q, mode, top_k): result }

    # Restore from persisted storage if vector_store is not yet loaded in session state
    if st.session_state.vector_store is None:
        try:
            persisted_vs = load_vector_store()
            if persisted_vs is not None:
                st.session_state.vector_store = persisted_vs
                registered = list_registered_sources()
                if registered:
                    doc_names = [s.get("source", "Document") for s in registered]
                    total_pages = sum(s.get("total_pages") or 0 for s in registered)
                    total_chunks = len(persisted_vs.docstore._dict)
                else:
                    doc_names = sorted(list({
                        doc.metadata.get("source", "Document")
                        for doc in persisted_vs.docstore._dict.values()
                    }))
                    total_chunks = len(persisted_vs.docstore._dict)
                    total_pages = len({
                        (doc.metadata.get("source"), doc.metadata.get("page"))
                        for doc in persisted_vs.docstore._dict.values()
                        if doc.metadata.get("page") is not None
                    })

                st.session_state.kb_stats = {
                    "processed": True,
                    "document_names": doc_names,
                    "total_pages": total_pages,
                    "total_chunks": total_chunks,
                    "file_size_kb": 0.0,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "groq_model": st.session_state.selected_model,
                }
        except Exception:
            st.session_state.vector_store = None


init_session_state()


def clear_knowledge_base():
    """Reset the vector database, clear cached indices, wipe source registry, and wipe chat history."""
    clear_persisted_vector_store()
    st.session_state.vector_store = None
    st.session_state.messages = []
    st.session_state.current_analysis = None
    st.session_state.analyses = {}
    st.session_state.active_url = None
    st.session_state.active_source_id = None
    st.session_state.active_source_url = None
    st.session_state.active_source_type = None
    st.session_state.active_analysis_id = None
    st.session_state.query_cache = {}
    st.session_state.current_view = "home"
    st.session_state.kb_stats = {
        "processed": False,
        "document_names": [],
        "total_pages": 0,
        "total_chunks": 0,
        "file_size_kb": 0.0,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "groq_model": GROQ_MODEL,
    }
    st.session_state.uploaded_file_fingerprint = ""


# --- SIDEBAR NAVIGATION & CONFIGURATION ---
with st.sidebar:
    # 1. Stitch Brand Identity
    st.markdown(
        """
        <div class="stitch-brand">
            <div class="stitch-logo-badge">D</div>
            <div class="stitch-title-block">
                <div class="stitch-brand-name">
                    DocuMind AI
                    <span class="stitch-version-chip">v2.5 Pro</span>
                </div>
                <div class="stitch-brand-subtitle">Knowledge Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. View Switcher Navigation Rail
    nav_options = ["🏠 Home", "🌐 Website Analysis", "💬 Global Chat"]
    current_idx = 0
    if st.session_state.current_view == "website_analysis":
        current_idx = 1
    elif st.session_state.current_view == "knowledge_chat":
        current_idx = 2

    chosen_nav = st.radio(
        "Workspace View",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed",
        key="nav_view_selector",
    )
    if chosen_nav == "🏠 Home" and st.session_state.current_view != "home":
        st.session_state.current_view = "home"
        st.rerun()
    elif chosen_nav == "🌐 Website Analysis" and st.session_state.current_view != "website_analysis":
        st.session_state.current_view = "website_analysis"
        st.rerun()
    elif chosen_nav == "💬 Global Chat" and st.session_state.current_view != "knowledge_chat":
        st.session_state.current_view = "knowledge_chat"
        st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #eaedff; margin: 1.1rem 0;'>", unsafe_allow_html=True)

    # 3. Groq API Configuration
    st.markdown("#### 🔑 Groq API Engine")
    current_saved_key = st.session_state.groq_api_key
    user_api_key = st.text_input(
        "Groq API Key",
        value=current_saved_key,
        type="password",
        placeholder="gsk_...",
        help="Get your free key from https://console.groq.com/keys",
        label_visibility="collapsed",
    )
    if user_api_key.strip() != current_saved_key:
        st.session_state.groq_api_key = user_api_key.strip()
        update_groq_api_key(user_api_key.strip())
        st.rerun()

    active_key = st.session_state.groq_api_key or GROQ_API_KEY
    if not is_groq_configured(active_key):
        st.warning(
            "⚠️ **API Key Required**\n\n"
            "Paste your Groq API key (`gsk_...`) above.\n\n"
            "[👉 Get a free Groq Key](https://console.groq.com/keys)"
        )
    else:
        model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="Select the Groq model for question answering.",
        )
        st.session_state.selected_model = selected_model
        st.caption(f"⚡ LLM: **Groq** (`{selected_model}`)")

    st.markdown("<hr style='border: none; border-top: 1px solid #eaedff; margin: 1.1rem 0;'>", unsafe_allow_html=True)

    # 4. Document Library Manager (PDFs)
    st.markdown("#### 📄 Document Library")
    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to build your semantic knowledge base.",
        label_visibility="collapsed",
    )

    duplicate_files = []
    new_files = []
    file_info_map = {}

    if uploaded_files:
        total_size_kb = sum(len(f.getvalue()) for f in uploaded_files) / 1024
        st.caption(f"📁 {len(uploaded_files)} file(s) attached ({total_size_kb:.1f} KB)")

        for f in uploaded_files:
            f_bytes = f.getvalue()
            file_kb = len(f_bytes) / 1024
            f_hash = compute_content_hash(f_bytes)
            file_info_map[f.name] = {"bytes": f_bytes, "hash": f_hash, "kb": file_kb}

            is_dup = is_source_indexed(f_hash) or (
                st.session_state.vector_store is not None
                and any(
                    doc.metadata.get("content_hash") == f_hash or doc.metadata.get("source") == f.name
                    for doc in st.session_state.vector_store.docstore._dict.values()
                )
            )

            if is_dup:
                duplicate_files.append(f)
            else:
                new_files.append(f)

            badge_html = (
                '<span style="background: #eef2ff; color: #4338ca; font-size: 0.72rem; font-weight: 600; padding: 1px 5px; border-radius: 4px;">Indexed</span>'
                if is_dup
                else f'<span style="color: #64748b; font-family: \'Geist\', sans-serif;">{file_kb:.0f} KB</span>'
            )

            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #eaedff; border-radius: 8px; padding: 0.4rem 0.6rem; margin-bottom: 4px; font-size: 0.8rem; display: flex; align-items: center; justify-content: space-between;">
                    <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px;">📄 {f.name}</span>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Action Buttons: PDF Ingestion / Re-index / Clear
    if uploaded_files and len(duplicate_files) == len(uploaded_files):
        col_use, col_reidx, col_clr = st.columns([1.1, 1.1, 0.9])
        with col_use:
            use_existing_btn = st.button("Use Existing", use_container_width=True)
        with col_reidx:
            reindex_btn = st.button("🔄 Re-index", type="primary", use_container_width=True)
        with col_clr:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        process_btn = False
    else:
        use_existing_btn = False
        reindex_btn = False
        col_proc, col_clr = st.columns([1.4, 1])
        with col_proc:
            process_btn = st.button("⚡ Process PDFs", type="primary", use_container_width=True, disabled=not bool(uploaded_files))
        with col_clr:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        clear_knowledge_base()
        st.toast("Knowledge base cleared.", icon="🗑️")
        st.rerun()

    if use_existing_btn:
        st.toast("Using existing indexed document.", icon="📄")

    # Re-indexing PDFs
    if reindex_btn and duplicate_files:
        with st.status("Re-indexing document(s)...", expanded=True) as status:
            try:
                embeddings = get_embedding_model()
                current_vs = st.session_state.vector_store

                for df in duplicate_files:
                    if current_vs is not None:
                        current_vs = remove_source_from_vector_store(current_vs, df.name, embeddings)
                    unregister_source_by_name_or_hash(df.name)

                reindex_tuples = [
                    (file_info_map[df.name]["bytes"], df.name, file_info_map[df.name]["hash"])
                    for df in duplicate_files
                ]
                chunks, stats = process_pdf_files(
                    file_sources=reindex_tuples,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

                if chunks:
                    current_vs = add_documents_to_vector_store(current_vs, chunks, embeddings)
                    save_vector_store(current_vs)

                    for proc_item in stats["processed_files"]:
                        register_source(
                            source=proc_item["filename"],
                            source_type="pdf",
                            title=proc_item["filename"],
                            content_hash=proc_item.get("content_hash", ""),
                            chunk_count=proc_item["chunks"],
                            total_pages=proc_item["pages"],
                        )

                    st.session_state.vector_store = current_vs
                    st.session_state.kb_stats["processed"] = True
                    st.session_state.kb_stats["document_names"] = sorted(list({
                        doc.metadata.get("source") for doc in current_vs.docstore._dict.values()
                    }))
                    st.session_state.kb_stats["total_chunks"] = len(current_vs.docstore._dict)
                    st.session_state.kb_stats["total_pages"] = sum(
                        s.get("total_pages", 0) or 0 for s in list_registered_sources()
                    )

                    status.update(label="✓ Document re-indexed successfully!", state="complete")
                    st.toast("Document re-indexed successfully!", icon="✅")
                    st.rerun()
                else:
                    status.update(label="No extractable text found", state="error")
            except Exception as ex:
                status.update(label="Re-indexing failed", state="error")
                st.error(f"Error during re-indexing: {str(ex)}")

    # Processing New PDFs
    if process_btn and uploaded_files:
        files_to_process = new_files if new_files else uploaded_files
        with st.status("Building semantic FAISS index...", expanded=True) as status:
            try:
                file_tuples = [
                    (file_info_map[f.name]["bytes"], f.name, file_info_map[f.name]["hash"])
                    for f in files_to_process
                ]
                chunks, stats = process_pdf_files(
                    file_sources=file_tuples,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )

                if stats["empty_files"]:
                    st.warning(f"⚠️ Empty PDF skipped: {', '.join(stats['empty_files'])}")

                if chunks:
                    embeddings = get_embedding_model()
                    vector_store = add_documents_to_vector_store(
                        st.session_state.vector_store, chunks, embeddings
                    )
                    save_vector_store(vector_store)

                    for proc_item in stats["processed_files"]:
                        register_source(
                            source=proc_item["filename"],
                            source_type="pdf",
                            title=proc_item["filename"],
                            content_hash=proc_item.get("content_hash", ""),
                            chunk_count=proc_item["chunks"],
                            total_pages=proc_item["pages"],
                        )

                    total_size_kb = sum(len(f.getvalue()) for f in uploaded_files) / 1024
                    st.session_state.vector_store = vector_store
                    st.session_state.kb_stats = {
                        "processed": True,
                        "document_names": sorted(list({
                            doc.metadata.get("source") for doc in vector_store.docstore._dict.values()
                        })),
                        "total_pages": sum(s.get("total_pages", 0) or 0 for s in list_registered_sources()),
                        "total_chunks": len(vector_store.docstore._dict),
                        "file_size_kb": total_size_kb,
                        "embedding_model": EMBEDDING_MODEL_NAME,
                        "groq_model": st.session_state.selected_model,
                    }
                    st.session_state.uploaded_file_fingerprint = "-".join(
                        sorted([f.name for f in uploaded_files])
                    )

                    status.update(label=f"✓ Vector index ready ({len(vector_store.docstore._dict)} chunks)!", state="complete")
                    st.toast("Knowledge base updated!", icon="✅")
                    st.rerun()
                else:
                    status.update(label="No text found in PDFs.", state="error")
                    st.error("Could not extract any text from the uploaded PDF documents.")
            except Exception as err:
                status.update(label="Processing failed", state="error")
                st.error(f"Error processing documents: {str(err)}")

    st.markdown("<hr style='border: none; border-top: 1px solid #eaedff; margin: 1.1rem 0;'>", unsafe_allow_html=True)

    # 5. Web / URL / YouTube Ingestion Manager
    st.markdown("#### 🌐 Web / URL / YouTube Ingestion")
    input_url = st.text_input(
        "Website or YouTube URL",
        placeholder="https://example.com/article or YouTube URL",
        help="Enter a public webpage or YouTube video URL to ingest and analyze.",
        label_visibility="collapsed",
        key="web_url_input_box",
    )

    url_is_indexed = False
    norm_url_preview = None
    raw_input_url = input_url.strip()
    detected_source_type = detect_source_type(raw_input_url) if raw_input_url else "webpage"
    is_youtube_input = detected_source_type == "youtube"

    if raw_input_url:
        try:
            norm_url_preview = normalize_youtube_url(raw_input_url) if is_youtube_input else validate_and_normalize_url(raw_input_url)
            url_is_indexed = is_url_indexed(norm_url_preview) or (
                st.session_state.vector_store is not None
                and any(
                    doc.metadata.get("source") == norm_url_preview
                    for doc in st.session_state.vector_store.docstore._dict.values()
                )
            )
        except Exception:
            url_is_indexed = False

    if url_is_indexed:
        st.info("▶ YouTube video already indexed." if is_youtube_input else "🌐 Webpage already indexed.")

    col_add_url, col_reidx_url = st.columns([1.2, 1.2])
    with col_add_url:
        if is_youtube_input:
            add_button_label = "+ Analyze Video" if not url_is_indexed else "Open Analysis"
        else:
            add_button_label = "+ Add Website" if not url_is_indexed else "Open Analysis"

        add_url_btn = st.button(
            add_button_label,
            type="primary" if not url_is_indexed else "secondary",
            use_container_width=True,
            disabled=not bool(raw_input_url),
        )
    with col_reidx_url:
        reindex_url_btn = st.button(
            "🔄 Re-index",
            use_container_width=True,
            disabled=not (bool(raw_input_url) and url_is_indexed),
            help="Re-fetch and replace indexed content for this URL.",
        )

    # Ingestion Execution Trigger
    if (add_url_btn or reindex_url_btn) and raw_input_url:
        try:
            if is_youtube_input:
                norm_target_url = normalize_youtube_url(raw_input_url)
            else:
                norm_target_url = validate_and_normalize_url(raw_input_url)

            # URL Change Detection: If user entered an existing URL or selected Open Analysis
            if add_url_btn and url_is_indexed:
                # Use Existing path: Load indexed content for this URL into an isolated analysis
                src_reg_entry = get_source_by_url(norm_target_url) or {}
                st.session_state.active_source_id = norm_target_url
                st.session_state.active_source_url = norm_target_url
                st.session_state.active_source_hash = src_reg_entry.get("content_hash", "")
                st.session_state.active_source_type = detected_source_type
                st.session_state.active_url = norm_target_url
                st.session_state.current_view = "website_analysis"

                if norm_target_url in st.session_state.analyses:
                    st.session_state.current_analysis = st.session_state.analyses[norm_target_url]
                    st.session_state.active_analysis_id = st.session_state.current_analysis.get("analysis_id")
                    st.session_state.active_source_hash = st.session_state.current_analysis.get("content_hash", st.session_state.active_source_hash)
                else:
                    # Build fresh analysis from stored chunks in vector store
                    if st.session_state.vector_store is not None:
                        url_chunks = [
                            doc for doc in st.session_state.vector_store.docstore._dict.values()
                            if doc.metadata.get("source") == norm_target_url
                        ]
                        page_title = url_chunks[0].metadata.get("title", norm_target_url) if url_chunks else norm_target_url
                        analysis = generate_structured_website_analysis(
                            url=norm_target_url,
                            page_title=page_title,
                            chunks=url_chunks,
                            llm_model=st.session_state.selected_model,
                            api_key=active_key,
                        )
                        st.session_state.current_analysis = analysis
                        st.session_state.analyses[norm_target_url] = analysis
                        st.session_state.active_analysis_id = analysis.get("analysis_id")
                        st.session_state.active_source_hash = analysis.get("content_hash", "")
                st.toast("Loaded existing analysis.", icon="▶" if is_youtube_input else "🌐")
                st.rerun()

            # Active Ingestion / Re-indexing Path:
            # 1. Reset per-page analysis session state to prevent URL A contamination
            st.session_state.active_source_id = norm_target_url
            st.session_state.active_source_url = norm_target_url
            st.session_state.active_source_hash = None
            st.session_state.active_source_type = detected_source_type
            st.session_state.active_url = norm_target_url
            st.session_state.current_analysis = None
            st.session_state.current_view = "website_analysis"

            status_banner = f"🎬 Analyzing YouTube: {norm_target_url}..." if is_youtube_input else f"🔄 Analyzing {norm_target_url}..."
            with st.status(status_banner, expanded=True) as web_status:
                if is_youtube_input:
                    web_status.write("🎬 Fetching YouTube video information & captions...")
                    raw_docs, web_stats = process_youtube_url(norm_target_url)
                    chunks = raw_docs  # already chunked by chunk_transcript_segments with timestamps
                else:
                    web_status.write("🌐 Fetching webpage safely...")
                    raw_docs, web_stats = process_web_url(norm_target_url)
                    web_status.write("✂️ Creating semantic chunks...")
                    chunks = chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

                final_url = web_stats["url"]
                page_title = web_stats["title"]
                content_hash = web_stats["content_hash"]
                final_source_type = web_stats.get("source_type", detected_source_type)

                embeddings = get_embedding_model()
                current_vs = st.session_state.vector_store

                # Remove previous chunks if re-indexing or already in index
                if is_url_indexed(final_url) or reindex_url_btn:
                    web_status.write("🔄 Updating indexed chunks...")
                    if current_vs is not None:
                        current_vs = remove_source_from_vector_store(current_vs, final_url, embeddings)
                    unregister_source_by_name_or_hash(final_url)

                web_status.write(f"🧹 Extracted content: **{page_title}**")

                if not chunks:
                    web_status.update(label="No extractable text found", state="error")
                    st.session_state.current_analysis = {
                        "analysis_id": str(uuid.uuid4()),
                        "source_url": final_url,
                        "source_title": page_title,
                        "source_type": final_source_type,
                        "status": "failed",
                        "summary": "Could not extract readable text from this source.",
                        "key_points": [],
                        "links": [],
                        "other_information": {},
                        "charts": [],
                        "citations": [],
                        "messages": [],
                        "error": "No readable content extracted.",
                    }
                    st.session_state.active_analysis_id = st.session_state.current_analysis["analysis_id"]
                else:
                    web_status.write(f"🧠 Indexing {len(chunks)} chunks in unified FAISS...")
                    current_vs = add_documents_to_vector_store(current_vs, chunks, embeddings)
                    save_vector_store(current_vs)

                    register_source(
                        source=final_url,
                        source_type=final_source_type,
                        title=page_title,
                        content_hash=content_hash,
                        chunk_count=len(chunks),
                        total_pages=None,
                    )

                    st.session_state.vector_store = current_vs
                    st.session_state.kb_stats["processed"] = True
                    st.session_state.kb_stats["document_names"] = sorted(list({
                        doc.metadata.get("source") for doc in current_vs.docstore._dict.values()
                    }))
                    st.session_state.kb_stats["total_chunks"] = len(current_vs.docstore._dict)

                    web_status.write("📊 Generating structured summary, insights & charts...")
                    analysis = generate_structured_website_analysis(
                        url=final_url,
                        page_title=page_title,
                        chunks=chunks,
                        web_stats=web_stats,
                        llm_model=st.session_state.selected_model,
                        api_key=active_key,
                    )

                    # Store isolated analysis
                    st.session_state.active_analysis_id = analysis["analysis_id"]
                    st.session_state.active_source_id = final_url
                    st.session_state.active_source_url = final_url
                    st.session_state.active_source_hash = content_hash
                    st.session_state.active_source_type = final_source_type
                    st.session_state.active_url = final_url
                    st.session_state.current_analysis = analysis
                    st.session_state.analyses[final_url] = analysis

                    web_status.update(label="✓ Analysis Ready!", state="complete")
                    st.toast(f"Analysis Ready: {page_title}", icon="▶" if is_youtube_input else "🌐")
                    st.rerun()

        except SSRFSecurityError as ssrf_err:
            st.session_state.current_analysis = {
                "analysis_id": str(uuid.uuid4()),
                "source_url": input_url.strip(),
                "source_title": "Security Exception",
                "source_type": detected_source_type,
                "status": "failed",
                "summary": "Security Block: This URL points to a private or restricted address.",
                "key_points": [],
                "links": [],
                "other_information": {},
                "charts": [],
                "citations": [],
                "messages": [],
                "error": str(ssrf_err),
            }
            st.session_state.active_analysis_id = st.session_state.current_analysis["analysis_id"]
            st.error(f"🛡️ Security Block: {str(ssrf_err)}")
        except YouTubeProcessingError as yt_err:
            st.session_state.current_analysis = {
                "analysis_id": str(uuid.uuid4()),
                "source_url": input_url.strip(),
                "source_title": "YouTube Video Ingestion Failed",
                "source_type": "youtube",
                "status": "failed",
                "summary": "YouTube video analysis failed.",
                "key_points": [],
                "links": [],
                "other_information": {},
                "charts": [],
                "citations": [],
                "messages": [],
                "error": str(yt_err),
            }
            st.session_state.active_analysis_id = st.session_state.current_analysis["analysis_id"]
            st.error(f"❌ YouTube Error: {str(yt_err)}")
        except WebProcessingError as web_err:
            st.session_state.current_analysis = {
                "analysis_id": str(uuid.uuid4()),
                "source_url": input_url.strip(),
                "source_title": "Ingestion Failed",
                "source_type": "url",
                "status": "failed",
                "summary": "Website analysis failed. Unable to fetch webpage.",
                "key_points": [],
                "links": [],
                "other_information": {},
                "charts": [],
                "citations": [],
                "messages": [],
                "error": str(web_err),
            }
            st.session_state.active_analysis_id = st.session_state.current_analysis["analysis_id"]
            st.error(f"❌ {str(web_err)}")
        except Exception as ex:
            st.session_state.current_analysis = {
                "analysis_id": str(uuid.uuid4()),
                "source_url": input_url.strip(),
                "source_title": "Processing Error",
                "source_type": detected_source_type,
                "status": "failed",
                "summary": "Failed to analyze source.",
                "key_points": [],
                "links": [],
                "other_information": {},
                "charts": [],
                "citations": [],
                "messages": [],
                "error": str(ex),
            }
            st.session_state.active_analysis_id = st.session_state.current_analysis["analysis_id"]
            st.error(f"Processing error: {str(ex)}")

    st.markdown("<hr style='border: none; border-top: 1px solid #eaedff; margin: 1.1rem 0;'>", unsafe_allow_html=True)

    # 6. Retrieval Settings
    st.markdown("#### ⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-K Citations",
        min_value=1,
        max_value=15,
        value=DEFAULT_TOP_K,
        help="Number of chunks retrieved for answer grounding.",
    )

    answer_mode_ui = st.radio(
        "Answer Mode",
        options=["🌐 Source + Fallback", "🔒 Indexed Source Only"],
        index=0,
        help="Source + Fallback: answers from source first and provides helpful fallback when source is insufficient. Indexed Source Only: strictly answers only from uploaded/indexed content.",
        key="answer_mode_toggle",
    )
    answer_mode = "SOURCE_FIRST_WITH_FALLBACK" if "Fallback" in answer_mode_ui else "STRICT_SOURCE"
    if answer_mode == "STRICT_SOURCE":
        st.caption("🔒 Answers only from uploaded/indexed content.")
    else:
        st.caption("🌐 Uses indexed content first; provides additional knowledge when insufficient.")

    debug_mode_enabled = st.checkbox(
        "🛠️ Retrieval Debug Mode",
        value=False,
        key="retrieval_debug_toggle",
        help="Display intent classification, query expansion variants, candidate pool size, relevance scores, and neighbor chunk diagnostics.",
    )

    # 7. User Profile Card
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #eaedff; border-radius: 12px; padding: 0.75rem; display: flex; align-items: center; gap: 10px; margin-top: 1.5rem;">
            <div class="stitch-logo-badge" style="width: 34px; height: 34px; font-size: 0.95rem; background: #4338ca;">G</div>
            <div style="flex: 1; overflow: hidden;">
                <div style="font-weight: 700; font-size: 0.88rem; color: #131b2e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Ganesh</div>
                <div style="font-size: 0.75rem; color: #64748b; font-family: 'Geist', sans-serif;">Enterprise Workspace</div>
            </div>
            <span style="color: #0d9488; font-size: 0.8rem; font-weight: 700; font-family: 'Geist', sans-serif;">● Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# VIEW 1: HOME PAGE (Clean, User-Focused, No Developer Jargon)
# =========================================================================
if st.session_state.current_view == "home":
    st.markdown(
        """
        <div class="stitch-top-header">
            <div class="stitch-breadcrumbs">
                <span>Workspace</span>
                <span style="color: #94a3b8; font-weight: bold;">/</span>
                <span class="stitch-breadcrumbs-active">Home Hub</span>
            </div>
            <div style="background: #f0fdf4; border: 1px solid #ccfbf1; color: #0d9488; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; font-family: 'Geist', sans-serif; display: flex; align-items: center; gap: 6px;">
                <span class="pulse-dot"></span>
                Syntropic Clarity Engine Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Branding Block
    st.markdown(
        """
        <div class="home-hero-card">
            <div class="home-hero-title">DocuMind AI</div>
            <div class="home-hero-tagline">Ask questions about your documents and websites.</div>
            <div class="home-hero-desc">
                DocuMind AI combines your PDFs and public web pages into a verified, hallucination-guarded semantic knowledge workspace. Every response is strictly grounded in verifiable source citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Knowledge Source Metrics Overview (Simple user metrics, no internal vector/chunk counts)
    all_registered = list_registered_sources()
    pdf_count = len([s for s in all_registered if s.get("source_type") == "pdf"])
    web_count = len([s for s in all_registered if s.get("source_type") in ("url", "webpage")])
    yt_count = len([s for s in all_registered if s.get("source_type") == "youtube"])
    total_sources = len(all_registered)

    st.markdown(
        f"""
        <div class="metrics-container">
            <div class="metric-pill">
                <div class="metric-icon-box" style="background: #fee2e2; color: #dc2626;">📄</div>
                <div>
                    <div class="metric-value">{pdf_count}</div>
                    <div class="metric-label">PDF Documents</div>
                </div>
            </div>
            <div class="metric-pill">
                <div class="metric-icon-box" style="background: #e0e7ff; color: #4338ca;">🌐</div>
                <div>
                    <div class="metric-value">{web_count}</div>
                    <div class="metric-label">Indexed Websites</div>
                </div>
            </div>
            <div class="metric-pill">
                <div class="metric-icon-box" style="background: #fee2e2; color: #dc2626;">▶</div>
                <div>
                    <div class="metric-value">{yt_count}</div>
                    <div class="metric-label">YouTube Videos</div>
                </div>
            </div>
            <div class="metric-pill">
                <div class="metric-icon-box" style="background: #f0fdf4; color: #059669;">📚</div>
                <div>
                    <div class="metric-value">{total_sources}</div>
                    <div class="metric-label">Total Sources</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Analysis Banner (if currently loaded)
    if st.session_state.current_analysis and st.session_state.current_analysis.get("status") == "completed":
        active_title = st.session_state.current_analysis.get("source_title", "Source")
        active_url_str = st.session_state.current_analysis.get("source_url", "")
        active_type = st.session_state.current_analysis.get("source_type", "url")
        is_yt_act = active_type == "youtube"
        banner_title = "Active YouTube Video Analysis" if is_yt_act else "Active Website Analysis"
        border_col = "#dc2626" if is_yt_act else "#4338ca"
        tag_col = "#dc2626" if is_yt_act else "#4338ca"
        badge_icon = "▶" if is_yt_act else "🌐"

        col_banner, col_btn = st.columns([8, 2])
        with col_banner:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #eaedff; border-left: 4px solid {border_col}; border-radius: 12px; padding: 0.9rem 1.25rem;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: {tag_col}; text-transform: uppercase;">{banner_title}</div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #131b2e;">{badge_icon} {active_title}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">{active_url_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("Open Analysis →", key="home_open_analysis_btn", use_container_width=True, type="primary"):
                st.session_state.current_view = "website_analysis"
                st.rerun()

    # Active & Recent Sources Shelf
    st.markdown("### 📚 Indexed Knowledge Sources")
    if not all_registered:
        st.info("👈 Upload PDF documents or enter a Website or YouTube URL in the sidebar to build your knowledge base.")
    else:
        for src in all_registered:
            src_type = src.get("source_type", "pdf")
            src_name = src.get("source", "")
            src_title = src.get("title", src_name)

            col_src_info, col_src_act = st.columns([8, 2])
            with col_src_info:
                if src_type == "youtube":
                    st.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #eaedff; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span style="font-weight: 700; font-size: 0.92rem; color: #131b2e;">▶ {src_title}</span>
                                <div style="font-size: 0.78rem; color: #64748b; font-family: 'Geist', sans-serif;">{src_name}</div>
                            </div>
                            <span style="background: #fee2e2; color: #dc2626; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;">YouTube</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif src_type in ("url", "webpage"):
                    st.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #eaedff; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span style="font-weight: 700; font-size: 0.92rem; color: #131b2e;">🌐 {src_title}</span>
                                <div style="font-size: 0.78rem; color: #64748b; font-family: 'Geist', sans-serif;">{src_name}</div>
                            </div>
                            <span style="background: #eef2ff; color: #4338ca; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;">Webpage</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    pages = src.get("total_pages", 1)
                    st.markdown(
                        f"""
                        <div style="background: #ffffff; border: 1px solid #eaedff; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span style="font-weight: 700; font-size: 0.92rem; color: #131b2e;">📄 {src_name}</span>
                                <div style="font-size: 0.78rem; color: #64748b; font-family: 'Geist', sans-serif;">{pages} page(s) indexed</div>
                            </div>
                            <span style="background: #fee2e2; color: #dc2626; font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;">PDF</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_src_act:
                if src_type in ("url", "webpage", "youtube"):
                    if st.button("Inspect ↗", key=f"inspect_src_{src_name}", use_container_width=True):
                        st.session_state.active_source_id = src_name
                        st.session_state.active_source_url = src_name
                        st.session_state.active_source_hash = src.get("content_hash", "")
                        st.session_state.active_source_type = src_type
                        st.session_state.active_url = src_name
                        st.session_state.current_view = "website_analysis"
                        if src_name in st.session_state.analyses:
                            st.session_state.current_analysis = st.session_state.analyses[src_name]
                            st.session_state.active_analysis_id = st.session_state.current_analysis.get("analysis_id")
                            st.session_state.active_source_hash = st.session_state.current_analysis.get("content_hash", st.session_state.active_source_hash)
                        else:
                            if st.session_state.vector_store is not None:
                                url_chunks = [
                                    doc for doc in st.session_state.vector_store.docstore._dict.values()
                                    if doc.metadata.get("source") == src_name
                                ]
                                analysis = generate_structured_website_analysis(
                                    url=src_name,
                                    page_title=src_title,
                                    chunks=url_chunks,
                                    llm_model=st.session_state.selected_model,
                                    api_key=active_key,
                                )
                                st.session_state.current_analysis = analysis
                                st.session_state.analyses[src_name] = analysis
                                st.session_state.active_analysis_id = analysis.get("analysis_id")
                        st.rerun()

    # Universal Search / Ask Dock
    home_query = st.chat_input("Ask anything across your documents and websites...")
    if home_query:
        st.session_state.current_view = "knowledge_chat"
        st.session_state.pending_query = home_query
        st.rerun()


# =========================================================================
# VIEW 2: WEBSITE / YOUTUBE ANALYSIS (Strict Source Isolation & Structured Cards)
# =========================================================================
elif st.session_state.current_view == "website_analysis":
    # Navigation Top Bar: Back to Home
    col_back, col_title_bar = st.columns([2, 8])
    with col_back:
        if st.button("← Back to Home", key="back_to_home_btn", use_container_width=True):
            st.session_state.current_view = "home"
            st.rerun()

    analysis = st.session_state.current_analysis
    expected_url = st.session_state.active_url

    # Stale Result Protection Guard
    if not analysis or not expected_url:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 1rem; color: #64748b;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌐</div>
                <h3 style="color: #131b2e;">No Source Selected</h3>
                <p>Enter a public webpage or YouTube video URL in the sidebar and click <strong>+ Add / Analyze</strong> to start an analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif analysis.get("status") == "failed":
        # Failed state: strictly do not render any previous URL result
        err_msg = analysis.get("error", "Failed to ingest source.")
        st.markdown(
            f"""
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 14px; padding: 2rem; margin: 1.5rem 0;">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">❌</div>
                <h3 style="color: #991b1b; margin-top: 0;">Source Analysis Failed</h3>
                <p style="color: #b91c1c; font-size: 0.95rem;">{err_msg}</p>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 1rem;">
                    No analysis available for: <strong>{analysis.get('source_url')}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not validate_analysis_session(analysis, expected_url):
        # Mismatched / Stale Result Protection
        st.warning("Stale analysis detected and safely discarded. Please re-select or re-analyze the source.")
    else:
        # Verified Current Analysis Session
        cur_url = analysis["source_url"]
        cur_title = analysis["source_title"]
        created_time = analysis.get("created_at", "Recently")
        is_yt_page = analysis.get("source_type") == "youtube"
        badge_header = '<span class="source-badge" style="background: #fee2e2; color: #dc2626;">▶ YOUTUBE ANALYSIS</span>' if is_yt_page else '<span class="source-badge">🌐 WEBSITE ANALYSIS</span>'
        source_label_text = "YouTube Video" if is_yt_page else "Webpage"
        ch_meta = ""
        channel_name = analysis.get("other_information", {}).get("channel") or analysis.get("other_information", {}).get("author")
        if channel_name:
            ch_meta = f'<span>•</span><span><strong>Channel:</strong> {channel_name}</span>'

        # Page-Level Source Header
        st.markdown(
            f"""
            <div class="page-source-header">
                <div class="source-header-top">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 800; color: #4338ca; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">ACTIVE SOURCE</div>
                        {badge_header}
                    </div>
                    <span class="status-badge-ready">✓ Indexed • Analysis Ready</span>
                </div>
                <h2 class="source-header-title">{cur_title}</h2>
                <div class="source-header-meta">
                    <span><strong>URL:</strong> <a href="{cur_url}" target="_blank" rel="noopener noreferrer">{cur_url} ↗</a></span>
                    <span>•</span>
                    <span><strong>Source:</strong> {source_label_text}</span>
                    {ch_meta}
                    <span>•</span>
                    <span><strong>Indexed:</strong> {created_time}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_main_left, col_main_right = st.columns([7, 4], gap="large")

        with col_main_left:
            # BOX 1 — SUMMARY
            summary_text = analysis.get("summary", "")
            st.markdown(
                f"""
                <div class="structured-card">
                    <div class="card-header-bar">
                        <div class="card-title-group">
                            <span>📝</span>
                            <span>Summary</span>
                        </div>
                        <span style="font-size: 0.74rem; font-weight: 600; color: #4338ca; background: #eef2ff; padding: 2px 7px; border-radius: 6px;">AI Grounded</span>
                    </div>
                    <div style="font-size: 0.94rem; line-height: 1.6; color: #1e293b;">
                        {summary_text}
                    </div>
                    <div class="card-footer-scope">
                        Based only on: <a href="{cur_url}" target="_blank" rel="noopener noreferrer">{cur_url} ↗</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # BOX 2 — KEY INSIGHTS
            key_insights = analysis.get("key_points", [])
            if key_insights:
                insights_html = "".join(
                    f'<div class="insight-bullet-item"><span class="insight-dot">•</span><div>{pt}</div></div>'
                    for pt in key_insights
                )
                st.markdown(
                    f"""
                    <div class="structured-card">
                        <div class="card-header-bar">
                            <div class="card-title-group">
                                <span>💡</span>
                                <span>Key Insights</span>
                            </div>
                            <span style="font-size: 0.74rem; font-weight: 600; color: #0d9488; background: #f0fdf4; padding: 2px 7px; border-radius: 6px;">{len(key_insights)} Key Points</span>
                        </div>
                        <div>
                            {insights_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # BOX 5 — DATA / CHARTS (Strictly rendered only if validated source-backed chart exists)
            charts = analysis.get("charts", [])
            if charts:
                for idx, ch in enumerate(charts):
                    st.markdown(
                        f"""
                        <div class="structured-card">
                            <div class="card-header-bar">
                                <div class="card-title-group">
                                    <span>📊</span>
                                    <span>Data Visualization: {ch['title']}</span>
                                </div>
                                <span style="font-size: 0.74rem; font-weight: 600; color: #4338ca; background: #eef2ff; padding: 2px 7px; border-radius: 6px;">Verified Source Data</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Prepare verified chart data
                    df_chart = pd.DataFrame(ch["data"]).set_index("label")
                    chart_type = ch.get("chart_type", "bar")
                    if chart_type == "line":
                        st.line_chart(df_chart, use_container_width=True)
                    elif chart_type == "area":
                        st.area_chart(df_chart, use_container_width=True)
                    else:
                        st.bar_chart(df_chart, use_container_width=True)

                    st.markdown(
                        f"""
                            <div class="card-footer-scope">
                                {ch.get('source_citation', f'Source: {cur_url}')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # SOURCE Q&A CHAT (Source-Scoped to Current URL / Video)
            q_heading = "💬 Ask About This Video" if is_yt_page else "💬 Ask About This Website"
            st.markdown(f"### {q_heading}")
            st.caption(f"Questions asked here are strictly grounded in: **{cur_title}**")

            # Render Per-Analysis Isolated Messages
            url_messages = analysis.get("messages", [])
            for msg in url_messages:
                if msg["role"] == "user":
                    t_str = msg.get("time", "Just now")
                    u_content = html.escape(str(msg.get("content", "")))
                    user_msg_html = (
                        '<div class="user-msg-row">'
                        '<div style="text-align: right; max-width: 82%;">'
                        f'<div style="font-size: 0.75rem; color: #64748b; margin-bottom: 3px;">You • {t_str}</div>'
                        f'<div class="user-msg-bubble">{u_content}</div>'
                        '</div>'
                        '</div>'
                    )
                    st.markdown(user_msg_html, unsafe_allow_html=True)
                else:
                    ans = msg["content"]
                    msg_citations = msg.get("citations", [])
                    cite_tags = ""
                    if msg_citations:
                        chip_list = []
                        for c in msg_citations:
                            if c.get("source_type") == "youtube" and c.get("timestamp_formatted"):
                                ts_link = c.get("timestamp_url") or cur_url
                                chip_list.append(
                                    f'<a href="{ts_link}" target="_blank" rel="noopener noreferrer" style="background:#fee2e2; color:#dc2626; padding:2px 8px; border-radius:6px; font-size:0.75rem; text-decoration:none; font-weight:600; margin-right:4px;">▶ {c["formatted"]} ↗</a>'
                                )
                            else:
                                sec_lbl = c.get("heading") or "Section"
                                chip_list.append(
                                    f'<span style="background:#eef2ff; color:#4338ca; padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; margin-right:4px;">📌 {sec_lbl}</span>'
                                )
                        cite_tags = f'<div style="margin-top: 8px; border-top: 1px solid #f1f5f9; padding-top: 6px;">{"".join(chip_list)}</div>'

                    bubble_html = format_chat_bubble_html(ans)
                    ai_msg_html = (
                        '<div class="ai-msg-row">'
                        '<div style="width: 32px; height: 32px; border-radius: 8px; background: #e0e7ff; color: #4338ca; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">✨</div>'
                        '<div class="ai-msg-bubble" style="flex: 1;">'
                        f'<div style="margin-bottom: 4px;">{bubble_html}</div>'
                        f'{cite_tags}'
                        '</div>'
                        '</div>'
                    )
                    st.markdown(ai_msg_html, unsafe_allow_html=True)
                    if debug_mode_enabled and msg.get("debug_info"):
                        render_retrieval_debug_expander(msg["debug_info"])

            # Isolated Chat Input Dock for This Source
            chat_ph = "Ask a question about this video..." if is_yt_page else "Ask a question about this website..."
            website_query = st.chat_input(chat_ph, key="source_chat_input_dock")
            if website_query:
                user_time = datetime.now().strftime("%I:%M %p")
                analysis["messages"].append({"role": "user", "content": website_query, "time": user_time})

                norm_q = " ".join(website_query.strip().lower().split())
                cache_key = (
                    cur_url,
                    analysis.get("content_hash", ""),
                    norm_q,
                    "source_scoped",
                    answer_mode,
                    RAG_PIPELINE_VERSION,
                    top_k,
                )

                if "query_cache" not in st.session_state:
                    st.session_state.query_cache = {}

                if cache_key in st.session_state.query_cache:
                    cached_res = st.session_state.query_cache[cache_key]
                    clean_ans = sanitize_final_response(cached_res.get("answer", ""))
                    analysis["messages"].append({
                        "role": "assistant",
                        "content": clean_ans,
                        "citations": cached_res.get("citations", []),
                        "debug_info": cached_res.get("debug_info"),
                    })
                    st.rerun()

                with st.spinner(f"Searching verified sections of {cur_title}..."):
                    try:
                        res = query_rag_pipeline(
                            vector_store=st.session_state.vector_store,
                            question=website_query,
                            top_k=top_k,
                            llm_model=st.session_state.selected_model,
                            api_key=active_key,
                            source_filter=cur_url,
                            source_type=analysis.get("source_type", "youtube" if is_yt_page else "webpage"),
                            answer_mode=answer_mode,
                        )
                        st.session_state.query_cache[cache_key] = res
                        clean_ans = sanitize_final_response(res.get("answer", ""))
                        analysis["messages"].append({
                            "role": "assistant",
                            "content": clean_ans,
                            "citations": res.get("citations", []),
                            "debug_info": res.get("debug_info"),
                        })
                        st.rerun()
                    except Exception as q_err:
                        st.error(f"Error querying source: {str(q_err)}")

        with col_main_right:
            # BOX 3 — IMPORTANT LINKS
            links = analysis.get("links", [])
            if links:
                links_html = "".join(
                    f'<a href="{lnk["url"]}" target="_blank" rel="noopener noreferrer" class="link-chip">'
                    f'<span class="link-chip-text">🔗 {lnk["text"]} ↗</span>'
                    f'<span class="link-chip-url">{lnk["url"]}</span>'
                    f'</a>'
                    for lnk in links[:12]
                )
                st.markdown(
                    f"""
                    <div class="structured-card">
                        <div class="card-header-bar">
                            <div class="card-title-group">
                                <span>🔗</span>
                                <span>Important Links</span>
                            </div>
                            <span style="font-size: 0.74rem; font-weight: 600; color: #4338ca; background: #eef2ff; padding: 2px 7px; border-radius: 6px;">{len(links)} Links</span>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 8px;">
                            {links_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # BOX 4 — OTHER INFORMATION
            meta_info = analysis.get("other_information", {})
            if meta_info:
                meta_rows = []
                channel_val = meta_info.get("channel") or (meta_info.get("author") if is_yt_page else None)
                if channel_val:
                    meta_rows.append(f'<div class="meta-tag-pill">👤 <strong>Channel:</strong> {channel_val}</div>')
                elif meta_info.get("author"):
                    meta_rows.append(f'<div class="meta-tag-pill">✍️ <strong>Author:</strong> {meta_info["author"]}</div>')
                if meta_info.get("date"):
                    meta_rows.append(f'<div class="meta-tag-pill">📅 <strong>Date:</strong> {meta_info["date"]}</div>')
                if meta_info.get("category"):
                    meta_rows.append(f'<div class="meta-tag-pill">🏷️ <strong>Category:</strong> {meta_info["category"]}</div>')
                if meta_info.get("topics"):
                    for tp in meta_info["topics"][:5]:
                        meta_rows.append(f'<div class="meta-tag-pill"># {tp}</div>')

                if meta_rows:
                    st.markdown(
                        f"""
                        <div class="structured-card">
                            <div class="card-header-bar">
                                <div class="card-title-group">
                                    <span>📌</span>
                                    <span>Other Information</span>
                                </div>
                            </div>
                            <div class="meta-chip-row">
                                {''.join(meta_rows)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # BOX 6 — SOURCES & CITATIONS
            citations = analysis.get("citations", [])
            st.markdown(
                f"""
                <div class="structured-card">
                    <div class="card-header-bar">
                        <div class="card-title-group">
                            <span>🔎</span>
                            <span>Sources & Citations</span>
                        </div>
                        <span style="font-size: 0.74rem; font-weight: 600; color: #0d9488; background: #f0fdf4; padding: 2px 7px; border-radius: 6px;">{len(citations)} Sections</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if citations:
                for idx, c in enumerate(citations[:8], start=1):
                    snip = c.get("snippet", "")
                    if is_yt_page and c.get("timestamp_formatted"):
                        loc_label = c.get("timestamp_formatted")
                        ts_target_url = c.get("timestamp_url") or cur_url
                        badge_html = f'<span style="font-size: 0.72rem; font-weight: 600; color: #dc2626; background: #fee2e2; padding: 2px 6px; border-radius: 4px;">{loc_label}</span>'
                        open_html = f'<a href="{ts_target_url}" target="_blank" rel="noopener noreferrer" style="font-size: 0.76rem; font-weight: 600; color: #dc2626; text-decoration: underline;">▶ Open Video at {loc_label} ↗</a>'
                    else:
                        head = c.get("heading") or "Overview"
                        badge_html = f'<span style="font-size: 0.72rem; font-weight: 600; color: #4338ca; background: #e0e7ff; padding: 2px 6px; border-radius: 4px;">{head}</span>'
                        open_html = f'<a href="{cur_url}" target="_blank" rel="noopener noreferrer" style="font-size: 0.76rem; font-weight: 600; color: #4338ca; text-decoration: underline;">Open Source ↗</a>'

                    st.markdown(
                        f"""
                        <div class="source-citation-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-weight: 750; font-size: 0.85rem; color: #131b2e;">{idx}. {cur_title}</span>
                                {badge_html}
                            </div>
                            <div style="font-size: 0.8rem; color: #475569; margin-bottom: 6px;">"{snip}"</div>
                            {open_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No sections available for citation.")

            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 3: GLOBAL KNOWLEDGE CHAT (Cross-Source Knowledge Chat)
# =========================================================================
elif st.session_state.current_view == "knowledge_chat":
    st.markdown(
        """
        <div class="stitch-top-header">
            <div class="stitch-breadcrumbs">
                <span>Workspace</span>
                <span style="color: #94a3b8; font-weight: bold;">/</span>
                <span class="stitch-breadcrumbs-active">Global Knowledge Chat</span>
            </div>
            <div style="background: #f0fdf4; border: 1px solid #ccfbf1; color: #0d9488; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; font-family: 'Geist', sans-serif; display: flex; align-items: center; gap: 6px;">
                <span class="pulse-dot"></span>
                Cross-Source Intelligence Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_chat, col_citations = st.columns([7, 4], gap="large")

    with col_chat:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="structured-card" style="padding: 2rem;">
                    <h3 style="margin-top: 0; color: #131b2e;">💬 Global Knowledge Chat</h3>
                    <p style="color: #64748b; font-size: 0.92rem;">
                        Ask questions across all indexed PDF documents and webpages. DocuMind AI will retrieve relevant passages across all sources and cite verified evidence.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                t_str = msg.get("time", "Just now")
                u_content = html.escape(str(msg.get("content", "")))
                user_msg_html = (
                    '<div class="user-msg-row">'
                    '<div style="text-align: right; max-width: 82%;">'
                    f'<div style="font-size: 0.75rem; color: #64748b; margin-bottom: 3px;">Ganesh • {t_str}</div>'
                    f'<div class="user-msg-bubble">{u_content}</div>'
                    '</div>'
                    '</div>'
                )
                st.markdown(user_msg_html, unsafe_allow_html=True)
            else:
                citations = msg.get("citations", [])
                latency_val = msg.get("latency", "350ms")
                citation_tags = ""
                for c in citations:
                    if c.get("source_type") == "youtube":
                        ts_url = c.get("timestamp_url") or c["source"]
                        citation_tags += f'<a href="{ts_url}" target="_blank" rel="noopener noreferrer" style="background:#fee2e2; color:#dc2626; padding:2px 8px; border-radius:6px; font-size:0.75rem; text-decoration:none; font-weight:600; margin-right:4px;">▶ {c["formatted"]} ↗</a> '
                    elif c.get("source_type") in ("url", "webpage"):
                        citation_tags += f'<a href="{c["source"]}" target="_blank" rel="noopener noreferrer" style="background:#eef2ff; color:#4338ca; padding:2px 8px; border-radius:6px; font-size:0.75rem; text-decoration:none; font-weight:600; margin-right:4px;">🌐 {c["formatted"]} ↗</a> '
                    else:
                        citation_tags += f'<span style="background:#f1f5f9; color:#334155; padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; margin-right:4px;">📌 {c["formatted"]}</span> '

                home_bubble_html = format_chat_bubble_html(msg["content"])
                meta_bar = (
                    f'<div style="font-size: 0.78rem; color: #64748b; margin-bottom: 6px; display: flex; justify-content: space-between;">'
                    f'<span>✓ Grounded in {len(citations)} source(s)</span>'
                    f'<span>⚡ {latency_val}</span>'
                    f'</div>'
                )
                cite_footer = f'<div style="margin-top: 8px; border-top: 1px solid #f1f5f9; padding-top: 6px;">{citation_tags}</div>' if citation_tags else ''
                global_ai_msg = (
                    '<div class="ai-msg-row">'
                    '<div style="width: 34px; height: 34px; border-radius: 10px; background: #4338ca; color: #ffffff; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">✨</div>'
                    '<div class="ai-msg-bubble" style="flex: 1;">'
                    f'{meta_bar}'
                    f'<div style="margin-bottom: 8px;">{home_bubble_html}</div>'
                    f'{cite_footer}'
                    '</div>'
                    '</div>'
                )
                st.markdown(global_ai_msg, unsafe_allow_html=True)
                if debug_mode_enabled and msg.get("debug_info"):
                    render_retrieval_debug_expander(msg["debug_info"])

    with col_citations:
        citation_header_card = (
            '<div class="structured-card">'
            '<div class="card-header-bar">'
            '<div class="card-title-group">'
            '<span>🔎</span>'
            '<span>Latest Citations</span>'
            '</div>'
            '</div>'
        )
        st.markdown(citation_header_card, unsafe_allow_html=True)

        latest_cites = []
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and msg.get("citations"):
                latest_cites = msg["citations"]
                break

        if latest_cites:
            for idx, c in enumerate(latest_cites[:6], start=1):
                src_tp = c.get("source_type")
                if src_tp == "youtube":
                    loc = f"{c.get('timestamp_formatted', 'Video')}"
                    tag_bg = "background: #fee2e2; color: #dc2626;"
                    open_lnk = f'<a href="{c.get("timestamp_url") or c["source"]}" target="_blank" rel="noopener noreferrer" style="font-size: 0.76rem; font-weight: 600; color: #dc2626; text-decoration: underline; margin-top: 4px; display: inline-block;">▶ Open Video at {loc} ↗</a>'
                elif src_tp in ("url", "webpage"):
                    loc = f"Section: {c.get('heading', 'Web')}"
                    tag_bg = "background: #e0e7ff; color: #4338ca;"
                    open_lnk = f'<a href="{c["source"]}" target="_blank" rel="noopener noreferrer" style="font-size: 0.76rem; font-weight: 600; color: #4338ca; text-decoration: underline; margin-top: 4px; display: inline-block;">Open Webpage ↗</a>'
                else:
                    loc = f"Page {c.get('page', 1)}"
                    tag_bg = "background: #f1f5f9; color: #334155;"
                    open_lnk = ""

                st.markdown(
                    f"""
                    <div class="source-citation-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 750; font-size: 0.85rem; color: #131b2e;">{idx}. {c.get('title') or c['source']}</span>
                            <span style="font-size: 0.72rem; font-weight: 600; {tag_bg} padding: 2px 6px; border-radius: 4px;">{loc}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569;">"{c.get('snippet', '')}"</div>
                        {open_lnk}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Ask a cross-document question to view verified evidence passages.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Universal Chat Input Dock
    global_query = st.session_state.pending_query or st.chat_input("Ask anything across your documents and websites...")
    st.session_state.pending_query = None

    if global_query:
        if not st.session_state.kb_stats["processed"] or st.session_state.vector_store is None:
            st.warning("⚠️ Please upload documents or add a website in the sidebar before querying.")
        elif not is_groq_configured(active_key):
            st.error("⚠️ Groq API key is missing or invalid. Please configure it in the sidebar.")
        else:
            cur_time_str = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": global_query, "time": cur_time_str})
            t0 = time.time()

            norm_gq = " ".join(global_query.strip().lower().split())
            cache_key = (
                "global_kb",
                "all_indexed_sources",
                norm_gq,
                "cross_source",
                answer_mode,
                RAG_PIPELINE_VERSION,
                top_k,
            )

            if "query_cache" not in st.session_state:
                st.session_state.query_cache = {}

            if cache_key in st.session_state.query_cache:
                res = st.session_state.query_cache[cache_key]
                elapsed_ms = 5
                clean_ans = sanitize_final_response(res.get("answer", ""))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clean_ans,
                    "citations": res.get("citations", []),
                    "latency": f"{elapsed_ms}ms (cached)",
                    "debug_info": res.get("debug_info"),
                })
                st.rerun()

            with st.spinner("Searching across all sources..."):
                try:
                    res = query_rag_pipeline(
                        vector_store=st.session_state.vector_store,
                        question=global_query,
                        top_k=top_k,
                        llm_model=st.session_state.selected_model,
                        api_key=active_key,
                        source_filter=None,  # Cross-document general mode
                        answer_mode=answer_mode,
                    )
                    st.session_state.query_cache[cache_key] = res
                    clean_ans = sanitize_final_response(res.get("answer", ""))
                    elapsed_ms = int((time.time() - t0) * 1000)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_ans,
                        "citations": res.get("citations", []),
                        "latency": f"{elapsed_ms}ms",
                        "debug_info": res.get("debug_info"),
                    })
                    st.rerun()
                except Exception as g_err:
                    st.error(f"Failed to generate answer: {str(g_err)}")
