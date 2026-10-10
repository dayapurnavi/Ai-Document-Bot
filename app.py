"""DocuMind AI Assistant - Exact Stitch UI Replication.

1:1 Direct Reproduction of the Stitch-generated UI (Dual Page UI Replicator):
- Stitch Left Sidebar (320px width, #121417 in dark / #FAFAFA in light):
  * + New Chat button (12px rounded-xl, 42px height, exact Stitch colors)
  * Search chats... input (dark #1a1d21 / light #ffffff, rounded-lg/xl)
  * RECENTS section with clean session history or "No recent conversations."
  * Pinned User Profile at the absolute bottom (Rose avatar DP, Daya Purnavi, no Patient label)
- Stitch Main Area (#0a0b0d in dark / #ffffff in light):
  * Clean Top Navbar with theme toggle (☀️ / 🌙) and engine settings (⚙️)
  * Center Hero section: "Where should we start?" + "Ask questions about your uploaded documents or attach medical records below."
  * Conversational chat messages with user bubbles and assistant responses + citations
- Stitch Bottom Dock:
  * Pill container (#16181d / #FBFBFC, 16px rounded-2xl, shadow-xl)
  * Document upload (+) attachment inside chat input
  * Models selector chip popover
  * Grok API key chip popover with emerald status indicator
  * Lavender circular send button with purple arrow
"""

import html
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
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
from src.embeddings import get_embedding_model
from src.vector_store import (
    add_documents_to_vector_store,
    save_vector_store,
    load_vector_store,
    clear_persisted_vector_store,
)
from src.source_registry import (
    compute_content_hash,
    is_source_indexed,
    register_source,
    unregister_source_by_name_or_hash,
    list_registered_sources,
)
from src.rag_chain import (
    query_rag_pipeline,
    REFUSAL_PHRASE,
    sanitize_final_response,
    RAG_PIPELINE_VERSION,
)

st.set_page_config(
    page_title="My AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Hide Streamlit top header */
header[data-testid="stHeader"] {
    display: none;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

/* Hide Streamlit deploy button */
.stDeployButton {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION & STATE HANDLING ---
def init_session_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "workspace_name" not in st.session_state:
        st.session_state.workspace_name = "DocuMind AI Assistant"
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Daya Purnavi"
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = GROQ_API_KEY if is_groq_configured(GROQ_API_KEY) else ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = (
            GROQ_MODEL
            if GROQ_MODEL in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
            else "openai/gpt-oss-20b"
        )
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "answer_mode" not in st.session_state:
        st.session_state.answer_mode = "STRICT_SOURCE"
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "active_documents" not in st.session_state:
        st.session_state.active_documents = []
    if "mobile_sidebar_open" not in st.session_state:
        st.session_state.mobile_sidebar_open = False

    # Restore persisted vector store and registered sources if available
    if st.session_state.vector_store is None:
        try:
            persisted_vs = load_vector_store()
            if persisted_vs is not None:
                st.session_state.vector_store = persisted_vs
        except Exception:
            st.session_state.vector_store = None

    # Sync active_documents from vector store or source registry
    if not st.session_state.active_documents:
        if st.session_state.vector_store is not None and hasattr(st.session_state.vector_store, "docstore") and hasattr(st.session_state.vector_store.docstore, "_dict"):
            try:
                seen_sources = set()
                for doc in st.session_state.vector_store.docstore._dict.values():
                    s_name = doc.metadata.get("title") or doc.metadata.get("source") or "Document"
                    if s_name not in seen_sources:
                        seen_sources.add(s_name)
                        st.session_state.active_documents.append({
                            "name": s_name,
                            "source": s_name,
                            "chunks": 1,
                            "pages": doc.metadata.get("total_pages", 1),
                        })
            except Exception:
                pass
        if not st.session_state.active_documents:
            try:
                reg_sources = list_registered_sources()
                for s in reg_sources:
                    st.session_state.active_documents.append({
                        "name": s.get("title") or s.get("source"),
                        "source": s.get("source"),
                        "chunks": s.get("chunk_count", 0),
                        "pages": s.get("total_pages", 1),
                    })
            except Exception:
                pass


init_session_state()


def clear_knowledge_base():
    """Reset the vector database and source registry."""
    clear_persisted_vector_store()
    st.session_state.vector_store = None
    st.session_state.query_cache = {}
    st.session_state.active_documents = []


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


# --- THEME VARIABLES & EXACT STITCH DESIGN TOKENS ---
is_dark = st.session_state.theme == "dark"
is_mobile_open = st.session_state.get("mobile_sidebar_open", False)

bg_app = "#0a0b0d" if is_dark else "#ffffff"
bg_sidebar = "#121417" if is_dark else "#FAFAFA"
sidebar_border = "#1e2227" if is_dark else "rgba(229, 231, 235, 0.8)"
text_main = "#ffffff" if is_dark else "#111827"
text_muted = "#a1a1aa" if is_dark else "#6b7280"
hero_sub = "#d4d4d8" if is_dark else "#6b7280"

new_chat_bg = "#1a1d22" if is_dark else "#ffffff"
new_chat_border = "#2b3038" if is_dark else "rgba(229, 231, 235, 0.9)"
new_chat_text = "#ffffff" if is_dark else "#1f2937"
new_chat_hover = "#23272e" if is_dark else "#f9fafb"

search_bg = "#1a1d21" if is_dark else "#ffffff"
search_border = "#262a30" if is_dark else "rgba(229, 231, 235, 0.9)"

dock_bg = "#16181d" if is_dark else "#FBFBFC"
dock_border = "#2b3038" if is_dark else "rgba(229, 231, 235, 0.8)"
dock_shadow = "0 20px 25px -5px rgba(0, 0, 0, 0.5)" if is_dark else "0 2px 10px rgba(0, 0, 0, 0.04)"

chip_bg = "#1f232b" if is_dark else "#ffffff"
chip_border = "#2e3440" if is_dark else "rgba(229, 231, 235, 0.6)"
chip_hover = "#282d37" if is_dark else "#f3f4f6"
chip_text = "#e4e4e7" if is_dark else "#4b5563"

chip_grok_bg = "#1a1d24" if is_dark else "#ffffff"
chip_grok_border = "#2d323e" if is_dark else "rgba(229, 231, 235, 0.6)"
chip_grok_hover = "#232731" if is_dark else "#f3f4f6"
chip_grok_text = "#d4d4d8" if is_dark else "#4b5563"

send_btn_bg = "#f1ebf9" if is_dark else "#EDE9FE"
send_btn_hover = "#e7dcf5" if is_dark else "#E4DEFD"
send_btn_icon = "#7c5fa6" if is_dark else "#7C3AED"

user_bubble_bg = "#1a1d22" if is_dark else "#f3f4f6"
user_bubble_border = "#2b3038" if is_dark else "#e5e7eb"
user_bubble_text = "#f4f4f5" if is_dark else "#111827"

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* ============================================================
       SECTION 1 — NUKE ALL DEFAULT STREAMLIT CHROME
       Remove every built-in Streamlit visual that is NOT in Stitch
    ============================================================ */

    /* Hide ALL Streamlit Cloud watermarks, badges, footer, header, toolbar, buttons */
    #MainMenu,
    footer,
    header,
    header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarHeader"],
    [data-testid="stLogoSpacer"],
    [data-testid="stSidebarResizeHandle"],
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    [data-testid="stStatusWidget"],
    div[class*="viewerBadge"],
    div[class*="ViewerBadge"],
    a[class*="viewerBadge"],
    a[class*="ViewerBadge"],
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    #manage-app-button,
    button[title="Manage app"],
    [data-testid="manage-app-button"],
    [data-testid="stManageAppButton"],
    .reportview-container footer,
    .reportview-container .main footer,
    div[data-testid="stCustomComponentV1"] iframe,
    svg[class*="streamlit"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
    }}

    /* Zero out ALL default Streamlit spacing/padding/margins */
    html, body {{
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }}

    html, body, .stApp {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background-color: {bg_app} !important;
        color: {text_main} !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }}

    /* Center the main block-container and match exactly with the bottom dock */
    .main .block-container,
    div[data-testid="stMainBlockContainer"],
    .block-container {{
        padding-top: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 14rem !important;
        max-width: 820px !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }}

    /* Remove default vertical gaps from stVerticalBlock and stElementContainer */
    div[data-testid="stVerticalBlock"] {{
        gap: 0 !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    div[data-testid="stElementContainer"] {{
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Remove default column borders and gaps */
    div[data-testid="stHorizontalBlock"] {{
        gap: 0 !important;
        border: none !important;
    }}
    div[data-testid="column"] {{
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }}

    /* ============================================================
       SECTION 2 — STITCH EXACT TWO-COLUMN LAYOUT
       DESKTOP (min-width: 769px): 320px fixed sidebar | remaining main canvas
    ============================================================ */

    @media (min-width: 769px) {{
        section[data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            bottom: 0 !important;
            width: 320px !important;
            min-width: 320px !important;
            max-width: 320px !important;
            height: 100vh !important;
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {sidebar_border} !important;
            transform: none !important;
            margin: 0 !important;
            padding: 0 !important;
            visibility: visible !important;
            display: flex !important;
            flex-direction: column !important;
            z-index: 100 !important;
            overflow: hidden !important;
            pointer-events: auto !important;
        }}

        /* Main area offset on desktop */
        .stApp > .main,
        section.stMain,
        section[data-testid="stAppScrollToBottomContainer"],
        section[data-testid="stMain"],
        section.main {{
            margin-left: 320px !important;
            width: calc(100% - 320px) !important;
            max-width: calc(100% - 320px) !important;
            min-height: 100vh !important;
            background-color: {bg_app} !important;
            box-sizing: border-box !important;
        }}

        div[data-testid="stBottom"] {{
            left: 320px !important;
            width: calc(100% - 320px) !important;
        }}

        div.st-key-btn_mobile_open_sidebar,
        div.st-key-btn_mobile_close_sidebar,
        .mobile-sidebar-backdrop {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
    }}

    /* stSidebarContent — the Streamlit-generated wrapper: full height flex column */
    div[data-testid="stSidebarContent"] {{
        padding: 1rem !important;
        height: 100vh !important;
        min-height: 100vh !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        background-color: {bg_sidebar} !important;
        flex: 1 !important;
    }}

    /* stSidebarUserContent — user's rendered widgets */
    div[data-testid="stSidebarUserContent"] {{
        padding: 0 !important;
        margin: 0 !important;
        height: 100% !important;
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        min-height: 0 !important;
    }}

    /* The auto-generated wrapper div inside stSidebarUserContent that Streamlit
       sets margin-top: 591px on — zero that out completely */
    div[data-testid="stSidebarUserContent"] > div {{
        margin: 0 !important;
        padding: 0 !important;
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;
        min-height: 0 !important;
    }}

    /* Vertical block inside sidebar fills full height as flex column */
    div[data-testid="stSidebarUserContent"] div[data-testid="stVerticalBlock"] {{
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 0 !important;
        width: 100% !important;
        min-height: 0 !important;
    }}

    /* Each element container inside sidebar */
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"] {{
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }}

    /* Profile box element container — auto margin pushes it to the very bottom */
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"]:has(.user-profile-box) {{
        margin-top: auto !important;
    }}

    /* ============================================================
       SECTION 3 — STITCH SIDEBAR COMPONENTS
    ============================================================ */

    /* New Chat button — full width, rounded-xl, 42px, exact Stitch colors */
    div.st-key-btn_new_chat {{
        width: 100% !important;
        margin-bottom: 1rem !important;
    }}
    div.st-key-btn_new_chat button {{
        background-color: {new_chat_bg} !important;
        border: 1px solid {new_chat_border} !important;
        border-radius: 12px !important;
        color: {new_chat_text} !important;
        font-weight: {'500' if is_dark else '600'} !important;
        font-size: 0.875rem !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        width: 100% !important;
        height: 42px !important;
        padding: 0 1rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        box-sizing: border-box !important;
        box-shadow: {'none' if is_dark else '0 1px 2px rgba(0,0,0,0.03)'} !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }}
    div.st-key-btn_new_chat button::before {{
        content: '';
        display: inline-block;
        width: 16px;
        height: 16px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='{'white' if is_dark else '%231f2937'}' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M12 4v16m8-8H4'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
        flex-shrink: 0;
    }}
    div.st-key-btn_new_chat button:hover {{
        background-color: {new_chat_hover} !important;
        border-color: {'#3f4654' if is_dark else '#d1d5db'} !important;
    }}
    div.st-key-btn_new_chat p {{
        margin: 0 !important;
        color: {new_chat_text} !important;
        font-weight: inherit !important;
    }}

    /* Search input — Stitch rounded-lg/xl, exact dark/light bg */
    div.st-key-search_chats_input {{
        width: 100% !important;
        margin-bottom: 1rem !important;
    }}
    div.st-key-search_chats_input div[data-baseweb="base-input"] {{
        position: relative !important;
        background-color: {search_bg} !important;
        border: 1px solid {search_border} !important;
        border-radius: {'8px' if is_dark else '12px'} !important;
    }}
    div.st-key-search_chats_input div[data-baseweb="base-input"]::before {{
        content: '';
        position: absolute;
        left: 12px;
        top: 50%;
        transform: translateY(-50%);
        width: 16px;
        height: 16px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='{'%2371717a' if is_dark else '%239ca3af'}' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-size: contain;
        pointer-events: none;
        z-index: 2;
    }}
    div.st-key-search_chats_input,
    div.st-key-search_chats_input > div,
    div.st-key-search_chats_input > div > div,
    div.st-key-search_chats_input div[data-baseweb="input"] {{
        background-color: {search_bg} !important;
        border-radius: {'8px' if is_dark else '12px'} !important;
        border: none !important;
    }}
    div.st-key-search_chats_input input {{
        background-color: {search_bg} !important;
        border: none !important;
        border-radius: {'8px' if is_dark else '12px'} !important;
        color: {text_main} !important;
        font-size: 0.875rem !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        width: 100% !important;
        height: 38px !important;
        padding: 0 0.85rem 0 38px !important;
        box-sizing: border-box !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    div.st-key-search_chats_input input::placeholder {{
        color: {'#71717a' if is_dark else '#9ca3af'} !important;
        opacity: 1 !important;
    }}
    div.st-key-search_chats_input input:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    /* RECENTS header */
    .recents-header {{
        font-size: {'0.75rem' if is_dark else '0.6875rem'};
        font-weight: {'600' if is_dark else '700'};
        color: {'#a1a1aa' if is_dark else '#9ca3af'};
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1rem 0 0.75rem 0;
        padding-left: 0.25rem;
        display: block;
    }}
    .empty-recents-text {{
        font-size: 0.875rem;
        color: {'#a1a1aa' if is_dark else '#6b7280'};
        padding-left: 0.25rem;
        font-weight: 400;
        display: block;
    }}

    /* Explicit margins between sidebar elements for clean rhythm */
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"]:has(div.st-key-btn_new_chat) {{
        margin-bottom: 12px !important;
    }}
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"]:has(div.st-key-search_chats_input) {{
        margin-bottom: 16px !important;
    }}
    div[data-testid="stSidebarUserContent"] div[data-testid="stElementContainer"]:has(.recents-header) {{
        margin-top: 6px !important;
        margin-bottom: 8px !important;
    }}

    /* Session buttons inside sidebar */
    [data-testid="stSidebar"] div[class*="st-key-session_btn_"] button,
    section[data-testid="stSidebar"] div[class*="st-key-session_btn_"] button {{
        background-color: {new_chat_bg} !important;
        border: 1px solid {new_chat_border} !important;
        color: {text_main} !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.8125rem !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        padding: 6px 10px !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        width: 100% !important;
        height: 38px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }}
    [data-testid="stSidebar"] div[class*="st-key-session_btn_"] button:hover,
    section[data-testid="stSidebar"] div[class*="st-key-session_btn_"] button:hover {{
        background-color: {new_chat_hover} !important;
        border-color: {'#3f4654' if is_dark else '#d1d5db'} !important;
    }}
    [data-testid="stSidebar"] div[class*="st-key-session_btn_act_"] button,
    section[data-testid="stSidebar"] div[class*="st-key-session_btn_act_"] button {{
        background-color: {'#232730' if is_dark else '#ebeef2'} !important;
        border: 1px solid {'#3b4252' if is_dark else '#d1d5db'} !important;
        color: {'#ffffff' if is_dark else '#111827'} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] div[class*="st-key-del_session_"] button,
    section[data-testid="stSidebar"] div[class*="st-key-del_session_"] button {{
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: {'#71717a' if is_dark else '#9ca3af'} !important;
        font-size: 0.75rem !important;
        padding: 0 !important;
        box-shadow: none !important;
        width: 100% !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stSidebar"] div[class*="st-key-del_session_"] button:hover,
    section[data-testid="stSidebar"] div[class*="st-key-del_session_"] button:hover {{
        color: #ef4444 !important;
        background-color: {'#2a1717' if is_dark else '#fee2e2'} !important;
        border-color: {'#ef4444' if is_dark else '#fca5a5'} !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-session_btn_"]) {{
        gap: 6px !important;
        align-items: center !important;
        margin-bottom: 6px !important;
    }}

    /* User Profile — pinned at absolute bottom of sidebar */
    div[data-testid="stSidebarUserContent"] > div:has(.user-profile-box) {{
        margin-top: auto !important;
        position: sticky !important;
        bottom: 0 !important;
        background-color: {bg_sidebar} !important;
        z-index: 10 !important;
        width: 100% !important;
        padding-top: 0.5rem !important;
    }}
    .user-profile-box {{
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        padding: 0.75rem 0.25rem 0.25rem 0.25rem !important;
        border-top: 1px solid transparent !important;
        background-color: {bg_sidebar} !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }}
    .avatar-circle {{
        width: {'40px' if is_dark else '36px'} !important;
        height: {'40px' if is_dark else '36px'} !important;
        min-width: {'40px' if is_dark else '36px'} !important;
        border-radius: 50% !important;
        background-color: #e11d48 !important;
        color: #ffffff !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: {'700' if is_dark else '600'} !important;
        font-size: {'0.875rem' if is_dark else '0.75rem'} !important;
        flex-shrink: 0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }}
    .user-name {{
        font-weight: 600 !important;
        font-size: {'1rem' if is_dark else '0.875rem'} !important;
        color: {text_main} !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        letter-spacing: {'normal' if is_dark else '-0.01em'} !important;
    }}

    /* ============================================================
       SECTION 4 — STITCH MAIN CANVAS CONTROLS
       Theme + settings buttons — positioned top-right, hidden by default
       (Not in Stitch UI — appears only on hover for functionality)
    ============================================================ */

    /* The st.columns navbar row — shrink it to zero height */
    div.st-key-btn_theme,
    div.st-key-btn_settings {{
        position: fixed !important;
        top: 12px !important;
        z-index: 200 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 36px !important;
        height: 36px !important;
        opacity: 0 !important;
        transition: opacity 0.2s ease !important;
    }}
    div.st-key-btn_theme:hover,
    div.st-key-btn_settings:hover {{
        opacity: 1 !important;
    }}
    div.st-key-btn_theme {{
        right: 56px !important;
    }}
    div.st-key-btn_settings {{
        right: 12px !important;
    }}
    div.st-key-btn_theme button,
    div.st-key-btn_settings button {{
        background-color: {new_chat_bg} !important;
        border: 1px solid {new_chat_border} !important;
        border-radius: 8px !important;
        color: {text_main} !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
    }}
    div.st-key-btn_theme button:hover,
    div.st-key-btn_settings button:hover {{
        background-color: {new_chat_hover} !important;
        border-color: {'#3f4654' if is_dark else '#d1d5db'} !important;
    }}
    div.st-key-btn_theme p,
    div.st-key-btn_settings p {{
        margin: 0 !important;
        font-size: 0.95rem !important;
    }}

    /* The containing navbar column row — collapse it completely */
    div[data-testid="stHorizontalBlock"]:has(div.st-key-btn_theme),
    div[data-testid="stHorizontalBlock"]:has(div.st-key-btn_settings) {{
        height: 0 !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        border: none !important;
    }}
    div[data-testid="column"]:has(div.st-key-btn_theme),
    div[data-testid="column"]:has(div.st-key-btn_settings),
    div[data-testid="column"]:has(span[data-testid="stText"]) {{
        height: 0 !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
    }}

    /* ============================================================
       SECTION 5 — STITCH CENTER HERO
    ============================================================ */
    .stitch-hero {{
        text-align: center;
        max-width: 672px;
        margin: 4rem auto 2rem auto;
        padding: 0 1rem;
    }}
    .stitch-hero-title {{
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: {'2.25rem' if is_dark else '2.125rem'} !important;
        font-weight: 700 !important;
        color: {text_main} !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 0.75rem !important;
        line-height: 1.2 !important;
    }}
    .stitch-hero-subtitle {{
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: {'1rem' if is_dark else '0.9375rem'} !important;
        color: {hero_sub} !important;
        line-height: 1.625 !important;
        max-width: 520px !important;
        margin: 0 auto !important;
        font-weight: 400 !important;
    }}

    /* Welcome Architecture & Getting Started Guide Section */
    .welcome-guide-container {{
        max-width: 820px;
        margin: 1.5rem auto 2.5rem auto;
        padding: 0 1rem;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .welcome-header {{
        text-align: left;
        margin-bottom: 1.5rem;
    }}
    .welcome-title {{
        font-size: {'2.25rem' if is_dark else '2.125rem'} !important;
        font-weight: 800 !important;
        color: {'#3b82f6' if is_dark else '#2563eb'} !important;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
    }}
    .welcome-subtitle {{
        font-size: {'0.9375rem' if is_dark else '0.9375rem'} !important;
        color: {'#94a3b8' if is_dark else '#64748b'} !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }}
    .rag-architecture-card {{
        background-color: {'#0f172a' if is_dark else '#ffffff'};
        border: 1px solid {'#1e293b' if is_dark else '#e2e8f0'};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: {'0 4px 20px rgba(0, 0, 0, 0.25)' if is_dark else '0 4px 16px rgba(0, 0, 0, 0.05)'};
    }}
    .rag-arch-title {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: {'#f8fafc' if is_dark else '#0f172a'} !important;
        margin: 0 0 10px 0 !important;
    }}
    .rag-arch-desc {{
        font-size: 0.875rem !important;
        line-height: 1.55 !important;
        color: {'#94a3b8' if is_dark else '#64748b'} !important;
        margin: 0 0 18px 0 !important;
    }}
    .rag-flow-container {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        row-gap: 10px;
    }}
    .rag-pill {{
        background-color: {'#0369a126' if is_dark else '#e0f2fe'};
        border: 1px solid {'#0284c7' if is_dark else '#38bdf8'};
        color: {'#38bdf8' if is_dark else '#0284c7'};
        font-size: 0.75rem;
        font-weight: 500;
        padding: 5px 12px;
        border-radius: 9999px;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }}
    .rag-arrow {{
        color: {'#64748b' if is_dark else '#94a3b8'};
        font-size: 0.8125rem;
        font-weight: 600;
    }}
    .get-started-section {{
        margin-top: 24px;
    }}
    .get-started-title {{
        font-size: 1.125rem !important;
        font-weight: 700 !important;
        color: {'#f8fafc' if is_dark else '#0f172a'} !important;
        margin: 0 0 16px 0 !important;
    }}
    .get-started-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 20px;
    }}
    .get-started-col {{
        display: flex;
        flex-direction: column;
    }}
    .step-num-title {{
        font-size: 0.9375rem;
        font-weight: 700;
        color: {'#f8fafc' if is_dark else '#0f172a'};
        margin-bottom: 6px;
    }}
    .step-desc {{
        font-size: 0.84375rem;
        line-height: 1.45;
        color: {'#94a3b8' if is_dark else '#64748b'};
    }}
    .step-desc strong {{
        color: {'#ffffff' if is_dark else '#0f172a'};
        font-weight: 600;
    }}
    .get-started-banner {{
        background-color: {'#0c2b45' if is_dark else '#f0fdf4'};
        border: 1px solid {'#0369a1' if is_dark else '#86efac'};
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 14px;
    }}
    .banner-icon {{
        font-size: 1.125rem;
    }}
    .banner-text {{
        font-size: 0.875rem;
        color: {'#38bdf8' if is_dark else '#0284c7'};
    }}
    .banner-text strong {{
        color: {'#60a5fa' if is_dark else '#0369a1'};
        font-weight: 700;
    }}

    /* ============================================================
       SECTION 6 — STITCH BOTTOM INPUT DOCK
    ============================================================ */
    div[data-testid="stBottom"] {{
        left: 0 !important;
        width: 100% !important;
        background: {'rgba(10, 11, 13, 0.97)' if is_dark else 'rgba(255, 255, 255, 0.97)'} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        padding-top: 14px !important;
        padding-bottom: 24px !important;
        border-top: 1px solid {'rgba(255, 255, 255, 0.06)' if is_dark else 'rgba(0, 0, 0, 0.06)'} !important;
    }}
    /* Streamlit inner wrapper with lavender/white auto-bg — force transparent */
    div[data-testid="stBottom"] > div:first-child {{
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}
    div[data-testid="stBottomBlockContainer"] {{
        position: relative !important;
        max-width: 820px !important;
        margin: 0 auto !important;
        padding: 0 1.5rem !important;
        background: transparent !important;
        background-color: transparent !important;
    }}
    div[data-testid="stBottom"] div[data-testid="stVerticalBlock"] {{
        gap: 6px !important;
        background: transparent !important;
    }}
    div[data-testid="stBottom"] div[data-testid="stElementContainer"] {{
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }}
    div[data-testid="stBottom"] div[data-testid="stLayoutWrapper"] {{
        background: transparent !important;
    }}
    /* Horizontal block with Models & Grok API chips & Active Docs */
    div[data-testid="stBottom"] div[data-testid="stHorizontalBlock"] {{
        max-width: 820px !important;
        margin: 0 auto 8px auto !important;
        gap: 8px !important;
        display: flex !important;
        align-items: center !important;
        background: transparent !important;
    }}
    div[data-testid="stBottom"] div[data-testid="column"],
    div[data-testid="stBottom"] div[data-testid="stColumn"] {{
        min-width: unset !important;
        padding: 0 !important;
        flex: 0 0 auto !important;
        height: auto !important;
        overflow: visible !important;
        background: transparent !important;
    }}

    /* Models / Grok chip popovers */
    div[data-testid="stPopover"] button {{
        background-color: {chip_bg} !important;
        border: 1px solid {chip_border} !important;
        border-radius: 8px !important;
        color: {chip_text} !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        padding: 2px 10px !important;
        height: 28px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
        box-shadow: none !important;
        white-space: nowrap !important;
        transition: background-color 0.15s ease !important;
    }}
    div[data-testid="stPopover"] button:hover {{
        background-color: {chip_hover} !important;
        border-color: {'#3b4252' if is_dark else '#d1d5db'} !important;
        color: {'#ffffff' if is_dark else '#111827'} !important;
    }}
    div[data-testid="stPopover"] p {{
        margin: 0 !important;
        font-size: 0.75rem !important;
        color: inherit !important;
    }}

    /* Active Document Fixed Pill in Question Bar */
    div[class*="st-key-btn_rm_doc_"] button {{
        background-color: {'#1c2027' if is_dark else '#f3f4f6'} !important;
        border: 1px solid {'#2e3440' if is_dark else '#e5e7eb'} !important;
        border-radius: 8px !important;
        color: {'#e4e4e7' if is_dark else '#374151'} !important;
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        padding: 2px 10px !important;
        height: 28px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        box-shadow: none !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }}
    div[class*="st-key-btn_rm_doc_"] button:hover {{
        background-color: {'#3b1818' if is_dark else '#fee2e2'} !important;
        border-color: #ef4444 !important;
        color: {'#fca5a5' if is_dark else '#b91c1c'} !important;
    }}
    div[class*="st-key-btn_rm_doc_"] p {{
        margin: 0 !important;
        font-size: 0.75rem !important;
        color: inherit !important;
    }}

    /* Chat input pill — 16px rounded-2xl with clean margins and internal padding */
    div[data-testid="stChatInput"] {{
        border: 1px solid {dock_border} !important;
        border-radius: 16px !important;
        box-shadow: {dock_shadow} !important;
        background: {dock_bg} !important;
        transition: all 0.2s ease !important;
        padding: 6px 14px !important;
        margin-top: 4px !important;
        margin-bottom: 6px !important;
    }}
    /* Override Streamlit's auto-generated inner div that gets white/lavender bg */
    div[data-testid="stChatInput"] > div {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    div[data-testid="stChatInput"] div {{
        background: transparent !important;
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: {'#52525b' if is_dark else '#d1d5db'} !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: 0.9375rem !important;
        color: {text_main} !important;
        padding: 8px 6px !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {'#71717a' if is_dark else '#9ca3af'} !important;
    }}

    /* Send button — lavender circle */
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    div[data-testid="stChatInput"] button:last-child {{
        background-color: {send_btn_bg} !important;
        border-radius: {'9999px' if is_dark else '12px'} !important;
        width: {'34px' if is_dark else '36px'} !important;
        height: {'34px' if is_dark else '36px'} !important;
        min-width: {'34px' if is_dark else '36px'} !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        box-shadow: none !important;
        flex-shrink: 0 !important;
        transition: background-color 0.15s ease !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]:hover,
    div[data-testid="stChatInput"] button:last-child:hover {{
        background-color: {send_btn_hover} !important;
    }}
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg,
    div[data-testid="stChatInput"] button:last-child svg {{
        color: {send_btn_icon} !important;
        fill: {send_btn_icon} !important;
        stroke: {send_btn_icon} !important;
    }}

    /* Attach (+) button */
    div[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]):first-of-type {{
        color: {'#a1a1aa' if is_dark else '#9ca3af'} !important;
        background: transparent !important;
        border-radius: 8px !important;
        padding: 4px !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]):first-of-type:hover {{
        color: {'#ffffff' if is_dark else '#1f2937'} !important;
        background-color: {'#23272e' if is_dark else '#f3f4f6'} !important;
    }}

    /* Hide char counter */
    div[data-testid="stChatInput"] [data-testid="stChatInputCharCounter"],
    div[data-testid="stChatInput"] small {{
        display: none !important;
    }}

    /* ============================================================
       SECTION 7 — CHAT BUBBLES
    ============================================================ */
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 2.25rem;
        margin-top: 1.5rem;
        margin-bottom: 3.5rem;
        padding-top: 1rem;
        width: 100%;
        max-width: 820px;
        margin-left: auto;
        margin-right: auto;
        box-sizing: border-box;
    }}
    .user-msg-row {{
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin-bottom: 0.5rem;
    }}
    .user-msg-bubble {{
        background-color: {user_bubble_bg};
        color: {user_bubble_text};
        border: 1px solid {user_bubble_border};
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.25rem;
        font-size: 0.9375rem;
        line-height: 1.55;
        max-width: 82%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
        word-wrap: break-word;
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .assistant-msg-row {{
        display: flex;
        flex-direction: column;
        width: 100%;
        gap: 12px;
        padding: 0.25rem 0;
    }}
    .assistant-msg-bubble {{
        background: transparent;
        color: {text_main};
        font-size: 0.95rem;
        line-height: 1.7;
        width: 100%;
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .action-icons-bar {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 6px;
        color: {text_muted};
        font-size: 0.875rem;
    }}
    .action-icon {{
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.15s ease;
        user-select: none;
    }}
    .action-icon:hover {{
        opacity: 1;
        color: {text_main};
    }}
    .citations-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed {sidebar_border};
    }}
    .citation-badge {{
        background: {dock_bg};
        color: {text_muted};
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        border: 1px solid {dock_border};
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: 'Inter', system-ui, sans-serif;
    }}

    /* ============================================================
       SECTION 8 — RESPONSIVE (MOBILE & ANDROID RATIO)
    ============================================================ */
    @media (max-width: 768px) {{
        /* Sidebar: sliding drawer overlay */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"][aria-expanded="true"],
        section[data-testid="stSidebar"][aria-expanded="false"] {{
            position: fixed !important;
            top: 0 !important;
            left: {'0' if is_mobile_open else '-320px'} !important;
            bottom: 0 !important;
            width: 290px !important;
            min-width: 290px !important;
            max-width: 85vw !important;
            height: 100vh !important;
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {sidebar_border} !important;
            transform: none !important;
            transition: left 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
            z-index: 10000 !important;
            box-shadow: {'4px 0 28px rgba(0, 0, 0, 0.65)' if is_mobile_open else 'none'} !important;
            pointer-events: {'auto' if is_mobile_open else 'none'} !important;
            visibility: {'visible' if is_mobile_open else 'hidden'} !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        /* Mobile backdrop overlay */
        .mobile-sidebar-backdrop {{
            display: {'block' if is_mobile_open else 'none'} !important;
            position: fixed !important;
            inset: 0 !important;
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(4px) !important;
            -webkit-backdrop-filter: blur(4px) !important;
            z-index: 9999 !important;
            cursor: pointer !important;
        }}

        /* Main canvas takes 100% full screen width */
        .stApp > .main,
        section.stMain,
        section[data-testid="stAppScrollToBottomContainer"],
        section[data-testid="stMain"],
        section.main {{
            margin-left: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            box-sizing: border-box !important;
        }}

        div[data-testid="stMainBlockContainer"],
        .block-container {{
            padding-left: 14px !important;
            padding-right: 14px !important;
            padding-top: 54px !important;
            padding-bottom: 120px !important;
            width: 100% !important;
            max-width: 100% !important;
            left: 0 !important;
            box-sizing: border-box !important;
        }}

        /* Hero typography on Android screens */
        .stitch-hero {{
            max-width: 100% !important;
            margin: 2.5rem auto 1.5rem auto !important;
            padding: 0 12px !important;
        }}
        .stitch-hero-title {{
            font-size: 1.625rem !important; /* 26px */
            line-height: 1.25 !important;
            margin-bottom: 0.5rem !important;
            word-wrap: break-word !important;
        }}
        .stitch-hero-subtitle {{
            font-size: 0.875rem !important; /* 14px */
            line-height: 1.45 !important;
            max-width: 100% !important;
        }}

        .welcome-guide-container {{
            max-width: 100% !important;
            margin: 1.5rem auto 1.5rem auto !important;
            padding: 0 10px !important;
        }}
        .welcome-title {{
            font-size: 1.75rem !important;
        }}
        .rag-architecture-card {{
            padding: 16px 14px !important;
        }}
        .get-started-grid {{
            grid-template-columns: 1fr !important;
            gap: 14px !important;
        }}

        /* Mobile Bottom Dock */
        div[data-testid="stBottom"] {{
            left: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
            padding-bottom: 14px !important;
            box-sizing: border-box !important;
        }}
        div[data-testid="stBottomBlockContainer"] {{
            max-width: 100% !important;
            padding: 0 !important;
        }}
        div[data-testid="stBottom"] div[data-testid="stHorizontalBlock"] {{
            max-width: 100% !important;
            padding: 0 2px !important;
            gap: 6px !important;
        }}
        div[data-testid="stChatInput"] {{
            max-width: 100% !important;
            padding: 3px 8px !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            font-size: 0.875rem !important;
        }}

        /* Mobile open button at top-left */
        div.st-key-btn_mobile_open_sidebar {{
            position: fixed !important;
            top: 10px !important;
            left: 10px !important;
            z-index: 998 !important;
            display: block !important;
            width: 36px !important;
            height: 36px !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div.st-key-btn_mobile_open_sidebar button {{
            background-color: {new_chat_bg} !important;
            border: 1px solid {new_chat_border} !important;
            border-radius: 8px !important;
            color: {text_main} !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.15rem !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
        }}
        div.st-key-btn_mobile_open_sidebar button:hover {{
            background-color: {new_chat_hover} !important;
        }}
        div.st-key-btn_mobile_open_sidebar div,
        div.st-key-btn_mobile_open_sidebar span,
        div.st-key-btn_mobile_open_sidebar p {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            margin: 0 !important;
            font-size: 1.15rem !important;
            color: {text_main} !important;
        }}

        /* Mobile close button header inside sidebar */
        div.st-key-btn_mobile_close_sidebar {{
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: 100% !important;
            margin-bottom: 12px !important;
            padding-bottom: 6px !important;
            border-bottom: 1px solid {sidebar_border} !important;
        }}
        div.st-key-btn_mobile_close_sidebar::before {{
            content: 'DocuMind AI' !important;
            font-family: 'Inter', system-ui, sans-serif !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            color: {text_main} !important;
            letter-spacing: -0.01em !important;
            display: flex !important;
            align-items: center !important;
        }}
        div.st-key-btn_mobile_close_sidebar > div {{
            display: flex !important;
            justify-content: flex-end !important;
            width: auto !important;
            margin-left: auto !important;
        }}
        div.st-key-btn_mobile_close_sidebar button {{
            background-color: transparent !important;
            border: 1px solid {sidebar_border} !important;
            border-radius: 8px !important;
            color: {text_muted} !important;
            width: 32px !important;
            height: 32px !important;
            min-width: 32px !important;
            max-width: 32px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.95rem !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
        }}
        div.st-key-btn_mobile_close_sidebar button:hover {{
            color: {text_main} !important;
            background-color: {new_chat_hover} !important;
            border-color: {new_chat_border} !important;
        }}
        div.st-key-btn_mobile_close_sidebar div,
        div.st-key-btn_mobile_close_sidebar span,
        div.st-key-btn_mobile_close_sidebar p {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            margin: 0 !important;
            font-size: 0.95rem !important;
            color: {text_muted} !important;
        }}

        div.st-key-btn_theme {{
            right: 52px !important;
            top: 10px !important;
            opacity: 0.8 !important;
        }}
        div.st-key-btn_settings {{
            right: 10px !important;
            top: 10px !important;
            opacity: 0.8 !important;
        }}
    }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <script>
    (function() {
        function purgeStreamlitBadges() {
            const selectors = [
                'div[class*="viewerBadge"]',
                'a[class*="viewerBadge"]',
                'div[class*="ViewerBadge"]',
                'a[class*="ViewerBadge"]',
                '.viewerBadge_container__1QSob',
                '.viewerBadge_link__1S137',
                '#manage-app-button',
                'button[title="Manage app"]',
                '[data-testid="manage-app-button"]',
                '[data-testid="stManageAppButton"]',
                '.stDeployButton',
                'footer',
                '#MainMenu'
            ];
            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.opacity = '0';
                    el.remove();
                });
            });

            // Actively remove any Grok API button if rendered by old cache
            document.querySelectorAll('button').forEach(btn => {
                const text = (btn.innerText || btn.textContent || '').toLowerCase();
                if (text.includes('grok') || text.includes('groq api')) {
                    const pop = btn.closest('div[data-testid="stPopover"]');
                    if (pop) { pop.style.display = 'none'; pop.remove(); }
                    btn.style.display = 'none';
                    btn.remove();
                }
            });
        }
        purgeStreamlitBadges();
        const obs = new MutationObserver(purgeStreamlitBadges);
        obs.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)



# --- GENERAL AI ANSWER GENERATOR ---
def get_general_ai_answer(
    question: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    has_docs: bool = False,
) -> str:
    """
    Generate authoritative, comprehensive answers for any user question.
    Answers general knowledge, machine learning (Deep Learning, NumPy, Scikit-learn, Pandas),
    medical concepts, coding, workflows, and handles typos like 'pands' for 'pandas'.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    key = api_key or GROQ_API_KEY
    model = llm_model or GROQ_MODEL

    system_prompt = (
        "You are DocuMind AI, an expert, intelligent AI assistant.\n"
        "Your task is to provide a clear, accurate, thorough, and well-structured answer to the user's question.\n"
        "- Format using clean GitHub-flavored Markdown with bold headings, bullet points, and code snippets where helpful.\n"
        "- If the user input contains typos (for example 'pands' for pandas, 'nump' for numpy, 'pytn' for python, 'scikit learn' for scikit-learn), "
        "interpret the user's intended concept accurately and provide a complete, expert explanation.\n"
        "- NEVER output raw HTML tags (such as <div>, <span>, <p>, <br>).\n"
        "- Provide comprehensive explanations with key concepts, practical examples, and use cases.\n"
        "- Be polite, professional, and directly address the user's query."
    )

    try:
        llm = ChatGroq(model=model, api_key=key, temperature=0.2, max_retries=3)
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
        raw_text = resp.content if hasattr(resp, "content") else str(resp)
        return sanitize_final_response(raw_text)
    except Exception as e:
        return f"Unable to generate response at this time: {str(e)}"


# --- WORKSPACE & ASSISTANT SETTINGS DIALOG ---
@st.dialog("⚙️ Workspace & Engine Settings")
def show_settings_dialog():
    st.markdown("Configure your AI model, document grounding, Groq API key, and profile.")

    # 1. Profile
    st.markdown("##### 👤 User Profile")
    u_name = st.text_input("Name", value=st.session_state.user_name)
    if u_name.strip():
        st.session_state.user_name = u_name.strip()

    # 2. Document Upload Option Inside Settings
    st.markdown("##### 📄 Knowledge Documents")
    uploaded_docs = st.file_uploader(
        "Upload Documents / Records",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, TXT, MD, lab reports, or prescriptions.",
        key="settings_doc_uploader",
    )
    if uploaded_docs:
        file_tuples = [(f.getvalue(), f.name, compute_content_hash(f.getvalue())) for f in uploaded_docs]
        if st.button("⚡ Index Uploaded Documents", type="primary", use_container_width=True):
            chunks, stats = process_pdf_files(file_tuples, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            if chunks:
                embs = get_embedding_model()
                st.session_state.vector_store = add_documents_to_vector_store(
                    st.session_state.vector_store, chunks, embs
                )
                save_vector_store(st.session_state.vector_store)
                for pf in stats["processed_files"]:
                    st_type = "pdf" if pf["filename"].lower().endswith(".pdf") else "document"
                    register_source(
                        source=pf["filename"],
                        source_type=st_type,
                        title=pf["filename"],
                        content_hash=pf.get("content_hash", ""),
                        chunk_count=pf["chunks"],
                        total_pages=pf["pages"],
                        metadata={"indexed_at": datetime.now().isoformat()},
                    )
                    if not any(d.get("name") == pf["filename"] for d in st.session_state.active_documents):
                        st.session_state.active_documents.append({
                            "name": pf["filename"],
                            "source": pf["filename"],
                            "pages": pf.get("pages", 1),
                            "chunks": pf.get("chunks", 0),
                        })
                st.toast(f"Successfully indexed {len(chunks)} chunks and pinned to question bar!", icon="✅")
                st.rerun()

    # 3. Groq API Engine
    st.markdown("##### 🔑 Groq API Engine")
    new_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Free API key from console.groq.com/keys",
    )
    if new_key.strip() != st.session_state.groq_api_key:
        st.session_state.groq_api_key = new_key.strip()
        update_groq_api_key(new_key.strip())
        st.toast("Groq API Key updated!", icon="🔑")

    # 4. Model Selector
    st.markdown("##### ⚡ AI Model")
    model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    st.session_state.selected_model = st.selectbox("LLM Model", options=model_options, index=idx)

    # 5. Citations & Grounding
    st.markdown("##### 🎯 Retrieval Parameters")
    st.session_state.top_k = st.slider(
        "Top-K Document Citations",
        min_value=1,
        max_value=15,
        value=st.session_state.top_k,
    )
    ans_mode = st.radio(
        "Grounding Mode",
        options=["🌐 Document + Knowledge Fallback (Answers All Questions)", "🔒 Strict Document / Record Only"],
        index=0 if st.session_state.answer_mode == "SOURCE_FIRST_WITH_FALLBACK" else 1,
    )
    st.session_state.answer_mode = "SOURCE_FIRST_WITH_FALLBACK" if "Fallback" in ans_mode else "STRICT_SOURCE"

    st.markdown("<hr style='margin: 1.25rem 0; border-color: #21262d;'>", unsafe_allow_html=True)
    c_clear, c_close = st.columns([1, 1])
    with c_clear:
        if st.button("🗑️ Clear Indexed Documents", use_container_width=True):
            clear_knowledge_base()
            st.toast("Knowledge base cleared.", icon="🗑️")
            st.rerun()
    with c_close:
        if st.button("✓ Apply & Close", type="primary", use_container_width=True):
            st.rerun()


# --- EXACT STITCH LEFT SIDEBAR ---
with st.sidebar:
    # Mobile-only close button header (brand title styled via CSS ::before)
    if st.button("✕", key="btn_mobile_close_sidebar"):
        st.session_state.mobile_sidebar_open = False
        st.rerun()

    # 1. New Chat Button (Stitch styled with SVG icon)
    if st.button("New Chat", key="btn_new_chat", use_container_width=True):
        st.session_state.mobile_sidebar_open = False
        new_session_id = f"session_{len(st.session_state.chat_sessions) + 1}_{int(time.time())}"
        st.session_state.chat_sessions[new_session_id] = {
            "title": "New Conversation",
            "messages": [],
            "created_at": datetime.now().strftime("%I:%M %p"),
        }
        st.session_state.current_session_id = new_session_id
        st.rerun()

    # 2. Search chats... input box (Stitch styled with SVG icon)
    search_query = st.text_input(
        "Search chats...",
        placeholder="Search chats...",
        label_visibility="collapsed",
        key="search_chats_input",
    )

    # 3. Recents Section
    st.markdown('<div class="recents-header">RECENTS</div>', unsafe_allow_html=True)

    session_items = list(reversed(list(st.session_state.chat_sessions.items())))
    if search_query.strip():
        session_items = [
            (s_id, s_data)
            for s_id, s_data in session_items
            if search_query.strip().lower() in s_data.get("title", "").lower()
        ]

    if not session_items:
        st.markdown('<div class="empty-recents-text">No recent conversations.</div>', unsafe_allow_html=True)
    else:
        for s_id, s_data in session_items:
            is_active = s_id == st.session_state.current_session_id
            s_title = s_data.get("title", "Conversation")
            btn_label = f"💬  {s_title}" if is_active else f"     {s_title}"

            s_key = f"session_btn_act_{s_id}" if is_active else f"session_btn_{s_id}"
            col_s_btn, col_s_del = st.columns([5, 1])
            with col_s_btn:
                if st.button(btn_label, key=s_key, use_container_width=True):
                    st.session_state.mobile_sidebar_open = False
                    st.session_state.current_session_id = s_id
                    st.rerun()
            with col_s_del:
                if len(st.session_state.chat_sessions) > 0:
                    if st.button("✕", key=f"del_session_{s_id}", help="Delete chat"):
                        del st.session_state.chat_sessions[s_id]
                        if st.session_state.current_session_id == s_id:
                            st.session_state.current_session_id = (
                                list(st.session_state.chat_sessions.keys())[0]
                                if st.session_state.chat_sessions
                                else None
                            )
                        st.rerun()

    # 4. User Profile Pinned at the Absolute Bottom (Daya Purnavi, DP Avatar, NO Patient text)
    st.markdown(
        f"""
        <div class="user-profile-box">
            <div class="avatar-circle">DP</div>
            <div class="user-name">Daya Purnavi</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- TOP NAVBAR CONTROLS (positioned-fixed via CSS, zero layout height) ---
col_nav_left, col_nav_right = st.columns([1, 9])
with col_nav_left:
    if st.button("☰", key="btn_mobile_open_sidebar"):
        st.session_state.mobile_sidebar_open = True
        st.rerun()

with col_nav_right:
    c_th, c_set = st.columns([1, 1])
    with c_th:
        theme_icon = "☀️" if is_dark else "🌙"
        if st.button(theme_icon, key="btn_theme", help="Toggle Dark / Light Mode"):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
    with c_set:
        if st.button("⚙️", key="btn_settings", help="Settings & Engine Config"):
            show_settings_dialog()

# Render mobile backdrop when sidebar is open on mobile
if is_mobile_open:
    st.markdown(
        """
        <div class="mobile-sidebar-backdrop" onclick="const b = document.querySelector('div.st-key-btn_mobile_close_sidebar button'); if(b) b.click();"></div>
        """,
        unsafe_allow_html=True,
    )


# --- MAIN CANVAS & MESSAGES ---
cur_session = (
    st.session_state.chat_sessions.get(st.session_state.current_session_id)
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions
    else None
)
messages = cur_session.get("messages", []) if cur_session else []

# Welcome Architecture & Getting Started Guide (displayed when no messages)
if not messages:
    welcome_html = (
        '<div class="welcome-guide-container">'
        '<div class="welcome-header">'
        '<h1 class="welcome-title">DocuMind AI</h1>'
        '<p class="welcome-subtitle">AI-Powered Document Question Answering System with Grounded Citations</p>'
        '</div>'
        '<div class="rag-architecture-card">'
        '<h2 class="rag-arch-title">Enterprise RAG Architecture</h2>'
        '<p class="rag-arch-desc">'
        'DocuMind AI transforms complex enterprise documents into an instant semantic knowledge base. Every answer is strictly grounded in your retrieved document chunks and guarded against hallucination.'
        '</p>'
        '<div class="rag-flow-container">'
        '<span class="rag-pill">1. PDF Upload</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">2. PyMuPDF Extraction</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">3. Recursive Chunking</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">4. MiniLM Embeddings</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">5. FAISS Vector DB</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">6. Semantic Search</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">7. Groq Grounded LLM</span>'
        '<span class="rag-arrow">→</span>'
        '<span class="rag-pill">8. Verified Page Citations</span>'
        '</div>'
        '</div>'
        '<div class="get-started-section">'
        '<h3 class="get-started-title">🚀 How to Get Started:</h3>'
        '<div class="get-started-grid">'
        '<div class="get-started-col">'
        '<div class="step-num-title">1. Upload Documents</div>'
        '<div class="step-desc">Upload one or multiple PDF documents via the sidebar.</div>'
        '</div>'
        '<div class="get-started-col">'
        '<div class="step-num-title">2. Process & Index</div>'
        '<div class="step-desc">Click <strong>Process PDFs</strong> to extract text and generate FAISS vectors.</div>'
        '</div>'
        '<div class="get-started-col">'
        '<div class="step-num-title">3. Ask with Confidence</div>'
        '<div class="step-desc">Ask natural language questions and receive cited, grounded answers.</div>'
        '</div>'
        '</div>'
        '<div class="get-started-banner">'
        '<span class="banner-icon">👉</span>'
        '<span class="banner-text">Upload your PDF files in the sidebar and click <strong>Process PDFs</strong> to start.</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(welcome_html, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in messages:
        if msg["role"] == "user":
            u_text = html.escape(str(msg.get("content", "")))
            st.markdown(
                f"""
                <div class="user-msg-row">
                    <div class="user-msg-bubble">{u_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            ans_raw = msg.get("content", "")
            clean_ans = sanitize_final_response(ans_raw)
            citations = msg.get("citations", [])

            st.markdown('<div class="assistant-msg-row"><div class="assistant-msg-bubble">', unsafe_allow_html=True)
            st.markdown(clean_ans)

            # Action icons bar: 📋, 👍, 👎, 🔊, ⋯
            st.markdown(
                """
                <div class="action-icons-bar">
                    <span class="action-icon" title="Copy">📋</span>
                    <span class="action-icon" title="Helpful">👍</span>
                    <span class="action-icon" title="Not helpful">👎</span>
                    <span class="action-icon" title="Read aloud">🔊</span>
                    <span class="action-icon" title="More">⋯</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Citations
            if citations:
                cite_chips = []
                for c in citations:
                    p_text = f" • Page {c.get('page', 1)}" if c.get("page") else ""
                    s_name = c.get("source", "Document")
                    cite_chips.append(f'<span class="citation-badge">📄 {s_name}{p_text}</span>')
                st.markdown(f'<div class="citations-row">{"".join(cite_chips)}</div>', unsafe_allow_html=True)

            if st.session_state.debug_mode and msg.get("debug_info"):
                with st.expander("🛠️ Retrieval Diagnostics", expanded=False):
                    st.json(msg["debug_info"])

            st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- STITCH BOTTOM INPUT DOCK ---
with st.bottom:
    active_docs = st.session_state.get("active_documents", [])
    if active_docs:
        col_widths = [1.5]
        for _ in active_docs[:3]:
            col_widths.append(2.4)
        total_used = sum(col_widths)
        if total_used < 10:
            col_widths.append(round(10 - total_used, 1))
        dock_cols = st.columns(col_widths)

        with dock_cols[0]:
            with st.popover("✨ Models", use_container_width=True):
                st.markdown("##### ⚡ Select AI Model")
                model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
                idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
                sel = st.selectbox("Model", options=model_options, index=idx, label_visibility="collapsed")
                if sel != st.session_state.selected_model:
                    st.session_state.selected_model = sel
                    st.rerun()

        for d_idx, doc_item in enumerate(active_docs[:3]):
            with dock_cols[1 + d_idx]:
                raw_n = doc_item.get("name", "Document")
                short_n = raw_n if len(raw_n) <= 15 else raw_n[:12] + "..."
                if st.button(f"📄 {short_n}  ✕", key=f"btn_rm_doc_{d_idx}", help=f"Click to remove {raw_n}", use_container_width=True):
                    removed = st.session_state.active_documents.pop(d_idx)
                    try:
                        unregister_source_by_name_or_hash(removed.get("source") or removed.get("name"))
                    except Exception:
                        pass
                    if not st.session_state.active_documents:
                        clear_knowledge_base()
                    st.toast(f"Removed {raw_n} from question bar", icon="🗑️")
                    st.rerun()
    else:
        c_m1, c_m_sp = st.columns([1.6, 8.4])
        with c_m1:
            with st.popover("✨ Models", use_container_width=True):
                st.markdown("##### ⚡ Select AI Model")
                model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
                idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
                sel = st.selectbox("Model", options=model_options, index=idx, label_visibility="collapsed")
                if sel != st.session_state.selected_model:
                    st.session_state.selected_model = sel
                    st.rerun()

    chat_submission = st.chat_input(
        "Describe your symptoms or ask a question...",
        accept_file="multiple",
        file_type=["pdf", "docx", "txt", "md"],
    )

# Handle submission (from chat_input or queued_prompt)
user_prompt_text = ""
attached_files = []

if st.session_state.queued_prompt:
    user_prompt_text = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
elif chat_submission:
    if isinstance(chat_submission, str):
        user_prompt_text = chat_submission
    elif isinstance(chat_submission, dict):
        user_prompt_text = chat_submission.get("text", "") or ""
        attached_files = chat_submission.get("files", []) or []
    else:
        user_prompt_text = getattr(chat_submission, "text", "") or ""
        attached_files = getattr(chat_submission, "files", []) or []

# Ingest any files attached directly in chat_input (+ button)
if attached_files:
    file_tuples = [(f.getvalue(), f.name, compute_content_hash(f.getvalue())) for f in attached_files]
    with st.spinner("Indexing attached document..."):
        try:
            chunks, stats = process_pdf_files(file_tuples, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            if chunks:
                embeddings = get_embedding_model()
                st.session_state.vector_store = add_documents_to_vector_store(
                    st.session_state.vector_store, chunks, embeddings
                )
                save_vector_store(st.session_state.vector_store)
                for pf in stats["processed_files"]:
                    st_type = "pdf" if pf["filename"].lower().endswith(".pdf") else "document"
                    register_source(
                        source=pf["filename"],
                        source_type=st_type,
                        title=pf["filename"],
                        content_hash=pf.get("content_hash", ""),
                        chunk_count=pf["chunks"],
                        total_pages=pf["pages"],
                        metadata={"indexed_at": datetime.now().isoformat()},
                    )
                    # Fixed in question bar: only removed when user clicks ✕!
                    if not any(d.get("name") == pf["filename"] for d in st.session_state.active_documents):
                        st.session_state.active_documents.append({
                            "name": pf["filename"],
                            "source": pf["filename"],
                            "pages": pf.get("pages", 1),
                            "chunks": pf.get("chunks", 0),
                        })
                st.toast(f"Document pinned to question bar! (Remove with ✕)", icon="📎")
        except Exception as file_err:
            st.error(f"Error reading attached document: {file_err}")

    # If the user only uploaded a file without asking a question, rerun immediately so the badge shows!
    if not user_prompt_text.strip():
        st.rerun()

# Process User Question
if user_prompt_text.strip():
    # If no session active, create one now
    if not st.session_state.current_session_id or st.session_state.current_session_id not in st.session_state.chat_sessions:
        new_session_id = f"session_{len(st.session_state.chat_sessions) + 1}_{int(time.time())}"
        st.session_state.chat_sessions[new_session_id] = {
            "title": user_prompt_text[:24].capitalize(),
            "messages": [],
            "created_at": datetime.now().strftime("%I:%M %p"),
        }
        st.session_state.current_session_id = new_session_id

    cur_sess = st.session_state.chat_sessions[st.session_state.current_session_id]
    cur_time = datetime.now().strftime("%I:%M %p")
    cur_sess["messages"].append({
        "role": "user",
        "content": user_prompt_text,
        "time": cur_time,
    })

    # Auto update session title
    if len(cur_sess["messages"]) <= 2:
        clean_title = user_prompt_text.strip().capitalize()
        if len(clean_title) > 26:
            clean_title = clean_title[:23] + "..."
        cur_sess["title"] = clean_title

    # Check Groq Key
    active_key = st.session_state.groq_api_key or GROQ_API_KEY
    if not is_groq_configured(active_key):
        cur_sess["messages"].append({
            "role": "assistant",
            "content": "⚠️ **Groq API Key Required**\nPlease click **⚙️** in the top navbar to configure your free Groq API key.",
            "time": datetime.now().strftime("%I:%M %p"),
            "citations": [],
        })
        st.rerun()

    # Query Knowledge Base & AI Engine (Guaranteed to answer all questions)
    with st.spinner("Analyzing & generating response..."):
        try:
            norm_q = " ".join(user_prompt_text.strip().lower().split())
            cache_key = (
                "chat_kb",
                norm_q,
                st.session_state.answer_mode,
                st.session_state.top_k,
                st.session_state.selected_model,
                RAG_PIPELINE_VERSION,
                bool(st.session_state.vector_store is not None),
            )

            if cache_key in st.session_state.query_cache:
                ans_text, citations = st.session_state.query_cache[cache_key]
            else:
                ans_text = ""
                citations = []

                if st.session_state.vector_store is not None:
                    # 1. Attempt RAG grounded query against verified documents
                    rag_res = query_rag_pipeline(
                        vector_store=st.session_state.vector_store,
                        question=user_prompt_text,
                        top_k=st.session_state.top_k,
                        llm_model=st.session_state.selected_model,
                        api_key=active_key,
                        source_filter=None,
                        source_type="pdf",
                        answer_mode=st.session_state.answer_mode,
                    )

                    # If RAG found document evidence and provided a factual answer
                    if not rag_res.get("is_refusal") and rag_res.get("answer"):
                        ans_text = rag_res["answer"]
                        citations = rag_res.get("citations", [])
                    else:
                        # Refuse when not documented in the indexed PDF
                        ans_text = "I couldn't find that information in the indexed PDF content."
                        citations = []
                else:
                    # No documents uploaded yet: notify user to upload PDF
                    ans_text = "I couldn't find that information in the indexed PDF content. Please upload your PDF files in the sidebar and click **Process PDFs** to start."
                    citations = []

                st.session_state.query_cache[cache_key] = (ans_text, citations)

            cur_sess["messages"].append({
                "role": "assistant",
                "content": ans_text,
                "citations": citations,
                "time": datetime.now().strftime("%I:%M %p"),
            })
            st.rerun()
        except Exception as query_err:
            cur_sess["messages"].append({
                "role": "assistant",
                "content": f"❌ Error retrieving answer: {str(query_err)}",
                "time": datetime.now().strftime("%I:%M %p"),
                "citations": [],
            })
            st.rerun()
