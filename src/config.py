"""Configuration module for DocuMind AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# Also support Streamlit Community Cloud st.secrets
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for secret_key in ("GROQ_API_KEY", "GROQ_MODEL", "EMBEDDING_MODEL_NAME"):
            if secret_key in st.secrets and not os.getenv(secret_key):
                os.environ[secret_key] = str(st.secrets[secret_key])
except Exception:
    pass

# API Keys and Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
).strip()

# Chunking Configuration
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
except ValueError:
    CHUNK_SIZE = 1000

try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
except ValueError:
    CHUNK_OVERLAP = 150

# Retrieval Configuration
try:
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "6"))
except ValueError:
    DEFAULT_TOP_K = 6

# Storage Paths
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def is_groq_configured(key: str | None = None) -> bool:
    """Check if a valid, non-placeholder Groq API key is present."""
    k = key if key is not None else GROQ_API_KEY
    if not k:
        return False
    k = k.strip()
    placeholder_keys = {
        "your_groq_api_key_here",
        "your_actual_api_key_here",
        "your_api_key_here",
        "gsk_your_actual_api_key_here",
    }
    if k.lower() in placeholder_keys or k.startswith("your_"):
        return False
    return len(k) > 15


def update_groq_api_key(new_key: str):
    """Save newly provided Groq API key to .env file and update runtime config."""
    global GROQ_API_KEY
    new_key = new_key.strip()
    GROQ_API_KEY = new_key
    os.environ["GROQ_API_KEY"] = new_key
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if "GROQ_API_KEY=" in content:
            import re
            content = re.sub(r"GROQ_API_KEY=.*", f"GROQ_API_KEY={new_key}", content)
        else:
            content += f"\nGROQ_API_KEY={new_key}\n"
        env_file.write_text(content, encoding="utf-8")
    else:
        env_file.write_text(f"GROQ_API_KEY={new_key}\nGROQ_MODEL={GROQ_MODEL}\n", encoding="utf-8")


def reload_env_config():
    """Reload environment variables if changed at runtime."""
    global GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K
    load_dotenv(BASE_DIR / ".env", override=True)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    try:
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    except ValueError:
        CHUNK_SIZE = 1000
    try:
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    except ValueError:
        CHUNK_OVERLAP = 150
    try:
        DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "6"))
    except ValueError:
        DEFAULT_TOP_K = 6
