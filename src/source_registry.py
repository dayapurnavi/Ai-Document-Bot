"""Source registry for tracking and deduplicating indexed knowledge sources."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import VECTORSTORE_DIR

REGISTRY_FILENAME = "sources.json"


def compute_content_hash(content: bytes | str) -> str:
    """Compute SHA-256 hash for document content (bytes or text string)."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_registry_path(folder_path: Path = VECTORSTORE_DIR) -> Path:
    """Return the Path to the sources.json registry file."""
    return Path(folder_path) / REGISTRY_FILENAME


def load_source_registry(folder_path: Path = VECTORSTORE_DIR) -> Dict[str, Any]:
    """Load the source registry from disk."""
    reg_path = get_registry_path(folder_path)
    if not reg_path.exists():
        return {"sources": {}}

    try:
        content = reg_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, dict) and "sources" in data:
            return data
        return {"sources": {}}
    except Exception:
        return {"sources": {}}


def save_source_registry(
    registry: Dict[str, Any], folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Persist the source registry to disk safely."""
    try:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        reg_path = get_registry_path(folder)

        # Temporary file write for atomic update
        tmp_path = folder / f"{REGISTRY_FILENAME}.tmp"
        tmp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        tmp_path.replace(reg_path)
        return True
    except Exception:
        return False


def is_source_indexed(
    content_hash: str, folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Check if a source with the exact content hash is already registered."""
    reg = load_source_registry(folder_path)
    sources = reg.get("sources", {})
    return any(s.get("content_hash") == content_hash for s in sources.values())


def get_source_by_hash(
    content_hash: str, folder_path: Path = VECTORSTORE_DIR
) -> Optional[Dict[str, Any]]:
    """Retrieve registered source metadata by its content hash."""
    reg = load_source_registry(folder_path)
    for src_id, src in reg.get("sources", {}).items():
        if src.get("content_hash") == content_hash:
            result = dict(src)
            result["source_id"] = src_id
            return result
    return None


def get_source_by_name(
    source_name: str, folder_path: Path = VECTORSTORE_DIR
) -> Optional[Dict[str, Any]]:
    """Retrieve registered source metadata by source filename or identifier."""
    reg = load_source_registry(folder_path)
    for src_id, src in reg.get("sources", {}).items():
        if src.get("source") == source_name:
            result = dict(src)
            result["source_id"] = src_id
            return result
    return None


def register_source(
    source: str,
    source_type: str = "pdf",
    title: Optional[str] = None,
    content_hash: str = "",
    chunk_count: int = 0,
    total_pages: Optional[int] = None,
    folder_path: Path = VECTORSTORE_DIR,
) -> Dict[str, Any]:
    """Register or update a source record in the persistent registry."""
    reg = load_source_registry(folder_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    source_id = f"{source_type}_{content_hash[:12]}" if content_hash else f"{source_type}_{hashlib.sha256(source.encode()).hexdigest()[:12]}"

    source_entry = {
        "source_id": source_id,
        "source_type": source_type,
        "source": source,
        "title": title or source,
        "content_hash": content_hash,
        "chunk_count": chunk_count,
        "total_pages": total_pages,
        "created_at": reg.get("sources", {}).get(source_id, {}).get("created_at", now_iso),
        "updated_at": now_iso,
    }

    if "sources" not in reg:
        reg["sources"] = {}
    reg["sources"][source_id] = source_entry

    save_source_registry(reg, folder_path)
    return source_entry


def unregister_source(
    source_id: str, folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Remove a source entry from the registry."""
    reg = load_source_registry(folder_path)
    if "sources" in reg and source_id in reg["sources"]:
        del reg["sources"][source_id]
        return save_source_registry(reg, folder_path)
    return False


def unregister_source_by_name_or_hash(
    identifier: str, folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Remove a source by its name, hash, or source_id."""
    reg = load_source_registry(folder_path)
    sources = reg.get("sources", {})
    to_delete = []

    for sid, s in sources.items():
        if sid == identifier or s.get("source") == identifier or s.get("content_hash") == identifier:
            to_delete.append(sid)

    if to_delete:
        for sid in to_delete:
            del sources[sid]
        reg["sources"] = sources
        return save_source_registry(reg, folder_path)
    return False


def clear_source_registry(folder_path: Path = VECTORSTORE_DIR) -> bool:
    """Delete the source registry file from disk."""
    reg_path = get_registry_path(folder_path)
    if reg_path.exists():
        try:
            reg_path.unlink()
        except Exception:
            return False
    return True


def list_registered_sources(
    folder_path: Path = VECTORSTORE_DIR,
) -> List[Dict[str, Any]]:
    """List all registered sources."""
    reg = load_source_registry(folder_path)
    return list(reg.get("sources", {}).values())


def get_source_by_url(
    url: str, folder_path: Path = VECTORSTORE_DIR
) -> Optional[Dict[str, Any]]:
    """Retrieve registered source metadata by its URL."""
    reg = load_source_registry(folder_path)
    for src_id, src in reg.get("sources", {}).items():
        if src.get("source") == url:
            result = dict(src)
            result["source_id"] = src_id
            return result
    return None


def is_url_indexed(
    url: str, folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Check if a URL is already registered in the knowledge base."""
    return get_source_by_url(url, folder_path) is not None


def has_url_content_changed(
    url: str, new_content_hash: str, folder_path: Path = VECTORSTORE_DIR
) -> bool:
    """Check if an already-indexed URL has different content compared to registry."""
    existing = get_source_by_url(url, folder_path)
    if not existing:
        return False
    return existing.get("content_hash") != new_content_hash
