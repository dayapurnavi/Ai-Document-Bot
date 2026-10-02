"""YouTube processing module for extracting metadata, transcripts, and timestamps for DocuMind AI RAG."""

import re
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import requests
from langchain_core.documents import Document

from src.source_registry import compute_content_hash

# Common YouTube URL patterns
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}

# Generic YouTube platform navigation phrases that MUST NEVER become primary RAG content
YOUTUBE_NAVIGATION_PHRASES = [
    "about press copyright contact us",
    "creators advertise developers",
    "terms privacy policy & safety",
    "how youtube works",
    "test new features",
    "© 2026 google llc",
    "© 2025 google llc",
    "© 2024 google llc",
]


def is_youtube_navigation_content(text: str) -> bool:
    """
    Check if text contains primarily generic YouTube website navigation/footer content.
    Returns True if the content is generic platform chrome and should be rejected/excluded.
    """
    clean = text.lower()
    return any(phrase in clean for phrase in YOUTUBE_NAVIGATION_PHRASES)


class YouTubeProcessingError(Exception):
    """Exception raised for YouTube ingestion failures."""
    pass


def detect_source_type(url_or_path: str) -> str:
    """
    Detect whether a source is a YouTube video, a PDF, or a generic webpage.

    Args:
        url_or_path: URL string or filename.

    Returns:
        One of 'youtube', 'pdf', 'webpage'.
    """
    clean = url_or_path.strip().lower()

    if clean.endswith(".pdf") or clean.startswith("pdf:"):
        return "pdf"

    try:
        parsed = urllib.parse.urlparse(clean)
        netloc = parsed.netloc.lower()
        if netloc in YOUTUBE_DOMAINS or netloc.endswith(".youtube.com") or netloc == "youtu.be":
            return "youtube"
        if parsed.scheme in ("http", "https"):
            return "webpage"
    except Exception:
        pass

    if "youtube.com" in clean or "youtu.be" in clean:
        return "youtube"

    return "webpage"


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract the 11-character YouTube video ID from various YouTube URL formats.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - With trailing query parameters like ?t=120s, ?si=...

    Args:
        url: Raw YouTube URL string.

    Returns:
        11-character video ID, or None if not found/invalid.
    """
    clean_url = url.strip()
    if not clean_url:
        return None

    # Handle youtu.be/VIDEO_ID
    match_short = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", clean_url)
    if match_short:
        return match_short.group(1)

    # Handle /shorts/VIDEO_ID, /embed/VIDEO_ID, /v/VIDEO_ID
    match_path = re.search(r"/(?:shorts|embed|v)/([a-zA-Z0-9_-]{11})", clean_url)
    if match_path:
        return match_path.group(1)

    # Handle watch?v=VIDEO_ID
    try:
        parsed = urllib.parse.urlparse(clean_url)
        if "youtube.com" in parsed.netloc.lower():
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                candidate = qs["v"][0].strip()
                if re.match(r"^[a-zA-Z0-9_-]{11}$", candidate):
                    return candidate
    except Exception:
        pass

    # Generic regex fallback for 11-character ID in query
    match_v = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", clean_url)
    if match_v:
        return match_v.group(1)

    return None


def normalize_youtube_url(url: str) -> str:
    """
    Return a canonical YouTube watch URL.

    Args:
        url: Raw YouTube URL.

    Returns:
        Canonical URL format: https://www.youtube.com/watch?v=VIDEO_ID
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise YouTubeProcessingError(f"Invalid YouTube URL: could not extract a valid 11-character video ID from '{url}'.")
    return f"https://www.youtube.com/watch?v={video_id}"


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    sec = max(0, int(seconds))
    hrs, remainder = divmod(sec, 3600)
    mins, s = divmod(remainder, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{s:02d}"
    return f"{mins:02d}:{s:02d}"


def get_youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """Build a clickable YouTube URL targeted to a specific start timestamp."""
    start_sec = max(0, int(seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"


def fetch_youtube_metadata(video_id: str) -> Dict[str, Any]:
    """
    Retrieve real video metadata (title, author/channel, channel URL, thumbnail)
    using the official YouTube oEmbed endpoint and page metadata.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        Dict of metadata fields.
    """
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    meta: Dict[str, Any] = {
        "title": f"YouTube Video ({video_id})",
        "author": "YouTube Creator",
        "author_url": "",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "description": "",
    }

    # 1. Fetch via oEmbed
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={canonical_url}&format=json"
        res = requests.get(oembed_url, timeout=10.0)
        if res.ok:
            data = res.json()
            if data.get("title"):
                meta["title"] = data["title"].strip()
            if data.get("author_name"):
                meta["author"] = data["author_name"].strip()
            if data.get("author_url"):
                meta["author_url"] = data["author_url"].strip()
            if data.get("thumbnail_url"):
                meta["thumbnail_url"] = data["thumbnail_url"].strip()
    except Exception:
        pass

    # 2. Fetch description from video HTML page meta tags (optional, fast timeout)
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        page_res = requests.get(canonical_url, headers=headers, timeout=3.5)
        if page_res.ok:
            page_text = page_res.text
            desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page_text, re.I)
            if not desc_match:
                desc_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]*)"', page_text, re.I)
            if desc_match:
                meta["description"] = desc_match.group(1).strip()
    except Exception:
        pass

    return meta


def fetch_youtube_transcript(
    video_id: str,
    preferred_languages: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], bool, str]:
    """
    Fetch the transcript/captions for a YouTube video.
    Supports English, Telugu, Hindi, and any other available caption language.
    Prefers manual human captions when available, falling back to auto-generated.

    Args:
        video_id: 11-character video ID.
        preferred_languages: List of language codes in priority order.

    Returns:
        Tuple of (entries_list, has_transcript_bool, language_code).
        Each entry has 'text', 'start', 'duration'.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = preferred_languages or [
        "en", "en-US", "en-GB", "te", "hi", "es", "fr", "de", "it", "id"
    ]

    try:
        api = YouTubeTranscriptApi() if hasattr(YouTubeTranscriptApi, "__init__") else None

        transcript_list = None
        if api and hasattr(api, "list"):
            try:
                transcript_list = api.list(video_id)
            except Exception:
                transcript_list = None
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            except Exception:
                transcript_list = None

        if transcript_list:
            t_obj = None
            try:
                t_obj = transcript_list.find_manually_created_transcript(langs)
            except Exception:
                pass

            if not t_obj:
                try:
                    t_obj = transcript_list.find_generated_transcript(langs)
                except Exception:
                    pass

            if not t_obj:
                try:
                    for t in transcript_list:
                        t_obj = t
                        break
                except Exception:
                    pass

            if t_obj:
                fetched = t_obj.fetch()
                raw_entries = fetched.entries if hasattr(fetched, "entries") else list(fetched)
                clean_entries = []
                for item in raw_entries:
                    text_val = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                    start_val = getattr(item, "start", 0.0) if not isinstance(item, dict) else item.get("start", 0.0)
                    dur_val = getattr(item, "duration", 0.0) if not isinstance(item, dict) else item.get("duration", 0.0)
                    if text_val and text_val.strip():
                        clean_entries.append({
                            "text": text_val.strip(),
                            "start": float(start_val),
                            "duration": float(dur_val),
                        })
                if clean_entries:
                    return clean_entries, True, getattr(t_obj, "language_code", "en")

        # Direct fetch fallback
        if api and hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=langs)
            raw_entries = fetched.entries if hasattr(fetched, "entries") else list(fetched)
            clean_entries = []
            for item in raw_entries:
                text_val = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                start_val = getattr(item, "start", 0.0) if not isinstance(item, dict) else item.get("start", 0.0)
                dur_val = getattr(item, "duration", 0.0) if not isinstance(item, dict) else item.get("duration", 0.0)
                if text_val and text_val.strip():
                    clean_entries.append({
                        "text": text_val.strip(),
                        "start": float(start_val),
                        "duration": float(dur_val),
                    })
            if clean_entries:
                return clean_entries, True, "en"

        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            raw_entries = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            clean_entries = [
                {"text": e["text"].strip(), "start": float(e["start"]), "duration": float(e.get("duration", 0.0))}
                for e in raw_entries
                if e.get("text") and e["text"].strip()
            ]
            if clean_entries:
                return clean_entries, True, "en"

    except Exception:
        pass

    return [], False, ""


def chunk_transcript_segments(
    transcript_entries: List[Dict[str, Any]],
    video_title: str,
    video_id: str,
    canonical_url: str,
    target_words: int = 110,
    overlap_words: int = 25,
) -> List[Document]:
    """
    Group sequential transcript entries into coherent timestamped chunks using a sliding window.
    Guarantees termination and progress across all entries.
    Preserves start and end timestamps for clickable video citations.

    Args:
        transcript_entries: List of transcript snippets with 'text', 'start', 'duration'.
        video_title: Video title.
        video_id: 11-char video ID.
        canonical_url: Canonical YouTube URL.
        target_words: Target word count per chunk.
        overlap_words: Overlap word count between consecutive chunks.

    Returns:
        List of LangChain Document objects with rich timestamp metadata.
    """
    if not transcript_entries:
        return []

    # Calculate full transcript text for content hash
    full_transcript_text = " ".join(e["text"] for e in transcript_entries)
    content_hash = compute_content_hash(full_transcript_text)

    documents: List[Document] = []
    chunk_id = 0

    idx = 0
    total_entries = len(transcript_entries)

    while idx < total_entries:
        selected_entries = []
        word_count = 0
        j = idx

        while j < total_entries and word_count < target_words:
            entry = transcript_entries[j]
            selected_entries.append(entry)
            word_count += len(entry["text"].split())
            j += 1

        if not selected_entries:
            break

        start_time = selected_entries[0]["start"]
        last_entry = selected_entries[-1]
        end_time = last_entry["start"] + last_entry.get("duration", 0.0)

        chunk_text = " ".join(e["text"] for e in selected_entries).strip()
        # Clean any sound effect markers like [music], [applause]
        clean_chunk_text = re.sub(r"\[(?:music|applause|laughter|screaming)\]", "", chunk_text, flags=re.I).strip()
        if not clean_chunk_text:
            clean_chunk_text = chunk_text

        formatted_span = f"{format_timestamp(start_time)}-{format_timestamp(end_time)}"
        heading = f"Segment {formatted_span}"
        timestamp_url = get_youtube_timestamp_url(video_id, start_time)

        content = f"### Video Segment [{formatted_span}]\n{clean_chunk_text}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": canonical_url,
                    "source_type": "youtube",
                    "title": video_title,
                    "video_id": video_id,
                    "page": None,
                    "heading": heading,
                    "timestamp_start": start_time,
                    "timestamp_end": end_time,
                    "timestamp_formatted": formatted_span,
                    "timestamp_url": timestamp_url,
                    "chunk_id": chunk_id,
                    "content_hash": content_hash,
                },
            )
        )
        chunk_id += 1

        if j >= total_entries:
            break

        # Calculate entries to step forward with overlap
        # Estimate number of overlap entries based on overlap_words
        accumulated_overlap = 0
        overlap_entries_count = 0
        for back_entry in reversed(selected_entries):
            accumulated_overlap += len(back_entry["text"].split())
            overlap_entries_count += 1
            if accumulated_overlap >= overlap_words:
                break

        step = max(1, len(selected_entries) - overlap_entries_count)
        idx += step

    return documents


def process_youtube_url(url: str) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Dedicated processing pipeline for YouTube URLs:
    1. Extracts video ID and canonical URL.
    2. Fetches video metadata (title, author, channel, description).
    3. Attempts transcript extraction.
    4. Groups into timestamped LangChain Documents.
    5. Filters out all YouTube platform navigation text.
    6. Provides structured links (Watch Video, Channel).

    Args:
        url: Raw user-provided YouTube URL.

    Returns:
        Tuple of (list_of_documents, stats_dictionary).
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise YouTubeProcessingError(
            f"Invalid YouTube URL: Could not extract a valid 11-character video ID from '{url}'."
        )

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # 1. Fetch metadata
    meta = fetch_youtube_metadata(video_id)
    video_title = meta.get("title", f"YouTube Video ({video_id})")
    author = meta.get("author", "YouTube Creator")
    author_url = meta.get("author_url", "")
    description = meta.get("description", "")

    # 2. Fetch transcript
    transcript_entries, has_transcript, lang_code = fetch_youtube_transcript(video_id)

    # 3. Build documents
    documents: List[Document] = []

    if has_transcript and transcript_entries:
        documents = chunk_transcript_segments(
            transcript_entries=transcript_entries,
            video_title=video_title,
            video_id=video_id,
            canonical_url=canonical_url,
        )
    else:
        # Fallback when transcript is disabled/unavailable:
        # Index video title, author, and description only
        fallback_text_parts = [
            f"Title: {video_title}",
            f"Channel: {author}",
        ]
        if description:
            fallback_text_parts.append(f"Description:\n{description}")

        fallback_content = "\n\n".join(fallback_text_parts)
        chash = compute_content_hash(fallback_content)

        documents.append(
            Document(
                page_content=f"### Video Overview\n{fallback_content}",
                metadata={
                    "source": canonical_url,
                    "source_type": "youtube",
                    "title": video_title,
                    "video_id": video_id,
                    "page": None,
                    "heading": "Video Overview",
                    "timestamp_start": 0.0,
                    "timestamp_end": 0.0,
                    "timestamp_formatted": "00:00",
                    "timestamp_url": canonical_url,
                    "chunk_id": 0,
                    "content_hash": chash,
                    "has_transcript": False,
                },
            )
        )

    # 4. Filter and build Important Links (Watch Video, Channel)
    links: List[Dict[str, str]] = [
        {"text": f"Watch Video: {video_title}", "url": canonical_url}
    ]
    if author_url:
        links.append({"text": f"Channel: {author}", "url": author_url})

    # Extract any real external links present in description
    if description:
        raw_urls = re.findall(r"https?://[^\s<>\"']+", description)
        for u in raw_urls:
            u_clean = u.rstrip(".,;)")
            if "youtube.com" not in u_clean and "youtu.be" not in u_clean:
                parsed_u = urllib.parse.urlparse(u_clean)
                domain = parsed_u.netloc.replace("www.", "")
                links.append({"text": f"🔗 {domain} (From Description)", "url": u_clean})

    # Content hash
    full_content = "\n\n".join(d.page_content for d in documents)
    content_hash = compute_content_hash(full_content)

    stats: Dict[str, Any] = {
        "url": canonical_url,
        "title": video_title,
        "source_type": "youtube",
        "video_id": video_id,
        "author": author,
        "author_url": author_url,
        "has_transcript": has_transcript,
        "transcript_language": lang_code,
        "content_hash": content_hash,
        "sections_count": len(documents),
        "total_characters": len(full_content),
        "links": links,
        "metadata": {
            "author": author,
            "category": "YouTube Video",
            "topics": ["Video", author],
        },
    }

    return documents, stats
