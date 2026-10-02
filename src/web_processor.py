"""Web processing module for validating, safely fetching, and extracting structured web content for RAG."""

import ipaddress
import html
import re
import socket
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.source_registry import compute_content_hash

# Default safe limits
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 20.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 DocuMindAI/2.4"
)


class WebProcessingError(Exception):
    """Base exception raised for web ingestion failures."""
    pass


class SSRFSecurityError(WebProcessingError):
    """Exception raised when a URL targets a private or internal network address."""
    pass


def validate_and_normalize_url(url: str) -> str:
    """
    Validate URL scheme, perform SSRF checks against internal/private addresses,
    and return a normalized URL.

    Args:
        url: Raw user-provided URL string.

    Returns:
        Normalized safe URL.

    Raises:
        SSRFSecurityError: If URL points to internal or private IP ranges.
        WebProcessingError: If URL is malformed or uses an unsupported scheme.
    """
    raw_url = url.strip()
    if not raw_url:
        raise WebProcessingError("URL cannot be empty.")

    # Check for invalid non-http/https schemes (e.g. javascript:alert(1), data:..., file:...)
    if ":" in raw_url:
        candidate_scheme = raw_url.split(":", 1)[0].lower().strip()
        if candidate_scheme not in ("http", "https") and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*$", candidate_scheme):
            raise WebProcessingError(
                f"Unsupported URL scheme '{candidate_scheme}'. Only 'http://' and 'https://' are supported."
            )

    # Auto-prefix https:// if user omitted scheme completely (e.g. 'example.com/page')
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_url):
        raw_url = "https://" + raw_url

    parsed = urllib.parse.urlsplit(raw_url)
    scheme = parsed.scheme.lower()

    if scheme not in ("http", "https"):
        raise WebProcessingError(
            f"Unsupported URL scheme '{scheme}'. Only 'http://' and 'https://' are supported."
        )

    hostname = parsed.hostname
    if not hostname:
        raise WebProcessingError("Invalid URL: missing hostname.")

    hostname_lower = hostname.lower()

    # Reject obvious localhost / internal hostnames
    blocked_hosts = {
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "ip6-localhost",
        "ip6-loopback",
    }
    if hostname_lower in blocked_hosts or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        raise SSRFSecurityError(
            f"Access to '{hostname}' is blocked for security reasons (internal address)."
        )

    # Resolve hostname to IP to protect against DNS rebinding and internal IPs
    try:
        # Check if hostname itself is already an IP address
        try:
            ip_obj = ipaddress.ip_address(hostname_lower)
            ips = [ip_obj]
        except ValueError:
            # Resolve DNS
            addr_info = socket.getaddrinfo(hostname_lower, None)
            ips = [ipaddress.ip_address(item[4][0]) for item in addr_info]

        for ip in ips:
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                raise SSRFSecurityError(
                    f"Access to '{hostname}' ({ip}) is blocked for security reasons (private network address)."
                )
    except socket.gaierror as e:
        raise WebProcessingError(f"DNS resolution failed for '{hostname}': {str(e)}") from e
    except SSRFSecurityError:
        raise
    except Exception as e:
        raise WebProcessingError(f"Failed to validate host '{hostname}': {str(e)}") from e

    # Normalize URL components
    port = parsed.port
    # Remove default ports
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = hostname_lower
    elif port:
        netloc = f"{hostname_lower}:{port}"
    else:
        netloc = hostname_lower

    path = parsed.path or "/"
    # Strip unnecessary trailing slash unless path is root
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Filter out common marketing/tracking query parameters while preserving content parameters
    filtered_query = ""
    if parsed.query:
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "ref", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid"
        }
        clean_pairs = [(k, v) for k, v in query_pairs if k.lower() not in tracking_params]
        if clean_pairs:
            filtered_query = urllib.parse.urlencode(clean_pairs)

    normalized = urllib.parse.urlunsplit((scheme, netloc, path, filtered_query, ""))
    return normalized


def safe_fetch_url(
    url: str,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    timeout: Tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT),
) -> Tuple[str, str]:
    """
    Safely fetch a webpage over HTTP with SSRF redirect revalidation, timeouts,
    size limits, and Content-Type enforcement.

    Args:
        url: Normalized target URL.
        max_redirects: Maximum allowable redirects.
        max_bytes: Maximum response size in bytes.
        timeout: (connect_timeout, read_timeout) tuple.

    Returns:
        Tuple of (response_html_text, final_normalized_url).

    Raises:
        SSRFSecurityError: If any redirect targets an internal network address.
        WebProcessingError: On network, HTTP, or Content-Type errors.
    """
    current_url = validate_and_normalize_url(url)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})

    redirect_count = 0

    while True:
        try:
            response = session.get(
                current_url,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.SSLError as e:
            raise WebProcessingError(f"SSL certificate verification failed for '{current_url}': {str(e)}") from e
        except requests.exceptions.Timeout as e:
            raise WebProcessingError(f"Connection timed out while fetching '{current_url}'.") from e
        except requests.exceptions.ConnectionError as e:
            raise WebProcessingError(f"Failed to connect to '{current_url}': Network or DNS error.") from e
        except Exception as e:
            raise WebProcessingError(f"Unexpected network error fetching '{current_url}': {str(e)}") from e

        # Handle redirects manually to revalidate SSRF at every hop
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            redirect_count += 1
            if redirect_count > max_redirects:
                raise WebProcessingError(f"Redirect limit exceeded ({max_redirects} redirects).")

            redirect_location = response.headers.get("Location")
            if not redirect_location:
                raise WebProcessingError(f"HTTP {response.status_code} redirect missing Location header.")

            # Resolve relative redirect URLs
            next_url = urllib.parse.urljoin(current_url, redirect_location)
            # Revalidate destination for SSRF
            current_url = validate_and_normalize_url(next_url)
            continue

        # Handle standard HTTP status codes
        if response.status_code == 403:
            raise WebProcessingError("Unable to access this webpage. The website returned HTTP 403 (Forbidden).")
        elif response.status_code == 404:
            raise WebProcessingError("Webpage not found. The website returned HTTP 404 (Not Found).")
        elif response.status_code == 429:
            raise WebProcessingError("Too many requests. The website returned HTTP 429 (Rate Limited).")
        elif response.status_code >= 500:
            raise WebProcessingError(f"The website server returned an error (HTTP {response.status_code}).")
        elif response.status_code != 200:
            raise WebProcessingError(f"Unable to fetch webpage. HTTP Status: {response.status_code}.")

        # Enforce Content-Type validation
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise WebProcessingError(
                f"This URL does not contain a supported HTML webpage (received: '{content_type}')."
            )

        # Enforce size limit during stream reading
        content_chunks = []
        bytes_read = 0
        for chunk in response.iter_content(chunk_size=16384):
            if chunk:
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    raise WebProcessingError(
                        f"Webpage exceeds the maximum safe limit ({max_bytes / (1024 * 1024):.0f} MB)."
                    )
                content_chunks.append(chunk)

        encoding = response.encoding or response.apparent_encoding or "utf-8"
        raw_bytes = b"".join(content_chunks)
        try:
            html_text = raw_bytes.decode(encoding, errors="replace")
        except Exception:
            html_text = raw_bytes.decode("utf-8", errors="replace")

        return html_text, current_url


def format_table_to_markdown(table_tag) -> str:
    """Convert an HTML table into a clean GitHub-flavored markdown table."""
    try:
        headers = []
        thead = table_tag.find("thead")
        if thead:
            th_tags = thead.find_all(["th", "td"])
            headers = [html.unescape(th.get_text(separator=" ")).strip() for th in th_tags]
        if not headers:
            first_tr = table_tag.find("tr")
            if first_tr:
                th_tags = first_tr.find_all("th")
                if th_tags:
                    headers = [html.unescape(th.get_text(separator=" ")).strip() for th in th_tags]

        tbody = table_tag.find("tbody") or table_tag
        data_rows = []
        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            cell_texts = [html.unescape(c.get_text(separator=" ")).strip() for c in cells]
            if any(cell_texts):
                if not headers and not data_rows and tr.find("th"):
                    headers = cell_texts
                else:
                    data_rows.append(cell_texts)

        if not data_rows and not headers:
            return ""

        num_cols = max(len(headers), max((len(r) for r in data_rows), default=0))
        if num_cols == 0:
            return ""

        if not headers:
            headers = [f"Column {i+1}" for i in range(num_cols)]
        else:
            while len(headers) < num_cols:
                headers.append(f"Column {len(headers)+1}")

        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * num_cols) + " |",
        ]
        for r in data_rows:
            row_copy = list(r)
            while len(row_copy) < num_cols:
                row_copy.append("")
            escaped_row = [c.replace("|", "\\|") for c in row_copy[:num_cols]]
            md_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(md_lines)
    except Exception:
        return ""


def compute_content_quality_score(
    clean_text: str,
    raw_html: str,
    sections: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Calculate text-to-html ratio, html tag noise count, duplicate line ratio,
    word count, unique word count, heading count, paragraph count, and structural section count.
    Return quality metrics and overall score between 0.0 and 1.0.
    """
    text_len = len(clean_text.strip())
    html_len = max(1, len(raw_html.strip()))

    words = clean_text.split()
    word_count = len(words)
    unique_word_count = len(set(clean_text.lower().split()))
    heading_count = len([s for s in sections if s.get("heading")])
    paragraph_count = len([p for p in clean_text.split("\n\n") if p.strip()])
    raw_tags = len(re.findall(r"<[^>]+>", raw_html))
    html_tag_ratio = round(raw_tags / max(1, word_count), 4)

    raw_ratio = text_len / html_len
    text_ratio_score = min(1.0, raw_ratio * 5.0)

    # Check for leaked HTML tags in clean_text (should be 0)
    tag_noise_matches = len(re.findall(r"<\/?(?:div|span|p|section|article|script|style|table|tr|td|a|h[1-6])[^>]*>", clean_text, re.I))
    tag_noise_score = max(0.0, 1.0 - (tag_noise_matches * 0.2))

    # Duplicate lines check
    lines = [line.strip() for line in clean_text.splitlines() if len(line.strip()) > 15]
    if lines:
        unique_lines = set(lines)
        duplicate_ratio = 1.0 - (len(unique_lines) / len(lines))
        duplicate_score = max(0.0, 1.0 - duplicate_ratio * 1.5)
    else:
        duplicate_ratio = 0.0
        duplicate_score = 0.5 if text_len > 50 else 0.0

    sec_count = len(sections)
    if sec_count >= 3:
        structure_score = 1.0
    elif sec_count == 2:
        structure_score = 0.8
    elif sec_count == 1:
        structure_score = 0.6 if text_len > 150 else 0.3
    else:
        structure_score = 0.0

    overall_score = round(
        (text_ratio_score * 0.30)
        + (tag_noise_score * 0.30)
        + (duplicate_score * 0.25)
        + (structure_score * 0.15),
        2
    )

    is_valid = overall_score >= 0.25 and text_len >= 50 and word_count >= 10

    return {
        "overall_score": overall_score,
        "text_ratio": round(raw_ratio, 4),
        "content_density": round(text_len / html_len, 4),
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "heading_count": heading_count,
        "paragraph_count": paragraph_count,
        "HTML_tag_ratio": html_tag_ratio,
        "tag_noise_count": tag_noise_matches,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "section_count": sec_count,
        "is_valid": is_valid,
    }


def extract_webpage_content(
    html_content: str,
    url: str,
) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Parse HTML content, strip noise and boilerplate, extract title,
    convert tables to markdown, and preserve meaningful headings and sections.

    Args:
        html_content: Raw HTML text.
        url: Webpage URL for title fallback and context.

    Returns:
        Tuple of (clean_full_text, title, sections_list)
        where sections_list contains {"heading": ..., "text": ...}.
    """
    if not html_content or not html_content.strip():
        raise WebProcessingError("Webpage content is completely empty.")

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Title Extraction: og:title -> <title> -> <h1> -> URL path
    title = ""
    og_title_meta = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "og:title"})
    if og_title_meta and og_title_meta.get("content"):
        title = html.unescape(og_title_meta["content"]).strip()
    elif soup.title and soup.title.string:
        title = html.unescape(soup.title.string).strip()
    elif soup.find("h1"):
        title = html.unescape(soup.find("h1").get_text()).strip()
    else:
        parsed = urllib.parse.urlparse(url)
        path_name = parsed.path.strip("/").split("/")[-1]
        title = path_name.replace("-", " ").replace("_", " ").title() if path_name else parsed.netloc

    # 2. Decommission non-content elements and boilerplate
    decommission_tags = [
        "script", "style", "noscript", "svg", "iframe", "nav", "footer",
        "header", "form", "button", "input", "select", "textarea", "aside",
        "dialog", "menu", "canvas", "template", "meta", "link"
    ]
    for tag in soup.find_all(decommission_tags):
        tag.decompose()

    # Remove hidden elements
    for hidden in soup.find_all(attrs={"aria-hidden": "true"}):
        hidden.decompose()
    for hidden in soup.find_all(style=re.compile(r"display:\s*none|visibility:\s*hidden", re.I)):
        hidden.decompose()

    # Remove advertisement, cookie banners, tracking, popups
    noise_pattern = re.compile(
        r"(cookie-banner|cookie-consent|cookie-notice|privacy-notice|ad-container|"
        r"advertisement|banner-ad|social-share|disclaimer|tracking|analytics|"
        r"modal-backdrop|newsletter-signup|subscribe-banner|promo-popup)",
        re.I
    )
    for noisy in soup.find_all(attrs={"class": noise_pattern}):
        noisy.decompose()
    for noisy in soup.find_all(attrs={"id": noise_pattern}):
        noisy.decompose()

    def clean_text_snippet(raw: str) -> str:
        if not raw:
            return ""
        # 1. Unescape HTML entities
        unescaped = html.unescape(raw)
        # 2. Strip any raw HTML tags (e.g. </div>, <div>, <span>, </p>, etc.)
        cleaned = re.sub(r"<[^>]+>", " ", unescaped)
        # 3. Strip any isolated closing or opening tag text fragments
        cleaned = re.sub(r"</?(?:div|span|p|section|article|aside|header|footer|td|tr|table|ul|ol|li)[^>]*>", "", cleaned, flags=re.I)
        # 4. Collapse multiple whitespace within line
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
        return cleaned

    # Convert HTML tables to Markdown tables before descendant extraction
    for tbl in soup.find_all("table"):
        md_table = format_table_to_markdown(tbl)
        if md_table:
            p_table = soup.new_tag("p")
            p_table.string = f"\n\n{md_table}\n\n"
            tbl.replace_with(p_table)
        else:
            tbl.decompose()

    # 3. Target main content container: <article> -> <main> -> <body>
    main_container = soup.find("article") or soup.find("main") or soup.find("body") or soup

    # 4. Extract structured sections by preserving headings
    sections: List[Dict[str, str]] = []
    current_heading = title or "Overview"
    current_paragraphs: List[str] = []

    # Iterate through content elements
    for el in main_container.descendants:
        if not hasattr(el, "name") or not el.name:
            continue

        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading_text = clean_text_snippet(el.get_text(separator=" "))
            if heading_text:
                if current_paragraphs:
                    sec_text = "\n\n".join(current_paragraphs).strip()
                    if sec_text:
                        sections.append({"heading": current_heading, "text": sec_text})
                    current_paragraphs = []
                current_heading = heading_text
        elif el.name == "li":
            li_text = clean_text_snippet(el.get_text(separator=" "))
            if li_text and len(li_text) > 3:
                current_paragraphs.append(f"- {li_text}")
        elif el.name in ("p", "blockquote", "pre", "code"):
            p_text = clean_text_snippet(el.get_text(separator=" "))
            # Avoid duplicate nested text
            if p_text and len(p_text) > 10 and (not current_paragraphs or p_text != current_paragraphs[-1]):
                current_paragraphs.append(p_text)

    # Flush final section
    if current_paragraphs:
        sec_text = "\n\n".join(current_paragraphs).strip()
        if sec_text:
            sections.append({"heading": current_heading, "text": sec_text})

    # Fallback if no sections extracted by tags
    if not sections:
        fallback_text = clean_text_snippet(main_container.get_text(separator="\n"))
        if fallback_text:
            sections.append({"heading": title or "Overview", "text": fallback_text})

    if not sections:
        raise WebProcessingError("No readable content could be extracted from this webpage.")

    # Guaranteed sanitization pass: ensure zero raw HTML fragments in section texts or full text
    html_tag_strip = re.compile(r"<\/?(?:div|span|p|section|article|aside|header|footer|td|tr|table|ul|ol|li|script|style|noscript|nav|main|h[1-6])[^>]*>", re.I)
    for s in sections:
        s["text"] = html_tag_strip.sub("", s["text"]).strip()
        s["heading"] = html_tag_strip.sub("", s["heading"]).strip()

    # Combine into clean full text for content hashing
    full_text_parts = [f"## {s['heading']}\n{s['text']}" for s in sections if s["text"]]
    full_clean_text = "\n\n".join(full_text_parts)
    full_clean_text = html_tag_strip.sub("", full_clean_text).strip()

    return full_clean_text, title, sections


def extract_page_links(
    html_content: str,
    base_url: str,
    max_links: int = 15,
) -> List[Dict[str, str]]:
    """
    Extract meaningful, deduplicated hyperlinks from webpage HTML.

    Args:
        html_content: Raw HTML text.
        base_url: Current webpage URL for relative link resolution.
        max_links: Maximum number of links to return.

    Returns:
        List of dicts with 'text' and 'url'.
    """
    if not html_content or not html_content.strip():
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    # Clean scripts/styles first
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    clean_base = base_url.rstrip("/")
    parsed_base = urllib.parse.urlparse(clean_base)
    seen_urls = set()
    links: List[Dict[str, str]] = []

    generic_texts = {
        "click here", "read more", "more", "here", "link", "learn more",
        "details", "view more", "continue", "next", "previous", "back",
        "home", "login", "sign in", "sign up", "register", "privacy policy",
        "terms of service", "cookie policy"
    }

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        resolved = urllib.parse.urljoin(base_url, href)
        parsed_link = urllib.parse.urlparse(resolved)

        if parsed_link.scheme not in ("http", "https"):
            continue

        # Strip fragment and trailing slash for deduplication
        clean_url = urllib.parse.urlunparse((
            parsed_link.scheme,
            parsed_link.netloc.lower(),
            parsed_link.path.rstrip("/") if parsed_link.path != "/" else "/",
            "",
            parsed_link.query,
            ""
        ))

        # Do not include the page itself
        if clean_url == clean_base or clean_url == f"{clean_base}/":
            continue

        if clean_url in seen_urls:
            continue

        # Extract anchor text
        anchor_text = a_tag.get_text(separator=" ").strip()
        if not anchor_text and a_tag.get("title"):
            anchor_text = a_tag["title"].strip()

        # Clean multiple spaces
        anchor_text = re.sub(r"\s+", " ", anchor_text)

        # Filter out overly short, numeric, or generic navigation labels
        if len(anchor_text) < 3 or anchor_text.lower() in generic_texts:
            continue

        seen_urls.add(clean_url)
        links.append({"text": anchor_text, "url": clean_url})

        if len(links) >= max_links:
            break

    return links


def extract_page_metadata(html_content: str) -> Dict[str, Any]:
    """
    Extract author, publication date, category, and keywords from HTML meta tags.

    Args:
        html_content: Raw HTML text.

    Returns:
        Dict of present metadata fields.
    """
    if not html_content or not html_content.strip():
        return {}

    soup = BeautifulSoup(html_content, "html.parser")
    meta_info: Dict[str, Any] = {}

    # Author
    author_meta = (
        soup.find("meta", attrs={"name": re.compile(r"^author$", re.I)})
        or soup.find("meta", property=re.compile(r"article:author", re.I))
    )
    if author_meta and author_meta.get("content"):
        meta_info["author"] = author_meta["content"].strip()

    # Date
    date_meta = (
        soup.find("meta", property=re.compile(r"article:published_time|publication_date", re.I))
        or soup.find("meta", attrs={"name": re.compile(r"^(date|pubdate|dc\.date)$", re.I)})
    )
    if date_meta and date_meta.get("content"):
        meta_info["date"] = date_meta["content"].strip()
    else:
        time_tag = soup.find("time")
        if time_tag:
            time_val = time_tag.get("datetime") or time_tag.get_text().strip()
            if time_val:
                meta_info["date"] = time_val

    # Category
    cat_meta = (
        soup.find("meta", property=re.compile(r"article:section", re.I))
        or soup.find("meta", attrs={"name": re.compile(r"^category$", re.I)})
    )
    if cat_meta and cat_meta.get("content"):
        meta_info["category"] = cat_meta["content"].strip()

    # Topics / Keywords
    kw_meta = soup.find("meta", attrs={"name": re.compile(r"^keywords$", re.I)})
    if kw_meta and kw_meta.get("content"):
        keywords = [k.strip() for k in kw_meta["content"].split(",") if k.strip()]
        if keywords:
            meta_info["topics"] = keywords[:8]

    return meta_info


def process_web_url(
    url: str,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Ingest a web URL, validate it against SSRF, fetch, clean, and convert into
    standard LangChain Document objects with standardized metadata.

    Args:
        url: User-provided web URL.

    Returns:
        Tuple of (list_of_documents, metadata_stats_dictionary).

    Raises:
        SSRFSecurityError: If URL points to internal or private IP ranges.
        WebProcessingError: On fetch, parse, or extraction errors.
    """
    # 1. Validate & normalize URL
    normalized_url = validate_and_normalize_url(url)

    # 2. Fetch safely with redirect SSRF checks
    html_text, final_url = safe_fetch_url(normalized_url)

    # 3. Extract content & headings
    full_text, title, sections = extract_webpage_content(html_text, final_url)

    # 4. Extract structured links & meta information
    extracted_links = extract_page_links(html_text, final_url)
    extracted_meta = extract_page_metadata(html_text)

    # 5. Content Quality Verification (Reject garbage pages, anti-bot screens, or empty shells)
    quality = compute_content_quality_score(full_text, html_text, sections)
    if not quality["is_valid"]:
        raise WebProcessingError(
            f"The extracted webpage content does not meet the minimum quality threshold "
            f"(quality score: {quality['overall_score']:.2f}, text length: {len(full_text)}). "
            f"This usually indicates anti-bot protection, a JavaScript-only application, or an empty page."
        )

    # 6. Generate deterministic content hash
    content_hash = compute_content_hash(full_text)

    # 7. Build standard LangChain Document objects per section
    documents: List[Document] = []
    for idx, sec in enumerate(sections):
        doc_content = f"### {sec['heading']}\n{sec['text']}"
        documents.append(
            Document(
                page_content=doc_content,
                metadata={
                    "source": final_url,
                    "source_id": final_url,
                    "source_type": "url",
                    "title": title,
                    "page": None,
                    "heading": sec["heading"],
                    "content_hash": content_hash,
                    "section_id": idx,
                    "chunk_id": idx,
                },
            )
        )

    stats: Dict[str, Any] = {
        "url": final_url,
        "source_id": final_url,
        "title": title,
        "content_hash": content_hash,
        "sections_count": len(sections),
        "total_characters": len(full_text),
        "quality_score": quality["overall_score"],
        "quality_metrics": quality,
        "links": extracted_links,
        "metadata": extracted_meta,
    }

    return documents, stats

