"""Structured analysis and validated AI chart generation for DocuMind AI.

Ensures strict source-grounding, URL isolation, structured output cards,
and zero-hallucination chart data validation.
"""

import json
import math
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag_chain import get_groq_llm, extract_citations
from src.config import GROQ_MODEL, is_groq_configured


def validate_chart_data(
    chart_spec: Dict[str, Any],
    source_text: str,
    source_url: str,
    section_heading: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Strictly validate a candidate chart specification against source text.
    Enforces the rule: NO DATA = NO CHART. Never invent or hallucinate numbers.

    Rules:
    - Must be a supported chart type: bar, line, pie, area, scatter.
    - Title, x_label, and y_label must be non-empty strings.
    - Data must have at least 2 data points.
    - Each data point must have a non-empty 'label' and numeric 'value'.
    - Crucially: Every numeric value MUST be verified to exist in the source_text.
      If any numeric value is not found in the source text, the entire chart is rejected.
    - Source URL and citation must match the current source.

    Args:
        chart_spec: Candidate chart dictionary.
        source_text: Plain text of retrieved source chunks.
        source_url: Normalized URL of the current source.
        section_heading: Optional section heading for citation.

    Returns:
        Validated chart dictionary, or None if validation fails.
    """
    if not isinstance(chart_spec, dict):
        return None

    chart_type = str(chart_spec.get("chart_type", "")).lower().strip()
    valid_types = {"bar", "line", "pie", "area", "scatter"}
    if chart_type not in valid_types:
        return None

    title = str(chart_spec.get("title", "")).strip()
    if not title:
        return None

    x_label = str(chart_spec.get("x_label", "Category")).strip()
    y_label = str(chart_spec.get("y_label", "Value")).strip()

    raw_data = chart_spec.get("data")
    if not isinstance(raw_data, list) or len(raw_data) < 2:
        return None

    validated_points = []
    seen_labels = set()

    for item in raw_data:
        if not isinstance(item, dict):
            return None

        raw_label = str(item.get("label", "")).strip()
        raw_val = item.get("value")

        if not raw_label or raw_label in seen_labels:
            return None

        # Parse numeric value
        try:
            val = float(raw_val)
            if math.isnan(val) or math.isinf(val):
                return None
        except (ValueError, TypeError):
            return None

        # Numeric verification against source text:
        # Check if the number appears in the source text in common formats:
        # e.g., 100, 100.0, 100%, 10,000, 100M, etc.
        val_int_str = str(int(val)) if val.is_integer() else None
        val_float_str = f"{val:.1f}" if val.is_integer() else str(val)

        # Regex patterns to search for this number in source text
        patterns = []
        if val_int_str:
            # Match standalone integer or with commas or suffixes
            # e.g. "100", "100,000", "100%", "$100"
            patterns.append(rf"(?<![\d\w]){re.escape(val_int_str)}(?![\d\w])")
        patterns.append(rf"(?<![\d\w]){re.escape(val_float_str)}(?![\d\w])")

        # Also support formatted thousand separators like 10,000
        if val_int_str and len(val_int_str) >= 4:
            formatted_thousands = f"{int(val):,}"
            patterns.append(rf"(?<![\d\w]){re.escape(formatted_thousands)}(?![\d\w])")

        found_in_source = any(re.search(pat, source_text) for pat in patterns)
        if not found_in_source:
            # Rejected: Number was not found in source text (possible hallucination)
            return None

        seen_labels.add(raw_label)
        validated_points.append({
            "label": raw_label,
            "value": int(val) if val.is_integer() else val,
        })

    # Prepare citation
    citation_text = f"Source: {source_url}"
    if section_heading:
        citation_text += f" — Section: {section_heading}"

    return {
        "chart_type": chart_type,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data": validated_points,
        "source_url": source_url,
        "source_citation": citation_text,
    }


def detect_and_extract_charts(
    chunks: List[Document],
    source_url: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Inspect source chunks for numerical/time-series data, extract candidate charts,
    and validate them strictly. Enforces: NO DATA = NO CHART.

    Args:
        chunks: List of Document chunks for the current source.
        source_url: Normalized URL of the current source.
        llm_model: Optional LLM model name.
        api_key: Optional Groq API key.

    Returns:
        List of validated chart specifications.
    """
    if not chunks:
        return []

    combined_text = "\n\n".join(c.page_content for c in chunks)

    # First check: Does source text contain digits?
    # If no numbers exist at all, return empty list immediately.
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", combined_text))
    if not has_numbers:
        return []

    # Check for structured numerical patterns (years, percentages, statistics, prices)
    numeric_matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", combined_text)
    if len(numeric_matches) < 2:
        return []

    charts: List[Dict[str, Any]] = []

    # 1. Deterministic Year/Value or Category/Value extraction for simple tabular sequences
    # E.g.: "2023: 100, 2024: 150, 2025: 200", "2023 = 100\n2024 = 150", or "100 in 2023, 150 in 2024"
    p1 = re.compile(r"\b(19\d\d|20\d\d)\b[\s:=-]+(?:[$€£])?([\d,]+(?:\.\d+)?)\b")
    p2 = re.compile(r"(?:[$€£])?([\d,]+(?:\.\d+)?)\s*(?:million|billion|k|%)?\s+(?:in|during|for|of)\s+\b(19\d\d|20\d\d)\b", re.I)
    p3 = re.compile(r"\b(19\d\d|20\d\d)\b[,\s]+(?:revenue|sales|profit|population|users|growth|count)?[\s]+(?:was|reached|generated|of|at|is)[\s]+(?:[$€£])?([\d,]+(?:\.\d+)?)\b", re.I)

    raw_pairs = p1.findall(combined_text)
    if not raw_pairs:
        raw_pairs = [(yr, val) for val, yr in p2.findall(combined_text)]
    if not raw_pairs:
        raw_pairs = p3.findall(combined_text)

    if len(raw_pairs) >= 2:
        candidate_data = []
        seen_yr = set()
        for year_str, val_str in raw_pairs:
            if year_str in seen_yr:
                continue
            seen_yr.add(year_str)
            clean_val = val_str.replace(",", "")
            try:
                candidate_data.append({"label": year_str, "value": float(clean_val)})
            except ValueError:
                continue

        candidate_data.sort(key=lambda d: d["label"])

        if len(candidate_data) >= 2:
            spec = {
                "chart_type": "line" if len(candidate_data) >= 3 else "bar",
                "title": f"Trends Across Years ({chunks[0].metadata.get('title', 'Data')})",
                "x_label": "Year",
                "y_label": "Value",
                "data": candidate_data,
            }
            validated = validate_chart_data(
                spec,
                source_text=combined_text,
                source_url=source_url,
                section_heading=chunks[0].metadata.get("heading"),
            )
            if validated:
                charts.append(validated)

    # 2. If LLM is configured, use structured LLM extraction for richer tables/metrics
    if not charts and is_groq_configured(api_key):
        try:
            llm = get_groq_llm(model=llm_model, api_key=api_key, temperature=0.0)
            chart_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a strict data extraction engine. Analyze the source text and identify any "
                        "structured numerical data tables, statistics, time series, or comparisons that can be represented as a chart.\n"
                        "CRITICAL RULES:\n"
                        "1. NO DATA = NO CHART. If the text does not contain clear numerical data with categories or years, return an empty array: {\"charts\": []}.\n"
                        "2. NEVER invent, hallucinate, or estimate numbers. Every number MUST appear verbatim in the text.\n"
                        "3. Respond ONLY with valid JSON in this exact structure:\n"
                        "{\n"
                        "  \"charts\": [\n"
                        "    {\n"
                        "      \"chart_type\": \"bar\",\n"
                        "      \"title\": \"Descriptive Title\",\n"
                        "      \"x_label\": \"Category/Year\",\n"
                        "      \"y_label\": \"Metric\",\n"
                        "      \"data\": [{\"label\": \"Item A\", \"value\": 100}, {\"label\": \"Item B\", \"value\": 200}]\n"
                        "    }\n"
                        "  ]\n"
                        "}",
                    ),
                    (
                        "human",
                        "Source Text:\n{text}\n\nExtract verified numerical chart data:",
                    ),
                ]
            )

            chain = chart_prompt | llm | StrOutputParser()
            # Feed first 4000 characters to stay within fast deterministic context
            raw_response = chain.invoke({"text": combined_text[:4000]})

            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                candidate_charts = parsed.get("charts", [])
                for candidate in candidate_charts:
                    val_chart = validate_chart_data(
                        candidate,
                        source_text=combined_text,
                        source_url=source_url,
                        section_heading=chunks[0].metadata.get("heading") if chunks else None,
                    )
                    if val_chart:
                        charts.append(val_chart)
                        break  # Limit to 1 clean validated chart per page
        except Exception:
            # Fall back safely: NO DATA = NO CHART
            pass

    return charts


def generate_structured_website_analysis(
    url: str,
    page_title: str,
    chunks: List[Document],
    web_stats: Optional[Dict[str, Any]] = None,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an isolated, source-grounded structured analysis for the specified URL.

    Output Schema:
    - analysis_id: Unique UUID string
    - source_url: Normalized target URL
    - source_title: Webpage title
    - source_type: 'url'
    - status: 'completed' | 'failed'
    - summary: Concise grounded summary
    - key_points: List of verified insights
    - links: Extracted deduplicated page links
    - other_information: Author, date, topics, category
    - charts: Validated source-backed charts
    - citations: Section citations
    - messages: Dedicated per-URL Q&A chat list
    - created_at: Timestamp

    Args:
        url: Normalized webpage URL.
        page_title: Webpage title.
        chunks: List of Document chunks belonging to this URL.
        web_stats: Metadata stats returned from process_web_url.
        llm_model: Optional model override.
        api_key: Optional Groq API key override.

    Returns:
        Structured analysis session dictionary.
    """
    stats = web_stats or {}
    analysis_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    source_type = stats.get("source_type") or (chunks[0].metadata.get("source_type") if chunks else "url")

    # If no chunks provided, return a failed analysis session
    if not chunks:
        return {
            "analysis_id": analysis_id,
            "source_url": url,
            "source_title": page_title or url,
            "source_type": source_type,
            "status": "failed",
            "summary": "No extractable content found for this URL.",
            "key_points": [],
            "links": [],
            "other_information": {},
            "charts": [],
            "citations": [],
            "messages": [],
            "created_at": created_at,
            "error": "No extractable chunks found.",
        }

    # Extract citations strictly belonging to this source
    citations = extract_citations(chunks)

    # Clean links and metadata from web_stats
    extracted_links = stats.get("links", [])
    extracted_meta = dict(stats.get("metadata", {}))
    if stats.get("author"):
        extracted_meta["channel" if source_type == "youtube" else "author"] = stats["author"]

    combined_text = "\n\n".join(c.page_content for c in chunks)

    # Grounded summary and key insights generation
    summary = ""
    key_points: List[str] = []

    if is_groq_configured(api_key):
        try:
            llm = get_groq_llm(model=llm_model, api_key=api_key, temperature=0.0)
            if source_type == "youtube":
                system_instruction = (
                    "You are DocuMind AI's video intelligence engine. "
                    "Analyze the provided video transcript and generate:\n"
                    "1. A concise, structured summary of the video content/story (overview, main characters, plot, and conclusion).\n"
                    "2. 4 to 6 key insights or takeaways as bullet points.\n\n"
                    "STRICT GROUNDING RULES:\n"
                    "- Use ONLY the provided transcript context.\n"
                    "- Never extrapolate, hallucinate, or make assumptions.\n"
                    "- Respond ONLY with valid JSON in this exact format:\n"
                    "{\n"
                    "  \"summary\": \"Concise grounded video summary...\",\n"
                    "  \"key_points\": [\n"
                    "    \"Insight 1...\",\n"
                    "    \"Insight 2...\",\n"
                    "    \"Insight 3...\",\n"
                    "    \"Insight 4...\"\n"
                    "  ]\n"
                    "}"
                )
            else:
                system_instruction = (
                    "You are DocuMind AI's structured analysis engine. "
                    "Analyze the provided webpage content and generate:\n"
                    "1. A concise, structured executive summary (3-5 sentences or short paragraphs).\n"
                    "2. 4 to 6 key insights or takeaways as bullet points.\n\n"
                    "STRICT GROUNDING RULES:\n"
                    "- Use ONLY the provided webpage context.\n"
                    "- Never extrapolate or make assumptions.\n"
                    "- Respond ONLY with valid JSON in this exact format:\n"
                    "{\n"
                    "  \"summary\": \"Concise grounded executive summary...\",\n"
                    "  \"key_points\": [\n"
                    "    \"Insight 1...\",\n"
                    "    \"Insight 2...\",\n"
                    "    \"Insight 3...\",\n"
                    "    \"Insight 4...\"\n"
                    "  ]\n"
                    "}"
                )

            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_instruction),
                    (
                        "human",
                        "Source Title: {title}\nURL: {url}\n\nContent:\n{content}\n\nStructured JSON:",
                    ),
                ]
            )

            chain = analysis_prompt | llm | StrOutputParser()
            raw_json = chain.invoke(
                {
                    "title": page_title,
                    "url": url,
                    "content": combined_text[:6000],
                }
            )

            json_match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                summary = str(parsed.get("summary", "")).strip()
                kp_raw = parsed.get("key_points", [])
                if isinstance(kp_raw, list):
                    key_points = [str(p).strip() for p in kp_raw if str(p).strip()]
        except Exception:
            pass

    # Fallback deterministic summary if LLM was unavailable or failed
    if not summary:
        first_clean = chunks[0].page_content.replace("###", "").strip()
        paragraphs = [p.strip() for p in first_clean.split("\n\n") if len(p.strip()) > 30]
        if paragraphs:
            summary = "\n\n".join(paragraphs[:3])
        else:
            summary = f"Structured document intelligence generated for {page_title}."

    if not key_points:
        key_points = []
        for c in chunks[:5]:
            heading = c.metadata.get("heading")
            if heading and heading != "Overview" and heading not in key_points:
                key_points.append(f"Section overview: {heading}")
        if not key_points:
            key_points = [
                f"Source content indexed from {url}",
                f"Extracted {len(chunks)} semantic chunks for grounded retrieval",
                "Ready for verified question-answering",
            ]

    # Detect and validate charts (NO DATA = NO CHART)
    charts = detect_and_extract_charts(
        chunks=chunks,
        source_url=url,
        llm_model=llm_model,
        api_key=api_key,
    )

    return {
        "analysis_id": analysis_id,
        "source_url": url,
        "source_title": page_title or url,
        "source_type": source_type,
        "status": "completed",
        "summary": summary,
        "key_points": key_points,
        "links": extracted_links,
        "other_information": extracted_meta,
        "charts": charts,
        "citations": citations,
        "messages": [],  # Isolated conversation for this URL session
        "created_at": created_at,
        "error": None,
    }


def validate_analysis_session(
    analysis: Optional[Dict[str, Any]],
    expected_url: str,
) -> bool:
    """
    Stale result protection guard.
    Verifies that the candidate analysis result matches the expected URL
    and is in a completed status.

    Args:
        analysis: Analysis dictionary or None.
        expected_url: Normalized URL currently being viewed.

    Returns:
        True if the analysis matches expected_url and is completed, False otherwise.
    """
    if not analysis or not isinstance(analysis, dict):
        return False

    source_url = analysis.get("source_url")
    if not source_url or source_url.rstrip("/") != expected_url.rstrip("/"):
        return False

    return analysis.get("status") == "completed"
