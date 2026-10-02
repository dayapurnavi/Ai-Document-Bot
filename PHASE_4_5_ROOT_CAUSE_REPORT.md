# DocuMind AI — Phase 4.5 Root-Cause Debugging Report

**Date & Time:** October 2, 2026  
**Environment:** Windows (PowerShell) / Streamlit 1.x / Python 3.12 / Groq LLM / FAISS / HuggingFace `all-MiniLM-L6-v2`  
**Test Suite Status:** 150/150 Tests Passing (9/9 Test Suites, 100% Pass Rate)  
**Live Application Status:** Running at `http://localhost:8501`

---

## 1. Exact Failure Reproduced

Two distinct production failures visible in live user sessions were reproduced and verified under exact runtime conditions:

### Bug 1: Leaking `</div>` Tag in Assistant Responses
- **Symptom:** Assistant responses (particularly refusals such as *"I couldn't find that information in the indexed webpage content."*) were displayed in the Streamlit UI with an unclosed raw `</div>` box or code block immediately following the message text.
- **Reproduction:** Calling `format_chat_bubble_html` on refusal strings inside Python indented multiline strings (`f""" ... <div class="ai-msg-row"> ... </div> """`) in `app.py`.
- **Observation:** `markdown-it-py` (CommonMark parser used by Streamlit) interpreted lines with $\ge 4$ leading spaces as indented code blocks, emitting `<pre><code>&lt;/div&gt;</code></pre>`, rendering literal `</div>` to user viewports.

### Bug 2: False Refusals on Grounded, Answerable Questions
- **Symptom:** Questions like *"What does this page mention about Duolingo?"* or *"What is GitHub Copilot used for according to this page?"* triggered false refusals (*"I couldn't find that information in the indexed webpage content."*) even when the indexed source explicitly contained the exact factual answers.
- **Reproduction:** Tracing `evaluate_evidence_coverage` with real chunks for `https://github.com/`. Conversational context words (`"according"`, `"page"`, `"website"`, `"mention"`) remained in the sub-question evaluation vocabulary. Because the marketing chunks did not contain the literal words `"according"` or `"page"`, the token match ratio fell below 50% ($1/3 = 33\%$), triggering `missing_subs` and leaving `covered_subs = []`.
- **Observation:** Line 469 of `src/rag_chain.py` (`if not covered_subs: state = "NONE"`) forced the evidence state to `NONE`, causing the Strict Refusal Gate to prematurely return a refusal without invoking the LLM.

---

## 2. Whether Retrieval Failed

**Finding: Retrieval did NOT fail.**
- For the test query `"how to push project files to the github"`, FAISS candidate retrieval returned 6 relevant chunks from `https://github.com/` based on keyword matches for `"github"` and `"push"` (e.g., Secret Scanning Push Protection).
- For `"What does this page mention about Duolingo?"` and `"What is GitHub Copilot used for according to this page?"`, FAISS retrieved the exact target chunks (Chunk 1 and Chunk 2) with semantic similarity rankings in the top 3 positions.
- Candidate retrieval count was 6, reranked count was 6, and context chunks was 6. Retrieval was operating accurately.

---

## 3. Whether Evidence Gate Failed

**Finding: YES — The Evidence Evaluation Gate failed on conversational questions (Bug 2).**
- In `src/rag_chain.py:evaluate_evidence_coverage`, the token evaluation logic extracted words from decomposed sub-questions.
- Conversational framing phrases (e.g., *"according to this page"*, *"what does this page mention about"*, *"based on this website"*) injected meta-words into `eval_sq_words` (`['duolingo', 'mention', 'page']`).
- Because technical source chunks contain domain facts and omit conversational filler words like `"page"` or `"mention"`, `sq_matches` was 1 out of 3 ($33.3\%$). The threshold required $\ge 2$ matches for length 3-4 words.
- This caused `covered_subs` to become empty (`[]`).
- The condition `if not covered_subs: state = "NONE"` immediately labeled valid retrieved context as `NONE`, triggering an artificial refusal before LLM generation.

---

## 4. Whether Indexing Failed

**Finding: Indexing did NOT fail.**
- Inspection of `vectorstore/sources.json` and the active FAISS index confirmed:
  - Source ID: `webpage_39110213e54d`
  - URL: `https://github.com/`
  - Chunk Count: 6 chunks
  - Content: Correctly extracted clean marketing copy of GitHub homepage (Copilot, Pull Requests, Security, CI/CD, Duolingo 25% velocity increase).
- Chunks contained clean plain text without HTML boilerplates, scripts, or broken markup.
- However, the indexed content for `https://github.com/` **did NOT contain Git CLI commands** (`git init`, `git add`, `git commit`, `git push origin main`). Therefore, for the query `"how to push project files to the github"`, refusing to invent Git push commands was **correct and strictly grounded**. The defect was purely in how the refusal was formatted and rendered.

---

## 5. Whether Stale Cache Was Involved

**Finding: Stale cache was a compounding risk factor.**
- In `app.py`, cached answers from `st.session_state.query_cache` were being rendered directly without passing through a final sanitization step.
- If an earlier pipeline iteration had generated a response containing closing HTML tags (`</div>`), that contaminated string remained in cache across subsequent queries.
- **Fix:** Both cache insertion and cache retrieval paths now mandate execution of `sanitize_final_response()`.

---

## 6. Whether HTML Extraction Failed

**Finding: HTML extraction in `src/web_processor.py` was clean.**
- Extracted text from `process_web_url` contained no `<script>`, `<style>`, `<div>`, or `</div>` artifacts.
- BeautifulSoup and readability parsing stripped all document-level HTML tags prior to document chunking.

---

## 7. Whether HTML Rendering Failed

**Finding: YES — Hand-rolled markdown regexes in `format_chat_bubble_html` broke Markdown structures.**
- In `app.py`, `format_chat_bubble_html` attempted to parse markdown by replacing `\n\n` with `<div style="margin-bottom: 0.65rem;">...</div>` and replacing internal `\n` with `<br/>`.
- This broke markdown tables (`| A | B |`), bullet lists (`- item`), numbered lists (`1. item`), and fenced code blocks (```` ```...``` ````).
- Moreover, it created raw `<div>` tags directly in the template that exacerbated the Streamlit indentation defect.

---

## 8. Whether Streamlit Rendering Failed

**Finding: YES — Indented multiline strings in `st.markdown(..., unsafe_allow_html=True)` was the direct root cause of Bug 1.**
- In `app.py` (lines 1705–1715 in Website Analysis and lines 1948–1965 in Global Chat), the assistant message HTML was written inside python multiline f-strings:
  ```python
  st.markdown(
      f"""
      <div class="ai-msg-row">
          <div class="ai-msg-bubble">
              {bubble_html}
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )
  ```
- Because of Python indentation inside `if/else` blocks, every line in the f-string had 24 to 32 leading spaces.
- CommonMark specification (implemented by `markdown-it-py` inside Streamlit) mandates that any line starting with $\ge 4$ spaces is an **Indented Code Block**.
- As a result, the closing `                            </div>` tag was converted to `<pre><code>&lt;/div&gt;</code></pre>`, which appeared on the user's screen as raw `</div>`!

---

## 9. Exact Affected Files

| File | Nature of Changes |
| :--- | :--- |
| `src/rag_chain.py` | Implemented and exported central `sanitize_final_response()`; added conversational context word filtering in `evaluate_evidence_coverage`; routed all return paths through `sanitize_final_response()`. |
| `app.py` | Replaced broken `format_chat_bubble_html` with MarkdownIt-backed safe renderer; converted indented multiline `st.markdown()` f-strings to unindented HTML blocks; sanitized cached query results. |
| `tests/test_runtime_answer_path.py` | New comprehensive test suite with 8 mandatory tests validating the runtime answer and refusal paths. |

---

## 10. Exact Fixes Implemented

### Fix A: Central Response Sanitizer (`src/rag_chain.py`)
```python
def sanitize_final_response(text: Optional[str]) -> str:
    """
    Central authoritative response sanitizer for DocuMind AI (Phase 4.5).
    Guarantees that no raw HTML tags (e.g., <div>, </div>, <span>, <p>, <section>)
    leak into user-facing responses across all execution paths.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Protect code blocks from tag stripping
    code_blocks: List[str] = []
    def _save_block(m):
        code_blocks.append(m.group(0))
        return f"__DOCUMIND_CODE_BLOCK_{len(code_blocks)-1}__"

    protected = re.sub(r"```[\s\S]*?```", _save_block, text)
    protected = re.sub(r"`[^`\n]+`", _save_block, protected)

    # 2. Decode HTML entities that might hide encoded tags
    decoded = html.unescape(protected)

    # 3. Strip script and style blocks completely
    cleaned = re.sub(r"<\s*(?:script|style)[^>]*>[\s\S]*?<\s*\/\s*(?:script|style)\s*>", "", decoded, flags=re.IGNORECASE)

    # 4. Convert block-level paragraph / section closing tags to double newlines
    cleaned = re.sub(r"<\s*\/\s*(?:p|section|article|header|footer|h[1-6]|blockquote)\s*>", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*\/\s*(?:tr|li)\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*br\s*\/?>", "\n", cleaned, flags=re.IGNORECASE)

    # 5. Strip all remaining structural / raw HTML tags
    cleaned = re.sub(r"<\/?\s*[a-zA-Z][a-zA-Z0-9_\-]*\b[^>]*\/?>", "", cleaned)
    cleaned = re.sub(r"<\s*\/\s*[a-zA-Z0-9_\-]+\s*>", "", cleaned)
    cleaned = re.sub(r"^\s*<\s*\/?\s*[a-zA-Z0-9_\-]+\s*>\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\bdiv\s*>", "", cleaned, flags=re.IGNORECASE)

    # 6. Restore protected code blocks
    for idx, block in enumerate(code_blocks):
        cleaned = cleaned.replace(f"__DOCUMIND_CODE_BLOCK_{idx}__", block)

    # 7. Clean up whitespace
    lines = [line.rstrip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
```

### Fix B: Conversational Framing Filtering (`src/rag_chain.py`)
```python
conversational_context_words = {
    "according", "accord", "accordance", "page", "pages", "website", "websites",
    "site", "sites", "document", "documents", "doc", "docs", "text", "content",
    "source", "sources", "article", "articles", "mention", "mentions", "mentioned",
    "say", "says", "said", "tell", "tells", "telling", "based", "here", "above",
    "below", "provide", "provides", "provided", "used", "use", "using", "state",
    "states", "stated", "show", "shows", "shown", "find", "finds", "found", "info",
    "information", "read", "reads", "view", "views"
}
all_filter_words = meta_directives | conversational_context_words

core_q_words = [w for w in q_words if w not in all_filter_words]
eval_q_words = core_q_words if core_q_words else [w for w in q_words if w not in meta_directives]
```

### Fix C: Streamlit Safe Unindented Rendering (`app.py`)
```python
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
```
Outer message wrappers now have ZERO leading indentation, preventing markdown-it from generating indented code blocks:
```python
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
```

---

## 11. New Regression Tests

Created `tests/test_runtime_answer_path.py` containing 8 tests:

| Test Name | Purpose | Result |
| :--- | :--- | :--- |
| `test_answerable_question_reaches_llm` | Proves answerable questions pass evidence gates, reach the LLM, and return factual data with citations. | **PASSED** |
| `test_unsupported_question_refuses` | Proves unsupported topics cleanly refuse with exact refusal phrase and ZERO HTML tags. | **PASSED** |
| `test_final_response_has_no_html` | Proves response sanitizer strips raw/encoded HTML tags while preserving tables, code blocks, lists, and links. | **PASSED** |
| `test_cached_response_is_sanitized` | Proves cached responses are sanitized on retrieval and cache keys are strictly isolated. | **PASSED** |
| `test_source_switch_changes_retrieval` | Proves changing active sources completely changes retrieved context with zero cross-source leakage. | **PASSED** |
| `test_active_source_is_enforced` | Proves all returned chunks and citations belong strictly to the active source filter. | **PASSED** |
| `test_relevant_chunks_do_not_trigger_false_refusal` | Proves conversational phrasing (Bug 2) never triggers false refusals when evidence is present. | **PASSED** |
| `test_empty_retrieval_returns_clean_refusal` | Proves empty retrieval yields a clean refusal with ZERO HTML tags. | **PASSED** |

### Complete Regression Suite Results
```
================= 150 passed, 1 warning in 130.82s (0:02:10) ==================
- test_phase3_production.py: 43/43 PASSED
- test_phase4_uat.py: 30/30 PASSED
- test_rag_pipeline.py: 11/11 PASSED
- test_runtime_answer_path.py: 8/8 PASSED
- test_source_isolation.py: 11/11 PASSED
- test_web_rag.py: 12/12 PASSED
- test_youtube_rag.py: 12/12 PASSED
- test_deep_retrieval.py: 11/11 PASSED
- test_evidence_eval.py: 12/12 PASSED
Total: 150/150 PASSED (100%)
```

---

## 12. Live Test Results

Executed live pipeline queries against the active indexed source `https://github.com/`:

### Live Test 1: Grounded Answerable Question
- **Question:** `"What does this page mention about Duolingo?"`
- **Question Type:** `FACTUAL`
- **Active Source:** `https://github.com/`
- **Candidates Retrieved:** 6 chunks
- **Evidence State:** `SUFFICIENT` (Coverage: 0.700)
- **Is Refusal:** `False`
- **Actual LLM Output:**
  ```markdown
  Duolingo boosts developer speed by **25 %** with GitHub Copilot.
  ```
- **Rendered HTML in Streamlit UI:**
  ```html
  <p>Duolingo boosts developer speed by <strong>25 %</strong> with GitHub Copilot.</p>
  ```
- **Raw HTML / `</div>` Present:** **NO (False)**
- **Citations:** 6 verified citations

### Live Test 2: Unsupported Procedural Operation
- **Question:** `"how to push project files to the github"`
- **Question Type:** `HOW`
- **Active Source:** `https://github.com/`
- **Candidates Retrieved:** 6 chunks
- **Evidence State:** `STRONG` (Retrieved Copilot / platform text based on keywords `"github"` and `"push"`)
- **Is Refusal:** `True` (LLM recognized absence of Git CLI push commands and adhered to strict grounding)
- **Actual LLM Output:**
  ```
  I couldn't find that information in the indexed webpage content.
  ```
- **Rendered HTML in Streamlit UI:**
  ```html
  <p>I couldn't find that information in the indexed webpage content.</p>
  ```
- **Raw HTML / `</div>` Present:** **NO (False)** — Clean refusal, zero leaks!
- **Citations:** 0 (citations suppressed on refusal)

### Live Test 3: Complex Answerable Multi-Point Question
- **Question:** `"What is GitHub Copilot used for according to this page?"`
- **Question Type:** `DEFINITION`
- **Active Source:** `https://github.com/`
- **Candidates Retrieved:** 6 chunks
- **Evidence State:** `STRONG` (Coverage: 0.767)
- **Is Refusal:** `False`
- **Actual LLM Output:**
  ```markdown
  GitHub Copilot is used to:

  - **Write, test, and fix code quickly** – from simple boilerplate to complex features [Chunk 1].
  - **Assist at every step of the software development lifecycle** – helping developers throughout the process [Chunk 2].
  - **Apply fixes in seconds** – reducing debugging time and building features faster with Copilot Autofix [Chunk 3].
  - **Help with code reviews** – assigning initial reviews to Copilot for greater speed and quality [Chunk 4].
  ```
- **Rendered HTML in Streamlit UI:**
  ```html
  <p>GitHub Copilot is used to:</p>
  <ul>
  <li><strong>Write, test, and fix code quickly</strong> – from simple boilerplate to complex features...</li>
  <li><strong>Assist at every step of the software development lifecycle</strong>...</li>
  <li><strong>Apply fixes in seconds</strong>...</li>
  <li><strong>Help with code reviews</strong>...</li>
  </ul>
  ```
- **Raw HTML / `</div>` Present:** **NO (False)**
- **Citations:** 6 verified citations

---

## Final Verification Checklist

- [x] Answerable source question produces an answer
- [x] Deep answerable question produces an answer
- [x] Cross-section answerable question produces an answer
- [x] Unsupported question correctly refuses
- [x] Refusal contains NO raw HTML
- [x] Successful answer contains NO raw HTML
- [x] Cached answer contains NO raw HTML
- [x] Source A cannot leak into Source B
- [x] Relevant retrieved evidence cannot incorrectly become NONE
- [x] Active source is visible and enforced
- [x] Latest indexed content is actually used
- [x] Streamlit runtime path is tested
- [x] Existing tests still pass (142/142)
- [x] New runtime tests pass (8/8)
- [x] Total automated tests: 150/150 PASSED
