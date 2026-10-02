"""Live Test Matrix runner for Phase 5 against indexed GitHub source."""

import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

from src.vector_store import load_vector_store
from src.rag_chain import query_rag_pipeline

def run_matrix():
    vs = load_vector_store()
    target_url = "https://github.com/"

    questions = [
        "What does this page mention about Duolingo?",
        "What is GitHub Copilot used for according to this page?",
        "how to push project files to the github",
        "How can I create a GitHub repository?",
        "Explain GitHub Copilot features mentioned on this page.",
        "What is the orbital velocity of Europa?",
        "Why is GitHub useful for developers?",
        "Explain the process step by step.",
    ]

    print("=" * 80)
    print("PHASE 5 LIVE TEST MATRIX EXECUTION")
    print(f"Target Source: {target_url}")
    print("=" * 80)

    results_summary = []

    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/8] Running: '{q}'")
        
        # Test in both modes: STRICT_SOURCE and SOURCE_FIRST_WITH_FALLBACK
        res_strict = query_rag_pipeline(
            vector_store=vs,
            question=q,
            source_filter=target_url,
            source_type="webpage",
            answer_mode="STRICT_SOURCE",
        )
        
        res_fallback = query_rag_pipeline(
            vector_store=vs,
            question=q,
            source_filter=target_url,
            source_type="webpage",
            answer_mode="SOURCE_FIRST_WITH_FALLBACK",
        )

        dbg_s = res_strict.get("debug_info", {})
        dbg_f = res_fallback.get("debug_info", {})

        entry = {
            "index": idx,
            "question": q,
            "intent": res_fallback.get("intent", "UNKNOWN"),
            "retrieval_relevance": res_fallback.get("retrieval_relevance", "N/A"),
            "answerability": res_fallback.get("answerability", "N/A"),
            "evidence_state": res_fallback.get("evidence_state", "N/A"),
            # STRICT MODE
            "strict_fallback_used": res_strict.get("fallback_used", False),
            "strict_llm_called": dbg_s.get("llm_called", False),
            "strict_refusal": res_strict.get("is_refusal", False),
            "strict_answer_source": res_strict.get("fallback_source") or ("Indexed Source" if not res_strict.get("is_refusal") else "None (Refused)"),
            "strict_citations_count": len(res_strict.get("citations", [])),
            "strict_answer_snippet": res_strict.get("answer", "")[:120].replace("\n", " "),
            # FALLBACK MODE
            "fallback_used": res_fallback.get("fallback_used", False),
            "fallback_llm_called": dbg_f.get("llm_called", False),
            "fallback_refusal": res_fallback.get("is_refusal", False),
            "fallback_answer_source": res_fallback.get("fallback_source", "Indexed Source"),
            "fallback_citations_count": len(res_fallback.get("citations", [])),
            "fallback_answer_snippet": res_fallback.get("answer", "")[:120].replace("\n", " "),
            "full_answer_fallback": res_fallback.get("answer", ""),
            "full_answer_strict": res_strict.get("answer", ""),
        }
        results_summary.append(entry)

        print(f"  Intent:              {entry['intent']}")
        print(f"  Retrieval Relevance: {entry['retrieval_relevance']}")
        print(f"  Answerability:       {entry['answerability']}")
        print(f"  Evidence State:      {entry['evidence_state']}")
        print(f"  [Strict Mode]        Refusal={entry['strict_refusal']} | Citations={entry['strict_citations_count']} | LLM={entry['strict_llm_called']}")
        print(f"  [Fallback Mode]      Fallback={entry['fallback_used']} ({entry['fallback_answer_source']}) | Citations={entry['fallback_citations_count']} | LLM={entry['fallback_llm_called']}")
        safe_snippet = entry['fallback_answer_snippet'].encode('ascii', 'replace').decode('ascii')
        print(f"  [Fallback Preview]   {safe_snippet}...")

    with open("phase5_matrix_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("\n✓ Matrix saved to phase5_matrix_results.json")

if __name__ == "__main__":
    run_matrix()
