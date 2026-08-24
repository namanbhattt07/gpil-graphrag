"""
Phase 8, Step 1A -- minimal terminal chatbot wrapper around the validated
Phase 7 guarded pipeline, via the Step 1 UI boundary
(src/inference/ui_adapter.py:run_diagnostic_query()).

WHAT THIS FILE IS FOR (plain language):
A temporary validation harness so real questions can be asked against the
Phase 7 pipeline from a terminal, before the Streamlit UI exists. As of
Step 1, this file no longer knows the index's file paths itself -- it
delegates entirely to run_diagnostic_query(), the same boundary function a
future Streamlit app will call, so this CLI now doubles as a live sanity
check that the boundary itself works. No internal object (QueryContext,
the LocalSearch engine, any DataFrame) is ever touched here, per the
Phase 8 boundary audit.
"""

from __future__ import annotations

from config.settings import get_settings  # import loads .env into the process env as a side effect

from src.inference.ui_adapter import DiagnosticQueryError, run_diagnostic_query


def _render_sources(result) -> str | None:
    """Problem 2 -- source traceability: one block per answer claim that
    has a resolved citation, showing claim -> citation id(s) -> the actual
    retrieved/available evidence text those ids point to. Built entirely
    from result.sources (src.inference.schemas.SourceCitation), which
    pipeline.py only ever populates from claims the grounding checker has
    already validated -- nothing here is invented or re-derived; this is
    purely a display pass. Returns None (never an empty string) when there
    is nothing to show, so callers can skip the section header cleanly."""
    if not result.sources:
        return None
    lines = ["Sources:"]
    for i, source in enumerate(result.sources, start=1):
        citations = ", ".join(source.citation_ids)
        lines.append(f"[{i}] {citations}")
        if source.claim_text:
            lines.append(f'    Claim: "{source.claim_text}"')
        evidence = source.evidence_text.strip().replace("\n", " ")
        if len(evidence) > 400:
            evidence = evidence[:400] + "..."
        lines.append(f"    Evidence: {evidence}")
    return "\n".join(lines)


def _debug_line(result) -> str:
    # PipelineResult.grounding_check always holds the FIRST draft's result;
    # when a retry happened (final_decision="regenerated" means it fixed
    # things), the status a user should see is the RETRY's, not the
    # first draft's -- otherwise a successful regeneration prints a
    # misleading "grounding=fail" next to a perfectly good final answer.
    if result.retry_attempted and result.retry_grounding_check is not None:
        grounding_status = result.retry_grounding_check.status
    elif result.grounding_check is not None:
        grounding_status = result.grounding_check.status
    else:
        grounding_status = "n/a"
    return (
        f"[final_decision={result.final_decision} "
        f"premise={result.premise_check.status} "
        f"grounding={grounding_status} "
        f"llm_calls={result.llm_calls_made}]"
    )


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("Warning: OPENAI_API_KEY is not set (.env missing or empty) -- queries will fail.")

    print("GPIL Diagnostic Assistant")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        try:
            result = run_diagnostic_query(question)
        except DiagnosticQueryError as exc:  # infra failures (API/index/config) -- fail closed, keep looping
            print(f"Assistant: [error] {exc}")
            print()
            continue

        print(f"Assistant: {result.final_text}")
        sources_block = _render_sources(result)
        if sources_block:
            print()
            print(sources_block)
        print(_debug_line(result))
        print()


if __name__ == "__main__":
    main()
