"""
Phase 8 -- GPIL Streamlit UI.

WHAT THIS FILE IS FOR (plain language):
This is a PURE RENDERING layer on top of the Phase 7 guarded pipeline, via
the Phase 8 Step 1 boundary (src/inference/ui_adapter.py:run_diagnostic_query).
It must never implement its own retrieval, inference, premise-checking,
grounding, or citation logic -- every fact shown on screen (the answer text,
evidence, citation ids, metric comparisons, SKU rankings, pass/fail status)
comes straight out of a PipelineResult field the backend already populated.
If a section would be empty, it is simply not drawn -- nothing here invents
placeholder evidence or a fabricated status.

The only backend calls this file makes are:
    - config.settings.get_settings()      (to check whether an API key is
      configured, exactly like src/cli.py already does)
    - src.inference.ui_adapter.run_diagnostic_query(question)

Launch with:
    streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/ui/streamlit_app.py` puts this file's own directory
# (src/ui/) on sys.path[0], not the project root -- so the top-level
# `config` and `src` packages aren't importable by default, even though
# running the same file via `python -m` or pytest (which add the project
# root instead) works fine. Prepending the project root here reproduces
# that same resolution for the Streamlit launch path specifically, without
# touching how any other entry point (src/cli.py, tests/) resolves imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import get_settings
from src.inference.ui_adapter import DiagnosticQueryError, run_diagnostic_query

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPIL | Sales & Distribution Intelligence",
    layout="wide",
)

# Questions below are restricted to states/periods that actually exist in
# the validated pilot index (verified by reading documents.parquet
# directly), and were each live-tested end to end through
# run_diagnostic_query() against the real pilot index before being listed
# here. As of the 62-document pilot (2026-08-30 finalization), most states
# have 2 consecutive indexed months, and six states (Bihar, Gujarat,
# Uttarakhand, Maharashtra, Karnataka, Rajasthan) have 3+ consecutive
# months -- Bihar Sep/Oct/Nov 2025 is the case exercised by
# src/inference/pipeline.py's own test suite and below.
EXAMPLE_QUESTIONS = [
    "What was Bihar's Service Level in October 2025?",
    "Why did Bihar's Service Level decline from October to November 2025?",
    "Compare Bihar's Service Level between October 2025 and November 2025.",
    "Which distributors served Maharashtra in August 2024?",
    "What was Bihar's Service Level in September 2025, October 2025, and November 2025?",
]

# final_decision (PipelineResult) -> user-facing label + badge styling.
# Deliberately not showing the raw enum value to the end user; the enum
# itself is still visible in st.expander("Evidence & Sources") debugging
# isn't exposed here at all, matching the "no internal names" requirement.
DECISION_DISPLAY = {
    "pass_through": ("Grounded answer", "pass"),
    "regenerated": ("Validated regenerated answer", "pass"),
    "hedged": ("Grounded with qualification", "warn"),
    "needs_clarification": ("Clarification required", "info"),
    "insufficient_evidence": ("Insufficient evidence", "fail"),
    "rejected": ("Insufficient evidence", "fail"),
}

# ---------------------------------------------------------------------------
# CSS -- contained entirely in this file, styles Streamlit's own DOM nodes
# rather than replacing them. No other frontend framework involved. Colors
# are defined for both light and dark so the appearance reads as intentional
# regardless of the viewer's Streamlit theme setting.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --gpil-navy: #0b1f3a;
        --gpil-navy-light: #14335c;
        --gpil-accent: #2f6fed;
        --gpil-pass: #1f8a52;
        --gpil-pass-bg: rgba(31, 138, 82, 0.12);
        --gpil-warn: #b7791f;
        --gpil-warn-bg: rgba(183, 121, 31, 0.14);
        --gpil-fail: #b3261e;
        --gpil-fail-bg: rgba(179, 38, 30, 0.12);
        --gpil-info: #2f6fed;
        --gpil-info-bg: rgba(47, 111, 237, 0.12);
        --gpil-card-border: rgba(120, 120, 120, 0.18);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---- Header ---- */
    .gpil-header {
        background: linear-gradient(135deg, var(--gpil-navy) 0%, var(--gpil-navy-light) 100%);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 0.75rem;
        color: #f4f6fb;
    }
    .gpil-header-eyebrow {
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0.72;
        margin-bottom: 0.15rem;
    }
    .gpil-header-title {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }
    .gpil-header-subtitle {
        font-size: 0.98rem;
        opacity: 0.85;
        max-width: 640px;
    }
    .gpil-status-row {
        margin: 0.6rem 0 1.4rem 0;
        font-size: 0.85rem;
    }
    .gpil-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
    }
    .gpil-status-ok { color: var(--gpil-pass); background: var(--gpil-pass-bg); }
    .gpil-status-warn { color: var(--gpil-warn); background: var(--gpil-warn-bg); }

    /* ---- Sidebar ---- */
    .gpil-wordmark {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: var(--gpil-accent);
        line-height: 1;
    }
    .gpil-wordmark-sub {
        font-size: 0.82rem;
        opacity: 0.7;
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
    }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid var(--gpil-card-border);
        padding: 0.35rem 0.2rem;
        margin-bottom: 0.6rem;
    }

    /* ---- Answer status badge ---- */
    .gpil-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        margin: 0.5rem 0 0.9rem 0;
    }
    .gpil-badge-pass { color: var(--gpil-pass); background: var(--gpil-pass-bg); }
    .gpil-badge-warn { color: var(--gpil-warn); background: var(--gpil-warn-bg); }
    .gpil-badge-fail { color: var(--gpil-fail); background: var(--gpil-fail-bg); }
    .gpil-badge-info { color: var(--gpil-info); background: var(--gpil-info-bg); }

    /* ---- Welcome / empty state ---- */
    .gpil-welcome {
        text-align: center;
        padding: 2.75rem 1rem 2rem 1rem;
    }
    .gpil-welcome-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .gpil-welcome-subtitle {
        opacity: 0.72;
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    /* ---- Example question buttons ---- */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid var(--gpil-card-border);
        text-align: left;
        white-space: normal;
        line-height: 1.3;
        padding: 0.6rem 0.9rem;
    }

    /* ---- Evidence expander ---- */
    .gpil-evidence-item {
        border-left: 3px solid var(--gpil-accent);
        padding: 0.15rem 0 0.15rem 0.85rem;
        margin-bottom: 0.9rem;
    }
    .gpil-citation-ids {
        font-family: monospace;
        font-size: 0.78rem;
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state -- conversation history only. Never holds API keys/secrets;
# get_settings() is called fresh each run and its result isn't cached here.
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "result": PipelineResult|None, "error": str|None}


def _reset_conversation() -> None:
    """Clears ONLY chat history -- never touches settings/env/secrets."""
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="gpil-wordmark">GPIL</div>', unsafe_allow_html=True)
    st.markdown('<div class="gpil-wordmark-sub">Sales &amp; Distribution Intelligence</div>', unsafe_allow_html=True)

    st.subheader("Workspace")
    st.button("New Conversation", use_container_width=True, on_click=_reset_conversation)

    st.subheader("About")
    st.markdown(
        "- GraphRAG-powered diagnostics\n"
        "- Evidence-grounded answers\n"
        "- KPI-aware reasoning"
    )

    st.subheader("Demo Index")
    st.markdown(
        "This demo runs against a **validated pilot index** -- a fixed, "
        "quality-checked set of Sales & Distribution documents used to "
        "prove out the diagnostic pipeline end to end."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

settings = get_settings()
api_key_configured = bool(settings.openai_api_key)

st.markdown(
    """
    <div class="gpil-header">
        <div class="gpil-header-eyebrow">GPIL</div>
        <div class="gpil-header-title">Sales &amp; Distribution Intelligence</div>
        <div class="gpil-header-subtitle">
            AI-powered diagnostic insights grounded in Sales &amp; Distribution data
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if api_key_configured:
    st.markdown(
        '<div class="gpil-status-row"><span class="gpil-status gpil-status-ok">'
        "&#9679; System Ready &mdash; GraphRAG Diagnostic Assistant</span></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="gpil-status-row"><span class="gpil-status gpil-status-warn">'
        "&#9679; API Key Not Configured</span></div>",
        unsafe_allow_html=True,
    )
    st.warning(
        "OPENAI_API_KEY is not set (.env missing or empty). Queries will fail until it is configured."
    )

# ---------------------------------------------------------------------------
# Rendering helpers -- every one of these reads ONLY fields already present
# on a PipelineResult (or its nested dataclasses). None of them compute,
# infer, or fabricate anything.
# ---------------------------------------------------------------------------


def _render_metric_comparison_table(comparisons, heading: str) -> None:
    if not comparisons:
        return
    st.markdown(f"**{heading}**")
    rows = [
        {
            "Metric": c.metric,
            "Period A": c.period_a,
            "Value A": c.value_a,
            "Period B": c.period_b,
            "Value B": c.value_b,
            "Direction": c.direction,
        }
        for c in comparisons
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_ranked_skus_table(ranked_skus: list[dict]) -> None:
    if not ranked_skus:
        return
    st.markdown("**Ranked SKUs**")
    st.dataframe(ranked_skus, hide_index=True, use_container_width=True)


def _render_evidence(sources) -> None:
    """Evidence & Sources expander -- built entirely from result.sources
    (SourceCitation objects), the same claim -> citation -> evidence text
    provenance src/cli.py's _render_sources() already displays in the
    terminal. Nothing here is invented; a claim with no resolved source
    simply doesn't appear."""
    if not sources:
        return
    with st.expander("Evidence & Sources", expanded=False):
        for i, source in enumerate(sources, start=1):
            st.markdown('<div class="gpil-evidence-item">', unsafe_allow_html=True)
            if source.claim_text:
                st.markdown(f"**Claim:** {source.claim_text}")
            st.markdown(f"**Evidence:** {source.evidence_text}")
            citation_ids = ", ".join(source.citation_ids)
            st.markdown(f'<div class="gpil-citation-ids">Citation ID(s): {citation_ids}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


def _render_pipeline_result(result) -> None:
    st.markdown(result.final_text)

    label, tone = DECISION_DISPLAY.get(result.final_decision, (result.final_decision, "info"))
    st.markdown(f'<span class="gpil-badge gpil-badge-{tone}">{label}</span>', unsafe_allow_html=True)

    # Structured insights -- only drawn when the backend actually produced
    # them; a question with no comparison/ranking shape leaves both empty.
    premise = result.premise_check
    if premise is not None and premise.metric_comparisons:
        _render_metric_comparison_table(premise.metric_comparisons, "Metric Comparison")
    elif premise is not None and premise.supplementary_comparisons:
        _render_metric_comparison_table(premise.supplementary_comparisons, "Related Metric Comparison")

    if result.evidence_sufficiency is not None and result.evidence_sufficiency.ranked_skus:
        _render_ranked_skus_table(result.evidence_sufficiency.ranked_skus)

    _render_evidence(result.sources)


def _process_question(question: str) -> None:
    """Renders the new user/assistant turn live, then appends it to history
    and reruns so the top-of-script history loop (below) takes over as the
    single source of truth on the next render -- avoids ever rendering the
    same turn two different ways."""
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Sales & Distribution data..."):
            result = None
            error_message = None
            try:
                result = run_diagnostic_query(question)
            except DiagnosticQueryError:
                # Fail closed on infrastructure errors without leaking
                # internals (no stack trace, no API key, no file paths) --
                # the session stays alive for the next question.
                error_message = (
                    "The diagnostic assistant could not complete this query right now. "
                    "Please try again, or rephrase the question."
                )

        if error_message:
            st.error(error_message)
        else:
            _render_pipeline_result(result)

    st.session_state.history.append({"question": question, "result": result, "error": error_message})
    st.rerun()


# ---------------------------------------------------------------------------
# Main conversational workspace
# ---------------------------------------------------------------------------

example_clicked: str | None = None

if not st.session_state.history:
    st.markdown(
        '<div class="gpil-welcome">'
        '<div class="gpil-welcome-title">Ask why a Sales &amp; Distribution KPI changed.</div>'
        '<div class="gpil-welcome-subtitle">Try one of these, or ask your own question below.</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
            example_clicked = q
else:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn["error"]:
                st.error(turn["error"])
            else:
                _render_pipeline_result(turn["result"])

typed_question = st.chat_input("Ask why a Sales & Distribution KPI changed...")

question_to_run = typed_question or example_clicked
if question_to_run:
    _process_question(question_to_run)
