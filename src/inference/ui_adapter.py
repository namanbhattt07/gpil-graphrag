"""
Phase 8, Step 1 -- the clean backend/UI boundary for the Phase 7 pipeline.
Phase 8, Step 2 -- adds process-lifetime reuse of the expensive GraphRAG
LocalSearch engine across repeated questions (see _get_query_context below).

WHAT THIS FILE IS FOR (plain language):
Every other caller of the Phase 7 pipeline so far (tests, src/cli.py) has
had to know the index's file paths and how to load .env itself. A future
Streamlit UI shouldn't need any of that -- it should only ever need to
hand over a question string and get back a PipelineResult. This module is
that one function:

    result = run_diagnostic_query("Why did Bihar's Service Level decline?")

It is a THIN wrapper, not a new implementation: it fixes the index paths,
makes sure .env is loaded, and calls the existing
src.inference.context.build_query_context() / src.inference.pipeline.run_pipeline()
-- nothing about retrieval, premise checking, answer generation, grounding,
or the one-bounded-retry behavior lives here or is duplicated here.
answer_fn/retry_fn (the test-injection seams on run_pipeline()) are
deliberately not exposed -- there would be nothing for a UI caller to do
with them.

ENGINE REUSE (Phase 8 Step 2):
build_query_context() does two things in one call: (1) load the index and
build a GraphRAG LocalSearch engine -- expensive, and identical for every
question since INDEX_ROOT/OUTPUT_DIR are fixed constants -- and (2) run
retrieval for one specific question -- cheap-ish but MUST happen fresh
every time. This module now caches only piece (1), the engine, for the
life of the process, and always calls
engine.context_builder.build_context(query=..., **engine.context_builder_params)
directly for piece (2) on every question. This was verified safe by
reading GraphRAG's LocalSearchMixedContext source directly: build_context()
only reads its own stored entities/relationships/embeddings and returns a
new ContextBuilderResult -- it never assigns to self.<anything>, so sharing
one engine across many questions (or concurrent sessions) cannot leak state
between them. Nothing question-specific (QueryContext, retrieved chunks,
premise/answer/grounding results, PipelineResult) is ever cached -- see
_get_query_context()'s docstring for the exact boundary.

ERROR BOUNDARY:
run_pipeline() already returns a normal PipelineResult (with
final_decision="insufficient_evidence"/"hedged"/etc.) for evidence-quality
problems -- that is not an error, and this module doesn't touch it.
Infrastructure failures (missing/invalid API key, unreachable API, a
broken index/config) are different: those raise. This module narrows
whatever exception type happens to surface (a GraphRAG config error, an
httpx/litellm error, a file error, ...) down to ONE type,
DiagnosticQueryError, so a UI caller only ever needs one except clause --
while chaining the original exception as __cause__ so the real traceback
is still visible in logs/a debugger.
"""

from __future__ import annotations

import threading
from pathlib import Path

from config.settings import get_settings  # importing this module loads .env as a side effect
from graphrag.query.structured_search.local_search.search import LocalSearch

from src.inference.context import QueryContext, build_query_context
from src.inference.pipeline import run_pipeline
from src.inference.schemas import PipelineResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_ROOT = PROJECT_ROOT / "data" / "graphrag_index" / "pilot_run_mini"
OUTPUT_DIR = INDEX_ROOT / "output_fixed_clean"

# Process-lifetime cache for the GraphRAG engine ONLY -- never for anything
# question-specific. Becomes stale only if the underlying index at
# INDEX_ROOT/OUTPUT_DIR is regenerated while this process is still running
# (not a concern today: the index is frozen); restarting the process (or a
# fresh `streamlit run`) always clears it and rebuilds from the current
# index on the next question. Guarded by _engine_lock so two questions
# arriving at (almost) the same instant on a cold cache -- e.g. two
# Streamlit sessions starting together -- can't each build their own engine.
_cached_engine: LocalSearch | None = None
_engine_lock = threading.Lock()


def _get_query_context(question: str) -> QueryContext:
    """Return a QueryContext for `question`, building (and caching) the
    GraphRAG engine only on the very first call this process makes.

    Every call -- cached or not -- ends up making exactly one
    engine.context_builder.build_context(query=question, ...) call, i.e.
    exactly one fresh retrieval/embedding per question, with no exceptions:
      - First ever call: build_query_context() builds the engine AND runs
        retrieval for `question` in one call (identical cost/behavior to
        the pre-Step-2 implementation); its context_result is reused as-is
        rather than re-retrieved a second time.
      - Every later call: reuses the cached engine and calls
        engine.context_builder.build_context() directly for `question`.
    """
    global _cached_engine

    if _cached_engine is None:
        with _engine_lock:
            if _cached_engine is None:  # re-check: another thread may have built it while this one waited for the lock
                qctx = build_query_context(index_root=INDEX_ROOT, output_dir=OUTPUT_DIR, query=question)
                _cached_engine = qctx.engine
                return qctx
            # else: fall through -- the engine is now cached, so run this
            # thread's own fresh retrieval for `question` below, same as
            # the already-warm path.

    engine = _cached_engine
    context_result = engine.context_builder.build_context(query=question, **engine.context_builder_params)
    return QueryContext(engine=engine, context_result=context_result)


class DiagnosticQueryError(RuntimeError):
    """Raised when run_diagnostic_query() cannot complete a query due to an
    infrastructure failure (missing/invalid API key, unreachable API,
    a broken index/config file) -- as opposed to an evidence-quality
    outcome, which the Phase 7 pipeline already reports through a normal
    PipelineResult rather than an exception. The original exception is
    chained as __cause__, so `raise`s of this type still carry the full
    underlying traceback for debugging."""


def run_diagnostic_query(question: str) -> PipelineResult:
    """The one function a Streamlit UI (or any other caller) needs: give
    it a question, get back a fully structured PipelineResult. Callers
    don't need to know about index paths, QueryContext, GraphRAG, engine
    caching, or the answer_fn/retry_fn test-injection seams -- those stay
    internal to the Phase 7 pipeline this wraps.

    Raises:
        ValueError: `question` is empty/whitespace-only -- a caller
            mistake, not an infrastructure failure, so this is NOT wrapped
            in DiagnosticQueryError.
        DiagnosticQueryError: the pipeline itself could not complete the
            query (see module docstring). Never raised for a question the
            pipeline could evaluate but had insufficient evidence for --
            that comes back as a normal PipelineResult instead.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    question = question.strip()
    get_settings()  # no-op if .env was already loaded by an earlier import; cheap either way

    try:
        qctx = _get_query_context(question)
        result = run_pipeline(question, qctx)
        # +1 for the embedding call _get_query_context() always makes (via
        # either build_query_context() or a direct build_context() call) --
        # identical accounting to the pre-Step-2 answer_question() wrapper,
        # since both paths make exactly one retrieval call per question.
        result.llm_calls_made += 1
        return result
    except Exception as exc:
        raise DiagnosticQueryError(f"{type(exc).__name__}: {exc}") from exc
