"""
Tests for the Phase 8 UI boundary (src/inference/ui_adapter.py):
  Step 1: the run_diagnostic_query() boundary itself.
  Step 2: process-lifetime GraphRAG engine reuse across repeated questions.

Pure unit tests -- no GraphRAG, no index, no API calls. build_query_context()
and run_pipeline() (the two calls run_diagnostic_query() now makes -- see
_get_query_context() in ui_adapter.py) are monkeypatched out with small
fakes, so these tests only exercise the adapter's OWN logic: fixed-path
wiring, engine caching/reuse, pass-through of the PipelineResult, input
validation, the exception boundary, and that the public signature stays
exactly (question: str).

Test-to-requirement map:
  1: test_first_query_calls_build_query_context_with_fixed_index_paths
  2: test_pipeline_result_is_returned_unchanged
  3: test_empty_and_whitespace_questions_are_rejected_without_calling_pipeline
  4: test_public_signature_exposes_only_question
  5: test_infrastructure_failure_is_wrapped_in_diagnostic_query_error
  A: test_first_query_creates_exactly_one_engine
  B: test_second_query_reuses_same_engine_object
  C: test_second_query_calls_build_context_directly_with_new_question
  D: test_pipeline_results_are_not_reused_across_queries
  E: (covered by section 3 below)
  F: (covered by section 5 below, both call sites)
  H: test_concurrent_first_calls_initialize_only_one_engine
"""

from __future__ import annotations

import inspect
import threading
import time

import pytest

import src.inference.ui_adapter as ui_adapter
from src.inference.context import QueryContext
from src.inference.schemas import PipelineResult, PremiseCheckResult
from src.inference.ui_adapter import (
    INDEX_ROOT,
    OUTPUT_DIR,
    DiagnosticQueryError,
    run_diagnostic_query,
)


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    """_cached_engine is module-level, process-lifetime state -- reset it
    before and after every test so tests can't leak a cached engine into
    each other (pytest runs every test in this file in one process)."""
    ui_adapter._cached_engine = None
    yield
    ui_adapter._cached_engine = None


def make_dummy_result(question: str = "dummy question") -> PipelineResult:
    premise = PremiseCheckResult(
        status="no_claim",
        method="deterministic",
        claimed_direction="neutral",
        target_state=None,
        target_period=None,
        baseline_period=None,
    )
    return PipelineResult(
        question=question,
        premise_check=premise,
        answer=None,
        grounding_check=None,
        final_decision="hedged",
        final_text="dummy final answer text",
        llm_calls_made=0,
    )


class FakeContextBuilder:
    """Stands in for engine.context_builder -- records every direct
    build_context() call so tests can prove retrieval happens fresh for
    every question, cached engine or not."""

    def __init__(self):
        self.build_context_calls: list[str] = []

    def build_context(self, query, **kwargs):
        self.build_context_calls.append(query)
        return f"context_result::{query}"


class FakeEngine:
    """Stands in for GraphRAG's LocalSearch -- only needs the two
    attributes _get_query_context() actually touches on a cache hit."""

    def __init__(self):
        self.context_builder = FakeContextBuilder()
        self.context_builder_params: dict = {}


def make_fake_build_query_context(engines: list):
    """engines is a list this fake appends every engine it constructs to,
    so a test can assert exactly how many times a NEW engine was built."""
    calls: list[str] = []

    def fake_build_query_context(*, index_root, output_dir, query, **kwargs):
        calls.append(query)
        engine = FakeEngine()
        engines.append(engine)
        return QueryContext(engine=engine, context_result=f"initial_context::{query}")

    return fake_build_query_context, calls


def make_fake_run_pipeline():
    calls: list[tuple] = []

    def fake_run_pipeline(question, qctx, **kwargs):
        calls.append((question, qctx))
        return make_dummy_result(question)

    return fake_run_pipeline, calls


def _patch_both(monkeypatch, engines):
    fake_bqc, bqc_calls = make_fake_build_query_context(engines)
    fake_rp, rp_calls = make_fake_run_pipeline()
    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", fake_bqc)
    monkeypatch.setattr("src.inference.ui_adapter.run_pipeline", fake_rp)
    return bqc_calls, rp_calls


# ---------------------------------------------------------------------------
# 1: fixed index/output paths -- the UI never supplies these
# ---------------------------------------------------------------------------


def test_first_query_calls_build_query_context_with_fixed_index_paths(monkeypatch):
    engines: list = []
    bqc_calls, _ = _patch_both(monkeypatch, engines)

    captured = {}
    fake_bqc = ui_adapter.build_query_context

    def spying_bqc(*, index_root, output_dir, query, **kwargs):
        captured["index_root"] = index_root
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        return fake_bqc(index_root=index_root, output_dir=output_dir, query=query, **kwargs)

    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", spying_bqc)

    run_diagnostic_query("Why did Bihar's Service Level decline?")

    assert captured["index_root"] == INDEX_ROOT
    assert captured["output_dir"] == OUTPUT_DIR
    # No test-only seams leak through the adapter to the real pipeline call.
    assert "answer_fn" not in captured["kwargs"]
    assert "retry_fn" not in captured["kwargs"]


def test_question_is_stripped_before_being_passed_through(monkeypatch):
    engines: list = []
    bqc_calls, rp_calls = _patch_both(monkeypatch, engines)

    run_diagnostic_query("  Why did Bihar's Service Level decline?  ")

    assert bqc_calls[0] == "Why did Bihar's Service Level decline?"
    assert rp_calls[0][0] == "Why did Bihar's Service Level decline?"


# ---------------------------------------------------------------------------
# 2: the PipelineResult comes back through unchanged (aside from the
#    documented +1 llm_calls_made bookkeeping, identical to pre-Step-2)
# ---------------------------------------------------------------------------


def test_pipeline_result_is_returned_unchanged(monkeypatch):
    engines: list = []
    _patch_both(monkeypatch, engines)

    result = run_diagnostic_query("some question")

    assert result.final_text == "dummy final answer text"
    assert result.final_decision == "hedged"
    # run_pipeline's dummy starts at 0; the adapter adds exactly 1 for the
    # retrieval call -- same accounting as the old answer_question() wrapper.
    assert result.llm_calls_made == 1


# ---------------------------------------------------------------------------
# 3: empty/whitespace/non-string input rejected before the pipeline is touched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_question", ["", "   ", "\n\t"])
def test_empty_and_whitespace_questions_are_rejected_without_calling_pipeline(monkeypatch, bad_question):
    engines: list = []
    bqc_calls, rp_calls = _patch_both(monkeypatch, engines)

    with pytest.raises(ValueError):
        run_diagnostic_query(bad_question)

    assert bqc_calls == []  # the pipeline must never be reached
    assert rp_calls == []


def test_non_string_question_rejected(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", explode)
    monkeypatch.setattr("src.inference.ui_adapter.run_pipeline", explode)

    with pytest.raises(ValueError):
        run_diagnostic_query(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4: the public function exposes ONLY `question` -- no answer_fn/retry_fn,
#    no index_root/output_dir, nothing test-only or GraphRAG-shaped
# ---------------------------------------------------------------------------


def test_public_signature_exposes_only_question():
    params = inspect.signature(run_diagnostic_query).parameters
    assert set(params.keys()) == {"question"}
    assert "answer_fn" not in params
    assert "retry_fn" not in params


# ---------------------------------------------------------------------------
# 5: infrastructure failures are narrowed to ONE exception type, with the
#    original error preserved for debugging -- from EITHER call site
#    (a cold-cache engine build, or a warm-cache retrieval/pipeline call)
# ---------------------------------------------------------------------------


def test_infrastructure_failure_during_engine_build_is_wrapped(monkeypatch):
    original = ConnectionError("could not reach the completion API")

    def raising_build_query_context(**kwargs):
        raise original

    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", raising_build_query_context)

    with pytest.raises(DiagnosticQueryError) as exc_info:
        run_diagnostic_query("Why did Bihar's Service Level decline?")

    assert exc_info.value.__cause__ is original
    assert "ConnectionError" in str(exc_info.value)
    assert "could not reach the completion API" in str(exc_info.value)


def test_infrastructure_failure_during_run_pipeline_is_wrapped(monkeypatch):
    """Same as above, but the failure happens in run_pipeline() on a
    WARM cache (engine already cached from a prior successful call) --
    proves the error boundary still applies after the caching change."""
    engines: list = []
    _patch_both(monkeypatch, engines)
    run_diagnostic_query("first question, warms the cache")

    original = TimeoutError("completion API timed out")

    def raising_run_pipeline(question, qctx, **kwargs):
        raise original

    monkeypatch.setattr("src.inference.ui_adapter.run_pipeline", raising_run_pipeline)

    with pytest.raises(DiagnosticQueryError) as exc_info:
        run_diagnostic_query("second question, cache is warm")

    assert exc_info.value.__cause__ is original
    assert "TimeoutError" in str(exc_info.value)


def test_pipeline_evidence_outcomes_are_not_treated_as_errors(monkeypatch):
    """A normal 'insufficient_evidence' PipelineResult is not an
    infrastructure failure -- it must come back as a plain return value,
    never raise DiagnosticQueryError."""
    engines: list = []
    fake_bqc, _ = make_fake_build_query_context(engines)

    def fake_run_pipeline(question, qctx, **kwargs):
        result = make_dummy_result(question)
        result.final_decision = "insufficient_evidence"
        return result

    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", fake_bqc)
    monkeypatch.setattr("src.inference.ui_adapter.run_pipeline", fake_run_pipeline)

    result = run_diagnostic_query("Why did Bihar's Service Level improve in a period with no baseline?")
    assert result.final_decision == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Phase 8 Step 2 -- engine caching/reuse behavior
# ---------------------------------------------------------------------------


def test_first_query_creates_exactly_one_engine(monkeypatch):
    engines: list = []
    bqc_calls, _ = _patch_both(monkeypatch, engines)

    run_diagnostic_query("question one")

    assert len(bqc_calls) == 1
    assert len(engines) == 1


def test_second_query_reuses_same_engine_object(monkeypatch):
    engines: list = []
    bqc_calls, rp_calls = _patch_both(monkeypatch, engines)

    run_diagnostic_query("question one")
    run_diagnostic_query("question two")

    assert len(bqc_calls) == 1  # the expensive path only ran once
    assert len(engines) == 1
    # both run_pipeline calls got a QueryContext wrapping the SAME engine object
    assert rp_calls[0][1].engine is engines[0]
    assert rp_calls[1][1].engine is engines[0]


def test_second_query_calls_build_context_directly_with_new_question(monkeypatch):
    engines: list = []
    _patch_both(monkeypatch, engines)

    run_diagnostic_query("question one")
    run_diagnostic_query("question two")

    engine = engines[0]
    # question one's context_result came for free from build_query_context()
    # itself (no direct build_context() call on the engine was needed for it).
    assert engine.context_builder.build_context_calls == ["question two"]


def test_first_query_context_result_comes_from_build_query_context_not_a_second_call(monkeypatch):
    """Requirement 5: the first query must not pay for retrieval twice --
    its context_result is whatever build_query_context() returned, not a
    second direct build_context() call on the freshly cached engine."""
    engines: list = []
    _, rp_calls = _patch_both(monkeypatch, engines)

    run_diagnostic_query("question one")

    assert engines[0].context_builder.build_context_calls == []
    assert rp_calls[0][1].context_result == "initial_context::question one"


def test_pipeline_results_are_not_reused_across_queries(monkeypatch):
    engines: list = []
    _patch_both(monkeypatch, engines)

    result1 = run_diagnostic_query("question one")
    result2 = run_diagnostic_query("question two")

    assert result1 is not result2
    assert result1.question == "question one"
    assert result2.question == "question two"


def test_three_questions_share_one_engine_but_get_distinct_contexts(monkeypatch):
    engines: list = []
    bqc_calls, rp_calls = _patch_both(monkeypatch, engines)

    for q in ["q1", "q2", "q3"]:
        run_diagnostic_query(q)

    assert len(bqc_calls) == 1
    assert len(engines) == 1
    assert [q for q, _ in rp_calls] == ["q1", "q2", "q3"]
    contexts = [qctx.context_result for _, qctx in rp_calls]
    assert contexts == ["initial_context::q1", "context_result::q2", "context_result::q3"]


# ---------------------------------------------------------------------------
# H: concurrent first calls must not build two engines
# ---------------------------------------------------------------------------


def test_concurrent_first_calls_initialize_only_one_engine(monkeypatch):
    engines: list = []
    bqc_calls: list = []
    call_started = threading.Event()
    release = threading.Event()

    def fake_build_query_context(*, index_root, output_dir, query, **kwargs):
        bqc_calls.append(query)
        call_started.set()
        assert release.wait(timeout=5), "second thread never reached the lock in time"
        engine = FakeEngine()
        engines.append(engine)
        return QueryContext(engine=engine, context_result=f"initial_context::{query}")

    def fake_run_pipeline(question, qctx, **kwargs):
        return make_dummy_result(question)

    monkeypatch.setattr("src.inference.ui_adapter.build_query_context", fake_build_query_context)
    monkeypatch.setattr("src.inference.ui_adapter.run_pipeline", fake_run_pipeline)

    results = []
    errors = []

    def worker(q):
        try:
            results.append(run_diagnostic_query(q))
        except Exception as exc:  # pragma: no cover - only hit on a real failure
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("question A",))
    t1.start()
    assert call_started.wait(timeout=5), "first thread never entered build_query_context"
    # t1 is now inside build_query_context, holding _engine_lock, paused on release.wait().
    t2 = threading.Thread(target=worker, args=("question B",))
    t2.start()
    time.sleep(0.05)  # give t2 time to reach _get_query_context() and block on _engine_lock
    release.set()  # let t1 finish building/caching the engine, then release the lock
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"worker thread(s) raised: {errors}"
    assert len(results) == 2
    assert len(bqc_calls) == 1  # build_query_context (the expensive path) called exactly once
    assert len(engines) == 1  # exactly one engine constructed, despite two concurrent callers
