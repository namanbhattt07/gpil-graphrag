"""
Tests for the Phase 7 pipeline orchestrator (src/inference/pipeline.py),
including the bounded single-retry behavior added on top of the original
premise-check -> generate -> grounding-check flow.

These test run_pipeline() -- the pure orchestration logic -- with a stub
QueryContext (a real ContextBuilderResult built from a synthetic sources
DataFrame, plus a placeholder engine object) and MOCKED answer_fn/retry_fn
functions, per the "use mocked LLMs in tests" instruction. No GraphRAG
index, no embedding call, no completion call, no API cost anywhere in this
file. (src/inference's actual completion-call plumbing is covered
separately, with a mocked model client, in test_inference_answer.py.)

Test-to-requirement map (see the bounded-retry spec this file implements):
  A: test_clean_answer_produces_no_retry
  B: test_one_unsupported_qualifier_triggers_retry_that_fixes_it
  C: test_retry_still_unsupported_fails_closed
  D: test_direction_contradiction_can_be_corrected_by_retry
  E: test_retry_is_never_attempted_more_than_once (+ folded into B/C via call counting)
  F: test_retry_success_preserves_unflagged_content_verbatim
  G: test_completion_call_counts_clean_vs_retry_success_vs_retry_failure
  H: test_real_bihar_service_level_concerning_case_is_fixed_by_retry
"""

import pandas as pd
import pytest

from src.inference.context import QueryContext
from src.inference.grounding_check import GroundingCheckResult
from src.inference.pipeline import answer_question, run_pipeline
from src.inference.schemas import AnswerResult, GroundingIssue
from tests.test_inference_premise_check import make_sources_df

from graphrag.query.context_builder.builders import ContextBuilderResult


def make_query_context(docs: list[tuple[str, str, dict]]) -> QueryContext:
    """Build a stub QueryContext with a real ContextBuilderResult (so
    run_pipeline exercises the actual dataclass shape) but no real
    GraphRAG engine -- fine, since these tests always inject mocked
    answer_fn/retry_fn that never actually touch `engine`."""
    sources = make_sources_df(docs)
    context_result = ContextBuilderResult(
        context_chunks="<formatted context text, unused by these tests>",
        context_records={"sources": sources, "reports": pd.DataFrame(columns=["id", "content"])},
    )
    return QueryContext(engine=object(), context_result=context_result)


async def forbidden_answer_fn(engine, query, context_result):
    raise AssertionError("answer_fn must not be called when the premise check fails")


async def forbidden_retry_fn(engine, query, context_result, draft, grounding):
    raise AssertionError("retry_fn must not be called when the first draft already passed grounding")


def make_stub_answer_fn(text: str):
    async def _stub(engine, query, context_result):
        return AnswerResult(text=text, claims=[], llm_calls=1)

    return _stub


def make_counting_retry_fn(text: str):
    """Returns a retry_fn stub that records every call it receives (draft,
    grounding, query) so tests can assert both the CONTENT it was called
    with and that it was called AT MOST ONCE."""
    calls: list[dict] = []

    async def _stub(engine, query, context_result, draft, grounding):
        calls.append({"engine": engine, "query": query, "context_result": context_result, "draft": draft, "grounding": grounding})
        return AnswerResult(text=text, claims=[], llm_calls=1)

    _stub.calls = calls
    return _stub


# ---------------------------------------------------------------------------
# Premise gate: unchanged, short-circuits before any generation/retry call
# ---------------------------------------------------------------------------


def test_false_premise_short_circuits_and_never_calls_answer_fn():
    qctx = make_query_context(
        [
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
            ("Bihar", "November 2025", {"Productivity": 89.0, "Service Level": 92.9}),
        ]
    )
    result = run_pipeline(
        "Why did Bihar's performance decline in October 2025?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
    )
    assert result.premise_check.status == "insufficient_data"
    assert result.final_decision == "insufficient_evidence"
    assert result.llm_calls_made == 0
    assert result.retry_attempted is False


def test_undefined_metric_premise_short_circuits_as_insufficient_evidence():
    """Superlative/ranking questions ('least performing') must fail closed
    the same way insufficient_data does -- zero generation calls, a
    deterministic hedge text, never handed to answer_fn."""
    qctx = make_query_context([("Bihar", "November 2025", {"Productivity": 89.0})])
    result = run_pipeline(
        "Which state was the least performing in 2025 and why?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
    )
    assert result.premise_check.status == "undefined_metric"
    assert result.final_decision == "insufficient_evidence"
    assert result.llm_calls_made == 0
    assert result.retry_attempted is False
    assert result.final_text == result.premise_check.explanation


def test_contradicted_premise_also_short_circuits():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 80.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
    )
    assert result.premise_check.status == "contradicted"
    assert result.final_decision == "hedged"
    assert result.llm_calls_made == 0


# ---------------------------------------------------------------------------
# A: clean answer -> no retry
# ---------------------------------------------------------------------------


def test_clean_answer_produces_no_retry():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    clean_answer = "Bihar's Productivity fell from 92.0% in September to 85.5% in October 2025 [Data: Sources (0)]."
    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(clean_answer),
        retry_fn=forbidden_retry_fn,  # must never be called
    )
    assert result.premise_check.status == "supported"
    assert result.grounding_check.status == "pass"
    assert result.retry_attempted is False
    assert result.retry_answer is None
    assert result.retry_grounding_check is None
    assert result.final_decision == "pass_through"
    assert result.final_text == clean_answer
    assert result.llm_calls_made == 1


def test_no_claim_question_also_produces_no_retry_when_clean():
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    result = run_pipeline(
        "What was Bihar's performance in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn("Bihar's Productivity in October 2025 was 85.5% [Data: Sources (0)]."),
        retry_fn=forbidden_retry_fn,
    )
    assert result.premise_check.status == "no_claim"
    assert result.final_decision == "pass_through"
    assert result.llm_calls_made == 1


# ---------------------------------------------------------------------------
# B: one unsupported qualifier -> one retry -> corrected answer passes
# ---------------------------------------------------------------------------


def test_one_unsupported_qualifier_triggers_retry_that_fixes_it():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    bad_draft = "Bihar's Productivity fell to an alarming 85.5% in October 2025 [Data: Sources (0)]."
    corrected = "Bihar's Productivity fell to 85.5% in October 2025 [Data: Sources (0)]."
    retry_stub = make_counting_retry_fn(corrected)

    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(bad_draft),
        retry_fn=retry_stub,
    )

    assert result.grounding_check.status == "fail"
    assert any(i.term == "alarming" for i in result.grounding_check.flagged_issues)
    assert result.retry_attempted is True
    assert len(retry_stub.calls) == 1  # exactly one retry call
    assert result.retry_grounding_check.status == "pass"
    assert result.final_decision == "regenerated"
    assert result.final_text == corrected
    assert result.llm_calls_made == 2


# ---------------------------------------------------------------------------
# C: retry still contains an unsupported claim -> fail closed
# ---------------------------------------------------------------------------


def test_retry_still_unsupported_fails_closed():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    bad_draft = "Bihar's Productivity fell to an alarming 85.5% in October 2025."
    still_bad = "Bihar's Productivity fell to a concerning 85.5% in October 2025."  # different unsupported word
    retry_stub = make_counting_retry_fn(still_bad)

    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(bad_draft),
        retry_fn=retry_stub,
    )

    assert result.retry_attempted is True
    assert len(retry_stub.calls) == 1  # retried exactly once, not looped
    assert result.retry_grounding_check.status == "fail"
    assert result.final_decision == "insufficient_evidence"
    # The still-unverified regenerated text must NOT be handed back as the answer.
    assert result.final_text != still_bad
    assert "concerning" in result.final_text  # explanation names the remaining issue
    assert result.llm_calls_made == 2


# ---------------------------------------------------------------------------
# D: direction contradiction -> retry can correct it
# ---------------------------------------------------------------------------


def test_direction_contradiction_can_be_corrected_by_retry():
    # Use a real "supported" decline premise (Sept 92.0 -> Oct 85.5, down)
    # so premise_check populates a real MetricComparison for grounding_check
    # to validate the draft's direction word against.
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    bad_draft = "Bihar's Productivity climbed in October 2025, reaching 85.5% [Data: Sources (0)]."
    corrected = "Bihar's Productivity fell in October 2025, reaching 85.5% [Data: Sources (0)]."
    retry_stub = make_counting_retry_fn(corrected)

    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(bad_draft),
        retry_fn=retry_stub,
    )

    assert any(i.issue_type == "direction_contradiction" for i in result.grounding_check.flagged_issues)
    assert len(retry_stub.calls) == 1
    assert result.retry_grounding_check.status == "pass"
    assert result.final_decision == "regenerated"
    assert result.final_text == corrected


# ---------------------------------------------------------------------------
# E: retry is never attempted more than once (explicit, beyond call counts above)
# ---------------------------------------------------------------------------


def test_retry_is_never_attempted_more_than_once_even_when_still_failing():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    retry_stub = make_counting_retry_fn("Still alarming and still concerning, 85.5%.")
    run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn("An alarming 85.5% in October 2025."),
        retry_fn=retry_stub,
    )
    # However bad the retry's own output is, run_pipeline has no loop back
    # into retry_fn -- exactly one call, always.
    assert len(retry_stub.calls) == 1


# ---------------------------------------------------------------------------
# F: supported claims/citations remain unchanged through a successful retry
# ---------------------------------------------------------------------------


def test_retry_success_preserves_unflagged_content_verbatim():
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    draft_text = "Bihar's Productivity fell to an alarming 85.5% [Data: Sources (0); Entities (4, 6)]."
    corrected = "Bihar's Productivity fell to 85.5% [Data: Sources (0); Entities (4, 6)]."
    retry_stub = make_counting_retry_fn(corrected)

    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(draft_text),
        retry_fn=retry_stub,
    )

    # retry_fn must receive the ORIGINAL draft object (with its original
    # text/citations) untouched, not some pipeline-mangled version.
    assert retry_stub.calls[0]["draft"].text == draft_text
    # The citation markers survive into the final regenerated text.
    assert "[Data: Sources (0); Entities (4, 6)]" in result.final_text
    # Numbers survive unchanged too.
    assert "85.5%" in result.final_text
    assert "alarming" not in result.final_text


# ---------------------------------------------------------------------------
# G: completion call counts -- clean=1, retry-success=2, retry-failure=2
# ---------------------------------------------------------------------------


def test_completion_call_counts_clean_vs_retry_success_vs_retry_failure():
    def make_qctx():
        return make_query_context(
            [
                ("Bihar", "September 2025", {"Productivity": 92.0}),
                ("Bihar", "October 2025", {"Productivity": 85.5}),
            ]
        )
    question = "Why did Bihar's Productivity decline in October 2025?"

    clean_result = run_pipeline(
        question, make_qctx(), answer_fn=make_stub_answer_fn("Fell to 85.5% [Data: Sources (0)]."), retry_fn=forbidden_retry_fn
    )
    assert clean_result.llm_calls_made == 1

    success_result = run_pipeline(
        question,
        make_qctx(),
        answer_fn=make_stub_answer_fn("An alarming 85.5%."),
        retry_fn=make_counting_retry_fn("A plain 85.5% [Data: Sources (0)]."),
    )
    assert success_result.llm_calls_made == 2

    failure_result = run_pipeline(
        question,
        make_qctx(),
        answer_fn=make_stub_answer_fn("An alarming 85.5%."),
        retry_fn=make_counting_retry_fn("Still a concerning 85.5%."),
    )
    assert failure_result.llm_calls_made == 2


# ---------------------------------------------------------------------------
# H: exact real-world replay -- the live Bihar Service Level "concerning" case
# ---------------------------------------------------------------------------

REAL_DRAFTED_ANSWER = """In examining the Service Level metrics for Bihar, we see that it experienced a decline from October to November 2025. Specifically, the Service Level was 94.3% in October, which then decreased to 92.9% in November. This drop of 1.4 percentage points raises immediate questions regarding the underlying factors contributing to this change.

Several factors could be at play behind this decline. Reports indicate that there are notable challenges within the supply chain and inventory management in Bihar, particularly affecting specific distributors such as Zhang, Brooks and Miles Distributors, which reported a Service Level significantly below the state average in November (82.6% compared to 92.9% state-wide) [Data: Reports (9); Entities (147); Sources (3)].

In conclusion, the decline in Bihar's Service Level from October to November 2025 can likely be attributed to significant operational challenges among distributors, particularly concerning inventory and out-of-stock issues; adjustments in distribution strategies may be necessary to enhance overall service performance in subsequent months."""

# Genuinely fully corrected: the claim-level+causal-language checker (this
# implementation) catches MORE than the original word-level checker did --
# not just "concerning" but also the two causal connectors ("contributing
# to", "attributed to") neither of which the cited evidence explicitly
# establishes. A real fix has to hedge those too, not just the one adjective.
REAL_CORRECTED_ANSWER = """In examining the Service Level metrics for Bihar, we see that it experienced a decline from October to November 2025. Specifically, the Service Level was 94.3% in October, which then decreased to 92.9% in November. This drop of 1.4 percentage points raises immediate questions about what happened during this period.

Several factors could be at play behind this decline. Reports indicate that there are notable challenges within the supply chain and inventory management in Bihar, particularly affecting specific distributors such as Zhang, Brooks and Miles Distributors, which reported a Service Level significantly below the state average in November (82.6% compared to 92.9% state-wide) [Data: Reports (9); Entities (147); Sources (3)].

In conclusion, the decline in Bihar's Service Level from October to November 2025 coincided with operational challenges among distributors, particularly inventory and out-of-stock issues; adjustments in distribution strategies may be necessary to enhance overall service performance in subsequent months."""


def test_real_bihar_service_level_concerning_case_is_fixed_by_retry():
    """Session-recorded replay of the live finding: the real drafted answer
    had an unsupported qualifier ('concerning') AND unsupported causal
    language ('contributing to', 'attributed to') in an otherwise
    accurate, well-cited answer. With the retry now implemented, that
    answer should come back FIXED instead of withheld entirely."""
    qctx = make_query_context(
        [
            ("Bihar", "October 2025", {"Service Level": 94.3}),
            ("Bihar", "November 2025", {"Service Level": 92.9}),
        ]
    )
    retry_stub = make_counting_retry_fn(REAL_CORRECTED_ANSWER)

    result = run_pipeline(
        "Why did Bihar's Service Level decline from October to November 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(REAL_DRAFTED_ANSWER),
        retry_fn=retry_stub,
    )

    assert result.premise_check.status == "supported"
    assert result.grounding_check.status == "fail"
    flagged_types = {i.issue_type for i in result.grounding_check.flagged_issues}
    assert any(i.term == "concerning" for i in result.grounding_check.flagged_issues)
    assert "unsupported_causal" in flagged_types
    assert len(retry_stub.calls) == 1
    # The flagged issues must have been passed into the retry call.
    retry_issue_types = {i.issue_type for i in retry_stub.calls[0]["grounding"].flagged_issues}
    assert "unsupported_causal" in retry_issue_types
    assert result.retry_grounding_check.status == "pass"
    assert result.final_decision == "regenerated"
    assert result.final_text == REAL_CORRECTED_ANSWER
    assert "concerning" not in result.final_text
    assert result.llm_calls_made == 2


# ---------------------------------------------------------------------------
# Phase 8 regression: the live Q1 replay. A distributor-deviation claim
# whose own metric field never names a category (just "Out-of-Stock Rate")
# must not let a same-shaped sentence about a DIFFERENT category (Candy,
# not Ferrero) satisfy grounding merely because the question itself never
# reaches the claim-level checker -- run_pipeline must thread `question`
# through to check_grounding() on both the first draft and the retry.
# ---------------------------------------------------------------------------

Q1_QUESTION = (
    "For Bihar in October 2025, what was the channel-level Out-of-Stock Rate for the "
    "Ferrero category, and which distributor(s) showed a significant deviation on "
    "Out-of-Stock Rate for Ferrero that month?"
)

REAL_BIHAR_OCTOBER_TEXT = (
    "State: Bihar\nPeriod: October 2025\n\n"
    "Distributor-level deviations, October 2025: "
    "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on "
    "Out-of-Stock Rate for the Ferrero category in October 2025: 42.9% vs. the state average "
    "of 5.2%, a gap of 37.7 percentage points. "
    "Distributor Long, Anderson and Irwin Distributors showed a significant deviation on "
    "Out-of-Stock Rate for the Candy category in October 2025: 28.6% vs. the state average "
    "of 10.4%, a gap of 18.2 percentage points. "
    "For the Ferrero category in Bihar during October 2025, the channel-level Out-of-Stock "
    "rate was 5.2%."
)


def make_query_context_from_text(record_id: str, text: str) -> QueryContext:
    """Same shape as make_query_context(), but with one raw source
    document's text supplied verbatim -- used when a test needs the exact
    distributor-deviation bullet phrasing the real pilot index uses, which
    make_source_doc()'s fixed KPI-table format doesn't produce."""
    sources = pd.DataFrame([{"id": record_id, "text": text}])
    context_result = ContextBuilderResult(
        context_chunks="<formatted context text, unused by these tests>",
        context_records={"sources": sources, "reports": pd.DataFrame(columns=["id", "content"])},
    )
    return QueryContext(engine=object(), context_result=context_result)


def make_claims_answer_fn(text: str, claims):
    async def _stub(engine, query, context_result):
        return AnswerResult(text=text, claims=claims, llm_calls=1)

    return _stub


def make_claims_retry_fn(text: str, claims):
    """Like make_claims_answer_fn(), but with the retry_fn signature -- so
    the retry's OWN grounding check (also claim-scoped) can be exercised,
    instead of make_counting_retry_fn()'s always-empty claims list."""
    async def _stub(engine, query, context_result, draft, grounding):
        return AnswerResult(text=text, claims=claims, llm_calls=1)

    return _stub


def test_run_pipeline_threads_question_to_reject_wrong_category_distributor():
    """The actual live Q1 regression, through the real (unmocked)
    check_grounding -- only answer_fn is stubbed. A drafted answer citing a
    genuine Ferrero deviation (Zhang, 42.9%) alongside a misattributed
    Candy-as-Ferrero deviation (Long, Anderson and Irwin, 28.6%) must fail
    grounding specifically on the Candy claim, proving run_pipeline passes
    `question` through to check_grounding()."""
    from src.inference.schemas import AnswerClaim

    qctx = make_query_context_from_text("0", REAL_BIHAR_OCTOBER_TEXT)
    claims = [
        AnswerClaim(
            claim_id=0, claim_text="Zhang 42.9%", claim_type="deviation",
            entity="Zhang, Brooks and Miles Distributors", metric="Out-of-Stock Rate",
            value=42.9, citations=["Sources (0)"],
        ),
        AnswerClaim(
            claim_id=1, claim_text="Long, Anderson and Irwin 28.6%", claim_type="deviation",
            entity="Long, Anderson and Irwin Distributors", metric="Out-of-Stock Rate",
            value=28.6, citations=["Sources (0)"],
        ),
    ]
    answer_text = (
        "Zhang, Brooks and Miles Distributors showed a Ferrero deviation of 42.9% [Data: Sources (0)]. "
        "Long, Anderson and Irwin Distributors showed a Ferrero deviation of 28.6% [Data: Sources (0)]."
    )

    # The retry path IS reached (first draft fails), so use a retry stub
    # that returns the SAME wrong content AND claims -- proving the failure
    # survives and is not an artifact of a mocked retry silently "fixing"
    # it, and that the retry's OWN grounding check also receives `question`.
    retry_stub = make_claims_retry_fn(answer_text, claims)
    result = run_pipeline(
        Q1_QUESTION,
        qctx,
        answer_fn=make_claims_answer_fn(answer_text, claims),
        retry_fn=retry_stub,
    )

    assert result.grounding_check.status == "fail"
    assert any(i.claim_id == 1 for i in result.grounding_check.flagged_issues)
    assert not any(i.claim_id == 0 for i in result.grounding_check.flagged_issues)
    assert result.retry_attempted is True
    assert result.retry_grounding_check.status == "fail"
    assert any(i.claim_id == 1 for i in result.retry_grounding_check.flagged_issues)


# ---------------------------------------------------------------------------
# answer_question(): thin real-world wrapper bookkeeping (still no real API)
# ---------------------------------------------------------------------------


def test_answer_question_wrapper_counts_the_embedding_call(monkeypatch):
    qctx = make_query_context(
        [
            ("Bihar", "October 2025", {"Productivity": 85.5}),
            ("Bihar", "November 2025", {"Productivity": 89.0}),
        ]
    )
    monkeypatch.setattr("src.inference.pipeline.build_query_context", lambda **kwargs: qctx)

    result = answer_question(
        question="Why did Bihar's performance decline in October 2025?",
        index_root="unused",
        output_dir="unused",
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
    )
    assert result.final_decision == "insufficient_evidence"
    # run_pipeline reports 0 (no generation call happened); the wrapper
    # adds +1 for the embedding call build_query_context always makes.
    assert result.llm_calls_made == 1


# ---------------------------------------------------------------------------
# SKU evidence sufficiency / granularity gate (Problems 1, 3, 4) + source
# provenance (Problem 2), end to end through run_pipeline().
# ---------------------------------------------------------------------------

SKU_KPI_FIXTURE_ROWS = [
    dict(state_name="Punjab", month="2025-03", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1000, qty_delivered=900,
         revenue=900000.0, service_level=0.9, billed_outlets=50, eligible_outlets=100,
         numeric_distribution=0.5, oos_pct=0.05),
    dict(state_name="Punjab", month="2025-03", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=200, qty_delivered=180,
         revenue=90000.0, service_level=0.9, billed_outlets=10, eligible_outlets=100,
         numeric_distribution=0.1, oos_pct=0.2),
]


@pytest.fixture
def sku_data_dir(tmp_path):
    pd.DataFrame(SKU_KPI_FIXTURE_ROWS).to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_sku_question_answerable_when_evidence_exists(sku_data_dir):
    """Problem 3's positive case: GraphRAG's own retrieval found NOTHING
    relevant (empty sources -- simulating a state/period the main indexed
    corpus doesn't cover well), but the deterministic SKU evidence lookup
    resolves real evidence for Punjab/March 2025, so generation proceeds
    and produces a grounded, cited answer."""
    qctx = make_query_context([])  # GraphRAG's own retrieval returns nothing usable
    # Two sentences on purpose: the ranking claim ("highest Revenue") names
    # only Alpha Pack 1, and the second sentence (no ranking/superlative
    # word) names Beta Pack 1 as supporting context -- keeps this a clean,
    # unambiguous top-1 claim for the prose-fallback ranking scanner (see
    # verify_sku_ranking_claim()'s own tests for the "who does a single
    # comparison sentence actually claim" ambiguity this sidesteps).
    answer_text = (
        "In Punjab during March 2025, Alpha Pack 1 had the highest Revenue at Rs 900,000 "
        "[Data: Sources (sku-Punjab-March_2025)]. Beta Pack 1 came next at Rs 90,000 "
        "[Data: Sources (sku-Punjab-March_2025)]."
    )
    result = run_pipeline(
        "Which SKU had the highest Revenue in Punjab in March 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(answer_text),
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.evidence_sufficiency is not None
    assert result.evidence_sufficiency.status == "sufficient"
    assert result.final_decision == "pass_through"
    assert result.grounding_check.status == "pass"
    assert result.final_text == answer_text
    assert result.llm_calls_made == 1
    # Provenance (Problem 2): even with no structured claims (the stub
    # answer_fn reports none), the plain "sources=[]" default holds -- the
    # real claim -> evidence resolution is exercised separately with a
    # claims-bearing stub below.


def test_sku_question_with_claims_produces_resolved_sources(sku_data_dir):
    """Problem 2 end to end: a claim citing the SKU evidence source
    resolves to real evidence text in the final PipelineResult."""
    from src.inference.schemas import AnswerClaim

    qctx = make_query_context([])
    claim = AnswerClaim(
        claim_id=0,
        claim_text="Alpha Pack 1 had the highest Revenue at Rs 900,000",
        claim_type="factual_numeric",
        entity="Alpha Pack 1",
        metric="Revenue",
        period="March 2025",
        value=900000,
        citations=["Sources (sku-Punjab-March_2025)"],
    )

    async def stub_with_claims(engine, query, context_result):
        return AnswerResult(
            text="Alpha Pack 1 had the highest Revenue in Punjab in March 2025 at Rs 900,000 "
                 "[Data: Sources (sku-Punjab-March_2025)].",
            claims=[claim],
            llm_calls=1,
        )

    result = run_pipeline(
        "Which SKU had the highest Revenue in Punjab in March 2025?",
        qctx,
        answer_fn=stub_with_claims,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "pass_through"
    assert len(result.sources) == 1
    assert result.sources[0].citation_ids == ["Sources (sku-Punjab-March_2025)"]
    assert "Alpha Pack 1" in result.sources[0].evidence_text
    assert "900,000" in result.sources[0].evidence_text


# ---------------------------------------------------------------------------
# Ambiguous SKU-ranking clarification gate (Design Decision: ask rather
# than guess a default "top SKU" metric)
# ---------------------------------------------------------------------------


def test_ambiguous_sku_ranking_asks_for_clarification_when_evidence_exists(sku_data_dir):
    """The question names real, in-range state/period (Punjab, March 2025
    -- matches the sku_data_dir fixture) but no ranking metric -- must
    short-circuit into needs_clarification BEFORE generation, zero LLM
    calls, naming the real available metrics, and still report
    evidence_sufficiency="sufficient" (evidence WAS resolved; the question
    is answerable in principle, just ambiguous about which metric)."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Tell me the top 3 SKUs in Punjab in March 2025.",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "needs_clarification"
    assert result.llm_calls_made == 0
    assert result.evidence_sufficiency is not None
    assert result.evidence_sufficiency.status == "sufficient"
    assert "Revenue" in result.final_text
    assert "Units Delivered" in result.final_text
    assert "Service Level" in result.final_text
    assert "Numeric Distribution" in result.final_text
    assert "Out-of-Stock Rate" in result.final_text


def test_ambiguous_sku_ranking_with_metric_named_is_not_short_circuited(sku_data_dir):
    """Regression guard: naming a real metric must proceed normally
    (existing test_sku_question_answerable_when_evidence_exists already
    covers this end to end) -- this test only checks the gate itself
    doesn't fire when ranking_metric is resolved."""
    from src.inference.query_requirements import detect_query_requirements

    req = detect_query_requirements("Top 3 SKUs in Punjab in March 2025 by Revenue")
    assert req.ranking_metric == "Revenue"


def test_missing_evidence_takes_priority_over_ranking_ambiguity(sku_data_dir):
    """When BOTH problems exist (ambiguous metric AND no data for the
    named state/period), the missing-evidence explanation must win --
    asking "which metric?" first would just be a wasted round trip before
    hitting the same missing-evidence wall regardless of the answer. This
    is the ordering fix behind
    test_sku_question_does_not_hallucinate_when_evidence_absent below."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Tell me our top 3 SKUs in Punjab in 2027.",  # ambiguous metric AND out-of-range year
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert result.evidence_sufficiency.status == "insufficient_scope"


def test_sku_question_does_not_hallucinate_when_evidence_absent(sku_data_dir):
    """Problem 3's negative case: the state/period named simply isn't in
    the (fixture) SKU KPI table -- must short-circuit BEFORE any
    generation call, with an explicit explanation, never a generic 'I
    don't have information' and never a hallucinated answer."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Tell me our top 3 SKUs in Punjab in 2027.",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert result.llm_calls_made == 0
    assert result.evidence_sufficiency is not None
    assert result.evidence_sufficiency.status == "insufficient_scope"
    # Must distinguish "the raw dataset has SKU data" from "no rows for
    # this scope" -- never claim the dataset itself has no SKU data.
    assert "DOES contain SKU-level" in result.final_text
    assert "Punjab" in result.final_text
    assert "2027" in result.final_text


def test_sku_question_with_no_state_named_is_explicit_granularity_gap(sku_data_dir):
    qctx = make_query_context([])
    result = run_pipeline(
        "Which SKU has the highest sales?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert result.llm_calls_made == 0
    assert "does not name" in result.final_text


def test_non_sku_question_has_no_evidence_sufficiency_check_and_is_unaffected(sku_data_dir):
    """Regression guard: an ordinary State-level question must be
    completely unaffected by the new SKU gate -- evidence_sufficiency
    stays None, and the existing premise/grounding flow runs exactly as
    it did before Problems 1/3/4 existed, even when a data_dir with real
    SKU data is supplied."""
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    clean_answer = "Bihar's Productivity fell from 92.0% in September to 85.5% in October 2025 [Data: Sources (0)]."
    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(clean_answer),
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.evidence_sufficiency is None
    assert result.final_decision == "pass_through"
    assert result.final_text == clean_answer


def test_sku_ranking_grounding_failure_triggers_retry_and_resolves(sku_data_dir):
    """End-to-end: a first draft that names the WRONG SKU as highest-
    Revenue fails grounding (via verify_sku_ranking_claim), triggers the
    one bounded retry, and the corrected retry passes -- proving the SKU
    evidence path is wired into the SAME retry mechanism every other
    question already uses, not a separate code path."""
    qctx = make_query_context([])
    wrong_answer = (
        "Beta Pack 1 had the highest Revenue in Punjab in March 2025 at Rs 90,000 "
        "[Data: Sources (sku-Punjab-March_2025)]."
    )
    fixed_answer = (
        "Alpha Pack 1 had the highest Revenue in Punjab in March 2025 at Rs 900,000 "
        "[Data: Sources (sku-Punjab-March_2025)]."
    )
    result = run_pipeline(
        "Which SKU had the highest Revenue in Punjab in March 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(wrong_answer),
        retry_fn=make_counting_retry_fn(fixed_answer),
        data_dir=sku_data_dir,
    )
    assert result.grounding_check.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.grounding_check.flagged_issues)
    assert result.retry_attempted is True
    assert result.final_decision == "regenerated"
    assert result.final_text == fixed_answer


# ---------------------------------------------------------------------------
# Deterministic SKU ranking (2026-08-23 stabilization pass, Fix 1) -- the
# LLM must never compute a top-N/bottom-N ranking itself. These prove the
# ranking is computed BEFORE generation (evidence_sufficiency.ranked_skus
# is populated even though answer_fn is a dumb stub that never looks at
# context), and that the deterministic block actually reaches the prompt
# text answer_fn/retry_fn would see in production.
# ---------------------------------------------------------------------------

RANKING_KPI_FIXTURE_ROWS = [
    dict(state_name="Gujarat", month="2026-04", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1000, qty_delivered=900,
         revenue=90000.0, service_level=0.90, billed_outlets=50, eligible_outlets=100,
         numeric_distribution=0.50, oos_pct=0.05),
    dict(state_name="Gujarat", month="2026-04", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=2000, qty_delivered=1900,
         revenue=190000.0, service_level=0.95, billed_outlets=80, eligible_outlets=100,
         numeric_distribution=0.80, oos_pct=0.02),
    dict(state_name="Gujarat", month="2026-04", sku_id="SKU0003", sku_name="Gamma Pack 1",
         franchise_name="Gamma", category_name="IPM", qty_ordered=500, qty_delivered=400,
         revenue=40000.0, service_level=0.80, billed_outlets=20, eligible_outlets=100,
         numeric_distribution=0.20, oos_pct=0.20),
]


@pytest.fixture
def ranking_sku_data_dir(tmp_path):
    pd.DataFrame(RANKING_KPI_FIXTURE_ROWS).to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_ranking_computed_deterministically_before_generation(ranking_sku_data_dir):
    """The ranking must exist in evidence_sufficiency.ranked_skus (computed
    entirely in sku_evidence.py) BEFORE answer_fn is ever called -- proven
    here by using a stub answer_fn that ignores context entirely, so any
    correctness in the final ranking numbers can only have come from
    deterministic computation, never from the (nonexistent, in this test)
    LLM reasoning over the facts."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Top 2 SKUs by Revenue in Gujarat in April 2026",
        qctx,
        answer_fn=make_stub_answer_fn("stub text, never reads context"),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert result.evidence_sufficiency.status == "sufficient"
    ranked = result.evidence_sufficiency.ranked_skus
    assert ranked is not None
    assert [e["sku_id"] for e in ranked] == ["SKU0002", "SKU0001"]  # Beta 190000 > Alpha 90000
    assert result.final_decision == "pass_through"


def test_ranking_block_reaches_the_generation_prompt(ranking_sku_data_dir):
    """The deterministic ranking must actually be visible in the text
    answer_fn/retry_fn receive (via context_result.context_records, the
    same object _context_data_with_facts() reads in production) -- not
    just present in evidence_sufficiency, which nothing downstream of
    run_pipeline would otherwise see."""
    from src.inference.fact_structuring import build_deterministic_ranking_block

    captured = {}

    async def capturing_answer_fn(engine, query, context_result):
        captured["block"] = build_deterministic_ranking_block(context_result.context_records)
        return AnswerResult(text="ok", claims=[], llm_calls=1)

    qctx = make_query_context([])
    run_pipeline(
        "Top 2 SKUs by Revenue in Gujarat in April 2026",
        qctx,
        answer_fn=capturing_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert "Deterministic SKU Ranking" in captured["block"]
    assert "Rank 1: SKU SKU0002" in captured["block"]
    assert "Rank 2: SKU SKU0001" in captured["block"]
    assert "do not recompute" in captured["block"].lower()


def test_bottom_n_ranking_also_computed_deterministically(ranking_sku_data_dir):
    qctx = make_query_context([])
    result = run_pipeline(
        "Bottom 2 SKUs by Out-of-Stock Rate in Gujarat in April 2026",
        qctx,
        answer_fn=make_stub_answer_fn("stub"),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    ranked = result.evidence_sufficiency.ranked_skus
    # Lowest OOS first: Beta (0.02) then Alpha (0.05).
    assert [e["sku_id"] for e in ranked] == ["SKU0002", "SKU0001"]


def test_most_selling_synonym_ranks_by_units_delivered(ranking_sku_data_dir):
    qctx = make_query_context([])
    result = run_pipeline(
        "What was the top 2 most selling SKUs in Gujarat in April 2026?",
        qctx,
        answer_fn=make_stub_answer_fn("stub"),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    ranked = result.evidence_sufficiency.ranked_skus
    assert ranked is not None
    assert ranked[0]["metric"] == "Units Delivered"
    assert [e["sku_id"] for e in ranked] == ["SKU0002", "SKU0001"]  # 1900 > 900 units delivered


def test_ranking_generation_still_grounded_when_model_correctly_restates_it(ranking_sku_data_dir):
    """End-to-end sanity: a well-behaved answer that correctly restates the
    (now deterministically-known) top-2 must still pass grounding cleanly
    -- the deterministic ranking doesn't bypass or weaken grounding, it
    just gives the model the right answer to describe."""
    qctx = make_query_context([])
    answer_text = (
        "In Gujarat during April 2026, the top 2 SKUs by Revenue were Beta Pack 1 at Rs 190,000 "
        "[Data: Sources (sku-Gujarat-April_2026)] and Alpha Pack 1 at Rs 90,000 "
        "[Data: Sources (sku-Gujarat-April_2026)]."
    )
    result = run_pipeline(
        "Top 2 SKUs by Revenue in Gujarat in April 2026",
        qctx,
        answer_fn=make_stub_answer_fn(answer_text),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert result.grounding_check.status == "pass"
    assert result.final_decision == "pass_through"


# ---------------------------------------------------------------------------
# SKU-name-only queries (Fix 2) -- a real SKU name must trigger the SKU
# evidence path even when the literal word "SKU" never appears. Uses
# ranking_sku_data_dir's real fixture SKU names ("Alpha Pack 1", "Beta
# Pack 1"), resolved via the default data_dir=None -> DEFAULT_DATA_DIR path
# is NOT exercised here (that would require the real project data/
# directory); instead data_dir is passed explicitly, and
# known_sku_names(data_dir) is called internally by run_pipeline() itself
# -- exactly the real production code path, just against a small fixture.
# ---------------------------------------------------------------------------


def test_bare_sku_name_question_resolves_sku_evidence_without_word_sku(ranking_sku_data_dir):
    qctx = make_query_context([])
    result = run_pipeline(
        "What was the revenue of Alpha Pack 1 in Gujarat in April 2026?",
        qctx,
        answer_fn=make_stub_answer_fn(
            "Alpha Pack 1 had a Revenue of Rs 90,000 in Gujarat in April 2026 "
            "[Data: Sources (sku-Gujarat-April_2026)]."
        ),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert result.evidence_sufficiency is not None
    assert result.evidence_sufficiency.status == "sufficient"
    assert result.evidence_sufficiency.requirements.required_granularity == "sku"
    assert result.final_decision == "pass_through"


def test_bare_sku_name_comparison_question_resolves_sku_evidence(ranking_sku_data_dir):
    """'Compare Alpha Pack 1 and Beta Pack 1 ...' -- two SKU names, no
    literal 'SKU' word, not a ranking question."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Compare Alpha Pack 1 and Beta Pack 1 in Gujarat in April 2026.",
        qctx,
        answer_fn=make_stub_answer_fn(
            "Beta Pack 1 had higher Revenue (Rs 190,000) than Alpha Pack 1 (Rs 90,000) in Gujarat "
            "in April 2026 [Data: Sources (sku-Gujarat-April_2026)]."
        ),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert result.evidence_sufficiency.status == "sufficient"
    assert result.evidence_sufficiency.requirements.required_granularity == "sku"


def test_ordinary_question_unaffected_by_known_sku_name_lookup(ranking_sku_data_dir):
    """A plain state-level question that happens to run against a data_dir
    WITH real SKU names must stay completely unaffected -- known_sku_names()
    is now read on every call (see pipeline.py), but must never cause a
    false-positive SKU-grain match for an unrelated question."""
    qctx = make_query_context(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    clean_answer = "Bihar's Productivity fell from 92.0% in September to 85.5% in October 2025 [Data: Sources (0)]."
    result = run_pipeline(
        "Why did Bihar's Productivity decline in October 2025?",
        qctx,
        answer_fn=make_stub_answer_fn(clean_answer),
        retry_fn=forbidden_retry_fn,
        data_dir=ranking_sku_data_dir,
    )
    assert result.evidence_sufficiency is None
    assert result.final_decision == "pass_through"


# ---------------------------------------------------------------------------
# Section 3 audit -- representative query-type coverage beyond what's
# already tested above (ranking, bare-name lookup/comparison, ambiguous
# metric, missing evidence, no-state-named). Each of these must either
# answer deterministically/grounded, ask a precise clarification, or fail
# closed -- never guess.
# ---------------------------------------------------------------------------


def test_ambiguous_best_performing_sku_also_asks_for_clarification(sku_data_dir):
    """'Best-performing SKUs' names no metric and no selling/sold word --
    genuinely ambiguous, same gate as bare 'top 3 SKUs' (Section 6)."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Which were the best-performing SKUs in Punjab in March 2025?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "needs_clarification"
    assert result.llm_calls_made == 0


def test_unknown_state_sku_question_is_explicit_granularity_gap(sku_data_dir):
    """A state this project's data doesn't cover (not one of the 28 Indian
    states) -- extract_state_from_question() returns None for it, so this
    behaves identically to 'no state named' (never silently substitutes a
    different state)."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Which SKU had the highest Revenue in Freedonia in March 2025?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert result.llm_calls_made == 0
    assert "does not name" in result.final_text


def test_unknown_period_sku_question_reports_indexed_range(sku_data_dir):
    """A real state but a period far outside what's indexed -- must state
    the actual indexed range, never a generic refusal."""
    qctx = make_query_context([])
    result = run_pipeline(
        "Which SKU had the highest Revenue in Punjab in December 2030?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert "Punjab" in result.final_text
    assert "December 2030" in result.final_text


def test_missing_state_and_period_sku_ranking_question(sku_data_dir):
    """No state, no period, no metric -- the state-missing granularity gap
    must win (nothing to rank at all without a state), same priority
    ordering as test_missing_evidence_takes_priority_over_ranking_ambiguity."""
    qctx = make_query_context([])
    result = run_pipeline(
        "What are our top SKUs?",
        qctx,
        answer_fn=forbidden_answer_fn,
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.final_decision == "insufficient_evidence"
    assert "does not name" in result.final_text


def test_malformed_query_still_fails_closed_not_crash(sku_data_dir):
    """A near-empty/garbled query must never raise -- it resolves no SKU
    or state/period requirement at all, and (since it asserts no
    decline/improve direction) reaches the ordinary "no_claim" path, which
    generates a plain answer rather than crashing or hanging on a
    malformed question."""
    qctx = make_query_context([])
    result = run_pipeline(
        "??? asdkfj",
        qctx,
        answer_fn=make_stub_answer_fn("Nothing specific was asked."),
        retry_fn=forbidden_retry_fn,
        data_dir=sku_data_dir,
    )
    assert result.evidence_sufficiency is None
    assert result.premise_check.status == "no_claim"
    assert result.final_decision == "pass_through"


# ---------------------------------------------------------------------------
# GPIL Knowledge Layer (2026-08-23): glossary rows merged into
# context_records["sources"] BEFORE generation, exactly like SKU evidence.
# ---------------------------------------------------------------------------


def make_capturing_answer_fn(text: str):
    """Like make_stub_answer_fn(), but also records the context_result it
    was actually called with, so a test can inspect whether pipeline.py
    merged glossary rows into context_records["sources"] before generation
    ran -- the thing that lets a correct, cited GPI/IPM answer pass
    grounding at all (see knowledge_layer.py's module docstring)."""
    calls: list[dict] = []

    async def _stub(engine, query, context_result):
        calls.append({"query": query, "context_result": context_result})
        return AnswerResult(text=text, claims=[], llm_calls=1)

    _stub.calls = calls
    return _stub


def test_glossary_row_merged_into_context_before_generation_for_known_term():
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    capturing = make_capturing_answer_fn(
        "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]."
    )
    result = run_pipeline("What is GPI?", qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn)
    assert len(capturing.calls) == 1
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert "glossary-gpi" in set(sources["id"])
    assert result.final_decision == "pass_through"


def test_no_glossary_rows_merged_for_a_question_naming_no_known_term():
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    capturing = make_capturing_answer_fn("Bihar's Productivity in October 2025 was 85.5% [Data: Sources (0)].")
    run_pipeline("What was Bihar's performance in October 2025?", qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn)
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert not any(str(i).startswith("glossary-") for i in sources["id"])


# ---------------------------------------------------------------------------
# Named-period document completion (retrieval-recall fix): GraphRAG's own
# retrieval doesn't guarantee it returns every state+period a question
# explicitly names, even when that document is genuinely indexed. See
# named_period_evidence.py's module docstring for the live 3-period
# failure this closes.
# ---------------------------------------------------------------------------


def test_missing_named_period_document_is_injected_before_generation(tmp_path):
    (tmp_path / "Gujarat_2026-04.txt").write_text(
        "State: Gujarat\nPeriod: April 2026\n\nService Level for Gujarat in April 2026 was 92.0%.",
        encoding="utf-8",
    )
    # GraphRAG's own retrieval only found June -- April is missing.
    qctx = make_query_context([("Gujarat", "June 2026", {"Service Level": 90.8})])
    capturing = make_capturing_answer_fn(
        "Gujarat's Service Level was 92.0% in April 2026 and 90.8% in June 2026 [Data: Sources (0)]."
    )
    run_pipeline(
        "Gujarat's Service Level in April 2026 versus June 2026 -- why did it change?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn, index_input_dir=tmp_path,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    ids = set(sources["id"])
    assert any(str(i).startswith("named-period-doc-") for i in ids)
    blob = "\n".join(str(t) for t in sources["text"])
    assert "April 2026" in blob and "92.0%" in blob


def test_already_retrieved_named_period_is_not_duplicated(tmp_path):
    (tmp_path / "Gujarat_2026-06.txt").write_text(
        "State: Gujarat\nPeriod: June 2026\n\nService Level for Gujarat in June 2026 was 90.8%.",
        encoding="utf-8",
    )
    qctx = make_query_context([("Gujarat", "June 2026", {"Service Level": 90.8})])
    capturing = make_capturing_answer_fn("Gujarat's Service Level was 90.8% in June 2026 [Data: Sources (0)].")
    run_pipeline(
        "Gujarat's Service Level in June 2026 -- why?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn, index_input_dir=tmp_path,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert not any(str(i).startswith("named-period-doc-") for i in sources["id"])


def test_local_search_narrows_sources_to_the_named_state_before_generation():
    """Live bug regression: a question naming exactly one state
    (local_search-shaped) must not hand the generation model OTHER
    states' distributor-deviation sentences -- a live run had the model
    borrow an unrelated state's '20.0% vs. 8.0%' figure for a Gujarat
    distributor because 7 other states' documents sat in the same
    retrieved context."""
    qctx = make_query_context([
        ("Gujarat", "June 2026", {"Service Level": 90.8}),
        ("Odisha", "January 2026", {"Service Level": 88.0}),
        ("Chhattisgarh", "June 2025", {"Service Level": 91.0}),
    ])
    capturing = make_capturing_answer_fn("Gujarat's Service Level was 90.8% in June 2026 [Data: Sources (0)].")
    run_pipeline(
        "Why did Baxter, Thomas and Williams in Gujarat show a deviation in June 2026?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    states_seen = {str(t).splitlines()[0] for t in sources["text"]}
    assert states_seen == {"State: Gujarat"}


def test_global_search_question_does_not_narrow_sources():
    """A broad/global-shaped question (names a state but uses recurring/
    pattern language) must keep the full, naturally wide retrieved set --
    narrowing to one state would be exactly the wrong direction for a
    cross-time/cross-entity scan."""
    qctx = make_query_context([
        ("Gujarat", "April 2026", {"Service Level": 92.0}),
        ("Gujarat", "June 2026", {"Service Level": 90.8}),
        ("Odisha", "January 2026", {"Service Level": 88.0}),
    ])
    capturing = make_capturing_answer_fn(
        "Gujarat's Service Level was 92.0% in April 2026 and 90.8% in June 2026 [Data: Sources (0), Sources (1)]."
    )
    run_pipeline(
        "Is there a recurring pattern in Gujarat's Service Level over time?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    state_docs = [t for t in sources["text"] if str(t).startswith("State:")]
    assert len(state_docs) == 3  # nothing dropped, including the Odisha row


def test_no_state_named_question_does_not_narrow_sources():
    qctx = make_query_context([
        ("Gujarat", "June 2026", {"Service Level": 90.8}),
        ("Odisha", "January 2026", {"Service Level": 88.0}),
    ])
    capturing = make_capturing_answer_fn("Service Level figures are shown above [Data: Sources (0), Sources (1)].")
    run_pipeline(
        "What was the Service Level in June 2026?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    state_docs = [t for t in sources["text"] if str(t).startswith("State:")]
    assert len(state_docs) == 2  # nothing dropped, including the Odisha row


# ---------------------------------------------------------------------------
# Category-ambiguity short-circuit ("Missing categories are NEVER guessed
# from retrieved evidence"): a category-scoped-ONLY metric (Out-of-Stock
# Rate/Numeric Distribution/ACV/Range Billing) asked with no category
# named must report every category's real value, never arbitrarily lead
# with whichever one sits nearest in the retrieved text.
# ---------------------------------------------------------------------------

_GOA_CATEGORY_BLOCK_TEXT = (
    "State: Goa\nPeriod: June 2025\n\n"
    "For the GPI category (franchises: GPI_Franchise_1) in Goa during June 2025: "
    "Numeric Distribution was 45.9%, ACV (definition pending) was 45.8%, Out-of-Stock rate was 6.9%, "
    "and average Range Billing (definition pending) was 4.7%.\n"
    "For the IPM category (franchises: Marlboro) in Goa during June 2025: "
    "Numeric Distribution was 55.8%, ACV (definition pending) was 55.8%, Out-of-Stock rate was 6.2%, "
    "and average Range Billing (definition pending) was 32.6%.\n"
    "For the Ferrero category (franchises: TicTac) in Goa during June 2025: "
    "Numeric Distribution was 10.5%, ACV (definition pending) was 10.5%, Out-of-Stock rate was 6.6%, "
    "and average Range Billing (definition pending) was 15.6%.\n"
    "For the Candy category (franchises: Candy_Franchise_1) in Goa during June 2025: "
    "Numeric Distribution was 10.0%, ACV (definition pending) was 10.0%, Out-of-Stock rate was 11.0%, "
    "and average Range Billing (definition pending) was 15.3%.\n"
)


def test_category_ambiguous_oos_question_reports_all_four_categories_zero_llm_calls():
    """Live bug regression: 'What was the out of stock in Goa in June
    2025?' silently answered with just Candy's figure (11.0%), arbitrarily
    -- must instead report all 4 categories' real values with zero LLM
    calls, never guessing one."""
    qctx = make_query_context_from_text("0", _GOA_CATEGORY_BLOCK_TEXT)
    result = run_pipeline(
        "What was the out of stock in Goa in June 2025?",
        qctx, answer_fn=forbidden_answer_fn, retry_fn=forbidden_retry_fn,
    )
    assert result.final_decision == "needs_clarification"
    assert result.llm_calls_made == 0
    for label, value in [("GPI", "6.9"), ("IPM", "6.2"), ("Ferrero", "6.6"), ("Candy", "11.0")]:
        assert label in result.final_text and value in result.final_text


def test_category_explicitly_named_bypasses_the_breakdown_short_circuit():
    """'...for Candy in Goa in June 2025' explicitly names a category --
    must proceed to normal generation, not the breakdown short-circuit."""
    qctx = make_query_context_from_text("0", _GOA_CATEGORY_BLOCK_TEXT)
    capturing = make_capturing_answer_fn("Candy's Out-of-Stock Rate in Goa was 11.0% in June 2025 [Data: Sources (0)].")
    result = run_pipeline(
        "What was the out of stock for Candy in Goa in June 2025?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    assert len(capturing.calls) == 1
    assert result.final_decision == "pass_through"


def test_category_ambiguous_but_no_matching_evidence_falls_through_normally():
    """No category-level facts exist for this state+period at all (e.g. it
    genuinely wasn't retrieved/indexed) -- the short-circuit must return
    None and let the normal pipeline (which will correctly find nothing)
    handle it, not silently do nothing and crash or hang."""
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    capturing = make_capturing_answer_fn("No category-level Out-of-Stock data is available [Data: Sources (0)].")
    result = run_pipeline(
        "What was the out of stock in Bihar in October 2025?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    assert len(capturing.calls) == 1  # fell through to normal generation, not the short-circuit


def test_state_level_metric_question_unaffected_by_category_ambiguity_check():
    """An ordinary state-level (category-agnostic) metric question must
    never trigger the category short-circuit at all."""
    qctx = make_query_context([("Bihar", "October 2025", {"Service Level": 94.3})])
    capturing = make_capturing_answer_fn("Bihar's Service Level was 94.3% in October 2025 [Data: Sources (0)].")
    result = run_pipeline(
        "What was Bihar's Service Level in October 2025?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn,
    )
    assert len(capturing.calls) == 1
    assert result.final_decision == "pass_through"


def test_named_period_not_in_pilot_corpus_leaves_pipeline_unaffected(tmp_path):
    """Gujarat July 2026 isn't one of the pilot's 62 indexed documents --
    nothing should be injected, and the pipeline must fall back to its
    normal insufficient-data handling rather than reaching outside the
    pilot's own scope."""
    qctx = make_query_context([("Gujarat", "June 2026", {"Service Level": 90.8})])
    capturing = make_capturing_answer_fn("Gujarat's Service Level was 90.8% in June 2026 [Data: Sources (0)].")
    run_pipeline(
        "Gujarat's Service Level in July 2026 -- why?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn, index_input_dir=tmp_path,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert not any(str(i).startswith("named-period-doc-") for i in sources["id"])


# ---------------------------------------------------------------------------
# Evidence-triggered glossary merge (2026-08-23, SKU live validation pass):
# "Did the top-selling SKU in Gujarat change between April 2026 and June
# 2026?" never says "IPM"/"GPI" in the question text, but the resolved SKU
# evidence always names each SKU's category (GPI/IPM/Ferrero/Candy) --
# and a live run found the model spontaneously mis-expanding "IPM" as
# "Integrated Pest Management" while discussing an IPM-category SKU, with
# no glossary block present to correct it since the question-text-only
# detection never fired. Fixed by ALSO scanning the resolved SKU evidence
# text for known terms once it's available (see pipeline.py's
# "EVIDENCE-TRIGGERED GLOSSARY MERGE" docstring section).
# ---------------------------------------------------------------------------

IPM_SKU_FIXTURE_ROWS = [
    dict(state_name="Gujarat", month="2026-04", sku_id="SKU0041", sku_name="Marlboro Pack 1",
         franchise_name="Marlboro", category_name="IPM", qty_ordered=1300000, qty_delivered=1167350,
         revenue=317881078.0, service_level=0.865, billed_outlets=900, eligible_outlets=1000,
         numeric_distribution=0.731, oos_pct=0.20),
]


@pytest.fixture
def ipm_sku_data_dir(tmp_path):
    pd.DataFrame(IPM_SKU_FIXTURE_ROWS).to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_evidence_only_category_term_gets_a_glossary_row_even_when_question_never_names_it(ipm_sku_data_dir):
    """The question below never says 'IPM' -- only the resolved SKU
    evidence (Marlboro Pack 1's own category) does. glossary-ipm must
    still be merged into context_records["sources"], not just glossary-gpi
    or nothing at all."""
    qctx = make_query_context([])
    capturing = make_capturing_answer_fn(
        "Marlboro Pack 1 was Gujarat's top-selling SKU by Units Delivered in April 2026, with 1,167,350 units "
        "[Data: Sources (sku-Gujarat-April_2026)]."
    )
    run_pipeline(
        "What was the top-selling SKU in Gujarat in April 2026?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn, data_dir=ipm_sku_data_dir,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    ids = set(sources["id"])
    assert "glossary-ipm" in ids


def test_evidence_triggered_glossary_row_is_the_real_authoritative_ipm_definition(ipm_sku_data_dir):
    qctx = make_query_context([])
    capturing = make_capturing_answer_fn(
        "Marlboro Pack 1 was Gujarat's top-selling SKU by Units Delivered in April 2026 "
        "[Data: Sources (sku-Gujarat-April_2026)]."
    )
    run_pipeline(
        "What was the top-selling SKU in Gujarat in April 2026?",
        qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn, data_dir=ipm_sku_data_dir,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    ipm_row = sources[sources["id"] == "glossary-ipm"].iloc[0]
    assert "Marlboro" in ipm_row["text"]
    assert "Integrated Pest Management" in ipm_row["text"]  # named only to rule it out, see knowledge_layer.py


def test_evidence_triggered_glossary_merge_does_not_duplicate_a_question_named_term():
    """When the question ALREADY names the term (e.g. 'What is GPI...'),
    the evidence-text scan must not add a second, duplicate glossary-gpi
    row -- exactly one row per matched term, regardless of how many times
    (question text + evidence text) it was actually detected."""
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    capturing = make_capturing_answer_fn("GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)].")
    run_pipeline("What is GPI?", qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn)
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert list(sources["id"]).count("glossary-gpi") == 1


def test_evidence_triggered_glossary_merge_is_a_no_op_for_ordinary_non_sku_questions():
    """The evidence-text scan only ever runs when SKU evidence was
    actually resolved -- an ordinary state-level question (no SKU
    granularity at all) must behave exactly as before."""
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    capturing = make_capturing_answer_fn("Bihar's Productivity in October 2025 was 85.5% [Data: Sources (0)].")
    run_pipeline("What was Bihar's performance in October 2025?", qctx, answer_fn=capturing, retry_fn=forbidden_retry_fn)
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    assert not any(str(i).startswith("glossary-") for i in sources["id"])


def test_mixed_definition_and_analytical_question_carries_both_glossary_and_kpi_evidence():
    """'What is GPI and how did it perform in Gujarat in April 2026?' --
    per the task's mixed-question requirement, the glossary supplies the
    business meaning while the ordinary retrieved evidence still supplies
    the KPI numbers side by side, in the SAME merged context_records --
    neither one displaces the other."""
    qctx = make_query_context([("Gujarat", "April 2026", {"Productivity": 74.0})])
    capturing = make_capturing_answer_fn(
        "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]. "
        "Gujarat's Productivity in April 2026 was 74.0% [Data: Sources (0)]."
    )
    result = run_pipeline(
        "What is GPI and how did it perform in Gujarat in April 2026?",
        qctx,
        answer_fn=capturing,
        retry_fn=forbidden_retry_fn,
    )
    sources = capturing.calls[0]["context_result"].context_records["sources"]
    ids = set(sources["id"])
    assert "glossary-gpi" in ids
    assert "0" in ids
    assert result.final_decision == "pass_through"


def test_pipeline_passes_through_when_answer_uses_correct_gpil_specific_meaning():
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    clean = "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]."
    result = run_pipeline(
        "What is GPI?", qctx, answer_fn=make_stub_answer_fn(clean), retry_fn=forbidden_retry_fn
    )
    assert result.grounding_check.status == "pass"
    assert result.final_decision == "pass_through"
    assert result.final_text == clean


def test_pipeline_fails_closed_when_answer_fabricates_wrong_gpi_expansion():
    """Even though the fabricated answer cites the REAL, resolvable
    "glossary-gpi" id, scan_glossary_term_misuse() catches the specific
    known-wrong expansion regardless of citation -- the live bug this
    Knowledge Layer exists to fix must fail grounding, not pass it."""
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    bad_draft = "GPI stands for General Product Inventory [Data: Sources (glossary-gpi)]."
    still_bad_retry = "GPI is short for General Product Inventory [Data: Sources (glossary-gpi)]."
    retry_stub = make_counting_retry_fn(still_bad_retry)

    result = run_pipeline("What is GPI?", qctx, answer_fn=make_stub_answer_fn(bad_draft), retry_fn=retry_stub)

    assert result.grounding_check.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in result.grounding_check.flagged_issues)
    assert len(retry_stub.calls) == 1
    assert result.retry_grounding_check.status == "fail"
    assert result.final_decision == "insufficient_evidence"


def test_pipeline_fails_closed_when_answer_fabricates_wrong_ipm_expansion():
    qctx = make_query_context([("Bihar", "October 2025", {"Productivity": 85.5})])
    bad_draft = "IPM stands for Integrated Pest Management [Data: Sources (glossary-ipm)]."
    retry_stub = make_counting_retry_fn(bad_draft)

    result = run_pipeline("What is IPM?", qctx, answer_fn=make_stub_answer_fn(bad_draft), retry_fn=retry_stub)

    assert result.grounding_check.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in result.grounding_check.flagged_issues)
    assert result.final_decision == "insufficient_evidence"
