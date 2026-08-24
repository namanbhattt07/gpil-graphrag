"""
Tests for src/cli.py::_debug_line() -- the terminal debug metadata line --
and _render_sources() -- the Problem 2 source-traceability display.

Forensic finding (see conversation history): PipelineResult.grounding_check
always holds the FIRST draft's grounding result, even when a retry fixed
things (final_decision="regenerated"). _debug_line() used to print that
first-draft status unconditionally, so a successfully regenerated answer
displayed the misleading "grounding=fail" next to a perfectly good final
answer. This only tests the fix to that display logic -- no pipeline,
grounding, or retry behavior is touched or exercised here.
"""

from __future__ import annotations

from src.cli import _debug_line, _render_sources
from src.inference.schemas import GroundingCheckResult, PipelineResult, PremiseCheckResult, SourceCitation


def make_premise(status: str = "supported") -> PremiseCheckResult:
    return PremiseCheckResult(
        status=status,
        method="deterministic",
        claimed_direction="decline",
        target_state="Bihar",
        target_period="November 2025",
        baseline_period="October 2025",
    )


def make_grounding(status: str) -> GroundingCheckResult:
    return GroundingCheckResult(status=status, method="deterministic")


def make_result(**overrides) -> PipelineResult:
    defaults = dict(
        question="Why did Bihar's Service Level decline from October to November 2025?",
        premise_check=make_premise(),
        answer=None,
        grounding_check=None,
        final_decision="pass_through",
        final_text="dummy final answer text",
        llm_calls_made=2,
        retry_attempted=False,
        retry_answer=None,
        retry_grounding_check=None,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


def test_pass_through_reports_initial_grounding_status():
    """No retry -- must still show the (only) grounding_check's status."""
    result = make_result(
        final_decision="pass_through",
        grounding_check=make_grounding("pass"),
        llm_calls_made=2,
    )
    line = _debug_line(result)
    assert "grounding=pass" in line
    assert "final_decision=pass_through" in line
    assert "llm_calls=2" in line


def test_regenerated_success_reports_retry_grounding_status_not_initial():
    """The exact forensic bug: initial draft failed, retry passed, final
    answer is the retry's -- the debug line must show pass, not fail."""
    result = make_result(
        final_decision="regenerated",
        grounding_check=make_grounding("fail"),  # first draft failed
        retry_attempted=True,
        retry_grounding_check=make_grounding("pass"),  # retry fixed it
        llm_calls_made=3,
    )
    line = _debug_line(result)
    assert "grounding=pass" in line
    assert "grounding=fail" not in line
    assert "final_decision=regenerated" in line
    assert "llm_calls=3" in line


def test_retry_still_failing_reports_retry_grounding_status_as_fail():
    """Retry was attempted but STILL failed (insufficient_evidence) -- the
    retry's own fail status should show, not silently swallowed."""
    result = make_result(
        final_decision="insufficient_evidence",
        grounding_check=make_grounding("fail"),
        retry_attempted=True,
        retry_grounding_check=make_grounding("fail"),
        llm_calls_made=3,
    )
    line = _debug_line(result)
    assert "grounding=fail" in line
    assert "final_decision=insufficient_evidence" in line


def test_hedged_question_with_no_grounding_check_reports_n_a():
    """Premise contradicted/unsupported -- no generation call was ever
    made, so grounding_check is None and nothing was retried."""
    result = make_result(
        final_decision="hedged",
        premise_check=make_premise(status="contradicted"),
        grounding_check=None,
        retry_attempted=False,
        retry_grounding_check=None,
        llm_calls_made=0,
    )
    line = _debug_line(result)
    assert "grounding=n/a" in line


def test_retry_attempted_but_no_retry_grounding_check_falls_back_to_initial():
    """Defensive fallback: if retry_attempted is True but
    retry_grounding_check is somehow None, fall back to the initial
    grounding_check's status rather than crashing or showing n/a."""
    result = make_result(
        final_decision="regenerated",
        grounding_check=make_grounding("fail"),
        retry_attempted=True,
        retry_grounding_check=None,
        llm_calls_made=3,
    )
    line = _debug_line(result)
    assert "grounding=fail" in line


# ---------------------------------------------------------------------------
# _render_sources() -- Problem 2, source traceability display
# ---------------------------------------------------------------------------


def test_render_sources_returns_none_when_no_sources():
    result = make_result(sources=[])
    assert _render_sources(result) is None


def test_render_sources_shows_citation_and_evidence_for_each_claim():
    result = make_result(
        sources=[
            SourceCitation(
                claim_id=0,
                claim_text="Service Level declined from 94.3% to 92.9%",
                citation_ids=["Sources (22)"],
                evidence_text="Service Level for Gujarat in April 2026 was 92.9%.",
            ),
        ]
    )
    block = _render_sources(result)
    assert block is not None
    assert "Sources:" in block
    assert "Sources (22)" in block
    assert "Service Level declined from 94.3% to 92.9%" in block
    assert "Service Level for Gujarat in April 2026 was 92.9%." in block


def test_render_sources_lists_each_claim_separately_even_with_shared_citation():
    """Two claims citing the same source must each get their own numbered
    entry -- 'Claim 1 -> Source X' and 'Claim 2 -> Source X', per Problem
    2's brief, not collapsed into one."""
    result = make_result(
        sources=[
            SourceCitation(claim_id=0, claim_text="Claim A", citation_ids=["Sources (5)"], evidence_text="Evidence A."),
            SourceCitation(claim_id=1, claim_text="Claim B", citation_ids=["Sources (5)"], evidence_text="Evidence A."),
        ]
    )
    block = _render_sources(result)
    assert block.count("Claim A") + block.count("Claim B") == 2
    assert "[1]" in block and "[2]" in block


def test_render_sources_truncates_very_long_evidence_text():
    long_text = "X" * 1000
    result = make_result(
        sources=[SourceCitation(claim_id=0, claim_text="Claim", citation_ids=["Sources (1)"], evidence_text=long_text)]
    )
    block = _render_sources(result)
    assert "..." in block
    assert len(block) < len(long_text)
