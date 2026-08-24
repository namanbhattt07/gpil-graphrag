"""
Tests for the Phase 7 deterministic premise checker
(src/inference/premise_check.py).

These are pure unit tests -- no GraphRAG, no index, no API calls. They
build small synthetic `context_records["sources"]` DataFrames that mimic
exactly the shape GraphRAG's local search context builder returns (one row
per retrieved text unit, with a "State: X\\nPeriod: Y" header and
fixed-template metric sentences, matching src/graph/build_documents.py's
output), and check that check_premise() reaches the right verdict.

The false-premise case here (test_no_baseline_is_insufficient_data,
test_no_baseline_surfaces_supplementary_comparisons) is a synthetic replay
of the real "Why did Bihar's performance decline in October 2025?" finding
from the live retrieval test: Bihar's mini-pilot index has no September
2025 document, so a decline-into-October claim can't be evaluated against
its real prior period -- only Productivity/Service Level for October vs.
November (the wrong direction) are actually indexed.
"""

import pandas as pd
import pytest

from src.inference.premise_check import (
    check_premise,
    extract_direction_claim,
    extract_metric_claim,
    extract_period_from_question,
    extract_state_from_question,
    extract_superlative_ranking_claim,
    _previous_period,
)


def make_source_doc(state: str, period: str, metrics: dict) -> str:
    """Build one document's text in the same fixed template
    build_documents.py writes -- 'State: X\\nPeriod: Y\\n\\n<Metric> for X
    in Y was <value>.' repeated per metric."""
    sentences = " ".join(f"{name} for {state} in {period} was {value}." for name, value in metrics.items())
    return f"State: {state}\nPeriod: {period}\n\n{sentences}\n"


def make_sources_df(docs: list[tuple[str, str, dict]]) -> pd.DataFrame:
    rows = [
        {"id": str(i), "text": make_source_doc(state, period, metrics)}
        for i, (state, period, metrics) in enumerate(docs)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# extract_direction_claim / extract_metric_claim / extract_period_from_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Why did Bihar's performance decline in October 2025?", "decline"),
        ("Why did Bihar's performance drop in October 2025?", "decline"),
        ("Why did Bihar's performance improve in October 2025?", "improve"),
        ("Why did Bihar's performance grow in October 2025?", "improve"),
        ("What was Bihar's performance in October 2025?", "neutral"),
        ("Did Bihar improve or decline in October 2025?", "neutral"),  # both present -> ambiguous
    ],
)
def test_extract_direction_claim(question, expected):
    assert extract_direction_claim(question) == expected


def test_extract_metric_claim_finds_named_metric():
    assert extract_metric_claim("Why did Bihar's Service Level decline in October 2025?") == "Service Level"


def test_extract_metric_claim_none_for_generic_performance():
    assert extract_metric_claim("Why did Bihar's performance decline in October 2025?") is None


def test_extract_period_from_question():
    assert extract_period_from_question("Why did Bihar decline in October 2025?") == "October 2025"
    assert extract_period_from_question("Why did Bihar decline?") is None


def test_previous_period_normal():
    assert _previous_period("October 2025") == "September 2025"


def test_previous_period_year_rollover():
    assert _previous_period("January 2026") == "December 2025"


def test_previous_period_unparseable_returns_none():
    assert _previous_period("not a period") is None


# ---------------------------------------------------------------------------
# extract_superlative_ranking_claim / check_premise's "undefined_metric" gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which state was the least performing in 2025 and why?", "least performing"),
        ("Which state was the worst performing state overall?", "worst performing"),
        ("Which state had the lowest performing distribution network?", "lowest performing"),
        ("Which distributor was the best performing this year?", "best performing"),
        ("Which state was the highest performing in 2025?", "highest performing"),
        ("LEAST PERFORMING state in 2025?", "least performing"),  # case-insensitive
    ],
)
def test_extract_superlative_ranking_claim_matches(question, expected):
    assert extract_superlative_ranking_claim(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What was Bihar's performance in October 2025?",
        "Why did Bihar's performance decline in October 2025?",
        # Names an actual metric instead of the bare superlative -- must
        # NOT match, this is a different (answerable) question shape.
        "Which state had the highest Productivity in March 2025?",
        "Which distributor performed best?",  # "performed", not "performing"
    ],
)
def test_extract_superlative_ranking_claim_none_for_non_superlative_questions(question):
    assert extract_superlative_ranking_claim(question) is None


def test_superlative_ranking_question_is_undefined_metric_and_skips_evidence_parsing():
    """The exact failing question from the live investigation: must fail
    closed as undefined_metric WITHOUT needing (or looking at) any
    retrieved evidence at all -- the gate fires before source parsing."""
    result = check_premise("which state was the least performing in 2025 and why?", {})
    assert result.status == "undefined_metric"
    assert result.method == "deterministic"
    assert result.target_state is None
    assert result.explanation  # non-empty, human-readable


def test_superlative_ranking_question_ignores_retrieved_evidence_even_when_present():
    """Even when sources ARE retrieved (so the LLM would otherwise have
    something to work with), the superlative gate still fires first --
    the project's lack of a defined ranking metric doesn't depend on what
    got retrieved for this particular query."""
    sources = make_sources_df([("Bihar", "November 2025", {"Productivity": 89.0})])
    result = check_premise("Which state was the worst performing in 2025?", {"sources": sources})
    assert result.status == "undefined_metric"


def test_superlative_gate_does_not_shadow_existing_direction_claim_behavior():
    """Regression: ordinary decline/improve questions (no superlative
    phrase) must reach check_premise()'s existing logic completely
    unchanged."""
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.status == "supported"


# ---------------------------------------------------------------------------
# check_premise -- end to end over synthetic context_records
# ---------------------------------------------------------------------------


def test_no_claim_for_descriptive_question():
    """A plain 'what was X' question has no directional claim to verify --
    should proceed straight through with status='no_claim', no comparison
    work needed at all."""
    sources = make_sources_df([("Bihar", "October 2025", {"Productivity": 85.5})])
    result = check_premise("What was Bihar's performance in October 2025?", {"sources": sources})
    assert result.status == "no_claim"


def test_no_baseline_is_insufficient_data():
    """Replays the real finding: Bihar's index has October and November
    2025 documents but no September -- a 'decline in October' claim has no
    real prior-period baseline to check against."""
    sources = make_sources_df(
        [
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
            ("Bihar", "November 2025", {"Productivity": 89.0, "Service Level": 92.9}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.status == "insufficient_data"
    assert result.target_state == "Bihar"
    assert result.target_period == "October 2025"
    assert result.baseline_period == "September 2025"


def test_no_baseline_surfaces_supplementary_comparisons_in_correct_chronological_order():
    """Even though November isn't the real baseline, it's still useful
    context -- and October must be period_a / November period_b (NOT the
    reverse), since chronological order matters and 'November' < 'October'
    lexicographically would get this backwards if compared as strings."""
    sources = make_sources_df(
        [
            ("Bihar", "October 2025", {"Productivity": 85.5}),
            ("Bihar", "November 2025", {"Productivity": 89.0}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert len(result.supplementary_comparisons) == 1
    comp = result.supplementary_comparisons[0]
    assert comp.period_a == "October 2025"
    assert comp.value_a == 85.5
    assert comp.period_b == "November 2025"
    assert comp.value_b == 89.0
    assert comp.direction == "up"


def test_true_decline_is_supported_when_real_baseline_shows_it():
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0, "Service Level": 96.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.status == "supported"
    assert {c.direction for c in result.metric_comparisons} == {"down"}


def test_contradicted_when_baseline_shows_improvement_not_decline():
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 80.0, "Service Level": 90.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.status == "contradicted"


def test_mixed_signal_across_headline_metrics_is_unsupported():
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 90.0, "Service Level": 90.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.status == "unsupported"


def test_named_metric_narrows_check_to_just_that_metric():
    """Question naming 'Service Level' explicitly should only compare that
    metric, not the full headline pair -- Productivity's mixed direction
    shouldn't matter here."""
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 80.0, "Service Level": 96.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did Bihar's Service Level decline in October 2025?", {"sources": sources})
    assert result.status == "supported"
    assert [c.metric for c in result.metric_comparisons] == ["Service Level"]


def test_empty_sources_is_insufficient_data():
    result = check_premise(
        "Why did Bihar's performance decline in October 2025?", {"sources": pd.DataFrame(columns=["id", "text"])}
    )
    assert result.status == "insufficient_data"


def test_explicit_period_in_question_is_trusted_over_top_retrieved_doc():
    """If the top-ranked retrieved doc happens to be a different period
    than the one the question names, the question's own stated period
    wins."""
    sources = make_sources_df(
        [
            ("Bihar", "November 2025", {"Productivity": 89.0}),  # ranked first
            ("Bihar", "October 2025", {"Productivity": 85.5}),
            ("Bihar", "September 2025", {"Productivity": 92.0}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.target_period == "October 2025"
    assert result.baseline_period == "September 2025"
    assert result.status == "supported"


# ---------------------------------------------------------------------------
# Audit fix A2: check_premise() must not silently substitute a DIFFERENT
# state's evidence for the state the question explicitly asks about.
# extract_state_from_question() is the state-side equivalent of the
# period-side trust already proven above (test_explicit_period_in_question_
# is_trusted_over_top_retrieved_doc).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Why did Bihar's performance decline in October 2025?", "Bihar"),
        ("Why did Andhra Pradesh's performance decline in October 2025?", "Andhra Pradesh"),
        ("Why did Uttar Pradesh's performance decline in October 2025?", "Uttar Pradesh"),
        ("Why did Uttarakhand's performance decline in October 2025?", "Uttarakhand"),
        ("What was performance in October 2025?", None),
    ],
)
def test_extract_state_from_question(question, expected):
    assert extract_state_from_question(question) == expected


def test_question_named_state_not_retrieved_is_insufficient_data_not_wrong_state():
    """1: the question explicitly names Bihar, but every retrieved document
    is actually about Gujarat (e.g. an imperfect retrieval ranking) --
    check_premise() must NOT silently validate the claim against Gujarat's
    evidence; it must report insufficient_data for Bihar instead."""
    sources = make_sources_df(
        [
            ("Gujarat", "October 2025", {"Productivity": 90.0, "Service Level": 95.0}),
            ("Gujarat", "September 2025", {"Productivity": 91.0, "Service Level": 96.0}),
        ]
    )
    result = check_premise("Why did Bihar performance decline in October 2025?", {"sources": sources})
    assert result.status == "insufficient_data"
    assert result.target_state == "Bihar"
    assert result.status != "supported"


def test_question_named_state_with_matching_evidence_is_validated_normally():
    """2: when the question names a state AND that state's evidence IS
    retrieved (even alongside other states, or not ranked first), the
    existing supported/contradicted/unsupported logic proceeds exactly as
    before, scoped correctly to the named state."""
    sources = make_sources_df(
        [
            ("Gujarat", "October 2025", {"Productivity": 99.0, "Service Level": 99.0}),  # ranked first, wrong state
            ("Bihar", "September 2025", {"Productivity": 92.0, "Service Level": 96.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did Bihar performance decline in October 2025?", {"sources": sources})
    assert result.status == "supported"
    assert result.target_state == "Bihar"
    assert result.target_period == "October 2025"
    assert result.baseline_period == "September 2025"
    assert {c.direction for c in result.metric_comparisons} == {"down"}


def test_no_state_named_falls_back_to_top_retrieved_doc_unchanged():
    """3: when the question does NOT explicitly name a state, behavior is
    unchanged from before this fix -- the top-ranked retrieved document's
    state is trusted, exactly like extract_period_from_question() already
    does for period when the question doesn't state one."""
    sources = make_sources_df(
        [
            ("Gujarat", "September 2025", {"Productivity": 92.0, "Service Level": 96.0}),
            ("Gujarat", "October 2025", {"Productivity": 85.5, "Service Level": 94.3}),
        ]
    )
    result = check_premise("Why did performance decline in October 2025?", {"sources": sources})
    assert result.status == "supported"
    assert result.target_state == "Gujarat"


def test_explicit_period_behavior_unchanged_when_state_also_named():
    """4: the question's own stated period still wins over the top-ranked
    retrieved doc's period, exactly as before, even now that the state is
    ALSO being validated explicitly -- proves the period-trust logic
    wasn't altered by the state-trust addition."""
    sources = make_sources_df(
        [
            ("Bihar", "November 2025", {"Productivity": 89.0}),  # ranked first
            ("Bihar", "October 2025", {"Productivity": 85.5}),
            ("Bihar", "September 2025", {"Productivity": 92.0}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.target_state == "Bihar"
    assert result.target_period == "October 2025"
    assert result.baseline_period == "September 2025"
    assert result.status == "supported"
    assert result.status == "supported"


# ---------------------------------------------------------------------------
# Gamble-Wright multi-period / explicit-baseline fix (known issue 3):
# "Gamble-Wright Distributors in Manipur had a Dropsize deviation and a GPI
# Out-of-Stock deviation in February 2025. By August 2025, did those two
# issues persist...?" was incorrectly rejected as insufficient_data,
# because (a) extract_direction_claim() misread the idiom "pick up" (as in
# "what new deviations did Gamble-Wright pick up instead") as the
# direction word "up" -> a false "improve" claim, and (b) check_premise()
# only ever took the FIRST-mentioned period as the target and always
# computed baseline as "previous calendar month", ignoring the question's
# own explicitly-supplied February 2025 baseline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What new deviations did Gamble-Wright pick up instead?", "neutral"),
        ("What new issues did the distributor pick up this month?", "neutral"),
        ("What deviations did Gamble-Wright pick up in February 2025?", "neutral"),
        # The bare word "up" on its own (not part of "pick up") must still work.
        ("Was Service Level up in October 2025?", "improve"),
        ("Did performance decline in October 2025?", "decline"),
    ],
)
def test_pick_up_idiom_does_not_trigger_false_improve_claim(question, expected):
    assert extract_direction_claim(question) == expected


def test_gamble_wright_question_is_no_claim_and_never_gated():
    """The exact known-issue question: with the 'pick up' idiom no longer
    misread as a direction word, this question asserts no decline/improve
    premise at all and must proceed straight to no_claim (generation is
    never blocked), regardless of what evidence is or isn't retrieved."""
    question = (
        "Gamble-Wright Distributors in Manipur had a Dropsize deviation and a GPI Out-of-Stock "
        "deviation in February 2025. By August 2025, did those two issues persist, and what new "
        "deviations did Gamble-Wright pick up instead?"
    )
    result = check_premise(question, {"sources": pd.DataFrame(columns=["id", "text"])})
    assert result.status == "no_claim"


def test_multi_period_question_uses_explicit_baseline_not_previous_calendar_month():
    """A genuinely directional two-period question ('...from February 2025
    ... improve by August 2025?') must use February 2025 -- the question's
    OWN explicitly-named earlier period -- as the baseline, not the
    previous-calendar-month computation (January 2025, which isn't even
    indexed here)."""
    sources = make_sources_df(
        [
            ("Manipur", "February 2025", {"Dropsize": 174.14}),
            ("Manipur", "August 2025", {"Dropsize": 178.77}),
            # January 2025 (the WRONG, previous-calendar-month baseline) is
            # deliberately NOT indexed, to prove it's never even looked up.
        ]
    )
    question = "Did the Dropsize deviation in Manipur from February 2025 improve by August 2025?"
    result = check_premise(question, {"sources": sources})
    assert result.status == "supported"
    assert result.target_period == "August 2025"
    assert result.baseline_period == "February 2025"
    assert result.metric_comparisons[0].direction == "up"


def test_multi_period_question_target_named_first_still_resolves_chronologically():
    """Calendar order decides baseline/target, not text order -- a
    (deliberately unusual) phrasing naming the LATER period first must
    still treat the earlier one as baseline."""
    sources = make_sources_df(
        [
            ("Manipur", "February 2025", {"Dropsize": 174.14}),
            ("Manipur", "August 2025", {"Dropsize": 178.77}),
        ]
    )
    question = "Did Manipur's Dropsize, which was 178.77 by August 2025, improve from February 2025?"
    result = check_premise(question, {"sources": sources})
    assert result.baseline_period == "February 2025"
    assert result.target_period == "August 2025"


def test_single_period_question_baseline_behavior_completely_unchanged():
    """Regression guard: a single-period question must still compute its
    baseline as the previous calendar month, exactly as before this fix --
    the multi-period branch must never engage for a single-period
    question."""
    sources = make_sources_df(
        [
            ("Bihar", "September 2025", {"Productivity": 92.0}),
            ("Bihar", "October 2025", {"Productivity": 85.5}),
        ]
    )
    result = check_premise("Why did Bihar's performance decline in October 2025?", {"sources": sources})
    assert result.target_period == "October 2025"
    assert result.baseline_period == "September 2025"
    assert result.status == "supported"


def test_extract_all_periods_from_question():
    from src.inference.premise_check import extract_all_periods_from_question

    assert extract_all_periods_from_question("Why did Bihar decline in October 2025?") == ["October 2025"]
    assert extract_all_periods_from_question("Why did Bihar decline?") == []
    assert extract_all_periods_from_question(
        "Gamble-Wright had a deviation in February 2025. By August 2025, did it persist?"
    ) == ["February 2025", "August 2025"]
    # Repeated mention of the same period is deduplicated, not treated as
    # a second distinct period.
    assert extract_all_periods_from_question(
        "Was October 2025 better or worse than October 2025?"
    ) == ["October 2025"]


def test_extract_all_periods_from_question_shared_year_range():
    """15-query SKU validation pass (2026-08-21): a two-month range with
    the year stated only once, after the second month ('January and
    February 2026' / 'January to February 2026' / 'January through
    February 2026'), must resolve BOTH months -- not silently drop the
    first one, which previously made a whole indexed month look
    unavailable to callers (evaluate_sku_evidence, check_premise)."""
    from src.inference.premise_check import extract_all_periods_from_question

    assert extract_all_periods_from_question(
        "Compare the top-selling SKU in Punjab between January and February 2026."
    ) == ["January 2026", "February 2026"]
    assert extract_all_periods_from_question(
        "SKU performance from March to May 2025."
    ) == ["March 2025", "May 2025"]
    assert extract_all_periods_from_question(
        "Service Level January through March 2025."
    ) == ["January 2025", "March 2025"]
    # Already-explicit-both-years phrasing must be completely unaffected.
    assert extract_all_periods_from_question(
        "Compare February 2025 to August 2025."
    ) == ["February 2025", "August 2025"]
