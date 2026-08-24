"""
Tests for the Phase 7 CLAIM-LEVEL grounding/contradiction checker
(src/inference/grounding_check.py).

Pure unit tests -- no GraphRAG, no index, no API calls.

Test-to-requirement map (see the claim-level grounding spec this
implements):
   1: test_correct_trend_claim_passes
   2: test_wrong_direction_fails
   3: test_wrong_delta_fails
   4: test_correct_entity_wrong_period_fails
   5: test_unsupported_alarming_fails                 (Example B)
   6: test_concern_variants_all_caught
   7: test_inefficiencies_from_unrelated_context_fails (Example E)
   8: test_explicit_source_supported_qualitative_term_passes
   9: test_explicit_deviation_claim_passes             (Example C)
  10: test_unsupported_causal_claim_fails              (Example D)
  11: test_explicitly_supported_causal_claim_passes
  12: test_cross_month_cross_entity_evidence_cannot_validate_claim
  13: test_citation_specific_evidence_used_not_global_context
  14/15/16/17: covered in test_inference_pipeline.py (retry orchestration)

Plus: the original prose-fallback scanners (still the secondary safety
net) and the real-world "concerning" replay, updated for the new
check_grounding(answer_text, context_records, claims, metric_comparisons)
signature.
"""

import pandas as pd
import pytest

from src.inference.grounding_check import (
    check_direction_consistency,
    check_grounding,
    resolve_claim_sources,
    scan_causal_language,
    scan_definition_language,
    scan_entity_numeric_claims,
    scan_glossary_term_misuse,
    scan_qualitative_language,
    scan_recommendation_language,
    validate_claim,
    verify_ranking_claim,
    verify_sku_ranking_claim,
    verify_thematic_completeness,
    _build_evidence_index,
    _check_entity_value_against_atomic_facts,
    _check_gap_against_atomic_fact,
    _decompose_state_category_entity,
    _detect_entity_fusion,
    _distributor_names_from_sources,
    _ensure_sentence_boundary,
    _extract_focus_categories,
    _is_future_or_prescriptive,
    _known_distributor_core_names,
    _known_distributor_entity_names,
    _period_sort_key,
    _resolve_citations,
    _resolve_entity_reference,
    _sentence_categories,
    _split_sentences,
    _trend_delta,
)
from src.inference.fact_structuring import AtomicFact, extract_atomic_facts
from src.inference.knowledge_layer import build_glossary_source_rows
from src.inference.schemas import AnswerClaim, MetricComparison


def make_context_records(entities=None, sources=None, reports=None, relationships=None) -> dict:
    def _df(rows, cols):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    return {
        "entities": _df(entities, ["id", "entity", "description"]),
        "sources": _df(sources, ["id", "text"]),
        "reports": _df(reports, ["id", "title", "content"]),
        "relationships": _df(relationships, ["id", "source", "target", "description"]),
        "claims": pd.DataFrame(columns=["id", "description"]),
    }


# A realistic pair of Bihar Service Level source documents, matching the
# real Phase 4 template and the real ids seen in live retrieval (0=Oct,
# 3=Nov).
BIHAR_OCT_NOV_SOURCES = [
    {"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nService Level for Bihar in October 2025 was 94.3%. Productivity for Bihar in October 2025 was 85.5%."},
    {"id": "3", "text": "State: Bihar\nPeriod: November 2025\n\nService Level for Bihar in November 2025 was 92.9%. Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on Service Level in November 2025: 82.6% vs. the state average of 92.9%, a gap of 10.4 percentage points."},
]


# ---------------------------------------------------------------------------
# 1-4: numeric / trend / delta / period claim-level checks
# ---------------------------------------------------------------------------


def test_correct_trend_claim_passes():
    """1: Example A -- Service Level 94.3 -> 92.9, claimed as 'declined by
    1.4pp', citing both real source docs -- should PASS."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined by 1.4 percentage points from October to November 2025.",
        claim_type="trend",
        entity="Bihar",
        metric="Service Level",
        period="November 2025",
        comparison_period="October 2025",
        value=92.9,
        comparison_value=94.3,
        direction="declined",
        delta=1.4,
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_wrong_direction_fails():
    """2: same real numbers, but claim says 'improved' -- direction
    contradicts the cited values -> FAIL."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level improved from October to November 2025.",
        metric="Service Level", period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3, direction="improved",
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "direction_contradiction" for i in issues)


def test_period_sort_key_parses_month_year():
    assert _period_sort_key("October 2025") == (2025, 10)
    assert _period_sort_key("not a period") is None


def test_trend_delta_canonical_order_unchanged():
    """value=later/comparison_value=earlier (the schema example's own
    convention) -- delta must be exactly value - comparison_value, same
    as before this fix existed."""
    claim = AnswerClaim(period="November 2025", comparison_period="October 2025", value=92.9, comparison_value=94.3)
    assert _trend_delta(claim) == pytest.approx(92.9 - 94.3)


def test_trend_delta_swapped_order_reflects_real_chronological_movement():
    """value=earlier/comparison_value=later -- delta must reflect the REAL
    later-minus-earlier movement (92.9 - 94.3, a decline), not the raw
    value-minus-comparison_value (94.3 - 92.9, which would misread as a
    rise) the old fixed-role assumption computed."""
    claim = AnswerClaim(period="October 2025", comparison_period="November 2025", value=94.3, comparison_value=92.9)
    assert _trend_delta(claim) == pytest.approx(92.9 - 94.3)


def test_trend_delta_falls_back_when_periods_missing_or_unparseable():
    no_periods = AnswerClaim(value=94.3, comparison_value=92.9)
    assert _trend_delta(no_periods) == pytest.approx(94.3 - 92.9)
    unparseable = AnswerClaim(period="Q1", comparison_period="Q2", value=94.3, comparison_value=92.9)
    assert _trend_delta(unparseable) == pytest.approx(94.3 - 92.9)


def test_trend_delta_falls_back_when_comparison_entity_set():
    cross_entity = AnswerClaim(
        comparison_entity="Maharashtra", period="April 2025", comparison_period="August 2024",
        value=6.6, comparison_value=6.5,
    )
    assert _trend_delta(cross_entity) == pytest.approx(6.6 - 6.5)


def test_chronologically_ordered_trend_claim_passes_even_when_period_is_earlier():
    """2026-08-23 stabilization pass, live-caught bug: a real multi-period
    SKU comparison question ('...between April 2026 and June 2026') had
    the model naturally set period=the EARLIER month (with its value) and
    comparison_period=the LATER month (with its comparison_value) --
    chronological order, not the schema example's 'value=later,
    comparison_value=baseline/earlier' convention. This is an accurate,
    correctly-cited 'declined' claim (94.3 -> 92.9 IS a real decline) and
    must PASS -- before the fix, the fixed value-minus-comparison_value
    assumption read this ordering backwards and always failed it."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend",
        entity="Bihar",
        metric="Service Level",
        period="October 2025",  # the EARLIER period, paired with `value`
        comparison_period="November 2025",  # the LATER period, paired with `comparison_value`
        value=94.3,
        comparison_value=92.9,
        direction="declined",
        delta=1.4,
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_chronologically_ordered_trend_claim_still_catches_a_real_wrong_direction():
    """Regression guard for the fix above: swapping period/comparison_period
    order must not blanket-disable the direction check -- a genuinely
    WRONG direction claim in the earlier-period-first ordering must still
    fail."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level improved from 94.3% in October 2025 to 92.9% in November 2025.",
        period="October 2025",
        comparison_period="November 2025",
        value=94.3,
        comparison_value=92.9,
        direction="improved",  # wrong: 94.3 -> 92.9 is a decline, not an improvement
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "direction_contradiction" for i in issues)


def test_chronological_swap_not_applied_when_comparison_entity_is_set():
    """A cross-entity comparison (comparison_entity set) has no 'later
    period' concept to infer from -- the chronological-order fix must only
    ever apply to same-entity trend claims, never silently change
    behavior for a two-different-entities comparison claim."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Sikkim's rate (6.6%) in April 2025 was higher than Maharashtra's (6.5%) in August 2024.",
        claim_type="comparison",
        entity="Sikkim",
        comparison_entity="Maharashtra",
        period="April 2025",
        comparison_period="August 2024",  # chronologically EARLIER than period, but irrelevant here
        value=6.6,
        comparison_value=6.5,
        direction="higher",
        citations=[],
    )
    # direction="higher" maps to "up" polarity; value(6.6) - comparison_value(6.5) = +0.1 = "up" -- consistent
    # either way for this claim, so this only proves the swap path isn't silently taken for comparison_entity
    # claims (verified by not raising/erroring on a comparison-shaped claim with period-order the swap would
    # otherwise touch); the real regression coverage for comparison_entity claims lives elsewhere in this file.
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type == "direction_contradiction" for i in issues)


def test_claim_text_direction_contradiction_detected_even_without_direction_field():
    """A live cross-entity comparison failure: claim_text itself used a
    direction word ('lower') that contradicts the claim's own value/
    comparison_value arithmetic, even though the structured `direction`
    field was left unset (so the OLDER claim.direction-only check has
    nothing to check) -- e.g. 'Sikkim's rate was significantly lower at
    6.6%...' when 6.6 is actually higher than the 6.5 it's compared to.
    Must FAIL via the claim_text scan specifically."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Sikkim's Ferrero Out-of-Stock Rate was significantly lower at 6.6% compared to Maharashtra's 6.5%.",
        claim_type="comparison",
        value=6.6,
        comparison_value=6.5,
        citations=[],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "direction_contradiction" and i.term == "down" for i in issues)


def test_claim_text_with_both_direction_words_is_not_falsely_flagged():
    """A well-phrased two-sided comparison ('X is lower, Y is higher') uses
    BOTH an up and a down word in the same claim_text -- this is genuinely
    ambiguous prose, not a contradiction, and must NOT be flagged (mirrors
    _infer_direction_word()'s existing conservative mixed-word handling)."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Maharashtra's rate was lower at 6.5%, while Sikkim's was higher at 6.6%.",
        claim_type="comparison",
        value=6.6,
        comparison_value=6.5,
        citations=[],
    )
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type == "direction_contradiction" for i in issues)


def test_wrong_delta_fails():
    """3: correct direction, but claimed delta (5.0) doesn't match the
    actual 1.4 -> FAIL."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined by 5.0 percentage points.",
        metric="Service Level", period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3, direction="declined", delta=5.0,
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_delta" for i in issues)


def test_correct_entity_wrong_period_fails():
    """4: entity is right (Bihar, and the cited doc IS about Bihar), but
    the claimed period isn't mentioned in the cited evidence -> FAIL."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bihar's Service Level in December 2025 was 92.9%.",
        entity="Bihar", metric="Service Level", period="December 2025", value=92.9,
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_period" for i in issues)


# ---------------------------------------------------------------------------
# 5-9: qualitative / deviation claim-level checks
# ---------------------------------------------------------------------------


def test_unsupported_alarming_fails():
    """5: Example B -- Ferrero OOS = 13.0%, claim calls it 'alarming', no
    such characterization in evidence -> FAIL."""
    sources = [{"id": "3", "text": "State: Bihar\nPeriod: November 2025\n\nFor the Ferrero category in Bihar during November 2025: Out-of-Stock rate was 13.0%."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Ferrero's out-of-stock rate was an alarming 13.0%.",
        claim_type="qualitative", metric="Out-of-Stock Rate", value=13.0,
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_qualifier" and i.term == "alarming" for i in issues)


def test_concern_variants_all_caught():
    """6: 'concern'/'concerns'/'concerning' (and 'concerned') must ALL be
    caught -- the original word list only had 'concerning'/'concerned'."""
    sources = [{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nService Level for Bihar in October 2025 was 94.3%."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    for word in ["concern", "concerns", "concerning", "concerned"]:
        claim = AnswerClaim(
            claim_text=f"This is a {word} for Bihar's Service Level.",
            metric="Service Level", citations=["Sources (0)"],
        )
        issues = validate_claim(claim, evidence_index)
        assert any(i.issue_type == "unsupported_qualifier" and i.term == word for i in issues), f"'{word}' was not caught"


def test_inefficiencies_from_unrelated_context_fails():
    """7: Example E -- the CITED report contains the word 'inefficiencies',
    but only in a passage about a different month's candy dynamics, never
    mentioning THIS claim's metric (Service Level) at all. Word presence
    alone must not be enough -- FAIL."""
    reports = [{
        "id": "9", "title": "Bihar Candy Supply Dynamics",
        "content": (
            "Concerns surrounding inventory management: Distributors in Bihar, including "
            "Weaver-Sherman, reported deviations in dropsize metrics, suggesting inefficiencies "
            "in managing candy inventory for November 2025."
        ),
    }]
    records = make_context_records(reports=reports)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Supply chain inefficiencies affected Bihar's Service Level.",
        claim_type="qualitative", entity="Bihar", metric="Service Level",
        citations=["Reports (9)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_qualifier" and i.term == "inefficiencies" for i in issues)


def test_explicit_source_supported_qualitative_term_passes():
    """8: the cited evidence itself uses the SAME word, about the SAME
    metric -- should PASS."""
    sources = [{"id": "3", "text": "State: Bihar\nPeriod: November 2025\n\nAn alarming Service Level deviation was recorded for one distributor."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level showed an alarming deviation.",
        metric="Service Level", citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type == "unsupported_qualifier" for i in issues)


def test_explicit_deviation_claim_passes():
    """9: Example C -- distributor Service Level 82.6% vs state average
    92.9%, source explicitly says 'significant deviation' -> PASS."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Zhang, Brooks and Miles Distributors showed a significant Service Level deviation.",
        claim_type="deviation", entity="Zhang, Brooks and Miles Distributors", metric="Service Level",
        value=82.6, comparison_value=92.9, citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


# ---------------------------------------------------------------------------
# 10-11: causal claims -- strictest category
# ---------------------------------------------------------------------------


def test_unsupported_causal_claim_fails():
    """10: Example D -- distributor Service Level = 82.6%, claim says it
    CAUSED the state decline. Evidence states the fact but never asserts
    causation -> FAIL."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="This distributor caused Bihar's Service Level decline.",
        claim_type="causal", entity="Zhang, Brooks and Miles Distributors",
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_causal" for i in issues)


def test_explicitly_supported_causal_claim_passes():
    """11: when the cited evidence DOES contain explicit causal language,
    a causal claim should pass the causal check."""
    sources = [{"id": "3", "text": "State: Bihar\nPeriod: November 2025\n\nThe stockout was due to a logistics delay that led to reduced Service Level."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="The logistics delay led to reduced Service Level.",
        claim_type="causal", citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type == "unsupported_causal" for i in issues)


# ---------------------------------------------------------------------------
# 12-13: evidence linkage -- citation-specific, not global blob
# ---------------------------------------------------------------------------


def test_cross_month_cross_entity_evidence_cannot_validate_claim():
    """12: a claim about Bihar November citing a report that's ACTUALLY
    about a different entity/period must not be validated by that
    citation just because some words happen to overlap."""
    reports = [{
        "id": "50", "title": "Kerala Candy Supply Dynamics, September 2025",
        "content": "Kerala's candy category showed a concerning out-of-stock pattern in September 2025.",
    }]
    records = make_context_records(reports=reports)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bihar's Service Level was concerning in November 2025.",
        entity="Bihar", metric="Service Level", period="November 2025",
        citations=["Reports (50)"],
    )
    issues = validate_claim(claim, evidence_index)
    # Entity, period, AND the qualitative term should all fail -- none of
    # them are actually established by a report about Kerala candy in September.
    issue_types = {i.issue_type for i in issues}
    assert "unsupported_entity" in issue_types
    assert "unsupported_period" in issue_types
    assert "unsupported_qualifier" in issue_types


def test_citation_specific_evidence_used_not_global_context():
    """13: the SAME evidence set contains a report elsewhere that WOULD
    support the qualitative term, but the claim doesn't cite it -- only
    the citation actually given should be checked, so this must still
    FAIL even though 'somewhere in context' the word is supported."""
    reports = [
        {"id": "9", "title": "Unrelated Candy Report", "content": "Candy dynamics only, unrelated distribution topic, nothing evaluative here."},
        {"id": "63", "title": "Bihar Economic Performance", "content": "Bihar's Service Level was flagged as a concerning metric this month."},
    ]
    records = make_context_records(reports=reports)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bihar's Service Level was concerning.",
        metric="Service Level",
        citations=["Reports (9)"],  # cites the WRONG report on purpose
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_qualifier" and i.term == "concerning" for i in issues)

    # Sanity check: citing the RIGHT report instead passes.
    claim_correct_citation = AnswerClaim(
        claim_text="Bihar's Service Level was concerning.",
        metric="Service Level",
        citations=["Reports (63)"],
    )
    issues_correct = validate_claim(claim_correct_citation, evidence_index)
    assert not any(i.issue_type == "unsupported_qualifier" for i in issues_correct)


def test_uncited_evaluative_claim_fails_closed_not_whole_blob_fallback():
    """An evaluative claim (qualitative/causal/deviation) with NO
    citations must fail closed, not silently pass because the whole
    retrieved blob happens to contain the word somewhere unrelated. This
    is what actually closes the Example E gap for claims the model
    reports without a citation (rather than a wrong one)."""
    # The word "concerning" and causal language legitimately exist
    # SOMEWHERE in this evidence set (about an unrelated report), but the
    # claim below cites nothing at all.
    reports = [{"id": "9", "title": "Bihar Candy Supply Dynamics", "content": "Concerns about candy inventory led to delays in an unrelated month."}]
    records = make_context_records(reports=reports)
    evidence_index = _build_evidence_index(records)

    qualitative_claim = AnswerClaim(claim_text="This was a concerning result.", metric="Service Level", citations=[])
    assert any(i.issue_type == "unsupported_qualifier" for i in validate_claim(qualitative_claim, evidence_index))

    causal_claim = AnswerClaim(claim_text="This led to the decline.", claim_type="causal", citations=[])
    assert any(i.issue_type == "unsupported_causal" for i in validate_claim(causal_claim, evidence_index))


def test_unresolvable_citation_fails_closed_instead_of_whole_blob_fallback():
    """Source-traceability fix: a claim that cites something which LOOKS
    like a citation but doesn't resolve to any known record (e.g. the
    model wrote "Atomic SKU Facts (4)" instead of a real "Sources (21)"
    id -- see answer.py's _ATOMIC_FACTS_CITATION_GUIDANCE) must fail
    closed with an "unresolved_citation" issue, NOT fall back to checking
    against the whole retrieved blob. Deliberately constructed so the
    claimed value DOES appear elsewhere in the whole blob -- if the old
    permissive whole-blob fallback were still active for this case, it
    would incorrectly let this claim pass."""
    sources = [{"id": "21", "text": "State: Punjab\nPeriod: March 2025\n\nSKU Alpha Pack 1 Revenue was Rs 95,267,583 in Punjab during March 2025."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)

    claim = AnswerClaim(
        claim_text="SKU Alpha Pack 1 had revenue of Rs 95,267,583.",
        metric="Revenue", value=95267583,
        citations=["Atomic SKU Facts (4)"],  # unresolvable -- not a real dataset name
    )
    issues = validate_claim(claim, evidence_index)
    assert len(issues) == 1
    assert issues[0].issue_type == "unresolved_citation"
    assert issues[0].severity == "error"
    assert "Atomic SKU Facts (4)" in issues[0].term

    # Sanity check: the SAME claim citing the REAL resolvable id passes cleanly.
    claim_correct = AnswerClaim(
        claim_text="SKU Alpha Pack 1 had revenue of Rs 95,267,583.",
        metric="Revenue", value=95267583,
        citations=["Sources (21)"],
    )
    assert validate_claim(claim_correct, evidence_index) == []


def make_sku_context_records(sku_source_id="sku-Punjab-February_2026"):
    """One real SKU-evidence source row (the shape sku_evidence.py's
    render_sku_evidence_document() actually produces), plus one unrelated
    numbered GraphRAG source, matching how pipeline.py's
    _augment_context_with_sku_evidence() appends SKU rows onto whatever
    GraphRAG itself retrieved."""
    sources = [
        {
            "id": sku_source_id,
            "text": (
                "SKU-level Sales & Distribution detail for Punjab, February 2026:\n"
                "SKU SKU0041 (Marlboro Pack 1, Marlboro franchise, IPM category) in Punjab during "
                "February 2026: 835,078 units delivered, revenue of Rs 227,400,090, Service Level "
                "90.0%, Numeric Distribution 68.2%, Out-of-Stock rate 12.5%.\n"
                "SKU SKU0016 (GPI_Franchise_3 Pack 1, GPI_Franchise_3 franchise, GPI category) in "
                "Punjab during February 2026: 304,315 units delivered, revenue of Rs 26,892,317, "
                "Service Level 90.3%, Numeric Distribution 26.9%, Out-of-Stock rate 12.5%."
            ),
        },
        {"id": "2", "text": "Some unrelated GraphRAG-retrieved entity/report text about a different topic."},
    ]
    return make_context_records(sources=sources)


# make_sku_context_records() renders 2 SKUs x 5 metrics = 10 SKU Atomic
# Facts, in fixed order (Units Delivered, Revenue, Service Level, Numeric
# Distribution, Out-of-Stock Rate per SKU, SKUs in the order they appear in
# the source text): fact 1 = Marlboro Pack 1's Units Delivered (835,078),
# fact 6 = GPI_Franchise_3 Pack 1's Units Delivered (304,315).
MARLBORO_UNITS_FACT_N = 1
GPI3_UNITS_FACT_N = 6


@pytest.mark.parametrize(
    "malformed_citation_template",
    ["Atomic SKU Facts ({n})", "SKU Facts ({n})", "SKU Fact ({n})", "SKU ({n})"],
)
def test_sku_atomic_fact_number_citation_resolves_to_real_sku_evidence(malformed_citation_template):
    """15-query SKU validation pass (2026-08-21): across every live SKU
    ranking/factual question, the model consistently cited the Atomic SKU
    Facts section's own FACT NUMBER instead of its 'Source:' field --
    four different spellings, all previously failing closed as
    unresolved and making every SKU answer un-groundable. This citation
    shape now resolves to the real, precisely-scoped evidence for that
    exact fact instead of being discarded."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    citation = malformed_citation_template.format(n=MARLBORO_UNITS_FACT_N)
    resolved = _resolve_citations([citation], evidence_index)
    assert "Marlboro Pack 1" in resolved
    assert "835,078" in resolved


def test_sku_atomic_fact_citation_resolves_to_only_that_skus_fact_not_a_different_sku():
    """THE core bug from the live validation pass: resolving 'Atomic SKU
    Facts (N)' to the WHOLE multi-SKU source blob (an earlier version of
    this fix) satisfied validate_claim()'s co-occurrence check but
    displayed a DIFFERENT SKU's sentence as the claim's "evidence" for
    source-traceability purposes, purely because that other SKU happened
    to render first in the shared blob. Resolving by exact fact number
    must return ONLY Marlboro Pack 1's own fact text -- never
    GPI_Franchise_3 Pack 1's, even though both live in the same source."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    resolved = _resolve_citations([f"Atomic SKU Facts ({MARLBORO_UNITS_FACT_N})"], evidence_index)
    assert "Marlboro Pack 1" in resolved
    assert "GPI_Franchise_3 Pack 1" not in resolved
    assert "304,315" not in resolved


def test_sku_atomic_fact_citation_claim_passes_validate_claim():
    """End-to-end: a claim citing the malformed 'Atomic SKU Facts (N)'
    form, whose entity/value/metric/period genuinely ARE stated together
    in the real SKU evidence, must now pass validate_claim() -- confirming
    the recovery resolves to evidence that still supports (or would
    correctly reject) the specific claim, not a permissive whole-blob."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    claim = AnswerClaim(
        claim_text="Marlboro Pack 1 had 835,078 units delivered in Punjab in February 2026.",
        entity="Marlboro Pack 1", metric="Units Delivered", period="February 2026",
        value=835078.0, citations=[f"Atomic SKU Facts ({MARLBORO_UNITS_FACT_N})"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_sku_atomic_fact_citation_still_rejects_wrong_value():
    """The recovery resolves to the SPECIFIC fact's own scope -- it does
    NOT make grounding permissive. A claim citing Marlboro Pack 1's own
    Units Delivered fact number but asserting a value that actually
    belongs to a DIFFERENT SKU (GPI_Franchise_3 Pack 1) must still fail,
    exactly the cross-SKU-misattribution check (E) this suite targets."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    claim = AnswerClaim(
        claim_text="Marlboro Pack 1 had 304,315 units delivered in Punjab in February 2026.",
        entity="Marlboro Pack 1", metric="Units Delivered", period="February 2026",
        value=304315.0,  # this is actually GPI_Franchise_3 Pack 1's figure, not Marlboro's
        citations=[f"Atomic SKU Facts ({MARLBORO_UNITS_FACT_N})"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


def test_sku_atomic_fact_citation_with_multiple_fact_numbers_for_same_sku():
    """The model's real citations name several fact numbers together, e.g.
    'Atomic SKU Facts (1, 2, 3, 4, 5)' -- the five metrics for one SKU.
    All five must resolve, all scoped to Marlboro Pack 1 alone."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    resolved = _resolve_citations(["Atomic SKU Facts (1, 2, 3, 4, 5)"], evidence_index)
    assert "Marlboro Pack 1" in resolved
    assert "GPI_Franchise_3 Pack 1" not in resolved


# ---------------------------------------------------------------------------
# Bare synthetic-row-id citation fallback (2026-08-23, SKU live validation
# pass): a live "bottom 3 SKUs by Revenue" question had the model write the
# bare id "sku-Gujarat-April_2026" as one of its structured claim's
# `citations` entries -- no "Sources (...)" wrapper at all -- which
# previously failed closed as "unresolved_citation" even though the id
# names a real, known Sources row. _resolve_citations() now falls back to
# an EXACT id lookup against the real "sources" table when a citation
# string has no "Name (ids)" shape to parse at all.
# ---------------------------------------------------------------------------


def test_bare_synthetic_row_id_citation_resolves_via_fallback():
    evidence_index = _build_evidence_index(
        make_context_records(sources=[{"id": "sku-Gujarat-April_2026", "text": "SKU-level Sales & Distribution detail for Gujarat, April 2026: some real evidence text."}])
    )
    resolved = _resolve_citations(["sku-Gujarat-April_2026"], evidence_index)
    assert "some real evidence text" in resolved


def test_bare_id_fallback_does_not_match_a_nonexistent_id():
    evidence_index = _build_evidence_index(
        make_context_records(sources=[{"id": "sku-Gujarat-April_2026", "text": "real evidence text"}])
    )
    resolved = _resolve_citations(["sku-Gujarat-June_2026"], evidence_index)
    assert resolved == ""


def test_bare_id_fallback_coexists_with_normal_wrapped_citations_in_the_same_list():
    evidence_index = _build_evidence_index(
        make_context_records(sources=[
            {"id": "0", "text": "state-level evidence text"},
            {"id": "sku-Gujarat-April_2026", "text": "sku-level evidence text"},
        ])
    )
    resolved = _resolve_citations(["Sources (0)", "sku-Gujarat-April_2026"], evidence_index)
    assert "state-level evidence text" in resolved
    assert "sku-level evidence text" in resolved


def test_bare_id_fallback_lets_a_real_claim_pass_validate_claim():
    """End-to-end: the exact live failure shape -- a structured claim
    citing the bare id, with an entity/value/period that genuinely IS
    stated together in the real evidence -- must now pass."""
    evidence_index = _build_evidence_index(
        make_context_records(sources=[{
            "id": "sku-Gujarat-April_2026",
            "text": (
                "SKU-level Sales & Distribution detail for Gujarat, April 2026:\n"
                "SKU SKU0053 (Candy_Franchise_1 Pack 3, Candy_Franchise_1 franchise, Candy category) "
                "in Gujarat during April 2026: 23,974 units delivered, revenue of Rs 319,831, "
                "Service Level 75.5%, Numeric Distribution 3.1%, Out-of-Stock rate 40.0%."
            ),
        }])
    )
    claim = AnswerClaim(
        claim_text="Candy_Franchise_1 Pack 3 had revenue of Rs 319,831 in Gujarat in April 2026.",
        entity="Candy_Franchise_1 Pack 3", metric="Revenue", period="April 2026",
        value=319831.0, citations=["sku-Gujarat-April_2026"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_bare_id_fallback_still_rejects_a_wrong_value_for_that_same_bare_citation():
    """The fallback resolves to the SAME scoped evidence a correctly-
    wrapped citation would -- it does not become permissive."""
    evidence_index = _build_evidence_index(
        make_context_records(sources=[{
            "id": "sku-Gujarat-April_2026",
            "text": (
                "SKU SKU0053 (Candy_Franchise_1 Pack 3, Candy_Franchise_1 franchise, Candy category) "
                "in Gujarat during April 2026: 23,974 units delivered, revenue of Rs 319,831, "
                "Service Level 75.5%, Numeric Distribution 3.1%, Out-of-Stock rate 40.0%."
            ),
        }])
    )
    claim = AnswerClaim(
        claim_text="Candy_Franchise_1 Pack 3 had revenue of Rs 999,999 in Gujarat in April 2026.",
        entity="Candy_Franchise_1 Pack 3", metric="Revenue", period="April 2026",
        value=999999.0, citations=["sku-Gujarat-April_2026"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


@pytest.mark.parametrize(
    "other_fact_type_citation",
    ["Atomic Distributor Facts (5)", "Atomic State KPI Facts (2)", "Atomic Category KPI Facts (1)", "Distributor Facts (5)"],
)
def test_non_sku_atomic_fact_citations_still_fail_closed(other_fact_type_citation):
    """Regression guard: the SKU-specific recovery must not accidentally
    sweep up the OTHER three Atomic Facts sections (Distributor/State/
    Category) -- those must keep failing closed exactly as the original
    unresolved_citation fix intended, since this suite only characterized
    (and only has real evidence for) the SKU-facts failure mode."""
    evidence_index = _build_evidence_index(make_sku_context_records())
    resolved = _resolve_citations([other_fact_type_citation], evidence_index)
    assert resolved == ""


# ---------------------------------------------------------------------------
# Deterministic SKU Ranking section (2026-08-23 stabilization pass) --
# adversarial citation/leakage checks for the new ranking block. Mirrors
# the "Atomic ... Facts (N)" citation-format guard above: a model citing
# the ranking SECTION NAME instead of a real Sources id must fail closed,
# never silently resolve via the permissive whole-blob fallback.
# ---------------------------------------------------------------------------


def make_sku_ranking_context_records():
    """One real SKU-evidence source row PLUS its companion
    'sku-ranking-'-id-prefixed row (the exact shape
    sku_evidence._build_ranking_source_row() produces), so tests can check
    a claim citing the ranking row's OWN id, or a claim mis-citing the
    section name itself."""
    sources = [
        {
            "id": "sku-Punjab-February_2026",
            "text": (
                "SKU-level Sales & Distribution detail for Punjab, February 2026:\n"
                "SKU SKU0041 (Marlboro Pack 1, Marlboro franchise, IPM category) in Punjab during "
                "February 2026: 835,078 units delivered, revenue of Rs 227,400,090, Service Level "
                "90.0%, Numeric Distribution 68.2%, Out-of-Stock rate 12.5%.\n"
                "SKU SKU0016 (GPI_Franchise_3 Pack 1, GPI_Franchise_3 franchise, GPI category) in "
                "Punjab during February 2026: 304,315 units delivered, revenue of Rs 26,892,317, "
                "Service Level 90.3%, Numeric Distribution 26.9%, Out-of-Stock rate 12.5%."
            ),
        },
        {
            "id": "sku-ranking-Punjab-February_2026",
            "text": (
                "This ranking was computed deterministically from the full underlying dataset for "
                "Punjab, February 2026, and is authoritative -- restate it exactly as given below. "
                "Do not recompute values, do not add or remove SKUs, and do not change the order.\n"
                "Ranked by Revenue (highest first):\n"
                "Rank 1: SKU SKU0041 (Marlboro Pack 1, Marlboro franchise, IPM category) -- "
                "Revenue: Rs 227,400,090. Source: Sources (sku-Punjab-February_2026)\n"
                "Rank 2: SKU SKU0016 (GPI_Franchise_3 Pack 1, GPI_Franchise_3 franchise, GPI category) -- "
                "Revenue: Rs 26,892,317. Source: Sources (sku-Punjab-February_2026)\n"
            ),
        },
    ]
    return make_context_records(sources=sources)


def test_ranking_row_never_becomes_extra_sku_atomic_facts():
    """The 'sku-ranking-' row must not be double-parsed into the Atomic
    SKU Facts section -- exactly 2 real SKUs x 5 metrics = 10 facts, not
    20, even though the ranking row's text also names both SKUs."""
    facts = extract_atomic_facts(make_sku_ranking_context_records())
    sku_facts = [f for f in facts if f.kind == "sku"]
    assert len(sku_facts) == 10


def test_citing_ranking_section_name_fails_closed():
    """A model mistake mirroring the exact 'Atomic SKU Facts (N)' failure
    mode this section's guidance was added to prevent -- citing the
    ranking section's own name/number ('Deterministic SKU Ranking (1)')
    instead of the real Sources id its own [Source: ...] tag names."""
    evidence_index = _build_evidence_index(make_sku_ranking_context_records())
    claim = AnswerClaim(
        claim_text="Marlboro Pack 1 had the highest Revenue at Rs 227,400,090.",
        entity="Marlboro Pack 1", metric="Revenue", value=227400090.0,
        citations=["Deterministic SKU Ranking (1)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert len(issues) == 1
    assert issues[0].issue_type == "unresolved_citation"


def test_citing_the_real_ranking_row_id_directly_still_validates_correctly():
    """A claim CAN legitimately cite the ranking row's own id directly
    (Sources (sku-ranking-...)) -- it's a real row in the sources table --
    and since the ranking is deterministically correct by construction,
    this must validate exactly like citing the underlying evidence row
    would. Proves the ranking row is inert extra evidence, not a permissive
    loophole: a WRONG value still fails even when cited this way."""
    evidence_index = _build_evidence_index(make_sku_ranking_context_records())
    correct = AnswerClaim(
        claim_text="Marlboro Pack 1 had Revenue of Rs 227,400,090.",
        entity="Marlboro Pack 1", metric="Revenue", value=227400090.0,
        citations=["Sources (sku-ranking-Punjab-February_2026)"],
    )
    assert validate_claim(correct, evidence_index) == []

    wrong = AnswerClaim(
        claim_text="Marlboro Pack 1 had Revenue of Rs 26,892,317.",
        entity="Marlboro Pack 1", metric="Revenue",
        value=26892317.0,  # this is actually GPI_Franchise_3 Pack 1's figure
        citations=["Sources (sku-ranking-Punjab-February_2026)"],
    )
    issues = validate_claim(wrong, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


def test_sku_ranking_cross_period_leakage_rejected():
    """Adversarial cross-PERIOD check (Section 4): the SAME SKU name
    appears in two different periods' evidence with two different real
    values -- a claim naming period A's SKU but citing/stating period B's
    value must fail, never silently pass because the entity name and some
    value both appear somewhere in the combined cited scope."""
    sources = [
        {
            "id": "sku-Punjab-January_2026",
            "text": (
                "SKU-level Sales & Distribution detail for Punjab, January 2026:\n"
                "SKU SKU0041 (Marlboro Pack 1, Marlboro franchise, IPM category) in Punjab during "
                "January 2026: 800,000 units delivered, revenue of Rs 200,000,000, Service Level "
                "90.0%, Numeric Distribution 68.0%, Out-of-Stock rate 12.0%."
            ),
        },
        {
            "id": "sku-Punjab-February_2026",
            "text": (
                "SKU-level Sales & Distribution detail for Punjab, February 2026:\n"
                "SKU SKU0041 (Marlboro Pack 1, Marlboro franchise, IPM category) in Punjab during "
                "February 2026: 835,078 units delivered, revenue of Rs 227,400,090, Service Level "
                "90.0%, Numeric Distribution 68.2%, Out-of-Stock rate 12.5%."
            ),
        },
    ]
    evidence_index = _build_evidence_index(make_context_records(sources=sources))
    # Claims January's period string but February's real revenue figure --
    # never stated together for January anywhere in the cited evidence.
    leaked = AnswerClaim(
        claim_text="Marlboro Pack 1 had Revenue of Rs 227,400,090 in January 2026.",
        entity="Marlboro Pack 1", metric="Revenue", period="January 2026",
        value=227400090.0,
        citations=["Sources (sku-Punjab-January_2026)", "Sources (sku-Punjab-February_2026)"],
    )
    issues = validate_claim(leaked, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)

    # Sanity: the correctly-paired claim (January period + January value) passes.
    correct = AnswerClaim(
        claim_text="Marlboro Pack 1 had Revenue of Rs 200,000,000 in January 2026.",
        entity="Marlboro Pack 1", metric="Revenue", period="January 2026",
        value=200000000.0,
        citations=["Sources (sku-Punjab-January_2026)", "Sources (sku-Punjab-February_2026)"],
    )
    assert validate_claim(correct, evidence_index) == []


def test_no_citation_at_all_still_uses_whole_blob_fallback_unchanged():
    """Regression guard: a claim that cites NOTHING (empty citations list)
    is a different, less severe case than citing something unresolvable --
    it must keep falling back to the whole-blob check (existing behavior),
    not be treated as an unresolved_citation failure."""
    sources = [{"id": "5", "text": "State: Punjab\nPeriod: March 2025\n\nSKU Alpha Pack 1 Revenue was Rs 95,267,583 in Punjab during March 2025."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)

    claim = AnswerClaim(claim_text="Revenue was Rs 95,267,583.", metric="Revenue", value=95267583, citations=[])
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type == "unresolved_citation" for i in issues)
    # The value is genuinely present in the whole blob, so the permissive
    # (but explicitly-flagged, per flag()'s used_fallback message) fallback
    # still lets this pass -- unchanged pre-existing behavior.
    assert not any(i.issue_type == "unsupported_numeric" for i in issues)


# ---------------------------------------------------------------------------
# Prose-level fallback scanners (secondary safety net) -- still active
# ---------------------------------------------------------------------------


def test_prose_fallback_flags_unsupported_adjective_with_no_citation():
    records = make_context_records(sources=[{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nOut-of-Stock rate was 5.2%."}])
    answer = "The out-of-stock rate was an alarming 5.2%."  # no [Data: ...] tag at all
    issues = scan_qualitative_language(answer, records)
    assert any(i.term == "alarming" and i.source == "prose_fallback" for i in issues)


def test_prose_fallback_causal_scanner_catches_unreported_causal_sentence():
    records = make_context_records(sources=[{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nService Level was 94.3%."}])
    answer = "Distributor issues led to the decline."  # causal, no citation, not in any structured claim
    issues = scan_causal_language(answer, records)
    assert any(i.issue_type == "unsupported_causal" and i.source == "prose_fallback" for i in issues)


def test_direction_consistency_still_flags_prose_contradiction():
    comparisons = [
        MetricComparison(metric="Productivity", period_a="September 2025", value_a=92.0, period_b="October 2025", value_b=85.5, direction="down"),
    ]
    answer = "Productivity climbed in October 2025, reaching 85.5%."
    issues = check_direction_consistency(answer, comparisons)
    assert any(i.issue_type == "direction_contradiction" for i in issues)


def test_direction_consistency_flat_never_flagged():
    comparisons = [
        MetricComparison(metric="Inventory Turns", period_a="October 2025", value_a=2.64, period_b="November 2025", value_b=2.64, direction="flat"),
    ]
    answer = "Inventory Turns declined slightly."
    issues = check_direction_consistency(answer, comparisons)
    assert issues == []


# ---------------------------------------------------------------------------
# check_grounding() end to end -- claims + prose fallback combined
# ---------------------------------------------------------------------------


def test_check_grounding_passes_clean_answer_with_no_claims():
    records = make_context_records(sources=[{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nProductivity for Bihar in October 2025 was 85.5%."}])
    answer = "Bihar's productivity in October 2025 was 85.5% [Data: Sources (0)]."
    result = check_grounding(answer, records, claims=[])
    assert result.status == "pass"


def test_check_grounding_fails_on_claim_level_issue_alone():
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    claims = [
        AnswerClaim(
            claim_text="Service Level improved from October to November 2025.",
            metric="Service Level", period="November 2025", comparison_period="October 2025",
            value=92.9, comparison_value=94.3, direction="improved",
            citations=["Sources (0)", "Sources (3)"],
        )
    ]
    # Prose itself is neutral (no flagged words), only the structured claim is wrong.
    answer = "Bihar's Service Level improved from October to November 2025 [Data: Sources (0); Sources (3)]."
    result = check_grounding(answer, records, claims=claims)
    assert result.status == "fail"
    assert any(i.source == "claim" and i.issue_type == "direction_contradiction" for i in result.flagged_issues)


def test_check_grounding_explanation_reports_claim_vs_prose_counts():
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    answer = "Bihar's Service Level was an alarming 92.9% in November 2025 [Data: Sources (3)]."
    result = check_grounding(answer, records, claims=[])
    assert result.status == "fail"
    assert "prose-fallback" in result.explanation


# ---------------------------------------------------------------------------
# Real-world replay: the actual second live drafted answer (with
# "concern"/"concerning" AND causal "contributing to"/"resulting in"/
# "attributed to" language)
# ---------------------------------------------------------------------------

REAL_EVIDENCE_SOURCES = [
    {"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nProductivity for Bihar in October 2025 was 85.5%. Service Level for Bihar in October 2025 was 94.3%."},
    {"id": "3", "text": (
        "State: Bihar\nPeriod: November 2025\n\nService Level for Bihar in November 2025 was 92.9%.\n"
        "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on Service Level "
        "in November 2025: 82.6% vs. the state average of 92.9%, a gap of 10.4 percentage points.\n"
        "For the Ferrero category in Bihar during November 2025: Out-of-Stock rate was 13.0%."
    )},
]
REAL_EVIDENCE_REPORTS = [
    {"id": "9", "title": "Bihar Candy Supply Dynamics", "content": "Concerns surrounding inventory management for candy in Bihar, November 2025, suggesting inefficiencies in managing dropsize metrics."},
]

REAL_DRAFTED_ANSWER_V2 = """Bihar's Service Level experienced a decline from 94.3% in October 2025 to 92.9% in November 2025. This decrease of 1.4 percentage points is a point of concern for stakeholders monitoring market performance in the region.

One of the primary factors contributing to this decline appears to be inventory management challenges within the state. Reports have indicated significant out-of-stock rates for various product categories, including notable discrepancies in the availability of candy and Ferrero products, which could hinder overall service delivery. For instance, the out-of-stock rate for Ferrero products reached 13.0% in November, raising concerns about supply chain disruptions and inventory control practices [Data: Reports (9); Entities (259, 264)].

Moreover, specific distributors, such as Zhang, Brooks and Miles Distributors, presented a significant deviation in their service levels, recording only 82.6% compared to the state average [Data: Sources (3)]. The combined impact of these individual distributor performance discrepancies and broader supply chain inefficiencies may have collectively influenced Bihar's overall service level, resulting in the observed decline.

In sum, the drop in service levels from October to November 2025 seems to stem from various factors encompassing out-of-stock issues, suboptimal inventory management, and challenges experienced by significant distributors, all contributing to reduced service efficacy [Data: Sources (3), Reports (9); Entities (4, 6)]."""


# The structured claims a real GPT-4o-mini following the new
# claim-extraction prompt would plausibly self-report for
# REAL_DRAFTED_ANSWER_V2 -- this is the REALISTIC end-to-end path (claim-
# level validation as the primary mechanism), not the degraded
# claims=[]/prose-only case covered separately above.
REAL_ANSWER_V2_CLAIMS = [
    AnswerClaim(
        claim_id=0, claim_text="Bihar's Service Level experienced a decline from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3, direction="declined", delta=1.4,
        citations=["Sources (0)", "Sources (3)"],
    ),
    AnswerClaim(
        claim_id=1, claim_text="This decrease of 1.4 percentage points is a point of concern for stakeholders.",
        claim_type="qualitative", metric="Service Level", citations=["Sources (3)"],
    ),
    AnswerClaim(
        claim_id=2,
        claim_text=(
            "The combined impact of these individual distributor performance discrepancies and broader "
            "supply chain inefficiencies may have collectively influenced Bihar's overall service level, "
            "resulting in the observed decline."
        ),
        claim_type="causal", entity="Bihar", metric="Service Level",
        citations=["Sources (3)", "Reports (9)"],
    ),
    AnswerClaim(
        claim_id=3,
        claim_text="Zhang, Brooks and Miles Distributors presented a significant deviation in their service levels.",
        claim_type="deviation", entity="Zhang, Brooks and Miles Distributors", metric="Service Level",
        citations=["Sources (3)"],
    ),
]


def test_real_v2_answer_flags_concern_and_causal_language():
    """Session-recorded replay of the second live GPT-4o-mini answer, WITH
    the structured claims a model following the new prompt would report.
    The OLD checker passed this whole answer (0 issues). The new checker
    must flag the 'point of concern' qualitative claim and the causal
    'contributing to'/'resulting in' claim -- while still passing the
    genuinely well-grounded trend and deviation claims (94.3->92.9, and
    the distributor deviation the source explicitly states)."""
    records = make_context_records(sources=REAL_EVIDENCE_SOURCES, reports=REAL_EVIDENCE_REPORTS)
    result = check_grounding(REAL_DRAFTED_ANSWER_V2, records, claims=REAL_ANSWER_V2_CLAIMS)
    assert result.status == "fail"

    claim_issues = [i for i in result.flagged_issues if i.source == "claim"]
    assert any(i.claim_id == 1 and i.issue_type == "unsupported_qualifier" and i.term == "concern" for i in claim_issues)
    assert any(i.claim_id == 2 and i.issue_type == "unsupported_causal" for i in claim_issues)
    # The genuinely grounded claims (trend numbers, real deviation) must NOT be flagged.
    assert not any(i.claim_id == 0 for i in claim_issues)
    assert not any(i.claim_id == 3 for i in claim_issues)


# ---------------------------------------------------------------------------
# Phase 7 hardening pass: evidence specificity + direction-scanner false
# positives on future/recommendation language (see forensic report on the
# live "Why did Bihar's Service Level decline from October to November
# 2025?" query -- the draft's real trend claim always passed; the false
# failures came from "various distributors" (an invented generic entity)
# and "...improve service levels moving forward" (a recommendation
# sentence misread as a historical UP claim).
#
# Test-to-requirement map (Part 5 of the hardening task):
#   1: test_historical_up_direction_detected
#   2: test_historical_down_direction_detected
#   3: test_future_up_recommendation_improve_does_not_false_positive
#   4: test_future_up_recommendation_increase_does_not_false_positive
#   5: test_future_recommendation_strengthen_next_month_does_not_false_positive
#   6: test_future_down_recommendation_not_treated_as_historical_decline
#   7: test_historical_causal_phrasing_still_flagged_when_contradicting
#   8: test_various_up_words_in_recommendations_do_not_false_positive,
#      test_various_down_words_in_recommendations_do_not_false_positive,
#      test_future_or_prescriptive_detects_recommendation_language
#   9: test_generic_distributor_aggregation_passes_with_specificity_warning
#  10: test_specific_named_distributors_pass_cleanly
#  11: test_unsupported_named_distributor_still_fails,
#      test_generic_group_reference_fails_when_evidence_lacks_plurality
#  12: test_future_recommendation_with_grounded_entities_passes
#  13: test_unsupported_recommendation_number_fails
# ---------------------------------------------------------------------------

GENERIC_DISTRIBUTOR_SOURCES = [
    {"id": "3", "text": (
        "State: Bihar\nPeriod: November 2025\n\n"
        "Distributor Landry Ltd Distributors showed a significant deviation on Out-of-Stock Rate "
        "in November 2025: 28.6% vs. the state average of 13.0%, a gap of 15.6 percentage points.\n"
        "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on Out-of-Stock "
        "Rate in November 2025: 20.0% vs. the state average of 7.3%, a gap of 12.7 percentage points.\n"
        "Distributor Bennett-Webster Distributors showed a significant deviation on Out-of-Stock Rate "
        "in November 2025: 20.0% vs. the state average of 7.3%, a gap of 12.7 percentage points.\n"
    )},
]


# --- 1-2: historical direction words are still correctly read as historical ---


def test_historical_up_direction_detected():
    """1: a plain historical UP sentence, actual direction genuinely up ->
    no contradiction (i.e. 'increased' is read as historical, not
    suppressed as if it were a recommendation)."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=90.0, period_b="November 2025", value_b=94.0, direction="up")]
    answer = "Service Level increased from October to November."
    assert check_direction_consistency(answer, comparisons) == []


def test_historical_down_direction_detected():
    """2: same, for a historical DOWN sentence."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    answer = "Service Level declined from October to November."
    assert check_direction_consistency(answer, comparisons) == []


# --- 3-6: future/prescriptive sentences must not trigger a historical
# contradiction, regardless of which directional word or phrasing is used ---


def test_future_up_recommendation_improve_does_not_false_positive():
    """3: the original live false positive -- actual direction is DOWN,
    but this sentence is a future recommendation, not a historical claim."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    answer = "Service Level should improve going forward."
    assert check_direction_consistency(answer, comparisons) == []


def test_future_up_recommendation_increase_does_not_false_positive():
    """4: same shape, alternative UP word -- proves the fix isn't
    hardcoded to the word 'improve'."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    answer = "Service Level should increase going forward."
    assert check_direction_consistency(answer, comparisons) == []


def test_future_recommendation_strengthen_next_month_does_not_false_positive():
    """5: a third UP word ('strengthen', newly added to UP_WORDS) plus a
    different future-time phrase ('next month') and a different subject
    ('Management should ...') -- still correctly suppressed."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    answer = "Management should strengthen Service Level next month."
    assert check_direction_consistency(answer, comparisons) == []


def test_future_down_recommendation_not_treated_as_historical_decline():
    """6: the DOWN-word mirror image -- a recommendation to REDUCE a
    metric must not be read as a historical claim that it already fell,
    even when the actual historical direction is the opposite ('up')."""
    comparisons = [MetricComparison(metric="OOS", period_a="October 2025", value_a=5.0, period_b="November 2025", value_b=9.0, direction="up")]
    answer = "Management should reduce OOS going forward."
    assert check_direction_consistency(answer, comparisons) == []


def test_historical_causal_phrasing_still_flagged_when_contradicting():
    """7: 'improved because...' has no modal verb, no future-time phrase,
    and no recommendation-action word -- 'because' alone must NOT be
    treated as future/prescriptive framing, so a genuine historical
    contradiction in causal phrasing must still be caught."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    answer = "Service Level improved because of strong distributor engagement."
    issues = check_direction_consistency(answer, comparisons)
    assert any(i.issue_type == "direction_contradiction" for i in issues)


# --- 8: breadth check -- many different words/phrases, not just "improve" ---


@pytest.mark.parametrize("answer", [
    "Service Level should increase going forward.",
    "Service Level should strengthen going forward.",
    "Service Level should recover going forward.",
    "Management should boost Service Level next quarter.",
    "The team should raise Service Level going forward.",
    "Service Level could climb in the future with the right interventions.",
    "Leadership should enhance Service Level moving forward.",
])
def test_various_up_words_in_recommendations_do_not_false_positive(answer):
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    assert check_direction_consistency(answer, comparisons) == [], f"false positive for: {answer!r}"


@pytest.mark.parametrize("answer", [
    "Management should lower Service Level risk exposure going forward.",
    "The business should decrease Service Level volatility next quarter.",
    "Leadership should shrink the Service Level gap moving forward.",
])
def test_various_down_words_in_recommendations_do_not_false_positive(answer):
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=90.0, period_b="November 2025", value_b=94.0, direction="up")]
    assert check_direction_consistency(answer, comparisons) == [], f"false positive for: {answer!r}"


@pytest.mark.parametrize("sentence", [
    "Service Level should improve going forward.",
    "The team must address Service Level concerns.",
    "Consider monitoring the OOS rate next quarter.",
    "Management needs to stabilize Service Level.",
    "In the future, the business could optimize distributor coverage.",
    "Continue to track Service Level closely.",
])
def test_future_or_prescriptive_detects_recommendation_language(sentence):
    assert _is_future_or_prescriptive(sentence) is True


@pytest.mark.parametrize("sentence", [
    "Service Level declined from 94.3% to 92.9%.",
    "Service Level improved because of strong distributor engagement.",
    "Productivity was 85.5% in October 2025.",
])
def test_future_or_prescriptive_false_for_historical_sentences(sentence):
    assert _is_future_or_prescriptive(sentence) is False


# --- 9-11: generic group entity references ("various distributors") ---


def test_generic_distributor_aggregation_passes_with_specificity_warning():
    """9: the evidence genuinely names 3 distinct distributors with OOS
    deviations -- 'various distributors' is a TRUE generic aggregation of
    them, so it must not fail grounding, but it should carry a
    warning-severity 'generic_aggregation' notice (not silent, and not
    a hard failure) so the caller/UI can see specificity was lost."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Various distributors showed OOS deviations.",
        claim_type="deviation", entity="various distributors", metric="Out-of-Stock Rate",
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert not any(i.severity == "error" for i in issues)
    assert any(i.issue_type == "generic_aggregation" and i.severity == "warning" for i in issues)

    # And at the check_grounding() level, the overall status stays "pass".
    answer = "Various distributors showed OOS deviations [Data: Sources (3)]."
    result = check_grounding(answer, records, claims=[claim])
    assert result.status == "pass"
    assert any(i.issue_type == "generic_aggregation" for i in result.flagged_issues)


def test_specific_named_distributors_pass_cleanly():
    """10: naming the specific, evidence-backed distributor and its real
    values -- the preferred, more-specific answer style -- passes clean,
    with no specificity warning either (there's nothing generic to warn
    about)."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Landry Ltd Distributors showed a 28.6% OOS rate versus the 13.0% state average.",
        claim_type="deviation", entity="Landry Ltd Distributors", metric="Out-of-Stock Rate",
        value=28.6, comparison_value=13.0, citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_unsupported_named_distributor_still_fails():
    """11a: a specific but entirely invented distributor name must still
    fail -- the generic-aggregation carve-out only applies to genuinely
    generic phrasing ('various X'), never to a fabricated proper name."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Acme Logistics Distributors showed a significant OOS deviation.",
        claim_type="deviation", entity="Acme Logistics Distributors", metric="Out-of-Stock Rate",
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_entity" and i.severity == "error" for i in issues)


def test_generic_group_reference_fails_when_evidence_lacks_plurality():
    """11b: 'various distributors' must NOT be automatically acceptable --
    if the cited evidence doesn't actually name multiple distributors,
    the generic aggregation is just as unsupported as any other invented
    entity and must fail closed, per the explicit safety constraint."""
    sources = [{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nService Level for Bihar in October 2025 was 94.3%."}]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Various distributors affected Service Level.",
        claim_type="deviation", entity="various distributors", metric="Service Level",
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_entity" and i.severity == "error" for i in issues)


# --- 12-13: prose-level future recommendations, end to end via check_grounding() ---


def test_future_recommendation_with_grounded_entities_passes():
    """12: a forward-looking recommendation that names real, evidence-
    backed distributors and explains why (their observed OOS deviations)
    -- should pass. The 'because of' rationale here is the recommendation's
    justification, not a historical causal claim about the KPI, so it must
    not be flagged as unsupported_causal either."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    answer = (
        "Going forward, monitor Landry Ltd Distributors and Zhang, Brooks and Miles Distributors "
        "because of their observed OOS deviations."
    )
    result = check_grounding(answer, records, claims=[])
    assert result.status == "pass"


def test_unsupported_recommendation_number_fails():
    """13: framing a fabricated figure as a recommendation must not exempt
    it from grounding -- '50 new warehouses' is not backed by any cited
    evidence."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    answer = "Going forward, open 50 new warehouses."
    result = check_grounding(answer, records, claims=[])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_recommendation" for i in result.flagged_issues)


def test_scan_recommendation_language_respects_citation_scope():
    """A recommendation number that IS real, but only in evidence NOT
    cited by that sentence, must still fail -- same citation-scoping
    discipline as every other mechanism in this file."""
    records = make_context_records(
        sources=[{"id": "0", "text": "State: Bihar\nPeriod: October 2025\n\nService Level for Bihar in October 2025 was 94.3%."}],
        reports=[{"id": "9", "title": "Unrelated", "content": "The warehouse count is 50."}],
    )
    answer = "Going forward, open 50 new warehouses [Data: Sources (0)]."
    issues = scan_recommendation_language(answer, records)
    assert any(i.issue_type == "unsupported_recommendation" for i in issues)


# ---------------------------------------------------------------------------
# Phase 7c: final grounding hardening pass.
#
# Test-to-requirement map:
#   1a: test_comparison_claim_both_periods_cited_passes
#   1b: test_comparison_claim_missing_comparison_period_citation_fails
#   1c: test_comparison_claim_both_periods_in_evidence_but_citation_omits_one_fails
#   1d: test_comparison_claim_value_correct_but_wrong_period_fails
#   3:  test_malformed_concatenated_multi_entity_string_still_fails
#   4:  test_future_recommendation_still_not_a_direction_false_positive_regression
#       (regression re-check of Phase 7b's fix, word-agnostic per item 4)
# ---------------------------------------------------------------------------


def test_comparison_claim_both_periods_cited_passes():
    """1a: both the primary (November) and comparison (October) source
    documents are cited, and each genuinely states its own period+value
    together -- must pass with no unsupported_comparison issue."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3,
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_comparison_claim_missing_comparison_period_citation_fails():
    """1b: only the primary (November) source is cited -- the comparison
    (October) side is entirely absent from the cited scope."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3,
        citations=["Sources (3)"],  # October source never cited
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_comparison" for i in issues)


def test_comparison_claim_both_periods_in_evidence_but_citation_omits_one_fails():
    """1c: the October document genuinely exists among the retrieved
    evidence (proving this isn't a retrieval gap -- sanity-checked below),
    but this claim's citation list only references November plus an
    unrelated report. Only what's CITED counts -- must still fail."""
    records = make_context_records(
        sources=BIHAR_OCT_NOV_SOURCES,
        reports=[{"id": "9", "title": "Unrelated", "content": "Unrelated candy report, nothing about Service Level."}],
    )
    evidence_index = _build_evidence_index(records)
    assert "October 2025" in evidence_index["sources"]["0"]  # sanity: Oct was genuinely retrieved
    claim = AnswerClaim(
        claim_text="Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3,
        citations=["Sources (3)", "Reports (9)"],  # Nov + an unrelated report, never Oct
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_comparison" for i in issues)


def test_comparison_claim_value_correct_but_wrong_period_fails():
    """1d: the cited evidence DOES state Service Level=94.3, but for
    September, not October -- and DOES mention 'October 2025', but only
    for a different metric (Productivity=80.0). Both independent
    old-style checks (value-present-anywhere, period-present-anywhere)
    would be individually satisfied; only the new pairing check catches
    that the claimed period and value are never stated together."""
    sources = [
        {"id": "3", "text": "State: Bihar\nPeriod: November 2025\n\nService Level for Bihar in November 2025 was 92.9%."},
        {"id": "7", "text": "State: Bihar\nPeriod: September 2025\n\nService Level for Bihar in September 2025 was 94.3%."},
        {"id": "8", "text": "State: Bihar\nPeriod: October 2025\n\nProductivity for Bihar in October 2025 was 80.0%."},
    ]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3,
        citations=["Sources (3)", "Sources (7)", "Sources (8)"],
    )
    # Confirm the OLD independent checks alone would have missed this:
    # 94.3 appears somewhere in scope (Sept), and "October 2025" appears
    # somewhere in scope (Productivity doc) -- just never together.
    scope = " ".join([evidence_index["sources"][i] for i in ("3", "7", "8")])
    assert "94.3" in scope and "October 2025" in scope

    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_comparison" for i in issues)


def test_malformed_concatenated_multi_entity_string_still_fails():
    """3: the live-observed malformed pattern -- several genuinely-named
    distributors concatenated into ONE entity string -- must still fail.
    This exact string never appears in evidence even though each
    individual distributor name does; the Phase 7b generic-aggregation
    carve-out must NOT swallow this (it only matches a bare quantifier +
    plural-noun shape like 'various distributors', not a proper-name
    list), so this is a regression test that the two mechanisms stay
    properly separated."""
    records = make_context_records(sources=GENERIC_DISTRIBUTOR_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bennett-Webster, Casey, Nguyen and Ramirez, Weaver-Sherman Distributors also faced elevated out-of-stock rates.",
        claim_type="deviation",
        entity="Bennett-Webster, Casey, Nguyen and Ramirez, Weaver-Sherman Distributors",
        metric="Out-of-Stock Rate",
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_entity" and i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Entity-fusion retry-instruction hardening: grounding still fails (never
# loosened) but the flagged issue's detail now carries a distinguishable
# marker for a specific, common failure shape -- two real, individually-
# supported distributor names fused into one entity field -- so answer.py
# can send a much more direct retry instruction (see test_inference_answer.py).
# ---------------------------------------------------------------------------

# Two distinct real distributors, each with their OWN deviation sentence --
# mirrors the live Himachal Pradesh failure (Mooney, Lamb and Weber
# Distributors / Scott-Norman Distributors), with fictional names.
FUSION_SOURCES = [
    {"id": "20", "text": (
        "State: Freedonia\nPeriod: May 2027\n\n"
        "Distributor Kowalski, Ortiz and Reyes Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the IPM category in May 2027: 20.0% vs. the state average of 6.2%, "
        "a gap of 13.8 percentage points.\n"
        "Distributor Vance-Whitfield Distributors showed a significant deviation on Out-of-Stock "
        "Rate for the IPM category in May 2027: 20.0% vs. the state average of 6.2%, a gap of 13.8 "
        "percentage points.\n"
    )},
]


def test_known_distributor_core_names_strips_trailing_distributors_suffix():
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    scope = evidence_index["sources"]["20"]
    cores = _known_distributor_core_names(scope)
    assert "Kowalski, Ortiz and Reyes" in cores
    assert "Vance-Whitfield" in cores
    assert "Kowalski, Ortiz and Reyes Distributors" not in cores  # suffix stripped


def test_detect_entity_fusion_finds_both_names_in_fused_entity():
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    scope = evidence_index["sources"]["20"]
    fused_entity = "Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors"
    matched = _detect_entity_fusion(fused_entity, scope)
    assert set(matched) == {"Kowalski, Ortiz and Reyes", "Vance-Whitfield"}


def test_detect_entity_fusion_empty_for_single_wrong_entity():
    """A genuinely wrong (not fused) entity like a state name that appears
    nowhere near any known distributor name must NOT be flagged as a
    fusion -- fewer than 2 known names found."""
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    scope = evidence_index["sources"]["20"]
    assert _detect_entity_fusion("Bihar", scope) == []


def test_detect_entity_fusion_empty_for_single_real_distributor_name():
    """Exactly one known name present must not count as a fusion (needs
    2+) -- a single correctly-named-but-uncited distributor is a plain
    unsupported entity, not a fusion."""
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    scope = evidence_index["sources"]["20"]
    assert _detect_entity_fusion("Vance-Whitfield Distributors", scope) == []


def test_validate_claim_fused_entity_gets_fusion_detail_message():
    """End-to-end: a claim whose entity fuses two real, individually cited
    distributors still fails (severity=error, unsupported_entity -- nothing
    loosened), but its detail now names both real distributors and carries
    the marker phrase answer.py's retry-formatter looks for."""
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    claim = AnswerClaim(
        claim_text="Kowalski, Ortiz and Reyes and Vance-Whitfield Distributors reported 20.0% OOS.",
        claim_type="deviation",
        entity="Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors",
        metric="Out-of-Stock Rate",
        value=20.0,
        citations=["Sources (20)"],
    )
    issues = validate_claim(claim, evidence_index)
    entity_issues = [i for i in issues if i.issue_type == "unsupported_entity"]
    assert entity_issues
    assert all(i.severity == "error" for i in entity_issues)  # grounding not loosened
    assert any("combines multiple distinct named entities" in i.detail for i in entity_issues)
    assert any("Kowalski, Ortiz and Reyes" in i.detail and "Vance-Whitfield" in i.detail for i in entity_issues)


def test_validate_claim_comparison_entity_fusion_also_detected():
    """Same fusion signal, applied to comparison_entity (Phase 8c's second-
    entity field) -- structurally identical risk, same detection."""
    evidence_index = _build_evidence_index(make_context_records(sources=FUSION_SOURCES))
    claim = AnswerClaim(
        claim_text="Freedonia's OOS was higher than Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors'.",
        claim_type="comparison",
        entity="Freedonia",
        comparison_entity="Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors",
        metric="Out-of-Stock Rate",
        comparison_value=20.0,
        citations=["Sources (20)"],
    )
    issues = validate_claim(claim, evidence_index)
    comparison_entity_issues = [
        i for i in issues if i.issue_type == "unsupported_entity" and i.term == claim.comparison_entity
    ]
    assert comparison_entity_issues
    assert any("combines multiple distinct named entities" in i.detail for i in comparison_entity_issues)


def test_future_recommendation_still_not_a_direction_false_positive_regression():
    """4: Phase 7c must not regress Phase 7b's direction-scanner fix.
    Re-checked here (in addition to the full Phase 7b suite still running
    green) with the task's own example sentence and a second, different
    UP word, against an actual DOWN historical direction."""
    comparisons = [MetricComparison(metric="Service Level", period_a="October 2025", value_a=94.3, period_b="November 2025", value_b=92.9, direction="down")]
    for answer in [
        "Going forward, the company should improve Service Level.",
        "Going forward, the company should strengthen Service Level.",
    ]:
        assert check_direction_consistency(answer, comparisons) == [], f"false positive for: {answer!r}"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def test_prompt_forbids_combining_multiple_entities_in_one_claim_field():
    """3 (generation-side): the claim-extraction instructions sent to the
    model must explicitly forbid concatenating multiple named entities
    into one entity field -- the deterministic test above covers the
    grounding-side safety net; this covers the prompt-side prevention.
    Whitespace-normalized so the assertion survives incidental rewrapping
    of the prompt text's line breaks."""
    from src.inference.answer import _CLAIM_EXTRACTION_SUFFIX

    assert "Never combine multiple named entities" in _normalize_whitespace(_CLAIM_EXTRACTION_SUFFIX)


def test_prompt_instructs_against_inventing_entity_level_facts():
    """2 (generation-side): the specificity guidance must explicitly
    forbid inventing/inferring entity-level facts not present in cited
    evidence, and must allow generic aggregation only when genuinely
    warranted -- not as a shortcut."""
    from src.inference.answer import _SPECIFICITY_AND_FRAMING_GUIDANCE

    normalized = _normalize_whitespace(_SPECIFICITY_AND_FRAMING_GUIDANCE).lower()
    assert "never invent or infer an entity-level fact" in normalized


# ---------------------------------------------------------------------------
# Phase 7d: citation-selection hardening -- behavioral proof (grounding
# side). Phase 7d changes ONLY prompt text (see test_inference_answer.py
# for the prompt-content tests); grounding_check.py itself is untouched.
# This test proves that fact directly: a Relationships citation must
# still validate cleanly, with zero code changes here, when the
# relationship itself genuinely is the claimed fact -- proving the fix
# is about which record the MODEL reaches for, never about the checker
# becoming more permissive of Relationships citations in general.
# ---------------------------------------------------------------------------


def test_relationship_citation_still_valid_when_relationship_is_the_claimed_fact():
    """C: citing a Relationships record must still pass when the
    relationship itself -- not a separate document fact -- is what's
    being claimed (the Phase 7d spec's own second example: 'Landry Ltd
    Distributors is associated with Bihar')."""
    relationships = [
        {
            "id": "10",
            "source": "Landry Ltd Distributors",
            "target": "Bihar",
            "description": "Landry Ltd Distributors operates in Bihar.",
        },
    ]
    records = make_context_records(relationships=relationships)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Landry Ltd Distributors is associated with Bihar.",
        claim_type="factual_numeric",
        entity="Landry Ltd Distributors",
        citations=["Relationships (10)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_relationship_only_citation_still_fails_when_fact_is_not_in_the_relationship():
    """Companion negative case: a Relationships record that merely
    connects two entities must NOT validate a claim about a metric/value
    the relationship's own text never states -- proving Phase 7d hasn't
    made Relationships citations more permissive in general, only
    confirmed they still work when they genuinely ARE the evidence."""
    relationships = [
        {
            "id": "10",
            "source": "Landry Ltd Distributors",
            "target": "Bihar",
            "description": "Landry Ltd Distributors operates in Bihar.",
        },
    ]
    records = make_context_records(relationships=relationships)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Landry Ltd Distributors showed an Out-of-Stock deviation of 28.6%.",
        claim_type="deviation",
        entity="Landry Ltd Distributors",
        metric="Out-of-Stock Rate",
        value=28.6,
        citations=["Relationships (10)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


# ---------------------------------------------------------------------------
# Phase 8 (P0): primary-claim entity+metric+value co-occurrence check.
# Fixes a live failure where a real, correctly-retrieved Candy Out-of-Stock
# Rate deviation for one distributor was restated as that SAME distributor's
# Ferrero Out-of-Stock Rate: both the entity name and the value (28.6) were
# genuinely present in the cited evidence independently (so the OLD,
# independent entity/value checks would each pass on their own) -- just
# never stated TOGETHER for the "Ferrero" category the claim actually named.
# Mirrors _comparison_side_supported()'s sentence-scoped co-occurrence
# pattern, applied to the primary entity/value/metric instead of the
# comparison side.
#
# Test-to-requirement map:
#   1a: test_correct_ferrero_deviation_claim_passes
#   1b: test_wrong_category_conflation_fails_even_though_entity_and_value_each_appear
#   2:  test_state_level_category_claim_still_passes_under_primary_cooccurrence_check
# ---------------------------------------------------------------------------

# The real Bihar October 2025 distributor-deviation bullets (see the Phase 8
# forensic audit) -- Zhang, Brooks and Miles' real Ferrero deviation and
# Long, Anderson and Irwin's real Candy deviation, adjacent in one document,
# exactly the shape that caused the live conflation.
BIHAR_DISTRIBUTOR_DEVIATION_SOURCE = [
    {
        "id": "0",
        "text": (
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
        ),
    },
]


def test_correct_ferrero_deviation_claim_passes():
    """1a: Zhang, Brooks and Miles + Ferrero Out-of-Stock Rate + 42.9 -- the
    entity, metric, and value ARE all stated together as one fact -- must
    PASS cleanly."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Zhang, Brooks and Miles Distributors showed a Ferrero Out-of-Stock Rate deviation of 42.9%.",
        claim_type="deviation",
        entity="Zhang, Brooks and Miles Distributors",
        metric="Ferrero Out-of-Stock Rate",
        value=42.9,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_wrong_category_conflation_fails_even_though_entity_and_value_each_appear():
    """1b: the live failure, reproduced exactly. Long, Anderson and Irwin
    Distributors' REAL deviation is Candy Out-of-Stock Rate 28.6% -- both
    'Long, Anderson and Irwin Distributors' and '28.6' genuinely appear in
    the cited evidence (proving the OLD independent entity/value checks
    would each pass on their own below), but never together for 'Ferrero'
    -- must FAIL."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (0)"], evidence_index)
    # Sanity check, same pattern as the Phase 7c comparison tests: prove the
    # OLD, independent checks would each pass alone, so this test actually
    # proves the NEW co-occurrence check -- not something else -- is firing.
    assert "Long, Anderson and Irwin Distributors" in scope
    assert "28.6" in scope

    claim = AnswerClaim(
        claim_text="Long, Anderson and Irwin Distributors showed a Ferrero Out-of-Stock Rate deviation of 28.6%.",
        claim_type="deviation",
        entity="Long, Anderson and Irwin Distributors",
        metric="Ferrero Out-of-Stock Rate",
        value=28.6,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(
        i.issue_type == "unsupported_numeric" and "together as one fact" in i.detail for i in issues
    )


def test_state_level_category_claim_still_passes_under_primary_cooccurrence_check():
    """2: a plain state-level category claim (not distributor-level) must
    still pass -- the new check must not be so strict that ordinary
    category/channel-level facts stop validating."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bihar's channel-level Ferrero Out-of-Stock rate was 5.2% in October 2025.",
        claim_type="factual_numeric",
        entity="Bihar",
        metric="Ferrero Out-of-Stock",
        value=5.2,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


# ---------------------------------------------------------------------------
# Phase 8 regression: fabricated "<State> <Category> Category" composite
# entity (e.g. "Bihar Ferrero Category"). Live failure: the P0 co-occurrence
# check (above) correctly rejects a Candy value misattributed as Ferrero for
# a NAMED DISTRIBUTOR entity -- but a retry, told to fix that, reframed the
# same wrong value under a fabricated state+category entity string instead
# of a real distributor name. No source document ever spells a state and a
# category as one three-word literal entity (they're always two separate
# facts/paragraphs), so the plain literal-entity check always failed this
# shape even when the underlying Bihar+Ferrero fact WAS genuinely
# well-supported -- collapsing an otherwise-correct answer into
# insufficient_evidence. The fix decomposes the fabricated phrase into its
# real parts (state, category) and requires THOSE to co-occur with the
# value/metric, preserving the P0 protection while not fabricating a hard
# failure out of a literal-string requirement nothing was ever meant to
# satisfy.
#
# Test-to-requirement map:
#   A: test_composite_state_category_entity_does_not_require_literal_phrase
#   B: test_distributor_deviation_delta_against_state_average_accepted
#   C: test_composite_entity_cannot_launder_wrong_category_value
#   D: test_composite_entity_wrong_delta_still_rejected
#   E: covered by the full suite above continuing to pass unchanged
# ---------------------------------------------------------------------------


def test_composite_state_category_entity_does_not_require_literal_phrase():
    """A: entity='Bihar Ferrero Category' is never spelled verbatim in any
    source document (state-level and category-level facts are separate
    sentences/paragraphs) -- the claim must still PASS when the decomposed
    parts (Bihar, Ferrero) and the value/metric genuinely are stated
    together as one fact, exactly as the real live Q1 answer needs."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (0)"], evidence_index)
    # Sanity check: prove the literal three-word phrase really is absent,
    # so this test is actually exercising the decomposition fix.
    assert "Bihar Ferrero Category" not in scope

    claim = AnswerClaim(
        claim_text="For the Ferrero category in Bihar, the channel-level Out-of-Stock rate was 5.2% in October 2025.",
        claim_type="factual_numeric",
        entity="Bihar Ferrero Category",
        metric="Out-of-Stock Rate",
        value=5.2,
        citations=["Sources (0)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_distributor_deviation_delta_against_state_average_accepted():
    """B: Zhang, Brooks and Miles Distributors' Ferrero Out-of-Stock Rate
    (42.9%) against the Bihar Ferrero state average (5.2%) -- delta 37.7 is
    the correct gap and must be accepted, not flagged."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text=(
            "Zhang, Brooks and Miles Distributors showed a Ferrero Out-of-Stock Rate of 42.9% "
            "versus the Bihar state average of 5.2%, a gap of 37.7 percentage points."
        ),
        claim_type="deviation",
        entity="Zhang, Brooks and Miles Distributors",
        metric="Out-of-Stock Rate",
        value=42.9,
        comparison_value=5.2,
        delta=37.7,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert not any(i.issue_type in ("unsupported_delta", "unsupported_numeric", "unsupported_entity") for i in issues)


def test_composite_entity_cannot_launder_wrong_category_value():
    """C: the actual live regression. A Candy distributor's real value
    (28.6%) must NOT be accepted as a Ferrero fact merely because the same
    cited source contains 'Bihar', 'Ferrero', and '28.6' somewhere each on
    their own -- the decomposed (Bihar, Ferrero) parts must co-occur with
    THIS value in one sentence, and they never do (28.6 only ever appears
    together with 'Candy', not 'Ferrero')."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (0)"], evidence_index)
    # Prove each ingredient really is present independently somewhere in
    # scope, so a pass here would prove the co-occurrence gap, not just
    # that the ingredients are missing outright.
    assert "Bihar" in scope and "Ferrero" in scope and "28.6" in scope

    claim = AnswerClaim(
        claim_text="For Bihar, the Ferrero category Out-of-Stock rate deviation was 28.6%.",
        claim_type="deviation",
        entity="Bihar Ferrero Category",
        metric="Out-of-Stock Rate",
        value=28.6,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" and "together as one fact" in i.detail for i in issues)
    # The fix must not resurrect the OLD literal-phrase failure mode either
    # -- this must fail for the right (misattribution) reason, not because
    # 'Bihar Ferrero Category' isn't a literal string.
    assert not any(i.issue_type == "unsupported_entity" for i in issues)


def test_composite_entity_wrong_delta_still_rejected():
    """D: an incorrect delta (15.6) for the mismatched 5.2-vs-28.6 pairing
    must still be rejected -- the composite-entity fix must not weaken the
    existing delta-vs-actual-difference check."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="For Bihar, the Ferrero category Out-of-Stock deviation was 28.6% vs. the state average of 5.2%, a gap of 15.6 points.",
        claim_type="deviation",
        entity="Bihar Ferrero Category",
        metric="Out-of-Stock Rate",
        value=28.6,
        comparison_value=5.2,
        delta=15.6,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_delta" for i in issues)


def test_composite_entity_without_category_suffix_word_still_decomposes():
    """A second live run of the same regression showed the model drop the
    trailing 'Category' word entirely and write just entity='Bihar Ferrero'
    -- the shorter, more common surface form must be caught the same way as
    'Bihar Ferrero Category', both to pass on the correct value and to
    reject a misattributed one."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)

    claim_ok = AnswerClaim(
        claim_text="For Bihar, the Ferrero category channel-level out-of-stock rate was 5.2%.",
        claim_type="factual_numeric",
        entity="Bihar Ferrero",
        metric="Out-of-Stock Rate",
        value=5.2,
        citations=["Sources (0)"],
    )
    assert validate_claim(claim_ok, evidence_index) == []

    claim_bad = AnswerClaim(
        claim_text="For Bihar, the Ferrero deviation was 28.6%.",
        claim_type="deviation",
        entity="Bihar Ferrero",
        metric="Out-of-Stock Rate",
        value=28.6,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim_bad, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" and "together as one fact" in i.detail for i in issues)


def test_composite_entity_with_hyphen_separator_still_decomposes():
    """A third live run of the same regression showed the model separate
    the two parts with a hyphen -- entity='Bihar - Ferrero' -- rather than
    a plain space. The trailing punctuation on the prefix must be trimmed,
    not treated as part of the state name, so the correct value still
    passes."""
    records = make_context_records(sources=BIHAR_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="In October 2025, the channel-level Out-of-Stock Rate for the Ferrero category in Bihar was 5.2%.",
        claim_type="factual_numeric",
        entity="Bihar - Ferrero",
        metric="Out-of-Stock Rate",
        value=5.2,
        citations=["Sources (0)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_real_literal_composite_entity_is_not_decomposed():
    """This corpus's own graph really does contain literal composite
    entities that happen to LOOK like the fabricated '<prefix> <Category>'
    shape -- e.g. 'ACV Ferrero', a real category-scoped-metric entity node
    -- and those must keep being checked by exact literal match, not
    weakened into loose (prefix, category) co-occurrence, since decomposing
    a real entity would let a wrong value (e.g. Candy's ACV, 27.7) launder
    through as long as 'ACV' and 'Ferrero' both appear somewhere in scope.
    Decomposition is only a fallback for when the literal phrase is absent
    from evidence altogether."""
    records = make_context_records(
        sources=[{
            "id": "1",
            "text": "ACV Ferrero was 30.0% in Bihar during October 2025. ACV Candy was 27.7% in Bihar during October 2025.",
        }]
    )
    evidence_index = _build_evidence_index(records)

    claim_ok = AnswerClaim(
        claim_text="ACV Ferrero was 30.0% in Bihar.",
        claim_type="factual_numeric",
        entity="ACV Ferrero",
        value=30.0,
        citations=["Sources (1)"],
    )
    assert validate_claim(claim_ok, evidence_index) == []

    claim_bad = AnswerClaim(
        claim_text="ACV Ferrero was 27.7% in Bihar.",
        claim_type="factual_numeric",
        entity="ACV Ferrero",
        value=27.7,
        citations=["Sources (1)"],
    )
    issues = validate_claim(claim_bad, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" and "together as one fact" in i.detail for i in issues)


def test_decompose_state_category_entity_helper():
    """Direct unit coverage of the decomposition helper itself: recognizes
    both the '<state> <Category>' and '<state> <Category> Category' shapes
    only when the category word is a real category name, and leaves every
    other entity shape alone."""
    assert _decompose_state_category_entity("Bihar Ferrero Category") == ("Bihar", "Ferrero")
    assert _decompose_state_category_entity("Bihar Candy Category") == ("Bihar", "Candy")
    assert _decompose_state_category_entity("Bihar Ferrero") == ("Bihar", "Ferrero")
    # A third live run showed a hyphen separator between the two parts --
    # the trailing punctuation must be trimmed off the prefix, not treated
    # as part of the state name (which would never match the literal
    # "State: Bihar" evidence phrasing).
    assert _decompose_state_category_entity("Bihar - Ferrero") == ("Bihar", "Ferrero")
    # Plain state/entity names, and a non-category last word, are left
    # alone (None) so callers fall back to normal literal-entity handling.
    assert _decompose_state_category_entity("Bihar") is None
    assert _decompose_state_category_entity("Zhang, Brooks and Miles Distributors") is None
    assert _decompose_state_category_entity("Bihar Service Category") is None
    assert _decompose_state_category_entity("Andhra Pradesh") is None


# ---------------------------------------------------------------------------
# Phase 8 regression, part 2: category cross-check via focus_categories.
# The real live index (not just the synthetic BIHAR_DISTRIBUTOR_DEVIATION_
# SOURCE fixture above) showed the actual failure mode is a DISTRIBUTOR-
# level claim whose own metric field never names a category at all (just
# "Out-of-Stock Rate", not "Ferrero Out-of-Stock Rate") -- the P0
# co-occurrence check (_primary_fact_supported) can't catch a cross-
# category misattribution from the claim's fields alone in that case, since
# nothing in the claim says which category it's about. This closes that gap
# by reading the category the CITED SENTENCE itself declares ("... for the
# Candy category ...") and rejecting a match against a category the
# question never asked about -- without requiring the model to always
# remember to qualify its metric field.
# ---------------------------------------------------------------------------

# A trimmed version of the real Bihar October 2025 pilot-index document (see
# the Phase 8b live forensic run) -- Zhang, Brooks and Miles' real Ferrero
# deviation, and Long, Anderson and Irwin's / Weaver-Sherman's real CANDY
# deviations, all reported with a bare "Out-of-Stock Rate" claim.metric
# (no category qualifier), exactly as the live model actually wrote it.
REAL_BIHAR_OCTOBER_SOURCE = [
    {
        "id": "0",
        "text": (
            "State: Bihar\nPeriod: October 2025\n\n"
            "Distributor-level deviations, October 2025: "
            "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on "
            "Out-of-Stock Rate for the Ferrero category in October 2025: 42.9% vs. the state average "
            "of 5.2%, a gap of 37.7 percentage points. "
            "Distributor Long, Anderson and Irwin Distributors showed a significant deviation on "
            "Out-of-Stock Rate for the Candy category in October 2025: 28.6% vs. the state average "
            "of 10.4%, a gap of 18.2 percentage points. "
            "Distributor Weaver-Sherman Distributors showed a significant deviation on "
            "Out-of-Stock Rate for the Candy category in October 2025: 28.6% vs. the state average "
            "of 10.4%, a gap of 18.2 percentage points. "
            "For the Ferrero category in Bihar during October 2025, the channel-level Out-of-Stock "
            "rate was 5.2%."
        ),
    },
]

Q1_QUESTION = (
    "For Bihar in October 2025, what was the channel-level Out-of-Stock Rate for the "
    "Ferrero category, and which distributor(s) showed a significant deviation on "
    "Out-of-Stock Rate for Ferrero that month?"
)


def test_extract_focus_categories_reads_category_named_in_question():
    """The question names 'Ferrero' twice -- focus_categories must be
    exactly {'Ferrero'}, and a question naming no category at all must
    yield an empty set (so the category cross-check never activates for
    ordinary, category-agnostic questions like a plain Service Level ask)."""
    assert _extract_focus_categories(Q1_QUESTION) == frozenset({"Ferrero"})
    assert _extract_focus_categories("What was Bihar's Service Level in October 2025?") == frozenset()
    assert _extract_focus_categories(None) == frozenset()


def test_sentence_categories_reads_for_the_x_category_phrase():
    """Direct unit coverage of the sentence-side category reader: both
    surface forms this corpus uses ('for the X category' and the bare
    '(X)' parenthetical), and no false positives on a state-level sentence
    that names no category at all."""
    assert _sentence_categories(
        "Distributor X showed a deviation on Out-of-Stock Rate for the Candy category: 28.6%."
    ) == frozenset({"Candy"})
    assert _sentence_categories("Out-of-Stock Rate (Ferrero) had no deviation this month.") == frozenset({"Ferrero"})
    assert _sentence_categories("Service Level for Bihar in October 2025 was 94.3%.") == frozenset()


def test_distributor_claim_with_generic_metric_field_still_accepts_correct_category():
    """The real live shape: claim.metric is plain 'Out-of-Stock Rate' (no
    category qualifier at all) -- Zhang, Brooks and Miles Distributors'
    genuine Ferrero deviation (42.9%) must still PASS when the question
    asks about Ferrero, even though the claim's own metric field never says
    so."""
    records = make_context_records(sources=REAL_BIHAR_OCTOBER_SOURCE)
    evidence_index = _build_evidence_index(records)
    focus = _extract_focus_categories(Q1_QUESTION)
    claim = AnswerClaim(
        claim_text="Zhang, Brooks and Miles Distributors reported an Out-of-Stock Rate of 42.9% in October 2025.",
        claim_type="deviation",
        entity="Zhang, Brooks and Miles Distributors",
        metric="Out-of-Stock Rate",
        value=42.9,
        citations=["Sources (0)"],
    )
    assert validate_claim(claim, evidence_index, focus_categories=focus) == []


def test_distributor_claim_with_generic_metric_field_rejects_wrong_category():
    """The actual live regression, reproduced with the claim shape the
    model really wrote (generic metric field, no category word anywhere in
    the claim). Long, Anderson and Irwin Distributors' 28.6% is a real
    value in the cited evidence -- but it's their CANDY deviation, not
    Ferrero. Without the category cross-check, entity + value + generic
    'Out-of-Stock Rate' metric words all co-occur in the Candy sentence, so
    the OLD P0 check alone would incorrectly pass this claim; the
    cross-check must catch what the claim's own fields cannot."""
    records = make_context_records(sources=REAL_BIHAR_OCTOBER_SOURCE)
    evidence_index = _build_evidence_index(records)
    focus = _extract_focus_categories(Q1_QUESTION)

    # Sanity check: prove the OLD (no focus_categories) check really would
    # pass this, so this test is actually exercising the new cross-check.
    assert validate_claim(
        AnswerClaim(
            claim_text="Long, Anderson and Irwin Distributors had an Out-of-Stock Rate of 28.6% in October 2025.",
            claim_type="deviation", entity="Long, Anderson and Irwin Distributors",
            metric="Out-of-Stock Rate", value=28.6, citations=["Sources (0)"],
        ),
        evidence_index,
    ) == []

    claim = AnswerClaim(
        claim_text="Long, Anderson and Irwin Distributors had an Out-of-Stock Rate of 28.6% in October 2025.",
        claim_type="deviation",
        entity="Long, Anderson and Irwin Distributors",
        metric="Out-of-Stock Rate",
        value=28.6,
        citations=["Sources (0)"],
    )
    issues = validate_claim(claim, evidence_index, focus_categories=focus)
    assert any(i.issue_type == "unsupported_numeric" and "together as one fact" in i.detail for i in issues)


def test_check_grounding_end_to_end_with_question_rejects_wrong_category_distributor():
    """End-to-end via check_grounding(..., question=...): an answer mixing
    one correct Ferrero claim with one misattributed Candy-as-Ferrero claim
    must fail grounding overall, and passing no question at all must
    reproduce the exact prior (pre-fix) behavior -- additive, not a
    behavior change for existing callers that don't pass a question."""
    records = make_context_records(sources=REAL_BIHAR_OCTOBER_SOURCE)
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
    with_question = check_grounding(answer_text, records, claims, None, Q1_QUESTION)
    assert with_question.status == "fail"
    assert any(i.claim_id == 1 for i in with_question.flagged_issues)
    assert not any(i.claim_id == 0 for i in with_question.flagged_issues)

    without_question = check_grounding(answer_text, records, claims, None)
    assert without_question.status == "pass"


# ---------------------------------------------------------------------------
# Phase 8c: cross-entity comparisons via AnswerClaim.comparison_entity.
# The Q5 live failure: the model sometimes fused two different states into
# one entity string (entity="Sikkim, Maharashtra") to express "Sikkim's
# Ferrero OOS was higher than Maharashtra's" -- that phrase never appears
# verbatim in evidence (state-level facts are always reported per-state,
# never fused), so it always failed the entity check even though the
# underlying comparison was fully supported by evidence. comparison_entity
# gives the model a field to name the SECOND entity properly, with its own
# comparison_value/comparison_period, validated by the same literal-first/
# decompose-fallback/co-occurrence pattern as the primary entity -- so a
# wrong-category or wrong-period value still can't be smuggled onto either
# side.
#
# Real Bihar/Maharashtra/Sikkim source text (see the Phase 8c live forensic
# run against the pilot index) -- Maharashtra's real Ferrero OOS is 6.5%
# (August 2024), Sikkim's real Ferrero OOS is 6.6% (April 2025), and
# Maharashtra's real Candy OOS is 7.8% (a wrong-category decoy value used
# below).
#
# Test-to-requirement map:
#   A: test_cross_entity_comparison_claim_passes
#   B: test_cross_entity_comparison_rejects_wrong_category_value
#   C: test_combined_entity_string_still_rejected_not_comparison_entity
#   D: test_distributor_deviation_against_comparison_entitys_state_average
#   E: covered by the full suite above continuing to pass unchanged
# ---------------------------------------------------------------------------

MAHARASHTRA_AUGUST_2024_SOURCE = [
    {
        "id": "1",
        "text": (
            "State: Maharashtra\nPeriod: August 2024\n\n"
            "For the Ferrero category (franchises: TicTac, Kinder_Joy) in Maharashtra during August 2024: "
            "Numeric Distribution was 23.2%, ACV was 23.2%, Out-of-Stock rate was 6.5%, and average Range "
            "Billing was 16.7%.\n\n"
            "For the Candy category (franchises: Candy_Franchise_1, Candy_Franchise_2) in Maharashtra "
            "during August 2024: Numeric Distribution was 20.8%, ACV was 20.8%, Out-of-Stock rate was "
            "7.8%, and average Range Billing was 16.6%."
        ),
    },
]

SIKKIM_APRIL_2025_SOURCE = [
    {
        "id": "15",
        "text": (
            "State: Sikkim\nPeriod: April 2025\n\n"
            "Distributor Hanson-Stewart Distributors showed a significant deviation on Out-of-Stock Rate "
            "for the Ferrero category in April 2025: 42.9% vs. the state average of 6.6%, a gap of 36.3 "
            "percentage points.\n\n"
            "For the Ferrero category (franchises: TicTac, Kinder_Joy) in Sikkim during April 2025: "
            "Numeric Distribution was 7.2%, ACV was 7.3%, Out-of-Stock rate was 6.6%, and average Range "
            "Billing was 15.0%."
        ),
    },
]


def test_cross_entity_comparison_claim_passes():
    """A: a properly-shaped comparison claim -- entity='Sikkim'/value=6.6,
    comparison_entity='Maharashtra'/comparison_value=6.5, each from its own
    period -- must PASS cleanly when both sides are genuinely supported by
    the (jointly cited) evidence."""
    records = make_context_records(sources=MAHARASHTRA_AUGUST_2024_SOURCE + SIKKIM_APRIL_2025_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Sikkim's Ferrero Out-of-Stock Rate (6.6%) was higher than Maharashtra's (6.5%).",
        claim_type="comparison",
        entity="Sikkim",
        comparison_entity="Maharashtra",
        metric="Out-of-Stock Rate",
        period="April 2025",
        comparison_period="August 2024",
        value=6.6,
        comparison_value=6.5,
        direction="higher",
        delta=0.1,
        citations=["Sources (1)", "Sources (15)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_cross_entity_comparison_rejects_wrong_category_value():
    """B: Maharashtra's real Candy Out-of-Stock Rate (7.8%) must NOT be
    accepted as its Ferrero figure merely because comparison_entity
    ('Maharashtra'), the value (7.8), the period ('August 2024'), and the
    generic metric words ('Out-of-Stock', 'Rate') ALL co-occur in one
    sentence -- Maharashtra's Candy paragraph states its own period and
    generic metric words too, so this is caught only by the SAME
    focus_categories cross-check that closes the equivalent gap for the
    primary entity (see _sentence_categories()'s docstring) -- the caller
    (check_grounding(), via pipeline.py's `question`) is responsible for
    supplying it, exactly as for a plain claim.entity."""
    records = make_context_records(sources=MAHARASHTRA_AUGUST_2024_SOURCE + SIKKIM_APRIL_2025_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (1)", "Sources (15)"], evidence_index)
    assert "Maharashtra" in scope and "7.8" in scope  # both present independently -- proves the co-occurrence gap

    # No direction word in claim_text -- keeps this test isolated to the
    # focus_categories/co-occurrence gap it targets, not entangled with the
    # separate claim-text direction-consistency check (see
    # test_claim_text_direction_contradiction_detected below).
    claim = AnswerClaim(
        claim_text="Sikkim's Ferrero Out-of-Stock Rate was 6.6%, compared with Maharashtra's 7.8%.",
        claim_type="comparison",
        entity="Sikkim",
        comparison_entity="Maharashtra",
        metric="Out-of-Stock Rate",
        period="April 2025",
        comparison_period="August 2024",
        value=6.6,
        comparison_value=7.8,
        citations=["Sources (1)", "Sources (15)"],
    )
    # Without focus_categories, entity+period+value+metric alone DOES
    # co-occur (Maharashtra's Candy paragraph states its own period and
    # generic metric words too) -- proving this is genuinely the category
    # cross-check's job, not the entity/period/value check's.
    assert validate_claim(claim, evidence_index) == []

    issues = validate_claim(claim, evidence_index, focus_categories=frozenset({"Ferrero"}))
    assert any(i.issue_type == "unsupported_comparison" for i in issues)


def test_combined_entity_string_still_rejected_not_comparison_entity():
    """C: the actual Q5 live failure -- entity='Sikkim, Maharashtra' (both
    states fused into one string) must still be rejected exactly as
    before; comparison_entity is the correct fix, not a reason to start
    accepting the fused-entity shape."""
    records = make_context_records(sources=MAHARASHTRA_AUGUST_2024_SOURCE + SIKKIM_APRIL_2025_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Sikkim, Maharashtra Ferrero Out-of-Stock Rate was 6.6%.",
        claim_type="factual_numeric",
        entity="Sikkim, Maharashtra",
        metric="Out-of-Stock Rate",
        value=6.6,
        citations=["Sources (1)", "Sources (15)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_entity" for i in issues)


def test_distributor_deviation_against_comparison_entitys_state_average():
    """D: Hanson-Stewart Distributors' real Ferrero OOS deviation (42.9%
    vs. Sikkim's 6.6% average, gap 36.3pp) -- a plain (non-comparison_
    entity) deviation claim -- must keep passing unaffected by the new
    comparison_entity machinery."""
    records = make_context_records(sources=SIKKIM_APRIL_2025_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Hanson-Stewart Distributors showed a Ferrero Out-of-Stock Rate deviation of 42.9%, a gap of 36.3pp.",
        claim_type="deviation",
        entity="Hanson-Stewart Distributors",
        metric="Out-of-Stock Rate",
        value=42.9,
        comparison_value=6.6,
        delta=36.3,
        citations=["Sources (15)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_same_entity_trend_comparison_still_uses_period_only_check():
    """E: a plain same-entity, two-period trend claim (comparison_entity
    unset) must keep working exactly as before -- the new comparison_entity
    branch must not interfere with (or replace) the pre-existing
    comparison_period-only co-occurrence check for this unrelated shape."""
    records = make_context_records(
        sources=[{
            "id": "0",
            "text": (
                "State: Bihar\nPeriod: October 2025\n\nService Level for Bihar in October 2025 was 94.3%.\n\n"
                "State: Bihar\nPeriod: November 2025\n\nService Level for Bihar in November 2025 was 92.9%."
            ),
        }]
    )
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Bihar's Service Level declined from 94.3% in October 2025 to 92.9% in November 2025.",
        claim_type="trend",
        entity="Bihar",
        metric="Service Level",
        period="November 2025",
        comparison_period="October 2025",
        value=92.9,
        comparison_value=94.3,
        direction="declined",
        delta=1.4,
        citations=["Sources (0)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_resolve_entity_reference_helper():
    """Direct unit coverage of the shared _resolve_entity_reference()
    helper: literal match wins when present, decomposition is the fallback
    only when the literal form is absent, and neither applies to a
    genuinely unsupported entity."""
    records = make_context_records(sources=MAHARASHTRA_AUGUST_2024_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (1)"], evidence_index)

    literal, composite = _resolve_entity_reference("Maharashtra", scope)
    assert literal is True and composite is None

    literal, composite = _resolve_entity_reference("Maharashtra Ferrero", scope)
    assert literal is False and composite == ("Maharashtra", "Ferrero")

    literal, composite = _resolve_entity_reference("Sikkim, Maharashtra", scope)
    assert literal is False and composite is None


def test_check_grounding_end_to_end_q5_cross_entity_comparison():
    """End-to-end via check_grounding(..., question=...): a correctly-shaped
    cross-entity comparison claim passes; the same claim with Maharashtra's
    wrong-category value fails, with `question` (naming Ferrero) as the
    only thing standing between the two -- proving pipeline.py's threading
    of `question` into check_grounding() is what makes this catch work."""
    records = make_context_records(sources=MAHARASHTRA_AUGUST_2024_SOURCE + SIKKIM_APRIL_2025_SOURCE)
    question = (
        "Compare Maharashtra's Ferrero Out-of-Stock Rate in August 2024 to Sikkim's Ferrero "
        "Out-of-Stock Rate in April 2025."
    )

    good_claim = AnswerClaim(
        claim_id=0, claim_text="Sikkim's Ferrero OOS (6.6%) was higher than Maharashtra's (6.5%).",
        claim_type="comparison", entity="Sikkim", comparison_entity="Maharashtra",
        metric="Out-of-Stock Rate", period="April 2025", comparison_period="August 2024",
        value=6.6, comparison_value=6.5, citations=["Sources (1)", "Sources (15)"],
    )
    good_answer_text = "Sikkim's Ferrero OOS (6.6%) was higher than Maharashtra's (6.5%) [Data: Sources (1); Sources (15)]."
    result = check_grounding(good_answer_text, records, [good_claim], None, question)
    assert result.status == "pass"

    bad_claim = AnswerClaim(
        claim_id=0, claim_text="Sikkim's Ferrero OOS (6.6%) was higher than Maharashtra's (7.8%).",
        claim_type="comparison", entity="Sikkim", comparison_entity="Maharashtra",
        metric="Out-of-Stock Rate", period="April 2025", comparison_period="August 2024",
        value=6.6, comparison_value=7.8, citations=["Sources (1)", "Sources (15)"],
    )
    bad_answer_text = "Sikkim's Ferrero OOS (6.6%) was higher than Maharashtra's (7.8%) [Data: Sources (1); Sources (15)]."
    result = check_grounding(bad_answer_text, records, [bad_claim], None, question)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_comparison" for i in result.flagged_issues)


# ---------------------------------------------------------------------------
# Q7 regression: Campbell PLC Distributors' real GPI Out-of-Stock Rate
# deviation (20.0%, gap 12.0pp) misattributed in the answer to Baxter,
# Thomas and Williams Distributors -- a real, different distributor with a
# real, different deviation that same period (Dropsize, 214.48 vs. 193.87).
# GUJARAT_APRIL_2026_SOURCE is verbatim real text from the pilot index's
# Gujarat_2026-04.txt document; GUJARAT_ENTITIES mirrors the two real
# distributor entity names retrieval would surface for this question.
# ---------------------------------------------------------------------------

GUJARAT_APRIL_2026_SOURCE = [
    {
        "id": "50",
        "text": (
            "State: Gujarat\nPeriod: April 2026\n\n"
            "Gujarat Zone 1 distributors: Baxter, Thomas and Williams Distributors, Diaz, Stewart and "
            "Patterson Distributors, Holmes LLC Distributors, Greene-Elliott Distributors, Campbell PLC "
            "Distributors.\n\n"
            "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
            "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap of 10.6 percent.\n"
            "Distributor Campbell PLC Distributors showed a significant deviation on Out-of-Stock Rate "
            "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
            "percentage points.\n"
            "Distributor Campbell PLC Distributors showed a significant deviation on Out-of-Stock Rate "
            "for the IPM category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
            "percentage points."
        ),
    }
]

GUJARAT_ENTITIES = [
    {"id": "e1", "entity": "Baxter, Thomas and Williams Distributors", "description": ""},
    {"id": "e2", "entity": "Campbell PLC Distributors", "description": ""},
]

# Real relationship-shaped text -- NO trailing period, matching the real
# pilot index (measured at 99.8% of relationships.parquet rows carrying no
# terminal punctuation at all). r2's description is a plausible DEVIATES_FROM
# edge shaped exactly like the real ones found for Baxter's June 2026
# deviations (this specific April/GPI one isn't itself a real edge in the
# pilot index -- the real graph never actually embeds this number in any
# relationship row -- but the shape, and critically the missing period, are
# both faithfully real).
RELATIONSHIP_NO_PERIOD_ROWS = [
    {
        "id": "r1",
        "source": "BAXTER, THOMAS AND WILLIAMS DISTRIBUTORS",
        "target": "GUJARAT",
        "description": "SERVES: Baxter, Thomas and Williams Distributors operates in Gujarat",
    },
    {
        "id": "r2",
        "source": "CAMPBELL PLC DISTRIBUTORS",
        "target": "GUJARAT GPI OUT-OF-STOCK RATE APRIL 2026",
        "description": (
            "DEVIATES_FROM: Campbell PLC Distributors' Out-of-Stock Rate (20.0%) was 12.0 percentage "
            "points above the Gujarat state average (8.0%) in April 2026 for the GPI category"
        ),
    },
]


def test_ensure_sentence_boundary_appends_period_only_when_missing():
    """Direct unit coverage of the Fix 1 helper: only ever ADDS a period
    when one (or !/?) isn't already there; never alters existing
    punctuation or strips content."""
    assert _ensure_sentence_boundary("no period here") == "no period here."
    assert _ensure_sentence_boundary("already ends.") == "already ends."
    assert _ensure_sentence_boundary("a question?") == "a question?"
    assert _ensure_sentence_boundary("an exclamation!") == "an exclamation!"
    assert _ensure_sentence_boundary("  trailing space   ") == "  trailing space."
    assert _ensure_sentence_boundary("") == ""


def test_multi_record_citation_boundary_prevents_cross_record_misattribution():
    """Fix 1 regression: relationship description text in this corpus has
    NO trailing punctuation in the overwhelming majority of rows. Before
    _ensure_sentence_boundary, citing TWO such records together let
    _resolve_citations's bare " ".join(parts) fuse them into one artificial
    "sentence", so an entity named in one cited record could co-occur with
    a completely different cited record's value/metric under
    _primary_fact_supported()'s per-sentence check. A claim wrongly
    attributing Campbell PLC Distributors' real GPI Out-of-Stock Rate
    figure to Baxter, Thomas and Williams Distributors, citing both
    (punctuation-less) records together, must be rejected; the same figure
    correctly attributed to Campbell must pass."""
    records = make_context_records(relationships=RELATIONSHIP_NO_PERIOD_ROWS)
    evidence_index = _build_evidence_index(records)

    scope = _resolve_citations(["Relationships (r1, r2)"], evidence_index)
    assert len(_split_sentences(scope)) >= 2, "citation join must insert a real sentence boundary between records"

    # claim_type deliberately left unset (infers to "factual_numeric" --
    # neither claim_text below says the word "deviation" literally, and the
    # synthetic relationship rows use "DEVIATES_FROM" as a relation label,
    # not the literal word "deviation") -- this test is isolated to the
    # entity+value+metric co-occurrence mechanism, not the separate
    # deviation-keyword check.
    wrong_claim = AnswerClaim(
        claim_text=(
            "Baxter, Thomas and Williams Distributors' Out-of-Stock Rate was 20.0% for the GPI "
            "category, a gap of 12.0 percentage points."
        ),
        entity="Baxter, Thomas and Williams Distributors",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=20.0,
        citations=["Relationships (r1, r2)"],
    )
    issues = validate_claim(wrong_claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)

    right_claim = AnswerClaim(
        claim_text=(
            "Campbell PLC Distributors' Out-of-Stock Rate was 20.0% for the GPI category, "
            "a gap of 12.0 percentage points."
        ),
        entity="Campbell PLC Distributors",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=20.0,
        citations=["Relationships (r1, r2)"],
    )
    assert validate_claim(right_claim, evidence_index) == []


def test_known_distributor_entity_names_scoped_to_this_querys_context():
    """_known_distributor_entity_names never hardcodes a name -- it only
    ever returns entities actually present in THIS query's own retrieved
    context_records, filtered to ones whose name contains 'Distributor'."""
    records = make_context_records(
        entities=GUJARAT_ENTITIES + [{"id": "e3", "entity": "Gujarat", "description": ""}]
    )
    names = _known_distributor_entity_names(records)
    assert set(names) == {"Baxter, Thomas and Williams Distributors", "Campbell PLC Distributors"}

    assert _known_distributor_entity_names(make_context_records()) == []


def test_scan_entity_numeric_claims_flags_misattributed_deviation_sentence():
    """Fix 2 regression: a deviation sentence in raw answer PROSE that
    misattributes a real distributor's real figure to a different named
    distributor, with NO structured claim covering it at all, must still
    be caught by the new prose-level safety net."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    answer_text = (
        "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, "
        "a gap of 12.0 percentage points [Data: Sources (50)]."
    )
    issues = scan_entity_numeric_claims(answer_text, records, claims=None)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


def test_scan_entity_numeric_claims_passes_correct_attribution():
    """Same sentence shape, correctly attributed to Campbell -- must not be
    flagged."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    answer_text = (
        "Distributor Campbell PLC Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, "
        "a gap of 12.0 percentage points [Data: Sources (50)]."
    )
    assert scan_entity_numeric_claims(answer_text, records, claims=None) == []


def test_scan_entity_numeric_claims_skips_pair_already_covered_by_a_claim():
    """A structured claim already reports this exact (entity, value) pair
    -- validate_claim() already ran the stricter check against it (and,
    for a wrong pairing, already flags it independently), so the prose
    scanner must not re-flag the same sentence a second time.

    The sentence states TWO percent figures for Baxter (20.0% and the 8.0%
    state average) -- since the A1 "vs." sentence-boundary fix, the scanner
    correctly sees both numbers in this one sentence (previously the "vs."
    mis-split hid the second from it entirely), so both must be covered by
    a structured claim for this "already covered, don't re-flag" test to
    isolate what it's actually testing."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    answer_text = (
        "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, "
        "a gap of 12.0 percentage points [Data: Sources (50)]."
    )
    claims = [
        AnswerClaim(
            entity="Baxter, Thomas and Williams Distributors",
            value=20.0,
            metric="Out-of-Stock Rate",
            citations=["Sources (50)"],
        ),
        AnswerClaim(
            entity="Baxter, Thomas and Williams Distributors",
            value=8.0,
            metric="Out-of-Stock Rate",
            citations=["Sources (50)"],
        ),
    ]
    assert scan_entity_numeric_claims(answer_text, records, claims=claims) == []


def test_scan_entity_numeric_claims_ignores_non_deviation_sentences():
    """The scanner only checks '...deviation...' sentences -- a roster or
    narrative sentence that merely names a distributor near an unrelated
    percentage must never be flagged."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    answer_text = (
        "Gujarat Zone 1 has five distributors including Baxter, Thomas and Williams Distributors, "
        "and roughly 20.0% of the state's outlets are served by this zone."
    )
    assert scan_entity_numeric_claims(answer_text, records, claims=None) == []


def test_scan_entity_numeric_claims_no_known_entities_returns_empty():
    """No 'entities' table with any 'Distributor'-named row in
    context_records -- scanner must return no issues, never raise."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)
    answer_text = (
        "Distributor Ghost Co Distributors showed a significant deviation on Out-of-Stock Rate: 99.0%."
    )
    assert scan_entity_numeric_claims(answer_text, records, claims=None) == []


def test_check_grounding_end_to_end_q7_misattributed_distributor_deviation_fails():
    """End-to-end via check_grounding(): the exact Q7 shape -- Campbell PLC
    Distributors' real GPI Out-of-Stock Rate deviation (20.0%, gap 12.0pp)
    misattributed to Baxter, Thomas and Williams Distributors, backed by a
    matching (also wrong) structured claim -- must fail overall; the
    correctly-attributed version must pass."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)

    wrong_text = (
        "Baxter, Thomas and Williams Distributors showed a significant deviation on Out-of-Stock Rate "
        "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
        "percentage points [Data: Sources (50)]."
    )
    wrong_claim = AnswerClaim(
        claim_id=0, claim_text=wrong_text, claim_type="deviation",
        entity="Baxter, Thomas and Williams Distributors", metric="Out-of-Stock Rate",
        period="April 2026", value=20.0, delta=12.0, citations=["Sources (50)"],
    )
    result = check_grounding(wrong_text, records, [wrong_claim])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_numeric" for i in result.flagged_issues)

    right_text = (
        "Campbell PLC Distributors showed a significant deviation on Out-of-Stock Rate "
        "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
        "percentage points [Data: Sources (50)]."
    )
    right_claim = AnswerClaim(
        claim_id=0, claim_text=right_text, claim_type="deviation",
        entity="Campbell PLC Distributors", metric="Out-of-Stock Rate",
        period="April 2026", value=20.0, delta=12.0, citations=["Sources (50)"],
    )
    result = check_grounding(right_text, records, [right_claim])
    assert result.status == "pass"


def test_primary_claim_correct_value_but_wrong_period_fails():
    """Q7 hardening, isolated to the period dimension: Campbell PLC
    Distributors' real GPI Out-of-Stock Rate deviation is 20.0% in April
    2026 (gap 12.0pp); a DIFFERENT month's deviation for the same
    distributor/metric/category is 35.0% in June 2026 (gap 27.0pp). A claim
    that takes the real April VALUE (20.0) but labels it June -- a period
    that IS genuinely mentioned elsewhere in the cited scope, just for a
    different fact -- must be rejected: before requiring period to
    co-occur with entity+value+metric (see validate_claim()'s primary
    co-occurrence block), the old independent-per-field checks (is 20.0
    anywhere in scope? is 'June 2026' anywhere in scope?) would BOTH pass
    on their own, letting this slip through with no issue raised at all."""
    sources = [
        {
            "id": "70",
            "text": (
                "State: Gujarat\nPeriod: April 2026\n\nDistributor Campbell PLC Distributors showed a "
                "significant deviation on Out-of-Stock Rate for the GPI category in April 2026: 20.0% "
                "vs. the state average of 8.0%, a gap of 12.0 percentage points."
            ),
        },
        {
            "id": "71",
            "text": (
                "State: Gujarat\nPeriod: June 2026\n\nDistributor Campbell PLC Distributors showed a "
                "significant deviation on Out-of-Stock Rate for the GPI category in June 2026: 35.0% "
                "vs. the state average of 8.0%, a gap of 27.0 percentage points."
            ),
        },
    ]
    records = make_context_records(sources=sources)
    evidence_index = _build_evidence_index(records)
    scope = _resolve_citations(["Sources (70)", "Sources (71)"], evidence_index)
    # Confirm the OLD independent checks alone would have missed this: 20.0
    # appears somewhere in scope (April's real fact), and 'June 2026'
    # appears somewhere in scope (a different, unrelated fact) -- just
    # never together.
    assert "20.0" in scope and "June 2026" in scope

    claim = AnswerClaim(
        claim_text=(
            "Campbell PLC Distributors' GPI Out-of-Stock Rate deviation was 20.0% in June 2026, "
            "a gap of 12.0 percentage points."
        ),
        claim_type="deviation",
        entity="Campbell PLC Distributors",
        metric="Out-of-Stock Rate",
        period="June 2026",
        value=20.0,
        delta=12.0,
        citations=["Sources (70)", "Sources (71)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)

    # Sanity: the SAME claim with the correct period passes cleanly.
    correct_claim = AnswerClaim(
        claim_text=(
            "Campbell PLC Distributors' GPI Out-of-Stock Rate deviation was 20.0% in April 2026, "
            "a gap of 12.0 percentage points."
        ),
        claim_type="deviation",
        entity="Campbell PLC Distributors",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=20.0,
        delta=12.0,
        citations=["Sources (70)", "Sources (71)"],
    )
    assert validate_claim(correct_claim, evidence_index) == []


# ---------------------------------------------------------------------------
# Audit fix A1: _split_sentences() must not treat "vs." as a sentence
# boundary. The corpus's standard deviation-sentence template ("X% vs. the
# state average of Y%, a gap of Z percentage points.") was being split at
# "vs.", severing a distributor's own value from its state-average/gap
# figures within what is really one fact-sentence -- which let a
# comparison_value be silently swapped in from a DIFFERENT distributor's
# sentence in the same cited record. Closing this required two changes:
#   1. _split_sentences() itself (the regex fix).
#   2. A new same-entity/same-period comparison_value co-occurrence check
#      in validate_claim() (mirroring the existing claim.value check) --
#      without it, comparison_value in this claim shape (comparison_entity
#      and comparison_period both unset -- the ordinary "distributor vs.
#      state average" deviation) was never sentence-scoped at all, so
#      fixing #1 alone wasn't sufficient to catch the misattribution.
# ---------------------------------------------------------------------------

TWO_DISTRIBUTOR_DEVIATION_SOURCE = [
    {
        "id": "3",
        "text": (
            "State: Bihar\nPeriod: November 2025\n\n"
            "Distributor-level deviations, November 2025: "
            "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on Service Level "
            "in November 2025: 82.6% vs. the state average of 92.9%, a gap of 10.4 percentage points. "
            "Distributor Landry Ltd Distributors showed a significant deviation on Service Level "
            "in November 2025: 70.0% vs. the state average of 88.0%, a gap of 18.0 percentage points."
        ),
    }
]


def test_vs_state_average_sentence_is_not_split_at_vs():
    """1: the standard '<value>% vs. the state average of <value>%, a gap
    of <value> percentage points.' template must remain ONE sentence, not
    two fragments split at 'vs.'."""
    text = (
        "Distributor Zhang, Brooks and Miles Distributors showed a significant deviation on Service Level "
        "in November 2025: 82.6% vs. the state average of 92.9%, a gap of 10.4 percentage points."
    )
    sentences = _split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_vs_split_fix_preserves_normal_sentence_boundaries():
    """Regular '.'/'!'/'?' boundaries must still split normally -- the fix
    is scoped to 'vs.' specifically, not a general loosening of the
    splitter."""
    text = "Service Level declined. Distributors were affected! Why did this happen?"
    assert _split_sentences(text) == [
        "Service Level declined.",
        "Distributors were affected!",
        "Why did this happen?",
    ]


def test_two_distributor_deviation_sentences_each_split_correctly():
    """Two back-to-back distributor-deviation sentences (the real
    multi-distributor document shape) must split into exactly two
    sentences -- one per distributor -- not four fragments at each 'vs.'."""
    records = make_context_records(sources=TWO_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    scope = evidence_index["sources"]["3"]
    sentences = [s for s in _split_sentences(scope) if "Distributor" in s and "deviation" in s]
    assert len(sentences) == 2
    assert "Zhang, Brooks and Miles" in sentences[0] and "92.9" in sentences[0]
    assert "Landry Ltd" in sentences[1] and "88.0" in sentences[1]


def test_cross_distributor_comparison_value_misattribution_is_rejected():
    """2: the previously reproduced bug -- a claim correctly attributes
    Zhang's own value (82.6%) but reports Landry's state average (88.0%)
    as if it were Zhang's comparison figure. Both numbers are genuinely
    present in the cited record (each is a real, correctly-stated fact --
    just for a DIFFERENT distributor), so this must fail via the
    comparison-side co-occurrence check, not the plain 'does this number
    appear anywhere' check."""
    records = make_context_records(sources=TWO_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Zhang, Brooks and Miles Distributors Service Level was 82.6% vs. the state average of 88.0%.",
        claim_type="deviation",
        entity="Zhang, Brooks and Miles Distributors",
        metric="Service Level",
        period="November 2025",
        value=82.6,
        comparison_value=88.0,  # Landry's real state average, not Zhang's
        citations=["Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_comparison" and i.term == "88.0" for i in issues)


def test_correct_distributor_state_average_attribution_still_passes():
    """3: the SAME claim shape, correctly attributing Zhang's own state
    average (92.9%), must still pass cleanly."""
    records = make_context_records(sources=TWO_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Zhang, Brooks and Miles Distributors Service Level was 82.6% vs. the state average of 92.9%.",
        claim_type="deviation",
        entity="Zhang, Brooks and Miles Distributors",
        metric="Service Level",
        period="November 2025",
        value=82.6,
        comparison_value=92.9,
        citations=["Sources (3)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_other_distributor_correct_attribution_also_passes():
    """Same fixture, the OTHER distributor's own correct value/state
    average -- proves the fix isn't one-sided (only checking the first
    distributor in the record)."""
    records = make_context_records(sources=TWO_DISTRIBUTOR_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Landry Ltd Distributors Service Level was 70.0% vs. the state average of 88.0%.",
        claim_type="deviation",
        entity="Landry Ltd Distributors",
        metric="Service Level",
        period="November 2025",
        value=70.0,
        comparison_value=88.0,
        citations=["Sources (3)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_check_grounding_end_to_end_rejects_cross_distributor_state_average_swap():
    """4: end-to-end via check_grounding() -- the misattributed comparison
    value must fail the overall check, and the correctly-attributed
    version must pass."""
    records = make_context_records(sources=TWO_DISTRIBUTOR_DEVIATION_SOURCE)
    wrong_claim = AnswerClaim(
        claim_id=0,
        claim_text="Zhang, Brooks and Miles Distributors Service Level was 82.6% vs. the state average of 88.0%.",
        claim_type="deviation", entity="Zhang, Brooks and Miles Distributors", metric="Service Level",
        period="November 2025", value=82.6, comparison_value=88.0, citations=["Sources (3)"],
    )
    result = check_grounding(wrong_claim.claim_text, records, [wrong_claim])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_comparison" for i in result.flagged_issues)

    right_claim = AnswerClaim(
        claim_id=0,
        claim_text="Zhang, Brooks and Miles Distributors Service Level was 82.6% vs. the state average of 92.9%.",
        claim_type="deviation", entity="Zhang, Brooks and Miles Distributors", metric="Service Level",
        period="November 2025", value=82.6, comparison_value=92.9, citations=["Sources (3)"],
    )
    result = check_grounding(right_claim.claim_text, records, [right_claim])
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# Audit fix A3: Dropsize's "gap" is a RELATIVE percent of the comparison
# value (see build_documents.py's find_deviating_distributors() "both"
# branch), not a raw-unit difference like every other metric's
# percentage-point gap. validate_claim()'s delta check must compare a
# Dropsize claim's delta against the relative-percent figure, while every
# other metric keeps the original raw-difference comparison unchanged.
# ---------------------------------------------------------------------------

DROPSIZE_DEVIATION_SOURCE = [
    {
        "id": "50",
        "text": (
            "State: Gujarat\nPeriod: April 2026\n\n"
            "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
            "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap of 10.6 percent."
        ),
    }
]


def test_dropsize_correct_relative_percent_delta_passes():
    """1: a Dropsize claim whose delta faithfully restates the evidence's
    own relative-percent gap (10.6) must pass, even though the raw unit
    difference (20.61) is a completely different number."""
    records = make_context_records(sources=DROPSIZE_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text=(
            "Baxter, Thomas and Williams Distributors Dropsize was 214.48 vs. the state average of "
            "193.87, a gap of 10.6 percent."
        ),
        claim_type="deviation",
        entity="Baxter, Thomas and Williams Distributors",
        metric="Dropsize",
        period="April 2026",
        value=214.48,
        comparison_value=193.87,
        delta=10.6,
        citations=["Sources (50)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_dropsize_incorrect_delta_fails():
    """2: a Dropsize claim whose delta does NOT match the evidence's own
    relative-percent gap (nor the raw difference) must still fail."""
    records = make_context_records(sources=DROPSIZE_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text=(
            "Baxter, Thomas and Williams Distributors Dropsize was 214.48 vs. the state average of "
            "193.87, a gap of 99.0 percent."
        ),
        claim_type="deviation",
        entity="Baxter, Thomas and Williams Distributors",
        metric="Dropsize",
        period="April 2026",
        value=214.48,
        comparison_value=193.87,
        delta=99.0,
        citations=["Sources (50)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_delta" for i in issues)


def test_dropsize_raw_unit_difference_as_delta_is_rejected():
    """Guards against a regression in the other direction: if a Dropsize
    claim's delta is the RAW unit difference (20.61) rather than the
    evidence's own relative-percent gap (10.6), it must still fail -- the
    fix teaches the checker Dropsize's actual semantics, it doesn't just
    loosen the tolerance."""
    records = make_context_records(sources=DROPSIZE_DEVIATION_SOURCE)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Baxter, Thomas and Williams Distributors Dropsize was 214.48 vs. 193.87, a gap of 20.61.",
        claim_type="deviation",
        entity="Baxter, Thomas and Williams Distributors",
        metric="Dropsize",
        period="April 2026",
        value=214.48,
        comparison_value=193.87,
        delta=20.61,
        citations=["Sources (50)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_delta" for i in issues)


def test_non_dropsize_delta_validation_unchanged_correct_case():
    """3a: a non-Dropsize (percentage-point) metric's correct delta --
    still the plain raw-difference comparison, unaffected by the Dropsize
    branch."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined by 1.4 percentage points from October to November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3, direction="declined", delta=1.4,
        citations=["Sources (0)", "Sources (3)"],
    )
    assert validate_claim(claim, evidence_index) == []


def test_non_dropsize_delta_validation_unchanged_incorrect_case():
    """3b: a non-Dropsize metric's incorrect delta (a relative-percent-
    shaped number substituted for the true percentage-point difference)
    must still fail via the raw-difference comparison, exactly as before."""
    records = make_context_records(sources=BIHAR_OCT_NOV_SOURCES)
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="Service Level declined by 1.5% from October to November 2025.",
        claim_type="trend", entity="Bihar", metric="Service Level",
        period="November 2025", comparison_period="October 2025",
        value=92.9, comparison_value=94.3, direction="declined",
        delta=1.5 / 94.3 * 100,  # a RELATIVE percent (~1.59), not the pp difference (1.4)
        citations=["Sources (0)", "Sources (3)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_delta" for i in issues)


# ---------------------------------------------------------------------------
# Live Q7 investigation fix: scan_entity_numeric_claims()'s known-distributor
# list must not depend exclusively on context_records["entities"]. A live
# run found GraphRAG's entity-vector retrieval and its source/text-unit
# retrieval surface DIFFERENT distributor sets for the same query --
# context_records["entities"] contained neither "Baxter, Thomas and
# Williams Distributors" nor "Campbell PLC Distributors" even though
# context_records["sources"] genuinely named both (see
# GUJARAT_APRIL_2026_SOURCE above, which is verbatim real pilot-index
# text). That left the prose-fallback scanner blind to a real,
# evidence-provable misattribution whenever structured claim extraction
# didn't cover it. _distributor_names_from_sources() closes this by also
# reading distributor names directly out of the cited source text itself.
# ---------------------------------------------------------------------------


def test_distributor_names_from_sources_reads_names_directly_from_source_text():
    """Direct unit coverage: both distributors are recovered from the
    source text alone, with no entities table involved at all."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)
    names = _distributor_names_from_sources(records)
    assert names == {"Baxter, Thomas and Williams Distributors", "Campbell PLC Distributors"}


def test_known_distributor_entity_names_falls_back_to_source_text_when_entities_table_empty():
    """1/2: cited source contains Distributor A (Baxter) and Distributor B
    (Campbell); context_records["entities"] does NOT contain either (empty
    table, the exact real Q7 condition) -- both must still end up in the
    known-distributor list via the source-text fallback."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)  # no entities at all
    names = _known_distributor_entity_names(records)
    assert "Baxter, Thomas and Williams Distributors" in names
    assert "Campbell PLC Distributors" in names


def test_known_distributor_entity_names_unions_entities_table_and_source_text():
    """The entities-table channel and the source-text channel are additive,
    not either/or -- a name present in ONLY the entities table (not named
    in this particular source text) is still recognized alongside names
    recovered from the source text."""
    records = make_context_records(
        sources=GUJARAT_APRIL_2026_SOURCE,
        entities=[{"id": "x1", "entity": "Holmes LLC Distributors", "description": ""}],
    )
    names = _known_distributor_entity_names(records)
    assert "Holmes LLC Distributors" in names
    assert "Baxter, Thomas and Williams Distributors" in names
    assert "Campbell PLC Distributors" in names


def test_scan_entity_numeric_claims_catches_misattribution_with_empty_entities_table():
    """3/4: prose contains a wrong numeric attribution to Distributor A
    (Baxter, GPI, 20.0%, 12.0pp -- all of which are really Campbell's) --
    with context_records["entities"] empty, this must still be detected
    and rejected, not silently pass because the fallback scanner had no
    known distributor names to check against."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)
    wrong_text = (
        "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, "
        "a gap of 12.0 percentage points [Data: Sources (50)]."
    )
    issues = scan_entity_numeric_claims(wrong_text, records, claims=None)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


def test_scan_entity_numeric_claims_passes_correct_attribution_with_empty_entities_table():
    """Companion positive case, same empty-entities-table condition:
    Campbell + GPI + OOS + April 2026 + 20.0% + 12.0pp is the real,
    correctly-attributed fact and must still pass cleanly."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)
    right_text = (
        "Distributor Campbell PLC Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, "
        "a gap of 12.0 percentage points [Data: Sources (50)]."
    )
    assert scan_entity_numeric_claims(right_text, records, claims=None) == []


def test_check_grounding_end_to_end_q7_live_path_with_empty_claims_and_entities():
    """End to end via check_grounding(), reproducing the EXACT live Q7
    condition: claims=[] (structured extraction returned nothing, as
    observed live) AND context_records["entities"] empty (the real
    retrieval-mismatch condition). Baxter's misattribution must still fail;
    Campbell's correct attribution must still pass -- preserving the
    existing correct Q7 behavior under the harder, live-accurate
    conditions this fix targets."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE)  # no entities

    wrong_text = (
        "Baxter, Thomas and Williams Distributors showed a significant deviation on Out-of-Stock Rate "
        "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
        "percentage points [Data: Sources (50)]."
    )
    result = check_grounding(wrong_text, records, claims=[])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_numeric" for i in result.flagged_issues)

    right_text = (
        "Campbell PLC Distributors showed a significant deviation on Out-of-Stock Rate "
        "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
        "percentage points [Data: Sources (50)]."
    )
    result = check_grounding(right_text, records, claims=[])
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# Live investigation fix: unsupported abbreviation/category-code
# definitions ("GPI stands for General Product Inventory") must not pass
# grounding. Neither the indexed corpus (build_documents.py's templates)
# nor any business-knowledge/ontology doc in this project ever states what
# GPI or IPM stand for -- confirmed live: the model fabricated both
# ("General Product Inventory", "Integrated Pest Management") with
# claims=[] and grounding=pass. scan_definition_language() is the
# deterministic safety net, mirroring scan_causal_language()'s existing
# pattern: flag any sentence asserting what a term "stands for"/"means"
# unless the cited (or whole, as fallback) evidence ALSO uses that same
# category of definitional language -- which, for this corpus, it never
# does, so any such sentence is correctly unsupported by construction.
# ---------------------------------------------------------------------------


def test_gpi_fabricated_definition_is_flagged():
    """1: 'What does GPI stand for?' -- a fabricated expansion, with no
    definitional language anywhere in the retrieved evidence, must be
    flagged as unsupported."""
    records = make_context_records(
        sources=[{"id": "22", "text": "State: Gujarat\nPeriod: April 2026\n\nFor the GPI category in Gujarat during April 2026: Numeric Distribution was 66.7%."}]
    )
    answer = 'The GPI category stands for "General Product Inventory" [Data: Sources (22)].'
    issues = scan_definition_language(answer, records)
    assert any(i.issue_type == "unsupported_definition" for i in issues)


def test_ipm_fabricated_definition_is_flagged():
    """2: 'What does IPM stand for?' -- same shape, the real live
    fabrication ('Integrated Pest Management')."""
    records = make_context_records(
        sources=[{"id": "22", "text": "State: Gujarat\nPeriod: April 2026\n\nFor the IPM category (franchises: Marlboro) in Gujarat during April 2026: Numeric Distribution was 75.0%."}]
    )
    answer = "IPM stands for Integrated Pest Management [Data: Sources (22)]."
    issues = scan_definition_language(answer, records)
    assert any(i.issue_type == "unsupported_definition" for i in issues)


def test_fabricated_definition_fails_check_grounding_end_to_end():
    """3: unsupported fabricated expansions must not pass grounding --
    end-to-end via check_grounding(), replaying the real live answer text
    for both GPI and IPM in one response, with claims=[] (exactly as
    observed live: the model self-reported no structured claim for
    either)."""
    records = make_context_records(
        sources=[{
            "id": "22",
            "text": (
                "State: Gujarat\nPeriod: April 2026\n\n"
                "For the GPI category (franchises: GPI_Franchise_1) in Gujarat during April 2026: "
                "Numeric Distribution was 66.7%.\n"
                "For the IPM category (franchises: Marlboro) in Gujarat during April 2026: "
                "Numeric Distribution was 75.0%."
            ),
        }]
    )
    answer = (
        'The GPI category stands for "General Product Inventory" [Data: Sources (22)]. '
        "IPM stands for Integrated Pest Management [Data: Sources (22)]."
    )
    result = check_grounding(answer, records, claims=[])
    assert result.status == "fail"
    definition_issues = [i for i in result.flagged_issues if i.issue_type == "unsupported_definition"]
    assert len(definition_issues) == 2


def test_definition_claim_supported_by_evidence_passes():
    """Sanity/negative case: if the cited evidence itself genuinely DOES
    use 'stands for'-shaped definitional language, the check must not
    reject it -- proves this is a real grounding check, not a blanket ban
    on the phrase."""
    records = make_context_records(
        sources=[{"id": "1", "text": "State: Test\nPeriod: January 2025\n\nIn this report, ACV stands for Available Capable Volume."}]
    )
    answer = "ACV stands for Available Capable Volume [Data: Sources (1)]."
    issues = scan_definition_language(answer, records)
    assert issues == []


def test_non_definitional_sentences_are_never_flagged():
    """Plain factual/deviation sentences (no 'stands for'/'means'-shaped
    language at all) must never trigger this scanner."""
    records = make_context_records(
        sources=[{"id": "22", "text": "State: Gujarat\nPeriod: April 2026\n\nOut-of-Stock rate was 8.0%."}]
    )
    answer = "The GPI category's Out-of-Stock rate was 8.0% in April 2026 [Data: Sources (22)]."
    assert scan_definition_language(answer, records) == []


# ---------------------------------------------------------------------------
# GPIL Knowledge Layer (2026-08-23): scan_glossary_term_misuse().
#
# The GPIL Knowledge Layer (knowledge_layer.py) fixes the original GPI/IPM
# fabrication bug above by merging a real, citable glossary row into
# context_records["sources"] BEFORE generation whenever the question names
# a known term (see pipeline.py's _augment_context_with_glossary()). That
# real evidence is what lets a CORRECT GPI/IPM answer pass
# scan_definition_language() at all -- but it also means a real,
# resolvable "glossary-gpi"/"glossary-ipm" row now always exists once the
# term is in scope, so a FABRICATED wrong expansion that happens to cite
# that real id would slip past scan_definition_language() too (it only
# checks whether SOME definitional-style text exists in the cited scope,
# never whether it says what the answer claims). scan_glossary_term_misuse()
# closes that specific gap with a small, citation-independent blocklist of
# expansions already known to be wrong.
# ---------------------------------------------------------------------------


def make_glossary_records(question: str) -> dict:
    """context_records["sources"] containing exactly the glossary row(s)
    knowledge_layer.build_glossary_source_rows() would produce for
    `question` -- i.e. what pipeline.py would have merged in before
    generation ran for this exact question."""
    return make_context_records(sources=build_glossary_source_rows(question))


def test_scan_glossary_term_misuse_flags_general_product_inventory():
    issues = scan_glossary_term_misuse("GPI stands for General Product Inventory.")
    assert len(issues) == 1
    assert issues[0].issue_type == "wrong_glossary_expansion"
    assert issues[0].term == "GPI / GPIL"


def test_scan_glossary_term_misuse_flags_integrated_pest_management():
    issues = scan_glossary_term_misuse("IPM stands for Integrated Pest Management.")
    assert len(issues) == 1
    assert issues[0].issue_type == "wrong_glossary_expansion"
    assert issues[0].term == "IPM"


def test_scan_glossary_term_misuse_is_case_insensitive():
    issues = scan_glossary_term_misuse("gpi stands for general product inventory.")
    assert len(issues) == 1


def test_scan_glossary_term_misuse_ignores_correct_gpil_specific_answers():
    issues = scan_glossary_term_misuse("GPI stands for GPIL's own cigarette brand portfolio.")
    assert issues == []


def test_scan_glossary_term_misuse_ignores_unrelated_text():
    assert scan_glossary_term_misuse("Bihar's Productivity fell to 85.5% in October 2025.") == []


def test_wrong_gpi_expansion_fails_check_grounding_even_when_citing_the_real_glossary_row():
    """The specific gap this scanner closes: a fabricated wrong definition
    that cites the REAL, resolvable "glossary-gpi" id (which
    scan_definition_language() alone would accept, since that row's own
    text genuinely IS definitional-style language) must still fail
    grounding overall."""
    records = make_glossary_records("What is GPI?")
    answer = "GPI stands for General Product Inventory [Data: Sources (glossary-gpi)]."
    result = check_grounding(answer, records, claims=[], question="What is GPI?")
    assert result.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in result.flagged_issues)


def test_wrong_ipm_expansion_fails_check_grounding_even_when_citing_the_real_glossary_row():
    records = make_glossary_records("What is IPM?")
    answer = "IPM stands for Integrated Pest Management [Data: Sources (glossary-ipm)]."
    result = check_grounding(answer, records, claims=[], question="What is IPM?")
    assert result.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in result.flagged_issues)


def test_correct_gpi_answer_citing_the_real_glossary_row_passes_check_grounding():
    records = make_glossary_records("What is GPI?")
    answer = "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]."
    result = check_grounding(answer, records, claims=[], question="What is GPI?")
    assert result.status == "pass"


def test_correct_ipm_answer_citing_the_real_glossary_row_passes_check_grounding():
    records = make_glossary_records("What is IPM?")
    answer = "IPM stands for the Marlboro product category GPIL sells under license [Data: Sources (glossary-ipm)]."
    result = check_grounding(answer, records, claims=[], question="What is IPM?")
    assert result.status == "pass"


def test_correct_definition_for_a_term_absent_from_context_still_passes_when_uncited():
    """Ferrero/Candy definitions rarely use "stands for" phrasing (they
    aren't abbreviations) -- scan_definition_language() only ever fires on
    that specific high-precision pattern, so an ordinary descriptive
    sentence about Ferrero must never be flagged at all, cited or not."""
    records = make_glossary_records("What is Ferrero?")
    answer = "Ferrero is one of GPIL's four tracked product categories, the licensed confectionery line."
    result = check_grounding(answer, records, claims=[], question="What is Ferrero?")
    assert result.status == "pass"


def test_mixed_question_correct_glossary_and_correct_kpi_claim_both_pass():
    """'What is GPI and how did it perform in Gujarat in April 2026?' --
    the glossary sentence and the KPI sentence must both independently
    pass grounding in the same answer."""
    rows = build_glossary_source_rows("What is GPI?")
    rows.append({
        "id": "22",
        "text": "State: Gujarat\nPeriod: April 2026\n\nFor the GPI category (franchises: GPI_Franchise_1) in Gujarat during April 2026: Numeric Distribution was 66.7%.",
    })
    records = make_context_records(sources=rows)
    answer = (
        "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]. "
        "In Gujarat during April 2026, GPI's Numeric Distribution was 66.7% [Data: Sources (22)]."
    )
    result = check_grounding(answer, records, claims=[], question="What is GPI and how did it perform in Gujarat in April 2026?")
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# Q4 fix proof: a claim citing Sources(23) -- the real, headerless
# continuation chunk from the live Gujarat/Ferrero/April 2026 investigation
# -- for the Out-of-Stock Rate value that previously failed (because the
# model cited a GraphRAG Entity description splitting period and value
# across two sentences instead) must PASS the EXISTING, UNMODIFIED
# check_grounding()/validate_claim(). No change to grounding_check.py was
# made to support this test -- it proves the fix belongs entirely in
# fact_structuring.py/citation choice, not in what grounding accepts.
# ---------------------------------------------------------------------------

# Verbatim (fictional-content-free -- this IS the real corpus text unit)
# continuation chunk text captured during the live Q4 investigation:
# Source(23) for Gujarat, April 2026 -- has NO "State: X\nPeriod: Y" header
# (only the FIRST chunk of a chunked document keeps it), yet the Ferrero
# category paragraph inside it is a complete, self-contained sentence.
GUJARAT_APRIL_2026_SOURCE_23_TEXT = (
    "Stock rate was 8.0%, and average Range Billing (share of the category's "
    "SKU range billed, among outlets that billed anything — definition pending "
    "GPIL confirmation) was 6.4%.\n\n"
    "For the IPM category (franchises: Marlboro) in Gujarat during April 2026: "
    "Numeric Distribution was 75.0%, ACV (weighted distribution — definition "
    "pending GPIL confirmation) was 75.2%, Out-of-Stock rate was 8.0%, and "
    "average Range Billing (share of the category's SKU range billed, among "
    "outlets that billed anything — definition pending GPIL confirmation) was "
    "39.4%.\n\n"
    "For the Ferrero category (franchises: TicTac, Kinder_Joy) in Gujarat "
    "during April 2026: Numeric Distribution was 21.6%, ACV (weighted "
    "distribution — definition pending GPIL confirmation) was 21.6%, "
    "Out-of-Stock rate was 2.9%, and average Range Billing (share of the "
    "category's SKU range billed, among outlets that billed anything — "
    "definition pending GPIL confirmation) was 16.8%.\n\n"
    "For the Candy category (franchises: Candy_Franchise_1, Candy_Franchise_2) "
    "in Gujarat during April 2026: Numeric Distribution was 19.8%, ACV "
    "(weighted distribution — definition pending GPIL confirmation) was "
    "19.8%, Out-of-Stock rate was 10.0%, and average Range Billing (share of "
    "the category's SKU range billed, among outlets that billed anything — "
    "definition pending GPIL confirmation) was 16.3%."
)


def test_q4_replay_claim_citing_source_23_for_ferrero_oos_passes_real_grounding():
    """The exact claim shape that previously failed grounding (entity=
    'Ferrero', metric='Out-of-Stock Rate', period='April 2026', value=2.9)
    -- but now cited against Sources (23) instead of the GraphRAG Entity
    description -- must pass validate_claim() outright, with zero flagged
    issues. This is what the category-KPI atomic-facts extractor is meant
    to make the model do (by giving it a clean, explicit "Source: Sources
    (23)" tag for exactly this fact), proven here directly against the
    real evidence text, independent of any actual LLM call."""
    records = make_context_records(sources=[{"id": "23", "text": GUJARAT_APRIL_2026_SOURCE_23_TEXT}])
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="The Ferrero category's Out-of-Stock Rate in Gujarat was 2.9% in April 2026.",
        claim_type="factual_numeric",
        entity="Ferrero",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=2.9,
        citations=["Sources (23)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert issues == []


def test_q4_replay_all_four_ferrero_metrics_pass_real_grounding_when_cited_to_source_23():
    """All four expected Q4 values (ND=21.6, ACV=21.6, OOS=2.9, RB=16.8),
    each as its own claim citing Sources (23), pass cleanly -- confirming
    the fix isn't a coincidence specific to the OOS sub-claim."""
    records = make_context_records(sources=[{"id": "23", "text": GUJARAT_APRIL_2026_SOURCE_23_TEXT}])
    evidence_index = _build_evidence_index(records)
    expected = [
        ("Numeric Distribution", 21.6),
        ("ACV", 21.6),
        ("Out-of-Stock Rate", 2.9),
        ("Range Billing", 16.8),
    ]
    for metric, value in expected:
        claim = AnswerClaim(
            claim_text=f"Ferrero {metric} in Gujarat in April 2026 was {value}%.",
            claim_type="factual_numeric",
            entity="Ferrero",
            metric=metric,
            period="April 2026",
            value=value,
            citations=["Sources (23)"],
        )
        issues = validate_claim(claim, evidence_index)
        assert issues == [], f"{metric} unexpectedly flagged: {issues}"


def test_q4_replay_entity_description_citation_still_fails_unchanged():
    """Negative control: a claim citing the OLD, problematic evidence shape
    (a GraphRAG Entity description that states the period in one sentence
    and the value in the next) must STILL fail -- proves grounding's own
    strictness is completely unchanged by the fact-structuring fix."""
    records = make_context_records(
        entities=[{
            "id": "691",
            "entity": "GUJARAT FERRERO OUT-OF-STOCK RATE APRIL 2026",
            "description": (
                'The "Gujarat Ferrero Out-Of-Stock Rate April 2026" reflects the availability of '
                "Ferrero products in the state of Gujarat during that month. Both descriptions "
                "indicate that the state-average Ferrero-scoped Out-of-Stock Rate for Gujarat was "
                "recorded at 2.9%."
            ),
        }]
    )
    evidence_index = _build_evidence_index(records)
    claim = AnswerClaim(
        claim_text="The Ferrero category's Out-of-Stock Rate in Gujarat was 2.9% in April 2026.",
        claim_type="factual_numeric",
        entity="Ferrero",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=2.9,
        citations=["Entities (691)"],
    )
    issues = validate_claim(claim, evidence_index)
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


# ---------------------------------------------------------------------------
# Entity/value misbinding hardening (Q3-K fix): scan_entity_numeric_claims()
# now (1) checks candidate (entity, value) pairs against the authoritative
# Atomic Facts list first, falling back to whole-evidence co-occurrence only
# for facts Atomic Facts don't cover, and (2) spans an entity+"deviation"
# clause and its value across up to two consecutive sentences when the
# model splits them, instead of requiring both in one sentence.
#
# GUJARAT_APRIL_2026_SOURCE / GUJARAT_ENTITIES (defined above) are reused
# throughout -- this is verbatim real pilot-index text.
# ---------------------------------------------------------------------------


def test_atomic_facts_correct_entity_correct_value_passes():
    """Requirement B, direction 1: Campbell -> 20.0% (its own real GPI OOS
    figure) must pass with zero issues."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    text = (
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage "
        "points [Data: Sources (50)]."
    )
    assert check_grounding(text, records, claims=[]).status == "pass"


def test_atomic_facts_wrong_entity_correct_value_fails_both_directions():
    """Requirement B, directions 2 and 3 (the core swap test): Baxter's
    name attached to Campbell's real 20.0% figure fails, AND Campbell's
    name attached to Baxter's real Dropsize 214.48 figure fails -- checked
    directly via _check_entity_value_against_atomic_facts()."""
    facts = extract_atomic_facts(make_context_records(sources=GUJARAT_APRIL_2026_SOURCE))

    result_a = _check_entity_value_against_atomic_facts("Baxter, Thomas and Williams Distributors", 20.0, facts)
    assert result_a is not None and not isinstance(result_a, AtomicFact)  # a GroundingIssue
    assert "Campbell PLC Distributors" in result_a.detail

    result_b = _check_entity_value_against_atomic_facts("Campbell PLC Distributors", 214.48, facts)
    assert result_b is not None and not isinstance(result_b, AtomicFact)
    assert "Baxter, Thomas and Williams Distributors" in result_b.detail


def test_atomic_facts_wrong_entity_wrong_value_falls_back_to_prose_check():
    """Requirement B, direction 4: an entity + a value that belongs to
    NEITHER distributor (never in Atomic Facts at all) defers to the
    whole-evidence prose check, which also correctly rejects it (the value
    simply never co-occurs with this entity anywhere)."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    text = (
        "Baxter, Thomas and Williams Distributors showed a significant Out-of-Stock Rate deviation "
        "for the GPI category in April 2026: 99.9% vs. the state average of 8.0%, a gap of 91.9 "
        "percentage points [Data: Sources (50)]."
    )
    result = check_grounding(text, records, claims=[])
    assert result.status == "fail"


def test_two_sentence_split_entity_value_misbinding_now_detected():
    """THE core Q3 regression test: the model splits the wrong distributor
    name + 'deviation' into one sentence and the actual value into the
    NEXT sentence -- previously produced ZERO issues (confirmed live, 2 of
    10 production runs slipped through this exact shape); must now be
    caught."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    two_sentence_wrong = (
        "In Gujarat, the distributor Baxter, Thomas and Williams Distributors showed a significant "
        "Out-of-Stock Rate deviation for the GPI category in April 2026. Their Out-of-Stock Rate was "
        "reported at 20.0%, which represented a gap of 12.0 percentage points above the state average "
        "of 8.0% for the same period [Data: Sources (50)]."
    )
    result = check_grounding(two_sentence_wrong, records, claims=[])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_numeric" for i in result.flagged_issues)
    assert any("Campbell PLC Distributors" in i.detail for i in result.flagged_issues)


def test_two_sentence_split_correct_attribution_still_passes():
    """Companion positive case: the SAME two-sentence phrasing, correctly
    naming Campbell, must still pass cleanly."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    two_sentence_right = (
        "In Gujarat, the distributor Campbell PLC Distributors showed a significant Out-of-Stock "
        "Rate deviation for the GPI category in April 2026. Their Out-of-Stock Rate was reported at "
        "20.0%, which represented a gap of 12.0 percentage points above the state average of 8.0% "
        "for the same period [Data: Sources (50)]."
    )
    result = check_grounding(two_sentence_right, records, claims=[])
    assert result.status == "pass"


def test_two_sentence_window_does_not_borrow_across_a_different_entity():
    """The two-sentence widening must NOT attribute a number to entity A
    when the intervening/next sentence actually introduces a DIFFERENT
    known entity -- that number belongs to the second entity's own
    sentence, handled on its own loop iteration."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    text = (
        "Baxter, Thomas and Williams Distributors showed a significant deviation. "
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage "
        "points [Data: Sources (50)]."
    )
    # Baxter's sentence has no number of its own and the NEXT sentence names
    # a different known entity (Campbell) -- must not borrow Campbell's
    # 20.0%/8.0% for Baxter. Campbell's own sentence is self-contained and
    # correct, so nothing should be flagged.
    result = check_grounding(text, records, claims=[])
    assert result.status == "pass"


def test_gap_delta_verified_against_atomic_fact_prose_only():
    """Requirement E (derived values), prose path: entity/value correctly
    bound, but the stated GAP doesn't match the Atomic Fact's own recorded
    gap -- must fail with unsupported_delta, quoting the correct gap."""
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    wrong_gap_text = (
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 25.0 percentage "
        "points [Data: Sources (50)]."
    )
    result = check_grounding(wrong_gap_text, records, claims=[])
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_delta" for i in result.flagged_issues)
    assert any("12.0" in i.detail for i in result.flagged_issues if i.issue_type == "unsupported_delta")


def test_gap_delta_correct_passes():
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    right_gap_text = (
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage "
        "points [Data: Sources (50)]."
    )
    assert check_grounding(right_gap_text, records, claims=[]).status == "pass"


# ---------------------------------------------------------------------------
# Ranking / extrema validation (requirement F)
# ---------------------------------------------------------------------------

DROPSIZE_RANKING_SOURCE = [
    {
        "id": "60",
        "text": (
            "State: Gujarat\nPeriod: April 2026\n\n"
            "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
            "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap of 10.6 percent.\n"
            "Distributor Walter-Li Distributors showed a significant deviation on Dropsize in April "
            "2026: 172.51 vs. the state average of 193.87, a gap of 11.0 percent.\n"
            "Distributor Lewis LLC Distributors showed a significant deviation on Dropsize in April "
            "2026: 217.88 vs. the state average of 193.87, a gap of 12.4 percent."
        ),
    }
]
RANKING_Q = "Which distributor in Gujarat had the highest Dropsize in April 2026?"
RANKING_MIN_Q = "Which distributor in Gujarat had the lowest Dropsize in April 2026?"


def test_ranking_correct_max_passes():
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    text = "Lewis LLC Distributors had the highest Dropsize in April 2026 at 217.88 [Data: Sources (60)]."
    result = check_grounding(text, records, claims=[], question=RANKING_Q)
    assert result.status == "pass"


def test_ranking_incorrect_but_real_candidate_fails():
    """The named entity's OWN value is real and correctly attributed to it
    (Baxter really is 214.48) -- but it is NOT the highest, so a plain
    entity/value grounding check alone would pass this. The ranking
    validator must still reject it."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    text = "Baxter, Thomas and Williams Distributors had the highest Dropsize in April 2026 at 214.48 [Data: Sources (60)]."
    result = check_grounding(text, records, claims=[], question=RANKING_Q)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)
    assert any("Lewis LLC Distributors" in i.detail for i in result.flagged_issues if i.issue_type == "unsupported_ranking")


def test_ranking_correct_min_passes():
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    text = "Walter-Li Distributors had the lowest Dropsize in April 2026 at 172.51 [Data: Sources (60)]."
    result = check_grounding(text, records, claims=[], question=RANKING_MIN_Q)
    assert result.status == "pass"


def test_ranking_incorrect_min_fails():
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    text = "Baxter, Thomas and Williams Distributors had the lowest Dropsize in April 2026 at 214.48 [Data: Sources (60)]."
    result = check_grounding(text, records, claims=[], question=RANKING_MIN_Q)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)


def test_ranking_tied_values_both_accepted():
    """When two candidates are tied for the true extremum, naming EITHER
    one is correct."""
    tied_source = [
        {
            "id": "61",
            "text": (
                "State: Gujarat\nPeriod: April 2026\n\n"
                "Distributor Alpha One Distributors showed a significant deviation on Dropsize in "
                "April 2026: 220.00 vs. the state average of 193.87, a gap of 13.5 percent.\n"
                "Distributor Beta Two Distributors showed a significant deviation on Dropsize in "
                "April 2026: 220.00 vs. the state average of 193.87, a gap of 13.5 percent."
            ),
        }
    ]
    records = make_context_records(sources=tied_source)
    q = "Which distributor had the highest Dropsize in April 2026?"
    text_a = "Alpha One Distributors had the highest Dropsize in April 2026 at 220.00 [Data: Sources (61)]."
    text_b = "Beta Two Distributors had the highest Dropsize in April 2026 at 220.00 [Data: Sources (61)]."
    assert check_grounding(text_a, records, claims=[], question=q).status == "pass"
    assert check_grounding(text_b, records, claims=[], question=q).status == "pass"


def test_ranking_structured_claim_entity_must_be_a_real_candidate():
    """2026-08-23 stabilization pass, live-caught bug: a structured claim
    whose `metric` field textually matches a real distributor metric name,
    but whose `entity` is something OTHER than one of the retrieved
    distributor names (e.g. a state name, or -- the live case -- a SKU
    name from an unrelated SKU-ranking question that happens to name the
    same metric, 'Out-of-Stock Rate'), must NOT be treated as a claimed
    distributor-ranking entity at all -- it should return [] (nothing to
    verify), not fabricate an 'unsupported_ranking' failure against an
    entity that was never a distributor candidate in the first place.
    Before this fix, verify_ranking_claim()'s structured-claim branch had
    no such guard (unlike its sibling verify_sku_ranking_claim(), which
    already checked `c.entity in by_entity`), so ANY claim naming this
    metric -- regardless of whose entity it was -- got treated as a
    distributor ranking claim and, since 'Bihar'/'Gujarat'/a SKU name is
    never one of the real distributor names, always failed."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    atomic_facts = extract_atomic_facts(records)
    claim = AnswerClaim(
        claim_text="Gujarat's lowest Dropsize was reported for a specific SKU, not a distributor.",
        entity="Gujarat",  # not a distributor name at all -- real Dropsize candidates are Baxter/Walter-Li/Lewis LLC
        metric="Dropsize",
        value=100.0,
    )
    issues = verify_ranking_claim(claim.claim_text, [claim], atomic_facts, RANKING_MIN_Q)
    assert issues == []


def test_ranking_structured_claim_entity_that_is_a_real_candidate_still_verified():
    """Regression guard for the fix above: a structured claim whose entity
    genuinely IS one of the real distributor candidates must still be
    verified exactly as before -- the new `c.entity in candidate_names`
    check must narrow false positives, not weaken real detection."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    atomic_facts = extract_atomic_facts(records)
    wrong_claim = AnswerClaim(
        claim_text="Baxter, Thomas and Williams Distributors had the highest Dropsize.",
        entity="Baxter, Thomas and Williams Distributors",  # real, but NOT the max (214.48 < 217.88)
        metric="Dropsize",
        value=214.48,
    )
    issues = verify_ranking_claim(wrong_claim.claim_text, [wrong_claim], atomic_facts, RANKING_Q)
    assert len(issues) == 1
    assert issues[0].issue_type == "unsupported_ranking"
    assert "Lewis LLC Distributors" in issues[0].detail


def test_ranking_skipped_when_question_names_no_metric():
    """A superlative word alone, with no real metric named, is not this
    validator's concern (premise_check's own undefined_metric gate handles
    the metric-less '<superlative> performing' shape separately)."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    issues = verify_ranking_claim("Someone did well.", [], extract_atomic_facts(records), "Who performed best?")
    assert issues == []


def test_ranking_skipped_with_fewer_than_two_candidates():
    single_source = [
        {
            "id": "62",
            "text": (
                "State: Gujarat\nPeriod: April 2026\n\n"
                "Distributor Alpha One Distributors showed a significant deviation on Dropsize in "
                "April 2026: 220.00 vs. the state average of 193.87, a gap of 13.5 percent."
            ),
        }
    ]
    records = make_context_records(sources=single_source)
    issues = verify_ranking_claim(
        "Alpha One Distributors had the highest Dropsize.", [], extract_atomic_facts(records), RANKING_Q
    )
    assert issues == []


# ---------------------------------------------------------------------------
# Thematic completeness signal (requirement G/L -- Dropsize omission)
# ---------------------------------------------------------------------------

MULTI_METRIC_SOURCE = [
    {
        "id": "70",
        "text": (
            "State: Gujarat\nPeriod: April 2026\n\n"
            "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
            "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap of 10.6 percent.\n"
            "Distributor Campbell PLC Distributors showed a significant deviation on Out-of-Stock "
            "Rate for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of "
            "12.0 percentage points."
        ),
    }
]
ATTENTION_Q = "Which distributors in Gujarat need attention in April 2026, and why?"


def test_thematic_completeness_flags_missing_metric_as_warning_not_failure():
    """The confirmed live Dropsize-omission shape: the answer covers the
    Out-of-Stock deviation (Campbell) but entirely omits the Dropsize
    deviation (Baxter), even though both were retrieved. Must be flagged
    for visibility but NEVER fail the overall grounding check on its own
    (severity='warning')."""
    records = make_context_records(sources=MULTI_METRIC_SOURCE, entities=GUJARAT_ENTITIES)
    text = (
        "Campbell PLC Distributors needs attention: its Out-of-Stock Rate for the GPI category in "
        "April 2026 was 20.0%, a gap of 12.0 percentage points above the state average of 8.0% "
        "[Data: Sources (50)]."
    )
    result = check_grounding(text, records, claims=[], question=ATTENTION_Q)
    incomplete_issues = [i for i in result.flagged_issues if i.issue_type == "incomplete_coverage"]
    assert len(incomplete_issues) == 1
    assert incomplete_issues[0].severity == "warning"
    assert "Dropsize" in incomplete_issues[0].detail
    assert "Baxter" in incomplete_issues[0].detail
    # A warning-only issue must never by itself fail the whole check.
    assert result.status == "pass"


def test_thematic_completeness_silent_when_all_metrics_covered():
    records = make_context_records(sources=MULTI_METRIC_SOURCE, entities=GUJARAT_ENTITIES)
    text = (
        "Two distributors need attention in Gujarat this month. Baxter, Thomas and Williams "
        "Distributors showed a significant deviation on Dropsize in April 2026: 214.48 vs. the "
        "state average of 193.87, a gap of 10.6 percent [Data: Sources (50)]. Campbell PLC "
        "Distributors showed a significant Out-of-Stock Rate deviation for the GPI category in "
        "April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage points "
        "[Data: Sources (50)]."
    )
    result = check_grounding(text, records, claims=[], question=ATTENTION_Q)
    assert not any(i.issue_type == "incomplete_coverage" for i in result.flagged_issues)


def test_thematic_completeness_skipped_for_metric_specific_question():
    """A question that already names a specific metric is legitimately
    scoped to it -- an answer discussing only that metric is not
    incomplete."""
    records = make_context_records(sources=MULTI_METRIC_SOURCE, entities=GUJARAT_ENTITIES)
    q = "Which distributor in Gujarat had a significant Out-of-Stock Rate deviation in April 2026?"
    text = (
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage "
        "points [Data: Sources (50)]."
    )
    result = check_grounding(text, records, claims=[], question=q)
    assert not any(i.issue_type == "incomplete_coverage" for i in result.flagged_issues)


def test_thematic_completeness_skipped_for_non_attention_shaped_question():
    records = make_context_records(sources=MULTI_METRIC_SOURCE, entities=GUJARAT_ENTITIES)
    q = "What happened in Gujarat in April 2026?"
    text = (
        "Campbell PLC Distributors showed a significant Out-of-Stock Rate deviation for the GPI "
        "category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 percentage "
        "points [Data: Sources (50)]."
    )
    result = check_grounding(text, records, claims=[], question=q)
    assert not any(i.issue_type == "incomplete_coverage" for i in result.flagged_issues)


# ---------------------------------------------------------------------------
# Thematic / multi-claim questions (requirement G): confirm the EXISTING
# architecture already fails the whole answer when ANY one of several
# claims is wrong -- claims are checked independently but any single
# error-severity issue flips the whole GroundingCheckResult to "fail".
# No code change was needed for this; this is a regression/confirmation
# test.
# ---------------------------------------------------------------------------


def test_one_bad_claim_among_several_fails_the_whole_answer():
    records = make_context_records(sources=GUJARAT_APRIL_2026_SOURCE, entities=GUJARAT_ENTITIES)
    good_claim = AnswerClaim(
        claim_id=0,
        claim_text="Campbell PLC Distributors' GPI Out-of-Stock Rate was 20.0%, a gap of 12.0 points.",
        claim_type="deviation",
        entity="Campbell PLC Distributors",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=20.0,
        delta=12.0,
        citations=["Sources (50)"],
    )
    bad_claim = AnswerClaim(
        claim_id=1,
        claim_text="Baxter, Thomas and Williams Distributors' GPI Out-of-Stock Rate was 20.0%.",
        claim_type="deviation",
        entity="Baxter, Thomas and Williams Distributors",
        metric="Out-of-Stock Rate",
        period="April 2026",
        value=20.0,
        citations=["Sources (50)"],
    )
    text = (
        "Campbell PLC Distributors' GPI Out-of-Stock Rate was 20.0%, a gap of 12.0 points. "
        "Baxter, Thomas and Williams Distributors' GPI Out-of-Stock Rate was 20.0%."
    )
    result = check_grounding(text, records, claims=[good_claim, bad_claim])
    assert result.status == "fail"
    claim_ids_with_errors = {i.claim_id for i in result.flagged_issues if i.severity == "error" and i.source == "claim"}
    assert 1 in claim_ids_with_errors
    assert 0 not in claim_ids_with_errors  # the good claim itself is never flagged


def test_ranking_does_not_pull_in_unrelated_states_candidate():
    """Live validation regression: a Goa/June-2025 Dropsize ranking
    question must not be able to pull in a same-metric distributor fact
    from a COMPLETELY UNRELATED state/period's retrieved source as a
    'higher' candidate (the confirmed live case: an unrelated Meghalaya/
    July-2026 distributor's real Dropsize value was briefly treated as
    ranking above Goa's real candidates)."""
    goa_source = [
        {
            "id": "38",
            "text": (
                "State: Goa\nPeriod: June 2025\n\n"
                "Distributor Pearson-Palmer Distributors showed a significant deviation on Dropsize "
                "in June 2025: 170.05 vs. the state average of 149.00, a gap of 14.1 percent.\n"
                "Distributor Moreno Inc Distributors showed a significant deviation on Dropsize in "
                "June 2025: 182.70 vs. the state average of 149.00, a gap of 22.6 percent."
            ),
        }
    ]
    unrelated_state_source = [
        {
            "id": "45",
            "text": (
                "State: Meghalaya\nPeriod: July 2026\n\n"
                "Distributor Tate-Gonzalez Distributors showed a significant deviation on Dropsize "
                "in July 2026: 188.85 vs. the state average of 171.15, a gap of 10.3 percent."
            ),
        }
    ]
    records = make_context_records(sources=goa_source + unrelated_state_source)
    q = "Which distributor in Goa had the highest Dropsize in June 2025?"

    # Moreno IS the real highest among Goa/June-2025 candidates -- must pass.
    text = "Moreno Inc Distributors had the highest Dropsize in Goa in June 2025 at 182.70 [Data: Sources (38)]."
    result = check_grounding(text, records, claims=[], question=q)
    assert result.status == "pass"
    assert not any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)


def test_ranking_state_scoping_still_rejects_a_real_wrong_candidate_within_the_same_state():
    """Companion negative case, same fixture: a real Goa candidate that is
    genuinely NOT the highest (Pearson-Palmer, 170.05 < Moreno's 182.70)
    must still be correctly rejected -- state-scoping narrows the
    candidate pool, it does not weaken the check itself."""
    goa_source = [
        {
            "id": "38",
            "text": (
                "State: Goa\nPeriod: June 2025\n\n"
                "Distributor Pearson-Palmer Distributors showed a significant deviation on Dropsize "
                "in June 2025: 170.05 vs. the state average of 149.00, a gap of 14.1 percent.\n"
                "Distributor Moreno Inc Distributors showed a significant deviation on Dropsize in "
                "June 2025: 182.70 vs. the state average of 149.00, a gap of 22.6 percent."
            ),
        }
    ]
    records = make_context_records(sources=goa_source)
    q = "Which distributor in Goa had the highest Dropsize in June 2025?"
    text = "Pearson-Palmer Distributors had the highest Dropsize in Goa in June 2025 at 170.05 [Data: Sources (38)]."
    result = check_grounding(text, records, claims=[], question=q)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)


# ---------------------------------------------------------------------------
# focus_categories false-positive fix (live Gamble-Wright regression): a
# question naming ONE category (e.g. "GPI") must not cause check_grounding()
# to reject an otherwise-correct claim about a DIFFERENT category, when that
# claim's OWN metric field already names its own category explicitly. This
# is exactly the multi-period, multi-category question shape ("had a GPI
# deviation in February... what NEW deviations by August?") that exposed it.
# ---------------------------------------------------------------------------

MANIPUR_MULTI_PERIOD_SOURCE = [
    {
        "id": "6",
        "text": (
            "State: Manipur\nPeriod: February 2025\n\n"
            "Distributor Gamble-Wright Distributors showed a significant deviation on Dropsize in "
            "February 2025: 194.27 vs. the state average of 174.14, a gap of 11.6 percent.\n"
            "Distributor Gamble-Wright Distributors showed a significant deviation on Out-of-Stock "
            "Rate for the GPI category in February 2025: 30.0% vs. the state average of 11.9%, a gap "
            "of 18.1 percentage points."
        ),
    },
    {
        "id": "42",
        "text": (
            "State: Manipur\nPeriod: August 2025\n\n"
            "Distributor Gamble-Wright Distributors showed a significant deviation on Service Level "
            "in August 2025: 75.0% vs. the state average of 92.9%, a gap of 17.8 percentage points.\n"
            "Distributor Gamble-Wright Distributors showed a significant deviation on Out-of-Stock "
            "Rate for the IPM category in August 2025: 20.0% vs. the state average of 7.5%, a gap of "
            "12.5 percentage points.\n"
            "Distributor Gamble-Wright Distributors showed a significant deviation on Out-of-Stock "
            "Rate for the Ferrero category in August 2025: 14.3% vs. the state average of 3.6%, a gap "
            "of 10.7 percentage points."
        ),
    },
]
GAMBLE_WRIGHT_Q = (
    "Gamble-Wright Distributors in Manipur had a Dropsize deviation and a GPI Out-of-Stock deviation "
    "in February 2025. By August 2025, did those two issues persist, and what new deviations did "
    "Gamble-Wright pick up instead?"
)


def test_multi_category_claims_not_wrongly_rejected_by_question_global_category():
    """The question names 'GPI' once (describing February), but the real,
    correctly-cited August claims are for IPM and Ferrero -- each claim's
    OWN metric field already names its category, so the question's
    unrelated 'GPI' mention must not reject them."""
    records = make_context_records(sources=MANIPUR_MULTI_PERIOD_SOURCE)
    claims = [
        AnswerClaim(claim_id=0, entity="Gamble-Wright Distributors", metric="Dropsize",
                    period="February 2025", value=194.27, citations=["Sources (6)"]),
        AnswerClaim(claim_id=1, entity="Gamble-Wright Distributors", metric="GPI Out-of-Stock Rate",
                    period="February 2025", value=30.0, citations=["Sources (6)"]),
        AnswerClaim(claim_id=2, entity="Gamble-Wright Distributors", metric="Service Level",
                    period="August 2025", value=75.0, comparison_value=92.9, delta=17.8,
                    citations=["Sources (42)"]),
        AnswerClaim(claim_id=3, entity="Gamble-Wright Distributors", metric="IPM Out-of-Stock Rate",
                    period="August 2025", value=20.0, comparison_value=7.5, delta=12.5,
                    citations=["Sources (42)"]),
        AnswerClaim(claim_id=4, entity="Gamble-Wright Distributors", metric="Ferrero Out-of-Stock Rate",
                    period="August 2025", value=14.3, comparison_value=3.6, delta=10.7,
                    citations=["Sources (42)"]),
    ]
    result = check_grounding("placeholder", records, claims, question=GAMBLE_WRIGHT_Q)
    assert result.status == "pass"


def test_focus_categories_still_rejects_wrong_category_when_claim_metric_is_silent_on_category():
    """Regression guard for the ORIGINAL bug this mechanism was built to
    fix (Phase 8 hardening): a claim whose metric field does NOT itself
    name a category (plain 'Out-of-Stock Rate') must still be rejected
    against a cited sentence for a DIFFERENT category than the question
    asked about -- the new fix only exempts claims that self-declare their
    own category, it must not weaken this original protection."""
    records = make_context_records(
        sources=[{
            "id": "50",
            "text": (
                "State: Gujarat\nPeriod: April 2026\n\n"
                "Distributor Campbell PLC Distributors showed a significant deviation on Out-of-Stock "
                "Rate for the IPM category in April 2026: 20.0% vs. the state average of 8.0%, a gap "
                "of 12.0 percentage points."
            ),
        }]
    )
    claim = AnswerClaim(
        entity="Campbell PLC Distributors", metric="Out-of-Stock Rate",  # no category named
        period="April 2026", value=20.0, citations=["Sources (50)"],
    )
    # Question asks specifically about GPI -- the cited sentence is IPM, a
    # different category, and the claim's own metric field is silent on
    # category, so this must still be rejected.
    issues = validate_claim(claim, _build_evidence_index(records), focus_categories=frozenset({"GPI"}))
    assert any(i.issue_type == "unsupported_numeric" for i in issues)


# ---------------------------------------------------------------------------
# SKU ranking claim verification (Problem 1/4) -- verify_sku_ranking_claim(),
# the top-N-capable sibling of verify_ranking_claim() for kind="sku" Atomic
# Facts. Fictional state/SKU names, mirroring the distributor ranking
# tests' style exactly.
# ---------------------------------------------------------------------------

SKU_RANKING_SOURCE = [
    {
        "id": "70",
        "text": (
            "SKU-level Sales & Distribution detail for Ruritania, May 2027:\n"
            "SKU SKU0001 (Alpha Pack 1, Alpha franchise, Zorn category) in Ruritania during "
            "May 2027: 12,000 units delivered, revenue of Rs 900,000, Service Level 91.0%, "
            "Numeric Distribution 60.0%, Out-of-Stock rate 3.0%.\n"
            "SKU SKU0002 (Beta Pack 1, Beta franchise, Zorn category) in Ruritania during "
            "May 2027: 9,000 units delivered, revenue of Rs 700,000, Service Level 88.0%, "
            "Numeric Distribution 40.0%, Out-of-Stock rate 6.0%.\n"
            "SKU SKU0003 (Gamma Pack 1, Gamma franchise, Zorn category) in Ruritania during "
            "May 2027: 500 units delivered, revenue of Rs 40,000, Service Level 70.0%, "
            "Numeric Distribution 10.0%, Out-of-Stock rate 22.0%.\n"
        ),
    }
]
SKU_TOP1_Q = "Which SKU has the highest Revenue in Ruritania in May 2027?"
SKU_TOP2_Q = "What are the top 2 SKUs by Revenue in Ruritania in May 2027?"
SKU_BOTTOM_Q = "Which SKU has the lowest Revenue in Ruritania in May 2027?"


def test_sku_ranking_correct_top1_passes():
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text = "Alpha Pack 1 had the highest Revenue in Ruritania in May 2027 at Rs 900,000 [Data: Sources (70)]."
    result = check_grounding(text, records, claims=[], question=SKU_TOP1_Q)
    assert result.status == "pass"


def test_sku_ranking_incorrect_top1_fails():
    """Gamma Pack 1's own Revenue figure is real and correctly attributed
    -- but it is nowhere near the highest, so plain entity/value grounding
    alone would pass this. The SKU ranking validator must still reject
    it."""
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text = "Gamma Pack 1 had the highest Revenue in Ruritania in May 2027 at Rs 40,000 [Data: Sources (70)]."
    result = check_grounding(text, records, claims=[], question=SKU_TOP1_Q)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)
    assert any("Alpha Pack 1" in i.detail for i in result.flagged_issues if i.issue_type == "unsupported_ranking")


def test_sku_ranking_top_n_accepts_any_of_the_true_top_n():
    """A 'top 2' question naming either of the two genuinely highest SKUs
    must pass -- this is the generalization verify_ranking_claim() (single
    extremum only) doesn't support."""
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text_a = "The top 2 SKUs by Revenue were Alpha Pack 1 and Beta Pack 1 [Data: Sources (70)]."
    result = check_grounding(text_a, records, claims=[], question=SKU_TOP2_Q)
    assert result.status == "pass"


def test_sku_ranking_top_n_rejects_an_entity_outside_the_true_top_n():
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text = "The top 2 SKUs by Revenue were Alpha Pack 1 and Gamma Pack 1 [Data: Sources (70)]."
    result = check_grounding(text, records, claims=[], question=SKU_TOP2_Q)
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)
    assert any("Gamma Pack 1" in i.detail for i in result.flagged_issues if i.issue_type == "unsupported_ranking")


def test_sku_ranking_bottom_direction_correct_passes():
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text = "Gamma Pack 1 had the lowest Revenue in Ruritania in May 2027 at Rs 40,000 [Data: Sources (70)]."
    result = check_grounding(text, records, claims=[], question=SKU_BOTTOM_Q)
    assert result.status == "pass"


def test_sku_ranking_skipped_when_question_names_no_sku_metric():
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    issues = verify_sku_ranking_claim("Someone did well.", [], extract_atomic_facts(records), "Who performed best?")
    assert issues == []


def test_sku_ranking_recognizes_selling_synonym_as_units_delivered():
    """'most selling' names no literal SKU metric, but query_requirements.py
    now resolves it to Units Delivered (2026-08-21 fix) -- this verifier
    must recognize the same synonym so a wrong 'most selling' ranking
    claim still gets caught, not silently skipped."""
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    text = "Gamma Pack 1 was the most selling SKU in Ruritania in May 2027, with 500 units [Data: Sources (70)]."
    result = check_grounding(
        text, records, claims=[], question="Which SKU was the most selling SKU in Ruritania in May 2027?"
    )
    assert result.status == "fail"
    assert any(i.issue_type == "unsupported_ranking" for i in result.flagged_issues)


def test_sku_ranking_skipped_with_fewer_candidates_than_n():
    records = make_context_records(sources=SKU_RANKING_SOURCE)
    issues = verify_sku_ranking_claim(
        "Alpha Pack 1 and Beta Pack 1 are the top 5 SKUs by Revenue.",
        [], extract_atomic_facts(records),
        "What are the top 5 SKUs by Revenue in Ruritania in May 2027?",
    )
    assert issues == []  # only 3 SKU candidates exist, fewer than the requested 5


def test_sku_ranking_does_not_cross_contaminate_distributor_ranking():
    """A question ranking distributors must not be affected by SKU Atomic
    Facts being present in the same retrieved context, and vice versa --
    the two validators are scoped to their own AtomicFact.kind."""
    combined_source = DROPSIZE_RANKING_SOURCE + SKU_RANKING_SOURCE
    records = make_context_records(sources=combined_source)
    text = "Lewis LLC Distributors had the highest Dropsize in April 2026 at 217.88 [Data: Sources (60)]."
    result = check_grounding(text, records, claims=[], question=RANKING_Q)
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# resolve_claim_sources() -- provenance facade (Problem 2)
# ---------------------------------------------------------------------------


def test_resolve_claim_sources_returns_evidence_text_for_cited_claim():
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    claim = AnswerClaim(
        claim_id=0,
        claim_text="Lewis LLC Distributors had the highest Dropsize in April 2026 at 217.88",
        entity="Lewis LLC Distributors", metric="Dropsize", value=217.88,
        citations=["Sources (60)"],
    )
    sources = resolve_claim_sources([claim], records)
    assert len(sources) == 1
    assert sources[0].claim_id == 0
    assert sources[0].citation_ids == ["Sources (60)"]
    assert "Lewis LLC Distributors" in sources[0].evidence_text
    assert "217.88" in sources[0].evidence_text


def test_resolve_claim_sources_skips_claims_with_no_citations():
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    claim = AnswerClaim(claim_id=1, claim_text="An uncited claim.", citations=[])
    assert resolve_claim_sources([claim], records) == []


def test_resolve_claim_sources_skips_claims_with_unresolvable_citations():
    """A citation that doesn't resolve to any real record must never
    produce a fabricated source -- nothing is shown, not a placeholder."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    claim = AnswerClaim(claim_id=2, claim_text="Something.", citations=["Sources (9999)"])
    assert resolve_claim_sources([claim], records) == []


def test_resolve_claim_sources_one_entry_per_claim_even_with_shared_citation():
    """Two different claims citing the SAME source each get their own
    SourceCitation entry -- this is what lets a UI show 'Claim 1 -> Source
    X' and 'Claim 2 -> Source X' separately, per Problem 2's brief."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    claim_a = AnswerClaim(claim_id=0, claim_text="Claim A", citations=["Sources (60)"])
    claim_b = AnswerClaim(claim_id=1, claim_text="Claim B", citations=["Sources (60)"])
    sources = resolve_claim_sources([claim_a, claim_b], records)
    assert len(sources) == 2
    assert {s.claim_id for s in sources} == {0, 1}


def test_resolve_claim_sources_narrows_multi_sku_blob_to_the_specific_sku():
    """THE core bug from the live 15-question SKU validation pass: a claim
    correctly citing the whole-blob 'Sources (sku-Punjab-February_2026)'
    id (a real, correctly-formatted citation, not the malformed
    "Atomic SKU Facts (N)" shape) previously displayed the ENTIRE ~59-SKU
    source blob as its "evidence" -- so a ranking answer's THREE different
    claims (about three different SKUs) all showed the identical opening
    lines of the shared record, none of them actually pointing at the SKU
    each claim was really about. Must now narrow to just that one SKU's
    own sentence."""
    records = make_sku_context_records()
    claim = AnswerClaim(
        claim_id=0,
        claim_text="GPI_Franchise_3 Pack 1 had 304,315 units delivered in Punjab in February 2026.",
        entity="GPI_Franchise_3 Pack 1", metric="Units Delivered", period="February 2026",
        value=304315.0, citations=["Sources (sku-Punjab-February_2026)"],
    )
    sources = resolve_claim_sources([claim], records)
    assert len(sources) == 1
    assert "GPI_Franchise_3 Pack 1" in sources[0].evidence_text
    assert "304,315" in sources[0].evidence_text
    # The other SKU sharing the same source record must NOT appear --
    # this is the precise assertion that would have caught the bug.
    assert "Marlboro Pack 1" not in sources[0].evidence_text
    assert "835,078" not in sources[0].evidence_text


def test_resolve_claim_sources_different_claims_same_blob_get_different_narrowed_evidence():
    """Two claims about two DIFFERENT SKUs, both citing the same whole-blob
    source id, must each display THEIR OWN SKU's evidence -- not the same
    text twice. This is the exact shape of the live failure: a 'top 3'
    answer's three claims all showed identical (and mostly irrelevant)
    evidence before this fix."""
    records = make_sku_context_records()
    claim_marlboro = AnswerClaim(
        claim_id=0, claim_text="Marlboro Pack 1 claim",
        entity="Marlboro Pack 1", metric="Units Delivered", period="February 2026",
        value=835078.0, citations=["Sources (sku-Punjab-February_2026)"],
    )
    claim_gpi = AnswerClaim(
        claim_id=1, claim_text="GPI_Franchise_3 Pack 1 claim",
        entity="GPI_Franchise_3 Pack 1", metric="Units Delivered", period="February 2026",
        value=304315.0, citations=["Sources (sku-Punjab-February_2026)"],
    )
    sources = resolve_claim_sources([claim_marlboro, claim_gpi], records)
    by_claim = {s.claim_id: s.evidence_text for s in sources}
    assert "Marlboro Pack 1" in by_claim[0] and "GPI_Franchise_3 Pack 1" not in by_claim[0]
    assert "GPI_Franchise_3 Pack 1" in by_claim[1] and "Marlboro Pack 1" not in by_claim[1]


def test_resolve_claim_sources_falls_back_to_full_text_when_not_narrowable():
    """A claim with no entity/value pair to anchor on (e.g. a pure
    qualitative/causal claim) can't be narrowed to one sentence -- must
    fall back to the full resolved text, never drop to empty."""
    records = make_context_records(sources=DROPSIZE_RANKING_SOURCE)
    claim = AnswerClaim(claim_id=0, claim_text="Something evaluative.", citations=["Sources (60)"])
    sources = resolve_claim_sources([claim], records)
    assert len(sources) == 1
    assert sources[0].evidence_text != ""
