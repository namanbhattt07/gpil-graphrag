"""
Tests for src/inference/query_requirements.py (Problem 4 -- granularity
mismatch detection). Pure question-text analysis, no I/O -- these tests
never touch context_records, a data_dir, or an index.
"""

import pytest

from src.inference.query_requirements import detect_query_requirements


def test_sku_word_triggers_sku_granularity():
    req = detect_query_requirements("Tell me our top 3 SKUs in Punjab in 2025.")
    assert req.required_granularity == "sku"
    assert req.target_state == "Punjab"
    assert req.target_year == 2025
    assert req.target_periods == []
    assert req.rank_n == 3
    assert req.is_ranking is True


@pytest.mark.parametrize(
    "question",
    [
        "Which SKU has the highest sales?",
        "Which SKUs are performing poorly in Punjab?",
        "Which SKU has weak distribution?",
        "Where do we need intervention at SKU level?",
        "Which SKUs are doing well vs underperforming?",
    ],
)
def test_various_sku_phrasings_all_detected(question):
    assert detect_query_requirements(question).required_granularity == "sku"


def test_ordinary_state_question_has_no_granularity_requirement():
    """A normal existing-shape question (no "SKU" word) must never trigger
    the SKU path -- this is what keeps every pre-existing question's
    behavior completely unchanged (see pipeline.py)."""
    req = detect_query_requirements("Why did Bihar's Service Level decline in November 2025?")
    assert req.required_granularity is None
    assert req.target_state == "Bihar"


def test_explicit_month_year_period_is_extracted_and_wins_over_bare_year():
    req = detect_query_requirements("Top 3 SKUs in Punjab in March 2025")
    assert req.target_periods == ["March 2025"]
    assert req.target_year is None  # explicit month wins; no separate bare-year parse


def test_bare_year_extracted_when_no_month_named():
    req = detect_query_requirements("Top SKUs in Gujarat in 2026")
    assert req.target_year == 2026
    assert req.target_periods == []


def test_no_state_named_leaves_target_state_none():
    req = detect_query_requirements("Which SKU has the highest sales?")
    assert req.target_state is None


def test_rank_n_defaults_to_none_without_explicit_count():
    req = detect_query_requirements("Which SKU has the highest sales in Punjab?")
    assert req.rank_n is None
    assert req.is_ranking is True


def test_bottom_n_also_parsed():
    req = detect_query_requirements("Bottom 5 SKUs in Kerala in 2025")
    assert req.rank_n == 5


def test_multi_period_question_populates_both_periods_chronologically_unsorted():
    """detect_query_requirements just forwards premise_check.py's own
    extraction (order of appearance, not chronological) -- sorting/pairing
    baseline vs. target is premise_check's job, not this module's."""
    req = detect_query_requirements(
        "SKU performance in Punjab: compare February 2025 to August 2025."
    )
    assert req.target_periods == ["February 2025", "August 2025"]


def test_shared_year_period_range_extracts_both_months():
    """15-query SKU validation pass (2026-08-21): 'between January and
    February 2026' (year stated once, after the second month) previously
    only extracted 'February 2026' -- the SKU evidence lookup then never
    got January's data, and the pipeline wrongly reported January as
    unavailable even though it's indexed. See
    premise_check.py's _expand_shared_year_period_ranges()."""
    req = detect_query_requirements(
        "Compare the top-selling SKU in Punjab between January and February 2026."
    )
    assert req.target_periods == ["January 2026", "February 2026"]


# ---------------------------------------------------------------------------
# ranking_metric -- ambiguous-ranking detection (Design Decision: ask
# rather than guess a default "top SKU" metric)
# ---------------------------------------------------------------------------


def test_ranking_metric_none_when_ambiguous():
    """The exact case pipeline.py's needs_clarification short-circuit
    exists to catch: a ranking-shaped SKU question naming no real metric."""
    req = detect_query_requirements("Tell me the top 3 SKUs in Punjab in 2025.")
    assert req.required_granularity == "sku"
    assert req.is_ranking is True
    assert req.ranking_metric is None


@pytest.mark.parametrize(
    "question,expected_metric",
    [
        ("Top 3 SKUs in Punjab by Revenue in March 2025", "Revenue"),
        ("Which SKU has the highest Units Delivered in Punjab?", "Units Delivered"),
        ("Bottom 5 SKUs by Service Level in Kerala in 2025", "Service Level"),
        ("Top SKUs by Numeric Distribution in Gujarat", "Numeric Distribution"),
        ("Which SKU has the highest Out-of-Stock Rate in Assam?", "Out-of-Stock Rate"),
    ],
)
def test_ranking_metric_detected_when_named(question, expected_metric):
    req = detect_query_requirements(question)
    assert req.ranking_metric == expected_metric


def test_ranking_metric_none_for_non_ranking_sku_question():
    """A plain factual (non-ranking) SKU question has nothing to
    disambiguate -- ranking_metric stays None, but this must never trigger
    the ambiguous-ranking clarification gate (that also checks is_ranking)."""
    req = detect_query_requirements("What was the revenue of SKU X in Punjab in March 2025?")
    assert req.is_ranking is False
    # "revenue" is still a real metric name, so it's detected even though
    # required_granularity=="sku" and is_ranking is False here -- harmless,
    # since pipeline.py's gate also requires is_ranking to be True.
    assert req.ranking_metric == "Revenue"


# ---------------------------------------------------------------------------
# ranking_metric -- "selling"/"sold" synonyms map to Units Delivered
# (15-query SKU validation pass, 2026-08-21): "most selling"/"top-selling"/
# "sold the most" don't literally contain any SKU_METRIC_FIELDS label, so
# without synonym recognition these were wrongly treated as ambiguous and
# sent to needs_clarification -- even though "selling"/"sold" unambiguously
# means sales volume, never revenue/service-level/distribution/OOS.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What were the top 3 most selling SKUs in Punjab in February 2026?",
        "Compare the top-selling SKU in Punjab between January and February 2026.",
        "Which SKU sold the most in Punjab in February 2026?",
        "Did the top-selling SKU in Punjab change from January 2026 to February 2026?",
        "What was the best selling SKU in Gujarat in April 2026?",
    ],
)
def test_selling_synonyms_resolve_to_units_delivered(question):
    req = detect_query_requirements(question)
    assert req.ranking_metric == "Units Delivered"


@pytest.mark.parametrize(
    "question",
    [
        "What were the top 3 SKUs in Punjab in February 2026?",
        "Which were the best-performing SKUs in Punjab in February 2026?",
    ],
)
def test_genuinely_ambiguous_ranking_phrases_still_return_none(question):
    """A bare 'top SKUs' or 'best-performing SKUs' names no metric and no
    sales-volume word -- must stay genuinely ambiguous (None), not get
    swept up by the new selling-synonym recognition. These must still
    reach pipeline.py's needs_clarification gate."""
    req = detect_query_requirements(question)
    assert req.is_ranking is True
    assert req.ranking_metric is None


def test_ranking_metric_none_for_non_sku_question():
    """ranking_metric is only ever computed for SKU-granularity questions
    -- an ordinary ranking-shaped non-SKU question (e.g. about
    distributors) must not pick up a spurious SKU metric name."""
    req = detect_query_requirements("Which distributor had the highest Service Level in Bihar?")
    assert req.required_granularity is None
    assert req.ranking_metric is None


# ---------------------------------------------------------------------------
# rank_direction (2026-08-23 stabilization pass -- Fix 1: deterministic SKU
# ranking architecture). Lets sku_evidence.py sort real rows itself instead
# of asking the LLM to compute a ranking from a pile of facts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Top 5 SKUs by Revenue in Gujarat in April 2026", "top"),
        ("Which SKU has the highest Units Delivered in Punjab?", "top"),
        ("What was the best selling SKU in Gujarat in April 2026?", "top"),
        ("Which SKU sold the most in Punjab in February 2026?", "top"),
        ("Bottom 5 SKUs by Service Level in Kerala in 2025", "bottom"),
        ("Which SKU has the lowest Numeric Distribution in Assam?", "bottom"),
        ("What was the worst-performing SKU by Revenue in Gujarat?", "bottom"),
    ],
)
def test_rank_direction_detected(question, expected):
    assert detect_query_requirements(question).rank_direction == expected


def test_rank_direction_none_for_non_ranking_question():
    req = detect_query_requirements("What was the revenue of SKU X in Punjab in March 2025?")
    assert req.rank_direction is None


def test_rank_direction_none_when_both_max_and_min_words_present():
    """A genuinely mixed question ('highest ... and lowest ...') must not
    silently pick one direction -- mirrors premise_check.py's own
    extract_direction_claim() 'only gate on an unambiguous claim'
    discipline."""
    req = detect_query_requirements("Which SKU had the highest and lowest Revenue in Punjab?")
    assert req.rank_direction is None


# ---------------------------------------------------------------------------
# known_sku_names param -- SKU-name-only queries (Fix 2). A question that
# names a real SKU (e.g. "Marlboro Pack 1") must be recognized as
# SKU-grain even when the literal word "SKU" never appears, WITHOUT any
# hardcoded name list living in this module -- the caller supplies the
# real name set (see sku_evidence.known_sku_names()).
# ---------------------------------------------------------------------------

_KNOWN_NAMES = frozenset({"Marlboro Pack 1", "GPI_Franchise_1 Pack 1", "TicTac Pack 2"})


def test_bare_sku_name_triggers_sku_granularity_when_known_names_given():
    req = detect_query_requirements(
        "What was the revenue of Marlboro Pack 1 in Gujarat in April 2026?", known_sku_names=_KNOWN_NAMES
    )
    assert req.required_granularity == "sku"
    assert req.target_state == "Gujarat"


def test_bare_sku_name_comparison_question_triggers_sku_granularity():
    req = detect_query_requirements(
        "Compare Marlboro Pack 1 and GPI_Franchise_1 Pack 1 in Gujarat in April 2026.",
        known_sku_names=_KNOWN_NAMES,
    )
    assert req.required_granularity == "sku"


def test_bare_sku_name_without_known_names_param_is_not_detected():
    """Omitting known_sku_names (the default) reproduces the exact prior
    literal-'SKU'-word-only behavior -- every existing caller/test that
    doesn't pass this parameter is completely unaffected."""
    req = detect_query_requirements("What was the revenue of Marlboro Pack 1 in Gujarat in April 2026?")
    assert req.required_granularity is None


def test_unknown_product_name_does_not_trigger_sku_granularity():
    """A name that ISN'T in known_sku_names (e.g. a state or distributor
    name, or a nonexistent product) must never spuriously trigger the SKU
    path -- this is a substring match against real names only, not a
    generic proper-noun detector."""
    req = detect_query_requirements(
        "Why did Bihar's Service Level decline in November 2025?", known_sku_names=_KNOWN_NAMES
    )
    assert req.required_granularity is None


def test_bare_sku_name_still_requires_no_metric_word_for_ranking_metric():
    """A bare-name factual lookup (not a ranking question) leaves
    ranking_metric/is_ranking exactly as for any other non-ranking SKU
    question -- the bare-name detection only affects required_granularity,
    nothing else in QueryRequirements."""
    req = detect_query_requirements(
        "The revenue of Marlboro Pack 1 in April 2026?", known_sku_names=_KNOWN_NAMES
    )
    assert req.required_granularity == "sku"
    assert req.is_ranking is False


# ---------------------------------------------------------------------------
# Live bug (2026-08-23, SKU inference validation pass): a franchise-derived
# SKU name's STORED form uses an underscore ("GPI_Franchise_1 Pack 1"), but
# the natural way a person asks about it uses a space ("GPI Franchise 1
# Pack 1") -- the exact substring match against the literal stored name
# silently failed to recognize this as a SKU question at all, degrading to
# required_granularity=None (an ordinary, non-SKU question). Fixed by
# normalizing underscores/whitespace on BOTH sides before comparing (see
# _normalize_sku_name_text()). "Marlboro"/"TicTac"/"Kinder_Joy" SKU names
# have no franchise-number suffix in their question-facing form, which is
# why this was invisible in every existing test/example using those names.
# ---------------------------------------------------------------------------


def test_bare_sku_name_with_space_instead_of_underscore_still_triggers_sku_granularity():
    req = detect_query_requirements(
        "What was the revenue of GPI Franchise 1 Pack 1 in Gujarat in April 2026?",
        known_sku_names=_KNOWN_NAMES,
    )
    assert req.required_granularity == "sku"
    assert req.target_state == "Gujarat"


def test_bare_sku_name_with_literal_underscore_still_works_after_normalization_fix():
    """Regression: the normalization fix must not break the pre-existing
    literal-underscore-form match."""
    req = detect_query_requirements(
        "What was the revenue of GPI_Franchise_1 Pack 1 in Gujarat in April 2026?",
        known_sku_names=_KNOWN_NAMES,
    )
    assert req.required_granularity == "sku"


def test_comparison_question_with_space_form_sku_name_triggers_sku_granularity():
    req = detect_query_requirements(
        "Compare Marlboro Pack 1 and GPI Franchise 1 Pack 1 in Gujarat in April 2026.",
        known_sku_names=_KNOWN_NAMES,
    )
    assert req.required_granularity == "sku"


def test_space_underscore_normalization_does_not_cause_false_positive_matches():
    """The normalization must not loosen matching enough to spuriously
    match an unrelated question -- only a question containing the FULL
    normalized name (all words, in order) should match."""
    req = detect_query_requirements(
        "Why did Bihar's Service Level decline in November 2025?", known_sku_names=_KNOWN_NAMES
    )
    assert req.required_granularity is None
