"""
Tests for src/inference/sku_evidence.py (Problems 1 and 3). Uses a small,
hand-built kpi_state_month_sku.csv fixture written to a tmp_path data_dir
-- never the real, large data/kpi_state_month_sku.csv -- so these tests are
fast, deterministic, and independent of whatever the live dataset happens
to contain.
"""

import pandas as pd
import pytest

from src.inference.query_requirements import detect_query_requirements
from src.inference.sku_evidence import (
    evaluate_sku_evidence,
    indexed_range_label,
    known_sku_names,
    load_sku_kpi_table,
    rank_sku_rows,
    render_sku_evidence_document,
    sku_rows_for_state_period,
    sku_rows_for_state_year,
)

FIXTURE_ROWS = [
    # Punjab, two months in 2025, two SKUs each.
    dict(state_name="Punjab", month="2025-01", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1000, qty_delivered=900,
         revenue=90000.0, service_level=0.9, billed_outlets=50, eligible_outlets=100,
         numeric_distribution=0.5, oos_pct=0.05),
    dict(state_name="Punjab", month="2025-01", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=200, qty_delivered=180,
         revenue=18000.0, service_level=0.9, billed_outlets=10, eligible_outlets=100,
         numeric_distribution=0.1, oos_pct=0.2),
    dict(state_name="Punjab", month="2025-02", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1100, qty_delivered=1000,
         revenue=100000.0, service_level=0.909, billed_outlets=55, eligible_outlets=100,
         numeric_distribution=0.55, oos_pct=0.04),
    dict(state_name="Punjab", month="2025-02", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=210, qty_delivered=190,
         revenue=19000.0, service_level=0.905, billed_outlets=12, eligible_outlets=100,
         numeric_distribution=0.12, oos_pct=0.18),
    # Gujarat, one month, for cross-state isolation checks.
    dict(state_name="Gujarat", month="2025-01", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=500, qty_delivered=450,
         revenue=45000.0, service_level=0.9, billed_outlets=20, eligible_outlets=100,
         numeric_distribution=0.2, oos_pct=0.1),
]


@pytest.fixture
def data_dir(tmp_path):
    df = pd.DataFrame(FIXTURE_ROWS)
    df.to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_load_sku_kpi_table_round_trips(data_dir):
    df = load_sku_kpi_table(data_dir)
    assert len(df) == len(FIXTURE_ROWS)
    assert df["month"].dtype == object  # kept as string, never coerced to int


def test_sku_rows_for_state_period_filters_correctly(data_dir):
    df = load_sku_kpi_table(data_dir)
    rows = sku_rows_for_state_period(df, "Punjab", "January 2025")
    assert set(rows["sku_id"]) == {"SKU0001", "SKU0002"}
    assert set(rows["state_name"]) == {"Punjab"}


def test_sku_rows_for_state_period_empty_for_unindexed_month(data_dir):
    df = load_sku_kpi_table(data_dir)
    rows = sku_rows_for_state_period(df, "Punjab", "December 2025")
    assert rows.empty


def test_sku_rows_for_state_year_aggregates_across_months(data_dir):
    df = load_sku_kpi_table(data_dir)
    rows = sku_rows_for_state_year(df, "Punjab", 2025)
    alpha = rows[rows["sku_id"] == "SKU0001"].iloc[0]
    assert alpha["qty_delivered"] == 900 + 1000
    assert alpha["revenue"] == pytest.approx(90000.0 + 100000.0)
    # Service Level recomputed from summed totals, not averaged as a ratio.
    assert alpha["service_level"] == pytest.approx((900 + 1000) / (1000 + 1100))
    assert alpha["n_months"] == 2


def test_sku_rows_for_state_year_empty_for_unindexed_year(data_dir):
    df = load_sku_kpi_table(data_dir)
    rows = sku_rows_for_state_year(df, "Punjab", 2027)
    assert rows.empty


def test_indexed_range_label_reflects_actual_data(data_dir):
    df = load_sku_kpi_table(data_dir)
    assert indexed_range_label(df) == "January 2025 through February 2025"


def test_render_sku_evidence_document_names_state_and_period(data_dir):
    df = load_sku_kpi_table(data_dir)
    rows = sku_rows_for_state_period(df, "Punjab", "January 2025")
    text = render_sku_evidence_document("Punjab", "January 2025", rows)
    assert "Alpha Pack 1" in text
    assert "Beta Pack 1" in text
    assert "in Punjab during January 2025" in text
    assert "Gujarat" not in text  # cross-state isolation


# ---------------------------------------------------------------------------
# evaluate_sku_evidence() -- the end-to-end entry point
# ---------------------------------------------------------------------------


def test_evaluate_sku_evidence_sufficient_for_named_month(data_dir):
    req = detect_query_requirements("Top 3 SKUs in Punjab in January 2025")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "sufficient"
    assert len(result.sku_source_rows) == 1
    assert "Punjab" in result.sku_source_rows[0]["text"]
    assert result.sku_source_rows[0]["id"].startswith("sku-Punjab-")


def test_evaluate_sku_evidence_sufficient_for_bare_year(data_dir):
    req = detect_query_requirements("Top 3 SKUs in Punjab in 2025")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "sufficient"
    assert "Alpha Pack 1" in result.sku_source_rows[0]["text"]


def test_evaluate_sku_evidence_insufficient_for_out_of_range_year(data_dir):
    req = detect_query_requirements("Top 3 SKUs in Punjab in 2027")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "insufficient_scope"
    assert result.sku_source_rows == []
    # Must distinguish "raw dataset has SKU data" from "no rows for this scope".
    assert "DOES contain SKU-level" in result.explanation
    assert "Punjab" in result.explanation
    assert "2027" in result.explanation


def test_evaluate_sku_evidence_insufficient_when_state_not_named(data_dir):
    req = detect_query_requirements("Which SKU has the highest sales in 2025?")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "insufficient_scope"
    assert "does not name" in result.explanation


def test_evaluate_sku_evidence_not_applicable_for_non_sku_question(data_dir):
    req = detect_query_requirements("Why did Bihar's Service Level decline in November 2025?")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "not_applicable"
    assert result.sku_source_rows == []


def test_evaluate_sku_evidence_defaults_to_latest_month_when_no_period_named(data_dir):
    req = detect_query_requirements("Which SKU has the highest sales in Punjab?")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "sufficient"
    assert "February 2025" in result.sku_source_rows[0]["text"]  # latest indexed month
    assert "January 2025" not in result.sku_source_rows[0]["text"]


def test_evaluate_sku_evidence_cross_state_isolation(data_dir):
    """A Gujarat question must never surface Punjab's SKU facts."""
    req = detect_query_requirements("Top SKUs in Gujarat in January 2025")
    result = evaluate_sku_evidence(req, data_dir=data_dir)
    assert result.status == "sufficient"
    text = result.sku_source_rows[0]["text"]
    assert "Gujarat" in text
    # The Gujarat row's own numbers, not Punjab's.
    assert "450 units delivered" in text


# ---------------------------------------------------------------------------
# Deterministic SKU ranking (2026-08-23 stabilization pass, Fix 1: the LLM
# must never compute a top-N/bottom-N ranking itself). A dedicated 4-SKU
# fixture makes "top 2 of 4" a meaningful, non-trivial check.
# ---------------------------------------------------------------------------

RANKING_FIXTURE_ROWS = [
    dict(state_name="Kerala", month="2026-04", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1000, qty_delivered=900,
         revenue=90000.0, service_level=0.90, billed_outlets=50, eligible_outlets=100,
         numeric_distribution=0.50, oos_pct=0.05),
    dict(state_name="Kerala", month="2026-04", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=2000, qty_delivered=1900,
         revenue=190000.0, service_level=0.95, billed_outlets=80, eligible_outlets=100,
         numeric_distribution=0.80, oos_pct=0.02),
    dict(state_name="Kerala", month="2026-04", sku_id="SKU0003", sku_name="Gamma Pack 1",
         franchise_name="Gamma", category_name="IPM", qty_ordered=500, qty_delivered=400,
         revenue=40000.0, service_level=0.80, billed_outlets=20, eligible_outlets=100,
         numeric_distribution=0.20, oos_pct=0.20),
    dict(state_name="Kerala", month="2026-04", sku_id="SKU0004", sku_name="Delta Pack 1",
         franchise_name="Delta", category_name="IPM", qty_ordered=3000, qty_delivered=2950,
         revenue=295000.0, service_level=0.98, billed_outlets=95, eligible_outlets=100,
         numeric_distribution=0.95, oos_pct=0.01),
    # A second month, so multi-period ranking can be tested independently.
    dict(state_name="Kerala", month="2026-05", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1200, qty_delivered=1100,
         revenue=110000.0, service_level=0.92, billed_outlets=55, eligible_outlets=100,
         numeric_distribution=0.55, oos_pct=0.04),
    dict(state_name="Kerala", month="2026-05", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=1800, qty_delivered=1700,
         revenue=170000.0, service_level=0.94, billed_outlets=70, eligible_outlets=100,
         numeric_distribution=0.70, oos_pct=0.03),
]


@pytest.fixture
def ranking_data_dir(tmp_path):
    df = pd.DataFrame(RANKING_FIXTURE_ROWS)
    df.to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_rank_sku_rows_top_n_by_revenue(ranking_data_dir):
    df = load_sku_kpi_table(ranking_data_dir)
    rows = sku_rows_for_state_period(df, "Kerala", "April 2026")
    ranked = rank_sku_rows(rows, "Revenue", "top", 2)
    assert list(ranked["sku_name"]) == ["Delta Pack 1", "Beta Pack 1"]  # 295000 > 190000


def test_rank_sku_rows_bottom_n_by_out_of_stock_rate(ranking_data_dir):
    df = load_sku_kpi_table(ranking_data_dir)
    rows = sku_rows_for_state_period(df, "Kerala", "April 2026")
    ranked = rank_sku_rows(rows, "Out-of-Stock Rate", "bottom", 2)
    assert list(ranked["sku_name"]) == ["Delta Pack 1", "Beta Pack 1"]  # 0.01 < 0.02 lowest OOS


def test_evaluate_sku_evidence_appends_ranking_row_when_unambiguous(ranking_data_dir):
    req = detect_query_requirements("Top 2 SKUs by Revenue in Kerala in April 2026")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    assert result.status == "sufficient"
    # Full unranked evidence row + one ranking row.
    assert len(result.sku_source_rows) == 2
    ranking_row = next(r for r in result.sku_source_rows if r["id"].startswith("sku-ranking-"))
    assert "Rank 1: SKU SKU0004" in ranking_row["text"]
    assert "Rank 2: SKU SKU0002" in ranking_row["text"]
    assert "do not recompute" in ranking_row["text"].lower()
    assert result.ranked_skus is not None
    assert [e["sku_id"] for e in result.ranked_skus] == ["SKU0004", "SKU0002"]
    assert result.ranked_skus[0]["value"] == pytest.approx(295000.0)


def test_evaluate_sku_evidence_no_ranking_row_when_metric_ambiguous(ranking_data_dir):
    """Mirrors pipeline.py's needs_clarification gate: an ambiguous ranking
    ('top 2 SKUs', no metric) must never get a fabricated deterministic
    ranking -- there is nothing correct to compute yet."""
    req = detect_query_requirements("Top 2 SKUs in Kerala in April 2026")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    assert result.status == "sufficient"
    assert len(result.sku_source_rows) == 1  # no ranking row appended
    assert result.ranked_skus is None


def test_evaluate_sku_evidence_no_ranking_row_for_non_ranking_question(ranking_data_dir):
    req = detect_query_requirements("What was the Revenue of that SKU in Kerala in April 2026?")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    assert result.status == "sufficient"
    assert len(result.sku_source_rows) == 1
    assert result.ranked_skus is None


def test_evaluate_sku_evidence_no_comparison_row_for_single_period_ranking(ranking_data_dir):
    """The Rank-1 cross-period comparison row only makes sense with 2+
    resolved periods -- a single-period ranking must not get one."""
    req = detect_query_requirements("Top 2 SKUs by Revenue in Kerala in April 2026")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    comparison_rows = [r for r in result.sku_source_rows if r["id"].startswith("sku-ranking-comparison-")]
    assert comparison_rows == []


# ---------------------------------------------------------------------------
# Deterministic Rank-1 cross-period comparison row (2026-08-23 fix):
# "Did the top-selling SKU in Gujarat change between April 2026 and June
# 2026?" live-failed grounding on BOTH the first draft and the bounded
# retry -- not because either period's own ranking was wrong (each,
# verified independently, was already correct), but because the model
# tried to construct its own cross-period trend claim (restating each
# period's value and/or subtracting them) and got the citation and/or the
# arithmetic wrong. This row gives the model the ALREADY-COMPUTED
# comparison to restate instead.
# ---------------------------------------------------------------------------


RANK1_COMPARISON_FIXTURE_ROWS = [
    # Alpha wins Revenue in BOTH months (1000 -> 1200); Beta wins Units
    # Delivered in January (200) but Alpha overtakes in February (300) --
    # one fixture, two crossing metrics, so both the "no change" and
    # "changed" cases can be tested against real, deterministic ranking
    # output rather than a hand-built ranked_skus list.
    dict(state_name="Rajasthan", month="2026-01", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=110, qty_delivered=100,
         revenue=1000.0, service_level=0.91, billed_outlets=10, eligible_outlets=100,
         numeric_distribution=0.1, oos_pct=0.05),
    dict(state_name="Rajasthan", month="2026-01", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=220, qty_delivered=200,
         revenue=500.0, service_level=0.91, billed_outlets=20, eligible_outlets=100,
         numeric_distribution=0.2, oos_pct=0.05),
    dict(state_name="Rajasthan", month="2026-02", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=330, qty_delivered=300,
         revenue=1200.0, service_level=0.91, billed_outlets=30, eligible_outlets=100,
         numeric_distribution=0.3, oos_pct=0.05),
    dict(state_name="Rajasthan", month="2026-02", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="GPI", qty_ordered=165, qty_delivered=150,
         revenue=600.0, service_level=0.91, billed_outlets=15, eligible_outlets=100,
         numeric_distribution=0.15, oos_pct=0.05),
]


@pytest.fixture
def rank1_comparison_data_dir(tmp_path):
    df = pd.DataFrame(RANK1_COMPARISON_FIXTURE_ROWS)
    df.to_csv(tmp_path / "kpi_state_month_sku.csv", index=False)
    return tmp_path


def test_rank1_comparison_row_states_no_change_and_exact_delta_when_same_sku_wins_both_periods(rank1_comparison_data_dir):
    req = detect_query_requirements("Top 1 SKU by Revenue in Rajasthan in January and February 2026")
    result = evaluate_sku_evidence(req, data_dir=rank1_comparison_data_dir)
    assert result.status == "sufficient"
    comparison_row = next(r for r in result.sku_source_rows if r["id"] == "sku-ranking-comparison-Rajasthan")
    assert "did NOT change" in comparison_row["text"]
    assert "Alpha Pack 1" in comparison_row["text"]
    assert "Rs 1,000" in comparison_row["text"] and "Rs 1,200" in comparison_row["text"]
    assert "a change of +Rs 200" in comparison_row["text"]


def test_rank1_comparison_row_states_changed_when_different_sku_wins_each_period(rank1_comparison_data_dir):
    req = detect_query_requirements("Top 1 SKU by Units Delivered in Rajasthan in January and February 2026")
    result = evaluate_sku_evidence(req, data_dir=rank1_comparison_data_dir)
    comparison_row = next(r for r in result.sku_source_rows if r["id"] == "sku-ranking-comparison-Rajasthan")
    assert "CHANGED" in comparison_row["text"]
    assert "Beta Pack 1" in comparison_row["text"]
    assert "Alpha Pack 1" in comparison_row["text"]
    assert "restate this conclusion exactly, do not recompute it" in comparison_row["text"]


def test_rank1_comparison_row_included_in_deterministic_ranking_block():
    """The comparison row's id ("sku-ranking-comparison-...") shares the
    "sku-ranking-" prefix by design, so it's automatically picked up by
    fact_structuring.build_deterministic_ranking_block() -- no separate
    block-builder or answer.py wiring needed."""
    from src.inference.fact_structuring import build_deterministic_ranking_block
    import pandas as pd

    rows = [
        {"id": "sku-ranking-Kerala-April_2026", "text": "Ranked by Revenue: Rank 1: SKU SKU0004 ...\n"},
        {"id": "sku-ranking-Kerala-May_2026", "text": "Ranked by Revenue: Rank 1: SKU SKU0002 ...\n"},
        {"id": "sku-ranking-comparison-Kerala", "text": "The Rank 1 SKU by Revenue CHANGED ...\n"},
    ]
    block = build_deterministic_ranking_block({"sources": pd.DataFrame(rows)})
    assert "The Rank 1 SKU by Revenue CHANGED" in block


def test_evaluate_sku_evidence_ranking_row_per_resolved_period(ranking_data_dir):
    """Multi-period ranking/comparison: 'top 2 ... in April and May 2026'
    must compute an INDEPENDENT, correctly-scoped ranking for each period,
    not one conflated ranking across both months. Also produces ONE
    deterministic Rank-1 cross-period comparison row (2026-08-23 fix,
    see test_evaluate_sku_evidence_rank1_comparison_row_* below) -- three
    "sku-ranking-"-prefixed rows total: one per period plus the comparison."""
    req = detect_query_requirements("Top 2 SKUs by Revenue in Kerala in April and May 2026")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    assert result.status == "sufficient"
    ranking_rows = [r for r in result.sku_source_rows if r["id"].startswith("sku-ranking-")]
    assert len(ranking_rows) == 3
    per_period_rows = [r for r in ranking_rows if not r["id"].startswith("sku-ranking-comparison-")]
    assert len(per_period_rows) == 2
    april_row = next(r for r in per_period_rows if "April" in r["id"])
    may_row = next(r for r in per_period_rows if "May" in r["id"])
    assert "Rank 1: SKU SKU0004" in april_row["text"]  # Delta only exists in April
    assert "Rank 1: SKU SKU0002" in may_row["text"]  # highest Revenue in May is Beta (170000)
    periods = {e["period"] for e in result.ranked_skus}
    assert periods == {"April 2026", "May 2026"}


def test_evaluate_sku_evidence_ranking_skipped_when_fewer_candidates_than_n(ranking_data_dir):
    """'Top 10 SKUs' when only 4 exist for that state/period -- must not
    fabricate a ranking padded with nonexistent entries; falls back to
    ordinary (unranked) evidence, still fully answerable, just not via the
    deterministic-ranking fast path."""
    req = detect_query_requirements("Top 10 SKUs by Revenue in Kerala in April 2026")
    result = evaluate_sku_evidence(req, data_dir=ranking_data_dir)
    assert result.status == "sufficient"
    assert result.ranked_skus is None
    assert len(result.sku_source_rows) == 1


# ---------------------------------------------------------------------------
# known_sku_names() -- data-derived SKU name lookup for bare-name query
# detection (Fix 2). Never a hardcoded list; always read from the same CSV
# every other SKU-evidence lookup in this module reads.
# ---------------------------------------------------------------------------


def test_known_sku_names_returns_every_distinct_name(data_dir):
    names = known_sku_names(data_dir)
    assert names == {"Alpha Pack 1", "Beta Pack 1"}


def test_known_sku_names_empty_when_file_missing(tmp_path):
    """A data_dir with no kpi_state_month_sku.csv at all must degrade to
    an empty set, never raise -- callers (pipeline.py) call this
    unconditionally on every question, including ones with nothing to do
    with SKUs."""
    assert known_sku_names(tmp_path) == frozenset()
