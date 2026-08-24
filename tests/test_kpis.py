"""
Tests for the Phase 3 KPI computation module.

Like the Phase 2 data tests, these don't check exact numbers (the
underlying data is randomly generated) -- they check invariants that
must hold no matter what the random draws were: ratios stay within
[0, 1] where a ratio is expected, denominators are never zero, and the
grain of each output table is exactly what we designed (one row per
State x Month, or State x Month x Category).
"""

import numpy as np
import pandas as pd
import pytest

from src.data_gen.generate_synthetic_data import generate_all, INDIAN_STATES, N_MONTHS
from src.kpis.compute_kpis import (
    build_kpi_state_month,
    build_kpi_state_month_category,
    build_kpi_state_month_sku,
)

SEED = 42


@pytest.fixture(scope="module")
def tables():
    """Reuse one generated dataset across every test in this file --
    generation is the expensive part, not the KPI math."""
    return generate_all(SEED)


@pytest.fixture(scope="module")
def kpi_state_month(tables):
    return build_kpi_state_month(tables)


@pytest.fixture(scope="module")
def kpi_state_month_category(tables):
    return build_kpi_state_month_category(tables)


@pytest.fixture(scope="module")
def kpi_state_month_sku(tables):
    return build_kpi_state_month_sku(tables)


def test_state_month_grain_is_complete(kpi_state_month):
    """Every state should report every month -- 28 states x 24 months,
    no gaps and no duplicates."""
    assert len(kpi_state_month) == len(INDIAN_STATES) * N_MONTHS
    assert not kpi_state_month.duplicated(subset=["state_name", "month"]).any()
    assert set(kpi_state_month["state_name"]) == set(INDIAN_STATES)


def test_state_month_category_grain_is_complete(kpi_state_month_category):
    """Every state x month x category combination should appear exactly
    once -- 28 states x 24 months x 4 categories."""
    expected_categories = {"GPI", "IPM", "Ferrero", "Candy"}
    assert len(kpi_state_month_category) == len(INDIAN_STATES) * N_MONTHS * len(expected_categories)
    assert not kpi_state_month_category.duplicated(subset=["state_name", "month", "category_name"]).any()
    assert set(kpi_state_month_category["category_name"]) == expected_categories


def test_no_missing_values(kpi_state_month, kpi_state_month_category):
    """Every state/month/category combination has real data behind it
    (every state has WDs, outlets and visits), so nothing should be NaN
    in either output table."""
    assert kpi_state_month.isna().sum().sum() == 0
    assert kpi_state_month_category.isna().sum().sum() == 0


def test_ratio_kpis_are_bounded_zero_to_one(kpi_state_month, kpi_state_month_category):
    """Productivity, Service Level, Numeric Distribution, ACV and OOS%
    are all "part over whole" ratios -- they must land in [0, 1]. (Range
    Billing and Inventory Turns are NOT bounded by 1, so they're
    excluded here.)"""
    for col in ["productivity", "service_level"]:
        assert kpi_state_month[col].between(0, 1).all(), f"{col} escaped [0, 1]"

    for col in ["numeric_distribution", "acv", "oos_pct"]:
        assert kpi_state_month_category[col].between(0, 1).all(), f"{col} escaped [0, 1]"


def test_positive_kpis_are_never_negative(kpi_state_month, kpi_state_month_category):
    """SKUs/Transaction, Dropsize, Inventory Turns/Days and Range Billing
    aren't ratios capped at 1, but they can never be negative or zero
    given every state has real visits/orders/inventory in every month."""
    for col in ["skus_per_transaction", "dropsize", "inventory_turns", "inventory_days"]:
        assert (kpi_state_month[col] > 0).all(), f"{col} was <= 0"

    assert (kpi_state_month_category["range_billing"] > 0).all()


def test_state_performance_factor_produces_real_spread(kpi_state_month):
    """Phase 2 seeded a per-state performance factor specifically so
    states differ consistently in order-placement rate. If every
    state's average productivity came out identical, that signal would
    have been lost somewhere in the KPI rollup."""
    state_avg = kpi_state_month.groupby("state_name")["productivity"].mean()
    assert state_avg.max() - state_avg.min() > 0.05, "no meaningful state-to-state spread in productivity"


def test_kpi_computation_is_reproducible(tables):
    """Running the KPI computation twice on the same input tables must
    give identical output -- there's no randomness in this module, only
    in the upstream data generation."""
    first = build_kpi_state_month(tables)
    second = build_kpi_state_month(tables)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Table E: State x Month x SKU (Problem 1 -- SKU-level business diagnostics)
# ---------------------------------------------------------------------------


def test_state_month_sku_grain_is_complete(kpi_state_month_sku, tables):
    """Every state x month x SKU combination should appear exactly once --
    28 states x 24 months x (however many SKUs this seed's product
    catalogue produced), no gaps and no duplicates."""
    n_skus = tables["products"]["sku_id"].nunique()
    assert len(kpi_state_month_sku) == len(INDIAN_STATES) * N_MONTHS * n_skus
    assert not kpi_state_month_sku.duplicated(subset=["state_name", "month", "sku_id"]).any()
    assert set(kpi_state_month_sku["sku_id"]) == set(tables["products"]["sku_id"])


def test_state_month_sku_no_missing_values(kpi_state_month_sku):
    assert kpi_state_month_sku.isna().sum().sum() == 0


def test_state_month_sku_ratio_kpis_are_bounded_zero_to_one(kpi_state_month_sku):
    """Service Level, Numeric Distribution and OOS% are the same KPI
    DEFINITIONS as Table A/B, just re-scoped to one SKU -- they must stay
    bounded in [0, 1] here too."""
    for col in ["service_level", "numeric_distribution", "oos_pct"]:
        assert kpi_state_month_sku[col].between(0, 1).all(), f"{col} escaped [0, 1]"


def test_state_month_sku_volume_kpis_are_never_negative(kpi_state_month_sku):
    for col in ["qty_ordered", "qty_delivered", "revenue", "billed_outlets"]:
        assert (kpi_state_month_sku[col] >= 0).all(), f"{col} went negative"


def test_state_month_sku_revenue_matches_units_times_catalogue_price(kpi_state_month_sku, tables):
    """Revenue is a derived field (delivered units x the order line's own
    unit_price) -- spot check the implied per-unit price against the SKU's
    catalogue price rather than trusting the arithmetic blindly."""
    sample = kpi_state_month_sku[kpi_state_month_sku["qty_delivered"] > 0].iloc[0]
    price = tables["products"].set_index("sku_id").loc[sample["sku_id"], "unit_price"]
    implied_price = sample["revenue"] / sample["qty_delivered"]
    assert implied_price == pytest.approx(price, rel=0.01)


def test_state_month_sku_franchise_category_match_products_catalogue(kpi_state_month_sku, tables):
    """Franchise/category labels must be copied verbatim from products.csv
    -- the same entity-name-drift risk every other table in this project
    guards against."""
    lookup = tables["products"].set_index("sku_id")[["franchise_name", "category_name"]]
    sample = kpi_state_month_sku.iloc[0]
    row = lookup.loc[sample["sku_id"]]
    assert sample["franchise_name"] == row["franchise_name"]
    assert sample["category_name"] == row["category_name"]


def test_state_month_sku_reproducible(tables):
    """No randomness in this module -- computing it twice from the same
    input tables must give identical output."""
    first = build_kpi_state_month_sku(tables)
    second = build_kpi_state_month_sku(tables)
    pd.testing.assert_frame_equal(first, second)


def test_hero_skus_dominate_revenue(kpi_state_month_sku):
    """Sanity check tying back to the Hero-SKU Pareto design (see
    generate_synthetic_data.py's select_hero_skus()): revenue should be
    heavily concentrated in a small number of SKUs, not flat across all of
    them -- otherwise 'top N SKUs' business questions would have no real
    signal to answer from this data."""
    totals = kpi_state_month_sku.groupby("sku_id")["revenue"].sum().sort_values(ascending=False)
    top_3_share = totals.head(3).sum() / totals.sum()
    assert top_3_share > 0.3, "no revenue concentration -- SKU ranking would carry no real signal"
