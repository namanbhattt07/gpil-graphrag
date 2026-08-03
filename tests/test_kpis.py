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
from src.kpis.compute_kpis import build_kpi_state_month, build_kpi_state_month_category

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
