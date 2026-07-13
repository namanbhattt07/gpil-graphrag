"""
Tests for the Phase 2 synthetic data generator.

These don't check exact values (the data is randomly generated) — they
check the things that must always be true regardless of the random seed's
specific draws: reproducibility, hierarchy integrity, and no impossible
values (negative quantities, deliveries exceeding orders, etc.).
"""

import pandas as pd
import pytest

from src.data_gen.generate_synthetic_data import generate_all

SEED = 42


@pytest.fixture(scope="module")
def tables():
    """Generate the dataset once and reuse it across tests in this file —
    generation takes real time (hundreds of thousands of rows), so we
    don't want every test paying that cost separately."""
    return generate_all(SEED)


def test_generation_is_reproducible(tables):
    """Running the generator twice with the same seed must give identical
    tables — Phase 9's evaluation and every other phase depends on this."""
    second = generate_all(SEED)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], second[name])


def test_all_tables_present_and_non_empty(tables):
    expected = {"geography", "outlets", "products", "inventory_snapshots", "visits", "orders"}
    assert set(tables.keys()) == expected
    for name, df in tables.items():
        assert len(df) > 0, f"{name} came back empty"


def test_geography_hierarchy_has_all_levels(tables):
    geo = tables["geography"]
    assert set(geo["unit_type"]) == {"State", "Zone", "WD", "SE"}
    assert (geo[geo["unit_type"] == "State"]["parent_unit_id"].isna()).all()
    assert (geo[geo["unit_type"] != "State"]["parent_unit_id"].notna()).all()


def test_foreign_keys_are_valid(tables):
    geo, outlets, products = tables["geography"], tables["outlets"], tables["products"]
    visits, orders = tables["visits"], tables["orders"]

    assert outlets["se_id"].isin(geo["unit_id"]).all()
    assert visits["outlet_id"].isin(outlets["outlet_id"]).all()
    assert orders["outlet_id"].isin(outlets["outlet_id"]).all()
    assert orders["sku_id"].isin(products["sku_id"]).all()
    assert orders["visit_id"].isin(visits["visit_id"]).all()


def test_order_quantities_are_sane(tables):
    orders = tables["orders"]
    assert (orders["qty_ordered"] >= 0).all()
    assert (orders["qty_delivered"] >= 0).all()
    assert (orders["qty_delivered"] <= orders["qty_ordered"]).all()


def test_inventory_has_no_negative_stock(tables):
    inv = tables["inventory_snapshots"]
    for col in ["opening_stock", "qty_received", "qty_sold", "closing_stock"]:
        assert (inv[col] >= 0).all(), f"{col} went negative"


def test_deliberate_imperfections_are_present(tables):
    """Definition of done requires visible stock-outs and partial
    fulfilment, not a perfectly clean dataset."""
    inv, orders = tables["inventory_snapshots"], tables["orders"]

    assert inv["stockout_flag"].mean() > 0, "no stock-outs were generated"
    partial_rate = (orders["qty_delivered"] < orders["qty_ordered"]).mean()
    assert partial_rate > 0, "no partially-fulfilled orders were generated"
