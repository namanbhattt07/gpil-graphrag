"""
Phase 3 -- KPI Computation for the GPIL GraphRAG project.

WHAT THIS SCRIPT DOES (plain language):
Phase 2 gave us raw "what actually happened" data: individual visits,
individual order lines, individual monthly stock snapshots. Nobody asks
GraphRAG questions about a single order line though -- they ask things
like "how did Punjab's distribution look in March?". This script rolls
the raw rows up into the business KPIs the requirements doc asks for,
at State x month grain (State is GPIL's real top geography level).

WHY TWO OUTPUT TABLES INSTEAD OF ONE:
Some KPIs (Productivity, Service Level, Inventory Turns/Days, Dropsize,
SKUs/Transaction) describe overall S&D health for a state in a month --
they don't refer to any one product category. Others (Numeric
Distribution, ACV, Out-of-Stock %, Range Billing) are only meaningful
*relative to a category's SKU set* -- "distribution" of nothing in
particular isn't a real number. Cramming both kinds into one row would
either duplicate the category-agnostic KPIs across every category row
(confusing) or hide the category split entirely (loses the GPI vs IPM
vs Ferrero vs Candy differences later "why" questions need). So:
    kpi_state_month.csv           - State x Month grain
    kpi_state_month_category.csv  - State x Month x Category grain

ASSUMPTIONS FLAGGED FOR GPIL CONFIRMATION (per the requirements doc's
own instruction to flag rather than silently guess -- Dropsize and Range
Billing definitions vary by company):
  - Dropsize (formerly "ACL"/Average Case Load) is computed here as the
    average QUANTITY (units) ordered per productive visit -- a volume
    metric. This is deliberately different from SKUs/Transaction (a
    line-count metric), since "average line per call" and "average case
    load" are two different things some companies conflate.
  - Range Billing is computed here as: for each outlet that billed
    anything in a category that month, what fraction of that category's
    full SKU range did it buy? Averaged across billed outlets.
  - ACV outlet-tier weights (Gold=3, Silver=2, Bronze=1) are an
    assumption standing in for real sales-volume weights, which we
    don't have without real GPIL data.
All three are marked "NEEDS GPIL CONFIRMATION" in code and in the
Phase 3 report -- treat them as best-guess placeholders, not facts.

Run with:  python -m src.kpis.compute_kpis
(from the project root, with the venv activated)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings

# Reused from Phase 2 so the "which channel types can carry which
# category" business rule stays in exactly one place instead of being
# redefined (and risking drift) here.
from src.data_gen.generate_synthetic_data import CHANNEL_CATEGORY_ELIGIBILITY

# NEEDS GPIL CONFIRMATION: stand-in weights for ACV until we have real
# sales-volume weights per outlet. Gold outlets count 3x as much as
# Bronze, Silver 2x -- a common rough proxy, not a GPIL-confirmed number.
OUTLET_TIER_WEIGHT = {"Gold": 3, "Silver": 2, "Bronze": 1}


def _load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read all six Phase 2 CSVs back into DataFrames, parsing the date
    columns each table actually has so month arithmetic works later."""
    geography = pd.read_csv(data_dir / "geography.csv")
    outlets = pd.read_csv(data_dir / "outlets.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(data_dir / "products.csv", parse_dates=["launch_date"])
    visits = pd.read_csv(data_dir / "visits.csv", parse_dates=["visit_date"])
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    inventory = pd.read_csv(data_dir / "inventory_snapshots.csv", parse_dates=["snapshot_month"])

    return {
        "geography": geography,
        "outlets": outlets,
        "products": products,
        "visits": visits,
        "orders": orders,
        "inventory_snapshots": inventory,
    }


def _month_start(series: pd.Series) -> pd.Series:
    """Collapse a date column down to its calendar month (as a Period),
    so a visit on 2024-08-17 and one on 2024-08-02 land in the same
    'month' bucket for aggregation. pd.to_datetime() normalises both
    CSV-loaded datetime64 columns and the plain Python date objects
    generate_all() returns in-memory (used directly by the test suite)
    to the same dtype before extracting the period.exact dates ko month buckets mein
    convert karta hai taaki monthly KPI nikale ja sake."""
    return pd.to_datetime(series).dt.to_period("M")


def _wd_to_state(geography: pd.DataFrame) -> pd.Series:
    """Build a wd_id -> state_name lookup. Inventory snapshots are keyed
    by wd_id, not state, so every inventory-based KPI needs this to roll
    stock positions up to State grain.Warehouse Distributor ID ko State ke saath connect karta hai taaki 
    inventory ko state level pe aggregate kar sake."""
    wd_rows = geography[geography["unit_type"] == "WD"]
    return wd_rows.set_index("unit_id")["state_name"]


# ---------------------------------------------------------------------------
# Table A: State x Month KPIs (category-agnostic overall health metrics)
# ---------------------------------------------------------------------------

def compute_productivity(visits: pd.DataFrame, outlets: pd.DataFrame) -> pd.DataFrame:
    """
    Productivity (formerly "Strike Rate") = productive (Order Placed)
    visits / total visits. Answers: "of all the calls an SE made, what
    fraction resulted in an order?" -- the single clearest visits-to-orders
    conversion KPI.
    """
    v = visits.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    v["month"] = _month_start(v["visit_date"])

    grouped = v.groupby(["state_name", "month"])
    total_visits = grouped.size().rename("total_visits")
    productive_visits = grouped["visit_outcome"].apply(lambda s: (s == "Order Placed").sum()).rename("productive_visits")

    out = pd.concat([total_visits, productive_visits], axis=1).reset_index()
    out["productivity"] = out["productive_visits"] / out["total_visits"]
    return out


def compute_order_level_kpis(orders: pd.DataFrame, outlets: pd.DataFrame) -> pd.DataFrame:
    """
    Computes, at State x Month grain:
      - SKUs/Transaction = total order lines / total distinct orders
      - Service Level    = total qty delivered / total qty ordered
    Both are pure roll-ups of the order-line table joined to outlets
    for state, and to month via order_date.
    """
    o = orders.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    o["month"] = _month_start(o["order_date"])

    grouped = o.groupby(["state_name", "month"])
    n_lines = grouped.size().rename("total_order_lines")
    n_orders = grouped["order_id"].nunique().rename("total_orders")
    qty_ordered = grouped["qty_ordered"].sum().rename("total_qty_ordered")
    qty_delivered = grouped["qty_delivered"].sum().rename("total_qty_delivered")

    out = pd.concat([n_lines, n_orders, qty_ordered, qty_delivered], axis=1).reset_index()
    out["skus_per_transaction"] = out["total_order_lines"] / out["total_orders"]
    out["service_level"] = out["total_qty_delivered"] / out["total_qty_ordered"]
    return out


def compute_dropsize(orders: pd.DataFrame, outlets: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """
    Dropsize (formerly "ACL"/Average Case Load) -- NEEDS GPIL CONFIRMATION
    on definition. Computed here as: total quantity ordered / number of
    productive (Order Placed) visits, i.e. average UNITS sold per
    successful call. This is a volume metric, deliberately distinct from
    SKUs/Transaction (which counts SKU lines, not units).
    """
    o = orders.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    o["month"] = _month_start(o["order_date"])
    qty_by_group = o.groupby(["state_name", "month"])["qty_ordered"].sum().rename("total_qty_ordered")

    v = visits.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    v["month"] = _month_start(v["visit_date"])
    productive = v[v["visit_outcome"] == "Order Placed"]
    productive_by_group = productive.groupby(["state_name", "month"]).size().rename("productive_visits")

    out = pd.concat([qty_by_group, productive_by_group], axis=1).reset_index()
    out["dropsize"] = out["total_qty_ordered"] / out["productive_visits"]
    return out[["state_name", "month", "dropsize"]]


def compute_inventory_kpis(inventory: pd.DataFrame, geography: pd.DataFrame, days_in_month: int = 30) -> pd.DataFrame:
    """
    Inventory Turns = total qty sold / average stock held that month
                       (average of opening and closing stock).
    Inventory Days  = days_in_month / Turns -- "how many days of stock
                       on hand", the more intuitive way ops people read
                       the same underlying signal.
    Rolled up from WD x SKU x month to State x month via the WD's
    parent state (WDs, not SKUs, are what's geographically located).
    """
    wd_state = _wd_to_state(geography)
    inv = inventory.copy()
    inv["state_name"] = inv["wd_id"].map(wd_state)
    inv["month"] = _month_start(inv["snapshot_month"])
    inv["avg_stock"] = (inv["opening_stock"] + inv["closing_stock"]) / 2

    grouped = inv.groupby(["state_name", "month"])
    qty_sold = grouped["qty_sold"].sum().rename("total_qty_sold")
    avg_stock = grouped["avg_stock"].sum().rename("total_avg_stock")

    out = pd.concat([qty_sold, avg_stock], axis=1).reset_index()
    out["inventory_turns"] = out["total_qty_sold"] / out["total_avg_stock"]
    out["inventory_days"] = days_in_month / out["inventory_turns"]
    return out


def build_kpi_state_month(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table A by joining every category-agnostic KPI piece on
    (state_name, month)."""
    productivity = compute_productivity(tables["visits"], tables["outlets"])
    order_kpis = compute_order_level_kpis(tables["orders"], tables["outlets"])
    dropsize = compute_dropsize(tables["orders"], tables["outlets"], tables["visits"])
    inventory_kpis = compute_inventory_kpis(tables["inventory_snapshots"], tables["geography"])

    out = productivity[["state_name", "month", "productivity"]]
    out = out.merge(order_kpis[["state_name", "month", "skus_per_transaction", "service_level"]],
                     on=["state_name", "month"], how="outer")
    out = out.merge(dropsize, on=["state_name", "month"], how="outer")
    out = out.merge(inventory_kpis[["state_name", "month", "inventory_turns", "inventory_days"]],
                     on=["state_name", "month"], how="outer")

    out = out.sort_values(["state_name", "month"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Table B: State x Month x Category KPIs (product-scoped distribution KPIs)
# ---------------------------------------------------------------------------

def _eligible_outlets_by_category(outlets: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the outlets table into one row per (outlet, category) for
    every category that outlet's channel_type is allowed to carry
    (using the same eligibility rule Phase 2 used to decide what an
    outlet could order). This is the denominator population for
    Numeric Distribution and ACV.
    """
    rows = []
    for channel, categories in CHANNEL_CATEGORY_ELIGIBILITY.items():
        subset = outlets[(outlets["channel_type"] == channel) & (outlets["is_active"])]
        for category in categories:
            expanded = subset[["outlet_id", "state_name", "outlet_tier"]].copy()
            expanded["category_name"] = category
            rows.append(expanded)
    return pd.concat(rows, ignore_index=True)


def compute_distribution_kpis(orders: pd.DataFrame, outlets: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric Distribution (ND) = outlets that billed >=1 SKU of the
        category that month / outlets eligible to carry that category.
    ACV = same ratio, but each outlet counted by its tier weight
        (Gold/Silver/Bronze -- NEEDS GPIL CONFIRMATION on real weights)
        instead of counted equally, so a few big Gold outlets billing
        can lift ACV even while raw outlet-count ND stays flat.
    """
    eligible = _eligible_outlets_by_category(outlets)
    eligible["tier_weight"] = eligible["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    eligible_counts = eligible.groupby(["state_name", "category_name"]).agg(
        eligible_outlets=("outlet_id", "nunique"),
        eligible_weight=("tier_weight", "sum"),
    ).reset_index()

    o = orders.merge(outlets[["outlet_id", "state_name", "outlet_tier"]], on="outlet_id", how="left")
    o = o.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])
    o["tier_weight"] = o["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    billed = o.groupby(["state_name", "month", "category_name", "outlet_id"]).agg(
        tier_weight=("tier_weight", "first"),
    ).reset_index()
    billed_counts = billed.groupby(["state_name", "month", "category_name"]).agg(
        billed_outlets=("outlet_id", "nunique"),
        billed_weight=("tier_weight", "sum"),
    ).reset_index()

    out = billed_counts.merge(eligible_counts, on=["state_name", "category_name"], how="left")
    out["numeric_distribution"] = out["billed_outlets"] / out["eligible_outlets"]
    out["acv"] = out["billed_weight"] / out["eligible_weight"]
    return out


def compute_oos_pct(inventory: pd.DataFrame, geography: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Out-of-Stock % = share of WD x SKU x month snapshots flagged
    stockout_flag=True, rolled up to State x month x category (via
    the SKU's category and the WD's parent state).
    """
    wd_state = _wd_to_state(geography)
    inv = inventory.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    inv["state_name"] = inv["wd_id"].map(wd_state)
    inv["month"] = _month_start(inv["snapshot_month"])

    grouped = inv.groupby(["state_name", "month", "category_name"])
    total_snapshots = grouped.size().rename("total_snapshots")
    stockout_snapshots = grouped["stockout_flag"].sum().rename("stockout_snapshots")

    out = pd.concat([total_snapshots, stockout_snapshots], axis=1).reset_index()
    out["oos_pct"] = out["stockout_snapshots"] / out["total_snapshots"]
    return out


def compute_range_billing(orders: pd.DataFrame, outlets: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Range Billing -- NEEDS GPIL CONFIRMATION on definition. Computed
    here as: for each outlet that billed anything in a category that
    month, what fraction of that category's full SKU range did it buy
    (distinct SKUs billed / SKUs available in the category), then
    averaged across all outlets that billed anything in that category
    that month. Measures assortment DEPTH among active buyers, as
    opposed to Numeric Distribution which measures assortment BREADTH
    (how many outlets buy at all).
    """
    skus_per_category = products.groupby("category_name")["sku_id"].nunique().rename("skus_in_category")

    o = orders.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    o = o.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])

    per_outlet = o.groupby(["state_name", "month", "category_name", "outlet_id"])["sku_id"].nunique().rename(
        "distinct_skus_billed").reset_index()
    per_outlet = per_outlet.merge(skus_per_category, on="category_name", how="left")
    per_outlet["outlet_range_ratio"] = per_outlet["distinct_skus_billed"] / per_outlet["skus_in_category"]

    out = per_outlet.groupby(["state_name", "month", "category_name"])["outlet_range_ratio"].mean().rename(
        "range_billing").reset_index()
    return out


def build_kpi_state_month_category(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table B by joining every category-scoped KPI piece on
    (state_name, month, category_name)."""
    dist = compute_distribution_kpis(tables["orders"], tables["outlets"], tables["products"])
    oos = compute_oos_pct(tables["inventory_snapshots"], tables["geography"], tables["products"])
    range_billing = compute_range_billing(tables["orders"], tables["outlets"], tables["products"])

    out = dist[["state_name", "month", "category_name", "numeric_distribution", "acv"]]
    out = out.merge(oos[["state_name", "month", "category_name", "oos_pct"]],
                     on=["state_name", "month", "category_name"], how="outer")
    out = out.merge(range_billing, on=["state_name", "month", "category_name"], how="outer")

    out = out.sort_values(["state_name", "month", "category_name"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Phase 2's CSVs and compute both KPI tables."""
    tables = _load_tables(data_dir)
    return {
        "kpi_state_month": build_kpi_state_month(tables),
        "kpi_state_month_category": build_kpi_state_month_category(tables),
    }


def main() -> None:
    settings = get_settings()
    data_dir = settings.project_root / "data"
    kpis = compute_all(data_dir)

    for name, df in kpis.items():
        path = data_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}.csv: {len(df):,} rows")


if __name__ == "__main__":
    main()
