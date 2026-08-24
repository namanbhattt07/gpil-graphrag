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
            expanded["channel_type"] = channel
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


# ---------------------------------------------------------------------------
# Table B2 (Step 3, GraphRAG doc pipeline): State x Month x Category x
# Channel Type KPIs -- same category-scoped formulas as Table B, with
# channel_type added as an extra grouping key, so documents can eventually
# compare e.g. Retail vs Dealer channel performance within a
# state/category/month.
#
# OOS% is deliberately NOT included here. Its only source,
# inventory_snapshots.csv, is keyed by (wd_id, sku_id, month) -- a
# distributor's warehouse stock position -- and carries no outlet_id or
# channel_type at all. A WD's stock isn't attributable to any one channel
# it serves downstream, so there is no real per-channel OOS% to compute
# from current data; inventing one would mean redefining the KPI on a
# dimension it doesn't have, which this step's brief explicitly rules out.
# ---------------------------------------------------------------------------

def compute_distribution_kpis_channel(
    orders: pd.DataFrame, outlets: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """
    Same Numeric Distribution / ACV formulas as compute_distribution_kpis(),
    grouped by (state_name, month, category_name, channel_type) instead of
    (state_name, month, category_name). _eligible_outlets_by_category()
    already knows each row's channel_type (that's how it applies
    CHANNEL_CATEGORY_ELIGIBILITY), so this only adds channel_type to the
    two groupby calls -- the eligible/billed counting logic is unchanged.
    """
    eligible = _eligible_outlets_by_category(outlets)
    eligible["tier_weight"] = eligible["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    eligible_counts = eligible.groupby(["state_name", "category_name", "channel_type"]).agg(
        eligible_outlets=("outlet_id", "nunique"),
        eligible_weight=("tier_weight", "sum"),
    ).reset_index()

    o = orders.merge(outlets[["outlet_id", "state_name", "outlet_tier", "channel_type"]], on="outlet_id", how="left")
    o = o.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])
    o["tier_weight"] = o["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    billed = o.groupby(["state_name", "month", "category_name", "channel_type", "outlet_id"]).agg(
        tier_weight=("tier_weight", "first"),
    ).reset_index()
    billed_counts = billed.groupby(["state_name", "month", "category_name", "channel_type"]).agg(
        billed_outlets=("outlet_id", "nunique"),
        billed_weight=("tier_weight", "sum"),
    ).reset_index()

    out = billed_counts.merge(eligible_counts, on=["state_name", "category_name", "channel_type"], how="left")
    out["numeric_distribution"] = out["billed_outlets"] / out["eligible_outlets"]
    out["acv"] = out["billed_weight"] / out["eligible_weight"]
    return out


def compute_range_billing_channel(
    orders: pd.DataFrame, outlets: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """Same Range Billing formula as compute_range_billing(), grouped by
    (state_name, month, category_name, channel_type) instead of
    (state_name, month, category_name)."""
    skus_per_category = products.groupby("category_name")["sku_id"].nunique().rename("skus_in_category")

    o = orders.merge(outlets[["outlet_id", "state_name", "channel_type"]], on="outlet_id", how="left")
    o = o.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])

    per_outlet = o.groupby(["state_name", "month", "category_name", "channel_type", "outlet_id"])["sku_id"].nunique().rename(
        "distinct_skus_billed").reset_index()
    per_outlet = per_outlet.merge(skus_per_category, on="category_name", how="left")
    per_outlet["outlet_range_ratio"] = per_outlet["distinct_skus_billed"] / per_outlet["skus_in_category"]

    out = per_outlet.groupby(["state_name", "month", "category_name", "channel_type"])["outlet_range_ratio"].mean().rename(
        "range_billing").reset_index()
    return out


def build_kpi_state_month_channel(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table B2 by joining the channel-scoped KPI pieces on
    (state_name, month, category_name, channel_type). No OOS% column --
    see the module comment above compute_distribution_kpis_channel."""
    dist = compute_distribution_kpis_channel(tables["orders"], tables["outlets"], tables["products"])
    range_billing = compute_range_billing_channel(tables["orders"], tables["outlets"], tables["products"])

    out = dist[["state_name", "month", "category_name", "channel_type", "numeric_distribution", "acv"]]
    out = out.merge(range_billing, on=["state_name", "month", "category_name", "channel_type"], how="outer")

    out = out.sort_values(["state_name", "month", "category_name", "channel_type"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
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
# Table C: WD (distributor) x Month KPIs (category-agnostic overall health
# metrics, same formulas as Table A but rolled up to the individual
# distributor instead of the whole state -- this is what unlocks "why did
# this specific distributor underperform" questions).
# ---------------------------------------------------------------------------

def _wd_info(geography: pd.DataFrame) -> pd.DataFrame:
    """wd_id -> (wd_name, state_name) lookup, purely for attaching
    descriptive context columns to WD-grain KPI tables (not part of the
    grouping grain itself)."""
    wd_rows = geography[geography["unit_type"] == "WD"]
    return wd_rows.rename(columns={"unit_id": "wd_id", "unit_name": "wd_name"})[
        ["wd_id", "wd_name", "state_name"]
    ]


def compute_productivity_wd(visits: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    """
    Same Productivity formula as compute_productivity(), grouped by
    (wd_id, month) instead of (state_name, month). visits.csv only carries
    se_id, not wd_id, so each visit is attributed to a distributor via the
    SE's parent_unit_id in geography.csv (SE -> WD).
    """
    se_to_wd = geography[geography["unit_type"] == "SE"].set_index("unit_id")["parent_unit_id"]
    v = visits.copy()
    v["wd_id"] = v["se_id"].map(se_to_wd)
    v["month"] = _month_start(v["visit_date"])

    grouped = v.groupby(["wd_id", "month"])
    total_visits = grouped.size().rename("total_visits")
    productive_visits = grouped["visit_outcome"].apply(lambda s: (s == "Order Placed").sum()).rename("productive_visits")

    out = pd.concat([total_visits, productive_visits], axis=1).reset_index()
    out["productivity"] = out["productive_visits"] / out["total_visits"]
    return out


def compute_order_level_kpis_wd(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Same SKUs/Transaction and Service Level formulas as
    compute_order_level_kpis(), grouped by (wd_id, month) using the wd_id
    orders.csv already carries on every line (verified identical to the
    outlet's own wd_id on outlets.csv, so no join is needed here).
    """
    o = orders.copy()
    o["month"] = _month_start(o["order_date"])

    grouped = o.groupby(["wd_id", "month"])
    n_lines = grouped.size().rename("total_order_lines")
    n_orders = grouped["order_id"].nunique().rename("total_orders")
    qty_ordered = grouped["qty_ordered"].sum().rename("total_qty_ordered")
    qty_delivered = grouped["qty_delivered"].sum().rename("total_qty_delivered")

    out = pd.concat([n_lines, n_orders, qty_ordered, qty_delivered], axis=1).reset_index()
    out["skus_per_transaction"] = out["total_order_lines"] / out["total_orders"]
    out["service_level"] = out["total_qty_delivered"] / out["total_qty_ordered"]
    return out


def compute_dropsize_wd(
    orders: pd.DataFrame, visits: pd.DataFrame, geography: pd.DataFrame
) -> pd.DataFrame:
    """Same Dropsize formula as compute_dropsize(), grouped by (wd_id, month).
    orders.csv already carries wd_id on every line, so no outlets join is
    needed for the order side."""
    o = orders.copy()
    o["month"] = _month_start(o["order_date"])
    qty_by_group = o.groupby(["wd_id", "month"])["qty_ordered"].sum().rename("total_qty_ordered")

    se_to_wd = geography[geography["unit_type"] == "SE"].set_index("unit_id")["parent_unit_id"]
    v = visits.copy()
    v["wd_id"] = v["se_id"].map(se_to_wd)
    v["month"] = _month_start(v["visit_date"])
    productive = v[v["visit_outcome"] == "Order Placed"]
    productive_by_group = productive.groupby(["wd_id", "month"]).size().rename("productive_visits")

    out = pd.concat([qty_by_group, productive_by_group], axis=1).reset_index()
    out["dropsize"] = out["total_qty_ordered"] / out["productive_visits"]
    return out[["wd_id", "month", "dropsize"]]


def compute_inventory_kpis_wd(inventory: pd.DataFrame, days_in_month: int = 30) -> pd.DataFrame:
    """
    Same Inventory Turns/Days formulas as compute_inventory_kpis(), grouped
    by (wd_id, month). No state lookup needed here -- inventory_snapshots.csv
    is already keyed by wd_id directly.
    """
    inv = inventory.copy()
    inv["month"] = _month_start(inv["snapshot_month"])
    inv["avg_stock"] = (inv["opening_stock"] + inv["closing_stock"]) / 2

    grouped = inv.groupby(["wd_id", "month"])
    qty_sold = grouped["qty_sold"].sum().rename("total_qty_sold")
    avg_stock = grouped["avg_stock"].sum().rename("total_avg_stock")

    out = pd.concat([qty_sold, avg_stock], axis=1).reset_index()
    out["inventory_turns"] = out["total_qty_sold"] / out["total_avg_stock"]
    out["inventory_days"] = days_in_month / out["inventory_turns"]
    return out


def build_kpi_wd_month(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table C by joining every category-agnostic KPI piece on
    (wd_id, month), mirroring build_kpi_state_month() at WD grain. Adds
    wd_name/state_name as descriptive context columns (not part of the
    grouping grain) so a WD row can still be traced back to its state."""
    productivity = compute_productivity_wd(tables["visits"], tables["geography"])
    order_kpis = compute_order_level_kpis_wd(tables["orders"])
    dropsize = compute_dropsize_wd(tables["orders"], tables["visits"], tables["geography"])
    inventory_kpis = compute_inventory_kpis_wd(tables["inventory_snapshots"])

    out = productivity[["wd_id", "month", "productivity"]]
    out = out.merge(order_kpis[["wd_id", "month", "skus_per_transaction", "service_level"]],
                     on=["wd_id", "month"], how="outer")
    out = out.merge(dropsize, on=["wd_id", "month"], how="outer")
    out = out.merge(inventory_kpis[["wd_id", "month", "inventory_turns", "inventory_days"]],
                     on=["wd_id", "month"], how="outer")

    out = out.merge(_wd_info(tables["geography"]), on="wd_id", how="left")
    out = out[["wd_id", "wd_name", "state_name", "month", "productivity", "skus_per_transaction",
               "service_level", "dropsize", "inventory_turns", "inventory_days"]]

    out = out.sort_values(["state_name", "wd_id", "month"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Table D: WD (distributor) x Month x Category KPIs (product-scoped
# distribution KPIs, same formulas as Table B but rolled up to the
# individual distributor instead of the whole state).
# ---------------------------------------------------------------------------

def _eligible_outlets_by_category_wd(outlets: pd.DataFrame) -> pd.DataFrame:
    """Same eligibility expansion as _eligible_outlets_by_category(), but
    keyed by wd_id instead of state_name."""
    rows = []
    for channel, categories in CHANNEL_CATEGORY_ELIGIBILITY.items():
        subset = outlets[(outlets["channel_type"] == channel) & (outlets["is_active"])]
        for category in categories:
            expanded = subset[["outlet_id", "wd_id", "outlet_tier"]].copy()
            expanded["category_name"] = category
            rows.append(expanded)
    return pd.concat(rows, ignore_index=True)


def compute_distribution_kpis_wd(orders: pd.DataFrame, outlets: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Same ND and ACV formulas as compute_distribution_kpis(), grouped by
    (wd_id, month, category_name)."""
    eligible = _eligible_outlets_by_category_wd(outlets)
    eligible["tier_weight"] = eligible["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    eligible_counts = eligible.groupby(["wd_id", "category_name"]).agg(
        eligible_outlets=("outlet_id", "nunique"),
        eligible_weight=("tier_weight", "sum"),
    ).reset_index()

    o = orders.merge(outlets[["outlet_id", "outlet_tier"]], on="outlet_id", how="left")
    o = o.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])
    o["tier_weight"] = o["outlet_tier"].map(OUTLET_TIER_WEIGHT)

    billed = o.groupby(["wd_id", "month", "category_name", "outlet_id"]).agg(
        tier_weight=("tier_weight", "first"),
    ).reset_index()
    billed_counts = billed.groupby(["wd_id", "month", "category_name"]).agg(
        billed_outlets=("outlet_id", "nunique"),
        billed_weight=("tier_weight", "sum"),
    ).reset_index()

    out = billed_counts.merge(eligible_counts, on=["wd_id", "category_name"], how="left")
    out["numeric_distribution"] = out["billed_outlets"] / out["eligible_outlets"]
    out["acv"] = out["billed_weight"] / out["eligible_weight"]
    return out


def compute_oos_pct_wd(inventory: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Same OOS% formula as compute_oos_pct(), grouped by (wd_id, month,
    category_name). No state lookup needed -- inventory_snapshots.csv is
    already keyed by wd_id directly.
    """
    inv = inventory.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    inv["month"] = _month_start(inv["snapshot_month"])

    grouped = inv.groupby(["wd_id", "month", "category_name"])
    total_snapshots = grouped.size().rename("total_snapshots")
    stockout_snapshots = grouped["stockout_flag"].sum().rename("stockout_snapshots")

    out = pd.concat([total_snapshots, stockout_snapshots], axis=1).reset_index()
    out["oos_pct"] = out["stockout_snapshots"] / out["total_snapshots"]
    return out


def compute_range_billing_wd(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Same Range Billing formula as compute_range_billing(), grouped by
    (wd_id, month, category_name). orders.csv already carries wd_id on
    every line, so no outlets join is needed here."""
    skus_per_category = products.groupby("category_name")["sku_id"].nunique().rename("skus_in_category")

    o = orders.merge(products[["sku_id", "category_name"]], on="sku_id", how="left")
    o["month"] = _month_start(o["order_date"])

    per_outlet = o.groupby(["wd_id", "month", "category_name", "outlet_id"])["sku_id"].nunique().rename(
        "distinct_skus_billed").reset_index()
    per_outlet = per_outlet.merge(skus_per_category, on="category_name", how="left")
    per_outlet["outlet_range_ratio"] = per_outlet["distinct_skus_billed"] / per_outlet["skus_in_category"]

    out = per_outlet.groupby(["wd_id", "month", "category_name"])["outlet_range_ratio"].mean().rename(
        "range_billing").reset_index()
    return out


def build_kpi_wd_month_category(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table D by joining every category-scoped KPI piece on
    (wd_id, month, category_name), mirroring build_kpi_state_month_category()
    at WD grain."""
    dist = compute_distribution_kpis_wd(tables["orders"], tables["outlets"], tables["products"])
    oos = compute_oos_pct_wd(tables["inventory_snapshots"], tables["products"])
    range_billing = compute_range_billing_wd(tables["orders"], tables["products"])

    out = dist[["wd_id", "month", "category_name", "numeric_distribution", "acv"]]
    out = out.merge(oos[["wd_id", "month", "category_name", "oos_pct"]],
                     on=["wd_id", "month", "category_name"], how="outer")
    out = out.merge(range_billing, on=["wd_id", "month", "category_name"], how="outer")

    out = out.merge(_wd_info(tables["geography"]), on="wd_id", how="left")
    out = out[["wd_id", "wd_name", "state_name", "month", "category_name",
               "numeric_distribution", "acv", "oos_pct", "range_billing"]]

    out = out.sort_values(["state_name", "wd_id", "month", "category_name"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Table E: State x Month x SKU KPIs (Problem 1 -- SKU-level business
# diagnostics). Same KPI DEFINITIONS as Tables B/C/D (Numeric Distribution,
# Out-of-Stock %, Service Level), just re-scoped from "any SKU in the
# category" down to ONE SKU -- no new metric is invented here. Adds Units
# Delivered and Revenue (delivered units x the order line's OWN unit_price,
# both real fields already on every orders.csv row -- no price assumption
# made here), since raw sales volume/value is what "top N SKUs" / "which
# SKU sells the most" questions actually ask about, and neither exists at
# any coarser grain (Table B describes distribution/OOS%, never volume).
#
# GRAIN AND SCALE: State x Month x SKU. products.csv has ~59 SKUs total (a
# small, fixed catalogue -- see build_products()), so this table is
# 28 states x 24 months x ~59 SKUs ~= 39,600 rows -- the same order of
# magnitude as kpi_wd_month_category.csv, never a raw-row dump of
# orders.csv's tens of millions of order lines. This is what makes a
# SKU-level narrative document corpus (src/graph/build_sku_documents.py)
# and a deterministic evidence lookup (src/inference/sku_evidence.py)
# tractable without ever feeding a raw order line into anything.
# ---------------------------------------------------------------------------


def _eligible_outlets_by_sku_state(outlets: pd.DataFrame) -> pd.DataFrame:
    """(state_name, category_name) -> count of outlets eligible to carry
    that category -- every SKU in a category shares the same eligible
    population (Numeric Distribution's denominator doesn't vary by SKU
    within a category, only by category, exactly like Table B's ND). Reuses
    _eligible_outlets_by_category() unchanged rather than re-deriving the
    channel/category eligibility rule a second time."""
    eligible = _eligible_outlets_by_category(outlets)
    return (
        eligible.groupby(["state_name", "category_name"])["outlet_id"]
        .nunique()
        .rename("eligible_outlets")
        .reset_index()
    )


def compute_sku_sales_kpis(orders: pd.DataFrame, outlets: pd.DataFrame) -> pd.DataFrame:
    """
    Per (state, month, sku_id): total units ordered/delivered, Service
    Level (delivered/ordered -- same formula as compute_order_level_kpis()),
    Revenue (delivered units x the order line's own unit_price), and the
    count of distinct outlets that billed >=1 unit of that SKU that month
    (the numerator for SKU-level Numeric Distribution below).
    """
    o = orders.merge(outlets[["outlet_id", "state_name"]], on="outlet_id", how="left")
    o["month"] = _month_start(o["order_date"])
    o["revenue"] = o["qty_delivered"] * o["unit_price"]

    grouped = o.groupby(["state_name", "month", "sku_id"])
    qty_ordered = grouped["qty_ordered"].sum().rename("qty_ordered")
    qty_delivered = grouped["qty_delivered"].sum().rename("qty_delivered")
    revenue = grouped["revenue"].sum().rename("revenue")
    billed_outlets = grouped["outlet_id"].nunique().rename("billed_outlets")

    out = pd.concat([qty_ordered, qty_delivered, revenue, billed_outlets], axis=1).reset_index()
    out["service_level"] = out["qty_delivered"] / out["qty_ordered"]
    return out


def compute_sku_oos_pct(inventory: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    """
    Same Out-of-Stock % formula as compute_oos_pct(), grouped by (state,
    month, sku_id) instead of (state, month, category) -- inventory_snapshots
    is already keyed by (wd_id, sku_id, month), so no category join is
    needed to compute this at SKU grain.
    """
    wd_state = _wd_to_state(geography)
    inv = inventory.copy()
    inv["state_name"] = inv["wd_id"].map(wd_state)
    inv["month"] = _month_start(inv["snapshot_month"])

    grouped = inv.groupby(["state_name", "month", "sku_id"])
    total_snapshots = grouped.size().rename("total_snapshots")
    stockout_snapshots = grouped["stockout_flag"].sum().rename("stockout_snapshots")

    out = pd.concat([total_snapshots, stockout_snapshots], axis=1).reset_index()
    out["oos_pct"] = out["stockout_snapshots"] / out["total_snapshots"]
    return out[["state_name", "month", "sku_id", "oos_pct"]]


def build_kpi_state_month_sku(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble Table E: State x Month x SKU. Joins products.csv for the
    SKU's descriptive columns (name/franchise/category -- fixed, don't vary
    by state/month) and the category's eligible-outlet count (for Numeric
    Distribution's denominator), then merges in sales and OOS% KPIs."""
    products = tables["products"][["sku_id", "sku_name", "franchise_name", "category_name"]]
    sales = compute_sku_sales_kpis(tables["orders"], tables["outlets"])
    oos = compute_sku_oos_pct(tables["inventory_snapshots"], tables["geography"])
    eligible = _eligible_outlets_by_sku_state(tables["outlets"])

    out = sales.merge(products, on="sku_id", how="left")
    out = out.merge(eligible, on=["state_name", "category_name"], how="left")
    out = out.merge(oos, on=["state_name", "month", "sku_id"], how="outer")

    out["numeric_distribution"] = out["billed_outlets"] / out["eligible_outlets"]

    out = out[[
        "state_name", "month", "sku_id", "sku_name", "franchise_name", "category_name",
        "qty_ordered", "qty_delivered", "revenue", "service_level",
        "billed_outlets", "eligible_outlets", "numeric_distribution", "oos_pct",
    ]]
    out = out.sort_values(["state_name", "month", "sku_id"]).reset_index(drop=True)
    out["month"] = out["month"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def compute_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Phase 2's CSVs and compute all six KPI tables (State and WD
    grain, each split into a category-agnostic and a category-scoped
    table; the State x Category x Channel Type table added in Step 3 of
    the GraphRAG doc pipeline; and the State x Month x SKU table added for
    SKU-level diagnostics)."""
    tables = _load_tables(data_dir)
    return {
        "kpi_state_month": build_kpi_state_month(tables),
        "kpi_state_month_category": build_kpi_state_month_category(tables),
        "kpi_state_month_channel": build_kpi_state_month_channel(tables),
        "kpi_wd_month": build_kpi_wd_month(tables),
        "kpi_wd_month_category": build_kpi_wd_month_category(tables),
        "kpi_state_month_sku": build_kpi_state_month_sku(tables),
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
