"""
Post-processing fix — re-rank outlet_tier (Gold/Silver/Bronze) by ACTUAL
sales performance, instead of the random label it was given at outlet
creation time.

WHY THIS EXISTS:
generate_synthetic_data.py currently assigns outlet_tier to every outlet
BEFORE any order exists (see build_outlets(), `tier = rng.choice(...)`),
purely by drawing from TIER_WEIGHTS_BY_COUNT. That label is then fed into
build_orders() as a volume multiplier, which shapes the AGGREGATE tier-level
volume share correctly (Gold ~65% of units from ~20% of outlets, etc.) —
but individual outlets still have a lot of other randomness stacked on top
(channel multiplier, Hero-SKU boost, per-state factor, per-visit noise), so
a single outlet's realized qty_delivered doesn't reliably match the label it
was randomly given at birth. That breaks the basic meaning of "Gold should
outsell Silver should outsell Bronze" at the individual-outlet level, which
Phase 6/7's diagnostic questions rely on.

THE FIX (does NOT touch any quantity/volume numbers, only the tier label):
1. Sum qty_delivered per outlet_id across the existing orders.csv (already
   generated — this script does not regenerate orders/visits).
2. Within each channel_type separately (tiers are a within-channel rank,
   never compared across channels), rank outlets by that total, highest
   first.
3. Re-label the top 20% Gold, the next 35% Silver, the bottom 45% Bronze
   (TIER_WEIGHTS_BY_COUNT from generate_synthetic_data.py, reused here so
   the two files can't silently drift apart).
4. Overwrite outlet_tier in outlets.csv with the new, performance-based
   label. Outlets with zero orders (inactive, or simply never billed)
   naturally rank last within their channel and land in Bronze.

Run with (from the project root, venv activated):
    python -u -m src.data_gen.reassign_outlet_tiers
The `-u` flag turns off Python's stdout buffering so the STEP prints below
show up in real time instead of appearing all at once at the end — useful
here because Step 1 reads a ~4.7GB orders.csv in chunks and can take a
few minutes.
"""

import shutil

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.data_gen.generate_synthetic_data import OUTLET_TIERS, TIER_WEIGHTS_BY_COUNT

# How many order rows to hold in memory at once while summing qty_delivered.
# orders.csv is far too large (~4.7GB, ~54M rows) to load in one go, so we
# stream it in chunks and keep only a running per-outlet total, never the
# raw rows.
ORDERS_CHUNK_SIZE = 5_000_000


def sum_qty_delivered_by_outlet(orders_path) -> pd.Series:
    """
    Stream orders.csv in chunks and return total qty_delivered per
    outlet_id, as a pandas Series indexed by outlet_id.

    We only read the two columns we actually need (outlet_id, qty_delivered)
    — orders.csv has 10 columns total, and skipping the other 8 (order_id,
    line_id, visit_id, order_date, wd_id, sku_id, qty_ordered, unit_price)
    roughly halves the memory/IO cost of each chunk.
    """
    totals = pd.Series(dtype="int64")
    rows_seen = 0

    reader = pd.read_csv(
        orders_path,
        usecols=["outlet_id", "qty_delivered"],
        dtype={"outlet_id": "string", "qty_delivered": "int64"},
        chunksize=ORDERS_CHUNK_SIZE,
    )
    for chunk_num, chunk in enumerate(reader, start=1):
        chunk_totals = chunk.groupby("outlet_id", sort=False)["qty_delivered"].sum()
        # .add(..., fill_value=0) merges this chunk's per-outlet sums into
        # the running total, treating an outlet missing from either side as 0
        # (an outlet may simply not appear in every chunk).
        totals = totals.add(chunk_totals, fill_value=0)
        rows_seen += len(chunk)
        print(f"    ...chunk {chunk_num}: {rows_seen:,} order lines read so far")

    return totals.astype("int64")


def assign_tier_by_rank(group: pd.DataFrame) -> pd.Series:
    """
    Within a single channel_type's outlets, rank by total_units (highest
    first) and cut into Gold/Silver/Bronze using TIER_WEIGHTS_BY_COUNT as
    percentile boundaries. Ties (e.g. many outlets with 0 units) are broken
    by row order (method="first"), which only matters for exactly which
    zero-volume outlet lands just inside vs. just outside the Bronze cutoff
    — it never affects the Gold/Silver/Bronze ORDERING itself.
    """
    # pct_rank: smallest value = highest total_units (rank 1 of n), largest
    # value (1.0) = lowest total_units. This turns "top 20% by count" into
    # a simple pct_rank <= 0.20 threshold check.
    pct_rank = group["total_units"].rank(method="first", ascending=False, pct=True)

    gold_cutoff = TIER_WEIGHTS_BY_COUNT[0]
    silver_cutoff = gold_cutoff + TIER_WEIGHTS_BY_COUNT[1]

    tier = np.select(
        [pct_rank <= gold_cutoff, pct_rank <= silver_cutoff],
        [OUTLET_TIERS[0], OUTLET_TIERS[1]],
        default=OUTLET_TIERS[2],
    )
    return pd.Series(tier, index=group.index)


def print_verification_tables(outlets: pd.DataFrame) -> None:
    """
    Print the two tables the fix must satisfy:
      1. Per-outlet AVERAGE units, sorted descending, for every
         (channel_type, outlet_tier) combination.
      2. OVERALL TOTAL units, sorted descending, for the same groups.
    Both are printed sorted so the ordering can be read off directly and
    compared against the expected trend by eye.
    """
    grouped = outlets.groupby(["channel_type", "outlet_tier"])["total_units"]

    print("\n--- Per-outlet AVERAGE units (channel x tier), sorted descending ---")
    avg_table = grouped.mean().sort_values(ascending=False)
    for (channel, tier), value in avg_table.items():
        print(f"    {tier:<7} {channel:<13} avg/outlet = {value:,.1f}")

    print("\n--- OVERALL TOTAL units (channel x tier), sorted descending ---")
    total_table = grouped.sum().sort_values(ascending=False)
    for (channel, tier), value in total_table.items():
        print(f"    {tier:<7} {channel:<13} total = {value:,.0f}")


def main() -> None:
    settings = get_settings()
    data_dir = settings.project_root / "data"
    orders_path = data_dir / "orders.csv"
    outlets_path = data_dir / "outlets.csv"

    print("[STEP 1] Computing total units (qty_delivered) per outlet...")
    total_units = sum_qty_delivered_by_outlet(orders_path)
    print(f"[STEP 1] Done — {len(total_units):,} outlets have at least one order line.")

    print("[STEP 2] Ranking outlets within each channel...")
    outlets = pd.read_csv(outlets_path, dtype=str)
    # Outlets with zero order lines never showed up in orders.csv, so they're
    # missing from total_units — .map(...).fillna(0) gives them an explicit
    # 0, which correctly ranks them last (Bronze) within their channel.
    outlets["total_units"] = outlets["outlet_id"].map(total_units).fillna(0).astype("int64")
    print("[STEP 2] Done — total_units attached to every outlet row.")

    print("[STEP 3] Assigning tiers based on percentile cutoffs...")
    old_tier = outlets["outlet_tier"].copy()
    outlets["outlet_tier"] = (
        outlets.groupby("channel_type", group_keys=False)
        .apply(assign_tier_by_rank, include_groups=False)
    )
    changed = (outlets["outlet_tier"] != old_tier).sum()
    print(f"[STEP 3] Done — {changed:,} of {len(outlets):,} outlets got a new tier label "
          f"({changed / len(outlets):.1%}).")

    print("[STEP 4] Updating outlets.csv with new tier assignments...")
    # Back up the pre-fix file first — outlets.csv is the output of a
    # ~30-min generation run, so a mistake here should be recoverable
    # without re-running generate_synthetic_data.py from scratch.
    backup_path = outlets_path.with_suffix(".csv.bak")
    shutil.copyfile(outlets_path, backup_path)
    print(f"    (backed up pre-fix file to {backup_path})")

    # total_units was only needed to compute the rank — it was never a
    # column in outlets.csv and shouldn't become one now.
    outlets.drop(columns=["total_units"]).to_csv(outlets_path, index=False)
    print(f"[STEP 4] Done — {outlets_path} overwritten with re-ranked outlet_tier.")

    print("[STEP 5] Verification — printing per-outlet and overall tables...")
    print_verification_tables(outlets)
    print("[STEP 5] Done.")


if __name__ == "__main__":
    main()
