"""
Phase 2 — Synthetic S&D Data Generator for the GPIL GraphRAG project.

WHAT THIS SCRIPT DOES (plain language):
We don't have access to GPIL's real ERP/distribution data, so this script
INVENTS a realistic-looking stand-in for it: a company-shaped hierarchy
(State -> Zone -> WD -> Sales Executive -> Outlet), a product catalogue
(Category -> Franchise -> SKU), and ~2 years of sales visits, orders and
distributor inventory that follow that hierarchy.

WHY IT LOOKS THE WAY IT DOES:
GPIL's real structure is State -> Zone -> WD (Wholesale Distributor) ->
SE (Sales Executive, who runs a fixed "beat" of retail outlets) -> Outlet.
Applied at GPIL's real ratios (28 states, 3-5 zones/state, ... 50-150
outlets/SE) this would produce ~600,000 outlets. We keep all 28 real
states (cheap, and needed for believable state-vs-state comparison
questions later); zones/state and WD/zone stay shrunk (2 and 2-3), but
SE/WD and outlets/SE were widened back up close to GPIL's real ratios,
landing around ~110,000-120,000 outlets. This is still safe for Phase 5
GraphRAG indexing cost: indexing runs on Phase 4's narrative documents,
which are built one per State x month (28 x 24 = 672 documents) regardless
of how many outlets sit underneath — outlet count only affects how long
Phase 2/3 take to run, not GraphRAG's token spend.

The data is not random noise: it has DELIBERATE, seeded imperfections
(distributor stock-outs, partially-fulfilled orders, festive-season spikes,
state-to-state performance differences) so that Phase 7's "why did X
happen" diagnostic questions actually have real signal to reason over,
per the project requirements doc.

OUTPUT: six CSV files under data/:
    geography.csv           - State/Zone/WD/SE hierarchy, one row per unit
    outlets.csv              - retail outlets, one row per outlet
    products.csv              - Category/Franchise/SKU catalogue
    visits.csv                - one row per SE visit to an outlet
    orders.csv                 - one row per SKU line on an order
    inventory_snapshots.csv    - one row per WD x SKU x month stock position

Run with:  python -m src.data_gen.generate_synthetic_data
(from the project root, with the venv activated)
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from config.settings import get_settings

# ---------------------------------------------------------------------------
# Reference data: things that are fixed facts about GPIL's world, not
# randomly generated. Keeping these as named constants at the top makes it
# easy to see (and change) the shape of the whole dataset in one place.
# ---------------------------------------------------------------------------

# All 28 Indian states (real names, since this is just geography, not a
# business fact we'd need GPIL to confirm).
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]

# Pilot-scale multipliers (shrunk from GPIL's real ratios, see module
# docstring). ZONES_PER_STATE is fixed; the others are (min, max) ranges
# sampled per node so the hierarchy isn't perfectly uniform.
ZONES_PER_STATE = 2
WD_PER_ZONE_RANGE = (2, 3)      # inclusive
SE_PER_WD_RANGE = (5, 15)        # inclusive
OUTLETS_PER_SE_RANGE = (70, 100)  # inclusive

# Retail channel types an outlet can be, with roughly how common each is.
# Paan shops/tobacconists are the classic cigarette outlet in India, hence
# the higher weight.
CHANNEL_TYPES = ["Paan/Tobacconist", "Kirana/General Store", "Modern Trade", "Dealer"]
CHANNEL_WEIGHTS = [0.35, 0.40, 0.15, 0.10]

# Which product categories each channel type typically carries. All four
# channels now carry all four categories (Paan/Tobacconist picked up
# Ferrero too, e.g. Tic Tac/Kinder Joy as impulse-buy counter items) —
# the dict is kept per-channel rather than collapsed to one shared list so
# a future channel-specific restriction is a one-line change, not a rewrite.
CHANNEL_CATEGORY_ELIGIBILITY = {
    "Paan/Tobacconist": ["GPI", "IPM", "Candy", "Ferrero"],
    "Kirana/General Store": ["GPI", "IPM", "Ferrero", "Candy"],
    "Modern Trade": ["GPI", "IPM", "Ferrero", "Candy"],
    "Dealer": ["GPI", "IPM", "Ferrero", "Candy"],
}

# Outlet "weight" tier, used later (Phase 3) to compute ACV as a
# sales-weighted version of numeric distribution. Most outlets are small
# (Bronze); a few big ones (Gold) carry disproportionate volume.
OUTLET_TIERS = ["Gold", "Silver", "Bronze"]
TIER_WEIGHTS = [0.15, 0.35, 0.50]

# Product hierarchy: Category -> Franchise. Franchise names are generic
# placeholders EXCEPT Marlboro (IPM), TicTac and Kinder_Joy (Ferrero),
# which the user named directly rather than us guessing them.
CATEGORY_FRANCHISES = {
    "GPI": [f"GPI_Franchise_{i}" for i in range(1, 7)],       # 6 franchises
    "IPM": ["Marlboro"],                                       # 1 franchise, many SKUs
    "Ferrero": ["TicTac", "Kinder_Joy"],                       # 2 franchises
    "Candy": ["Candy_Franchise_1", "Candy_Franchise_2"],       # 2 franchises
}

# How many SKU variants (pack sizes) each franchise gets. Marlboro gets more
# because the brief specifically calls out "different SKUs like Marlboro".
SKUS_PER_FRANCHISE_RANGE = {
    "GPI": (4, 8),
    "IPM": (10, 14),
    "Ferrero": (3, 5),
    "Candy": (3, 5),
}

# Roughly realistic price bands per category, in INR. Cigarettes are priced
# per pack, confectionery per small unit/multipack.
UNIT_PRICE_RANGE = {
    "GPI": (150, 400),
    "IPM": (250, 500),
    "Ferrero": (10, 150),
    "Candy": (5, 100),
}

# Data spans 24 months ending on the current month, so the dataset always
# feels "up to date" relative to when it's generated.
DATA_END_MONTH = date(2026, 7, 1)
N_MONTHS = 24

# Deliberate imperfections, expressed as base probabilities. These get
# nudged up/down per state and per month by the skew/seasonality factors
# below, so the imperfections aren't uniform — that's what gives the later
# "why" questions something real to latch onto.
BASE_ORDER_PLACED_PROB = 0.58
BASE_CLOSED_PROB = 0.07
STOCKOUT_EVENT_PROB = 0.08          # chance a WD x SKU x month has a supply disruption

# Festive/seasonal multipliers by calendar month number (1=Jan ... 12=Dec).
# Oct/Nov = Diwali season -> confectionery demand spikes; Nov-Jan also sees
# a mild uptick in tobacco sales (winter). Everything else is neutral.
MONTH_SEASONALITY = {m: 1.0 for m in range(1, 13)}
MONTH_SEASONALITY.update({10: 1.25, 11: 1.30, 12: 1.10, 1: 1.10})

# Extra seasonal boost applied only to Ferrero/Candy SKU selection odds
# during the festive months, on top of the general seasonality above.
FESTIVE_CATEGORY_BOOST = {"Ferrero": 1.6, "Candy": 1.6}
FESTIVE_MONTHS = {10, 11}


def month_range(end_month: date, n_months: int) -> list[date]:
    """Return n_months consecutive month-start dates, ending at end_month."""
    periods = pd.period_range(end=pd.Period(end_month, freq="M"), periods=n_months, freq="M")
    return [p.to_timestamp().date() for p in periods]


# ---------------------------------------------------------------------------
# 1. Geography: State -> Zone -> WD -> SE, flattened into one table with a
#    unit_type column and a parent_unit_id, instead of four separate CSVs.
# ---------------------------------------------------------------------------

def build_geography(rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """
    Build the distribution hierarchy as a single flat table: every State,
    Zone, WD and SE is one row, tagged with unit_type, and pointing at its
    parent via parent_unit_id. This is easier to store/join than four
    separate small dimension tables, and later phases can filter by
    unit_type when they only want one level.
    """
    rows = []
    zone_counter = wd_counter = se_counter = 0

    for s_idx, state_name in enumerate(INDIAN_STATES, start=1):
        state_id = f"ST{s_idx:02d}"
        rows.append({
            "unit_id": state_id, "unit_type": "State", "unit_name": state_name,
            "parent_unit_id": None, "state_name": state_name,
        })

        for z in range(1, ZONES_PER_STATE + 1):
            zone_counter += 1
            zone_id = f"ZN{zone_counter:04d}"
            zone_name = f"{state_name} Zone {z}"
            rows.append({
                "unit_id": zone_id, "unit_type": "Zone", "unit_name": zone_name,
                "parent_unit_id": state_id, "state_name": state_name,
            })

            n_wd = rng.integers(WD_PER_ZONE_RANGE[0], WD_PER_ZONE_RANGE[1] + 1)
            for _ in range(n_wd):
                wd_counter += 1
                wd_id = f"WD{wd_counter:04d}"
                # fake.company() gives a plausible distributor business name.
                wd_name = f"{fake.company()} Distributors"
                rows.append({
                    "unit_id": wd_id, "unit_type": "WD", "unit_name": wd_name,
                    "parent_unit_id": zone_id, "state_name": state_name,
                })

                n_se = rng.integers(SE_PER_WD_RANGE[0], SE_PER_WD_RANGE[1] + 1)
                for _ in range(n_se):
                    se_counter += 1
                    se_id = f"SE{se_counter:05d}"
                    se_name = fake.name()  # SE is a person, not a company
                    rows.append({
                        "unit_id": se_id, "unit_type": "SE", "unit_name": se_name,
                        "parent_unit_id": wd_id, "state_name": state_name,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Outlets: retail points, each attached to one SE (and, through the SE,
#    to a WD/Zone/State).
# ---------------------------------------------------------------------------

def build_outlets(geography: pd.DataFrame, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """
    Generate OUTLETS_PER_SE_RANGE retail outlets under every SE. Each
    outlet also carries a denormalised wd_id/zone_id/state_id so downstream
    tables (visits, orders) don't need a 4-way join just to find out which
    state an outlet is in.
    """
    # Build a quick lookup: unit_id -> row, so we can walk SE -> WD -> Zone.
    lookup = geography.set_index("unit_id").to_dict(orient="index")

    se_rows = geography[geography["unit_type"] == "SE"]
    rows = []
    outlet_counter = 0

    for se in se_rows.itertuples():
        wd_id = se.parent_unit_id
        zone_id = lookup[wd_id]["parent_unit_id"]
        state_id = lookup[zone_id]["parent_unit_id"]

        n_outlets = rng.integers(OUTLETS_PER_SE_RANGE[0], OUTLETS_PER_SE_RANGE[1] + 1)
        channel = rng.choice(CHANNEL_TYPES, size=n_outlets, p=CHANNEL_WEIGHTS)
        tier = rng.choice(OUTLET_TIERS, size=n_outlets, p=TIER_WEIGHTS)

        for i in range(n_outlets):
            outlet_counter += 1
            onboarded = fake.date_between(start_date=date(2015, 1, 1), end_date=date(2024, 7, 31))
            rows.append({
                "outlet_id": f"OUT{outlet_counter:06d}",
                "outlet_name": f"{fake.company()}",
                "se_id": se.unit_id,
                "wd_id": wd_id,
                "zone_id": zone_id,
                "state_id": state_id,
                "state_name": se.state_name,
                "channel_type": channel[i],
                "outlet_tier": tier[i],
                "onboarded_date": onboarded,
                # 95% of outlets stay active for the whole data window.
                "is_active": bool(rng.random() > 0.05),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Products: Category -> Franchise -> SKU.
# ---------------------------------------------------------------------------

def build_products(rng: np.random.Generator) -> pd.DataFrame:
    """
    Build the SKU catalogue. Category and Franchise are just descriptive
    columns on each SKU row rather than their own small lookup tables,
    since there are only 4 categories and 11 franchises total.
    """
    rows = []
    sku_counter = 0

    for category, franchises in CATEGORY_FRANCHISES.items():
        lo, hi = SKUS_PER_FRANCHISE_RANGE[category]
        price_lo, price_hi = UNIT_PRICE_RANGE[category]

        for franchise_name in franchises:
            n_skus = rng.integers(lo, hi + 1)
            for pack_idx in range(1, n_skus + 1):
                sku_counter += 1
                launch_date = date(2015 + int(rng.integers(0, 9)), int(rng.integers(1, 13)), 1)
                rows.append({
                    "sku_id": f"SKU{sku_counter:04d}",
                    "sku_name": f"{franchise_name} Pack {pack_idx}",
                    "franchise_name": franchise_name,
                    "category_name": category,
                    "pack_size": f"Variant {pack_idx}",
                    "unit_price": round(float(rng.uniform(price_lo, price_hi)), 2),
                    "launch_date": launch_date,
                    "is_active": True,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Inventory snapshots: WD x SKU x month stock position. Generated BEFORE
#    orders, because a WD's available stock is what limits how well outlet
#    orders can be fulfilled that month (see build_orders below).
# ---------------------------------------------------------------------------

def build_inventory(
    geography: pd.DataFrame, products: pd.DataFrame, months: list[date], rng: np.random.Generator
) -> pd.DataFrame:
    """
    Simulate a simple monthly stock walk for every WD x SKU combination:
    opening stock + received stock - sold stock = closing stock (which
    becomes next month's opening). A random subset of WD x SKU x months
    get a "supply disruption" (qty_received cut sharply), which is what
    creates the stock-outs that later drag down Out-of-Stock % and Service
    Level for that WD in that month.
    """
    wd_ids = geography.loc[geography["unit_type"] == "WD", "unit_id"].tolist()
    sku_ids = products["sku_id"].tolist()

    rows = []
    for wd_id in wd_ids:
        for sku_id in sku_ids:
            opening = float(rng.integers(100, 800))
            for m in months:
                base_supply = float(rng.integers(80, 400))
                disrupted = rng.random() < STOCKOUT_EVENT_PROB
                qty_received = base_supply * (0.2 if disrupted else 1.0)

                # Demand is a noisy fraction of whatever is available; this
                # keeps stock levels wandering instead of exploding or
                # collapsing to zero every month.
                available = opening + qty_received
                demand = available * float(rng.uniform(0.5, 0.95))
                qty_sold = min(available, demand)
                closing = max(available - qty_sold, 0.0)

                rows.append({
                    "snapshot_month": m,
                    "wd_id": wd_id,
                    "sku_id": sku_id,
                    "opening_stock": round(opening, 1),
                    "qty_received": round(qty_received, 1),
                    "qty_sold": round(qty_sold, 1),
                    "closing_stock": round(closing, 1),
                    "stockout_flag": bool(disrupted or closing < 5),
                })
                opening = closing

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Visits: one row per SE call on an outlet.
# ---------------------------------------------------------------------------

def build_visits(
    outlets: pd.DataFrame, months: list[date], rng: np.random.Generator
) -> pd.DataFrame:
    """
    Give every outlet 0-2 visits per month (average ~1.35). Each visit gets
    an outcome (Order Placed / No Order / Closed) whose probability is
    nudged by that month's seasonality factor and that state's random
    "performance factor" — this is the region-wise skew the requirements
    doc asks for: some states just convert visits into orders more often
    than others, consistently, across the whole 24 months.
    """
    # One random performance multiplier per state, fixed for the whole
    # dataset — this is what makes "why did State X underperform" a
    # question with a real, consistent answer.
    state_names = outlets["state_name"].unique()
    state_factor = {
        s: float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3)) for s in state_names
    }

    rows = []
    visit_counter = 0

    outlet_records = outlets[["outlet_id", "se_id", "state_name"]].itertuples()
    for o in outlet_records:
        factor = state_factor[o.state_name]
        for m in months:
            n_visits = rng.choice([0, 1, 2], p=[0.05, 0.55, 0.40])
            seasonality = MONTH_SEASONALITY[m.month]

            for _ in range(n_visits):
                visit_counter += 1
                day = int(rng.integers(1, 29))  # avoid month-length edge cases
                visit_date = date(m.year, m.month, day)

                p_order = float(np.clip(BASE_ORDER_PLACED_PROB * seasonality * factor, 0.15, 0.90))
                p_closed = BASE_CLOSED_PROB
                p_no_order = max(1.0 - p_order - p_closed, 0.0)
                # Renormalise so the three probabilities sum to exactly 1.
                total = p_order + p_closed + p_no_order
                outcome = rng.choice(
                    ["Order Placed", "No Order", "Closed"],
                    p=[p_order / total, p_no_order / total, p_closed / total],
                )

                rows.append({
                    "visit_id": f"VIS{visit_counter:07d}",
                    "visit_date": visit_date,
                    "se_id": o.se_id,
                    "outlet_id": o.outlet_id,
                    "visit_outcome": outcome,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Orders: one row per SKU line, only for visits with outcome
#    "Order Placed". Fulfilment is capped by that WD's stock position for
#    that SKU/month, which is what ties Service Level back to inventory.
# ---------------------------------------------------------------------------

def build_orders(
    visits: pd.DataFrame,
    outlets: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For every "Order Placed" visit, pick a handful of SKUs the outlet's
    channel type is allowed to carry (with a festive-season boost for
    Ferrero/Candy in Oct/Nov), order a random quantity of each, and deliver
    slightly less than ordered whenever that WD had a stock disruption on
    that SKU in that month — otherwise deliver in full most of the time.
    """
    outlet_lookup = outlets.set_index("outlet_id")[["wd_id", "channel_type"]].to_dict(orient="index")

    # Fast lookup: (wd_id, sku_id, month) -> stockout_flag
    inv_lookup = {
        (r.wd_id, r.sku_id, r.snapshot_month): r.stockout_flag
        for r in inventory.itertuples()
    }

    skus_by_category = products.groupby("category_name")["sku_id"].apply(list).to_dict()
    sku_price = products.set_index("sku_id")["unit_price"].to_dict()

    rows = []
    order_counter = line_counter = 0

    order_visits = visits[visits["visit_outcome"] == "Order Placed"]
    for v in order_visits.itertuples():
        outlet_info = outlet_lookup[v.outlet_id]
        wd_id = outlet_info["wd_id"]
        channel = outlet_info["channel_type"]
        month_key = date(v.visit_date.year, v.visit_date.month, 1)

        eligible_categories = CHANNEL_CATEGORY_ELIGIBILITY[channel]
        candidate_skus = []
        weights = []
        for cat in eligible_categories:
            boost = FESTIVE_CATEGORY_BOOST.get(cat, 1.0) if month_key.month in FESTIVE_MONTHS else 1.0
            for sku_id in skus_by_category.get(cat, []):
                candidate_skus.append(sku_id)
                weights.append(boost)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        n_lines = int(rng.integers(1, 7))
        n_lines = min(n_lines, len(candidate_skus))
        chosen_skus = rng.choice(candidate_skus, size=n_lines, replace=False, p=weights)

        order_counter += 1
        order_id = f"ORD{order_counter:07d}"

        for sku_id in chosen_skus:
            line_counter += 1
            qty_ordered = int(rng.integers(1, 25))

            disrupted = inv_lookup.get((wd_id, sku_id, month_key), False)
            if disrupted:
                # Supply-constrained: deliver a noticeably reduced amount.
                fulfilment_rate = float(rng.uniform(0.2, 0.7))
            else:
                # Normally fulfilled, with a small tail of minor shortfalls.
                fulfilment_rate = float(rng.choice([1.0, rng.uniform(0.6, 0.99)], p=[0.85, 0.15]))

            qty_delivered = int(round(qty_ordered * fulfilment_rate))

            rows.append({
                "order_id": order_id,
                "line_id": f"LN{line_counter:07d}",
                "visit_id": v.visit_id,
                "order_date": v.visit_date,
                "outlet_id": v.outlet_id,
                "wd_id": wd_id,
                "sku_id": sku_id,
                "qty_ordered": qty_ordered,
                "qty_delivered": qty_delivered,
                "unit_price": sku_price[sku_id],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(seed: int) -> dict[str, pd.DataFrame]:
    """Run every generator in dependency order and return the six tables."""
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    months = month_range(DATA_END_MONTH, N_MONTHS)

    geography = build_geography(rng, fake)
    outlets = build_outlets(geography, rng, fake)
    products = build_products(rng)
    inventory = build_inventory(geography, products, months, rng)
    visits = build_visits(outlets, months, rng)
    orders = build_orders(visits, outlets, products, inventory, rng)

    return {
        "geography": geography,
        "outlets": outlets,
        "products": products,
        "inventory_snapshots": inventory,
        "visits": visits,
        "orders": orders,
    }


def save_all(tables: dict[str, pd.DataFrame], data_dir: Path) -> None:
    """Write every table to <data_dir>/<name>.csv."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(data_dir / f"{name}.csv", index=False)


def main() -> None:
    settings = get_settings()
    tables = generate_all(settings.random_seed)
    save_all(tables, settings.project_root / "data")

    for name, df in tables.items():
        print(f"{name}.csv: {len(df):,} rows")


if __name__ == "__main__":
    main()
