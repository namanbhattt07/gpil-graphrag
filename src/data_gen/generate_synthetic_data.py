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

from datetime import date, timedelta
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
ZONES_PER_STATE = 2                    # no change
WD_PER_ZONE_RANGE = (4, 7)             # was (2, 3)    → ~2.2x
SE_PER_WD_RANGE = (12, 22)             # was (5, 15)   → ~1.7x
OUTLETS_PER_SE_RANGE = (140, 180)      # was (70, 100) → ~1.9x

# Retail channel types an outlet can be, with roughly how common each is.
# "Retail" outlets (paan shops/tobacconists, the classic cigarette outlet in
# India) are the most common by far, hence the highest weight.

CHANNEL_TYPES = ["Retail", "Hawkers", "Modern Trade", "Dealer"]

# Retail=8,00,000  Hawkers=10,000  MT=2,500  Dealer=40,000  (total 8,52,500)
# Modern Trade's outlet count was cut from 7,000 to 2,500 (real-world large-
# format/supermarket outlets genuinely are much rarer than street Hawkers).
# Kept exactly as-is per explicit instruction — still used below to draw
# each outlet's channel_type.
CHANNEL_WEIGHTS_BY_COUNT = [0.9385, 0.0117, 0.0029, 0.0469]

# Original target share of UNITS SOLD (sales volume), NOT revenue: Retail
# 35% / Hawkers 9% / Modern Trade 3% / Dealer 53%. Kept exactly as-is per
# explicit instruction, but NOTE: it is no longer wired into an actual
# quantity multiplier (see CHANNEL_BASE_MULTIPLIER below for why). Deriving
# a channel multiplier as volume_share/count_share (the old approach) ties
# per-outlet quantity directly to these aggregate targets, which makes it
# mathematically impossible to also satisfy the strict per-outlet cascade
# "Bronze Dealer > Gold Modern Trade > Gold Hawkers > Gold Retail" once
# combined with any non-trivial tier spread — worked out in detail on
# 2026-07-31 (see project memory). This constant is retained here purely as
# a record of the original aggregate-share ambition, not as a live input.
CHANNEL_WEIGHTS_BY_VOLUME = [0.35, 0.09, 0.03, 0.53]

# Fixed per-outlet BASE quantity multiplier, applied in build_orders()
# BEFORE the tier multiplier (see TIER_BASE_MULTIPLIER below). Unlike the
# old volume_share/count_share approach, these are chosen directly so the
# strict 12-way cascade (Gold Dealer > ... > Bronze Retail) holds by
# construction, regardless of outlet counts.
#
# 2026-08-03 retune: the previous 24/8/3/1 set satisfied the cascade with a
# lot of room to spare (adjacent-channel margin ~1.78-2x against the 1.5x
# Gold:Bronze tier spread) but left Hawkers/Modern Trade's realised volume
# share badly short of target (measured ~1.65%/1.09% vs the 9%/3% ambition)
# because their outlet-count share is so tiny (1.17%/0.29%) that only a much
# bigger multiplier moves their aggregate share at all. Re-solved by grid
# search over (Dealer, Modern Trade, Hawkers) holding Retail=1 fixed,
# maximising closeness to the 53/35/9/3 target subject to keeping every
# adjacent-channel cascade margin >= 1.2x (i.e. mult_higher > 1.2 * 1.5 *
# mult_lower for every step Dealer>MT>Hawkers>Retail) — tight enough to move
# the needle, but still a real safety buffer over the 1.0x line
# verify_cascade() treats as a hard failure, not the 1.01-1.05x knife-edge
# the same search found if pushed further. Landed at Dealer=33/MT=18/
# Hawkers=10 (Retail=1), projected shares Retail 35.3% / Dealer 58.3% /
# Hawkers 4.4% / Modern Trade 2.0% -- Retail and Dealer now much closer to
# target, Hawkers/MT meaningfully improved (~2.7x / ~1.8x closer) though
# still short of 9%/3% exactly, because hitting those exactly would need
# multipliers whose ratios violate the 1.5x tier spread (worked out
# analytically: the exact-fit multipliers are only ~1.09x-1.35x apart
# channel-to-channel, less than the 1.5x the tier spread needs, so an exact
# fit and a passing cascade are mutually exclusive at these outlet-count
# shares). Dealer overshooting to ~58% stays comfortably under the "no
# channel > 80%" realism guardrail.
CHANNEL_BASE_MULTIPLIER = {
    "Dealer": 33.0,
    "Modern Trade": 18.0,
    "Hawkers": 10.0,
    "Retail": 1.0,
}

# Which product categories each channel type typically carries. All four
# channels carry all four categories (Retail outlets picked up Ferrero too,
# e.g. Tic Tac/Kinder Joy as impulse-buy counter items) — the dict is kept
# per-channel rather than collapsed to one shared list so a future
# channel-specific restriction is a one-line change, not a rewrite.
CHANNEL_CATEGORY_ELIGIBILITY = {
    "Retail": ["GPI", "IPM", "Candy", "Ferrero"],
    "Hawkers": ["GPI", "IPM", "Ferrero", "Candy"],
    "Modern Trade": ["GPI", "IPM", "Ferrero", "Candy"],
    "Dealer": ["GPI", "IPM", "Ferrero", "Candy"],
}

# Inactive-outlet rate range per channel: Retail shops close down most
# often (small, low-investment businesses), Dealers/Hawkers rarely. A
# single rate is rolled per channel (not per outlet) so the whole dataset's
# Retail inactive share lands somewhere in 10-12%, etc.
CHANNEL_INACTIVE_RATE_RANGE = {
    "Retail": (0.10, 0.12),
    "Hawkers": (0.02, 0.03),
    "Modern Trade": (0.03, 0.06),
    "Dealer": (0.02, 0.04),
}

# Outlet "weight" tier, used later (Phase 3) to compute ACV as a
# sales-weighted version of numeric distribution. Most outlets are small
# (Bronze); a few big ones (Gold) carry disproportionate volume.
OUTLET_TIERS = ["Gold", "Silver", "Bronze"]

# outlet-count share. Kept exactly as-is per explicit instruction — still
# used below to draw each outlet's random outlet_tier.
TIER_WEIGHTS_BY_COUNT = [0.20, 0.35, 0.45]

# Original target share of UNITS SOLD (sales volume), NOT revenue. Kept
# exactly as-is per explicit instruction, but — like CHANNEL_WEIGHTS_BY_VOLUME
# above — no longer wired into an actual quantity multiplier. Deriving a
# tier multiplier as volume_share/count_share gives a Gold:Bronze ratio of
# (0.65/0.20)/(0.10/0.45) = ~14.6x, which is what previously pushed Bronze
# Dealer below Gold Modern Trade in the strict per-outlet cascade. Retained
# here purely as a record of the original aggregate-share ambition.
TIER_WEIGHTS_BY_VOLUME = [0.65, 0.25, 0.10]

# Fixed per-outlet BASE multiplier applied ON TOP of CHANNEL_BASE_MULTIPLIER
# above, chosen to give a modest 1.2x-1.5x Gold:Bronze spread (not 14.6x)
# so tier differences never overwhelm the channel gap the cascade depends
# on — e.g. Bronze Dealer (24*1.0=24) still clears Gold Modern Trade
# (8*1.5=12) by a comfortable 2x margin.
TIER_BASE_MULTIPLIER = {
    "Gold": 1.5,
    "Silver": 1.2,
    "Bronze": 1.0,
}

# The 4 GPI franchises (of 6) and the 1 IPM franchise whose flagship
# ("Pack 1") SKU becomes a "Hero" SKU in build_orders — boosted in both
# pick-rate and quantity so a handful of SKUs naturally carry the bulk of
# units sold, instead of all 63 SKUs selling at roughly the same flat
# volume. See select_hero_skus() below. These two values were tuned
# empirically (small-scale trial runs) to land Hero SKUs at ~70% of total
# units delivered, per the target concentration.
HERO_SKU_WEIGHT_BOOST = 25.0     # multiplies a Hero SKU's chance of being picked on an order line
HERO_SKU_QTY_MULTIPLIER = 1.4    # multiplies qty_ordered once a Hero SKU is picked

# Channel, tier and Hero multipliers can all land on the same order line
# (e.g. a Gold-tier Dealer ordering a Hero SKU) and compound — with the
# 2026-08-03 CHANNEL_BASE_MULTIPLIER retune (Dealer now 33, was 24), the
# worst case is ~channel_mult(33) x tier_mult(1.5) x hero_mult(1.4) x
# base_qty(24) = ~1,663, which would already exceed the old 1500 cap. Cap
# raised to 2000 to keep clear of that new worst case (was 250 at one point,
# which clipped 12-20% of Dealer/Hawkers/Modern Trade lines and
# systematically suppressed their aggregate volume share while inflating
# Retail's, since Retail's multiplier is below 1 and never hits the cap —
# exactly the failure mode this retune is trying to fix, so leaving the cap
# too tight here would quietly undo it).
MAX_QTY_ORDERED_PER_LINE = 2000

# Weight on state_factor (vs. independent per-state noise) when deriving
# each state's Dropsize scaling factor in build_orders() (see DROPSIZE_STATE_FACTOR_WEIGHT
# usage there). Using state_factor directly (weight=1.0) makes Productivity
# and Dropsize near-perfectly correlated (measured 0.81 on a full production
# run on 2026-07-30): with tens of thousands of visits/orders per state x
# month, per-visit randomness averages out, leaving both KPIs as almost
# deterministic functions of the same state_factor. Blending in independent
# per-state noise dilutes that down to a believable positive correlation
# instead of a suspiciously perfect one. Tuned empirically (small-scale
# Monte Carlo simulation of just the two KPI formulas, same approach as
# CATEGORY_SKU_WEIGHT_MULTIPLIER) to land Productivity-vs-Dropsize
# correlation at ~0.3-0.6 (target range) instead of coupling them 1:1 —
# 0.5 measured ~0.42-0.50 correlation across several seeds.
DROPSIZE_STATE_FACTOR_WEIGHT = 0.5
# Small-scale calibration predicted ~0.42-0.50 correlation at weight=0.5;
# the actual full-scale run (2026-07-30, seed 42) measured 0.605 -- just
# above the 0.3-0.6 target but accepted rather than re-tuning further.

# Product hierarchy: Category -> Franchise. Franchise names are generic
# placeholders EXCEPT Marlboro (IPM), TicTac and Kinder_Joy (Ferrero),
# which the user named directly rather than us guessing them.
CATEGORY_FRANCHISES = {
    "GPI": [f"GPI_Franchise_{i}" for i in range(1, 7)],       # 6 franchises
    "IPM": ["Marlboro"],                                       # 1 franchise, many SKUs
    "Ferrero": ["TicTac", "Kinder_Joy"],                       # 2 franchises
    "Candy": ["Candy_Franchise_1", "Candy_Franchise_2"],       # 2 franchises
}

# target share of UNITS SOLD (sales volume) per category, NOT revenue.
# Cigarettes (GPI+IPM) dominate volume; Ferrero/Candy are small impulse-buy
# categories. Used by build_orders() to weight SKU-selection probability by
# category_name — unlike CHANNEL_BASE_MULTIPLIER/TIER_BASE_MULTIPLIER (fixed
# numbers, chosen for the cascade), this one IS still live-wired into
# build_orders() to target the requested category split. This stacks with
# (doesn't replace) the Hero-SKU boost below — Hero SKUs get an extra
# multiplier on top of their category's base weight, since all 5 Hero SKUs
# live inside GPI/IPM.
CATEGORY_WEIGHTS_BY_VOLUME = {"GPI": 0.46, "IPM": 0.45, "Ferrero": 0.05, "Candy": 0.045}

# Naively splitting CATEGORY_WEIGHTS_BY_VOLUME evenly across each category's
# own SKUs (CATEGORY_WEIGHTS_BY_VOLUME[cat] / n_skus_in_cat) does NOT actually
# land units-sold at those targets, because the Hero-SKU boost below (25x
# pick-weight x 1.4x quantity = ~35x combined) is concentrated inside GPI (4
# Hero SKUs out of 40) and IPM (1 Hero SKU out of 5) but ZERO Hero SKUs sit in
# Ferrero/Candy. A naive split measured out at GPI 57.7% / IPM 39.0% /
# Ferrero 1.7% / Candy 1.5% in a full production run on 2026-07-29 — nowhere
# close to the 46/45/5/4.5 targets. Correcting this analytically is hard
# because build_orders() samples SKUs WITHOUT replacement from a highly
# skewed weight distribution (numpy's weighted sample-without-replacement
# isn't simply proportional to raw weight once one item's weight dominates),
# so these multipliers were found empirically instead: start from the naive
# per-SKU split, run a fast Monte Carlo simulation of just the SKU-selection
# logic, measure the resulting category share, rescale each category's
# weight by target/measured, and repeat until it converges (6 rounds was
# enough) — the same "tune empirically, verify by measuring the output"
# approach already used for HERO_SKU_WEIGHT_BOOST/HERO_SKU_QTY_MULTIPLIER.
# Verified this way to land within ~1pp of every target. If SKUS_PER_FRANCHISE_RANGE
# or the Hero SKU set ever changes, this needs re-calibrating the same way.
CATEGORY_SKU_WEIGHT_MULTIPLIER = {"GPI": 0.51, "IPM": 0.92, "Ferrero": 1.97, "Candy": 1.96}

# How many SKU variants (pack sizes) each franchise gets. Marlboro gets more
# because the brief specifically calls out "different SKUs like Marlboro".
SKUS_PER_FRANCHISE_RANGE = {
    "GPI": (4, 8),
    "IPM": (4,6), 
    "Ferrero": (3, 5),
    "Candy": (3, 5),
}

# Roughly realistic price bands per category, in INR. Cigarettes are priced
# per pack, confectionery per small unit/multipack.
UNIT_PRICE_RANGE = {
    "GPI": (70, 320),
    "IPM": (100, 400),  
    "Ferrero": (10, 150),
    "Candy": (5, 100),
}

# population (2024 estimate) and GSDP FY23-24 in ₹ Crore, current prices.
# Source: Wikipedia "List of states and union territories of India by population"
# and StatisticsTimes.com GSDP table (underlying data for the Wikipedia GDP page).
# Used by build_visits() to ground each state's business-volume factor in
# real population/GDP instead of pure randomness — see state_factor there.
STATE_POPULATION_GDP = {
    "Andhra Pradesh":    {"population": 53_340_000,  "gsdp_cr": 1_422_094},
    "Arunachal Pradesh": {"population": 1_576_000,   "gsdp_cr": 38_565},
    "Assam":             {"population": 36_047_000,  "gsdp_cr": 569_287},
    "Bihar":             {"population": 128_592_000, "gsdp_cr": 877_197},
    "Chhattisgarh":      {"population": 30_524_000,  "gsdp_cr": 512_107},
    "Goa":               {"population": 1_583_000,   "gsdp_cr": 106_533},
    "Gujarat":           {"population": 72_367_000,  "gsdp_cr": 2_425_804},
    "Haryana":           {"population": 30_573_000,  "gsdp_cr": 1_085_510},
    "Himachal Pradesh":  {"population": 7_505_000,   "gsdp_cr": 212_169},
    "Jharkhand":         {"population": 39_963_000,  "gsdp_cr": 465_638},
    "Karnataka":         {"population": 68_115_000,  "gsdp_cr": 2_557_241},
    "Kerala":            {"population": 35_920_000,  "gsdp_cr": 1_135_372},
    "Madhya Pradesh":    {"population": 87_610_000,  "gsdp_cr": 1_353_809},
    "Maharashtra":       {"population": 127_528_000, "gsdp_cr": 4_055_847},
    "Manipur":           {"population": 3_253_000,   "gsdp_cr": 43_414},
    "Meghalaya":         {"population": 3_379_000,   "gsdp_cr": 53_223},
    "Mizoram":           {"population": 1_250_000,   "gsdp_cr": 33_277},
    "Nagaland":          {"population": 2_253_000,   "gsdp_cr": 39_809},
    "Odisha":            {"population": 46_566_000,  "gsdp_cr": 798_969},
    "Punjab":            {"population": 30_926_000,  "gsdp_cr": 771_744},
    "Rajasthan":         {"population": 81_897_000,  "gsdp_cr": 1_521_510},
    "Sikkim":            {"population": 695_000,     "gsdp_cr": 48_937},
    "Tamil Nadu":        {"population": 77_089_000,  "gsdp_cr": 2_688_963},
    "Telangana":         {"population": 38_272_000,  "gsdp_cr": 1_461_836},
    "Tripura":           {"population": 4_184_000,   "gsdp_cr": 79_434},
    "Uttar Pradesh":     {"population": 241_066_874, "gsdp_cr": 2_642_877},
    "Uttarakhand":       {"population": 11_755_000,  "gsdp_cr": 332_998},
    "West Bengal":       {"population": 99_563_000,  "gsdp_cr": 1_651_374},
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

def build_outlets(
    geography: pd.DataFrame, months: list[date], rng: np.random.Generator, fake: Faker
) -> pd.DataFrame:
    """
    Generate OUTLETS_PER_SE_RANGE retail outlets under every SE. Each
    outlet also carries a denormalised wd_id/zone_id/state_id so downstream
    tables (visits, orders) don't need a 4-way join just to find out which
    state an outlet is in.

    Inactive rate depends on channel type (see CHANNEL_INACTIVE_RATE_RANGE)
    instead of one flat rate for everyone. An "inactive" outlet is one that
    had already permanently closed BEFORE the 24-month data window even
    starts (closure_date is sampled between its onboarded_date and the day
    before window_start) — so it must show ZERO visits/orders anywhere in
    the dataset, not just after some mid-window closure event. build_visits()
    enforces this by checking closure_date before generating each visit_date
    (and build_orders() inherits that automatically, since orders are only
    ever built from visits that already exist).
    """
    # Build a quick lookup: unit_id -> row, so we can walk SE -> WD -> Zone.
    lookup = geography.set_index("unit_id").to_dict(orient="index")

    # One inactive-rate roll per channel (not per outlet), so e.g. the whole
    # dataset's Retail inactive share lands somewhere in 10-12% rather than
    # every outlet independently rolling its own rate.
    channel_inactive_rate = {
        ch: float(rng.uniform(lo, hi)) for ch, (lo, hi) in CHANNEL_INACTIVE_RATE_RANGE.items()
    }

    # closure_date for an inactive outlet must fall BEFORE the data window
    # starts (see docstring) so it never has a single transaction inside the
    # 24-month period this dataset covers.
    window_start = months[0]
    last_possible_closure = window_start - timedelta(days=1)

    se_rows = geography[geography["unit_type"] == "SE"]
    rows = []
    outlet_counter = 0

    for se in se_rows.itertuples():
        wd_id = se.parent_unit_id
        zone_id = lookup[wd_id]["parent_unit_id"]
        state_id = lookup[zone_id]["parent_unit_id"]

        n_outlets = rng.integers(OUTLETS_PER_SE_RANGE[0], OUTLETS_PER_SE_RANGE[1] + 1)
        channel = rng.choice(CHANNEL_TYPES, size=n_outlets, p=CHANNEL_WEIGHTS_BY_COUNT)
        tier = rng.choice(OUTLET_TIERS, size=n_outlets, p=TIER_WEIGHTS_BY_COUNT)

        for i in range(n_outlets):
            outlet_counter += 1
            onboarded = fake.date_between(start_date=date(2015, 1, 1), end_date=date(2024, 7, 31))

            is_inactive = rng.random() < channel_inactive_rate[channel[i]]
            closure_date = (
                fake.date_between(
                    start_date=min(onboarded, last_possible_closure),
                    end_date=last_possible_closure,
                )
                if is_inactive else None
            )

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
                "is_active": not is_inactive,
                "closure_date": closure_date,
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


def select_hero_skus(products: pd.DataFrame) -> set[str]:
    """
    Pick a small, fixed set of "Hero" SKUs — the flagship ("Pack 1") SKU of
    4 of GPI's 6 franchises, plus Marlboro's flagship SKU — that
    build_orders() boosts in both pick-rate and quantity so they naturally
    end up carrying the bulk of units sold. This mirrors the Pareto-style
    demand concentration real cigarette/FMCG portfolios show (a handful of
    SKUs driving most volume, everything else a long tail), instead of all
    63 SKUs selling at roughly the same flat volume.
    """
    heroes = set()

    for franchise in CATEGORY_FRANCHISES["GPI"][:4]:
        franchise_skus = products[products["franchise_name"] == franchise].sort_values("sku_id")
        heroes.add(franchise_skus.iloc[0]["sku_id"])

    ipm_franchise = CATEGORY_FRANCHISES["IPM"][0]
    ipm_skus = products[products["franchise_name"] == ipm_franchise].sort_values("sku_id")
    heroes.add(ipm_skus.iloc[0]["sku_id"])

    return heroes


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
                disrupted = rng.random() < STOCKOUT_EVENT_PROB #Aaj supply chain kharab hui ya nahi. Probability

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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Give every outlet 0-2 visits per month (average ~1.35), stopping once an
    outlet has passed its closure_date (see build_outlets). Each visit gets
    an outcome (Order Placed / No Order / Closed) whose probability is
    nudged by that month's seasonality factor and that state's
    population/GDP-based "performance factor" — this is the region-wise
    skew the requirements doc asks for: some states just convert visits
    into orders more often than others, consistently, across the whole 24
    months.

    Also returns state_factor (state_name -> float) so build_orders() can
    reuse the exact same per-state performance signal when sizing orders —
    this is what makes Productivity (visit conversion) and Dropsize (SKUs
    per successful order) positively correlated: a state that's good at
    converting visits into orders is, for the same underlying reason, also
    good at getting outlets to order more lines per visit.
    """
    # State factor is grounded in real population + GSDP (bigger/richer
    # states generate more business) instead of pure noise — see
    # STATE_POPULATION_GDP. Population/GSDP span orders of magnitude (e.g.
    # Uttar Pradesh's population is ~350x Sikkim's), so we log-scale both
    # and min-max normalise them to 0-1 before combining, keeping the
    # resulting factor spread in a believable business range rather than
    # literally mirroring a 350x population gap. A small random wobble is
    # layered on top so state rankings aren't 100% identical every run.
    state_names = outlets["state_name"].unique()

    log_population = {s: np.log(STATE_POPULATION_GDP[s]["population"]) for s in state_names}
    log_gsdp = {s: np.log(STATE_POPULATION_GDP[s]["gsdp_cr"]) for s in state_names}

    def _minmax(values: dict[str, float]) -> dict[str, float]:
        lo, hi = min(values.values()), max(values.values())
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}

    population_norm = _minmax(log_population)
    gsdp_norm = _minmax(log_gsdp)

    state_factor = {}
    for s in state_names:
        composite = (population_norm[s] + gsdp_norm[s]) / 2.0   # 0 (smallest) .. 1 (largest)
        base = 0.75 + 0.6 * composite                            # maps to ~0.75-1.35
        wobble = float(rng.uniform(-0.05, 0.05))                 # so every run differs slightly
        state_factor[s] = float(np.clip(base + wobble, 0.65, 1.45))

    # Year-on-year variation: festive months stay the same (Oct/Nov is
    # always Diwali season), but the SIZE of that spike gets a small random
    # adjustment per calendar year, so year 2 isn't a carbon copy of year 1.
    years_in_range = sorted({m.year for m in months})
    year_seasonality_adjustment = {y: float(rng.uniform(0.90, 1.10)) for y in years_in_range}

    # Columnar accumulation (one list per output column) instead of a list
    # of per-row dicts: at ~24M visit rows, a list of dicts costs tens of
    # GB in pure Python object overhead (~500+ bytes/dict on top of the
    # actual data) and can push the process into swap. Plain lists of
    # scalars carry none of that per-row overhead, and pd.DataFrame(dict)
    # builds the same output DataFrame from them directly.
    visit_ids: list[str] = []
    visit_dates: list[date] = []
    se_ids: list[str] = []
    outlet_ids: list[str] = []
    visit_outcomes: list[str] = []
    visit_counter = 0

    outlet_records = outlets[["outlet_id", "se_id", "state_name", "closure_date"]].itertuples()
    for o in outlet_records:
        factor = state_factor[o.state_name]
        closure = o.closure_date  # None for outlets that never close

        for m in months:
            month_start = date(m.year, m.month, 1)
            if closure is not None and month_start > closure:
                # months is chronological, so every later month is also
                # past closure — nothing more to generate for this outlet.
                break

            n_visits = rng.choice([0, 1, 2], p=[0.05, 0.55, 0.40])
            base_seasonality = MONTH_SEASONALITY[m.month]
            year_adj = year_seasonality_adjustment[m.year]
            seasonality = 1.0 + (base_seasonality - 1.0) * year_adj

            for _ in range(n_visits):
                day = int(rng.integers(1, 29))  # avoid month-length edge cases
                visit_date = date(m.year, m.month, day)

                if closure is not None and visit_date > closure:
                    continue  # outlet had already closed by this specific day

                visit_counter += 1

                p_order = float(np.clip(BASE_ORDER_PLACED_PROB * seasonality * factor, 0.15, 0.90))
                p_closed = BASE_CLOSED_PROB
                p_no_order = max(1.0 - p_order - p_closed, 0.0)
                # Renormalise so the three probabilities sum to exactly 1.
                total = p_order + p_closed + p_no_order
                outcome = rng.choice(
                    ["Order Placed", "No Order", "Closed"],
                    p=[p_order / total, p_no_order / total, p_closed / total],
                )

                visit_ids.append(f"VIS{visit_counter:07d}")
                visit_dates.append(visit_date)
                se_ids.append(o.se_id)
                outlet_ids.append(o.outlet_id)
                visit_outcomes.append(outcome)

    visits = pd.DataFrame({
        "visit_id": visit_ids,
        "visit_date": visit_dates,
        "se_id": se_ids,
        "outlet_id": outlet_ids,
        "visit_outcome": visit_outcomes,
    })
    return visits, state_factor


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
    hero_sku_ids: set[str],
    state_factor: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For every "Order Placed" visit, pick a handful of SKUs the outlet's
    channel type is allowed to carry. SKU-selection probability is weighted
    by category_name (CATEGORY_WEIGHTS_BY_VOLUME, so GPI/IPM cigarettes
    dominate volume over Ferrero/Candy), with a festive-season boost for
    Ferrero/Candy in Oct/Nov stacked on top, and a much bigger boost for
    Hero SKUs stacked on top of that (see select_hero_skus) — Hero SKUs get
    an extra lift on top of their category's base weight, not instead of it.
    Order a quantity shaped by channel type, outlet tier and Hero-SKU status
    (CHANNEL_BASE_MULTIPLIER / TIER_BASE_MULTIPLIER / HERO_SKU_QTY_MULTIPLIER),
    and deliver slightly less than ordered whenever that WD had a stock
    disruption on that SKU in that month — otherwise deliver in full most of
    the time. CHANNEL_BASE_MULTIPLIER and TIER_BASE_MULTIPLIER are fixed
    numbers (not derived from the volume-share targets), which is what makes
    the strict per-outlet cascade "Gold Dealer > Silver Dealer > Bronze
    Dealer > Gold Modern Trade > ... > Bronze Retail" hold by construction.

    The number of lines per order (n_lines) is also scaled by the outlet's
    state -- via a "dropsize_factor" blended from state_factor (the same
    per-state performance signal build_visits() uses to decide how often a
    visit turns into an order) and independent per-state noise, weighted
    by DROPSIZE_STATE_FACTOR_WEIGHT. Without any link to state_factor,
    n_lines was a pure random draw and Productivity/Dropsize came out with
    near-zero correlation; using state_factor directly (no blend) made them
    *too* strongly correlated (~0.81 -- see DROPSIZE_STATE_FACTOR_WEIGHT's
    comment) since per-visit randomness averages out at this scale, leaving
    both metrics almost deterministic functions of the same input. The
    blend keeps a believable positive correlation without that 1:1 coupling.

    The three qty multipliers can all land on the same line (e.g. a
    Gold-tier Dealer ordering a Hero SKU) and compound, so the final
    qty_ordered is capped at MAX_QTY_ORDERED_PER_LINE to avoid unrealistic
    extremes.
    """
    outlet_lookup = outlets.set_index("outlet_id")[
        ["wd_id", "channel_type", "outlet_tier", "state_name"]
    ].to_dict(orient="index")

    # Per-state Dropsize scaling factor: a blend of state_factor and
    # independent per-state noise (see DROPSIZE_STATE_FACTOR_WEIGHT and the
    # docstring above for why a straight state_factor pass-through over-
    # correlates Productivity and Dropsize). Computed once per state here
    # (not per order) so a state's Dropsize tendency stays consistent
    # across the whole dataset, the same way state_factor does.
    dropsize_factor = {}
    for s in outlets["state_name"].unique():
        independent_noise = float(rng.uniform(0.65, 1.45))
        dropsize_factor[s] = (
            DROPSIZE_STATE_FACTOR_WEIGHT * state_factor[s]
            + (1 - DROPSIZE_STATE_FACTOR_WEIGHT) * independent_noise
        )

    # Fast lookup: (wd_id, sku_id, month) -> stockout_flag
    inv_lookup = {
        (r.wd_id, r.sku_id, r.snapshot_month): r.stockout_flag
        for r in inventory.itertuples()
    }

    skus_by_category = products.groupby("category_name")["sku_id"].apply(list).to_dict()
    sku_price = products.set_index("sku_id")["unit_price"].to_dict()

    # Per-SKU base weight from CATEGORY_WEIGHTS_BY_VOLUME, spread evenly
    # across that category's own SKU count so the category's SKUs would sum
    # to its target volume share BEFORE the Hero-SKU boost stacks on top —
    # then CATEGORY_SKU_WEIGHT_MULTIPLIER corrects for the fact that the
    # Hero boost stacking is NOT evenly distributed across categories (see
    # that constant's comment for why the naive split alone lands nowhere
    # near its targets).
    category_sku_weight = {
        cat: (CATEGORY_WEIGHTS_BY_VOLUME.get(cat, 1.0) / len(skus))
        * CATEGORY_SKU_WEIGHT_MULTIPLIER.get(cat, 1.0)
        for cat, skus in skus_by_category.items()
    }

    # Columnar accumulation (see build_visits' comment on why) — at ~40-50M
    # order-line rows, a list of per-row dicts is the difference between
    # this finishing in minutes and thrashing the machine's swap.
    order_ids: list[str] = []
    line_ids: list[str] = []
    line_visit_ids: list[str] = []
    order_dates: list[date] = []
    line_outlet_ids: list[str] = []
    line_wd_ids: list[str] = []
    line_sku_ids: list[str] = []
    qtys_ordered: list[int] = []
    qtys_delivered: list[int] = []
    line_unit_prices: list[float] = []
    order_counter = line_counter = 0

    order_visits = visits[visits["visit_outcome"] == "Order Placed"]
    for v in order_visits.itertuples():
        outlet_info = outlet_lookup[v.outlet_id]
        wd_id = outlet_info["wd_id"]
        channel = outlet_info["channel_type"]
        tier = outlet_info["outlet_tier"]
        state_name = outlet_info["state_name"]
        month_key = date(v.visit_date.year, v.visit_date.month, 1)

        eligible_categories = CHANNEL_CATEGORY_ELIGIBILITY[channel]
        candidate_skus = []
        weights = []
        for cat in eligible_categories:
            boost = FESTIVE_CATEGORY_BOOST.get(cat, 1.0) if month_key.month in FESTIVE_MONTHS else 1.0
            cat_weight = category_sku_weight.get(cat, 1.0)
            for sku_id in skus_by_category.get(cat, []):
                candidate_skus.append(sku_id)
                hero_weight = HERO_SKU_WEIGHT_BOOST if sku_id in hero_sku_ids else 1.0
                weights.append(cat_weight * boost * hero_weight)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        # Scale the max number of lines by this state's dropsize_factor
        # (partly derived from state_factor, partly independent noise —
        # see DROPSIZE_STATE_FACTOR_WEIGHT), so high-performing states
        # tend to produce bigger orders without being perfectly coupled
        # to Productivity.
        base_max_lines = 6
        scaled_max = max(2, int(round(base_max_lines * dropsize_factor[state_name])))
        n_lines = int(rng.integers(1, scaled_max + 1))
        n_lines = min(n_lines, len(candidate_skus))
        chosen_skus = rng.choice(candidate_skus, size=n_lines, replace=False, p=weights)

        order_counter += 1
        order_id = f"ORD{order_counter:07d}"

        channel_mult = CHANNEL_BASE_MULTIPLIER[channel]
        tier_mult = TIER_BASE_MULTIPLIER[tier]

        for sku_id in chosen_skus:
            line_counter += 1
            base_qty = int(rng.integers(1, 25))
            hero_mult = HERO_SKU_QTY_MULTIPLIER if sku_id in hero_sku_ids else 1.0
            qty_ordered = int(np.clip(
                round(base_qty * channel_mult * tier_mult * hero_mult),
                1, MAX_QTY_ORDERED_PER_LINE,
            ))

            disrupted = inv_lookup.get((wd_id, sku_id, month_key), False)
            if disrupted:
                # Supply-constrained: deliver a noticeably reduced amount.
                fulfilment_rate = float(rng.uniform(0.2, 0.7))
            else:
                # Normally fulfilled, with a small tail of minor shortfalls.
                fulfilment_rate = float(rng.choice([1.0, rng.uniform(0.6, 0.99)], p=[0.85, 0.15]))

            qty_delivered = int(round(qty_ordered * fulfilment_rate))

            order_ids.append(order_id)
            line_ids.append(f"LN{line_counter:07d}")
            line_visit_ids.append(v.visit_id)
            order_dates.append(v.visit_date)
            line_outlet_ids.append(v.outlet_id)
            line_wd_ids.append(wd_id)
            line_sku_ids.append(sku_id)
            qtys_ordered.append(qty_ordered)
            qtys_delivered.append(qty_delivered)
            line_unit_prices.append(sku_price[sku_id])

    return pd.DataFrame({
        "order_id": order_ids,
        "line_id": line_ids,
        "visit_id": line_visit_ids,
        "order_date": order_dates,
        "outlet_id": line_outlet_ids,
        "wd_id": line_wd_ids,
        "sku_id": line_sku_ids,
        "qty_ordered": qtys_ordered,
        "qty_delivered": qtys_delivered,
        "unit_price": line_unit_prices,
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(seed: int, verbose: bool = False) -> dict[str, pd.DataFrame]:
    """
    Run every generator in dependency order and return the six tables.

    verbose=True prints a one-line progress marker after every major step
    (geography, outlets, products, inventory, visits, orders) — useful for
    the full-scale run, which takes 30+ minutes, so it's obvious the process
    is still working rather than hung. Off by default so the test suite
    (which calls this twice per test module) doesn't get noisy.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    months = month_range(DATA_END_MONTH, N_MONTHS)

    log("[1/6] Building geography (State -> Zone -> WD -> SE)...")
    geography = build_geography(rng, fake)
    log(f"[1/6] Done — {len(geography):,} geography rows.")

    log("[2/6] Building outlets...")
    outlets = build_outlets(geography, months, rng, fake)
    log(f"[2/6] Done — {len(outlets):,} outlets.")

    log("[3/6] Building product catalogue...")
    products = build_products(rng)
    hero_sku_ids = select_hero_skus(products)
    log(f"[3/6] Done — {len(products):,} SKUs, {len(hero_sku_ids)} Hero SKUs.")

    log("[4/6] Building inventory snapshots (WD x SKU x month)...")
    inventory = build_inventory(geography, products, months, rng)
    log(f"[4/6] Done — {len(inventory):,} inventory rows.")

    log("[5/6] Building visits...")
    visits, state_factor = build_visits(outlets, months, rng)
    log(f"[5/6] Done — {len(visits):,} visits.")

    log("[6/6] Building orders (this is the slow step)...")
    orders = build_orders(visits, outlets, products, inventory, hero_sku_ids, state_factor, rng)
    log(f"[6/6] Done — {len(orders):,} order lines.")

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


def verify_cascade(outlets: pd.DataFrame, orders: pd.DataFrame) -> bool:
    """
    Check the three business rules this regeneration exists to satisfy,
    using qty_delivered (actual units sold, not just ordered) as "units":
      1. Per-outlet AVERAGE units for all 12 (channel, tier) combinations,
         in the exact required strict order.
      2. TOTAL units by channel: Dealer > Retail > Hawkers > Modern Trade.
      3. Within-channel tier order: Gold > Silver > Bronze, for every
         channel individually.
    Prints all three tables regardless of outcome, and returns True only if
    every check passes.
    """
    merged = orders.merge(
        outlets[["outlet_id", "channel_type", "outlet_tier"]], on="outlet_id", how="left"
    )
    units_by_group = merged.groupby(["channel_type", "outlet_tier"])["qty_delivered"].sum()
    outlet_counts = outlets.groupby(["channel_type", "outlet_tier"])["outlet_id"].count()
    avg_per_outlet = units_by_group.reindex(outlet_counts.index).fillna(0) / outlet_counts

    expected_order = [
        ("Dealer", "Gold"), ("Dealer", "Silver"), ("Dealer", "Bronze"),
        ("Modern Trade", "Gold"), ("Modern Trade", "Silver"), ("Modern Trade", "Bronze"),
        ("Hawkers", "Gold"), ("Hawkers", "Silver"), ("Hawkers", "Bronze"),
        ("Retail", "Gold"), ("Retail", "Silver"), ("Retail", "Bronze"),
    ]

    print("\n=== CHECK 1: Per-outlet AVERAGE units, all 12 (channel x tier) combos ===")
    print("(required order: Gold Dealer > Silver Dealer > Bronze Dealer > Gold Modern "
          "Trade > Silver Modern Trade > Bronze Modern Trade > Gold Hawkers > Silver "
          "Hawkers > Bronze Hawkers > Gold Retail > Silver Retail > Bronze Retail)")
    values = []
    for channel, tier in expected_order:
        val = float(avg_per_outlet.get((channel, tier), float("nan")))
        values.append(val)
        print(f"    {tier:<7} {channel:<13} avg/outlet = {val:,.2f}")
    check1_pass = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    print(f"    CHECK 1 (strict 12-way cascade): {'PASS' if check1_pass else 'FAIL'}")

    print("\n=== CHECK 2: TOTAL units by channel ===")
    total_by_channel = merged.groupby("channel_type")["qty_delivered"].sum().sort_values(ascending=False)
    for channel, val in total_by_channel.items():
        print(f"    {channel:<13} total = {val:,.0f}")
    check2_pass = list(total_by_channel.index) == ["Dealer", "Retail", "Hawkers", "Modern Trade"]
    print(f"    CHECK 2 (Dealer > Retail > Hawkers > Modern Trade): {'PASS' if check2_pass else 'FAIL'}")

    print("\n=== CHECK 3: Within-channel tier order (Gold > Silver > Bronze) ===")
    check3_pass = True
    for channel in CHANNEL_TYPES:
        gold = float(avg_per_outlet.get((channel, "Gold"), float("nan")))
        silver = float(avg_per_outlet.get((channel, "Silver"), float("nan")))
        bronze = float(avg_per_outlet.get((channel, "Bronze"), float("nan")))
        ok = gold > silver > bronze
        check3_pass = check3_pass and ok
        print(f"    {channel:<13} Gold={gold:,.2f}  Silver={silver:,.2f}  Bronze={bronze:,.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"    CHECK 3 (Gold > Silver > Bronze within every channel): {'PASS' if check3_pass else 'FAIL'}")

    all_pass = check1_pass and check2_pass and check3_pass
    print(f"\n=== OVERALL: {'ALL 3 CHECKS PASSED' if all_pass else 'AT LEAST ONE CHECK FAILED'} ===\n")
    return all_pass


def main() -> None:
    settings = get_settings()
    data_dir = settings.project_root / "data"

    print("Deleting existing CSVs before regenerating...", flush=True)
    table_names = ["geography", "outlets", "products", "inventory_snapshots", "visits", "orders"]
    for name in table_names:
        csv_path = data_dir / f"{name}.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"    deleted {csv_path.name}", flush=True)

    print("\nStarting full regeneration...", flush=True)
    tables = generate_all(settings.random_seed, verbose=True)

    print("\nSaving tables to CSV...", flush=True)
    save_all(tables, data_dir)
    for name, df in tables.items():
        print(f"    {name}.csv: {len(df):,} rows", flush=True)

    print("\nRunning cascade verification...", flush=True)
    all_pass = verify_cascade(tables["outlets"], tables["orders"])
    if not all_pass:
        raise SystemExit(
            "Cascade verification FAILED — see tables above. Do not treat this "
            "regeneration as done; investigate CHANNEL_BASE_MULTIPLIER / "
            "TIER_BASE_MULTIPLIER in this file before re-running."
        )


if __name__ == "__main__":
    main()
