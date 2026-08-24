"""
Phase 4 -- Graph-Ready Document Construction for the GPIL GraphRAG project.

WHAT THIS SCRIPT DOES (plain language):
GraphRAG (the Microsoft library we'll use in Phase 5) doesn't read CSV rows
directly. It reads plain text documents, and an LLM extracts entities
("Andhra Pradesh", "Marlboro", "TicTac") and relationships between them
("Andhra Pradesh HAS Out-of-Stock % of 9% for Ferrero in August 2024") out
of the prose. So Phase 3's two KPI tables (numbers in a grid) need to become
sentences before GraphRAG can do anything useful with them.

This script turns every (state, month) combination -- 672 of them, matching
Phase 3's kpi_state_month.csv grain -- into one short narrative .txt file.
Each document:
  1. Opens by naming the state and the month explicitly (in words, e.g.
     "August 2024", not just "2024-08") -- once a document's text is inside
     the graph, nothing else anchors it in time, so the time reference has
     to live in the sentence itself.
  2. Describes that state's overall KPIs for that month (from
     kpi_state_month.csv): Productivity, SKUs/Transaction, Service Level,
     Dropsize, Inventory Turns, Inventory Days.
  3. Names the Wholesale Distributors (WDs) operating in that state (from
     geography.csv), so GraphRAG has a state->WD relationship to extract
     even though we don't have WD-level KPIs yet.
  4. Adds one paragraph per product category (GPI, IPM, Ferrero, Candy)
     with that category's KPIs for the state/month (from
     kpi_state_month_category.csv: Numeric Distribution, ACV,
     Out-of-Stock %, Range Billing) and names the franchises that sit
     under that category (from products.csv).

Every entity name (state, WD, category, franchise) is copied verbatim from
Phase 2/3's source tables -- never reworded or abbreviated -- so that the
same real-world thing is spelled identically across all 672 documents.
If GraphRAG saw "Andhra Pradesh" in one document and "AP" in another, it
might treat them as two different entities instead of linking them.

OUTPUT:
  data/graphrag_input/<State>_<YYYY-MM>.txt   -- one file per state/month
  data/graphrag_docs_manifest.csv             -- filename -> state/month
    lookup table, so later phases (retrieval, citations) can trace an
    answer back to the exact source document without re-parsing the text.

Run with:  python -m src.graph.build_documents
(from the project root, with the venv activated)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import get_settings

# Fixed category order so every document's category paragraphs appear in
# the same sequence -- makes documents easier to compare/read and keeps
# generation deterministic (no dependence on dict/groupby ordering).
CATEGORY_ORDER = ["GPI", "IPM", "Ferrero", "Candy"]

# Three of the ten KPIs are explicitly marked "NEEDS GPIL CONFIRMATION" in
# compute_kpis.py (Dropsize's definition, ACV's tier-weighting, and Range
# Billing's definition). Per the document-structure design doc Section 2,
# only these three metric mentions get a confirmation-status qualifier in
# document text -- adding it to all ten would train extraction/readers to
# treat every number as equally uncertain, which isn't true.
DROPSIZE_QUALIFIER = (
    "units ordered per productive visit — definition pending GPIL confirmation"
)
ACV_QUALIFIER = "weighted distribution — definition pending GPIL confirmation"
RANGE_BILLING_QUALIFIER = (
    "share of the category's SKU range billed, among outlets that billed "
    "anything — definition pending GPIL confirmation"
)

# Design Decision #1 / #1a (BUSINESS_KNOWLEDGE_DICTIONARY.md Section 6): a
# distributor's KPI value is called out in document text only if it differs
# from its state's average by more than this many percentage points, in the
# unfavorable direction (lower for Productivity/Service Level/Numeric
# Distribution/ACV/Range Billing, higher for Out-of-Stock %). These are
# agreed tunable design choices, not statistically derived thresholds.
# Originally 6pp; tightened to 10pp after the 7-document pilot showed too
# many call-outs per document at 6pp (e.g. Maharashtra 2024-08 had 20
# call-outs across 11 distributors, Sikkim 2025-04 had 28 across 13).
PP_DEVIATION_THRESHOLD = 10.0

# Dropsize is an absolute unit count, not a ratio, so a percentage-point
# threshold doesn't apply -- Design Decision #1a instead uses a relative
# deviation from the state average, and (per its own worked example) treats
# deviation in *either* direction as notable, unlike the other metrics.
# Same 6% -> 10% tightening as PP_DEVIATION_THRESHOLD, for consistency.
DROPSIZE_RELATIVE_THRESHOLD_PCT = 10.0

# (state-grain column, display label, "lower_bad" | "higher_bad" | "both")
STATE_GRAIN_DEVIATION_METRICS = [
    ("productivity", "Productivity", "lower_bad"),
    ("service_level", "Service Level", "lower_bad"),
]

# Category-grain deviation metrics, same tuple shape, evaluated once per
# category present in a given state/month.
CATEGORY_GRAIN_DEVIATION_METRICS = [
    ("numeric_distribution", "Numeric Distribution", "lower_bad"),
    ("acv", "ACV", "lower_bad"),
    ("oos_pct", "Out-of-Stock Rate", "higher_bad"),
    ("range_billing", "Range Billing", "lower_bad"),
]

# Category x Channel-grain deviation metrics (Step 4) -- same three of the
# four category-grain metrics that exist at channel grain. OOS% is absent
# on purpose: kpi_state_month_channel.csv has no oos_pct column, since
# inventory_snapshots.csv (OOS%'s only source) carries no channel_type
# dimension to split by (see compute_kpis.py's Table B2 module comment).
CHANNEL_GRAIN_DEVIATION_METRICS = [
    ("numeric_distribution", "Numeric Distribution", "lower_bad"),
    ("acv", "ACV", "lower_bad"),
    ("range_billing", "Range Billing", "lower_bad"),
]


def _month_to_words(month_str: str) -> str:
    """
    Turn "2024-08" into "August 2024".

    The KPI CSVs store months as "YYYY-MM" strings (sortable, compact).
    But a GraphRAG document is meant to be read like a short report, and
    "August 2024" is unambiguous prose whereas "2024-08" could be
    misread or extracted as a plain number by the LLM doing entity
    extraction. We keep the words-only form in the document text.
    """
    return datetime.strptime(month_str, "%Y-%m").strftime("%B %Y")


def _safe_filename(state_name: str, month_str: str) -> str:
    """
    Build a filesystem-safe, unique filename for one state/month document.

    Spaces in state names (e.g. "Andhra Pradesh") aren't ideal in
    filenames on every OS/tool, so they're replaced with underscores.
    The month stays in "YYYY-MM" form here (not the word form) so files
    sort chronologically when listed alphabetically.
    """
    safe_state = state_name.replace(" ", "_")
    return f"{safe_state}_{month_str}.txt"


def _format_pct(value: float) -> str:
    """Render a 0-1 ratio KPI as a percentage string, e.g. 0.4619 -> '46.2%'."""
    return f"{value * 100:.1f}%"


def load_source_tables(data_dir: Path) -> dict:
    """
    Load every table Phase 4 needs: Phase 3's two state-grain KPI CSVs, the
    two WD-grain KPI CSVs (for the distributor-deviations section), plus
    Phase 2's geography.csv (for WD names) and products.csv (for franchise
    names).

    Returns a dict of DataFrames keyed by table name, so the rest of the
    script can pass one object around instead of six separate arguments.
    """
    return {
        "kpi_state_month": pd.read_csv(data_dir / "kpi_state_month.csv"),
        "kpi_state_month_category": pd.read_csv(
            data_dir / "kpi_state_month_category.csv"
        ),
        "kpi_state_month_channel": pd.read_csv(
            data_dir / "kpi_state_month_channel.csv"
        ),
        "kpi_wd_month": pd.read_csv(data_dir / "kpi_wd_month.csv"),
        "kpi_wd_month_category": pd.read_csv(
            data_dir / "kpi_wd_month_category.csv"
        ),
        "geography": pd.read_csv(data_dir / "geography.csv"),
        "products": pd.read_csv(data_dir / "products.csv"),
    }


def build_zone_lookup(geography: pd.DataFrame) -> dict:
    """
    Build a {state_name: [(zone_name, [WD names in that zone]), ...]}
    lookup from geography.csv, ordered by each row's unit_id (zero-padded,
    e.g. "ZN0027" < "ZN0028"), so zones and their WDs always list in the
    same order across runs.

    A WD row only carries its immediate parent (a Zone's unit_id) in
    parent_unit_id, not the zone's name, so WDs are matched to zones by
    joining on that id.
    """
    zone_rows = geography[geography["unit_type"] == "Zone"].sort_values("unit_id")
    wd_rows = geography[geography["unit_type"] == "WD"].sort_values("unit_id")

    lookup = {}
    for _, zone_row in zone_rows.iterrows():
        state_name = zone_row["state_name"]
        zone_wds = wd_rows[wd_rows["parent_unit_id"] == zone_row["unit_id"]][
            "unit_name"
        ].tolist()
        lookup.setdefault(state_name, []).append((zone_row["unit_name"], zone_wds))
    return lookup


def build_franchise_lookup(products: pd.DataFrame) -> dict:
    """
    Build a {category_name: [franchise names]} lookup from products.csv.

    Franchise names don't vary by state or month (they're a fixed part of
    GPIL's product hierarchy), so this lookup only needs to be built once
    and reused for every document.
    """
    return (
        products.groupby("category_name")["franchise_name"]
        .unique()
        .apply(list)
        .to_dict()
    )


def find_deviating_distributors(
    wd_rows: pd.DataFrame,
    metric_col: str,
    state_value: float,
    direction: str,
    name_col: str = "wd_name",
    sort_col: str = "wd_id",
) -> list:
    """
    Compare every distributor's value for one KPI against the state
    average for that same KPI, and return the ones that cross the
    Design Decision #1/#1a threshold (BUSINESS_KNOWLEDGE_DICTIONARY.md
    Section 6).

    direction controls which side of the average counts as "unfavorable"
    (threshold values come from PP_DEVIATION_THRESHOLD /
    DROPSIZE_RELATIVE_THRESHOLD_PCT above, not hard-coded here):
      "lower_bad"  -- a WD more than PP_DEVIATION_THRESHOLD points BELOW
                       the state average triggers (e.g. Productivity,
                       Service Level, Numeric Distribution, ACV, Range
                       Billing)
      "higher_bad" -- a WD more than PP_DEVIATION_THRESHOLD points ABOVE
                       the state average triggers (Out-of-Stock %, where
                       higher is worse)
      "both"       -- Dropsize only: more than
                       DROPSIZE_RELATIVE_THRESHOLD_PCT relative deviation
                       in EITHER direction triggers, since Dropsize has no
                       inherent "good" direction the way a rate does

    name_col/sort_col let this same function double as the Step 4
    channel-deviation check: pass name_col="channel_type",
    sort_col="channel_type" to compare channels instead of distributors --
    the threshold logic itself doesn't change, only which column names
    the compared entity.

    Returns a list of (name, value, gap) tuples, sorted by sort_col so
    output order is deterministic across runs. gap is always a positive
    number -- percentage points for "lower_bad"/"higher_bad", relative
    percent for "both".
    """
    wd_rows = wd_rows.sort_values(sort_col)
    deviations = []
    for _, row in wd_rows.iterrows():
        wd_value = row[metric_col]
        if direction == "lower_bad":
            gap = (state_value - wd_value) * 100
            crosses = gap > PP_DEVIATION_THRESHOLD
        elif direction == "higher_bad":
            gap = (wd_value - state_value) * 100
            crosses = gap > PP_DEVIATION_THRESHOLD
        else:  # "both" -- Dropsize's relative-deviation rule
            gap = (wd_value - state_value) / state_value * 100
            crosses = abs(gap) > DROPSIZE_RELATIVE_THRESHOLD_PCT
        if crosses:
            deviations.append((row[name_col], wd_value, abs(gap)))
    return deviations


def render_distributor_deviations(
    state_name: str,
    month_words: str,
    state_row: pd.Series,
    category_rows: pd.DataFrame,
    wd_month_rows: pd.DataFrame,
    wd_month_category_rows: pd.DataFrame,
) -> str:
    """
    Render the "Distributor-level deviations" section (design doc Section
    3): one sentence per distributor that crosses the anomaly threshold on
    a given metric, unchanged from before -- that's the actual signal.

    What changed: metrics with zero triggers that month no longer each get
    their own "No distributor showed a significant deviation..." sentence
    (Design Decision #2's original form). With up to 11 metrics per
    document and most months triggering nothing, that produced 8-11
    near-identical filler sentences per document. Instead, every no-
    deviation metric for the month is now collected into a single
    "No distributor deviation this month for: ..." line, so absence-of-
    anomaly is still stated explicitly and remains queryable per metric
    (each metric name still appears verbatim), just without one full
    sentence of boilerplate per metric.

    Deliberately states deviations as fact only ("X vs. state average of
    Y, a gap of Z") with no causal "because" language -- see design doc
    Section 4.
    """
    lines = [f"Distributor-level deviations, {month_words}:"]
    deviation_lines = []
    no_deviation_labels = []

    # State-grain metrics (Productivity, Service Level) from
    # kpi_wd_month.csv, compared against kpi_state_month.csv's row.
    for col, label, direction in STATE_GRAIN_DEVIATION_METRICS:
        state_value = state_row[col]
        deviations = find_deviating_distributors(
            wd_month_rows, col, state_value, direction
        )
        if deviations:
            for wd_name, wd_value, gap in deviations:
                deviation_lines.append(
                    f"Distributor {wd_name} showed a significant deviation "
                    f"on {label} in {month_words}: {_format_pct(wd_value)} "
                    f"vs. the state average of {_format_pct(state_value)}, "
                    f"a gap of {gap:.1f} percentage points."
                )
        else:
            no_deviation_labels.append(label)

    # Dropsize -- also state-grain, but its own relative-deviation rule
    # (see find_deviating_distributors' "both" branch) and unit-count
    # formatting instead of a percentage.
    dropsize_state_value = state_row["dropsize"]
    dropsize_deviations = find_deviating_distributors(
        wd_month_rows, "dropsize", dropsize_state_value, "both"
    )
    if dropsize_deviations:
        for wd_name, wd_value, gap in dropsize_deviations:
            deviation_lines.append(
                f"Distributor {wd_name} showed a significant deviation on "
                f"Dropsize in {month_words}: {wd_value:.2f} vs. the state "
                f"average of {dropsize_state_value:.2f}, a gap of "
                f"{gap:.1f} percent."
            )
    else:
        no_deviation_labels.append("Dropsize")

    # Category-grain metrics (ND, ACV, OOS%, Range Billing) from
    # kpi_wd_month_category.csv, one pass per category present this
    # state/month, compared against kpi_state_month_category.csv's row
    # for that same category.
    category_rows_by_name = category_rows.set_index("category_name")
    for category in CATEGORY_ORDER:
        if category not in category_rows_by_name.index:
            continue
        cat_state_row = category_rows_by_name.loc[category]
        cat_wd_rows = wd_month_category_rows[
            wd_month_category_rows["category_name"] == category
        ]
        for col, label, direction in CATEGORY_GRAIN_DEVIATION_METRICS:
            state_value = cat_state_row[col]
            scoped_label = f"{label} ({category})"
            deviations = find_deviating_distributors(
                cat_wd_rows, col, state_value, direction
            )
            if deviations:
                for wd_name, wd_value, gap in deviations:
                    deviation_lines.append(
                        f"Distributor {wd_name} showed a significant "
                        f"deviation on {label} for the {category} category "
                        f"in {month_words}: {_format_pct(wd_value)} vs. "
                        f"the state average of {_format_pct(state_value)}, "
                        f"a gap of {gap:.1f} percentage points."
                    )
            else:
                no_deviation_labels.append(scoped_label)

    # No-deviation metrics first, as one dense summary line -- keeps every
    # metric name queryable without one boilerplate sentence each.
    if no_deviation_labels:
        lines.append(
            "No distributor deviation this month for: "
            + ", ".join(no_deviation_labels) + "."
        )

    # Then every real deviation, still one explicit sentence each -- this
    # is the actual signal and stays uncompressed.
    lines.extend(deviation_lines)

    if not no_deviation_labels and not deviation_lines:
        # Only reachable if a state/month has zero categories and the
        # state-grain loops produced nothing, which shouldn't happen given
        # every state always has Productivity/Service Level/Dropsize rows
        # -- kept as a safety net, not expected to render in practice.
        lines.append(
            f"No distributor-level deviation data available for "
            f"{state_name} in {month_words}."
        )

    return "\n".join(lines)


def render_channel_performance(
    state_name: str,
    month_words: str,
    category_rows: pd.DataFrame,
    channel_month_rows: pd.DataFrame,
) -> str:
    """
    Render the "Channel-level performance" section (Step 4): same
    deviation-detection style as render_distributor_deviations(), just
    comparing channel_type values (from kpi_state_month_channel.csv)
    against the category's state average instead of comparing
    distributors. Reuses find_deviating_distributors() unchanged, passing
    name_col/sort_col="channel_type" instead of the wd defaults.

    Only Numeric Distribution, ACV, and Range Billing are checked --
    CHANNEL_GRAIN_DEVIATION_METRICS omits Out-of-Stock Rate on purpose,
    since it doesn't exist at channel grain (see that constant's comment).
    """
    lines = [f"Channel-level performance, {month_words}:"]
    deviation_lines = []
    no_deviation_labels = []

    category_rows_by_name = category_rows.set_index("category_name")
    for category in CATEGORY_ORDER:
        if category not in category_rows_by_name.index:
            continue
        cat_state_row = category_rows_by_name.loc[category]
        cat_channel_rows = channel_month_rows[
            channel_month_rows["category_name"] == category
        ]
        for col, label, direction in CHANNEL_GRAIN_DEVIATION_METRICS:
            state_value = cat_state_row[col]
            scoped_label = f"{label} ({category})"
            deviations = find_deviating_distributors(
                cat_channel_rows,
                col,
                state_value,
                direction,
                name_col="channel_type",
                sort_col="channel_type",
            )
            if deviations:
                for channel_name, channel_value, gap in deviations:
                    deviation_lines.append(
                        f"The {channel_name} channel showed a significant "
                        f"deviation on {label} for the {category} category "
                        f"in {month_words}: {_format_pct(channel_value)} "
                        f"vs. the state average of {_format_pct(state_value)}, "
                        f"a gap of {gap:.1f} percentage points."
                    )
            else:
                no_deviation_labels.append(scoped_label)

    if no_deviation_labels:
        lines.append(
            "No channel deviation this month for: "
            + ", ".join(no_deviation_labels) + "."
        )
    lines.extend(deviation_lines)

    if not no_deviation_labels and not deviation_lines:
        # Safety net mirroring render_distributor_deviations' -- not
        # expected to render since every state always has at least one
        # category present.
        lines.append(
            f"No channel-level performance data available for "
            f"{state_name} in {month_words}."
        )

    return "\n".join(lines)


def render_document(
    state_name: str,
    month_str: str,
    state_row: pd.Series,
    category_rows: pd.DataFrame,
    zone_lookup: dict,
    franchise_lookup: dict,
    wd_month_rows: pd.DataFrame,
    wd_month_category_rows: pd.DataFrame,
    channel_month_rows: pd.DataFrame,
) -> str:
    """
    Render the full narrative text for one state/month document.

    Takes the one matching row from kpi_state_month.csv (state_row) and
    the (up to 4) matching rows from kpi_state_month_category.csv
    (category_rows), plus the Zone/WD and franchise name lookups and the
    matching rows from kpi_state_month_channel.csv (channel_month_rows),
    and returns the finished document as a single string.
    """
    month_words = _month_to_words(month_str)

    lines = []

    # 0. Entity tag lines -- State/Period, in the exact "EntityType: name"
    # form the design doc specifies (Section 1), so an extractor doesn't
    # have to infer entity type/casing from prose alone.
    lines.append(f"State: {state_name}\nPeriod: {month_words}")

    # Zone section -- Step 2: names the state's Zones so GraphRAG has a
    # State->Zone relationship, right after the entity tag lines and
    # before everything else (per the requested placement).
    zones = zone_lookup.get(state_name, [])
    if zones:
        zone_names = [zone_name for zone_name, _ in zones]
        lines.append(
            f"{state_name} is divided into {len(zone_names)} zones: "
            + ", ".join(zone_names) + "."
        )

    # 1. Opening sentence -- names the state and month explicitly so the
    # document is self-contained about *when* it describes.
    lines.append(
        f"This document reports Sales & Distribution performance for "
        f"{state_name} in {month_words}."
    )

    # 2. State-level (category-agnostic) KPI paragraph -- one sentence per
    # metric, "[Metric] for [Scope] in [Period] was [Value]." (design doc
    # Section 2), with the confirmation-status qualifier on Dropsize only.
    lines.append(
        f"Productivity for {state_name} in {month_words} was "
        f"{_format_pct(state_row['productivity'])}. "
        f"Service Level for {state_name} in {month_words} was "
        f"{_format_pct(state_row['service_level'])}. "
        f"SKUs per Transaction for {state_name} in {month_words} was "
        f"{state_row['skus_per_transaction']:.2f}. "
        f"Dropsize ({DROPSIZE_QUALIFIER}) for {state_name} in "
        f"{month_words} was {state_row['dropsize']:.2f}. "
        f"Inventory Turns for {state_name} in {month_words} was "
        f"{state_row['inventory_turns']:.2f} times. "
        f"Inventory Days for {state_name} in {month_words} was "
        f"{state_row['inventory_days']:.1f} days."
    )

    # 3. WD paragraph -- names the Distributors operating in this state,
    # grouped by Zone, so GraphRAG can extract both a State->Distributor
    # relationship and a Zone->Distributor relationship. Always
    # "Distributor", never "WD", per the schema's naming-collision warning
    # ("WD" vs. the unrelated "Dealer" channel-type value).
    zones = zone_lookup.get(state_name, [])
    if zones:
        zone_wd_lines = [
            f"{zone_name} distributors: " + ", ".join(zone_wds) + "."
            for zone_name, zone_wds in zones
            if zone_wds
        ]
        if zone_wd_lines:
            lines.append("\n".join(zone_wd_lines))

    # 4. Distributor-level deviations -- new section, flags any WD whose
    # KPI crosses the Design Decision #1/#1a threshold vs. its state
    # average, with an explicit no-anomaly sentence per metric otherwise.
    lines.append(
        render_distributor_deviations(
            state_name=state_name,
            month_words=month_words,
            state_row=state_row,
            category_rows=category_rows,
            wd_month_rows=wd_month_rows,
            wd_month_category_rows=wd_month_category_rows,
        )
    )

    # 4b. Channel-level performance -- Step 4: same deviation-flagging
    # style as section 4, but comparing channel_type values (Retail,
    # Dealer, Hawkers, Modern Trade) against the category's state average
    # instead of comparing distributors. No OOS% here -- see
    # CHANNEL_GRAIN_DEVIATION_METRICS' comment for why.
    lines.append(
        render_channel_performance(
            state_name=state_name,
            month_words=month_words,
            category_rows=category_rows,
            channel_month_rows=channel_month_rows,
        )
    )

    # 5. One paragraph per category, in a fixed order, using the category-
    # scoped KPIs and the franchises that belong to that category.
    # Structure unchanged from the original documents (design doc Section
    # 7) -- only the ACV/Range Billing confirmation-status qualifiers are
    # new here.
    category_rows_by_name = category_rows.set_index("category_name")
    for category in CATEGORY_ORDER:
        if category not in category_rows_by_name.index:
            continue
        row = category_rows_by_name.loc[category]
        franchises = franchise_lookup.get(category, [])
        franchise_list = ", ".join(franchises) if franchises else "no franchises on record"
        lines.append(
            f"For the {category} category (franchises: {franchise_list}) in "
            f"{state_name} during {month_words}: Numeric Distribution was "
            f"{_format_pct(row['numeric_distribution'])}, ACV "
            f"({ACV_QUALIFIER}) was {_format_pct(row['acv'])}, "
            f"Out-of-Stock rate was {_format_pct(row['oos_pct'])}, and "
            f"average Range Billing ({RANGE_BILLING_QUALIFIER}) was "
            f"{_format_pct(row['range_billing'])}."
        )

    return "\n\n".join(lines) + "\n"


def build_all_documents(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Build all 672 state/month documents and write them to output_dir.

    Returns the manifest DataFrame (filename -> state_name/month) that
    also gets written to disk, so callers (including tests) can inspect
    it without re-reading the CSV.
    """
    tables = load_source_tables(data_dir)
    kpi_state_month = tables["kpi_state_month"]
    kpi_state_month_category = tables["kpi_state_month_category"]
    kpi_state_month_channel = tables["kpi_state_month_channel"]
    kpi_wd_month = tables["kpi_wd_month"]
    kpi_wd_month_category = tables["kpi_wd_month_category"]

    zone_lookup = build_zone_lookup(tables["geography"])
    franchise_lookup = build_franchise_lookup(tables["products"])

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Iterate the state x month grain in a fixed, sorted order so output
    # is deterministic regardless of the source CSV's row order.
    sorted_rows = kpi_state_month.sort_values(["state_name", "month"])
    total = len(sorted_rows)
    for i, (_, state_row) in enumerate(sorted_rows.iterrows(), start=1):
        state_name = state_row["state_name"]
        month_str = state_row["month"]

        category_rows = kpi_state_month_category[
            (kpi_state_month_category["state_name"] == state_name)
            & (kpi_state_month_category["month"] == month_str)
        ]
        wd_month_rows = kpi_wd_month[
            (kpi_wd_month["state_name"] == state_name)
            & (kpi_wd_month["month"] == month_str)
        ]
        wd_month_category_rows = kpi_wd_month_category[
            (kpi_wd_month_category["state_name"] == state_name)
            & (kpi_wd_month_category["month"] == month_str)
        ]
        channel_month_rows = kpi_state_month_channel[
            (kpi_state_month_channel["state_name"] == state_name)
            & (kpi_state_month_channel["month"] == month_str)
        ]

        document_text = render_document(
            state_name=state_name,
            month_str=month_str,
            state_row=state_row,
            category_rows=category_rows,
            zone_lookup=zone_lookup,
            franchise_lookup=franchise_lookup,
            wd_month_rows=wd_month_rows,
            wd_month_category_rows=wd_month_category_rows,
            channel_month_rows=channel_month_rows,
        )

        filename = _safe_filename(state_name, month_str)
        (output_dir / filename).write_text(document_text, encoding="utf-8")

        # Print progress for every document so a rerun is never a silent
        # black box -- with 672 files, seeing each one written (and the
        # running count) makes it obvious the script is alive and exactly
        # how far it's gotten.
        print(f"[{i}/{total}] wrote {filename}", flush=True)

        manifest_rows.append(
            {"filename": filename, "state_name": state_name, "month": month_str}
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(data_dir / "graphrag_docs_manifest.csv", index=False)
    return manifest


def build_pilot_documents(
    data_dir: Path, output_dir: Path, state_months: list
) -> dict:
    """
    Render and write documents for a specific, small list of (state_name,
    month_str) pairs -- used to pilot a template change against a handful
    of real documents before paying the cost of regenerating all 672.

    Unlike build_all_documents(), this does NOT touch
    graphrag_docs_manifest.csv (the full manifest still describes the full
    672-document corpus; a partial pilot run shouldn't overwrite that).

    Returns {filename: document_text} for the caller to inspect/print.
    """
    tables = load_source_tables(data_dir)
    kpi_state_month = tables["kpi_state_month"]
    kpi_state_month_category = tables["kpi_state_month_category"]
    kpi_state_month_channel = tables["kpi_state_month_channel"]
    kpi_wd_month = tables["kpi_wd_month"]
    kpi_wd_month_category = tables["kpi_wd_month_category"]

    zone_lookup = build_zone_lookup(tables["geography"])
    franchise_lookup = build_franchise_lookup(tables["products"])

    output_dir.mkdir(parents=True, exist_ok=True)

    documents = {}
    for state_name, month_str in state_months:
        state_row = kpi_state_month[
            (kpi_state_month["state_name"] == state_name)
            & (kpi_state_month["month"] == month_str)
        ].iloc[0]
        category_rows = kpi_state_month_category[
            (kpi_state_month_category["state_name"] == state_name)
            & (kpi_state_month_category["month"] == month_str)
        ]
        wd_month_rows = kpi_wd_month[
            (kpi_wd_month["state_name"] == state_name)
            & (kpi_wd_month["month"] == month_str)
        ]
        wd_month_category_rows = kpi_wd_month_category[
            (kpi_wd_month_category["state_name"] == state_name)
            & (kpi_wd_month_category["month"] == month_str)
        ]
        channel_month_rows = kpi_state_month_channel[
            (kpi_state_month_channel["state_name"] == state_name)
            & (kpi_state_month_channel["month"] == month_str)
        ]

        document_text = render_document(
            state_name=state_name,
            month_str=month_str,
            state_row=state_row,
            category_rows=category_rows,
            zone_lookup=zone_lookup,
            franchise_lookup=franchise_lookup,
            wd_month_rows=wd_month_rows,
            wd_month_category_rows=wd_month_category_rows,
            channel_month_rows=channel_month_rows,
        )

        filename = _safe_filename(state_name, month_str)
        (output_dir / filename).write_text(document_text, encoding="utf-8")
        documents[filename] = document_text

    return documents


def main():
    """Entry point for `python -m src.graph.build_documents`."""
    settings = get_settings()
    data_dir = settings.project_root / "data"
    output_dir = data_dir / "graphrag_input"

    manifest = build_all_documents(data_dir, output_dir)

    print(f"Wrote {len(manifest)} documents to {output_dir}")
    print(f"Manifest saved to {data_dir / 'graphrag_docs_manifest.csv'}")


if __name__ == "__main__":
    main()
