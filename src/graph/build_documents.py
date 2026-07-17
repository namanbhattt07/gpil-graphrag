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
     kpi_state_month.csv): Strike Rate, SKUs/Transaction, Service Level,
     ACL, Inventory Turns, Inventory Days.
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
    Load every table Phase 4 needs: Phase 3's two KPI CSVs, plus Phase 2's
    geography.csv (for WD names) and products.csv (for franchise names).

    Returns a dict of DataFrames keyed by table name, so the rest of the
    script can pass one object around instead of four separate arguments.
    """
    return {
        "kpi_state_month": pd.read_csv(data_dir / "kpi_state_month.csv"),
        "kpi_state_month_category": pd.read_csv(
            data_dir / "kpi_state_month_category.csv"
        ),
        "geography": pd.read_csv(data_dir / "geography.csv"),
        "products": pd.read_csv(data_dir / "products.csv"),
    }


def build_wd_lookup(geography: pd.DataFrame) -> dict:
    """
    Build a {state_name: [WD names]} lookup from geography.csv.

    geography.csv already carries state_name on every row (including WD
    rows), so we don't need to walk the Zone parent chain to figure out
    which state a WD belongs to -- we can filter unit_type == "WD" and
    group directly by state_name.
    """
    wd_rows = geography[geography["unit_type"] == "WD"]
    return wd_rows.groupby("state_name")["unit_name"].apply(list).to_dict()


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


def render_document(
    state_name: str,
    month_str: str,
    state_row: pd.Series,
    category_rows: pd.DataFrame,
    wd_lookup: dict,
    franchise_lookup: dict,
) -> str:
    """
    Render the full narrative text for one state/month document.

    Takes the one matching row from kpi_state_month.csv (state_row) and
    the (up to 4) matching rows from kpi_state_month_category.csv
    (category_rows), plus the WD and franchise name lookups, and returns
    the finished document as a single string.
    """
    month_words = _month_to_words(month_str)

    lines = []

    # 1. Opening sentence -- names the state and month explicitly so the
    # document is self-contained about *when* it describes.
    lines.append(
        f"This document reports Sales & Distribution performance for "
        f"{state_name} in {month_words}."
    )

    # 2. State-level (category-agnostic) KPI paragraph.
    lines.append(
        f"In {state_name} during {month_words}, the Strike Rate "
        f"(share of sales visits that resulted in an order) was "
        f"{_format_pct(state_row['strike_rate'])}, and the average "
        f"Service Level (share of ordered quantity actually delivered) "
        f"was {_format_pct(state_row['service_level'])}. "
        f"Sales executives averaged {state_row['skus_per_transaction']:.2f} "
        f"SKUs per transaction and an Average Case Load (ACL) of "
        f"{state_row['acl']:.2f} units per productive visit. "
        f"Distributor inventory turned over {state_row['inventory_turns']:.2f} "
        f"times during the month, equivalent to {state_row['inventory_days']:.1f} "
        f"days of stock on hand on average."
    )

    # 3. WD paragraph -- names the Wholesale Distributors operating in
    # this state so GraphRAG can extract a state->WD relationship, even
    # though we have no WD-level KPI numbers yet.
    wds = wd_lookup.get(state_name, [])
    if wds:
        wd_list = ", ".join(wds)
        lines.append(
            f"{state_name} is served by {len(wds)} Wholesale Distributor(s): "
            f"{wd_list}."
        )

    # 4. One paragraph per category, in a fixed order, using the category-
    # scoped KPIs and the franchises that belong to that category.
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
            f"{_format_pct(row['numeric_distribution'])}, ACV (weighted "
            f"distribution) was {_format_pct(row['acv'])}, Out-of-Stock rate "
            f"was {_format_pct(row['oos_pct'])}, and average Range Billing "
            f"(share of the category's SKU range billed, among outlets that "
            f"billed anything) was {_format_pct(row['range_billing'])}."
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

    wd_lookup = build_wd_lookup(tables["geography"])
    franchise_lookup = build_franchise_lookup(tables["products"])

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # Iterate the state x month grain in a fixed, sorted order so output
    # is deterministic regardless of the source CSV's row order.
    for _, state_row in kpi_state_month.sort_values(
        ["state_name", "month"]
    ).iterrows():
        state_name = state_row["state_name"]
        month_str = state_row["month"]

        category_rows = kpi_state_month_category[
            (kpi_state_month_category["state_name"] == state_name)
            & (kpi_state_month_category["month"] == month_str)
        ]

        document_text = render_document(
            state_name=state_name,
            month_str=month_str,
            state_row=state_row,
            category_rows=category_rows,
            wd_lookup=wd_lookup,
            franchise_lookup=franchise_lookup,
        )

        filename = _safe_filename(state_name, month_str)
        (output_dir / filename).write_text(document_text, encoding="utf-8")

        manifest_rows.append(
            {"filename": filename, "state_name": state_name, "month": month_str}
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(data_dir / "graphrag_docs_manifest.csv", index=False)
    return manifest


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
