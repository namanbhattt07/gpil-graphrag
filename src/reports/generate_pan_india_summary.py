"""
Builds GPIL_Pan_India_Analysis.xlsx -- a ten-sheet summary workbook for
a quick Pan-India business review, built directly from the Phase 2 raw
tables (orders.csv, outlets.csv, products.csv). This is a one-off
reporting script, separate from the KPI tables in src/kpis (those are
State x Month grain for GraphRAG; this is simple category/state/
franchise/channel roll-ups for a spreadsheet).

Each of the five roll-ups (Category, State, Franchise, Channel x
Category, Channel x Tier) gets a totals sheet and a "_Monthly" sheet
that adds a leading month column (calendar month, "YYYY-MM", from
order_date) so trends over time are visible, not just Pan-India totals.

DEFINITIONS:
  - "units sold" = qty_delivered (what actually shipped), not
    qty_ordered, so it lines up with how src/kpis/compute_kpis.py
    treats delivered quantity as the real sales number.
  - revenue = qty_delivered * unit_price, using orders.csv's unit_price
    (the actual transaction price on that order line), not the catalog
    unit_price in products.csv.
  - units_sold_lakh = units / 1e5, revenue_cr = revenue / 1e7 (Indian
    lakh/crore units, since this is a Pan-India report).

Run with:  python -m src.reports.generate_pan_india_summary
(from the project root, with the venv activated)
"""

from pathlib import Path

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "GPIL_Pan_India_Analysis.xlsx"

LAKH = 1e5
CRORE = 1e7


def load_merged_orders() -> pd.DataFrame:
    """Join orders to products (for category/franchise) and outlets
    (for state/channel/tier), and compute the units/revenue columns
    every sheet below rolls up from."""
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    outlets = pd.read_csv(DATA_DIR / "outlets.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    merged = orders.merge(
        products[["sku_id", "franchise_name", "category_name"]], on="sku_id", how="left"
    ).merge(
        outlets[["outlet_id", "state_name", "channel_type", "outlet_tier"]],
        on="outlet_id",
        how="left",
    )

    merged["units_sold"] = merged["qty_delivered"]
    merged["revenue"] = merged["qty_delivered"] * merged["unit_price"]
    merged["month"] = pd.to_datetime(merged["order_date"]).dt.to_period("M").astype(str)
    return merged


def _units_revenue(m: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Shared group-by-and-convert-to-lakh/cr step behind the Category,
    State and Franchise roll-ups (both totals and monthly)."""
    grouped = (
        m.groupby(group_cols, as_index=False)
        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
    )
    grouped["units_sold_lakh"] = grouped["units_sold"] / LAKH
    grouped["revenue_cr"] = grouped["revenue"] / CRORE
    return grouped[group_cols + ["units_sold_lakh", "revenue_cr"]]


def _with_cigarettes_row(out: pd.DataFrame, m: pd.DataFrame, month_col: bool) -> pd.DataFrame:
    """Append a combined GPI+IPM row labelled 'Cigarettes' (GPIL's two
    cigarette franchises' shared parent category), one per month if
    month_col is set, else a single Pan-India row."""
    cigarettes = m[m["category_name"].isin(["GPI", "IPM"])]
    group_cols = ["month"] if month_col else []
    # category_name is fixed to "Cigarettes" for every row, so group only
    # by month (if present) and overwrite the label afterwards.
    cig = (
        cigarettes.groupby(group_cols, as_index=False).agg(
            units_sold=("units_sold", "sum"), revenue=("revenue", "sum")
        )
        if group_cols
        else pd.DataFrame(
            {"units_sold": [cigarettes["units_sold"].sum()], "revenue": [cigarettes["revenue"].sum()]}
        )
    )
    cig["category_name"] = "Cigarettes"
    cig["units_sold_lakh"] = cig["units_sold"] / LAKH
    cig["revenue_cr"] = cig["revenue"] / CRORE
    cig = cig[out.columns]
    return pd.concat([out, cig], ignore_index=True)


def build_category_units(m: pd.DataFrame) -> pd.DataFrame:
    """Sheet 1: units/revenue by category, plus a combined GPI+IPM row
    labelled 'Cigarettes' (GPIL's two cigarette franchises' shared
    parent category, per the GPIL hierarchy)."""
    out = _units_revenue(m, ["category_name"])
    return _with_cigarettes_row(out, m, month_col=False)


def build_category_units_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly version of Sheet 1: units/revenue by month x category,
    plus a per-month 'Cigarettes' (GPI+IPM) row, sorted by month then
    units_sold_lakh desc."""
    out = _units_revenue(m, ["month", "category_name"])
    out = _with_cigarettes_row(out, m, month_col=True)
    return out.sort_values(["month", "units_sold_lakh"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_state_units(m: pd.DataFrame) -> pd.DataFrame:
    """Sheet 2: units/revenue by state, sorted by units_sold_lakh desc."""
    out = _units_revenue(m, ["state_name"])
    return out.sort_values("units_sold_lakh", ascending=False).reset_index(drop=True)


def build_state_units_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly version of Sheet 2: units/revenue by month x state,
    sorted by month then units_sold_lakh desc."""
    out = _units_revenue(m, ["month", "state_name"])
    return out.sort_values(["month", "units_sold_lakh"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_franchise_units(m: pd.DataFrame) -> pd.DataFrame:
    """Sheet 3: units/revenue by franchise x category, sorted by
    units_sold_lakh desc. category_name is kept alongside franchise_name
    since it's the parent grouping GPIL reports franchises under."""
    out = _units_revenue(m, ["franchise_name", "category_name"])
    return out.sort_values("units_sold_lakh", ascending=False).reset_index(drop=True)


def build_franchise_units_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly version of Sheet 3: units/revenue by month x franchise x
    category, sorted by month then units_sold_lakh desc."""
    out = _units_revenue(m, ["month", "franchise_name", "category_name"])
    return out.sort_values(["month", "units_sold_lakh"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_channel_category(m: pd.DataFrame) -> pd.DataFrame:
    """Sheet 4: pivot of units_sold_lakh, rows=channel_type, columns=category_name."""
    pivot = pd.pivot_table(
        m, index="channel_type", columns="category_name", values="units_sold", aggfunc="sum"
    )
    pivot = (pivot / LAKH).reset_index()
    pivot.columns.name = None
    return pivot


def build_channel_category_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly version of Sheet 4: pivot of units_sold_lakh, rows=(month,
    channel_type), columns=category_name."""
    pivot = pd.pivot_table(
        m,
        index=["month", "channel_type"],
        columns="category_name",
        values="units_sold",
        aggfunc="sum",
    )
    pivot = (pivot / LAKH).reset_index()
    pivot.columns.name = None
    return pivot


def build_channel_tier_units(m: pd.DataFrame) -> pd.DataFrame:
    """Sheet 5: pivot of units_sold_lakh, rows=channel_type, columns=outlet_tier."""
    pivot = pd.pivot_table(
        m, index="channel_type", columns="outlet_tier", values="units_sold", aggfunc="sum"
    )
    pivot = (pivot / LAKH).reset_index()
    pivot.columns.name = None
    return pivot


def build_channel_tier_units_monthly(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly version of Sheet 5: pivot of units_sold_lakh, rows=(month,
    channel_type), columns=outlet_tier."""
    pivot = pd.pivot_table(
        m,
        index=["month", "channel_type"],
        columns="outlet_tier",
        values="units_sold",
        aggfunc="sum",
    )
    pivot = (pivot / LAKH).reset_index()
    pivot.columns.name = None
    return pivot


def format_sheet(worksheet) -> None:
    """Bold the header row and apply a 2-decimal number format to every
    numeric column, then widen columns to fit their content."""
    for cell in worksheet[1]:
        cell.font = Font(bold=True)

    for col_idx, column_cells in enumerate(worksheet.iter_cols(min_row=2), start=1):
        max_len = len(str(worksheet.cell(row=1, column=col_idx).value))
        for cell in column_cells:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.00"
            max_len = max(max_len, len(f"{cell.value}"))
        worksheet.column_dimensions[get_column_letter(col_idx)].width = max_len + 2


def main() -> None:
    merged = load_merged_orders()

    sheets = {
        "Category_Units": build_category_units(merged),
        "State_Units": build_state_units(merged),
        "Franchise_Units": build_franchise_units(merged),
        "Channel_Category": build_channel_category(merged),
        "Channel_Tier_Units": build_channel_tier_units(merged),
        "Category_Units_Monthly": build_category_units_monthly(merged),
        "State_Units_Monthly": build_state_units_monthly(merged),
        "Franchise_Units_Monthly": build_franchise_units_monthly(merged),
        "Channel_Category_Monthly": build_channel_category_monthly(merged),
        "Channel_Tier_Units_Monthly": build_channel_tier_units_monthly(merged),
    }

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_sheet(writer.sheets[sheet_name])

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
