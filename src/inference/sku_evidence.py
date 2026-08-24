"""
Phase 7e -- deterministic SKU-level evidence lookup (Problems 1 and 3).

WHAT THIS FILE IS FOR (plain language):
data/orders.csv carries real SKU-level facts (tens of millions of rows),
but is far too large to feed into GraphRAG directly, and the existing
GraphRAG-indexed document corpus (src/graph/build_documents.py) never
aggregates or renders anything at SKU grain -- so a SKU question gets zero
usable evidence out of GraphRAG's own retrieval, no matter how it's
phrased. src/kpis/compute_kpis.py's build_kpi_state_month_sku() already
solves the "aggregate, don't dump raw rows" half of that problem (see its
own module comment for the scale argument): a small State x Month x SKU
table, computed the exact same way every other KPI table in this project
is computed. This file solves the other half -- turning that table into
evidence the inference pipeline can actually use, WITHOUT re-indexing
GraphRAG:

  - render_sku_evidence_document() renders a fixed-template narrative
    sentence per SKU (mirrors src/graph/build_documents.py's own
    fixed-sentence-per-fact style, parsed back deterministically by
    src/inference/fact_structuring.py's matching regex -- see that
    module's "Atomic SKU Facts" section).
  - evaluate_sku_evidence() resolves a QueryRequirements (from
    query_requirements.py) against data/kpi_state_month_sku.csv and
    returns an EvidenceSufficiencyResult: either real, traceable evidence
    (as synthetic Sources-shaped rows pipeline.py merges into
    context_records before generation) or an EXPLICIT, honest explanation
    of why not -- distinguishing "the raw dataset really has no SKU data
    for this" (never true in this project) from "the aggregated
    representation has no rows for this specific state/period" (the real,
    common failure mode this function exists to name precisely, per
    Problem 3's brief).

WHY THIS IS NOT WIRED THROUGH GRAPHRAG'S OWN RETRIEVAL:
GraphRAG's local-search retrieval only ever returns evidence for whatever
happens to already be indexed and semantically close to the query
embedding. Re-indexing the corpus to include SKU documents would mean
re-running LLM entity extraction over the whole corpus again (real
completion + embedding spend) just to make ~39,600 already-structured,
already-known rows "retrievable" -- when a plain, deterministic lookup
keyed on the SAME state/period the question already names (exactly what
premise_check.py already does for state-level evidence, minus the
embedding step) answers the same question for free and with perfect
recall. This module is a parallel, ADDITIVE evidence source: it changes
nothing about how GraphRAG's own retrieval works for any other question,
and callers merge its output into context_records the same way
fact_structuring.py's Atomic Facts are additive to context_chunks.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.inference.schemas import EvidenceSufficiencyResult, QueryRequirements

# Mirrors ui_adapter.py's own PROJECT_ROOT computation, so this module has
# a sensible default data directory without importing config.settings (and
# therefore without a hard dependency on .env being loadable) at import time.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"

_SKU_KPI_FILENAME = "kpi_state_month_sku.csv"

# The exact dataframe column each real SKU metric label sorts by, for
# deterministic ranking (see _build_ranking_source_row() below) -- SAME
# five labels as fact_structuring.SKU_METRIC_FIELDS, in the SAME raw
# (pre-formatting) units render_sku_evidence_document() reads from before
# multiplying the three ratio columns by 100 for display. Sorting on the
# raw fraction vs. the displayed percentage gives an identical order (a
# monotonic *100 transform never changes a sort), so this deliberately
# does NOT duplicate that formatting step, just the column name mapping.
_METRIC_COLUMN_BY_LABEL = {
    "Units Delivered": "qty_delivered",
    "Revenue": "revenue",
    "Service Level": "service_level",
    "Numeric Distribution": "numeric_distribution",
    "Out-of-Stock Rate": "oos_pct",
}


@lru_cache(maxsize=8)
def _known_sku_names_cached(data_dir_str: str) -> frozenset[str]:
    try:
        df = load_sku_kpi_table(Path(data_dir_str))
    except FileNotFoundError:
        return frozenset()
    return frozenset(df["sku_name"].dropna().unique())


def known_sku_names(data_dir: Path | None = None) -> frozenset[str]:
    """Every distinct sku_name in data/kpi_state_month_sku.csv (e.g.
    'Marlboro Pack 1', 'GPI_Franchise_1 Pack 1') -- lets
    query_requirements.py recognize a SKU-grain question by an explicit
    SKU NAME even when the literal word 'SKU' is absent, without
    hardcoding a name list anywhere in code: the names come from the SAME
    single source of truth (this CSV) every other SKU-evidence lookup in
    this module already reads.

    Cached per data_dir (unlike load_sku_kpi_table(), deliberately NOT
    cached): the real data/kpi_state_month_sku.csv is the full ~39,600-row
    state x month x SKU table (~5.8MB), not a small per-query slice --
    since pipeline.py now calls this on EVERY question (see its
    2026-08-23 stabilization-pass docstring note), re-reading the whole
    file every time would be real, avoidable per-query overhead. Caching
    is keyed on the resolved directory string, so distinct test fixtures
    (each pytest tmp_path is a unique directory) never share a stale
    cache entry with each other or with the real project data/ directory;
    a test that writes its CSV once before ever calling this (the
    universal pattern in this project's tests) is unaffected either way.
    Returns an empty set (never raises) when the file is missing -- a
    caller with no SKU data available simply gets no bare-name matches,
    degrading to the pre-existing literal-'SKU'-word-only detection
    rather than crashing an otherwise-unrelated question."""
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return _known_sku_names_cached(str(data_dir))


def load_sku_kpi_table(data_dir: Path) -> pd.DataFrame:
    """Read data/kpi_state_month_sku.csv back into a DataFrame. Not cached
    at module scope on purpose -- this is called at most once per question
    (unlike, say, the GraphRAG engine ui_adapter.py caches for the whole
    process), the file is small (~39,600 rows), and an uncached read keeps
    this function trivially safe to call from tests against a fresh
    tmp_path data_dir every time, with no cross-test cache-staleness risk."""
    path = Path(data_dir) / _SKU_KPI_FILENAME
    return pd.read_csv(path, dtype={"month": str})


def _month_words_to_yyyymm(period_words: str) -> str | None:
    """'March 2025' -> '2025-03' (the kpi_state_month_sku.csv month
    format). None for anything unparseable."""
    try:
        return datetime.strptime(period_words, "%B %Y").strftime("%Y-%m")
    except ValueError:
        return None


def yyyymm_to_words(yyyymm: str) -> str:
    """'2025-03' -> 'March 2025' -- the inverse of _month_words_to_yyyymm().
    Public (not module-private) because build_sku_documents.py reuses it
    directly to label each per-month document it writes, rather than
    re-implementing the same date formatting a second time."""
    return datetime.strptime(yyyymm, "%Y-%m").strftime("%B %Y")


def indexed_range_label(df: pd.DataFrame) -> str:
    """A human-readable 'Month YYYY through Month YYYY' label for whatever
    months actually exist in the loaded SKU KPI table -- computed from the
    data itself (never a hardcoded constant), so it can never drift out of
    sync with however many months the table actually covers."""
    months = sorted(df["month"].dropna().unique())
    if not months:
        return "no indexed months"
    return f"{yyyymm_to_words(months[0])} through {yyyymm_to_words(months[-1])}"


def sku_rows_for_state_period(df: pd.DataFrame, state: str, period_words: str) -> pd.DataFrame:
    """Every SKU row for one state x one named month, sorted by sku_id for
    deterministic output. Empty (not an error) when the month doesn't
    parse or isn't present in the table."""
    month = _month_words_to_yyyymm(period_words)
    if month is None:
        return df.iloc[0:0]
    return df[(df["state_name"] == state) & (df["month"] == month)].sort_values("sku_id")


def sku_rows_for_state_year(df: pd.DataFrame, state: str, year: int) -> pd.DataFrame:
    """Aggregate every SKU's rows across all indexed months in `year` for
    `state` into one row per SKU: Units Ordered/Delivered and Revenue are
    summed (real totals across the year); Numeric Distribution and OOS%
    are averaged across the months that had data for that SKU (a plain
    mean over matched months -- this project has not confirmed a
    volume-weighting convention for these ratios at ANY grain, so this
    doesn't invent one here either); Service Level is RECOMPUTED from the
    summed totals (delivered/ordered), not averaged as a ratio-of-ratios,
    to avoid the classic mean-of-percentages bias. billed_outlets/
    eligible_outlets are dropped at this grain -- an outlet counted in
    March and again in April isn't "2 outlets," so a naive sum/mean of
    those columns across months would misrepresent distribution breadth;
    Numeric Distribution's own average is the honest year-level distribution
    signal instead. n_months records how many of the year's months are
    actually present, so callers can state partial-year coverage honestly.
    """
    year_rows = df[(df["state_name"] == state) & (df["month"].str.startswith(f"{year}-"))]
    if year_rows.empty:
        return year_rows

    agg = year_rows.groupby(
        ["sku_id", "sku_name", "franchise_name", "category_name"], as_index=False
    ).agg(
        qty_ordered=("qty_ordered", "sum"),
        qty_delivered=("qty_delivered", "sum"),
        revenue=("revenue", "sum"),
        numeric_distribution=("numeric_distribution", "mean"),
        oos_pct=("oos_pct", "mean"),
        n_months=("month", "nunique"),
    )
    agg["service_level"] = agg["qty_delivered"] / agg["qty_ordered"]
    agg["state_name"] = state
    return agg.sort_values("sku_id")


def _year_period_label(year: int, n_months: int | None) -> str:
    if n_months is None or n_months >= 12:
        return f"the year {year} (12 months aggregated)"
    return f"the year {year} ({n_months} of 12 months aggregated)"


def _latest_period_for_state(df: pd.DataFrame, state: str) -> str | None:
    """The most recently indexed month for `state`, in word form -- the
    deterministic default used only when a SKU question names no
    period/year at all (mirrors how a plain descriptive state-level
    question implicitly defaults to 'whatever's most recently retrieved';
    here there is no retrieval ranking to defer to, so the most recent
    indexed month is the only defensible deterministic choice)."""
    months = sorted(df.loc[df["state_name"] == state, "month"].dropna().unique())
    if not months:
        return None
    return yyyymm_to_words(months[-1])


def render_sku_evidence_document(state: str, period_label: str, rows: pd.DataFrame) -> str:
    """Render one fixed-template sentence per SKU row -- the SAME grammar
    src/inference/fact_structuring.py's SKU regex parses back
    deterministically (see that module's _SKU_FACT_RE), so this is the
    ONE place that knows this sentence's exact wording; the regex mirrors
    it, never redefines it independently (matching this codebase's
    established "generator and parser agree on GRAMMAR, not any specific
    value" convention -- see fact_structuring.py's module docstring).
    Self-contained per sentence (names its own state/period, no document
    header dependency), exactly like build_documents.py's category-block
    paragraphs -- so this text works identically whether it came from a
    real on-disk file (src/graph/build_sku_documents.py) or was rendered
    on the fly for one query (sku_evidence.py callers), and never depends
    on anything else appearing before it in the same context blob.

    `rows` must have the build_kpi_state_month_sku()/sku_rows_for_state_year()
    column shape (state_name is used only when the per-row column is
    absent, e.g. the year-aggregate path already sets it)."""
    lines = [
        f"SKU-level Sales & Distribution detail for {state}, {period_label}:",
    ]
    for _, row in rows.iterrows():
        units = row["qty_delivered"]
        revenue = row["revenue"]
        service_level = row["service_level"] * 100
        nd = row["numeric_distribution"] * 100
        oos = row["oos_pct"] * 100
        lines.append(
            f"SKU {row['sku_id']} ({row['sku_name']}, {row['franchise_name']} franchise, "
            f"{row['category_name']} category) in {state} during {period_label}: "
            f"{units:,.0f} units delivered, revenue of Rs {revenue:,.0f}, "
            f"Service Level {service_level:.1f}%, Numeric Distribution {nd:.1f}%, "
            f"Out-of-Stock rate {oos:.1f}%."
        )
    return "\n".join(lines) + "\n"


_RANKING_ROW_ID_PREFIX = "sku-ranking-"


def _format_metric_value(metric_label: str, raw_value: float) -> str:
    """Mirrors render_sku_evidence_document()'s own per-metric formatting
    (Revenue as a rupee amount, the three ratio metrics as a percentage,
    Units Delivered as a plain count) so the ranking block states each
    value in the SAME shape the underlying Atomic SKU Fact sentence does
    -- a reader (or grounding_check's co-occurrence check) comparing the
    two never sees the same number written two different ways."""
    if metric_label == "Revenue":
        return f"Rs {raw_value:,.0f}"
    if metric_label == "Units Delivered":
        return f"{raw_value:,.0f} units"
    return f"{raw_value * 100:.1f}%"


def rank_sku_rows(rows: pd.DataFrame, metric_label: str, direction: str, n: int) -> pd.DataFrame:
    """Sort `rows` (one row per SKU, the build_kpi_state_month_sku()/
    sku_rows_for_state_period() column shape) by `metric_label`'s raw
    column, descending for direction="top" / ascending for "bottom", and
    return the first `n` -- the deterministic ranking computation itself
    (Design Decision: the LLM must never be asked to compute a ranking
    from a pile of facts; this is the one place that actually sorts). A
    plain, boring pandas sort -- no LLM, no heuristics, no tie-breaking
    beyond pandas' own stable sort (ties keep their original sku_id order,
    which is already deterministic since sku_rows_for_state_period() sorts
    by sku_id)."""
    column = _METRIC_COLUMN_BY_LABEL[metric_label]
    ascending = direction == "bottom"
    return rows.sort_values(column, ascending=ascending, kind="stable").head(n)


def _build_ranking_source_row(
    state: str, period_label: str, metric_label: str, direction: str, ranked_rows: pd.DataFrame, evidence_source_id: str
) -> tuple[dict, list[dict]]:
    """Render the deterministically-ranked `ranked_rows` (already sorted
    and truncated to N by rank_sku_rows()) into ONE synthetic Sources-shaped
    row (id/text) plus the equivalent plain structured data
    (EvidenceSufficiencyResult.ranked_skus), for a single state/period scope.

    The rendered text is deliberately NOT the same sentence grammar
    render_sku_evidence_document() uses (fact_structuring._SKU_FACT_RE
    would otherwise re-parse it as ordinary, unordered Atomic SKU Facts,
    duplicating -- not replacing -- the ranking work this exists to avoid)
    -- see fact_structuring.build_deterministic_ranking_block(), which
    reads rows by this exact id PREFIX rather than by regex-matching this
    row's text shape. Each ranked entry still cites `evidence_source_id`
    (the SAME id as the regular, full sku_source_rows entry for this
    state/period) so grounding_check's existing citation resolution and
    entity+value+metric+period co-occurrence check work completely
    unchanged: the ranking block only tells the model WHICH SKUs and in
    WHAT ORDER, never a value that isn't independently, identically
    verifiable against the real per-SKU Atomic Fact sentence."""
    label = "highest" if direction == "top" else "lowest"
    lines = [
        f"This ranking was computed deterministically from the full underlying dataset for "
        f"{state}, {period_label}, and is authoritative -- restate it exactly as given below. "
        f"Do not recompute values, do not add or remove SKUs, and do not change the order.",
        f"Ranked by {metric_label} ({label} first):",
    ]
    ranked_skus: list[dict] = []
    for rank, (_, row) in enumerate(ranked_rows.iterrows(), start=1):
        value = float(row[_METRIC_COLUMN_BY_LABEL[metric_label]])
        # "Source: Sources (id)" -- NO square brackets, deliberately
        # matching every OTHER Atomic Facts section's own unbracketed
        # "Source: {id}" field convention (see fact_structuring.py's
        # _facts_from_source_text() et al.), instead of this project's
        # real in-text citation tag shape "[Data: dataset (id)]". Live-
        # caught bug (2026-08-23): an earlier version of this line used
        # "[Source: Sources (id)]" WITH brackets -- close enough to a real
        # "[Data: ...]" tag that the model, told to "restate this exactly
        # as given," sometimes copied it verbatim into the drafted prose
        # instead of writing a proper "[Data: Sources (id)]" tag. Since
        # _extract_citations_from_sentence() only recognizes the literal
        # "[Data: ...]" shape, those sentences were then read as citing
        # NOTHING, and (having also skipped the structured claims block
        # that same run) fell through to whole-blob/prose-fallback
        # scanning -- producing a cascade of spurious "does not mention"
        # failures for facts that were genuinely, correctly cited.
        lines.append(
            f"Rank {rank}: SKU {row['sku_id']} ({row['sku_name']}, {row['franchise_name']} franchise, "
            f"{row['category_name']} category) -- {metric_label}: {_format_metric_value(metric_label, value)}. "
            f"Source: Sources ({evidence_source_id})"
        )
        ranked_skus.append(
            {
                "rank": rank,
                "sku_id": row["sku_id"],
                "sku_name": row["sku_name"],
                "franchise_name": row["franchise_name"],
                "category_name": row["category_name"],
                "state": state,
                "period": period_label,
                "metric": metric_label,
                "direction": direction,
                "value": value,
                "source_id": evidence_source_id,
            }
        )
    safe_period = period_label.replace(" ", "_").replace("(", "").replace(")", "")
    safe_state = state.replace(" ", "_")
    row_id = f"{_RANKING_ROW_ID_PREFIX}{safe_state}-{safe_period}"
    return {"id": row_id, "text": "\n".join(lines) + "\n"}, ranked_skus


_RANK_COMPARISON_ROW_ID_PREFIX = "sku-ranking-comparison-"


def _build_rank_comparison_row(state: str, metric_label: str, ranked_skus: list[dict]) -> dict | None:
    """Deterministically compare the #1-ranked SKU across every period a
    multi-period ranking question resolved, and render ONE authoritative
    sentence per period-pair stating whether it's the SAME SKU or a
    DIFFERENT one -- and, when it's the same SKU, the exact numeric change
    in `metric_label` between the two periods, already subtracted.

    WHY THIS EXISTS (2026-08-23, SKU live validation pass): "Did the
    top-selling SKU in Gujarat change between April 2026 and June 2026?"
    live-failed grounding on both the first draft AND the bounded retry --
    not because the underlying ranking was wrong (each period's own
    ranking, verified separately, was already correct and already
    passing), but because the model tried to construct its OWN cross-
    period trend claim (restating each period's Units Delivered figure
    and/or computing their difference) and got the citation and/or the
    arithmetic wrong. Both of those are exactly the "LLM manually sorting/
    computing something deterministic code already knows" failure mode
    this project's whole SKU architecture exists to eliminate (see this
    module's docstring) -- the fix is not a better retry prompt, it's
    giving the model the ALREADY-COMPUTED comparison to restate, the same
    way _build_ranking_source_row() already does for a single period's
    ranking.

    Deliberately scoped to the #1 rank only (not every rank 1..N): every
    live example of this question shape asks about "the top-selling SKU"
    (singular), never "did the top-3 change" -- extending this to
    per-rank-position comparison across N>1 would be speculative scope
    this project has no evidence it needs yet. Returns None when fewer
    than 2 distinct periods contributed a ranking (nothing to compare) or
    `ranked_skus` is empty."""
    periods_seen: list[str] = []
    top_by_period: dict[str, dict] = {}
    for entry in ranked_skus:
        if entry["rank"] != 1:
            continue
        period = entry["period"]
        if period not in top_by_period:
            top_by_period[period] = entry
            periods_seen.append(period)
    if len(periods_seen) < 2:
        return None

    lines = [
        f"Deterministic cross-period comparison for {state}, Rank 1 by {metric_label}: "
        f"whether the top-ranked SKU changed between the periods below, and by how much when it "
        f"didn't, is computed exactly from the rankings above -- restate this conclusion exactly, "
        f"do not recompute it.",
    ]
    baseline_period = periods_seen[0]
    baseline = top_by_period[baseline_period]
    for period in periods_seen[1:]:
        other = top_by_period[period]
        if baseline["sku_id"] == other["sku_id"]:
            delta = other["value"] - baseline["value"]
            sign = "+" if delta >= 0 else "-"
            lines.append(
                f"The Rank 1 SKU by {metric_label} did NOT change between {baseline_period} and {period}: "
                f"{baseline['sku_name']} was #1 in both periods. {metric_label} moved from "
                f"{_format_metric_value(metric_label, baseline['value'])} ({baseline_period}) to "
                f"{_format_metric_value(metric_label, other['value'])} ({period}), a change of "
                f"{sign}{_format_metric_value(metric_label, abs(delta))}."
            )
        else:
            lines.append(
                f"The Rank 1 SKU by {metric_label} CHANGED between {baseline_period} and {period}: "
                f"{baseline['sku_name']} (#1 in {baseline_period}) vs. {other['sku_name']} (#1 in {period})."
            )
    row_id = f"{_RANK_COMPARISON_ROW_ID_PREFIX}{state.replace(' ', '_')}"
    return {"id": row_id, "text": "\n".join(lines) + "\n"}


def _source_id_for(state: str, period_label: str) -> str:
    """A stable, distinctive synthetic Sources id -- prefixed 'sku-' so it
    can never collide with GraphRAG's own numeric text-unit ids when the
    two are concatenated into one context_records["sources"] table (see
    pipeline.py)."""
    safe_period = period_label.replace(" ", "_").replace("(", "").replace(")", "")
    safe_state = state.replace(" ", "_")
    return f"sku-{safe_state}-{safe_period}"


def evaluate_sku_evidence(requirements: QueryRequirements, data_dir: Path | None = None) -> EvidenceSufficiencyResult:
    """The main entry point. Resolves `requirements` (from
    query_requirements.py) against data/kpi_state_month_sku.csv and
    returns whether real evidence was found, distinguishing an
    out-of-scope state/period from "the dataset has no SKU data" (which
    this project's data always has -- see this module's docstring)."""
    if requirements.required_granularity != "sku":
        return EvidenceSufficiencyResult(status="not_applicable", requirements=requirements)

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    if requirements.target_state is None:
        return EvidenceSufficiencyResult(
            status="insufficient_scope",
            requirements=requirements,
            explanation=(
                "The question asks a SKU-level question but does not name one of this "
                "project's 28 indexed states, so SKU evidence cannot be scoped to any "
                "single state. The underlying dataset DOES contain SKU-level order and "
                "inventory data (data/orders.csv, data/products.csv, "
                "data/inventory_snapshots.csv), aggregated per state/month into "
                "data/kpi_state_month_sku.csv -- it is only meaningful once scoped to a "
                "specific state."
            ),
        )

    df = load_sku_kpi_table(data_dir)
    rows_by_period: list[tuple[str, pd.DataFrame]] = []

    if requirements.target_periods:
        for period in requirements.target_periods:
            rows = sku_rows_for_state_period(df, requirements.target_state, period)
            if not rows.empty:
                rows_by_period.append((period, rows))
    elif requirements.target_year is not None:
        rows = sku_rows_for_state_year(df, requirements.target_state, requirements.target_year)
        if not rows.empty:
            n_months = int(rows["n_months"].iloc[0]) if "n_months" in rows.columns else None
            label = _year_period_label(requirements.target_year, n_months)
            rows_by_period.append((label, rows))
    else:
        latest = _latest_period_for_state(df, requirements.target_state)
        if latest is not None:
            rows = sku_rows_for_state_period(df, requirements.target_state, latest)
            if not rows.empty:
                rows_by_period.append((latest, rows))

    if not rows_by_period:
        if requirements.target_periods:
            scope_desc = ", ".join(requirements.target_periods)
        elif requirements.target_year is not None:
            scope_desc = str(requirements.target_year)
        else:
            scope_desc = "(no period named, and no indexed month exists for this state)"
        return EvidenceSufficiencyResult(
            status="insufficient_scope",
            requirements=requirements,
            explanation=(
                f"The underlying dataset DOES contain SKU-level order and inventory data "
                f"(data/orders.csv, data/products.csv, data/inventory_snapshots.csv), "
                f"aggregated per state/month into data/kpi_state_month_sku.csv -- but no "
                f"rows exist there for {requirements.target_state} at {scope_desc}. "
                f"The current indexed range is {indexed_range_label(df)}."
            ),
        )

    source_rows = []
    ranked_skus: list[dict] = []
    # Deterministic ranking (Design Decision: the LLM must never compute a
    # top-N/bottom-N SKU ranking itself): only attempted when the question
    # is unambiguously ranking-shaped (a real metric AND a clear
    # top/bottom direction are both already known -- see
    # query_requirements.py; an ambiguous-metric ranking question never
    # reaches here at all, short-circuited by pipeline.py's
    # needs_clarification gate first). One ranking computed PER resolved
    # period (rows_by_period may hold 2+ entries for a multi-period
    # question), so a "top 3 SKUs ... in January and February 2026"
    # question gets two independently-correct, independently-labeled
    # rankings, not one conflated/averaged one this project has never
    # defined a business rule for.
    can_rank = (
        requirements.is_ranking
        and requirements.ranking_metric is not None
        and requirements.rank_direction is not None
    )
    for period_label, rows in rows_by_period:
        evidence_id = _source_id_for(requirements.target_state, period_label)
        source_rows.append(
            {"id": evidence_id, "text": render_sku_evidence_document(requirements.target_state, period_label, rows)}
        )
        if can_rank:
            n = requirements.rank_n or 1
            if len(rows) >= n:
                ranked_rows = rank_sku_rows(rows, requirements.ranking_metric, requirements.rank_direction, n)
                ranking_row, entries = _build_ranking_source_row(
                    requirements.target_state, period_label, requirements.ranking_metric,
                    requirements.rank_direction, ranked_rows, evidence_id,
                )
                source_rows.append(ranking_row)
                ranked_skus.extend(entries)

    if can_rank:
        comparison_row = _build_rank_comparison_row(requirements.target_state, requirements.ranking_metric, ranked_skus)
        if comparison_row is not None:
            source_rows.append(comparison_row)

    periods_desc = ", ".join(p for p, _ in rows_by_period)
    return EvidenceSufficiencyResult(
        status="sufficient",
        requirements=requirements,
        sku_source_rows=source_rows,
        explanation=(
            f"SKU-level evidence resolved for {requirements.target_state} ({periods_desc}) "
            f"from data/kpi_state_month_sku.csv."
        ),
        ranked_skus=ranked_skus or None,
    )
