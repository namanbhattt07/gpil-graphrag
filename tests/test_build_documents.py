"""
Tests for the Phase 4 GraphRAG document builder.

Like the Phase 2/3 tests, these regenerate data in-memory from a fixed
seed (rather than depending on data/ CSVs already existing on disk) so
the suite is self-contained and reproducible on a fresh checkout. The
KPI/geography/product tables are written out to a temporary directory
so build_all_documents() can be exercised exactly as it runs in
production (reading CSVs, writing .txt files + a manifest).
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data_gen.generate_synthetic_data import generate_all, INDIAN_STATES, N_MONTHS
from src.kpis.compute_kpis import build_kpi_state_month, build_kpi_state_month_category
from src.graph.build_documents import (
    build_all_documents,
    _month_to_words,
    _safe_filename,
)

SEED = 42
N_STATES = len(INDIAN_STATES)
EXPECTED_CATEGORIES = {"GPI", "IPM", "Ferrero", "Candy"}


@pytest.fixture(scope="module")
def tables():
    """Reuse one generated dataset across every test in this file."""
    return generate_all(SEED)


@pytest.fixture(scope="module")
def kpi_state_month(tables):
    return build_kpi_state_month(tables)


@pytest.fixture(scope="module")
def kpi_state_month_category(tables):
    return build_kpi_state_month_category(tables)


@pytest.fixture(scope="module")
def built(tmp_path_factory, tables, kpi_state_month, kpi_state_month_category):
    """
    Write the four source CSVs build_documents.py expects into a
    temporary data_dir, run build_all_documents(), and return everything
    a test might want: the manifest, the data_dir, and the output_dir.
    """
    data_dir = tmp_path_factory.mktemp("data")
    output_dir = data_dir / "graphrag_input"

    tables["geography"].to_csv(data_dir / "geography.csv", index=False)
    tables["products"].to_csv(data_dir / "products.csv", index=False)
    kpi_state_month.to_csv(data_dir / "kpi_state_month.csv", index=False)
    kpi_state_month_category.to_csv(
        data_dir / "kpi_state_month_category.csv", index=False
    )

    manifest = build_all_documents(data_dir, output_dir)
    return {"manifest": manifest, "data_dir": data_dir, "output_dir": output_dir}


def test_month_to_words():
    """Sanity-check the date formatting helper directly, since every
    document's opening sentence depends on it being correct."""
    assert _month_to_words("2024-08") == "August 2024"
    assert _month_to_words("2026-01") == "January 2026"


def test_safe_filename_replaces_spaces():
    assert _safe_filename("Andhra Pradesh", "2024-08") == "Andhra_Pradesh_2024-08.txt"


def test_correct_number_of_documents_written(built):
    """One document per state x month -- 28 states x 24 months = 672."""
    output_files = list(built["output_dir"].glob("*.txt"))
    assert len(output_files) == N_STATES * N_MONTHS
    assert len(built["manifest"]) == N_STATES * N_MONTHS


def test_manifest_matches_written_files(built):
    """Every filename in the manifest must correspond to a real file on
    disk, and vice versa -- no orphaned files, no phantom manifest rows."""
    manifest_filenames = set(built["manifest"]["filename"])
    disk_filenames = {p.name for p in built["output_dir"].glob("*.txt")}
    assert manifest_filenames == disk_filenames


def test_every_document_names_its_state_and_month(built, kpi_state_month):
    """Each document must mention its own state name and an explicit
    (word-form) month string, so the document is self-contained about
    what it describes -- this is the core Phase 4 requirement."""
    for _, row in kpi_state_month.sample(20, random_state=1).iterrows():
        filename = _safe_filename(row["state_name"], row["month"])
        text = (built["output_dir"] / filename).read_text(encoding="utf-8")
        assert row["state_name"] in text
        assert _month_to_words(row["month"]) in text


def test_every_document_mentions_all_categories(built):
    """Every document should have a paragraph for all 4 categories --
    if any state/month were missing a category row, this would catch a
    silently-dropped paragraph."""
    sample_files = list(built["output_dir"].glob("*.txt"))[:15]
    for path in sample_files:
        text = path.read_text(encoding="utf-8")
        for category in EXPECTED_CATEGORIES:
            assert category in text, f"{path.name} is missing category {category}"


def test_entity_names_match_source_tables_verbatim(built, tables):
    """Franchise and WD names inside documents must be copied exactly
    from products.csv / geography.csv -- if names drifted (e.g. reworded
    or truncated), GraphRAG would treat the same real-world entity as
    two different ones across documents."""
    sample_path = next(built["output_dir"].glob("Andhra_Pradesh_*.txt"))
    text = sample_path.read_text(encoding="utf-8")

    ap_wds = tables["geography"][
        (tables["geography"]["unit_type"] == "WD")
        & (tables["geography"]["state_name"] == "Andhra Pradesh")
    ]["unit_name"]
    for wd_name in ap_wds:
        assert wd_name in text

    # Franchises don't vary by month, so checking this one document is
    # enough to confirm every franchise name is copied verbatim.
    franchises = tables["products"]["franchise_name"].unique()
    for franchise_name in franchises:
        assert franchise_name in text


def test_no_nan_leaks_into_document_text(built):
    """A missing/NaN KPI value rendered into an f-string shows up as the
    standalone word 'nan' -- this would signal a silent data gap. We
    match on a word boundary (not a plain substring check) because real
    entity names can innocently contain "nan" as letters, e.g. the
    Faker-generated distributor surname "Hernandez"."""
    nan_word = re.compile(r"\bnan\b", re.IGNORECASE)
    sample_files = list(built["output_dir"].glob("*.txt"))[:30]
    for path in sample_files:
        text = path.read_text(encoding="utf-8")
        assert not nan_word.search(text), f"{path.name} contains a NaN leak"


def test_build_is_reproducible(built, tables, kpi_state_month, kpi_state_month_category, tmp_path_factory):
    """Running the build twice on the same input tables must produce
    byte-identical documents and manifest -- no randomness in this
    module, only in the upstream data generation."""
    data_dir_2 = tmp_path_factory.mktemp("data2")
    output_dir_2 = data_dir_2 / "graphrag_input"

    tables["geography"].to_csv(data_dir_2 / "geography.csv", index=False)
    tables["products"].to_csv(data_dir_2 / "products.csv", index=False)
    kpi_state_month.to_csv(data_dir_2 / "kpi_state_month.csv", index=False)
    kpi_state_month_category.to_csv(
        data_dir_2 / "kpi_state_month_category.csv", index=False
    )

    manifest_2 = build_all_documents(data_dir_2, output_dir_2)
    pd.testing.assert_frame_equal(
        built["manifest"].reset_index(drop=True), manifest_2.reset_index(drop=True)
    )

    sample_filename = built["manifest"]["filename"].iloc[0]
    first_text = (built["output_dir"] / sample_filename).read_text(encoding="utf-8")
    second_text = (output_dir_2 / sample_filename).read_text(encoding="utf-8")
    assert first_text == second_text
