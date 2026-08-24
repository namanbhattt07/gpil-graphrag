"""
Tests for src/graph/build_sku_documents.py (Problem 1 -- SKU-level
narrative document builder).

Uses a small, hand-built kpi_state_month_sku.csv fixture (not the real
data/kpi_state_month_sku.csv, and not a full generate_all() run) so this
suite is fast and independent of whatever the live dataset contains.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.graph.build_sku_documents import _safe_filename, build_all_sku_documents

FIXTURE_ROWS = [
    dict(state_name="Andhra Pradesh", month="2024-08", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=1000, qty_delivered=900,
         revenue=90000.0, service_level=0.9, billed_outlets=50, eligible_outlets=100,
         numeric_distribution=0.5, oos_pct=0.05),
    dict(state_name="Andhra Pradesh", month="2024-08", sku_id="SKU0002", sku_name="Beta Pack 1",
         franchise_name="Beta", category_name="IPM", qty_ordered=200, qty_delivered=180,
         revenue=18000.0, service_level=0.9, billed_outlets=10, eligible_outlets=100,
         numeric_distribution=0.1, oos_pct=0.2),
    dict(state_name="Bihar", month="2024-08", sku_id="SKU0001", sku_name="Alpha Pack 1",
         franchise_name="Alpha", category_name="GPI", qty_ordered=500, qty_delivered=450,
         revenue=45000.0, service_level=0.9, billed_outlets=20, eligible_outlets=100,
         numeric_distribution=0.2, oos_pct=0.1),
]


@pytest.fixture
def built(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = data_dir / "graphrag_sku_input"
    pd.DataFrame(FIXTURE_ROWS).to_csv(data_dir / "kpi_state_month_sku.csv", index=False)

    manifest = build_all_sku_documents(data_dir, output_dir)
    return {"manifest": manifest, "data_dir": data_dir, "output_dir": output_dir}


def test_safe_filename_has_sku_suffix_and_replaces_spaces():
    assert _safe_filename("Andhra Pradesh", "2024-08") == "Andhra_Pradesh_2024-08_sku.txt"


def test_one_document_per_state_month_combination(built):
    """Two (state, month) groups in the fixture -> two documents, one per
    group, not one per row."""
    output_files = list(built["output_dir"].glob("*.txt"))
    assert len(output_files) == 2
    assert len(built["manifest"]) == 2


def test_manifest_matches_written_files(built):
    manifest_filenames = set(built["manifest"]["filename"])
    disk_filenames = {p.name for p in built["output_dir"].glob("*.txt")}
    assert manifest_filenames == disk_filenames


def test_document_contains_all_skus_for_that_state_month(built):
    path = built["output_dir"] / "Andhra_Pradesh_2024-08_sku.txt"
    text = path.read_text()
    assert "Alpha Pack 1" in text
    assert "Beta Pack 1" in text
    assert "in Andhra Pradesh during August 2024" in text


def test_document_does_not_leak_a_different_states_skus(built):
    """Bihar's document must never mention Andhra Pradesh's SKU rows, and
    vice versa -- cross-state isolation, same guarantee build_documents.py's
    own tests already require of the main corpus."""
    ap_text = (built["output_dir"] / "Andhra_Pradesh_2024-08_sku.txt").read_text()
    bihar_text = (built["output_dir"] / "Bihar_2024-08_sku.txt").read_text()
    assert "Bihar" not in ap_text
    assert "Andhra Pradesh" not in bihar_text


def test_manifest_records_state_and_month(built):
    row = built["manifest"][built["manifest"]["filename"] == "Bihar_2024-08_sku.txt"].iloc[0]
    assert row["state_name"] == "Bihar"
    assert row["month"] == "2024-08"
