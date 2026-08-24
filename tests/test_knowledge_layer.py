"""
Tests for the GPIL Knowledge Layer (src/inference/knowledge_layer.py):
deterministic business-term detection, the glossary itself, and the
synthetic-Sources-row rendering that lets pipeline.py merge a matched
term's definition into the answer-generation context as real, citable
evidence.

Pure unit tests -- no GraphRAG, no index, no API calls.
"""

import pandas as pd
import pytest

from src.inference.knowledge_layer import (
    GPIL_GLOSSARY,
    build_glossary_block,
    build_glossary_reminder_block,
    build_glossary_source_rows,
    detect_glossary_terms,
)


# ---------------------------------------------------------------------------
# Detection: each required term resolves to the right GPIL-specific entry
# ---------------------------------------------------------------------------


def test_gpi_detected_as_godfrey_phillips_india_context():
    entries = detect_glossary_terms("What is GPI?")
    assert [e.term_id for e in entries] == ["gpi"]
    assert "Godfrey Phillips India" in entries[0].definition


def test_gpil_detected_as_same_company_entry_as_gpi():
    entries = detect_glossary_terms("Tell me about GPIL as a company.")
    assert [e.term_id for e in entries] == ["gpi"]
    assert "Godfrey Phillips India Ltd." in entries[0].definition


def test_ipm_detected_as_marlboro_context():
    entries = detect_glossary_terms("What is IPM?")
    assert [e.term_id for e in entries] == ["ipm"]
    assert "Marlboro" in entries[0].definition


def test_ferrero_detected_as_gpil_product_category():
    entries = detect_glossary_terms("What is Ferrero?")
    assert [e.term_id for e in entries] == ["ferrero"]
    assert "GPIL" in entries[0].definition
    assert "confectionery" in entries[0].definition.lower()


def test_candy_detected_as_gpil_product_category():
    entries = detect_glossary_terms("What is Candy?")
    assert [e.term_id for e in entries] == ["candy"]
    assert "GPIL" in entries[0].definition


def test_distributor_and_wd_resolve_to_the_same_entry():
    for question in ("What is a Distributor?", "What does WD mean?", "What is a Wholesale Distributor?"):
        entries = detect_glossary_terms(question)
        assert [e.term_id for e in entries] == ["distributor"], question


def test_dealer_is_a_distinct_entry_from_distributor():
    entries = detect_glossary_terms("What is a Dealer?")
    assert [e.term_id for e in entries] == ["dealer"]
    # The Dealer entry must itself explain the Dealer/Distributor distinction.
    assert "distributor" in entries[0].definition.lower()


def test_hero_sku_detected():
    entries = detect_glossary_terms("What is a Hero SKU?")
    assert [e.term_id for e in entries] == ["hero_sku"]


@pytest.mark.parametrize(
    "term_id,question",
    [
        ("acv", "What is ACV?"),
        ("numeric_distribution", "What is Numeric Distribution?"),
        ("range_billing", "What is Range Billing?"),
        ("out_of_stock_rate", "What is the Out-of-Stock Rate?"),
        ("service_level", "What is Service Level?"),
        ("dropsize", "What is Dropsize?"),
        ("productivity", "What is Productivity?"),
    ],
)
def test_established_kpi_terms_detected(term_id, question):
    entries = detect_glossary_terms(question)
    assert [e.term_id for e in entries] == [term_id]


def test_oos_percent_alias_matches_with_percent_sign():
    entries = detect_glossary_terms("Compare OOS% for GPI vs IPM in Gujarat.")
    assert {e.term_id for e in entries} == {"gpi", "ipm", "out_of_stock_rate"}


# ---------------------------------------------------------------------------
# The specific live bug this layer fixes: GPI/IPM must never become the
# generic industry expansions.
# ---------------------------------------------------------------------------


def test_gpi_entry_explicitly_rules_out_general_product_inventory():
    entries = detect_glossary_terms("What is GPI?")
    assert "general product inventory" in entries[0].definition.lower()


def test_ipm_entry_explicitly_rules_out_integrated_pest_management():
    entries = detect_glossary_terms("What is IPM?")
    assert "integrated pest management" in entries[0].definition.lower()


# ---------------------------------------------------------------------------
# Unrelated terms must never be force-matched into the Knowledge Layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What is a CEO?",
        "What is an ATM machine?",
        "Explain NATO to me.",
        "What is GDP growth in India?",
        "What does an SLA mean?",
        "Tell me about the CFO's role.",
    ],
)
def test_unrelated_acronyms_do_not_match_the_glossary(question):
    assert detect_glossary_terms(question) == []


def test_word_boundary_prevents_substring_match_inside_a_longer_word():
    # "GPIO" (a hardware term) must not be read as containing "GPI".
    assert detect_glossary_terms("How does a Raspberry Pi GPIO pin work?") == []
    # "candystore" as one token must not be read as containing "Candy".
    assert detect_glossary_terms("Is there a candystore chain in this dataset?") == []


def test_case_insensitive_matching():
    entries = detect_glossary_terms("what is gpi and how does it compare to ipm?")
    assert {e.term_id for e in entries} == {"gpi", "ipm"}


# ---------------------------------------------------------------------------
# Multi-term questions ("What is GPI and IPM?")
# ---------------------------------------------------------------------------


def test_multiple_terms_named_in_one_question_all_detected():
    entries = detect_glossary_terms("What are GPI and IPM?")
    assert [e.term_id for e in entries] == ["gpi", "ipm"]


def test_multiple_terms_render_in_glossarys_own_fixed_order_not_question_order():
    # IPM named first in the question text, but GPI still renders first
    # (GPIL_GLOSSARY's own declared order), matching every entry's fixed
    # numbering convention elsewhere in this project.
    entries = detect_glossary_terms("What are IPM and GPI?")
    assert [e.term_id for e in entries] == ["gpi", "ipm"]


def test_mixed_definition_and_analytical_question_still_detects_the_term():
    entries = detect_glossary_terms("What is GPI and how did it perform in Gujarat in April 2026?")
    assert [e.term_id for e in entries] == ["gpi"]


def test_ordinary_analytical_question_with_no_glossary_term_detects_nothing():
    assert detect_glossary_terms("What was the value in Bihar in November 2025?") == []


# ---------------------------------------------------------------------------
# build_glossary_source_rows() -- the synthetic-Sources-row shape
# pipeline.py merges into context_records["sources"]
# ---------------------------------------------------------------------------


def test_build_glossary_source_rows_shape_and_id_prefix():
    rows = build_glossary_source_rows("What is GPI?")
    assert len(rows) == 1
    assert rows[0]["id"] == "glossary-gpi"
    assert set(rows[0].keys()) == {"id", "text"}
    assert "Term: GPI / GPIL" in rows[0]["text"]
    assert "Source: Sources (glossary-gpi)" in rows[0]["text"]


def test_build_glossary_source_rows_empty_for_ordinary_question():
    assert build_glossary_source_rows("What was the value in Bihar in November 2025?") == []


def test_build_glossary_source_rows_one_row_per_matched_term():
    rows = build_glossary_source_rows("What are GPI and IPM?")
    assert [r["id"] for r in rows] == ["glossary-gpi", "glossary-ipm"]


# ---------------------------------------------------------------------------
# build_glossary_block() -- the rendered prompt-text section, reading
# already-merged context_records["sources"] (mirrors
# fact_structuring.build_deterministic_ranking_block()'s own contract)
# ---------------------------------------------------------------------------


def test_build_glossary_block_renders_header_and_matched_terms():
    rows = build_glossary_source_rows("What are GPI and IPM?")
    context_records = {"sources": pd.DataFrame(rows)}
    block = build_glossary_block(context_records)
    assert block.startswith("-----GPIL Knowledge Layer-----")
    assert "Term: GPI / GPIL" in block
    assert "Term: IPM" in block


def test_build_glossary_block_empty_when_no_glossary_rows_present():
    context_records = {"sources": pd.DataFrame([{"id": "21", "text": "Productivity for Bihar in November 2025 was 89.0%."}])}
    assert build_glossary_block(context_records) == ""


def test_build_glossary_block_empty_when_sources_missing_or_empty():
    assert build_glossary_block({}) == ""
    assert build_glossary_block({"sources": pd.DataFrame(columns=["id", "text"])}) == ""


def test_build_glossary_block_ignores_non_glossary_rows_alongside_glossary_ones():
    rows = build_glossary_source_rows("What is GPI?")
    rows.append({"id": "21", "text": "Productivity for Bihar in November 2025 was 89.0%."})
    context_records = {"sources": pd.DataFrame(rows)}
    block = build_glossary_block(context_records)
    assert "Term: GPI / GPIL" in block
    assert "Productivity for Bihar" not in block


# ---------------------------------------------------------------------------
# build_glossary_reminder_block() -- 2026-08-23 "What is IPM?" single-term
# investigation fix: the same matched definitions rendered a SECOND time
# under a distinct header, for bookending GraphRAG's own retrieved text
# (see answer.py's _context_data_with_facts()).
# ---------------------------------------------------------------------------


def test_build_glossary_reminder_block_renders_the_same_matched_terms():
    rows = build_glossary_source_rows("What are GPI and IPM?")
    context_records = {"sources": pd.DataFrame(rows)}
    reminder = build_glossary_reminder_block(context_records)
    assert reminder.startswith("-----GPIL Knowledge Layer (Reminder")
    assert "Term: GPI / GPIL" in reminder
    assert "Term: IPM" in reminder


def test_build_glossary_reminder_block_empty_when_no_glossary_rows_present():
    context_records = {"sources": pd.DataFrame([{"id": "21", "text": "Productivity for Bihar in November 2025 was 89.0%."}])}
    assert build_glossary_reminder_block(context_records) == ""


def test_build_glossary_reminder_block_has_a_distinct_header_from_the_main_block():
    rows = build_glossary_source_rows("What is IPM?")
    context_records = {"sources": pd.DataFrame(rows)}
    main_block = build_glossary_block(context_records)
    reminder_block = build_glossary_reminder_block(context_records)
    assert main_block.split("\n\n")[0] != reminder_block.split("\n\n")[0]


def test_glossary_block_preface_warns_length_and_repetition_are_not_correctness():
    rows = build_glossary_source_rows("What is IPM?")
    context_records = {"sources": pd.DataFrame(rows)}
    block = build_glossary_block(context_records)
    assert "not evidence of correctness" in block.lower()
