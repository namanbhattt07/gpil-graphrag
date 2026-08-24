"""
Tests for src/inference/fact_structuring.py: build_atomic_facts_block().

These use entirely fictional distributor names/metrics/categories/values
(never Baxter/Campbell/Gujarat/the live repro numbers) to prove the parser
works off the sentence GRAMMAR alone, not any reproduction-specific case.
"""

import pandas as pd

from src.inference.fact_structuring import (
    AtomicFact,
    build_atomic_facts_block,
    build_deterministic_ranking_block,
    extract_atomic_facts,
)


def _sources_df(rows: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    return {"sources": pd.DataFrame([{"id": rid, "text": text} for rid, text in rows])}


# ---------------------------------------------------------------------------
# Arbitrary fictional names / different metrics / different categories
# ---------------------------------------------------------------------------


def test_parses_category_grain_percentage_deviation_with_fictional_names():
    text = (
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in March 2031: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("101", text)]))

    assert "FACT 1" in block
    assert "Distributor: Quibble & Sons Distributors" in block
    assert "Metric: Out-of-Stock Rate" in block
    assert "Category: Zorn" in block
    assert "Period: March 2031" in block
    assert "Value: 33.3%" in block
    assert "Average: 9.1%" in block
    assert "Gap: 24.2 percentage points" in block
    assert "Source: Sources (101)" in block


def test_parses_different_metric_and_category_independently():
    text = (
        "Distributor Marlowe-Finch Distributors showed a significant deviation on "
        "ACV for the Halvorsen category in June 2029: 71.0% vs. the state average "
        "of 55.4%, a gap of 15.6 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("7", text)]))

    assert "Distributor: Marlowe-Finch Distributors" in block
    assert "Metric: ACV" in block
    assert "Category: Halvorsen" in block


# ---------------------------------------------------------------------------
# No category (state-grain % and Dropsize templates)
# ---------------------------------------------------------------------------


def test_parses_state_grain_percentage_deviation_with_no_category():
    text = (
        "Distributor Freedonia Distributors showed a significant deviation on "
        "Service Level in January 2028: 88.4% vs. the state average of 94.0%, "
        "a gap of 5.6 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("5", text)]))

    assert "Distributor: Freedonia Distributors" in block
    assert "Metric: Service Level" in block
    assert "Category: (none)" in block
    assert "Value: 88.4%" in block


def test_parses_dropsize_deviation_with_no_percent_sign_and_no_category():
    text = (
        "Distributor Halvorsen LLC Distributors showed a significant deviation on "
        "Dropsize in July 2030: 210.11 vs. the state average of 188.40, a gap of "
        "11.5 percent."
    )
    block = build_atomic_facts_block(_sources_df([("9", text)]))

    assert "Distributor: Halvorsen LLC Distributors" in block
    assert "Metric: Dropsize" in block
    assert "Category: (none)" in block
    assert "Value: 210.11" in block
    assert "Value: 210.11%" not in block
    assert "Average: 188.40" in block
    assert "Gap: 11.5 percent" in block


# ---------------------------------------------------------------------------
# Repeated identical numeric values
# ---------------------------------------------------------------------------


def test_repeated_identical_values_still_produce_separate_facts_per_distributor():
    text = (
        "Distributor Aldridge Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Novak category in May 2027: 20.0% vs. the "
        "state average of 8.0%, a gap of 12.0 percentage points.\n"
        "Distributor Blackwood Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Novak category in May 2027: 20.0% vs. the "
        "state average of 8.0%, a gap of 12.0 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("42", text)]))

    assert block.count("FACT ") == 2
    assert "Distributor: Aldridge Distributors" in block
    assert "Distributor: Blackwood Distributors" in block
    # Both facts keep their OWN distributor bound to the shared value/gap --
    # this is exactly the atomic-binding property the whole layer exists for.
    fact_1, fact_2 = block.split("FACT 2")
    assert "Aldridge" in fact_1
    assert "Value: 20.0%" in fact_1
    assert "Blackwood" in fact_2
    assert "Value: 20.0%" in fact_2


# ---------------------------------------------------------------------------
# Multiple adjacent deviation sentences in one source record
# ---------------------------------------------------------------------------


def test_multiple_adjacent_sentences_all_parsed_in_order():
    text = (
        "Distributor-level deviations, April 2026:\n"
        "Distributor Ashworth Distributors showed a significant deviation on "
        "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap "
        "of 10.6 percent.\n"
        "Distributor Whitfield Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Renner category in April 2026: 20.0% vs. the "
        "state average of 8.0%, a gap of 12.0 percentage points.\n"
        "Distributor Osei Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Sorrel category in April 2026: 20.0% vs. the "
        "state average of 8.0%, a gap of 12.0 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("22", text)]))

    assert block.count("FACT ") == 3
    assert block.index("Ashworth") < block.index("Whitfield") < block.index("Osei")
    assert "Category: (none)" in block  # Ashworth's Dropsize fact
    assert "Category: Renner" in block
    assert "Category: Sorrel" in block


# ---------------------------------------------------------------------------
# Multiple source ids
# ---------------------------------------------------------------------------


def test_facts_from_different_source_rows_keep_their_own_source_id():
    text_a = (
        "Distributor Nakamura Distributors showed a significant deviation on "
        "Dropsize in February 2029: 150.00 vs. the state average of 140.00, a "
        "gap of 7.1 percent."
    )
    text_b = (
        "Distributor Petrov Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Ives category in February 2029: 25.0% vs. "
        "the state average of 10.0%, a gap of 15.0 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("11", text_a), ("12", text_b)]))

    assert block.count("FACT ") == 2
    fact_1, fact_2 = block.split("FACT 2")
    assert "Nakamura" in fact_1 and "Source: Sources (11)" in fact_1
    assert "Petrov" in fact_2 and "Source: Sources (12)" in fact_2


# ---------------------------------------------------------------------------
# Malformed / non-matching text must not crash
# ---------------------------------------------------------------------------


def test_non_matching_text_returns_empty_string_without_crashing():
    text = "This document has no deviation sentences of any kind in it at all."
    assert build_atomic_facts_block(_sources_df([("1", text)])) == ""


def test_empty_and_missing_sources_table_do_not_crash():
    assert build_atomic_facts_block({}) == ""
    assert build_atomic_facts_block({"sources": pd.DataFrame(columns=["id", "text"])}) == ""
    assert build_atomic_facts_block({"sources": pd.DataFrame()}) == ""


def test_none_and_non_string_text_values_are_skipped_without_crashing():
    df = pd.DataFrame([{"id": "1", "text": None}, {"id": "2", "text": 12345}])
    assert build_atomic_facts_block({"sources": df}) == ""


def test_truncated_deviation_sentence_does_not_match_and_does_not_crash():
    text = "Distributor Ferreira Distributors showed a significant deviation on Dropsize in"
    assert build_atomic_facts_block(_sources_df([("1", text)])) == ""


def test_source_table_missing_required_columns_returns_empty_string():
    df = pd.DataFrame([{"id": "1"}])  # no "text" column
    assert build_atomic_facts_block({"sources": df}) == ""


# ---------------------------------------------------------------------------
# No matched facts -> caller preserves existing context unchanged (this
# module only proves the empty-string contract; answer.py's own tests prove
# the append-nothing behavior at the call site)
# ---------------------------------------------------------------------------


def test_no_matches_across_multiple_rows_returns_empty_string():
    df_rows = [
        ("1", "Just a narrative paragraph about Gujarat's zones."),
        ("2", "Productivity for Gujarat in April 2026 was 74.0%."),
    ]
    assert build_atomic_facts_block(_sources_df(df_rows)) == ""


# ---------------------------------------------------------------------------
# State-level KPI atomic facts (additive extension) -- entirely fictional
# state/period/values, mirroring the fictional-names convention above.
# ---------------------------------------------------------------------------


def _state_doc(state: str, period: str, metrics: dict) -> str:
    """One document's text in the real build_documents.py header+sentence
    shape -- 'State: X\\nPeriod: Y\\n\\n<Metric> for X in Y was <value>.'
    repeated per metric, exactly the grammar _state_metric_facts_from_
    source_text() targets."""
    sentences = " ".join(f"{name} for {state} in {period} was {value}." for name, value in metrics.items())
    return f"State: {state}\nPeriod: {period}\n\n{sentences}\n"


def test_state_metric_sentence_produces_atomic_state_kpi_fact():
    text = _state_doc("Ruritania", "May 2027", {"Productivity": "77.4%"})
    block = build_atomic_facts_block(_sources_df([("1", text)]))

    assert "-----Atomic State KPI Facts-----" in block
    assert "FACT 1" in block
    assert "State: Ruritania" in block
    assert "Metric: Productivity" in block
    assert "Period: May 2027" in block
    assert "Value: 77.4%" in block
    assert "Source: Sources (1)" in block


def test_all_six_state_metrics_parsed_independently():
    text = _state_doc(
        "Ruritania", "May 2027",
        {
            "Productivity": "77.4%",
            "Service Level": "88.1%",
            "SKUs per Transaction": "3.10",
            "Dropsize": "160.00",
            "Inventory Turns": "2.50 times",
            "Inventory Days": "12.0 days",
        },
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    assert block.count("-----Atomic State KPI Facts-----") == 1
    for metric, value in [
        ("Productivity", "77.4%"), ("Service Level", "88.1%"),
        ("SKUs per Transaction", "3.10"), ("Dropsize", "160.00"),
        ("Inventory Turns", "2.50"), ("Inventory Days", "12.0"),
    ]:
        assert f"Metric: {metric}" in block
        assert f"Value: {value}" in block


def test_dropsize_parenthetical_definition_aside_does_not_break_parsing():
    text = (
        "State: Ruritania\nPeriod: May 2027\n\n"
        "Dropsize (units ordered per productive visit — definition pending GPIL confirmation) "
        "for Ruritania in May 2027 was 160.00."
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    assert "Metric: Dropsize" in block
    assert "Value: 160.00" in block


def test_category_grain_sentence_is_not_mistaken_for_state_level_fact():
    """The category-block grammar ('For the X category ... during <period>:
    Numeric Distribution was ...') must never match the state-level
    '<Metric> for <State> in <Period> was <value>' pattern, even though
    both mention a state, a period, and 'was <value>'."""
    text = (
        "State: Ruritania\nPeriod: May 2027\n\n"
        "For the Zorn category (franchises: Foo) in Ruritania during May 2027: "
        "Numeric Distribution was 51.5%, ACV was 51.3%."
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    assert block == ""


def test_state_kpi_facts_do_not_interleave_or_renumber_distributor_facts():
    """Both extractions running over the SAME source row must each keep
    their own independent FACT numbering and their own section header --
    proves the distributor-facts section's existing output is byte-for-
    byte unaffected by this extension."""
    text = (
        "State: Ruritania\nPeriod: May 2027\n\n"
        "Productivity for Ruritania in May 2027 was 77.4%. "
        "Service Level for Ruritania in May 2027 was 88.1%.\n\n"
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in May 2027: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points."
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))

    assert "-----Atomic Distributor Facts-----" in block
    assert "-----Atomic State KPI Facts-----" in block
    distributor_section, state_section = block.split("-----Atomic State KPI Facts-----")
    assert "FACT 1" in distributor_section and distributor_section.count("FACT ") == 1
    assert "Distributor: Quibble & Sons Distributors" in distributor_section
    # State section has its OWN independent numbering, starting at 1 again.
    assert state_section.count("FACT ") == 2
    assert "State: Ruritania" in state_section
    assert "Metric: Productivity" in state_section
    assert "Metric: Service Level" in state_section


def test_state_facts_present_even_when_no_distributor_deviations_exist():
    """A source row with ONLY state-level KPI sentences (no distributor
    deviations at all) must still produce a state-facts block -- this is
    new behavior (previously build_atomic_facts_block() returned "" here),
    an intentional, additive extension, not a regression."""
    text = _state_doc("Ruritania", "May 2027", {"Productivity": "77.4%"})
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    assert "-----Atomic Distributor Facts-----" not in block
    assert "-----Atomic State KPI Facts-----" in block


def test_state_kpi_fact_missing_header_returns_no_state_facts():
    """A bare metric sentence with no 'State: X\\nPeriod: Y' header (e.g. a
    stray fragment) must not produce a state-KPI fact -- the header is
    what pins down which literal state/period string the sentence must
    name, exactly like premise_check.py's own use of the same header."""
    text = "Productivity for Ruritania in May 2027 was 77.4%."  # no header
    assert build_atomic_facts_block(_sources_df([("1", text)])) == ""


def test_state_facts_from_different_rows_keep_their_own_source_id():
    text_a = _state_doc("Ruritania", "May 2027", {"Productivity": "77.4%"})
    text_b = _state_doc("Freedonia", "June 2028", {"Service Level": "90.0%"})
    block = build_atomic_facts_block(_sources_df([("11", text_a), ("22", text_b)]))

    assert "State: Ruritania" in block and "Source: Sources (11)" in block
    assert "State: Freedonia" in block and "Source: Sources (22)" in block


# ---------------------------------------------------------------------------
# Category-level KPI atomic facts (Q4 fix) -- entirely fictional
# state/category/period/values, mirroring the fictional-names convention
# used throughout this file. Real corpus category names (GPI/IPM/Ferrero/
# Candy) are also exercised separately below since the extractor matches
# category NAME generically (any letters), not a hardcoded list.
# ---------------------------------------------------------------------------


def _category_block(category: str, state: str, period: str, values: dict, acv_qualifier: str, rb_qualifier: str) -> str:
    """One category-KPI paragraph in the real build_documents.py grammar --
    caller supplies the ACV/Range-Billing parenthetical qualifier text so
    both old- and new-corpus wording can be exercised with the same
    helper."""
    return (
        f"For the {category} category (franchises: {values['franchises']}) in "
        f"{state} during {period}: Numeric Distribution was {values['nd']}, "
        f"ACV ({acv_qualifier}) was {values['acv']}, "
        f"Out-of-Stock rate was {values['oos']}, and "
        f"average Range Billing ({rb_qualifier}) was {values['rb']}."
    )


NEW_ACV_QUALIFIER = "weighted distribution — definition pending GPIL confirmation"
NEW_RB_QUALIFIER = (
    "share of the category's SKU range billed, among outlets that billed "
    "anything — definition pending GPIL confirmation"
)
OLD_ACV_QUALIFIER = "weighted distribution"
OLD_RB_QUALIFIER = "share of the category's SKU range billed, among outlets that billed anything"


# 1: one category paragraph -> exactly 4 atomic facts.
def test_one_category_paragraph_produces_exactly_four_atomic_facts():
    text = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "Zorn_Franchise_1", "nd": "51.5%", "acv": "51.3%", "oos": "8.8%", "rb": "4.8%"},
        NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))

    assert "-----Atomic Category KPI Facts-----" in block
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert category_section.count("FACT ") == 4
    assert "Category: Zorn" in category_section
    assert "State: Ruritania" in category_section
    assert "Period: May 2027" in category_section
    for metric, value in [
        ("Numeric Distribution", "51.5%"), ("ACV", "51.3%"),
        ("Out-of-Stock Rate", "8.8%"), ("Range Billing", "4.8%"),
    ]:
        assert f"Metric: {metric}" in category_section
        assert f"Value: {value}" in category_section


# 2: exact Q4-style headerless continuation -- category paragraph names its
# own state/period, no "State: X\nPeriod: Y" document header anywhere.
def test_headerless_continuation_chunk_still_parses_like_real_q4_source_23():
    text = (
        "Stock rate was 8.0%, and average Range Billing (share of the category's "
        "SKU range billed, among outlets that billed anything — definition pending "
        "GPIL confirmation) was 6.4%.\n\n"
        "For the Ferrero category (franchises: TicTac, Kinder_Joy) in Gujarat "
        "during April 2026: Numeric Distribution was 21.6%, ACV (weighted "
        "distribution — definition pending GPIL confirmation) was 21.6%, "
        "Out-of-Stock rate was 2.9%, and average Range Billing (share of the "
        "category's SKU range billed, among outlets that billed anything — "
        "definition pending GPIL confirmation) was 16.8%.\n"
    )
    assert "State:" not in text and "Period:" not in text  # genuinely headerless
    block = build_atomic_facts_block(_sources_df([("23", text)]))

    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert "State: Gujarat" in category_section
    assert "Category: Ferrero" in category_section
    assert "Period: April 2026" in category_section
    assert "Metric: Out-of-Stock Rate" in category_section
    assert "Value: 2.9%" in category_section
    assert "Source: Sources (23)" in category_section


# 3: all four real corpus categories parse independently in one row.
def test_all_four_real_categories_parse_independently():
    paragraphs = "\n\n".join(
        _category_block(
            cat, "Gujarat", "April 2026",
            {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
            NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
        )
        for cat in ["GPI", "IPM", "Ferrero", "Candy"]
    )
    block = build_atomic_facts_block(_sources_df([("1", paragraphs)]))
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert category_section.count("FACT ") == 16  # 4 categories x 4 metrics
    for cat in ["GPI", "IPM", "Ferrero", "Candy"]:
        assert f"Category: {cat}" in category_section


# 4: lowercase source "Out-of-Stock rate" canonicalizes to "Out-of-Stock Rate".
def test_lowercase_source_oos_rate_canonicalizes_in_output():
    text = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
        NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
    )
    assert "Out-of-Stock rate" in text  # confirms the input fixture uses lowercase, as the real template does
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert "Metric: Out-of-Stock Rate" in category_section
    assert "Metric: Out-of-Stock rate" not in category_section


# 5: old corpus qualifier wording parses.
def test_old_qualifier_wording_parses():
    text = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
        OLD_ACV_QUALIFIER, OLD_RB_QUALIFIER,
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert category_section.count("FACT ") == 4
    assert "Value: 50.0%" in category_section
    assert "Value: 5.0%" in category_section
    assert "Value: 10.0%" in category_section


# 6: new corpus qualifier wording (with the "-- definition pending GPIL
# confirmation" suffix) parses -- covered directly by test 1/2/3 above too,
# but isolated here per the task's explicit numbered requirement.
def test_new_qualifier_wording_parses():
    text = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
        NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
    )
    assert "definition pending GPIL confirmation" in text
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]
    assert category_section.count("FACT ") == 4


# 7: truncated/mid-sentence category paragraph -> zero category facts.
def test_truncated_category_paragraph_produces_no_category_facts():
    text = (
        "For the Ferrero category (franchises: TicTac, Kinder_Joy) in Gujarat "
        "during April 2026: Numeric Distribution was 21.6%, ACV (weighted "
        "distribution) was 21.6%, Out-of-Stock rate was 2.9%, and average Range "
        # cut off before "Billing (...) was <value>." -- mirrors the real
        # Source(22) truncation exactly (mid-paragraph, no closing period).
    )
    assert build_atomic_facts_block(_sources_df([("22", text)])) == ""


# 8: existing distributor atomic facts remain byte-for-byte unchanged when
# a category paragraph is ALSO present in the same row.
def test_distributor_facts_unaffected_by_category_facts_in_same_row():
    text = (
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in March 2031: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points.\n\n"
        + _category_block(
            "Zorn", "Ruritania", "March 2031",
            {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
            NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
        )
    )
    block = build_atomic_facts_block(_sources_df([("101", text)]))
    distributor_section = block.split("-----Atomic State KPI Facts-----")[0] if "-----Atomic State KPI Facts-----" in block else block.split("-----Atomic Category KPI Facts-----")[0]

    assert "-----Atomic Distributor Facts-----" in distributor_section
    assert distributor_section.count("FACT ") == 1
    assert "Distributor: Quibble & Sons Distributors" in distributor_section
    assert "Value: 33.3%" in distributor_section
    assert "Average: 9.1%" in distributor_section
    assert "Gap: 24.2 percentage points" in distributor_section


# 9: existing state atomic facts remain unchanged when a category
# paragraph is ALSO present in the same row.
def test_state_facts_unaffected_by_category_facts_in_same_row():
    text = (
        _state_doc("Ruritania", "May 2027", {"Productivity": "77.4%"})
        + "\n"
        + _category_block(
            "Zorn", "Ruritania", "May 2027",
            {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
            NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
        )
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    state_section = block.split("-----Atomic State KPI Facts-----")[1].split("-----Atomic Category KPI Facts-----")[0]

    assert state_section.count("FACT ") == 1
    assert "Metric: Productivity" in state_section
    assert "Value: 77.4%" in state_section


# 10: three atomic sections coexist without numbering collisions.
def test_three_atomic_sections_coexist_without_numbering_collisions():
    text = (
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in May 2027: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points.\n\n"
        + _state_doc("Ruritania", "May 2027", {"Productivity": "77.4%", "Service Level": "88.1%"})
        + "\n"
        + _category_block(
            "Zorn", "Ruritania", "May 2027",
            {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
            NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
        )
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))

    assert "-----Atomic Distributor Facts-----" in block
    assert "-----Atomic State KPI Facts-----" in block
    assert "-----Atomic Category KPI Facts-----" in block

    distributor_section = block.split("-----Atomic State KPI Facts-----")[0]
    state_section = block.split("-----Atomic State KPI Facts-----")[1].split("-----Atomic Category KPI Facts-----")[0]
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]

    # Each section starts its own numbering at 1, independent of the others.
    assert distributor_section.count("FACT ") == 1 and "FACT 1" in distributor_section
    assert state_section.count("FACT ") == 2 and "FACT 1" in state_section and "FACT 2" in state_section
    assert category_section.count("FACT ") == 4 and "FACT 1" in category_section and "FACT 4" in category_section


def test_category_facts_from_different_rows_keep_their_own_source_id():
    text_a = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
        NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
    )
    text_b = _category_block(
        "Halvorsen", "Freedonia", "June 2028",
        {"franchises": "Y", "nd": "60.0%", "acv": "60.0%", "oos": "6.0%", "rb": "12.0%"},
        OLD_ACV_QUALIFIER, OLD_RB_QUALIFIER,
    )
    block = build_atomic_facts_block(_sources_df([("11", text_a), ("22", text_b)]))
    category_section = block.split("-----Atomic Category KPI Facts-----")[1]

    assert "Category: Zorn" in category_section and "Source: Sources (11)" in category_section
    assert "Category: Halvorsen" in category_section and "Source: Sources (22)" in category_section


def test_category_block_grammar_not_mistaken_for_state_or_distributor_facts():
    """A category paragraph alone (no distributor deviations, no state
    header/headline sentences) must produce ONLY the category section --
    proves the three sections' grammars stay properly disjoint."""
    text = _category_block(
        "Zorn", "Ruritania", "May 2027",
        {"franchises": "X", "nd": "50.0%", "acv": "50.0%", "oos": "5.0%", "rb": "10.0%"},
        NEW_ACV_QUALIFIER, NEW_RB_QUALIFIER,
    )
    block = build_atomic_facts_block(_sources_df([("1", text)]))
    assert "-----Atomic Distributor Facts-----" not in block
    assert "-----Atomic State KPI Facts-----" not in block
    assert "-----Atomic Category KPI Facts-----" in block


# ---------------------------------------------------------------------------
# extract_atomic_facts() -- structured, additive parallel to
# build_atomic_facts_block()'s text rendering. Reuses the SAME regexes;
# these tests confirm the structured output matches what the rendered text
# already says, and that build_atomic_facts_block()'s own text output is
# byte-for-byte UNCHANGED by this addition existing (a hard requirement --
# the LLM prompt must not change).
# ---------------------------------------------------------------------------


def test_extract_atomic_facts_distributor_fact_matches_rendered_text():
    text = (
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in March 2031: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points."
    )
    records = _sources_df([("101", text)])
    facts = extract_atomic_facts(records)
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "distributor"
    assert f.entity == "Quibble & Sons Distributors"
    assert f.metric == "Out-of-Stock Rate"
    assert f.category == "Zorn"
    assert f.period == "March 2031"
    assert f.value == 33.3
    assert f.average == 9.1
    assert f.gap == "24.2 percentage points"
    assert f.source_id == "101"
    assert f.numeric_fields() == [33.3, 9.1, 24.2]

    # build_atomic_facts_block()'s own rendered text is untouched by this.
    block = build_atomic_facts_block(records)
    assert "Distributor: Quibble & Sons Distributors" in block
    assert "Value: 33.3%" in block


def test_extract_atomic_facts_no_category_distributor_fact():
    text = (
        "Distributor Freedonia Distributors showed a significant deviation on "
        "Service Level in January 2028: 88.4% vs. the state average of 94.0%, "
        "a gap of 5.6 percentage points."
    )
    facts = extract_atomic_facts(_sources_df([("5", text)]))
    assert len(facts) == 1
    assert facts[0].category is None
    assert facts[0].entity == "Freedonia Distributors"
    assert facts[0].value == 88.4


def test_extract_atomic_facts_state_kpi_fact():
    text = "State: Ruritania\nPeriod: May 2027\n\nProductivity for Ruritania in May 2027 was 74.0%."
    facts = extract_atomic_facts(_sources_df([("9", text)]))
    state_facts = [f for f in facts if f.kind == "state"]
    assert len(state_facts) == 1
    f = state_facts[0]
    assert f.state == "Ruritania"
    assert f.metric == "Productivity"
    assert f.period == "May 2027"
    assert f.value == 74.0
    assert f.entity is None


def test_extract_atomic_facts_category_kpi_fact():
    text = (
        "For the Zorn category (franchises: X, Y) in Ruritania during May 2027: "
        "Numeric Distribution was 66.7%, ACV (weighted distribution) was 66.8%, "
        "Out-of-Stock rate was 8.0%, and average Range Billing (share billed) was 6.4%."
    )
    facts = extract_atomic_facts(_sources_df([("10", text)]))
    category_facts = [f for f in facts if f.kind == "category"]
    assert len(category_facts) == 4
    by_metric = {f.metric: f.value for f in category_facts}
    assert by_metric == {
        "Numeric Distribution": 66.7,
        "ACV": 66.8,
        "Out-of-Stock Rate": 8.0,
        "Range Billing": 6.4,
    }
    assert all(f.state == "Ruritania" and f.category == "Zorn" and f.period == "May 2027" for f in category_facts)


def test_extract_atomic_facts_two_distributors_stay_independently_bound():
    """The exact real Gujarat/April 2026 shape: Campbell's real GPI 20.0%
    fact and Baxter's real Dropsize 214.48 fact, from the SAME source row,
    must extract as two SEPARATE facts with no cross-contamination -- the
    same guarantee build_atomic_facts_block()'s own text rendering already
    provides, now available as structured data."""
    text = (
        "State: Gujarat\nPeriod: April 2026\n\n"
        "Distributor Baxter, Thomas and Williams Distributors showed a significant deviation on "
        "Dropsize in April 2026: 214.48 vs. the state average of 193.87, a gap of 10.6 percent.\n"
        "Distributor Campbell PLC Distributors showed a significant deviation on Out-of-Stock Rate "
        "for the GPI category in April 2026: 20.0% vs. the state average of 8.0%, a gap of 12.0 "
        "percentage points."
    )
    facts = extract_atomic_facts(_sources_df([("22", text)]))
    distributor_facts = [f for f in facts if f.kind == "distributor"]
    by_entity = {f.entity: f for f in distributor_facts}
    assert by_entity["Baxter, Thomas and Williams Distributors"].value == 214.48
    assert by_entity["Baxter, Thomas and Williams Distributors"].metric == "Dropsize"
    assert by_entity["Campbell PLC Distributors"].value == 20.0
    assert by_entity["Campbell PLC Distributors"].metric == "Out-of-Stock Rate"
    assert by_entity["Campbell PLC Distributors"].category == "GPI"


def test_extract_atomic_facts_empty_when_no_sources():
    assert extract_atomic_facts({}) == []
    assert extract_atomic_facts({"sources": pd.DataFrame(columns=["id", "text"])}) == []


def test_extract_atomic_facts_never_raises_on_malformed_rows():
    records = {"sources": pd.DataFrame([{"id": "1", "text": None}, {"id": "2", "text": "no facts here"}])}
    assert extract_atomic_facts(records) == []


# ---------------------------------------------------------------------------
# SKU-level KPI atomic facts (Problem 1) -- fictional state/SKU names, same
# discipline as the sections above: proves the parser works off the fixed
# GRAMMAR sku_evidence.render_sku_evidence_document() writes, not any
# specific reproduction case.
# ---------------------------------------------------------------------------


def _sku_fact_text() -> str:
    return (
        "SKU-level Sales & Distribution detail for Ruritania, May 2027:\n"
        "SKU SKU0099 (Zorn Widget Pack 1, Zorn Widgets franchise, Zorn category) in "
        "Ruritania during May 2027: 12,000 units delivered, revenue of Rs 500,000, "
        "Service Level 91.5%, Numeric Distribution 62.0%, Out-of-Stock rate 3.2%.\n"
        "SKU SKU0100 (Zorn Widget Pack 2, Zorn Widgets franchise, Zorn category) in "
        "Ruritania during May 2027: 800 units delivered, revenue of Rs 40,000, "
        "Service Level 70.0%, Numeric Distribution 15.0%, Out-of-Stock rate 22.5%.\n"
    )


def test_build_atomic_facts_block_renders_sku_section():
    block = build_atomic_facts_block(_sources_df([("40", _sku_fact_text())]))
    assert "-----Atomic SKU Facts-----" in block
    assert "SKU: SKU0099" in block
    assert "SKU Name: Zorn Widget Pack 1" in block
    assert "Franchise: Zorn Widgets" in block
    assert "Category: Zorn" in block
    assert "State: Ruritania" in block
    assert "Period: May 2027" in block
    assert "Source: Sources (40)" in block
    # Five metrics per SKU sentence -> 5 facts per SKU, numbered 1..10.
    assert "FACT 10" in block


def test_extract_atomic_facts_sku_kind_has_five_metrics_per_sku():
    facts = extract_atomic_facts(_sources_df([("41", _sku_fact_text())]))
    sku_facts = [f for f in facts if f.kind == "sku"]
    assert len(sku_facts) == 10  # 2 SKUs x 5 metrics each

    by_sku = {}
    for f in sku_facts:
        by_sku.setdefault(f.entity, {})[f.metric] = f.value

    assert by_sku["Zorn Widget Pack 1"] == {
        "Units Delivered": 12000.0,
        "Revenue": 500000.0,
        "Service Level": 91.5,
        "Numeric Distribution": 62.0,
        "Out-of-Stock Rate": 3.2,
    }
    assert by_sku["Zorn Widget Pack 2"]["Units Delivered"] == 800.0
    assert all(f.state == "Ruritania" and f.category == "Zorn" and f.period == "May 2027" for f in sku_facts)


# ---------------------------------------------------------------------------
# Deterministic SKU Ranking block (2026-08-23 stabilization pass, Design
# Decision: the LLM must never compute a SKU ranking itself) -- proves the
# "sku-ranking-"-id-prefixed row is rendered under its own labeled section
# and never double-counted as an ordinary, unordered Atomic SKU Fact.
# ---------------------------------------------------------------------------


def test_build_deterministic_ranking_block_renders_ranking_rows():
    text = (
        "This ranking was computed deterministically from the full underlying dataset for "
        "Ruritania, May 2027, and is authoritative -- restate it exactly as given below. "
        "Do not recompute values, do not add or remove SKUs, and do not change the order.\n"
        "Ranked by Revenue (highest first):\n"
        "Rank 1: SKU SKU0099 (Zorn Widget Pack 1, Zorn Widgets franchise, Zorn category) -- "
        "Revenue: Rs 500,000. Source: Sources (sku-Ruritania-May_2027)\n"
    )
    block = build_deterministic_ranking_block(_sources_df([("sku-ranking-Ruritania-May_2027", text)]))
    assert "-----Deterministic SKU Ranking (Authoritative -- do not recompute)-----" in block
    assert "Rank 1: SKU SKU0099" in block
    assert "do not recompute" in block.lower()


def test_build_deterministic_ranking_block_empty_without_ranking_rows():
    block = build_deterministic_ranking_block(_sources_df([("sku-Ruritania-May_2027", _sku_fact_text())]))
    assert block == ""


def test_ranking_row_never_double_counted_as_atomic_sku_fact():
    """A "sku-ranking-"-prefixed row's text is deliberately NOT in the
    _SKU_FACT_RE sentence shape -- build_atomic_facts_block()'s SKU
    section must never pick it up, or the model would see the same SKU
    twice (once ranked, once as an ordinary unordered fact) with no way to
    tell the two apart."""
    ranking_text = (
        "Rank 1: SKU SKU0099 (Zorn Widget Pack 1, Zorn Widgets franchise, Zorn category) -- "
        "Revenue: Rs 500,000. Source: Sources (sku-Ruritania-May_2027)\n"
    )
    facts = extract_atomic_facts(_sources_df([("sku-ranking-Ruritania-May_2027", ranking_text)]))
    assert facts == []


def test_ranking_block_coexists_with_full_atomic_sku_facts_section():
    records = _sources_df(
        [
            ("sku-Ruritania-May_2027", _sku_fact_text()),
            (
                "sku-ranking-Ruritania-May_2027",
                "This ranking was computed deterministically... Ranked by Revenue (highest first):\n"
                "Rank 1: SKU SKU0099 (Zorn Widget Pack 1, Zorn Widgets franchise, Zorn category) -- "
                "Revenue: Rs 500,000. Source: Sources (sku-Ruritania-May_2027)\n",
            ),
        ]
    )
    atomic_block = build_atomic_facts_block(records)
    ranking_block = build_deterministic_ranking_block(records)
    assert "-----Atomic SKU Facts-----" in atomic_block
    assert "FACT 10" in atomic_block  # unchanged: still 2 SKUs x 5 metrics from the real evidence row
    assert "-----Deterministic SKU Ranking" in ranking_block


def test_sku_facts_do_not_interfere_with_distributor_or_category_sections():
    """A source row combining a distributor-deviation sentence and a SKU
    sentence must extract BOTH independently -- no cross-contamination
    between sections (mirrors the existing "two distributors stay
    independently bound" guarantee, extended to the new SKU kind)."""
    text = (
        "Distributor Quibble & Sons Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Zorn category in May 2027: 33.3% vs. the state "
        "average of 9.1%, a gap of 24.2 percentage points.\n"
    ) + _sku_fact_text()
    facts = extract_atomic_facts(_sources_df([("42", text)]))
    kinds = {f.kind for f in facts}
    assert kinds == {"distributor", "sku"}
    assert sum(1 for f in facts if f.kind == "distributor") == 1
    assert sum(1 for f in facts if f.kind == "sku") == 10
