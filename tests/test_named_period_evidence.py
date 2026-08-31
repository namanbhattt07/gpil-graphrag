"""
Tests for src/inference/named_period_evidence.py: deterministic
document-completion for state+period(s) a question explicitly names but
GraphRAG's own retrieval didn't return, scoped strictly to the pilot's own
already-indexed 62-document input directory.

Pure unit tests -- no GraphRAG, no index, no API calls. Uses a temp
directory standing in for data/graphrag_index/input_pilot/, so these
tests never depend on (or risk drifting against) the real pilot corpus.
"""

from src.inference.named_period_evidence import named_period_source_rows


def _write_doc(tmp_path, filename, state, period, extra=""):
    (tmp_path / filename).write_text(
        f"State: {state}\nPeriod: {period}\n\nService Level for {state} in {period} was 90.0%.\n{extra}",
        encoding="utf-8",
    )


def test_named_period_not_in_retrieved_sources_is_injected(tmp_path):
    _write_doc(tmp_path, "Gujarat_2026-05.txt", "Gujarat", "May 2026")
    rows = named_period_source_rows(
        "Gujarat's Service Level in May 2026 -- why?",
        existing_sources_text="",  # nothing retrieved at all
        index_input_dir=tmp_path,
    )
    assert len(rows) == 1
    assert "Gujarat" in rows[0]["text"]
    assert "May 2026" in rows[0]["text"]
    assert rows[0]["id"].startswith("named-period-doc-")


def test_already_retrieved_period_is_not_duplicated(tmp_path):
    _write_doc(tmp_path, "Gujarat_2026-06.txt", "Gujarat", "June 2026")
    already_retrieved = "State: Gujarat\nPeriod: June 2026\n\nService Level for Gujarat in June 2026 was 90.8%."
    rows = named_period_source_rows(
        "Gujarat's Service Level in June 2026 -- why?",
        existing_sources_text=already_retrieved,
        index_input_dir=tmp_path,
    )
    assert rows == []


def test_multi_period_question_fills_in_only_the_missing_periods(tmp_path):
    """Live bug regression: 'April to May 2026 then ... June 2026' only
    had June actually retrieved by GraphRAG -- April and May must both be
    injected, June must not be duplicated."""
    _write_doc(tmp_path, "Gujarat_2026-04.txt", "Gujarat", "April 2026")
    _write_doc(tmp_path, "Gujarat_2026-05.txt", "Gujarat", "May 2026")
    _write_doc(tmp_path, "Gujarat_2026-06.txt", "Gujarat", "June 2026")
    already_retrieved = "State: Gujarat\nPeriod: June 2026\n\nService Level for Gujarat in June 2026 was 90.8%."
    rows = named_period_source_rows(
        "Gujarat's Service Level rose from April to May 2026 then fell in June 2026 -- why?",
        existing_sources_text=already_retrieved,
        index_input_dir=tmp_path,
    )
    assert len(rows) == 2
    assert any("April 2026" in r["text"] for r in rows)
    assert any("May 2026" in r["text"] for r in rows)


def test_period_not_in_pilot_corpus_returns_nothing(tmp_path):
    """Gujarat_2026-07.txt genuinely isn't one of the 62 indexed pilot
    documents (even though it exists in the full, un-indexed 672-document
    corpus) -- this must stay unretrievable, not silently reach outside
    the pilot's own scope."""
    # tmp_path deliberately has no Gujarat_2026-07.txt file at all.
    rows = named_period_source_rows(
        "Gujarat's Service Level in July 2026 -- why?",
        existing_sources_text="",
        index_input_dir=tmp_path,
    )
    assert rows == []


def test_no_state_named_returns_nothing(tmp_path):
    _write_doc(tmp_path, "Gujarat_2026-05.txt", "Gujarat", "May 2026")
    rows = named_period_source_rows(
        "How did Service Level trend in May 2026?",
        existing_sources_text="",
        index_input_dir=tmp_path,
    )
    assert rows == []


def test_no_period_named_returns_nothing(tmp_path):
    _write_doc(tmp_path, "Gujarat_2026-05.txt", "Gujarat", "May 2026")
    rows = named_period_source_rows(
        "How is Gujarat performing?",
        existing_sources_text="",
        index_input_dir=tmp_path,
    )
    assert rows == []


def test_default_index_input_dir_points_at_the_real_pilot_corpus():
    from src.inference.named_period_evidence import DEFAULT_INDEX_INPUT_DIR

    assert DEFAULT_INDEX_INPUT_DIR.name == "input_pilot"
    assert (DEFAULT_INDEX_INPUT_DIR / "Gujarat_2026-04.txt").is_file()
