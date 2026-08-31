"""
Deterministic document-completion for explicitly-named state+period(s) --
a retrieval-recall fix, additive to GraphRAG's own retrieval.

WHAT THIS FILE IS FOR (plain language):
GraphRAG's local-search retrieval is a semantic (embedding-similarity)
top-k search -- it does NOT guarantee that a document a question
explicitly names by state + period is actually among the rows it returns,
even when that exact document is one of the 62 the pilot index was built
from. A live run of "Gujarat's Service Level rose from April to May 2026
then fell in June 2026" only had GraphRAG's retrieval return the June 2026
Gujarat document -- April and May were silently missing from
context_records["sources"], and the generation model, given only June's
numbers, invented plausible-sounding April/May figures to answer a
question its own retrieved evidence never actually covered. Grounding
correctly caught the fabricated numbers and failed closed both times, but
that's the wrong reason to fail: the real April/May 2026 Gujarat documents
ARE in the 62-document pilot index, just not retrieved for this query.

This module closes that gap the SAME way sku_evidence.py and
knowledge_layer.py already close their own retrieval gaps: read the EXACT
already-indexed document straight off disk and merge it into
context_records["sources"] as an additional citable Sources row -- purely
additive, never replacing what GraphRAG's own retrieval already found, and
never invoking GraphRAG or the embedding API a second time.

WHY ONLY data/graphrag_index/input_pilot/, NEVER data/graphrag_input/:
input_pilot/ holds exactly the 62 documents GraphRAG actually indexed for
the live pilot (same filenames, same bytes, per src/graph/build_documents.py's
naming convention) -- the true, complete "existing indexed data" this
project's guard pipeline is scoped to. data/graphrag_input/ is the full
672-document corpus Phase 4 generates; the vast majority of it was
deliberately never indexed for this pilot. Reading a document from there
would silently expand the answerable scope beyond the pilot index this
system is meant to be validated against -- exactly the thing this module
must NOT do. A period whose file doesn't exist under input_pilot/ (even if
it exists under the full corpus) is treated as genuinely not indexed, the
same honest "insufficient evidence" outcome as any other real gap.

WHY THIS ISN'T "changing GraphRAG retrieval internals": nothing here
touches GraphRAG's embedding search, its context builder, or the index
itself -- it only fills a documented recall gap for the ONE case this
project's guard pipeline can check deterministically and cheaply: the
question's own explicitly-named state + period(s), the same "trust the
question's own explicit statement over retrieval" precedent
premise_check.py already established for state/period extraction (see
extract_state_from_question/extract_all_periods_from_question, reused
here rather than re-implemented).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.inference.premise_check import (
    extract_all_periods_from_question,
    extract_period_from_question,
    extract_state_from_question,
)

# Mirrors sku_evidence.py's own PROJECT_ROOT computation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_INPUT_DIR = _PROJECT_ROOT / "data" / "graphrag_index" / "input_pilot"

_ROW_ID_PREFIX = "named-period-doc-"

# Matches every Sources row's own "State: X\nPeriod: Y" header (the SAME
# fixed template build_documents.py writes, already relied on elsewhere --
# see premise_check.py's _SOURCE_HEADER_RE) -- used only to detect whether
# a state+period this question names is ALREADY present in what GraphRAG
# retrieved, so this module never injects a duplicate of a document
# retrieval already found on its own.
_SOURCE_HEADER_RE = re.compile(r"State:\s*(?P<state>[^\n]+)\s*\n\s*Period:\s*(?P<period>[^\n]+)")


def _filename_for(state: str, period: str) -> str | None:
    """'Gujarat', 'April 2026' -> 'Gujarat_2026-04.txt' -- the exact naming
    convention src/graph/build_documents.py uses (verbatim state name,
    underscores for spaces, zero-padded YYYY-MM). None if `period` isn't a
    parseable 'Month YYYY' string."""
    try:
        dt = datetime.strptime(period, "%B %Y")
    except ValueError:
        return None
    return f"{state.replace(' ', '_')}_{dt.year:04d}-{dt.month:02d}.txt"


def _already_retrieved(sources_text_blob: str, state: str, period: str) -> bool:
    """True if any retrieved Sources row's own header already names this
    exact state+period -- checked against the CONCATENATED text of all
    Sources rows (cheap; each row's header is a small, fixed prefix), so
    this module never injects a document retrieval already surfaced."""
    for m in _SOURCE_HEADER_RE.finditer(sources_text_blob):
        if m.group("state").strip() == state and m.group("period").strip() == period:
            return True
    return False


def named_period_source_rows(
    question: str,
    existing_sources_text: str,
    index_input_dir: Path | None = None,
) -> list[dict]:
    """Every {"id", "text"} Sources-shaped row for a state+period this
    `question` explicitly names (via premise_check.py's own state/period
    extractors) that (a) is not already covered by `existing_sources_text`
    (GraphRAG's own retrieval) and (b) exists as a real file under the 62-
    document pilot's own input directory. Returns [] (never raises) when
    the question names no state, names no period, every named period is
    already covered, or a named period simply isn't one of the 62 indexed
    documents -- all of these are legitimate "nothing to add" outcomes,
    not errors; a period genuinely outside the pilot's 62 documents stays
    exactly as unretrievable as before this module existed."""
    state = extract_state_from_question(question)
    if not state:
        return []

    periods = extract_all_periods_from_question(question)
    if not periods:
        single = extract_period_from_question(question)
        periods = [single] if single else []
    if not periods:
        return []

    base_dir = Path(index_input_dir) if index_input_dir is not None else DEFAULT_INDEX_INPUT_DIR

    rows: list[dict] = []
    for period in periods:
        if _already_retrieved(existing_sources_text, state, period):
            continue
        filename = _filename_for(state, period)
        if filename is None:
            continue
        path = base_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Underscored period ("April_2026"), matching sku_evidence.py's own
        # id convention ("sku-Gujarat-April_2026") -- a raw space inside a
        # citation id is unnecessary friction for the model to reproduce
        # exactly in its own "[Data: Sources (id)]" tags.
        row_id = f"{_ROW_ID_PREFIX}{state.replace(' ', '_')}-{period.replace(' ', '_')}"
        rows.append({"id": row_id, "text": text})
    return rows
