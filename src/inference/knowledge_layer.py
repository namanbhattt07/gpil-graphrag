"""
GPIL Knowledge Layer -- a small, structured business-terminology glossary,
detected deterministically in the question text and injected into the
answer-generation context alongside the existing Atomic Facts / SKU Ranking
evidence.

WHAT THIS FILE IS FOR (plain language):
A live investigation found the assistant answering "What is GPI?" with the
generic industry meaning ("General Product Inventory") and "What is IPM?"
with "Integrated Pest Management" -- both wrong for this project, where GPI
and IPM are GPIL's own product-category codes. The root cause: the indexed
corpus (src/graph/build_documents.py's output) only ever uses these codes
inside numeric KPI sentences ("For the GPI category ... Revenue was ..."),
never in a defining sentence, and GraphRAG's own local-search system prompt
explicitly invites the model to fall back on "relevant general knowledge"
when the retrieved evidence doesn't cover a question -- so an under-grounded
acronym question gets the globally common, wrong-for-this-project answer.

This module is the fix: a small, hand-curated dict of GPIL-specific terms
and their real business meaning (GLOSSARY), a deterministic, closed-
vocabulary detector that finds which of those terms a question actually
names (detect_glossary_terms), and a row-builder
(build_glossary_source_rows) that renders matched entries into the SAME
synthetic-Sources-row shape sku_evidence.py's ranking rows already use.

WHY SYNTHETIC SOURCES ROWS, NOT A PLAIN PROMPT STRING (important):
grounding_check.py's scan_definition_language() -- itself the fix for the
original GPI/IPM hallucination -- flags ANY sentence that asserts what a
term "stands for"/"means" unless the CITED (or whole retrieved) evidence
ALSO contains that kind of definitional language (see that function's own
docstring). If the glossary text were only appended to the prompt string
and never touched context_records, a correct GPI answer would fail that
same scanner for the same reason the wrong answer used to trigger it: no
evidence backing the definitional sentence. Routing glossary entries
through pipeline.py's existing additive-merge-into-context_records["sources"]
mechanism (the same one sku_evidence.py's ranking rows use) means the
definition becomes real, citable evidence BEFORE generation and grounding
both run, so a correctly-cited GPI/IPM answer passes grounding the same way
a correctly-cited KPI number does -- no special-casing needed in
grounding_check.py at all.

WHY A CLOSED, HAND-CURATED VOCABULARY, NOT AN LLM-GUESSED ONE:
Matching only the fixed alias list below means an unrelated acronym the
question happens to contain (anything not in GLOSSARY) can never trigger
this layer -- there is nothing for it to match against. This is the same
"grammar, not guessing" principle fact_structuring.py and query_requirements.py
already apply to Atomic Facts parsing and SKU-name detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_GLOSSARY_ROW_ID_PREFIX = "glossary-"


@dataclass(frozen=True)
class GlossaryEntry:
    """One GPIL business term. `aliases` are every surface form
    detect_glossary_terms() matches (case-insensitive, whole-word/whole-
    phrase only); `label` is the display name used in the rendered prompt
    block and citations; `definition` is the GPIL-specific meaning, written
    to explicitly state what the term does and does NOT mean in this
    system wherever a generic collision exists (GPI/IPM/ACV/WD).

    `forbidden_phrases` (default: none) are specific generic expansions
    already KNOWN to be wrong for this term -- e.g. GPI's "General Product
    Inventory", IPM's "Integrated Pest Management", both live-observed
    failures. grounding_check.py's scan_glossary_term_misuse() flags any
    answer using one of these, checked against the answer's own words
    directly and independent of citation -- see that function's docstring
    for why citation-scoped checking alone (scan_definition_language())
    isn't enough to catch a fabricated expansion that happens to cite a
    real, resolvable glossary row."""

    term_id: str
    aliases: tuple[str, ...]
    label: str
    definition: str
    forbidden_phrases: tuple[str, ...] = ()


# Order here is also the order entries render in when multiple terms are
# named in one question (e.g. "What are GPI and IPM?") -- fixed, not
# alphabetical, so the four product categories always group together.
GPIL_GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(
        term_id="gpi",
        aliases=("GPIL", "GPI"),
        label="GPI / GPIL",
        definition=(
            "GPIL is Godfrey Phillips India Ltd., the company this Sales & Distribution "
            "system belongs to. In this system, \"GPI\" stands for GPIL's own cigarette "
            "brand portfolio -- one of GPIL's four tracked product categories, distinct "
            "from IPM (licensed Marlboro), Ferrero (licensed confectionery), and Candy "
            "(in-house confectionery). GPI/GPIL NEVER means a generic industry term such "
            "as \"General Product Inventory\" in this system -- it always refers to "
            "Godfrey Phillips India Ltd. or its own cigarette category."
        ),
        forbidden_phrases=("general product inventory",),
    ),
    GlossaryEntry(
        term_id="ipm",
        aliases=("IPM",),
        label="IPM",
        definition=(
            "In this system, \"IPM\" stands for the Marlboro product category that GPIL "
            "sells under license -- one of GPIL's four tracked product categories, "
            "distinct from GPIL's own GPI cigarette category. IPM NEVER means "
            "\"Integrated Pest Management\" in this system -- it always refers to GPIL's "
            "licensed Marlboro/premium-tobacco category."
        ),
        forbidden_phrases=("integrated pest management",),
    ),
    GlossaryEntry(
        term_id="ferrero",
        aliases=("Ferrero",),
        label="Ferrero",
        definition=(
            "Ferrero is one of GPIL's four tracked product categories -- the licensed "
            "confectionery line (e.g. TicTac, Kinder Joy franchises) GPIL distributes "
            "alongside its tobacco categories (GPI, IPM)."
        ),
    ),
    GlossaryEntry(
        term_id="candy",
        aliases=("Candy",),
        label="Candy",
        definition=(
            "Candy is one of GPIL's four tracked product categories -- GPIL's own "
            "in-house confectionery/sweets line, distinct from the licensed Ferrero "
            "confectionery category."
        ),
    ),
    GlossaryEntry(
        term_id="distributor",
        aliases=("Distributor", "Distributors", "WD", "Wholesale Distributor"),
        label="Distributor / WD",
        definition=(
            "Distributor -- also written \"WD\", and WD stands for Wholesale "
            "Distributor -- is the entity that receives stock from GPIL and supplies "
            "outlets within a state/zone. \"Distributor\"/\"WD\" is a distinct role from "
            "\"Dealer\" (an outlet channel type, not a distributor) -- do not conflate "
            "the two."
        ),
    ),
    GlossaryEntry(
        term_id="dealer",
        aliases=("Dealer", "Dealers"),
        label="Dealer",
        definition=(
            "Dealer is one of the four outlet Channel Types tracked in this system "
            "(alongside Retail, Hawkers, and Modern Trade) -- an outlet classification, "
            "not a Distributor/WD. A Dealer outlet receives stock FROM a Distributor; it "
            "is not itself a distributor."
        ),
    ),
    GlossaryEntry(
        term_id="hero_sku",
        aliases=("Hero SKU", "Hero SKUs"),
        label="Hero SKU",
        definition=(
            "Hero SKU is an informal but business-critical term for the flagship SKU of "
            "each of GPIL's product franchises (the flagship SKU of each of GPI's 4 "
            "franchises, plus Marlboro's flagship) -- these 5 SKUs are deliberately "
            "weighted to sell far more than the rest of the portfolio, mirroring a "
            "real-world Pareto/80-20 pattern. \"Hero SKU\" is not a data column; it is a "
            "fixed, named set of SKUs."
        ),
    ),
    GlossaryEntry(
        term_id="acv",
        aliases=("ACV",),
        label="ACV",
        definition=(
            "In this system, \"ACV\" stands for GPIL's tier-weighted distribution-reach "
            "KPI for a product category -- its exact tier-weighting formula is documented "
            "elsewhere in this project as pending final GPIL confirmation. ACV here does "
            "NOT mean \"Actual Cash Value\" or \"Annual Contract Value\"."
        ),
    ),
    GlossaryEntry(
        term_id="numeric_distribution",
        aliases=("Numeric Distribution",),
        label="Numeric Distribution",
        definition=(
            "Numeric Distribution is the percentage of eligible outlets that billed at "
            "least one SKU of a given category/franchise in a period -- a reach/"
            "weighted-distribution KPI, not a sales-value KPI."
        ),
    ),
    GlossaryEntry(
        term_id="range_billing",
        aliases=("Range Billing",),
        label="Range Billing",
        definition=(
            "Range Billing is the share of a category's/franchise's full SKU range that "
            "an outlet actually orders from, among outlets that order at least one SKU "
            "-- it distinguishes \"wide but shallow\" distribution (many outlets, few "
            "SKUs each) from deep distribution."
        ),
    ),
    GlossaryEntry(
        term_id="out_of_stock_rate",
        aliases=("Out-of-Stock Rate", "OOS%", "OOS Rate"),
        label="Out-of-Stock Rate (OOS%)",
        definition=(
            "Out-of-Stock Rate (OOS%) is the share of SKU-stocking-points found stocked "
            "out at a distributor in a period -- an inventory-availability KPI."
        ),
    ),
    GlossaryEntry(
        term_id="service_level",
        aliases=("Service Level",),
        label="Service Level",
        definition=(
            "Service Level is the share of ordered quantity actually delivered "
            "(delivered units divided by ordered units) -- GPIL's core order-fulfilment "
            "KPI."
        ),
    ),
    GlossaryEntry(
        term_id="dropsize",
        aliases=("Dropsize",),
        label="Dropsize",
        definition=(
            "Dropsize is the average number of units ordered per productive visit (a "
            "productive visit being one that results in an order). Its exact formula is "
            "documented elsewhere in this project as pending final GPIL confirmation, but "
            "the working definition used throughout this system is units-ordered-per-"
            "productive-visit."
        ),
    ),
    GlossaryEntry(
        term_id="productivity",
        aliases=("Productivity",),
        label="Productivity",
        definition=(
            "Productivity is the share of sales visits that resulted in an order (a "
            "\"productive\" visit) -- GPIL's core sales-effectiveness KPI."
        ),
    ),
)


def _alias_pattern(alias: str) -> re.Pattern:
    """Whole-word/whole-phrase, case-insensitive match for one alias.
    Deliberately uses lookaround on "is this an alnum character" rather
    than \\b: a handful of aliases end in punctuation (e.g. "OOS%"), where
    \\b's word/non-word transition rule does not reliably sit at the
    boundary a human reader would expect -- this lookaround form treats any
    non-alphanumeric neighbor (space, punctuation, string edge) as a valid
    boundary regardless of what the alias itself ends with."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)


_TERM_PATTERNS: tuple[tuple[GlossaryEntry, tuple[re.Pattern, ...]], ...] = tuple(
    (entry, tuple(_alias_pattern(alias) for alias in entry.aliases)) for entry in GPIL_GLOSSARY
)


def detect_glossary_terms(question: str) -> list[GlossaryEntry]:
    """Every GPIL_GLOSSARY entry named in `question` -- case-insensitive,
    closed-vocabulary matching against this fixed glossary only, so a term
    NOT in the list (an unrelated acronym) can never spuriously match.
    Returns entries in GPIL_GLOSSARY's own fixed order, each at most once,
    even if more than one of its aliases appears (e.g. a question naming
    both "GPI" and "GPIL" still returns that one entry once)."""
    matches: list[GlossaryEntry] = []
    for entry, patterns in _TERM_PATTERNS:
        if any(pattern.search(question) for pattern in patterns):
            matches.append(entry)
    return matches


def _glossary_row_text(entry: GlossaryEntry) -> str:
    """One synthetic Sources-row body, in the SAME Term/Definition/Source
    shape every other Atomic Facts section already uses -- see this
    module's docstring for why the "Source:" field matters (it's what lets
    the model cite this exact row via a real "[Data: Sources (id)]" tag,
    which is what makes a resulting definitional sentence pass
    grounding_check.py's scan_definition_language())."""
    return (
        f"Term: {entry.label}\n"
        f"Definition: {entry.definition}\n"
        f"Source: Sources ({_GLOSSARY_ROW_ID_PREFIX}{entry.term_id})"
    )


def build_glossary_source_rows(question: str) -> list[dict]:
    """Every glossary entry `question` names, rendered as a synthetic
    Sources-shaped row ({"id", "text"}) -- the SAME shape
    sku_evidence.py's ranking rows use, so pipeline.py can merge these into
    context_records["sources"] with the exact same additive-merge mechanism
    (see pipeline.py's _augment_context_with_glossary()). Returns [] (never
    raises) when the question names no known GPIL term -- the common case,
    and safe to always call unconditionally before every question."""
    return [
        {"id": f"{_GLOSSARY_ROW_ID_PREFIX}{entry.term_id}", "text": _glossary_row_text(entry)}
        for entry in detect_glossary_terms(question)
    ]


_GLOSSARY_BLOCK_HEADER = "-----GPIL Knowledge Layer-----"

# Co-located with the definitions themselves (inside build_glossary_block()'s
# own returned text), not just in the far-downstream _GLOSSARY_CITATION_
# GUIDANCE paragraph in answer.py -- 2026-08-23 "What is GPI and IPM?" live
# investigation found the model reading GraphRAG's OWN indexed Entity
# description for "IPM" (auto-written at index time from sparse context,
# and drifting toward "agricultural practices" language) as legitimate
# competing evidence, filling the rest in from its own strong pretrained
# "Integrated Pest Management" prior -- i.e. the failure wasn't the model
# ignoring an instruction so much as never being told this block outranks
# OTHER RETRIEVED evidence for the same term, not just outside/world
# knowledge. Repeating the override right next to the definitions means it
# survives even if a caller only reads/quotes this block in isolation.
#
# 2026-08-23 FOLLOW-UP ("What is IPM?" single-term investigation): the
# real indexed "IPM" Entity description turned out to be a long,
# confident, MULTI-PARAGRAPH narrative that repeats agriculture-adjacent
# phrasing across seven separate states/periods ("agricultural cycles",
# "regional agricultural practices", "diverse agricultural matrix",
# "enhanced agricultural productivity", ...) -- length and repetition this
# project's own corpus never actually supports (the corpus is pure numeric
# KPI sentences; every one of those descriptive phrases was invented by the
# index-time summarizer). A single-term question concentrates GraphRAG's
# retrieval budget entirely on that one entity's neighborhood, surfacing
# this description prominently, whereas a multi-term question splits that
# budget across both terms and was not observed to fail the same way. The
# added sentence below names this specific failure mode explicitly (long/
# repeated /confident-sounding is not evidence of correctness) rather than
# just repeating "prefer this over other evidence" a different way.
_GLOSSARY_BLOCK_PREFACE = (
    "The definitions below are this project's OWN confirmed, authoritative meaning for each "
    "term named. For any term defined here, this section OVERRIDES every other description of "
    "that same term anywhere else in the data tables (including any Entity or Community Report "
    "description auto-generated from the corpus, which may be vague, incomplete, or misleading) "
    "as well as any generic/world-knowledge meaning of the same word or acronym. A retrieved "
    "Entity or Community Report description for the same term may be long, detailed, and repeat "
    "its own framing across many sentences -- length, detail, and repetition are NOT evidence of "
    "correctness; that description was auto-generated without ever seeing a confirmed definition, "
    "and does not become more reliable the longer or more often it repeats itself. This section is "
    "authoritative ONLY for the exact terms it defines below -- it says nothing about, and must "
    "not be extended to, any other term."
)

# 2026-08-23 ("What is IPM?" single-term investigation): a SECOND copy of
# the same matched definitions, rendered under this distinct header and
# placed immediately AFTER context_chunks in answer.py's
# _context_data_with_facts() -- i.e. bookending GraphRAG's own retrieved
# text (which is where the long, competing "IPM" Entity narrative lives)
# with the authoritative definition on BOTH sides, not just before it. A
# single opening block, however strongly worded, is comparatively short
# next to a seven-paragraph competing narrative and can be outweighed in
# practice, especially once a single-term question's retrieval budget
# concentrates entirely on that one entity. Restating the exact same facts
# again right after that content -- and immediately before the rest of the
# generation guidance -- gives the correct definition the same kind of
# textual recency the competing narrative would otherwise have on its own,
# without editing GraphRAG's own retrieved text, adding a new forbidden-
# phrase blocklist entry, or hardcoding a specific final answer.
_GLOSSARY_REMINDER_HEADER = (
    "-----GPIL Knowledge Layer (Reminder -- still authoritative, still overrides any "
    "conflicting description above, no matter how long or detailed)-----"
)


def _matched_glossary_row_texts(context_records: dict[str, "object"]) -> list[str]:
    """Every "glossary-"-id-prefixed row's text in context_records["sources"],
    in table order -- the shared read shared by build_glossary_block() and
    build_glossary_reminder_block() so both render from exactly the same
    matched rows (never two independently-maintained reads that could
    silently drift apart)."""
    df = context_records.get("sources") if context_records else None
    if df is None or df.empty or "id" not in df.columns or "text" not in df.columns:
        return []
    texts: list[str] = []
    for _, row in df.iterrows():
        row_id = str(row.get("id", ""))
        if not row_id.startswith(_GLOSSARY_ROW_ID_PREFIX):
            continue
        text = row.get("text")
        if isinstance(text, str) and text:
            texts.append(text.strip())
    return texts


def build_glossary_block(context_records: dict[str, "object"]) -> str:
    """Every "glossary-"-id-prefixed row in context_records["sources"],
    concatenated under one labeled header -- "" (never an error) when none
    exist, i.e. every question that names no known GPIL term. Mirrors
    fact_structuring.build_deterministic_ranking_block() exactly (reads by
    id PREFIX, not by re-detecting terms from the question a second time),
    so this module owns detection+rendering end to end and answer.py's
    prompt-assembly code only ever reads already-resolved context_records,
    the same contract every other additive block already follows."""
    texts = _matched_glossary_row_texts(context_records)
    if not texts:
        return ""
    return _GLOSSARY_BLOCK_HEADER + "\n\n" + _GLOSSARY_BLOCK_PREFACE + "\n\n" + "\n\n".join(texts)


def build_glossary_reminder_block(context_records: dict[str, "object"]) -> str:
    """The SAME matched glossary rows as build_glossary_block(), rendered a
    second time under a distinct "Reminder" header -- see
    _GLOSSARY_REMINDER_HEADER's comment above for why this exists (bookend
    GraphRAG's own retrieved text with the authoritative definition on both
    sides). "" (never an error) under the exact same condition
    build_glossary_block() returns "" for."""
    texts = _matched_glossary_row_texts(context_records)
    if not texts:
        return ""
    return _GLOSSARY_REMINDER_HEADER + "\n\n" + "\n\n".join(texts)
