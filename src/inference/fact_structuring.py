"""
Atomic Distributor Facts -- a deterministic, additive restructuring of the
retrieved Sources evidence for the answer-generation prompt.

WHAT THIS FILE IS FOR (plain language):
Diagnosis of a live misattribution failure (Baxter's name paired with
Campbell's numbers) found that the answer-generation LLM receives Sources
evidence as one long block of prose: several distributor-deviation
sentences back to back, each in the exact same shape, with no visual
grouping of a single fact's own fields. The extraction/retrieval layers
were confirmed correct (see the Phase 5/6 diagnosis) -- the risk is purely
in how densely-packed, structurally-repetitive prose gets read by the
generation model.

Every "Distributor X showed a significant deviation on..." sentence in the
corpus is produced by exactly one of three fixed f-string templates in
src/graph/build_documents.py (state-grain %, Dropsize, category-grain %) --
never free-form LLM prose. That determinism means the sentence can be
parsed back into its fields with a plain regex, with no ambiguity and no
second model call: build_atomic_facts_block() reads context_records
["sources"] (data already retrieved -- no new fetch) and emits one
explicit FACT block per matched sentence, each field kept together, each
still tagged with its originating Sources record id so the model can cite
it exactly as it already cites Sources today.

This is a strictly ADDITIVE transform: nothing here removes or rewrites
the original Sources prose (answer.py appends this block onto the existing
context_chunks, never replaces it), so if some future sentence shape
doesn't match the regex, the raw text -- today's only representation -- is
still present as a fallback. It also touches no retrieval, index, or
extraction code, and contains no distributor/state/metric/value-specific
logic: the regex matches the sentence GRAMMAR, not any particular fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Mirrors the three deviation-sentence templates in
# src/graph/build_documents.py's _render_distributor_deviations (state-grain
# %, Dropsize, category-grain %) as one pattern:
#   "Distributor {name} showed a significant deviation on {metric}"
#   ["for the {category} category"] "in {period}: {value} vs. the state
#   average of {average}, a gap of {gap}."
# `metric` is non-greedy so it stops at the first point the rest of the
# pattern can match -- for a category-grain sentence that is right before
# " for the {category} category", for the other two templates it is right
# before " in {period}". Nothing here names a specific metric, category,
# distributor, or value -- only the fixed grammar the generator always uses.
_DEVIATION_SENTENCE_RE = re.compile(
    r"Distributor (?P<distributor>[A-Z][^:]*?) showed a significant deviation on "
    r"(?P<metric>[^:]+?)(?: for the (?P<category>\w+) category)? in "
    r"(?P<period>[A-Za-z]+ \d{4}): "
    r"(?P<value>[\d.]+%?) vs\. the state average of (?P<average>[\d.]+%?), "
    r"a gap of (?P<gap>[\d.]+ (?:percentage points|percent))\."
)

_FACTS_HEADER = "-----Atomic Distributor Facts-----"

# ---------------------------------------------------------------------------
# State-level KPI atomic facts -- additive extension, entirely separate from
# the distributor-deviation extraction above (neither its regex nor its
# output is touched by this section).
#
# WHAT THIS SECTION IS FOR (plain language):
# The distributor-deviation facts above exist because dense, structurally-
# repetitive prose is hard for the generation model to read correctly. The
# same risk applies to state-level headline KPI sentences -- "Productivity
# for Bihar in November 2025 was 89.0%." -- which sit in the SAME dense
# Sources prose block as everything else, with no structured scaffolding of
# their own. A live cross-state comparison failure showed the model cite
# the wrong record for exactly this shape of fact (attributing a
# state+metric+period+value fact to the wrong source), even though the
# correct fact was present, verbatim, elsewhere in the SAME retrieved
# context. This section gives the model the same explicit
# State/Metric/Period/Value/Source scaffolding for headline KPI sentences
# that distributor deviations already get, so the model has a clean,
# citable fact to reach for instead of having to bind state+metric+period+
# value back together itself out of a paragraph of prose.
#
# Every "<Metric> for <State> in <Period> was <value>." sentence is written
# by exactly one fixed f-string template in src/graph/build_documents.py
# (state-grain KPI block) -- never free-form LLM prose -- so, exactly like
# the distributor case above, this can be parsed back into its fields with
# a plain regex, no ambiguity, no second model call.
# ---------------------------------------------------------------------------

# Mirrors the fixed "State: X\nPeriod: Y" header every Phase 4 document
# opens with (see src/graph/build_documents.py / premise_check.py's own
# copy of this same header regex) -- used here only to know, per source
# row, which literal state/period string to require the metric sentence to
# name, so a category-level sentence ("... for the Ferrero category ...
# during December 2025: Numeric Distribution was ...") can never be
# mistaken for a state-level headline sentence: that shape uses "during"
# and a colon, never "<state> in <period> was <value>".
_STATE_HEADER_RE = re.compile(r"State:\s*(?P<state>[^\n]+)\s*\n\s*Period:\s*(?P<period>[^\n]+)")

# The six state-level metrics Phase 4 writes into every document's opening
# paragraph (see premise_check.py's own copy of this same list) -- kept as
# a plain, duplicated constant rather than an import, matching this
# module's own established convention (see _DEVIATION_SENTENCE_RE above)
# of depending only on the fixed sentence GRAMMAR, never on another
# inference-layer module.
_STATE_METRIC_NAMES = (
    "Productivity",
    "Service Level",
    "SKUs per Transaction",
    "Dropsize",
    "Inventory Turns",
    "Inventory Days",
)

_STATE_FACTS_HEADER = "-----Atomic State KPI Facts-----"


def _state_metric_facts_from_source_text(source_id: str, text: str) -> list[str]:
    """One formatted FACT paragraph per state-level headline KPI sentence
    matched in this single Sources record's text, in the SAME
    State/Metric/Period/Value/Source shape the distributor facts above
    use. Returns [] (never raises) when the record has no parseable
    "State: X\\nPeriod: Y" header, or none of the six known metric
    sentences match -- exactly the same "append nothing, never an error"
    contract _facts_from_source_text() already follows."""
    header_match = _STATE_HEADER_RE.search(text)
    if not header_match:
        return []
    state = header_match.group("state").strip()
    period = header_match.group("period").strip()
    if not state or not period:
        return []

    facts: list[str] = []
    for name in _STATE_METRIC_NAMES:
        # Optional non-capturing group handles Dropsize's parenthetical
        # definition aside, exactly like premise_check.py's own version of
        # this pattern ("Dropsize (units ordered per productive visit --
        # definition pending GPIL confirmation) for Bihar in ... was ...").
        pattern = (
            rf"{re.escape(name)}(?:\s*\([^)]*\))?\s+for\s+{re.escape(state)}\s+in\s+"
            rf"{re.escape(period)}\s+was\s+([\d,]+\.?\d*%?)"
        )
        m = re.search(pattern, text)
        if not m:
            continue
        facts.append(
            "State: {state}\n"
            "Metric: {metric}\n"
            "Period: {period}\n"
            "Value: {value}\n"
            "Source: Sources ({source_id})".format(
                state=state, metric=name, period=period,
                value=m.group(1), source_id=source_id,
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Category-level KPI atomic facts -- additive extension, entirely separate
# from both the distributor-deviation and state-headline sections above
# (neither's regex, output, or numbering is touched by this section).
#
# WHAT THIS SECTION IS FOR (plain language):
# A live Q4 diagnosis (Gujarat/Ferrero/April 2026) found the SAME kind of
# citation-selection failure the state-headline section above was built to
# prevent, but for category-scoped KPI paragraphs instead of state
# headlines: the state's document was chunked by GraphRAG's own indexer
# into two retrieved Source records, and the category paragraph containing
# the correct answer survived intact only in the SECOND (continuation)
# chunk -- which, unlike the first chunk, carries no "State: X\nPeriod: Y"
# header at all (only the chunk that happens to start at the top of the
# document keeps that header; a chunk boundary falling mid-document drops
# it). The model, with no clean scaffolded fact to reach for, cited a
# GraphRAG-generated Entity description instead -- whose own auto-written
# prose stated the period in one sentence and the value in the next,
# which the (unmodified, still strict) grounding checker correctly
# rejected, since it never states period+value together as one fact.
#
# WHY THIS EXTRACTOR MUST NOT DEPEND ON THE DOCUMENT HEADER (unlike the
# state-headline section above): the state-headline sentence ("Productivity
# for Bihar in November 2025 was 89.0%.") does not repeat the period in a
# form distinguishable from other sentences without the document's own
# header telling us which state/period this row is even about. The
# category-block sentence is different -- it is fully self-contained:
# "For the Ferrero category (...) in Gujarat during April 2026: ..." names
# its own state and period directly, in the same sentence as all four
# metric values. That is exactly what lets this extractor work on a
# header-less continuation chunk, the same way the distributor-deviation
# extractor above already does (each deviation sentence is also fully
# self-contained, with no header dependency).
#
# Every "For the {category} category (...) in {state} during {period}: ..."
# paragraph is written by exactly one fixed f-string template in
# src/graph/build_documents.py's per-category loop -- never free-form LLM
# prose -- so, exactly like the other two sections, this can be parsed back
# into its fields with a plain regex, no ambiguity, no second model call.
# ---------------------------------------------------------------------------

# The parenthetical qualifiers after "ACV" and "Range Billing" have varied
# across corpus-generation script versions (an older wording with no
# trailing "-- definition pending GPIL confirmation" clause exists
# alongside a newer one that adds it) -- this extractor deliberately never
# hardcodes that qualifier text, matching ANY parenthetical content via
# `\([^)]*\)`, exactly the same "match the grammar, not a specific
# qualifier string" principle _state_metric_facts_from_source_text() already
# applies to Dropsize's parenthetical aside. "Out-of-Stock rate" is matched
# with a literal lowercase "rate" -- this exact category-block template is
# the one place in the whole corpus that spells it this way (the
# distributor-deviation template elsewhere capitalizes it as "Rate") -- the
# canonical capitalized "Out-of-Stock Rate" is only used in this function's
# OWN OUTPUT (the FACT block's Metric: field), never assumed of the input.
_CATEGORY_BLOCK_RE = re.compile(
    r"For the (?P<category>[A-Za-z]+) category \(franchises:[^)]*\) in "
    r"(?P<state>.+?) during (?P<period>[A-Za-z]+ \d{4}): "
    r"Numeric Distribution was (?P<nd>[\d,]+\.?\d*%?), "
    r"ACV\s*\([^)]*\) was (?P<acv>[\d,]+\.?\d*%?), "
    r"Out-of-Stock rate was (?P<oos>[\d,]+\.?\d*%?), and "
    r"average Range Billing\s*\([^)]*\) was (?P<rb>[\d,]+\.?\d*%?)\."
)

_CATEGORY_FACTS_HEADER = "-----Atomic Category KPI Facts-----"

# (regex group name, canonical output Metric: label) -- fixed order matches
# the template's own fixed field order, so facts render in the same order
# a reader would encounter them in the source sentence.
_CATEGORY_METRIC_FIELDS = (
    ("nd", "Numeric Distribution"),
    ("acv", "ACV"),
    ("oos", "Out-of-Stock Rate"),  # canonicalized from the source's lowercase "rate"
    ("rb", "Range Billing"),
)


def _category_metric_facts_from_source_text(source_id: str, text: str) -> list[str]:
    """One formatted FACT paragraph per METRIC (four per matched category
    paragraph) out of every "For the {category} category (...) in {state}
    during {period}: ..." sentence in this single Sources record's text --
    in the SAME State/Metric/Period/Value/Source shape the other two
    sections use, plus a Category field. Deliberately does NOT require (or
    even look at) a "State: X\\nPeriod: Y" document header -- see the
    module-level comment above for why the category sentence's own
    self-contained state+period makes that unnecessary, unlike the
    state-headline section. Returns [] (never raises) when no category
    paragraph in this text matches the known template -- e.g. a paragraph
    truncated mid-sentence by chunking never matches (the regex requires
    the whole sentence through its closing period), so a truncated
    paragraph correctly produces zero facts rather than a partial one."""
    facts: list[str] = []
    for match in _CATEGORY_BLOCK_RE.finditer(text):
        category = match.group("category")
        state = match.group("state").strip()
        period = match.group("period").strip()
        for group_name, metric_label in _CATEGORY_METRIC_FIELDS:
            facts.append(
                "State: {state}\n"
                "Category: {category}\n"
                "Metric: {metric}\n"
                "Period: {period}\n"
                "Value: {value}\n"
                "Source: Sources ({source_id})".format(
                    state=state, category=category, metric=metric_label,
                    period=period, value=match.group(group_name), source_id=source_id,
                )
            )
    return facts


def _facts_from_source_text(source_id: str, text: str) -> list[str]:
    """One formatted FACT paragraph per deviation sentence matched in this
    single Sources record's text -- numbered later by the caller so
    numbering stays sequential across the whole context, not per-row."""
    facts = []
    for match in _DEVIATION_SENTENCE_RE.finditer(text):
        category = match.group("category") or "(none)"
        facts.append(
            "Distributor: {distributor}\n"
            "Metric: {metric}\n"
            "Category: {category}\n"
            "Period: {period}\n"
            "Value: {value}\n"
            "Average: {average}\n"
            "Gap: {gap}\n"
            "Source: Sources ({source_id})".format(
                distributor=match.group("distributor"),
                metric=match.group("metric"),
                category=category,
                period=match.group("period"),
                value=match.group("value"),
                average=match.group("average"),
                gap=match.group("gap"),
                source_id=source_id,
            )
        )
    return facts


def _all_atomic_fact_texts(
    context_records: dict[str, "pd.DataFrame"],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """The four raw (unnumbered) fact-text lists build_atomic_facts_block()
    renders into "FACT N" blocks -- (distributor, state, category, sku),
    each in the SAME order/indexing its own "-----Atomic ... Facts-----"
    section numbers them 1..k. Split out from build_atomic_facts_block()
    so grounding_check.py can index into the SKU list directly by fact
    number (see sku_atomic_fact_text_at()) to resolve a citation to the
    ONE specific fact it names, not the whole multi-SKU source blob it
    came from -- see that function's docstring for why precision matters
    here specifically."""
    df = context_records.get("sources") if context_records else None
    if df is None or df.empty or "text" not in df.columns or "id" not in df.columns:
        return [], [], [], []

    distributor_facts: list[str] = []
    state_facts: list[str] = []
    category_facts: list[str] = []
    sku_facts: list[str] = []
    for _, row in df.iterrows():
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        source_id = str(row.get("id"))
        distributor_facts.extend(_facts_from_source_text(source_id, text))
        state_facts.extend(_state_metric_facts_from_source_text(source_id, text))
        category_facts.extend(_category_metric_facts_from_source_text(source_id, text))
        sku_facts.extend(_sku_metric_facts_from_source_text(source_id, text))
    return distributor_facts, state_facts, category_facts, sku_facts


def sku_atomic_fact_texts(context_records: dict[str, "pd.DataFrame"]) -> list[str]:
    """The raw (unnumbered) list of SKU Atomic Fact texts, in the SAME
    order build_atomic_facts_block() numbers them 1..k in the
    "-----Atomic SKU Facts-----" section -- i.e. index N-1 of this list is
    exactly what the model saw labeled "FACT N" for that section. Public
    so grounding_check.py can resolve an "Atomic SKU Facts (N)"-shaped
    citation to the ONE specific fact it names (see
    _SKU_ATOMIC_FACTS_NAME_ALIASES there), not the whole multi-SKU source
    blob that fact came from -- a live 15-question validation pass caught
    a correct, grounding-passed "Marlboro Pack 1" claim displaying
    GPI_Franchise_1 Pack 1's sentence as its "evidence" purely because
    that SKU's fact happened to render first in the same source blob;
    indexing directly into this SAME list the model was shown avoids
    that."""
    return _all_atomic_fact_texts(context_records)[3]


def build_atomic_facts_block(context_records: dict[str, "pd.DataFrame"]) -> str:
    """Parse every distributor-deviation sentence, every state-level
    headline KPI sentence, every category-level KPI paragraph, AND every
    SKU-level KPI sentence (Problem 1 -- see
    _sku_metric_facts_from_source_text()) out of context_records["sources"],
    and return them as up to four separate sections -- "-----Atomic
    Distributor Facts-----", "-----Atomic State KPI Facts-----",
    "-----Atomic Category KPI Facts-----", and "-----Atomic SKU
    Facts-----" -- each with its own independent FACT numbering starting
    at 1, so adding a later section never renumbers or otherwise changes a
    single character of an earlier section's own output. Returns "" only
    when NONE of the four sections found anything to report -- callers
    must treat "" as "append nothing", never as an error, since the raw
    Sources text is always still present as the existing fallback either
    way.
    """
    distributor_facts, state_facts, category_facts, sku_facts = _all_atomic_fact_texts(context_records)

    blocks: list[str] = []
    if distributor_facts:
        numbered = [f"FACT {i}\n{fact}" for i, fact in enumerate(distributor_facts, start=1)]
        blocks.append(_FACTS_HEADER + "\n\n" + "\n\n".join(numbered))
    if state_facts:
        numbered = [f"FACT {i}\n{fact}" for i, fact in enumerate(state_facts, start=1)]
        blocks.append(_STATE_FACTS_HEADER + "\n\n" + "\n\n".join(numbered))
    if category_facts:
        numbered = [f"FACT {i}\n{fact}" for i, fact in enumerate(category_facts, start=1)]
        blocks.append(_CATEGORY_FACTS_HEADER + "\n\n" + "\n\n".join(numbered))
    if sku_facts:
        numbered = [f"FACT {i}\n{fact}" for i, fact in enumerate(sku_facts, start=1)]
        blocks.append(_SKU_FACTS_HEADER + "\n\n" + "\n\n".join(numbered))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Structured Atomic Facts -- additive, parallel to everything above.
#
# WHAT THIS SECTION IS FOR (plain language):
# Everything above this point renders Atomic Facts as PROMPT TEXT for the
# generation model to read. Nothing above it gives any OTHER part of the
# pipeline a first-class, structured way to ask "what does the evidence
# actually say Campbell's GPI Out-of-Stock Rate was in April 2026?" without
# re-deriving the answer from raw prose a second time. The entity/value
# misbinding investigation (Baxter's name attached to Campbell's real GPI
# figure) found that this forced grounding_check.py's fallback scanner to
# re-implement its own, weaker, prose-scoped version of "find this
# distributor's number" -- duplicating logic that already exists here,
# with its own separate blind spots (see grounding_check.py's
# scan_entity_numeric_claims() docstring for the specific gap this closes).
#
# extract_atomic_facts() parses the SAME context_records["sources"] text
# with the EXACT SAME compiled regexes (_DEVIATION_SENTENCE_RE,
# _STATE_HEADER_RE/_STATE_METRIC_NAMES, _CATEGORY_BLOCK_RE) the text-
# rendering functions above already use -- so there is exactly one place
# that knows how to recognize a distributor-deviation/state-KPI/category-KPI
# sentence, never two independent implementations that could quietly drift
# apart. This function is purely ADDITIVE: it does not call, get called by,
# or alter a single character of build_atomic_facts_block()'s own output --
# the prompt text the generation model sees is completely unchanged by this
# section existing. Consumed by grounding_check.py's always-on entity/value
# co-occurrence check, not by answer.py's prompt-building path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicFact:
    """One structured entity/state/category + metric + period + value
    relationship, deterministically parsed from Source text -- the
    authoritative representation requirement A ("Atomic Facts are
    authoritative for covered numeric facts") refers to. `kind` is one of
    "distributor", "state", or "category", matching the three sections
    build_atomic_facts_block() renders. `entity` is set only for
    kind="distributor" (a distributor name); `state` is set for all three
    kinds -- a distributor-deviation sentence never repeats its own state
    name, but every source document opens with its own "State: X" header,
    which _distributor_atomic_facts_from_source_text() reads once per
    source row and attaches to every distributor fact from that row (see
    that function's docstring for the live ranking-validator bug this
    closes: without it, a same-metric distributor fact from a completely
    unrelated retrieved state/period could be silently compared against
    the question's actual target state)."""

    kind: str
    entity: str | None
    state: str | None
    category: str | None
    metric: str
    period: str
    value: float
    average: float | None
    gap: str | None
    source_id: str

    def numeric_fields(self) -> list[float]:
        """Every number this fact actually carries -- its own value, its
        average (when present), and the leading number of its gap text
        (when present and parseable) -- i.e. every number that legitimately
        belongs to THIS fact, for checking whether a claimed figure belongs
        here or was pulled in from a neighboring fact."""
        out = [self.value]
        if self.average is not None:
            out.append(self.average)
        if self.gap:
            m = re.match(r"[\d.]+", self.gap)
            if m:
                try:
                    out.append(float(m.group(0)))
                except ValueError:
                    pass
        return out


def _parse_atomic_value(raw: str | None) -> float | None:
    """'20.0%' / '214.48' / '1,234.5' -> float. None (not raised) for
    anything unparseable -- a fact whose own value can't be parsed is
    simply skipped by its caller rather than crashing extraction."""
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _distributor_atomic_facts_from_source_text(source_id: str, text: str) -> list[AtomicFact]:
    """Live ranking-validator investigation fix: a distributor-deviation
    sentence never repeats its own state name (build_documents.py's
    template is just "Distributor X showed a significant deviation on
    ... in <period>: ...", no state), so `state` was always left None here
    -- which let a ranking comparison (grounding_check.py's
    verify_ranking_claim(), scoped only by metric+category) silently pull
    in a same-named-metric distributor fact from a COMPLETELY UNRELATED
    state/period's retrieved source and rank it alongside the question's
    actual target state (confirmed live: a Goa/June-2025 Dropsize ranking
    question pulled in "Tate-Gonzalez Distributors", whose real Dropsize
    fact is Meghalaya/July 2026). Every document DOES open with its own
    "State: X\\nPeriod: Y" header (the same one _STATE_HEADER_RE already
    reads for the state-KPI section) -- reusing it here means every
    distributor fact now carries the state its OWN source document is
    actually about, at zero extra parsing cost (one header match per
    source row, shared across every distributor fact from that row)."""
    header_match = _STATE_HEADER_RE.search(text)
    state = header_match.group("state").strip() if header_match else None

    facts: list[AtomicFact] = []
    for match in _DEVIATION_SENTENCE_RE.finditer(text):
        value = _parse_atomic_value(match.group("value"))
        if value is None:
            continue
        facts.append(
            AtomicFact(
                kind="distributor",
                entity=match.group("distributor").strip(),
                state=state,
                category=match.group("category"),
                metric=match.group("metric").strip(),
                period=match.group("period").strip(),
                value=value,
                average=_parse_atomic_value(match.group("average")),
                gap=match.group("gap"),
                source_id=source_id,
            )
        )
    return facts


def _state_atomic_facts_from_source_text(source_id: str, text: str) -> list[AtomicFact]:
    header_match = _STATE_HEADER_RE.search(text)
    if not header_match:
        return []
    state = header_match.group("state").strip()
    period = header_match.group("period").strip()
    if not state or not period:
        return []
    facts: list[AtomicFact] = []
    for name in _STATE_METRIC_NAMES:
        pattern = (
            rf"{re.escape(name)}(?:\s*\([^)]*\))?\s+for\s+{re.escape(state)}\s+in\s+"
            rf"{re.escape(period)}\s+was\s+([\d,]+\.?\d*%?)"
        )
        m = re.search(pattern, text)
        if not m:
            continue
        value = _parse_atomic_value(m.group(1))
        if value is None:
            continue
        facts.append(
            AtomicFact(
                kind="state", entity=None, state=state, category=None,
                metric=name, period=period, value=value, average=None, gap=None,
                source_id=source_id,
            )
        )
    return facts


def _category_atomic_facts_from_source_text(source_id: str, text: str) -> list[AtomicFact]:
    facts: list[AtomicFact] = []
    for match in _CATEGORY_BLOCK_RE.finditer(text):
        category = match.group("category")
        state = match.group("state").strip()
        period = match.group("period").strip()
        for group_name, metric_label in _CATEGORY_METRIC_FIELDS:
            value = _parse_atomic_value(match.group(group_name))
            if value is None:
                continue
            facts.append(
                AtomicFact(
                    kind="category", entity=None, state=state, category=category,
                    metric=metric_label, period=period, value=value, average=None, gap=None,
                    source_id=source_id,
                )
            )
    return facts


def extract_atomic_facts(context_records: dict[str, "pd.DataFrame"]) -> list["AtomicFact"]:
    """All Atomic Facts (distributor + state + category + sku) as
    structured data -- see the module comment above for why this exists
    alongside build_atomic_facts_block(). Returns [] (never raises) under
    the exact same conditions build_atomic_facts_block() returns "" for."""
    df = context_records.get("sources") if context_records else None
    if df is None or df.empty or "text" not in df.columns or "id" not in df.columns:
        return []

    facts: list[AtomicFact] = []
    for _, row in df.iterrows():
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        source_id = str(row.get("id"))
        facts.extend(_distributor_atomic_facts_from_source_text(source_id, text))
        facts.extend(_state_atomic_facts_from_source_text(source_id, text))
        facts.extend(_category_atomic_facts_from_source_text(source_id, text))
        facts.extend(_sku_atomic_facts_from_source_text(source_id, text))
    return facts


# ---------------------------------------------------------------------------
# SKU-level KPI atomic facts (Problem 1) -- additive extension, entirely
# separate from the three sections above (none of their regexes, output, or
# numbering is touched by this section).
#
# WHAT THIS SECTION IS FOR (plain language):
# src/inference/sku_evidence.py resolves SKU-level questions to real
# evidence by rendering src/kpis/compute_kpis.py's kpi_state_month_sku.csv
# rows through render_sku_evidence_document() -- ONE fixed sentence per SKU,
# self-contained (names its own state/period, exactly like the category
# block above), appended into context_records["sources"] by pipeline.py
# alongside whatever GraphRAG itself retrieved. This section parses that
# SAME fixed sentence back into AtomicFacts, exactly the way the other
# three sections parse build_documents.py's fixed sentences -- so the
# ranking/grounding machinery downstream (grounding_check.py's
# verify_sku_ranking_claim()) works over real, structured SKU data the same
# way it already works over distributor/state/category data, with no
# separate code path. _SKU_FACT_RE mirrors render_sku_evidence_document()'s
# exact wording -- see that function's docstring for why the two are kept
# in sync by convention rather than by sharing an f-string (matching every
# other section's established "grammar, not code" convention in this file).
# ---------------------------------------------------------------------------

_SKU_FACT_RE = re.compile(
    r"SKU (?P<sku_id>SKU\d+) \((?P<sku_name>[^,]+), (?P<franchise>[^,]+) franchise, "
    r"(?P<category>[A-Za-z]+) category\) in (?P<state>.+?) during (?P<period>.+?): "
    r"(?P<units>[\d,]+) units delivered, revenue of Rs (?P<revenue>[\d,]+), "
    r"Service Level (?P<service_level>[\d.]+)%, Numeric Distribution (?P<nd>[\d.]+)%, "
    r"Out-of-Stock rate (?P<oos>[\d.]+)%\."
)

_SKU_FACTS_HEADER = "-----Atomic SKU Facts-----"

# One AtomicFact per METRIC per matched SKU sentence (five per sentence),
# mirroring _CATEGORY_METRIC_FIELDS' "one fact per metric" shape -- lets
# verify_sku_ranking_claim() (grounding_check.py) rank by any one of these
# independently (Units Delivered, Revenue, Service Level, Numeric
# Distribution, Out-of-Stock Rate) without needing a composite score this
# project has never defined.
#
# Deliberately public (no leading underscore), unlike this file's other
# *_FIELDS constants: this is the one canonical list of "which SKU metric
# names are real" in the whole project, so query_requirements.py's ambiguous-
# ranking clarification gate imports it directly instead of re-deriving its
# own copy of the same five names (which would risk drifting out of sync
# with this file's actual parsing grammar).
SKU_METRIC_FIELDS = (
    ("units", "Units Delivered"),
    ("revenue", "Revenue"),
    ("service_level", "Service Level"),
    ("nd", "Numeric Distribution"),
    ("oos", "Out-of-Stock Rate"),
)

# Common phrasings that mean "ranked by sales volume" without literally
# naming a metric -- "most selling", "best-selling", "top-selling", "sold
# the most". All map to "Units Delivered", never "Revenue" -- "selling"/
# "sold" is a volume word, not a value word, and this project has no
# business default that would make "top selling" mean revenue instead.
# Deliberately narrow: a bare "top SKUs" or "best-performing SKUs" (no
# selling/sold word) must NOT match this and must still be asked about --
# see resolve_sku_ranking_metric_synonym()'s docstring.
_SELLING_SYNONYM_RE = re.compile(
    r"\b(?:top[- ]selling|best[- ]selling|most[- ]selling|sold\s+the\s+most|selling\s+the\s+most)\b",
    re.IGNORECASE,
)


def resolve_sku_ranking_metric_synonym(question: str) -> str | None:
    """'top-selling'/'most selling'/'sold the most' -> 'Units Delivered'.
    These phrases don't literally contain any of SKU_METRIC_FIELDS' exact
    labels, but unambiguously mean sales volume -- unlike a genuinely
    ambiguous ranking phrase ("top SKUs", "best-performing SKUs") that
    names no metric or volume/value word at all and must still trigger a
    clarification request rather than being guessed.

    Shared by query_requirements.py (the ambiguous-ranking clarification
    gate) and grounding_check.py (post-hoc ranking verification) so both
    recognize the exact same phrasing as unambiguous, rather than risking
    the two drifting out of sync with independently-maintained regexes."""
    return "Units Delivered" if _SELLING_SYNONYM_RE.search(question) else None


def _sku_metric_facts_from_source_text(source_id: str, text: str) -> list[str]:
    """Prompt-text rendering, mirroring _category_metric_facts_from_source_text()."""
    facts: list[str] = []
    for match in _SKU_FACT_RE.finditer(text):
        sku_id = match.group("sku_id")
        sku_name = match.group("sku_name")
        franchise = match.group("franchise")
        category = match.group("category")
        state = match.group("state").strip()
        period = match.group("period").strip()
        for group_name, metric_label in SKU_METRIC_FIELDS:
            facts.append(
                "SKU: {sku_id}\n"
                "SKU Name: {sku_name}\n"
                "Franchise: {franchise}\n"
                "Category: {category}\n"
                "State: {state}\n"
                "Metric: {metric}\n"
                "Period: {period}\n"
                "Value: {value}\n"
                "Source: Sources ({source_id})".format(
                    sku_id=sku_id, sku_name=sku_name, franchise=franchise,
                    category=category, state=state, metric=metric_label,
                    period=period, value=match.group(group_name), source_id=source_id,
                )
            )
    return facts


# ---------------------------------------------------------------------------
# Deterministic SKU Ranking block -- additive, entirely separate from the
# four Atomic Facts sections above (2026-08-23 stabilization pass, Design
# Decision: the LLM must never be asked to compute a top-N/bottom-N SKU
# ranking itself).
#
# WHAT THIS SECTION IS FOR (plain language):
# sku_evidence.py's evaluate_sku_evidence() already computes the correct
# ranking deterministically (sort real rows, take N) whenever a SKU
# question is both ranking-shaped AND names a real metric+direction, and
# renders it as one extra synthetic Sources row per resolved period, id-
# prefixed "sku-ranking-" (see sku_evidence._build_ranking_source_row()).
# This function reads exactly those rows -- by id PREFIX, never by
# regex-matching their text shape -- and renders them into their own
# labeled prompt section, kept separate from "-----Atomic SKU Facts-----"
# so the model sees an explicit, authoritative, already-sorted answer
# instead of having to derive one from the full unordered SKU fact list.
# ---------------------------------------------------------------------------

_RANKING_ROW_ID_PREFIX = "sku-ranking-"
_RANKING_BLOCK_HEADER = "-----Deterministic SKU Ranking (Authoritative -- do not recompute)-----"


def build_deterministic_ranking_block(context_records: dict[str, "pd.DataFrame"]) -> str:
    """Every "sku-ranking-"-id-prefixed row in context_records["sources"],
    concatenated under one labeled header -- "" (never an error) when none
    exist, i.e. every question except an unambiguous SKU ranking question
    with real evidence resolved. Purely additive: does not read, alter, or
    get read by anything the other build_atomic_facts_block() sections
    already do -- a "sku-ranking-" row's text is deliberately NOT in the
    _SKU_FACT_RE sentence shape, so it never also becomes a duplicate,
    unordered "Atomic SKU Facts" entry."""
    df = context_records.get("sources") if context_records else None
    if df is None or df.empty or "id" not in df.columns or "text" not in df.columns:
        return ""
    blocks: list[str] = []
    for _, row in df.iterrows():
        row_id = str(row.get("id", ""))
        if not row_id.startswith(_RANKING_ROW_ID_PREFIX):
            continue
        text = row.get("text")
        if isinstance(text, str) and text:
            blocks.append(text.strip())
    if not blocks:
        return ""
    return _RANKING_BLOCK_HEADER + "\n\n" + "\n\n".join(blocks)


def _sku_atomic_facts_from_source_text(source_id: str, text: str) -> list[AtomicFact]:
    """Structured AtomicFact rendering, mirroring
    _category_atomic_facts_from_source_text(). `entity` is the SKU's
    human-readable name (not its bare sku_id) -- matching how every other
    kind's `entity` is whatever name would actually appear in generated
    prose (a distributor's display name, never an internal id), since
    that's what grounding_check.py's entity-mention scanning matches
    against answer text."""
    facts: list[AtomicFact] = []
    for match in _SKU_FACT_RE.finditer(text):
        state = match.group("state").strip()
        category = match.group("category")
        period = match.group("period").strip()
        entity = match.group("sku_name").strip()
        for group_name, metric_label in SKU_METRIC_FIELDS:
            value = _parse_atomic_value(match.group(group_name))
            if value is None:
                continue
            facts.append(
                AtomicFact(
                    kind="sku", entity=entity, state=state, category=category,
                    metric=metric_label, period=period, value=value,
                    average=None, gap=None, source_id=source_id,
                )
            )
    return facts
