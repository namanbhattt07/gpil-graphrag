"""
Phase 7, Stage 2 -- claim-level grounding / contradiction check.

WHAT THIS FILE IS FOR (plain language):
This is the guard that runs AFTER the answer-generation model has drafted a
response, and checks it against the same retrieved evidence it was
supposed to be grounded in.

WHY THIS IS CLAIM-LEVEL, NOT WORD-LEVEL:
The original version of this checker asked one question per flagged word:
"does this word appear ANYWHERE in the retrieved context?" That caught the
first live failure ("alarming"/"concerning" appearing nowhere at all), but
a second live test exposed the real gap: the model wrote "...supply chain
inefficiencies may have collectively influenced Bihar's overall service
level" and the word "inefficiencies" DID appear somewhere in the retrieved
context -- just in a community report about a completely different month's
candy out-of-stock dynamics, nothing to do with Bihar's Service Level. A
whole-context word search can't tell those apart.

The fix: every structured claim the model reports (see answer.py's
claim-extraction prompt) carries its OWN `citations` -- the specific
entities/sources/reports it's actually citing. validate_claim() resolves
those citations to the corresponding context_records rows and checks the
claim ONLY against that scoped text, never the whole blob. A claim that
cites nothing at all gets checked against the whole blob as an explicit,
visible fallback (flagged in its detail message), not a silent one.

A claim that DOES cite something, but that something doesn't resolve to
any known record (e.g. the model wrote "Atomic SKU Facts (4)" instead of a
real "Sources (21)" id -- see answer.py's _ATOMIC_FACTS_CITATION_GUIDANCE,
added after this exact failure mode was observed live), is a DIFFERENT,
worse case than citing nothing: it looks like grounded provenance but
isn't. validate_claim() treats it as unresolved and fails closed (an
"unresolved_citation" issue) rather than silently falling back to the
whole-blob check the way "cited nothing" does -- a whole-blob pass here
would let an invented-looking citation still slip through as "grounded,"
which is exactly the source-traceability gap this distinction closes.

THE ORIGINAL WORD/DIRECTION SCANNERS STILL RUN, AS A SAFETY NET:
scan_qualitative_language(), scan_causal_language(), and
check_direction_consistency() still run over the full answer text
unscoped-by-claim -- they catch anything the model didn't self-report as a
structured claim. They're citation-scoped per-sentence where a sentence
has its own [Data: ...] tag (same fix as above, applied to prose), and
fall back to whole-blob only when a sentence cites nothing. Every issue
carries `source="claim"` or `source="prose_fallback"` so it's visible
which mechanism caught what.

WHY CAUSAL CLAIMS ARE THE STRICTEST CATEGORY:
Phase 4's source documents (src/graph/build_documents.py) are pure
fact-listing -- they never say "X caused Y." A causal claim ("contributed
to", "led to", "due to", "resulted in", etc.) only passes if the CITED
evidence itself contains an explicit causal marker, not just a
co-occurring fact. In practice this means most causal language the model
generates will correctly fail and get hedged on retry ("coincided with"
instead of "caused") -- that's intentional, not a bug: this corpus never
actually establishes causation, so claiming it always should fail.

Still no LLM-escalation path -- everything here is deterministic string/
regex/arithmetic work over data already in memory. No second retrieval,
no second LLM call for grounding itself.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from src.inference.fact_structuring import (
    AtomicFact,
    extract_atomic_facts,
    resolve_sku_ranking_metric_synonym,
    sku_atomic_fact_texts,
)
from src.inference.knowledge_layer import GPIL_GLOSSARY
from src.inference.premise_check import extract_period_from_question, extract_state_from_question
from src.inference.schemas import (
    AnswerClaim,
    GroundingCheckResult,
    GroundingIssue,
    MetricComparison,
    SourceCitation,
)

# ---------------------------------------------------------------------------
# Qualitative word families -- curated, not a generic stemmer, per the
# explicit "no fuzzy semantic matching" constraint. Each entry is a real
# surface form; _stem() below adds a SMALL amount of safe suffix-stripping
# on top so novel inflections of these same words (not arbitrary words)
# still match.
# ---------------------------------------------------------------------------
QUALITATIVE_FLAGS = {
    "alarm", "alarms", "alarmed", "alarming", "alarmingly",
    "concern", "concerns", "concerning", "concerned",
    "worry", "worries", "worried", "worrying", "worrisome",
    "critical", "critically",
    "problem", "problems", "problematic",
    "trouble", "troubled", "troubling",
    "inefficient", "inefficiency", "inefficiencies", "inefficiently",
    "severe", "severely",
    "unacceptable", "unacceptably",
    "dire", "direly",
    "red flag",
}

UP_WORDS = {
    "climbed", "climbing", "climb", "climbs", "rose", "rising", "risen", "rise", "rises",
    "increased", "increasing", "increase", "increases",
    "grew", "growing", "grown", "grow", "growth", "grows",
    "gained", "gaining", "gain", "gains",
    "improved", "improving", "improvement", "improve", "improves",
    "higher", "up", "surged", "surging", "surge", "surges",
    "strengthened", "strengthening", "strengthen", "strengthens",
    "enhanced", "enhancing", "enhance", "enhances",
    "boosted", "boosting", "boost", "boosts",
    "raised", "raising", "raise", "raises",
    "recovered", "recovering", "recover", "recovers", "recovery",
    "restored", "restoring", "restore", "restores",
    "expanded", "expanding", "expand", "expands",
}
DOWN_WORDS = {
    "fell", "falling", "fallen", "fall", "falls",
    "dropped", "dropping", "drop", "drops",
    "declined", "declining", "decline", "declines",
    "decreased", "decreasing", "decrease", "decreases",
    "worsened", "worsening", "worsen", "worsens",
    "lower", "down", "slipped", "slipping", "slip",
    "dipped", "dipping", "dip", "reduction", "reduced",
    "deteriorated", "deteriorating", "deteriorate", "deteriorates",
    "reduce", "reduces", "reducing",
    "shrank", "shrunk", "shrinking", "shrink", "shrinks",
    "contracted", "contracting", "contract", "contracts",
}

# Neutral/future-oriented action words: NOT directional on their own (a
# recommendation to "monitor" or "address" a metric says nothing about
# which way it moved historically), but their presence is one of the
# signals _is_future_or_prescriptive() uses to recognize a forward-looking
# recommendation sentence rather than a historical observation.
RECOMMENDATION_ACTION_WORDS = {
    "maintain", "maintains", "maintaining", "maintained",
    "stabilize", "stabilizes", "stabilizing", "stabilized",
    "prevent", "prevents", "preventing", "prevented",
    "avoid", "avoids", "avoiding", "avoided",
    "mitigate", "mitigates", "mitigating", "mitigated",
    "address", "addresses", "addressing", "addressed",
    "monitor", "monitors", "monitoring", "monitored",
    "manage", "manages", "managing", "managed",
    "control", "controls", "controlling", "controlled",
    "optimize", "optimizes", "optimizing", "optimized",
    "consider", "considers", "considering", "considered",
    "recommend", "recommends", "recommending", "recommended",
    "continue", "continues", "continuing", "continued",
    "track", "tracks", "tracking", "tracked",
    "review", "reviews", "reviewing", "reviewed",
    "assess", "assesses", "assessing", "assessed",
    "watch", "watches", "watching", "watched",
}

# Modal/auxiliary verbs that mark a clause as prescriptive or predictive
# rather than a statement of historical fact ("Service Level should
# improve" vs. "Service Level improved").
_MODAL_FUTURE_RE = re.compile(
    r"\b(should|could|might|must|will|would|need(?:s)?\s+to|ought\s+to)\b", re.IGNORECASE
)

# Explicit forward-looking time phrases -- these alone are enough to mark a
# sentence as being about the future, regardless of which verb it uses.
_FUTURE_PHRASE_RE = re.compile(
    r"\b(going forward|moving forward|in the future|in future|"
    r"next (?:period|month|quarter|year)|over the coming|upcoming|"
    r"henceforth|from now on)\b",
    re.IGNORECASE,
)

_CAUSAL_PATTERN_RE = re.compile(
    r"\b(caus\w*|contribut\w*\s+to|due to|led to|leads?\s+to|"
    r"result(?:s|ed|ing)?\s+(?:in|from)|attribut\w*\s+to|stemm?\w*\s+from|"
    r"driven by|because of|as a result of|responsible for)\b",
    re.IGNORECASE,
)

_SUFFIXES = ("ically", "ingly", "edly", "ness", "ment", "ing", "ied", "ies", "ed", "es", "ally", "ly", "s")


def _stem(word: str) -> str:
    """Strip at most one suffix from a small fixed list -- deterministic,
    not a real lemmatizer. Only used to compare against the curated
    QUALITATIVE_FLAGS set, never for open-ended semantic matching."""
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_QUALITATIVE_STEMS = {_stem(term) for term in QUALITATIVE_FLAGS if " " not in term}


def _is_causal_text(text: str) -> bool:
    return bool(_CAUSAL_PATTERN_RE.search(text))


def _qualitative_tokens_in(text: str) -> set[str]:
    """Every word/phrase in `text` that belongs to a QUALITATIVE_FLAGS
    word-family (exact match or safe stem match) -- presence only, no
    judgment about whether it's supported."""
    found: set[str] = set()
    lower = text.lower()
    for term in QUALITATIVE_FLAGS:
        if " " in term and term in lower:
            found.add(term)
    for token in re.findall(r"[a-zA-Z]+", text):
        lower_token = token.lower()
        if lower_token in QUALITATIVE_FLAGS or _stem(lower_token) in _QUALITATIVE_STEMS:
            found.add(lower_token)
    return found


def _find_unsupported_qualitative_terms(text: str, evidence_scope: str) -> list[str]:
    """Qualitative terms in `text` that are NOT backed by `evidence_scope`
    -- either the exact word/stem-family appears there, or the sentence
    and the evidence both explicitly say 'deviation' (the one built-in
    override, since the project's own document template uses that word
    for real, quantified anomalies)."""
    evidence_lower = evidence_scope.lower()
    evidence_stems = {_stem(t) for t in re.findall(r"[a-zA-Z]+", evidence_lower)}
    lower_text = text.lower()
    unsupported = []
    for term in _qualitative_tokens_in(text):
        if " " in term:
            supported = term in evidence_lower
        else:
            supported = _stem(term) in evidence_stems
        if not supported and not ("deviation" in lower_text and "deviation" in evidence_lower):
            unsupported.append(term)
    return sorted(set(unsupported))


def _infer_direction_word(text: str) -> str | None:
    """Return 'up'/'down' if `text` unambiguously uses one direction's
    vocabulary, else None (no direction word, or genuinely mixed). Works
    on a full sentence or a short phrase like an AnswerClaim.direction
    value -- both are just token sets."""
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    has_up = bool(tokens & UP_WORDS)
    has_down = bool(tokens & DOWN_WORDS)
    if has_up and not has_down:
        return "up"
    if has_down and not has_up:
        return "down"
    return None


def _is_future_or_prescriptive(text: str) -> bool:
    """True if `text` reads as a future/prescriptive statement (a
    recommendation or prediction) rather than a report of something the
    data already shows happened.

    WHY THIS EXISTS: a live failure found the deterministic direction
    scanner treating "Service Level should improve going forward" as a
    historical claim that Service Level went UP -- 'improve' is in
    UP_WORDS regardless of tense. The general problem isn't the word
    'improve' specifically: ANY up/down word (increase, reduce, decline,
    strengthen, ...) can appear in a recommendation sentence without
    asserting anything about the past. This function is the shared,
    word-agnostic fix: it looks for the linguistic markers of future/
    prescriptive framing (modal verbs, explicit future-time phrases, or
    neutral recommendation-action verbs like 'monitor'/'address') and, if
    found, callers treat the sentence as NOT making a historical claim --
    regardless of which specific directional word it also contains.

    Deliberately conservative in one direction only: per the "prefer
    avoiding a false contradiction over incorrectly rejecting a grounded
    answer" instruction, presence of ANY one signal is enough to call a
    sentence future/prescriptive (no requirement to combine multiple
    markers) -- some genuinely historical sentences that happen to also
    contain a recommendation-shaped word (rare) will be under-checked as a
    result, which is the intentionally-chosen tradeoff, not an oversight.
    """
    if _MODAL_FUTURE_RE.search(text):
        return True
    if _FUTURE_PHRASE_RE.search(text):
        return True
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & RECOMMENDATION_ACTION_WORDS)


# Excludes "vs." from sentence-boundary detection: the corpus's standard
# deviation-sentence template ("82.6% vs. the state average of 92.9%, a gap
# of 10.4 percentage points.") uses "vs." as an abbreviation, not a sentence
# end. Without this, a bare period-based splitter treats "vs." as ending the
# sentence, severing a distributor's own value from its state-average/gap
# figures within what is really one fact-sentence -- which broke the
# entity+value+period co-occurrence check's ability to catch a comparison
# value silently swapped in from a different entity's sentence in the same
# cited record. `(?<!vs\.)` is a fixed-width (3-char) negative lookbehind,
# so this stays a single regex, no NLP tokenizer needed.
_SENTENCE_SPLIT_RE = re.compile(r"(?<!vs\.)(?<=[.!?])\s+", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter -- good enough for scanning, not meant to be
    a real NLP sentence tokenizer."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Evidence index + citation resolution -- the core of claim-level scoping.
# ---------------------------------------------------------------------------

_DATASET_TEXT_COLUMNS = {
    "entities": ["entity", "description"],
    "relationships": ["source", "target", "description"],
    "reports": ["title", "content"],
    "sources": ["text"],
    "claims": ["description"],
}
_DATASET_NAME_ALIASES = {
    "entity": "entities", "entities": "entities",
    "relationship": "relationships", "relationships": "relationships",
    "report": "reports", "reports": "reports",
    "source": "sources", "sources": "sources",
    "claim": "claims", "claims": "claims",
}
_CITATION_GROUP_RE = re.compile(r"([A-Za-z][A-Za-z ]*?)\s*\(([^)]*)\)")

# Live-observed SKU-ranking-question failure (Problem: model cites the
# Atomic SKU Facts section's own FACT NUMBER instead of the "Source:" id
# _ATOMIC_FACTS_CITATION_GUIDANCE (answer.py) tells it to use): across a
# 15-question validation pass, the model NEVER once produced a resolvable
# citation for a SKU claim -- it wrote "Atomic SKU Facts (143)", "SKU Facts
# (1)", "SKU Fact (201)", and plain "SKU (201)", four different spellings
# of the same wrong idea, all failing closed under the unresolved_citation
# fix above and making every SKU ranking/factual answer un-groundable.
# Unlike the unresolved_citation fix (which correctly rejects a citation
# that resolves to NOTHING), this is a citation that resolves to something
# very SPECIFIC and recoverable: every fact in the "Atomic SKU Facts"
# section comes from exactly the SKU evidence rows sku_evidence.py injects
# (id-prefixed "sku-"), so a citation shaped like this can be resolved to
# that SAME precisely-scoped evidence -- not the whole retrieved blob --
# rather than discarded. Matches the four spellings above (and close
# variants) by normalizing away spaces/case, not by literal string list,
# since the model's exact wording isn't fully predictable.
_SKU_ATOMIC_FACTS_NAME_ALIASES = frozenset({"sku", "skufact", "skufacts", "atomicskufact", "atomicskufacts"})


def _normalize_citation_name(name: str) -> str:
    """'Atomic SKU Facts' -> 'atomicskufacts' -- strips everything but
    letters and lowercases, so citation-group name matching is robust to
    the model's exact spacing/capitalization without needing a literal
    string list for every spelling it might produce."""
    return re.sub(r"[^a-z]", "", name.lower())


def _ensure_sentence_boundary(text: str) -> str:
    """Guarantee `text` ends with a sentence-terminating character (./!/?)
    so that concatenating this record's text with a DIFFERENT record's text
    (see _resolve_citations, _all_evidence_text) always creates a real
    sentence boundary for _split_sentences() to find -- two different cited
    records can never silently fuse into one artificial "sentence".

    WHY THIS EXISTS: a live failure (Q7) showed relationships.parquet /
    entities.parquet description text has NO trailing punctuation at all in
    the overwhelming majority of rows (e.g. "HAS_OBSERVATION: Campbell PLC
    Distributors has a measured GPI Out-of-Stock Rate fact for April
    2026") -- measured at 99.8% of relationship rows in the real pilot
    index. When a claim cited two such records together, the naive
    " ".join(parts) in _resolve_citations concatenated them with nothing
    but a bare space between them. _split_sentences (which only splits on
    "./!/?" + whitespace) then read the whole multi-record blob as ONE
    "sentence", letting one record's entity co-occur with a completely
    different record's value/metric in the eyes of
    _primary_fact_supported()'s co-occurrence check -- exactly how a real
    Bihar/Gujarat-style claim could cite Baxter's own (unrelated) record
    alongside Campbell's real GPI Out-of-Stock Rate record and have the
    checker treat "Baxter ... 20.0% ... Out-of-Stock Rate" as one
    (fabricated) fact.

    Applied once, at the per-record level in _build_evidence_index, so
    every downstream consumer of evidence_index (_resolve_citations,
    _all_evidence_text) gets the guarantee automatically -- never removes
    or alters a record's own content, only appends a period when one isn't
    already present.
    """
    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?":
        return stripped + "."
    return stripped


_SKU_ATOMIC_FACTS_PSEUDO_DATASET = "_sku_atomic_facts"


def _build_evidence_index(context_records: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """dataset name -> {record id (as string): concatenated text of that
    ONE record}. This is what lets a claim be checked against just the
    handful of records it actually cites, instead of the whole context.

    Also builds one internal pseudo-dataset, keyed
    _SKU_ATOMIC_FACTS_PSEUDO_DATASET (leading underscore -- excluded from
    _all_evidence_text()'s whole-blob fallback, since its content is
    already a re-slicing of the real "sources" rows, not new evidence):
    {fact number (as string): that ONE SKU Atomic Fact's own text}, in the
    exact same 1-based numbering fact_structuring.build_atomic_facts_block()
    shows the model as "FACT N" in the "-----Atomic SKU Facts-----"
    section. This is what lets _resolve_citations() resolve an "Atomic SKU
    Facts (N)"-shaped citation to the ONE specific fact it names, not the
    whole multi-SKU source blob that fact came from -- see
    sku_atomic_fact_texts()'s docstring for the live failure this fixes."""
    index: dict[str, dict[str, str]] = {}
    for dataset, text_cols in _DATASET_TEXT_COLUMNS.items():
        df = context_records.get(dataset)
        if df is None or df.empty or "id" not in df.columns:
            continue
        table: dict[str, str] = {}
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in text_cols if c in df.columns and pd.notna(row.get(c))]
            table[str(row["id"])] = _ensure_sentence_boundary(" ".join(parts))
        index[dataset] = table

    sku_facts = sku_atomic_fact_texts(context_records)
    if sku_facts:
        index[_SKU_ATOMIC_FACTS_PSEUDO_DATASET] = {
            str(i): _ensure_sentence_boundary(fact) for i, fact in enumerate(sku_facts, start=1)
        }
    return index


def _all_evidence_text(evidence_index: dict[str, dict[str, str]]) -> str:
    """Every record's text, concatenated -- the pre-claim-level fallback
    scope, used only when a claim/sentence cites nothing at all. Skips the
    _SKU_ATOMIC_FACTS_PSEUDO_DATASET entry -- its content is already
    included via the real "sources" table's sku-* rows, so including both
    would just duplicate the same text, not add new evidence."""
    parts: list[str] = []
    for dataset, table in evidence_index.items():
        if dataset == _SKU_ATOMIC_FACTS_PSEUDO_DATASET:
            continue
        parts.extend(table.values())
    return " ".join(parts)


def _resolve_citations(citations: list[str], evidence_index: dict[str, dict[str, str]]) -> str:
    """'Entities (4, 6)' + evidence_index -> the concatenated text of
    JUST entities 4 and 6. Unknown dataset names or missing ids are
    silently skipped (not every citation the model writes will resolve
    cleanly) rather than raising -- a claim with no resolvable citation
    text ends up with scope_text="", which callers treat as "cites
    nothing" and fall back to the whole blob, visibly (see validate_claim).

    One recognized exception: a citation group whose name normalizes to
    one of _SKU_ATOMIC_FACTS_NAME_ALIASES ("Atomic SKU Facts", "SKU
    Facts", "SKU Fact", or bare "SKU", each followed by a fact number) is
    resolved against _SKU_ATOMIC_FACTS_PSEUDO_DATASET -- i.e. to the ONE
    specific SKU fact that number names, never the whole multi-SKU source
    blob it came from. This is the model citing the Atomic Facts
    section's own display number instead of its "Source:" field, a
    live-observed failure mode _ATOMIC_FACTS_CITATION_GUIDANCE (answer.py)
    tries to prevent but doesn't fully stop. Resolving to the whole SKU
    source blob instead (every fact in that section comes from the SAME
    one or two sku-* rows) would still satisfy validate_claim()'s
    sentence-scoped co-occurrence check, but attaches the WRONG per-claim
    evidence for source-traceability display (Problem 2) -- a live
    15-question validation pass caught exactly this: a correct,
    grounding-passed "Marlboro Pack 1" claim displayed a DIFFERENT SKU's
    sentence as its "evidence" purely because that SKU's fact rendered
    first in the shared blob. Resolving by exact fact number avoids that.

    A SECOND recognized exception (2026-08-23, SKU live validation pass):
    a citation string with NO "Name (ids)" shape at all -- e.g. the model
    wrote the bare id "sku-Gujarat-April_2026" instead of the correct
    "Sources (sku-Gujarat-April_2026)" -- is looked up directly against
    the "sources" table by EXACT id match. This is a live-observed
    formatting slip specific to this project's own synthetic Sources rows
    (sku-*/sku-ranking-*/glossary-* ids -- see sku_evidence.py/
    knowledge_layer.py), which sit right next to real prose in the prompt
    and apparently invite the model to drop the citation tag's own
    "Sources (...)" wrapper sometimes. Deliberately still a HARD exact-
    match against a REAL, already-known row id -- this never resolves a
    citation that doesn't already correspond to genuine evidence, so it
    tightens formatting tolerance without loosening what counts as
    'cited', the same standard every other resolution path in this
    function already holds to."""
    parts: list[str] = []
    for citation in citations:
        matched_a_group = False
        for name, ids in _CITATION_GROUP_RE.findall(citation):
            matched_a_group = True
            normalized_name = _normalize_citation_name(name)
            if normalized_name in _SKU_ATOMIC_FACTS_NAME_ALIASES:
                dataset = _SKU_ATOMIC_FACTS_PSEUDO_DATASET
            else:
                dataset = _DATASET_NAME_ALIASES.get(name.strip().lower())
                if dataset is None:
                    continue
            table = evidence_index.get(dataset, {})
            for record_id in ids.split(","):
                record_id = record_id.strip()
                if not record_id or record_id == "+more":
                    continue
                text = table.get(record_id)
                if text:
                    parts.append(text)
        if not matched_a_group:
            bare_id = citation.strip()
            text = evidence_index.get("sources", {}).get(bare_id)
            if text:
                parts.append(text)
    return " ".join(parts)


_DATA_TAG_RE = re.compile(r"\[Data:\s*[^\]]+\]")


def _extract_citations_from_sentence(sentence: str) -> list[str]:
    """Pull citation groups out of a sentence's own [Data: ...] tag, e.g.
    'Sales fell [Data: Sources (3); Entities (147)].' ->
    ['Sources (3)', 'Entities (147)']. Handles both comma- and semicolon-
    separated groups since the system prompt's own example uses both."""
    tag_match = re.search(r"\[Data:\s*([^\]]+)\]", sentence)
    if not tag_match:
        return []
    return [f"{name.strip()} ({ids.strip()})" for name, ids in _CITATION_GROUP_RE.findall(tag_match.group(1))]


def _strip_data_tag(sentence: str) -> str:
    """Remove a sentence's own [Data: ...] tag, e.g. for number-extraction
    that must not confuse a citation's record ids (e.g. the '147' in
    'Entities (147)') with a number the sentence is actually claiming."""
    return _DATA_TAG_RE.sub("", sentence)


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers_in(text: str) -> set[float]:
    out: set[float] = set()
    for m in _NUMBER_RE.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _value_in_evidence(value: float, evidence_text: str, tol: float = 0.05) -> bool:
    return any(abs(value - n) <= tol for n in _numbers_in(evidence_text))


def _mentions(evidence_text: str, term: str) -> bool:
    return term.strip().lower() in evidence_text.lower()


def _comparison_side_supported(
    scope_text: str, period: str, value: float, metric: str | None, tol: float = 0.05
) -> bool:
    """Phase 7c hardening: True only if some SINGLE sentence within the
    cited scope states `period` and a number within `tol` of `value`
    together (and `metric`, when given) -- i.e. the cited evidence
    actually pairs this period with this value as one fact, not just two
    unrelated mentions that both happen to appear somewhere across a
    multi-record citation blob.

    WHY THIS EXISTS: claim.value/claim.period and claim.comparison_value/
    claim.comparison_period were previously checked independently
    (_value_in_evidence(value, scope_text) and _mentions(scope_text,
    period) separately) -- both against the WHOLE concatenated cited
    scope. That misses a real gap: citing a source that mentions the
    right VALUE for an unrelated period, and a (possibly different)
    source that mentions the right PERIOD for an unrelated metric/value,
    would satisfy both independent checks despite the cited evidence
    never actually stating "this metric was this value in this period" as
    one fact. This function is sentence-scoped, not per-citation-record-
    scoped, so citing the comparison fact and the primary fact in two
    DIFFERENT records is still fine (per the "multiple cited sources are
    allowed" requirement) -- only cross-contamination (this period's
    string co-occurring with a DIFFERENT fact's value) is rejected.

    Deliberately additive: callers should keep the existing independent
    value/period checks too (they still catch the simple "not cited at
    all" cases with a clearer, more specific message) -- this only adds
    the co-occurrence check on top, so nothing that used to fail can now
    pass.
    """
    for sentence in _split_sentences(scope_text):
        if period.lower() not in sentence.lower():
            continue
        if not _value_in_evidence(value, sentence, tol=tol):
            continue
        if metric and metric.lower() not in sentence.lower():
            continue
        return True
    return False


def _metric_words_present(metric: str, sentence_lower: str) -> bool:
    """True if every whitespace-separated token of `metric` appears
    somewhere in `sentence_lower` -- not necessarily contiguous or in the
    same order. Source documents phrase category-scoped metrics as
    "[metric] for the [Category] category" (e.g. "Out-of-Stock Rate for the
    Ferrero category"), the reverse word order from a natural claim field
    like "Ferrero Out-of-Stock Rate" -- a contiguous substring match (as
    _comparison_side_supported uses for its own, differently-shaped period
    check) would wrongly reject a correct claim over word order alone.
    Splitting on whitespace only (not "-") keeps compound tokens like
    "out-of-stock" intact as one meaningful unit rather than fragmenting
    into generic single words that would match almost anything."""
    return all(word in sentence_lower for word in metric.lower().split())


def _primary_fact_supported(
    scope_text: str,
    entity: str,
    value: float,
    metric: str | None,
    tol: float = 0.05,
    extra_term: str | None = None,
    focus_categories: frozenset[str] | None = None,
    period: str | None = None,
) -> bool:
    """Same sentence-scoped co-occurrence pattern as _comparison_side_supported()
    above, applied to the PRIMARY claim (entity + value + metric) instead of
    the comparison side (period + value + metric). True only if some SINGLE
    sentence within scope_text states `entity` and a number within `tol` of
    `value` together (and `metric`, when given) -- i.e. the cited evidence
    actually attributes THIS value to THIS entity as one fact, not just two
    independently-true mentions (this entity named somewhere in scope, this
    value stated somewhere else in scope) that validate_claim()'s separate
    entity-mention and value-mention checks would each pass on their own.

    WHY THIS EXISTS: a live failure showed a claim citing evidence that
    correctly named the entity (a distributor) AND separately, correctly
    stated the claimed value -- just for a DIFFERENT category/metric than the
    one the claim actually named (a real Candy Out-of-Stock Rate figure
    restated as that distributor's Ferrero Out-of-Stock Rate). Both
    independent checks passed; nothing verified the three were ever stated
    TOGETHER. This closes that gap the same way Phase 7c closed the
    equivalent gap for comparison_value/comparison_period.

    `extra_term`, when given, must ALSO appear in the same sentence --
    used by validate_claim() for a decomposed state+category composite
    entity (see _decompose_state_category_entity()) so BOTH the state token
    and the category token are required together with the value/metric,
    not just the state token alone.

    `focus_categories`, when given (non-empty), rejects a sentence that
    names a DIFFERENT category than the ones the caller cares about, even
    though entity/value/metric all otherwise match -- see
    _sentence_categories()'s docstring for why this exists: a claim whose
    OWN metric field never named a category at all (e.g. plain "Out-of-
    Stock Rate", not "Ferrero Out-of-Stock Rate") would otherwise co-occur
    just as validly with a same-shaped sentence about a completely
    different category for the same distributor.

    `period`, when given, must ALSO appear in the same sentence -- used by
    validate_claim() for BOTH the PRIMARY claim (claim.period, Q7
    hardening) and the COMPARISON side of a cross-entity comparison claim
    (claim.comparison_period; see comparison_entity on AnswerClaim), where
    the second entity's figure may be from a different period than the
    primary entity's (e.g. Maharashtra's August 2024 figure vs. Sikkim's
    April 2025 figure) -- so entity+value+metric alone isn't enough to pin
    down the right period's fact. Omitted (None) whenever the claim doesn't
    set that period field at all, in which case this adds no requirement,
    preserving prior behavior exactly for period-less claims.

    Deliberately additive, exactly like _comparison_side_supported(): callers
    keep the existing independent value/entity checks too (clearer, more
    specific messages for the simple "not cited at all" cases) -- this only
    adds the co-occurrence check on top, so nothing that used to fail can now
    pass.
    """
    return (
        _find_primary_fact_sentence(scope_text, entity, value, metric, tol, extra_term, focus_categories, period)
        is not None
    )


def _find_primary_fact_sentence(
    scope_text: str,
    entity: str,
    value: float,
    metric: str | None,
    tol: float = 0.05,
    extra_term: str | None = None,
    focus_categories: frozenset[str] | None = None,
    period: str | None = None,
) -> str | None:
    """The actual matching sentence _primary_fact_supported() checks for
    (same parameters, same logic) -- factored out so a caller that needs
    the SENTENCE itself, not just whether one exists, can get it without
    re-implementing the search. Used by resolve_claim_sources() (Problem
    2) to narrow a claim's displayed provenance down to the ONE sentence
    that actually supports it, instead of the whole (possibly many-SKU)
    cited record -- see that function's docstring for the live failure
    this closes. Returns None on no match, exactly mirroring
    _primary_fact_supported()'s False."""
    for sentence in _split_sentences(scope_text):
        if entity.lower() not in sentence.lower():
            continue
        if extra_term and extra_term.lower() not in sentence.lower():
            continue
        if period and period.lower() not in sentence.lower():
            continue
        if not _value_in_evidence(value, sentence, tol=tol):
            continue
        if metric and not _metric_words_present(metric, sentence.lower()):
            continue
        if focus_categories:
            sentence_categories = _sentence_categories(sentence)
            if sentence_categories and not (sentence_categories & focus_categories):
                continue
        return sentence
    return None


# Category names this corpus uses (see src/graph/build_documents.py's
# CATEGORY_ORDER) -- duplicated here as a plain constant, not an import, so
# this module's only dependency stays on the fixed set of category names
# every document uses verbatim, not on the document-generation code itself.
_CATEGORY_NAMES = {"GPI", "IPM", "Ferrero", "Candy"}

_CATEGORY_NAMES_LOWER = {c.lower() for c in _CATEGORY_NAMES}
_CANONICAL_CATEGORY_BY_LOWER = {c.lower(): c for c in _CATEGORY_NAMES}

# Source documents mark a category-scoped fact one of two ways: "... for
# the Ferrero category ..." (distributor deviation bullets) or a bare
# "(Ferrero)" parenthetical (the "No distributor deviation this month for:
# ... Out-of-Stock Rate (Ferrero) ..." lines). Both are matched so a
# sentence's own stated category can be read back out deterministically.
_SENTENCE_CATEGORY_RE = re.compile(
    r"for the ([A-Za-z]+) category|\(([A-Za-z]+)\)", re.IGNORECASE
)


def _sentence_categories(sentence: str) -> frozenset[str]:
    """Every real category name (see _CATEGORY_NAMES) that `sentence`
    explicitly states itself, via either surface form above -- e.g. "...
    Out-of-Stock Rate for the Candy category in October 2025: 28.6% ..."
    -> {'Candy'}. Empty when the sentence names no category at all (a
    state-level, category-agnostic metric like Service Level or
    Productivity), which callers must treat as "no information", not "no
    match" -- see _primary_fact_supported()'s focus_categories usage.

    WHY THIS EXISTS: a live failure showed a distributor-deviation claim
    whose OWN metric field never named a category (just "Out-of-Stock
    Rate") pass the entity+value+metric co-occurrence check against a
    sentence that WAS category-scoped -- just for a different category than
    the one actually asked about. The claim's metric field alone can't
    catch that (it's silent on category); reading the category the cited
    SENTENCE itself declares, and comparing it against what the question
    asked about, closes the gap without requiring the model to always
    remember to qualify every metric field with its category.
    """
    found: set[str] = set()
    for match in _SENTENCE_CATEGORY_RE.finditer(sentence):
        word = (match.group(1) or match.group(2) or "").strip().lower()
        canonical = _CANONICAL_CATEGORY_BY_LOWER.get(word)
        if canonical:
            found.add(canonical)
    return frozenset(found)


def _extract_focus_categories(question: str | None) -> frozenset[str]:
    """Every real category name (see _CATEGORY_NAMES) mentioned by name in
    the ORIGINAL question -- e.g. "...Out-of-Stock Rate for the Ferrero
    category..." -> {'Ferrero'}. Empty for a question that never names a
    category (most questions -- state-level KPIs like Service Level or
    Dropsize have no category), in which case callers must skip the
    category-consistency check entirely (see _primary_fact_supported()),
    not treat "no category asked about" as "reject every category-scoped
    sentence"."""
    if not question:
        return frozenset()
    return frozenset(
        canonical
        for lower, canonical in _CANONICAL_CATEGORY_BY_LOWER.items()
        if re.search(rf"\b{re.escape(lower)}\b", question, re.IGNORECASE)
    )


# A model-invented state+category composite phrase -- e.g. "Bihar Ferrero"
# or "Bihar Ferrero Category" -- the source documents state a state and a
# category as two SEPARATE facts (a state-level paragraph, a category-level
# paragraph naming the state), never fused into one entity string, so
# neither shape ever appears verbatim anywhere in evidence even when the
# underlying state+category fact is genuinely well-supported. The trailing
# "Category" word is optional: a live run showed the model drop it and
# write just "Bihar Ferrero", not only the "Bihar Ferrero Category" shape
# from the original forensic failure -- both are the same underlying
# fabricated-composite-entity problem, just two surface spellings of it.
_COMPOSITE_CATEGORY_ENTITY_RE = re.compile(r"^(.+?)\s+([A-Za-z]+)(?:\s+Category)?$", re.IGNORECASE)


def _decompose_state_category_entity(entity: str) -> tuple[str, str] | None:
    """If `entity` matches the fabricated '<prefix> <Category>' or
    '<prefix> <Category> Category' shape AND the category word is one of
    this corpus's real category names, return (prefix, category) -- e.g.
    'Bihar Ferrero' -> ('Bihar', 'Ferrero'), 'Bihar Ferrero Category' ->
    ('Bihar', 'Ferrero'). Returns None for every other entity shape (plain
    state names, distributor names, or a last word that isn't a real
    category), so callers fall back to normal literal-entity handling
    unchanged for everything else -- in particular, a genuine two-word
    place name (e.g. "Andhra Pradesh") never matches, since none of this
    corpus's fixed category names ("GPI", "IPM", "Ferrero", "Candy")
    coincide with any real state/place name here.

    WHY THIS EXISTS: see _primary_fact_supported()'s docstring and the
    module-level Phase 8 regression notes -- a live failure had the model
    report entity="Bihar Ferrero Category" (and, in a later live run,
    the shorter "Bihar Ferrero") for a state+category-level fact. Treating
    either phrase as a literal entity name always fails (see above, no
    document ever spells it that way), even though the genuine
    Bihar+Ferrero fact is well-supported. This lets validate_claim() check
    the DECOMPOSED parts for co-occurrence with the value/metric instead --
    still requiring both parts together in one sentence (see
    _primary_fact_supported()'s extra_term), so a value that only pairs with
    one of the two parts (e.g. a Candy figure that mentions "Bihar" but not
    "Ferrero" together with that value) is still correctly rejected.
    """
    match = _COMPOSITE_CATEGORY_ENTITY_RE.match(entity.strip())
    if not match:
        return None
    # A live run also showed a hyphen/colon separator between the two parts
    # ("Bihar - Ferrero"), which the lazy prefix group swallows whole
    # (prefix="Bihar -") -- strip trailing separator punctuation so the
    # prefix is just the bare state name, matching how it actually appears
    # in evidence ("State: Bihar", not "Bihar -").
    prefix = re.sub(r"[\s\-–—:]+$", "", match.group(1)).strip()
    category = match.group(2).strip()
    if not prefix or category.lower() not in _CATEGORY_NAMES_LOWER:
        return None
    return prefix, category


def _resolve_entity_reference(entity: str, scope_text: str) -> tuple[bool, tuple[str, str] | None]:
    """Shared by BOTH the primary claim.entity check and the
    claim.comparison_entity check (Phase 8c) -- decides how an entity
    string should be validated against `scope_text`: (entity_literal,
    composite).

    entity_literal=True means the literal string appears verbatim in
    scope_text -- composite is then always None, since decomposition is
    only a FALLBACK for when the literal form is absent (see
    _decompose_state_category_entity()'s docstring: some entities really
    are literal composite strings, e.g. "ACV Ferrero", and must keep being
    checked exactly as before).

    composite is the decomposed (prefix, category) parts when `entity`
    looks like a fabricated state+category composite (e.g. "Bihar
    Ferrero") AND the literal form is absent.

    Both entity_literal=False and composite=None means `entity` is neither
    literally present nor a recognizable fabricated composite -- callers
    treat this as a plain unsupported-entity failure.
    """
    entity_literal = _mentions(scope_text, entity)
    composite = _decompose_state_category_entity(entity) if not entity_literal else None
    return entity_literal, composite


# A quantifier word followed by a bare plural noun ("various distributors",
# "multiple franchises", "several zones") -- deliberately domain-agnostic
# (no hardcoded list of entity-type nouns like "distributor") so this
# recognizes the SHAPE of a generic group reference for any entity type the
# corpus happens to have, not just distributors.
_GENERIC_GROUP_RE = re.compile(
    r"^(?:various|multiple|several|some|many|numerous|a number of|these|those)\s+([a-z]+s)$",
    re.IGNORECASE,
)


def _generic_group_noun(entity: str) -> str | None:
    """'various distributors' -> 'distributors'; 'Bihar' -> None (not a
    quantifier+plural-noun shape at all)."""
    m = _GENERIC_GROUP_RE.match(entity.strip())
    return m.group(1).lower() if m else None


def _count_word_occurrences(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# PRIMARY mechanism: claim-level validation
# ---------------------------------------------------------------------------


def _infer_claim_type(claim: AnswerClaim) -> str:
    """Best-effort claim_type when the model didn't set one (or set one
    inconsistent with its own fields) -- used to decide which extra checks
    (deviation/causal) apply on top of the always-run numeric/entity/
    period/direction/qualitative checks."""
    if claim.claim_type:
        return claim.claim_type
    text = claim.claim_text or ""
    if _is_causal_text(text):
        return "causal"
    if claim.comparison_entity:
        return "comparison"
    if claim.value is not None and claim.comparison_value is not None:
        return "trend"
    if claim.entity and "deviation" in text.lower():
        return "deviation"
    if _qualitative_tokens_in(text):
        return "qualitative"
    return "factual_numeric"


def _known_distributor_core_names(scope_text: str) -> set[str]:
    """Every distinct distributor name stated in `scope_text` via the
    corpus's own "Distributor <name> showed a significant deviation ..."
    template (see _SOURCE_DISTRIBUTOR_DEVIATION_RE, defined later in this
    module -- referencing it here is safe: Python resolves a module-level
    name at CALL time, not at function-definition time, and by the time
    any caller actually invokes this function the whole module has already
    finished loading), with each name's own trailing " Distributors" word
    stripped off. The strip matters because a claim that fuses two real
    distributors typically keeps only ONE shared trailing "Distributors"
    word for the whole fused phrase (e.g. real names "Mooney, Lamb and
    Weber Distributors" + "Scott-Norman Distributors" fused into "Mooney,
    Lamb and Weber, Scott-Norman Distributors") -- checking for the full
    name WITH its own suffix would never match the fused phrase at all,
    since the first name's own "Distributors" word is exactly the part
    that got dropped when it was fused with the second. Names shorter than
    4 characters after stripping are discarded as too generic to be a
    reliable, low-false-positive signal on their own."""
    cores: set[str] = set()
    for raw in _SOURCE_DISTRIBUTOR_DEVIATION_RE.findall(scope_text):
        name = raw.strip()
        if name.endswith(" Distributors"):
            name = name[: -len(" Distributors")].strip()
        if len(name) >= 4:
            cores.add(name)
    return cores


def _detect_entity_fusion(entity: str, scope_text: str) -> list[str]:
    """Return the distinct known distributor core names (see
    _known_distributor_core_names()) that `entity` literally contains as
    substrings, when 2 OR MORE are found -- the signature of a model
    fusing multiple real, individually-supported distributor names into
    one entity field, rather than a single genuinely wrong or uncited
    entity name (e.g. "Bihar", which contains zero known distributor core
    names). Returns [] whenever fewer than 2 are found -- callers MUST
    treat that as "not a detected fusion" (fall back to the plain
    unsupported-entity message), not as proof the entity isn't a fusion of
    some other kind (e.g. two fused STATE names, which this
    distributor-specific signal doesn't cover).

    Purely a MESSAGE-selection helper: it never changes whether a claim
    passes or fails, and is only ever called from inside a branch that was
    already about to flag "unsupported_entity" regardless of what this
    returns -- see validate_claim()'s two call sites.

    A matched core name that is itself a substring of ANOTHER matched core
    name (e.g. a short name nested inside a longer one) is dropped before
    the >=2 count, so one real name spanning that whole substring can't be
    double-counted as two distinct fused entities.
    """
    known_cores = _known_distributor_core_names(scope_text)
    matched = sorted((c for c in known_cores if c in entity), key=len, reverse=True)
    deduped: list[str] = []
    for core in matched:
        if not any(core != longer and core in longer for longer in deduped):
            deduped.append(core)
    return deduped if len(deduped) >= 2 else []


def _period_sort_key(period: str) -> tuple[int, int] | None:
    """'October 2025' -> (2025, 10), for chronological comparison -- None
    for anything unparseable. A local duplicate of premise_check.py's own
    (private) _period_key() -- matching this module's established
    convention of depending only on the shared GRAMMAR every module
    already assumes ('Month YYYY'), never on another inference-layer
    module's private helpers (see e.g. STATE_METRIC_NAMES's own
    documented duplication precedent)."""
    try:
        dt = datetime.strptime(period, "%B %Y")
    except ValueError:
        return None
    return (dt.year, dt.month)


def _trend_delta(claim: AnswerClaim) -> float:
    """claim.value - claim.comparison_value, EXCEPT when both
    claim.period and claim.comparison_period parse as chronologically-
    ordered 'Month YYYY' strings AND claim.period is the chronologically
    EARLIER of the two -- in that case the roles are swapped so the
    returned delta always reflects (later period's value - earlier
    period's value), regardless of which of value/comparison_value the
    model happened to attach to which period.

    WHY THIS EXISTS (live multi-period SKU ranking bug, 2026-08-23
    stabilization pass): answer.py's own claim-extraction schema example
    always shows value as the LATER period and comparison_value as the
    EARLIER baseline (Service Level: value=92.9 for November,
    comparison_value=94.3 for October -- a "declined INTO November"
    framing) -- but nothing in the schema or prompt actually requires
    that role assignment. A live "Compare the top-selling SKU in Gujarat
    between April 2026 and June 2026" question had the model naturally
    order period=April 2026 (earlier, mentioned first) with value=
    1,167,350 and comparison_period=June 2026 (later) with
    comparison_value=1,122,188 -- an accurate, well-cited claim
    describing a real decline from April to June. The OLD fixed-role
    assumption (actual_delta = value - comparison_value = +45,162)
    read this as metric "up", flagging the correct 'declined' claim as a
    direction_contradiction. Determining direction from the ACTUAL
    chronological order of the two periods (when both are stated and
    parseable) fixes this for either ordering the model produces, instead
    of only ever accepting the one order the original example happened to
    show.

    Only applies to a same-entity two-period TREND claim
    (comparison_entity unset) -- a cross-entity comparison (Sikkim vs.
    Maharashtra) has no "later" side to infer this way, and its own
    period/comparison_period pair does not describe a trend over time at
    all. Falls back to the original value-minus-comparison_value
    assumption whenever either period is missing/unparseable or
    comparison_entity is set -- unchanged behavior for every case this
    fix doesn't apply to."""
    if not claim.comparison_entity and claim.period and claim.comparison_period:
        period_key = _period_sort_key(claim.period)
        comparison_key = _period_sort_key(claim.comparison_period)
        if period_key is not None and comparison_key is not None and period_key < comparison_key:
            return claim.comparison_value - claim.value
    return claim.value - claim.comparison_value


def validate_claim(
    claim: AnswerClaim,
    evidence_index: dict[str, dict[str, str]],
    focus_categories: frozenset[str] | None = None,
) -> list[GroundingIssue]:
    """Check ONE structured claim against ONLY the evidence its own
    `citations` resolve to. This is the primary grounding mechanism for
    numeric, trend, comparison, deviation, causal, and qualitative claims
    alike -- see the module docstring for why citation-scoping (not
    whole-context word search) is the fix this version implements.

    `focus_categories`, when given (see _extract_focus_categories()), is
    forwarded to the primary-claim co-occurrence check so a claim whose own
    metric field doesn't name a category can't be satisfied by a cited
    sentence about a DIFFERENT category than the question actually asked
    about -- see _primary_fact_supported()'s docstring.

    Live Gamble-Wright investigation fix: `focus_categories` is derived
    from the WHOLE question (_extract_focus_categories()) and was
    previously forwarded unconditionally to every claim, including ones
    whose OWN metric field already names its own category (e.g. "IPM
    Out-of-Stock Rate") -- exactly the case _primary_fact_supported()'s
    focus_categories parameter was NEVER meant to apply to (its own
    docstring: "a claim whose OWN metric field never named a category at
    all"). A live question that legitimately asks about a category CHANGE
    over two periods ("had a GPI deviation in February... what NEW
    deviations [i.e. other categories] by August?") named "GPI" once in
    the question, which then wrongly rejected every correctly-cited IPM/
    Ferrero claim about the LATER period just for not being GPI. Since
    each such claim already self-declares its own category in `metric`,
    the question-derived focus_categories is redundant AND harmful for it
    -- only applied here when the claim's own metric field is silent on
    category, preserving the original protection unchanged for that case.

    Source-traceability fix: a claim whose `citations` field is non-empty
    but resolves to NO real evidence (e.g. the model wrote "Atomic SKU
    Facts (4)" instead of a real "Sources (21)" id) fails closed here,
    immediately, with a single "unresolved_citation" issue -- it does NOT
    fall through to the whole-blob permissive check below. That fallback
    is reserved for claims that cite nothing at all; treating an
    unresolvable-but-present citation the same way would let a
    fabricated-looking citation still pass as "grounded," which is exactly
    the gap a live investigation found (a citation that resolves to zero
    records let the whole-blob fallback wave the claim through)."""
    cited_text = _resolve_citations(claim.citations, evidence_index)
    if claim.citations and not cited_text:
        return [
            GroundingIssue(
                issue_type="unresolved_citation",
                sentence=claim.claim_text or "",
                term=", ".join(claim.citations),
                detail=(
                    f"This claim cites {list(claim.citations)} but none of those citation "
                    f"ids resolve to any known evidence record (entities/relationships/"
                    f"reports/sources/claims) -- it cannot be treated as grounded, and is "
                    f"not checked against the whole retrieved context as an unscoped claim "
                    f"would be, since an unresolvable citation is a stronger signal of a "
                    f"problem than citing nothing at all."
                ),
                source="claim",
                claim_id=claim.claim_id,
                severity="error",
            )
        ]
    used_fallback = not cited_text
    scope_text = cited_text if cited_text else _all_evidence_text(evidence_index)
    reporting_text = claim.claim_text or ""
    claim_type = _infer_claim_type(claim)
    issues: list[GroundingIssue] = []

    if focus_categories and claim.metric and any(cat.lower() in claim.metric.lower() for cat in _CATEGORY_NAMES):
        focus_categories = None

    def flag(issue_type: str, term: str | None, detail: str, severity: str = "error") -> None:
        if used_fallback:
            detail += " (this claim cited no resolvable evidence, so it was checked against the full retrieved context instead of a scoped citation.)"
        issues.append(
            GroundingIssue(
                issue_type=issue_type, sentence=reporting_text, term=term,
                detail=detail, source="claim", claim_id=claim.claim_id, severity=severity,
            )
        )

    if claim.value is not None and not _value_in_evidence(claim.value, scope_text):
        flag("unsupported_numeric", str(claim.value), f"Claimed value {claim.value} does not appear in the cited evidence.")

    if claim.comparison_value is not None and not _value_in_evidence(claim.comparison_value, scope_text):
        flag(
            "unsupported_numeric",
            str(claim.comparison_value),
            f"Claimed comparison value {claim.comparison_value} does not appear in the cited evidence.",
        )

    if claim.entity:
        group_noun = _generic_group_noun(claim.entity)
        # Decomposition is a FALLBACK, not a replacement, for literal
        # matching: some entities in this corpus really are composite
        # strings that appear verbatim (e.g. "ACV Ferrero", "Numeric
        # Distribution GPI" -- category-scoped metric entities from the
        # graph) and must keep being checked exactly as before. Only fall
        # back to decomposed (state, category) co-occurrence when the
        # literal phrase does NOT appear anywhere in the cited scope at
        # all -- i.e. exactly the fabricated-composite-entity shape this
        # fix targets, never a real literal entity that happens to also
        # look decomposable.
        entity_literal, composite = (False, None) if group_noun else _resolve_entity_reference(claim.entity, scope_text)
        if group_noun:
            # "various distributors"-shaped entity: not a hallucination
            # check against ONE literal string (that exact phrase will
            # almost never appear verbatim in evidence either way) -- the
            # real question is whether the cited evidence actually names
            # more than one of this entity type. If it does, the
            # aggregation is TRUE, just less specific than the evidence
            # allows -- flag it as a non-failing specificity notice
            # instead of a grounding failure (see Part 3 of the Phase 7
            # hardening pass). If it doesn't, this is exactly as
            # unsupported as any other invented entity -- fail closed.
            mention_count = _count_word_occurrences(scope_text, group_noun)
            if mention_count < 2:
                flag(
                    "unsupported_entity", claim.entity,
                    f"The claim generically references '{claim.entity}', but the cited evidence names "
                    f"fewer than two distinct {group_noun} -- a generic group reference isn't supported here.",
                )
            else:
                flag(
                    "generic_aggregation", claim.entity,
                    f"'{claim.entity}' is a generic group reference; the cited evidence does name multiple "
                    f"{group_noun} individually ({mention_count} mentions of '{group_noun}') -- consider "
                    "naming them specifically for a more useful answer.",
                    severity="warning",
                )
        elif composite:
            # A fabricated "<state> <Category> Category" phrase (see
            # _decompose_state_category_entity()) is never expected to
            # appear verbatim -- check that its two real parts (state,
            # category) are EACH mentioned somewhere in the cited scope
            # instead. This is deliberately the weaker, independent-mention
            # check (mirroring the plain-entity branch above); the stricter
            # co-occurrence requirement (both parts + value + metric
            # together in one sentence) is enforced below, same as for any
            # other entity.
            prefix, category = composite
            if not (_mentions(scope_text, prefix) and _mentions(scope_text, category)):
                flag(
                    "unsupported_entity", claim.entity,
                    f"The cited evidence does not mention both '{prefix}' and '{category}' "
                    f"(claim entity '{claim.entity}').",
                )
        elif not entity_literal:
            fused_names = _detect_entity_fusion(claim.entity, scope_text)
            if fused_names:
                flag(
                    "unsupported_entity", claim.entity,
                    f"The claim entity '{claim.entity}' combines multiple distinct named entities "
                    f"({', '.join(fused_names)}) into one field -- the cited evidence never states "
                    "them as one combined entity.",
                )
            else:
                flag("unsupported_entity", claim.entity, f"The cited evidence does not mention '{claim.entity}'.")

        # Primary-claim co-occurrence: entity + category + metric + value +
        # period must ALL be stated TOGETHER as one fact, not just
        # independently present somewhere in the (possibly multi-record)
        # cited scope. Skipped for generic group entities ("various
        # distributors") -- that shape is handled entirely by the branch
        # above, and the literal group phrase is never expected to co-occur
        # with a value in one sentence. For a decomposed state+category
        # composite entity, both parts (not just the fabricated literal
        # phrase) must co-occur with the value/metric -- see
        # _primary_fact_supported()'s extra_term.
        #
        # `period=claim.period` (Q7 hardening): mirrors the SAME parameter
        # the comparison_entity block below already passes as
        # `period=claim.comparison_period` -- a live failure (Q7) showed a
        # claim attribute one real distributor's real deviation figure
        # (Campbell PLC Distributors' 20.0% GPI Out-of-Stock Rate, gap
        # 12.0pp, April 2026) to a DIFFERENT real distributor (Baxter,
        # Thomas and Williams Distributors) that had a genuinely different
        # deviation that same period. Before this, claim.period was only
        # ever checked independently ("is this period mentioned ANYWHERE in
        # the cited scope") a few lines below -- never required to co-occur
        # with the entity+value+metric as one fact, so a claim citing a
        # scope that separately contained (a) Baxter's name, (b) Campbell's
        # 20.0% GPI figure, and (c) the shared period string could satisfy
        # every check independently without ever stating the four together.
        # Omitted (None) whenever the model doesn't set period at all --
        # exactly as before, since _primary_fact_supported treats a falsy
        # period as "no additional requirement" (see its docstring).
        if not group_noun and claim.value is not None:
            cooccurs = (
                _primary_fact_supported(
                    scope_text, composite[0], claim.value, claim.metric,
                    extra_term=composite[1], focus_categories=focus_categories, period=claim.period,
                )
                if composite
                else _primary_fact_supported(
                    scope_text, claim.entity, claim.value, claim.metric,
                    focus_categories=focus_categories, period=claim.period,
                )
            )
            if not cooccurs:
                flag(
                    "unsupported_numeric",
                    str(claim.value),
                    f"The cited evidence does not state{f' {claim.metric}' if claim.metric else ''} "
                    f"{claim.value}{f' for {claim.period}' if claim.period else ''} for '{claim.entity}' "
                    "together as one fact -- the entity, value, and period each appear somewhere in the "
                    "cited evidence independently, but are never stated together (the value may belong to "
                    "a different metric/category/period for this same entity).",
                )

        # Same-entity, SAME-period comparison co-occurrence (the ordinary
        # "distributor vs. the state average" deviation shape, where
        # comparison_entity/comparison_period are both left unset since
        # it's one entity's value against a state/group average from the
        # SAME period/document). Mirrors the claim.value co-occurrence
        # check just above, applied to comparison_value instead -- without
        # this, comparison_value only ever got the weak, unscoped "appears
        # somewhere in the whole cited record" check near the top of this
        # function, so one distributor's real state-average figure could be
        # silently substituted for a DIFFERENT distributor's real
        # state-average figure from the same multi-distributor cited
        # record, with nothing to catch it. Skipped when comparison_entity
        # or comparison_period is set -- those shapes are already covered
        # by the (stronger, or differently-scoped) checks below/above.
        if (
            not group_noun
            and not claim.comparison_entity
            and not claim.comparison_period
            and claim.comparison_value is not None
            and not (
                _primary_fact_supported(
                    scope_text, composite[0], claim.comparison_value, claim.metric,
                    extra_term=composite[1], focus_categories=focus_categories, period=claim.period,
                )
                if composite
                else _primary_fact_supported(
                    scope_text, claim.entity, claim.comparison_value, claim.metric,
                    focus_categories=focus_categories, period=claim.period,
                )
            )
        ):
            flag(
                "unsupported_comparison",
                str(claim.comparison_value),
                f"The cited evidence does not state{f' {claim.metric}' if claim.metric else ''} "
                f"{claim.comparison_value} for '{claim.entity}'{f' in {claim.period}' if claim.period else ''} "
                "together as one fact -- the comparison value and the entity each appear somewhere in the "
                "cited evidence independently, but are never stated together (the value may belong to a "
                "different entity's comparison figure in the same cited record).",
            )

    if claim.comparison_entity:
        # Phase 8c: the SECOND entity in a comparison BETWEEN TWO DIFFERENT
        # entities (see AnswerClaim.comparison_entity's docstring) --
        # validated with the exact same literal-first, decompose-as-
        # fallback, then co-occurrence pattern as the primary claim.entity
        # block above, just checked against comparison_value/
        # comparison_period instead of value/period. This is what lets the
        # system represent "Sikkim's 6.6% vs. Maharashtra's 6.5%" as ONE
        # claim with two properly separated entity/value pairs, instead of
        # the model fusing both states into a single unsupported entity
        # string like entity="Sikkim, Maharashtra" (which never appears
        # verbatim in evidence and correctly fails the plain entity check
        # above on its own merits, with or without this block).
        comparison_entity_literal, comparison_composite = _resolve_entity_reference(claim.comparison_entity, scope_text)
        if comparison_composite:
            comparison_prefix, comparison_category = comparison_composite
            if not (_mentions(scope_text, comparison_prefix) and _mentions(scope_text, comparison_category)):
                flag(
                    "unsupported_entity", claim.comparison_entity,
                    f"The cited evidence does not mention both '{comparison_prefix}' and '{comparison_category}' "
                    f"(claim comparison_entity '{claim.comparison_entity}').",
                )
        elif not comparison_entity_literal:
            fused_comparison_names = _detect_entity_fusion(claim.comparison_entity, scope_text)
            if fused_comparison_names:
                flag(
                    "unsupported_entity", claim.comparison_entity,
                    f"The claim comparison_entity '{claim.comparison_entity}' combines multiple "
                    f"distinct named entities ({', '.join(fused_comparison_names)}) into one field "
                    "-- the cited evidence never states them as one combined entity.",
                )
            else:
                flag(
                    "unsupported_entity", claim.comparison_entity,
                    f"The cited evidence does not mention comparison entity '{claim.comparison_entity}'.",
                )

        # Comparison-side co-occurrence: comparison_entity + comparison_value
        # + metric (+ comparison_period, when the two entities' figures come
        # from different periods) must be stated TOGETHER as one fact --
        # the same protection _primary_fact_supported() already gives the
        # PRIMARY entity/value pair, mirrored onto the comparison side so a
        # wrong-category or wrong-period value can't be smuggled in merely
        # because comparison_entity and comparison_value each appear
        # somewhere, independently, in the cited scope.
        if claim.comparison_value is not None:
            comparison_cooccurs = (
                _primary_fact_supported(
                    scope_text, comparison_composite[0], claim.comparison_value, claim.metric,
                    extra_term=comparison_composite[1], focus_categories=focus_categories,
                    period=claim.comparison_period,
                )
                if comparison_composite
                else _primary_fact_supported(
                    scope_text, claim.comparison_entity, claim.comparison_value, claim.metric,
                    focus_categories=focus_categories, period=claim.comparison_period,
                )
            )
            if not comparison_cooccurs:
                flag(
                    "unsupported_comparison",
                    str(claim.comparison_value),
                    f"The cited evidence does not state{f' {claim.metric}' if claim.metric else ''} "
                    f"{claim.comparison_value} for comparison entity '{claim.comparison_entity}'"
                    f"{f' in {claim.comparison_period}' if claim.comparison_period else ''} together as one fact "
                    "-- the comparison entity and its value each appear somewhere in the cited evidence "
                    "independently, but are never stated together (the value may belong to a different "
                    "entity/metric/category/period).",
                )

    if claim.period and not _mentions(scope_text, claim.period):
        flag("unsupported_period", claim.period, f"The cited evidence does not mention the period '{claim.period}'.")

    if claim.comparison_period and not _mentions(scope_text, claim.comparison_period):
        flag(
            "unsupported_period",
            claim.comparison_period,
            f"The cited evidence does not mention the comparison period '{claim.comparison_period}'.",
        )

    # This period-only comparison check is for the SAME-entity, two-period
    # shape (a trend -- comparison_entity unset). When comparison_entity IS
    # set, the comparison_entity block above already runs the strictly
    # stronger entity+period+value+metric co-occurrence check -- running
    # this weaker, entity-blind check on top would be redundant at best and
    # at worst could pass on a period+value pairing that belongs to a
    # DIFFERENT entity than comparison_entity names.
    if (
        not claim.comparison_entity
        and claim.comparison_period is not None
        and claim.comparison_value is not None
        and not _comparison_side_supported(scope_text, claim.comparison_period, claim.comparison_value, claim.metric)
    ):
        flag(
            "unsupported_comparison",
            f"{claim.comparison_period}={claim.comparison_value}",
            f"The cited evidence does not state{f' {claim.metric}' if claim.metric else ''} "
            f"{claim.comparison_value} for {claim.comparison_period} together as one fact -- the comparison "
            "side of this claim isn't supported, even if the number or period each appear separately "
            "elsewhere in the cited evidence.",
        )

    if claim.value is not None and claim.comparison_value is not None:
        actual_delta = _trend_delta(claim)
        actual_direction = "up" if actual_delta > 1e-9 else "down" if actual_delta < -1e-9 else "flat"
        if claim.direction:
            claimed_polarity = _infer_direction_word(claim.direction)
            if claimed_polarity and claimed_polarity != actual_direction:
                flag(
                    "direction_contradiction",
                    claim.direction,
                    f"Claim direction '{claim.direction}' contradicts the cited values: "
                    f"{claim.comparison_value} -> {claim.value} is actually '{actual_direction}'.",
                )
        # A claim's own free-text claim_text can be internally
        # self-contradictory even when the structured `direction` field is
        # fine (or left unset) -- a live cross-entity comparison failure
        # had claim_text read "...was significantly lower at 6.6%..." in
        # one clause and correctly concluded "...had a higher rate" in the
        # next, with a value/comparison_value pair that was actually 'up'.
        # Reuses the SAME conservative _infer_direction_word() used above
        # (returns None -- no check -- when the text uses BOTH an up and a
        # down word, e.g. a well-phrased two-sided "X is lower, Y is
        # higher" comparison), so this only fires on a genuinely
        # unambiguous, single-direction word that contradicts the claim's
        # own arithmetic -- not on legitimate two-sided phrasing.
        text_polarity = _infer_direction_word(reporting_text)
        if text_polarity and text_polarity != actual_direction:
            flag(
                "direction_contradiction",
                text_polarity,
                f"This claim's own text uses '{text_polarity}'-direction language, but the cited values "
                f"{claim.comparison_value} vs {claim.value} are actually '{actual_direction}'.",
            )
        if claim.delta is not None:
            # Dropsize's "gap" is a RELATIVE percent of the comparison value
            # (see build_documents.py's find_deviating_distributors() "both"
            # branch: gap = (wd_value - state_value) / state_value * 100),
            # unlike every other metric's percentage-POINT gap, where the
            # gap IS simply value - comparison_value. Comparing a Dropsize
            # claim's delta against the raw unit difference (e.g. 20.61
            # units) instead of the evidence's own relative-percent gap
            # (e.g. 10.6%) would wrongly reject a claim that faithfully
            # restates what the evidence actually says.
            is_dropsize_metric = bool(claim.metric) and claim.metric.strip().lower() == "dropsize"
            if is_dropsize_metric and claim.comparison_value:
                expected_delta = abs(actual_delta) / abs(claim.comparison_value) * 100
            else:
                expected_delta = abs(actual_delta)
            if abs(expected_delta - abs(claim.delta)) > 0.15:
                flag(
                    "unsupported_delta",
                    str(claim.delta),
                    f"Claimed delta {claim.delta} does not match the actual difference "
                    f"({expected_delta:.2f}) between {claim.comparison_value} and {claim.value}.",
                )

    # Evaluative categories (deviation/causal/qualitative) are where an
    # uncited claim is most dangerous -- "no citation" for a plain number
    # is at least checkable against the whole context, but "no citation"
    # for "this was alarming" or "X caused Y" means literally nothing was
    # offered to support the judgment. So unlike the numeric/entity/period
    # checks above, these three FAIL CLOSED on a claim that resolved no
    # citation at all, instead of falling back to whole-blob permissive
    # checking -- that fallback is exactly what let an unrelated report's
    # incidental word choice validate an unrelated claim before.
    if claim_type == "deviation" and (used_fallback or "deviation" not in scope_text.lower()):
        flag("unsupported_deviation", "deviation", "The cited evidence does not explicitly describe this as a deviation.")

    is_causal = claim_type == "causal" or _is_causal_text(reporting_text)
    if is_causal and (used_fallback or not _is_causal_text(scope_text)):
        flag(
            "unsupported_causal",
            None,
            "This claim asserts or implies causation, but the cited evidence only states facts, not an "
            "explicit causal relationship. A number or fact co-occurring with another is not sufficient "
            "evidence of causation.",
        )

    # Topic alignment: a cited record can legitimately contain a flagged
    # word while being about a completely different metric/topic than this
    # claim (e.g. citing a report whose "inefficiencies" mention is about a
    # different month's candy out-of-stock dynamics, not this claim's
    # Service Level). If the claim names a metric and the cited evidence
    # never mentions that metric at all -- or the claim cited nothing and
    # we're looking at the whole blob -- no qualitative word found can
    # establish THIS claim; flag every one of them, not just the ones that
    # also happen to be textually absent.
    topic_mismatch = used_fallback or (bool(claim.metric) and not _mentions(scope_text, claim.metric))
    if topic_mismatch:
        for term in _qualitative_tokens_in(reporting_text):
            reason = (
                "no citation was given for this claim, so nothing supports it"
                if used_fallback
                else f"the cited evidence does not mention '{claim.metric}', so it cannot establish '{term}' about it"
            )
            flag("unsupported_qualifier", term, f"{reason} (found '{term}' elsewhere in context, but that doesn't count).")
    else:
        for term in _find_unsupported_qualitative_terms(reporting_text, scope_text):
            flag("unsupported_qualifier", term, f"'{term}' does not appear (or share a word-family) in the cited evidence.")

    return issues


# ---------------------------------------------------------------------------
# SECONDARY mechanism: prose-level fallback scanners (safety net)
# ---------------------------------------------------------------------------

_PERCENT_NUMBER_RE = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*%")


_SOURCE_DISTRIBUTOR_DEVIATION_RE = re.compile(r"Distributor ([A-Z][^:]*?) showed a significant deviation")


def _distributor_names_from_sources(context_records: dict[str, pd.DataFrame]) -> set[str]:
    """Every distributor name that appears in a "Distributor <name> showed a
    significant deviation ..." sentence in context_records["sources"]' own
    retrieved text -- the exact template src/graph/build_documents.py
    writes. This is a second, independent source of "known distributor
    names" alongside _known_distributor_entity_names()'s entities-table
    scan, added because a live Q7 investigation found GraphRAG's local
    search retrieves entities (via description-embedding similarity) and
    sources/text units (via a different ranking) independently -- for a
    real query, context_records["entities"] contained NEITHER of the two
    distributors ("Baxter, Thomas and Williams Distributors", "Campbell PLC
    Distributors") that were genuinely named in the retrieved source text,
    which left scan_entity_numeric_claims() with an empty known-entities
    list and blind to a real, evidence-provable misattribution. Reading
    names directly out of the cited source text itself doesn't depend on
    that alignment holding."""
    df = context_records.get("sources")
    if df is None or df.empty or "text" not in df.columns:
        return set()
    names: set[str] = set()
    for text in df["text"].dropna():
        names.update(m.strip() for m in _SOURCE_DISTRIBUTOR_DEVIATION_RE.findall(str(text)))
    return names


def _known_distributor_entity_names(context_records: dict[str, pd.DataFrame]) -> list[str]:
    """Every distinct distributor name relevant to THIS query's own
    retrieved evidence -- the union of (a) every entity name in
    context_records["entities"] whose name contains "Distributor", and (b)
    every distributor name _distributor_names_from_sources() reads directly
    out of context_records["sources"]' own text (see that function's
    docstring for why (a) alone isn't reliable). This corpus names every
    distributor entity "<Name> Distributors" (see
    src/graph/build_documents.py's deviation-sentence template), so this is
    a data-driven way to scope scan_entity_numeric_claims() to exactly the
    entity TYPE its Q7 regression is about, without ever hardcoding a
    specific distributor's name -- only names retrieval itself already
    surfaced as relevant to this question are ever checked."""
    entity_names: set[str] = set()
    df = context_records.get("entities")
    if df is not None and not df.empty and "entity" in df.columns:
        entity_names = {str(n).strip() for n in df["entity"].dropna() if "distributor" in str(n).lower()}

    return sorted(entity_names | _distributor_names_from_sources(context_records), key=len, reverse=True)


_GAP_STATEMENT_RE = re.compile(r"gap of (?P<gap_value>[\d.]+)\s*(?:percentage points|percent|pp)\b", re.IGNORECASE)


def _check_gap_against_atomic_fact(window_text: str, fact: AtomicFact, tol: float = 0.15) -> GroundingIssue | None:
    """Requirement E (derived values): when an answer states a "gap of N
    ..." figure for an (entity, value) pair already confirmed correct
    against `fact`, recompute nothing -- fact.gap is ALREADY the
    authoritative, deterministically-extracted gap for this exact fact
    (fact_structuring.py's own regex captured it straight from the source
    sentence) -- so this only needs a direct comparison, not new
    arithmetic. Returns None when no "gap of ..." phrase is present, or
    `fact` doesn't carry a gap, or the stated figure matches within
    tolerance (same 0.15 tolerance validate_claim()'s existing delta check
    already uses, for consistency)."""
    if not fact.gap:
        return None
    m = _GAP_STATEMENT_RE.search(window_text)
    if not m:
        return None
    try:
        stated_gap = float(m.group("gap_value"))
    except ValueError:
        return None
    fact_gap_match = re.match(r"[\d.]+", fact.gap)
    if not fact_gap_match:
        return None
    try:
        correct_gap = float(fact_gap_match.group(0))
    except ValueError:
        return None
    if abs(stated_gap - correct_gap) <= tol:
        return None
    return GroundingIssue(
        issue_type="unsupported_delta",
        sentence=window_text,
        term=str(stated_gap),
        detail=(
            f"This answer states a gap of {stated_gap}, but the Atomic Fact for '{fact.entity}' "
            f"({fact.metric}" + (f" for the {fact.category} category" if fact.category else "") +
            f" in {fact.period}) records a gap of {fact.gap}, Source: Sources ({fact.source_id})."
        ),
        source="prose_fallback",
    )


def _check_entity_value_against_atomic_facts(
    entity: str, value: float, atomic_facts: list[AtomicFact], tol: float = 0.05
) -> GroundingIssue | None | AtomicFact:
    """Check one (entity, value) pair found in ANSWER prose against the
    authoritative Atomic Facts list (fact_structuring.extract_atomic_facts)
    -- requirement A: "Atomic Facts are authoritative for covered numeric
    facts." Returns a tri-state result:

      an AtomicFact     -- the specific Atomic Fact for THIS entity that
                           carries this value: confirmed correct. Returning
                           the fact itself (not just True) lets callers also
                           verify a nearby stated "gap" against this exact
                           fact's own recorded gap (requirement E).
      a GroundingIssue  -- Atomic Facts confidently show this pairing is
                           WRONG (this entity's own facts don't carry this
                           value, and/or a DIFFERENT entity's fact does) --
                           the issue's `detail` names the correct Atomic
                           Fact (entity/metric/category/period/value/
                           source) so a retry prompt can quote it directly
                           (requirement I: corrective retry).
      None              -- this value doesn't appear in ANY Atomic Fact at
                           all (for this entity or any other) -- genuinely
                           not represented by Atomic Facts (e.g. only in
                           free narrative prose), so callers must fall back
                           to the existing whole-evidence co-occurrence
                           check instead of rejecting or accepting based on
                           Atomic Facts alone.
    """
    entity_lower = entity.strip().lower()
    this_entity_facts = [f for f in atomic_facts if f.entity and f.entity.strip().lower() == entity_lower]
    for fact in this_entity_facts:
        if any(abs(value - n) <= tol for n in fact.numeric_fields()):
            return fact

    other_matches = [
        f
        for f in atomic_facts
        if f.entity
        and f.entity.strip().lower() != entity_lower
        and any(abs(value - n) <= tol for n in f.numeric_fields())
    ]

    if not this_entity_facts and not other_matches:
        return None  # not represented by Atomic Facts at all -- defer

    def _describe(fact: AtomicFact) -> str:
        parts = [f"Distributor: {fact.entity}", f"Metric: {fact.metric}"]
        if fact.category:
            parts.append(f"Category: {fact.category}")
        parts.append(f"Period: {fact.period}")
        parts.append(f"Value: {fact.value}")
        if fact.average is not None:
            parts.append(f"Average: {fact.average}")
        if fact.gap:
            parts.append(f"Gap: {fact.gap}")
        parts.append(f"Source: Sources ({fact.source_id})")
        return " | ".join(parts)

    if other_matches:
        correct = other_matches[0]
        detail = (
            f"'{entity}' is stated with the value {value}, but the Atomic Facts extracted from the "
            f"retrieved evidence show {value} belongs to a DIFFERENT distributor, not '{entity}'. "
            f"The correct Atomic Fact is: {_describe(correct)}."
        )
    else:
        known = ", ".join(f"{f.metric}={f.value}" + (f" ({f.category})" if f.category else "") for f in this_entity_facts)
        detail = (
            f"'{entity}' is stated with the value {value}, but none of the Atomic Facts extracted for "
            f"'{entity}' ({known}) include this value -- this figure does not belong to '{entity}' in "
            "the retrieved evidence."
        )
    return GroundingIssue(
        issue_type="unsupported_numeric",
        sentence="",  # filled in by the caller, which has the actual sentence/window text
        term=f"{entity}={value}",
        detail=detail,
        source="prose_fallback",
    )


def scan_entity_numeric_claims(
    answer_text: str,
    context_records: dict[str, pd.DataFrame],
    claims: list[AnswerClaim] | None = None,
) -> list[GroundingIssue]:
    """Fallback safety net: for every "...showed a significant deviation
    ...: N% vs. ..." -shaped sentence (or PAIR of adjacent sentences, see
    below) in the full answer text, verify the cited evidence actually
    states N% for the NAMED distributor as one fact -- unless a structured
    claim already covers this exact (entity, value) pair, in which case
    validate_claim() already ran the stricter version of this same check
    and re-checking here would be redundant.

    WHY THIS EXISTS: validate_claim()'s entity+value+metric co-occurrence
    check (_primary_fact_supported) only ever runs for claims the model
    actually self-reported in the structured claims block. A live failure
    (Q7) showed the model narrate a real distributor's real deviation
    figure (Campbell PLC Distributors' 20.0% GPI Out-of-Stock Rate
    deviation, gap 12.0 percentage points) but attribute it, in prose, to a
    DIFFERENT real distributor (Baxter, Thomas and Williams Distributors)
    that had no such deviation that period -- the fact never became (or
    didn't correctly become) a matching structured claim, so nothing
    checked the pairing. This scanner closes that gap the same way
    scan_qualitative_language/scan_causal_language already act as a safety
    net for their own categories: citation-scoped per sentence, falling
    back to the whole retrieved blob only when a sentence cites nothing.

    TWO-SENTENCE HARDENING (Q3-F fix): a live investigation found this
    scanner originally required the word "deviation" AND a percent number
    to co-occur in the exact same sentence -- claims==[] runs where the
    model phrased "...Baxter... showed a significant deviation..." as one
    sentence and "Their rate was 20.0%..." as the NEXT sentence produced
    ZERO issues, because neither sentence alone satisfied the gate, even
    though the pairing was exactly as wrong as the single-sentence version
    this scanner already caught. When a sentence names EXACTLY ONE known
    distributor, mentions "deviation", but states no number of its own,
    this scanner now also looks at the immediately following sentence for a
    number -- but ONLY when that next sentence doesn't itself name a
    DIFFERENT known distributor (which would mean the number belongs to
    THAT entity instead, handled on its own loop iteration). Ordinary
    single-sentence attribution (still position-sensitive: nearest
    PRECEDING entity within the same sentence, per the original design) is
    completely unchanged when a sentence already states its own number, or
    names more than one distributor.

    ATOMIC-FACTS HARDENING (requirement A/B): each candidate (entity,
    value) pair is now checked against fact_structuring.extract_atomic_facts()
    FIRST -- the authoritative, structurally-parsed fact list -- before
    falling back to the previous whole-evidence-text co-occurrence check
    (_primary_fact_supported), which remains the fallback for facts Atomic
    Facts don't cover (e.g. free narrative prose). When Atomic Facts
    confidently reject a pairing, the flagged issue's `detail` names the
    CORRECT Atomic Fact (entity/metric/category/period/value/source) so a
    retry prompt can quote the correction directly instead of just saying
    "this is wrong."

    Deliberately narrow otherwise, to avoid false positives on ordinary
    narrative sentences that merely mention a distributor's name near an
    unrelated number: only '%'-suffixed numbers are checked (never raw
    counts/Dropsize units), and a sentence/window with more than one
    distinct distributor name always falls back to the original,
    position-sensitive single-sentence attribution instead of guessing
    across a sentence boundary.
    """
    known_entities = _known_distributor_entity_names(context_records)
    if not known_entities:
        return []

    evidence_index = _build_evidence_index(context_records)
    whole_blob = _all_evidence_text(evidence_index)
    atomic_facts = extract_atomic_facts(context_records)

    covered: set[tuple[str, float]] = set()
    for c in claims or []:
        if c.entity and c.value is not None:
            covered.add((c.entity.strip().lower(), round(c.value, 1)))
        if c.comparison_entity and c.comparison_value is not None:
            covered.add((c.comparison_entity.strip().lower(), round(c.comparison_value, 1)))

    def _already_covered(entity: str, value: float) -> bool:
        key_entity = entity.strip().lower()
        return any(key_entity == e and abs(value - v) <= 0.05 for e, v in covered)

    def _entity_spans(lower_sentence: str) -> list[tuple[int, str]]:
        return [
            (idx, entity)
            for entity in known_entities
            for idx in [lower_sentence.find(entity.lower())]
            if idx != -1
        ]

    sentences = _split_sentences(answer_text)
    issues: list[GroundingIssue] = []

    for i, sentence in enumerate(sentences):
        clean_sentence = _strip_data_tag(sentence)
        lower_sentence = clean_sentence.lower()
        has_deviation = "deviation" in lower_sentence
        entity_spans = _entity_spans(lower_sentence)
        if not entity_spans:
            continue

        percent_matches = list(_PERCENT_NUMBER_RE.finditer(clean_sentence))
        borrowed_next_sentence: str | None = None

        if has_deviation and not percent_matches and len(entity_spans) == 1 and i + 1 < len(sentences):
            next_sentence = sentences[i + 1]
            next_clean = _strip_data_tag(next_sentence)
            next_entity_spans = _entity_spans(next_clean.lower())
            other_entity_in_next = any(e != entity_spans[0][1] for _, e in next_entity_spans)
            if not other_entity_in_next:
                next_percent_matches = list(_PERCENT_NUMBER_RE.finditer(next_clean))
                if next_percent_matches:
                    percent_matches = next_percent_matches
                    borrowed_next_sentence = next_sentence

        if not has_deviation or not percent_matches:
            continue

        reporting_text = sentence if borrowed_next_sentence is None else f"{sentence} {borrowed_next_sentence}"
        citations = _extract_citations_from_sentence(sentence)
        if borrowed_next_sentence is not None:
            citations = citations + _extract_citations_from_sentence(borrowed_next_sentence)
        scope = _resolve_citations(citations, evidence_index) if citations else whole_blob
        scope = scope if scope else whole_blob

        for pm in percent_matches:
            try:
                value = float(pm.group(1).replace(",", ""))
            except ValueError:
                continue

            if borrowed_next_sentence is not None:
                # Only one known entity anywhere in this window -- no
                # position-relative-to-value math needed or possible.
                entity = entity_spans[0][1]
            else:
                preceding = [(pos, e) for pos, e in entity_spans if pos < pm.start()]
                if not preceding:
                    continue
                entity = max(preceding, key=lambda pe: pe[0])[1]

            if _already_covered(entity, value):
                continue

            atomic_result = _check_entity_value_against_atomic_facts(entity, value, atomic_facts)
            if isinstance(atomic_result, AtomicFact):
                # Atomic Facts confirm this (entity, value) pairing is
                # correct -- also check any nearby stated "gap" against
                # this exact fact's own recorded gap (requirement E).
                gap_issue = _check_gap_against_atomic_fact(reporting_text, atomic_result)
                if gap_issue is not None:
                    issues.append(gap_issue)
                continue
            if isinstance(atomic_result, GroundingIssue):
                atomic_result.sentence = reporting_text
                issues.append(atomic_result)
                continue  # Atomic Facts confidently rejected this pairing

            # atomic_result is None -- not represented by Atomic Facts at
            # all; fall back to the original whole-evidence-text check.
            if not _primary_fact_supported(scope, entity, value, None):
                issues.append(
                    GroundingIssue(
                        issue_type="unsupported_numeric",
                        sentence=reporting_text,
                        term=f"{entity}={value}",
                        detail=(
                            f"'{entity}' and {value}% are both stated in this answer, but the "
                            f"{'cited' if citations else 'retrieved'} evidence never states {value}% for "
                            f"'{entity}' together as one fact -- the figure may belong to a different "
                            "distributor mentioned elsewhere in the retrieved context."
                        ),
                        source="prose_fallback",
                    )
                )
    return issues


def scan_qualitative_language(answer_text: str, context_records: dict[str, pd.DataFrame]) -> list[GroundingIssue]:
    """Fallback safety net: scan the FULL answer text sentence by
    sentence for qualitative-flag word families, regardless of whether a
    structured claim covered that sentence. Citation-scoped when a
    sentence carries its own [Data: ...] tag; whole-blob otherwise."""
    evidence_index = _build_evidence_index(context_records)
    whole_blob = _all_evidence_text(evidence_index)
    issues: list[GroundingIssue] = []
    for sentence in _split_sentences(answer_text):
        citations = _extract_citations_from_sentence(sentence)
        scope = _resolve_citations(citations, evidence_index) if citations else whole_blob
        scope = scope if scope else whole_blob
        for term in _find_unsupported_qualitative_terms(sentence, scope):
            issues.append(
                GroundingIssue(
                    issue_type="unsupported_qualifier",
                    sentence=sentence,
                    term=term,
                    detail=(
                        f"'{term}' does not appear (or share a word-family) in the "
                        f"{'cited' if citations else 'retrieved'} evidence."
                    ),
                    source="prose_fallback",
                )
            )
    return issues


def scan_causal_language(answer_text: str, context_records: dict[str, pd.DataFrame]) -> list[GroundingIssue]:
    """Fallback safety net: scan the FULL answer text sentence by
    sentence for causal-connector language, catching causal claims the
    model didn't self-report in the structured claims block.

    Skips sentences that are future/prescriptive (see
    _is_future_or_prescriptive): "monitor X because of Y" is a
    recommendation's rationale, not a claim that Y caused something in the
    historical data -- it doesn't belong to the same "did the model invent
    causation" check as "Service Level declined due to X". A future
    recommendation's own factual content (numbers it states) is still
    checked, by scan_recommendation_language() below.
    """
    evidence_index = _build_evidence_index(context_records)
    whole_blob = _all_evidence_text(evidence_index)
    issues: list[GroundingIssue] = []
    for sentence in _split_sentences(answer_text):
        if not _is_causal_text(sentence):
            continue
        if _is_future_or_prescriptive(sentence):
            continue
        citations = _extract_citations_from_sentence(sentence)
        scope = _resolve_citations(citations, evidence_index) if citations else whole_blob
        scope = scope if scope else whole_blob
        if not _is_causal_text(scope):
            issues.append(
                GroundingIssue(
                    issue_type="unsupported_causal",
                    sentence=sentence,
                    term=None,
                    detail=(
                        f"This sentence asserts or implies causation, but the {'cited' if citations else 'retrieved'} "
                        "evidence only states facts, not an explicit causal relationship."
                    ),
                    source="prose_fallback",
                )
            )
    return issues


def check_direction_consistency(
    answer_text: str, metric_comparisons: list[MetricComparison]
) -> list[GroundingIssue]:
    """Fallback safety net: for every metric the premise checker already
    computed a real baseline-vs-target direction for, check whether any
    sentence mentioning that metric by name uses direction language that
    CONTRADICTS the computed direction.

    Skips sentences that are future/prescriptive (see
    _is_future_or_prescriptive) -- "Service Level should improve going
    forward" doesn't claim Service Level already went up, regardless of
    which up/down word it uses, so it isn't a historical claim this check
    can contradict. This is the GENERAL fix for the "improve" false
    positive: it's word-agnostic (works for any UP_WORDS/DOWN_WORDS term)
    and tense/framing-aware rather than a special case for one word.
    """
    issues: list[GroundingIssue] = []
    for comp in metric_comparisons:
        if comp.direction == "flat":
            continue
        for sentence in _split_sentences(answer_text):
            if comp.metric.lower() not in sentence.lower():
                continue
            if _is_future_or_prescriptive(sentence):
                continue
            stated = _infer_direction_word(sentence)
            if stated is not None and stated != comp.direction:
                issues.append(
                    GroundingIssue(
                        issue_type="direction_contradiction",
                        sentence=sentence,
                        term=stated,
                        detail=(
                            f"Sentence implies {comp.metric} went '{stated}', but retrieved "
                            f"evidence shows {comp.value_a} ({comp.period_a}) -> "
                            f"{comp.value_b} ({comp.period_b}), i.e. '{comp.direction}'."
                        ),
                    )
                )
    return issues


def scan_recommendation_language(answer_text: str, context_records: dict[str, pd.DataFrame]) -> list[GroundingIssue]:
    """Fallback safety net over future/prescriptive sentences (see
    _is_future_or_prescriptive): being framed as a recommendation exempts
    a sentence from the HISTORICAL direction/causal checks above, but it
    does NOT exempt it from stating only real numbers. "Going forward,
    open 50 new warehouses" is still a claim that '50' is a meaningful,
    evidence-grounded figure -- if that number never appears in the
    evidence, the recommendation is fabricated, not just optimistic.

    Deliberately narrow: only checks numbers a recommendation sentence
    states, the same deterministic check validate_claim() already uses for
    structured numeric claims. It does not attempt to verify recommended
    entities/actions with no associated number (e.g. a fabricated place
    name with no digit) -- see grounding_check.py's Phase 7 hardening
    notes for why that's an accepted, documented gap rather than a bigger
    NLP subsystem.

    Numbers are extracted from the sentence's PROSE only, with its own
    [Data: ...] tag stripped first -- a live validation run found record
    ids inside a citation tag (the '147' in 'Entities (147)') otherwise
    getting misread as a factual number the sentence claims, which fails
    almost any real citation tag with 2+ ids for no real reason."""
    evidence_index = _build_evidence_index(context_records)
    whole_blob = _all_evidence_text(evidence_index)
    issues: list[GroundingIssue] = []
    for sentence in _split_sentences(answer_text):
        if not _is_future_or_prescriptive(sentence):
            continue
        numbers = _numbers_in(_strip_data_tag(sentence))
        if not numbers:
            continue
        citations = _extract_citations_from_sentence(sentence)
        scope = _resolve_citations(citations, evidence_index) if citations else whole_blob
        scope = scope if scope else whole_blob
        for value in sorted(numbers):
            if not _value_in_evidence(value, scope):
                issues.append(
                    GroundingIssue(
                        issue_type="unsupported_recommendation",
                        sentence=sentence,
                        term=str(value),
                        detail=(
                            f"This recommendation states {value}, which does not appear in the "
                            f"{'cited' if citations else 'retrieved'} evidence."
                        ),
                        source="prose_fallback",
                    )
                )
    return issues


# Deliberately high-precision phrasing only ("stands for", "is short for",
# "is an abbreviation for", "is the full form of") -- NOT a looser word
# like "means", which is common in ordinary explanatory English ("this
# means the distributor underperformed") and would false-positive on
# legitimate, non-definitional sentences far more often than it would
# catch a real fabricated expansion.
_DEFINITION_PATTERN_RE = re.compile(
    r"\b(?:stands? for|is (?:an )?abbreviat(?:ion|ed)\s+for|is short for|is the full form of|"
    r"full form of \w+ is)\b",
    re.IGNORECASE,
)


def _is_definitional_text(text: str) -> bool:
    return bool(_DEFINITION_PATTERN_RE.search(text))


def scan_definition_language(answer_text: str, context_records: dict[str, pd.DataFrame]) -> list[GroundingIssue]:
    """Fallback safety net: flag any sentence asserting what a term or
    abbreviation "stands for"/"means" (e.g. "GPI stands for General
    Product Inventory") unless the cited (or whole, as fallback) evidence
    ALSO uses that same category of definitional language -- the same
    citation-scoped, "does the evidence itself ever say this kind of
    thing" pattern scan_causal_language() already uses for causal
    connectors, applied to definitional phrasing instead.

    WHY THIS EXISTS: a live investigation found the model asked "What does
    GPI stand for?" answered with a fully fabricated expansion ("General
    Product Inventory") and, separately, "Integrated Pest Management" for
    IPM -- neither appears anywhere in the indexed corpus. This project's
    source documents (src/graph/build_documents.py's templates) are pure
    numeric fact statements; they never define what any category code
    expands to, and neither does any business-knowledge/ontology doc in
    this project (docs/BUSINESS_KNOWLEDGE_DICTIONARY.md explicitly flags
    the real-world company mapping as an unconfirmed author's guess, "not
    asserted anywhere in comments or docs"). The model also self-reported
    no structured claim for either sentence (this claim shape has no
    numeric value or entity for validate_claim() to check even if it had),
    so this prose-level scanner is the only mechanism that can catch it.

    Not specific to GPI/IPM or any hardcoded term -- this checks the
    PATTERN of asserting a definition, the same domain-agnostic approach
    the other prose scanners already use, so it also protects any other
    category/metric abbreviation a user might ask about."""
    evidence_index = _build_evidence_index(context_records)
    whole_blob = _all_evidence_text(evidence_index)
    issues: list[GroundingIssue] = []
    for sentence in _split_sentences(answer_text):
        if not _is_definitional_text(sentence):
            continue
        citations = _extract_citations_from_sentence(sentence)
        scope = _resolve_citations(citations, evidence_index) if citations else whole_blob
        scope = scope if scope else whole_blob
        if not _is_definitional_text(scope):
            issues.append(
                GroundingIssue(
                    issue_type="unsupported_definition",
                    sentence=sentence,
                    term=None,
                    detail=(
                        "This sentence states what a term or abbreviation stands for/means, but the "
                        f"{'cited' if citations else 'retrieved'} evidence never defines it -- this "
                        "expansion is not grounded in the retrieved data."
                    ),
                    source="prose_fallback",
                )
            )
    return issues


# GPIL Knowledge Layer guard (2026-08-23) -- see knowledge_layer.py's
# GlossaryEntry.forbidden_phrases docstring for why this exists ALONGSIDE
# scan_definition_language() above rather than folding into it:
# scan_definition_language() only checks whether SOME definitional-style
# sentence exists in the cited/whole evidence, never WHAT it says. Once a
# question names a known GPIL term, pipeline.py always merges that term's
# real glossary row into context_records["sources"] BEFORE generation --
# so a fabricated wrong expansion that happens to cite that real,
# resolvable row (e.g. "GPI stands for General Product Inventory [Data:
# Sources (glossary-gpi)]") would otherwise pass scan_definition_language
# cleanly: the cited evidence genuinely does contain "stands for"-style
# language, just about a different meaning than what the answer claims.
# This scanner closes that specific gap with a small, explicit, hand-
# curated blocklist of expansions already KNOWN to be wrong (not a general
# semantic fact-checker, which this project's deterministic-scanner
# philosophy doesn't attempt anywhere else) -- checked against the
# answer's own words directly, independent of citation, so it catches the
# fabrication even when citation-scoped checking above does not.
_GLOSSARY_TERMS_WITH_FORBIDDEN_PHRASES = tuple(
    (entry.label, phrase) for entry in GPIL_GLOSSARY for phrase in entry.forbidden_phrases
)


def scan_glossary_term_misuse(answer_text: str) -> list[GroundingIssue]:
    """Flag the answer if it uses one of a GPIL_GLOSSARY entry's known-
    WRONG generic expansions anywhere in its text -- e.g. "General Product
    Inventory" for GPI, "Integrated Pest Management" for IPM, both
    live-observed failures this Knowledge Layer was built to fix. See the
    module comment above _GLOSSARY_TERMS_WITH_FORBIDDEN_PHRASES for why
    this check is citation-independent, unlike every other scanner in this
    file."""
    lower_text = answer_text.lower()
    issues: list[GroundingIssue] = []
    for label, phrase in _GLOSSARY_TERMS_WITH_FORBIDDEN_PHRASES:
        if phrase in lower_text:
            issues.append(
                GroundingIssue(
                    issue_type="wrong_glossary_expansion",
                    sentence=answer_text,
                    term=label,
                    detail=(
                        f'The answer uses "{phrase}", a generic meaning explicitly known to be '
                        f'WRONG for "{label}" in this system -- see knowledge_layer.py\'s '
                        f"GPIL_GLOSSARY entry for the correct GPIL-specific meaning."
                    ),
                    source="prose_fallback",
                    severity="error",
                )
            )
    return issues


_RANKING_MAX_WORDS = {"highest", "maximum", "max", "top", "best", "most"}
_RANKING_MIN_WORDS = {"lowest", "minimum", "min", "bottom", "worst", "least"}
_RANKING_SUPERLATIVE_RE = re.compile(
    r"\b(?:highest|lowest|maximum|minimum|max|min|top|bottom|best|worst|most|least)\b", re.IGNORECASE
)
_RANKING_DIRECTION_LABEL = {"max": "highest", "min": "lowest"}


def _extract_ranking_direction(question: str) -> str | None:
    """'max'/'min' if the question unambiguously asks a highest-or-lowest
    -shaped ranking question (a named-metric sibling of premise_check.py's
    extract_superlative_ranking_claim(), which only ever matches the
    metric-less '<superlative> performing' phrase and blocks generation
    entirely -- a question naming a REAL metric, e.g. "highest Out-of-Stock
    Rate", does NOT match that gate and proceeds to normal generation,
    where nothing previously verified the claimed entity was actually the
    extremum). None for a non-ranking or ambiguous (both max- and min-words
    present) question."""
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    has_max = bool(tokens & _RANKING_MAX_WORDS)
    has_min = bool(tokens & _RANKING_MIN_WORDS)
    if has_max and not has_min:
        return "max"
    if has_min and not has_max:
        return "min"
    return None


def _extract_ranking_metric(question: str, atomic_facts: list[AtomicFact]) -> str | None:
    """The exact distributor-fact metric name (as it appears in the
    retrieved Atomic Facts) the question names, if any -- matched only
    against metric names actually PRESENT in `atomic_facts` (never a fixed
    hardcoded list), longest name first so e.g. 'Out-of-Stock Rate' is
    never shadowed by a shorter partial match."""
    metric_names = sorted({f.metric for f in atomic_facts if f.kind == "distributor"}, key=len, reverse=True)
    lower_q = question.lower()
    for name in metric_names:
        if name.lower() in lower_q:
            return name
    return None


def verify_ranking_claim(
    answer_text: str,
    claims: list[AnswerClaim] | None,
    atomic_facts: list[AtomicFact],
    question: str | None,
) -> list[GroundingIssue]:
    """Requirement F: for a highest/lowest/max/min/top/bottom/best/worst
    question naming a real metric, grounding that only checks "does the
    claimed entity's own stated value exist in evidence" is NOT sufficient
    -- that value can be perfectly real and correctly attributed to that
    entity while still not being the actual extremum among the retrieved
    candidates. This deterministically computes the true max/min across
    every matching distributor-kind Atomic Fact and checks the answer
    actually names an entity that holds it.

    Deliberately scoped, not a general-purpose ranking engine: only
    distributor-kind Atomic Facts are ranked (the corpus's real ranking
    shape -- "which distributor had the highest X"), matched on an EXACT
    metric name the question itself names (never guessed), further
    narrowed to a named category when the question names one, AND narrowed
    to the question's own named state/period when it names them (a live
    validation run found a Goa/June-2025 Dropsize ranking question pull in
    an unrelated Meghalaya/July-2026 distributor's real Dropsize fact as a
    "higher" candidate before this state/period scoping existed -- see
    fact_structuring.py's _distributor_atomic_facts_from_source_text() for
    where each distributor fact's own state now comes from). When the
    question names no state/period at all, no scoping is applied on that
    axis (unchanged, permissive default) -- this only narrows, never
    widens, what counts as a candidate. The ranking is only ever computed
    over whatever Atomic Facts THIS query's own retrieval actually
    returned -- a known, documented scope limit (not a full-corpus ranking
    guarantee), not something a post-hoc validator can see past without
    changing retrieval. Returns [] whenever the question isn't
    ranking-shaped, names no real metric, or fewer than 2 candidates are
    available to rank at all (nothing to be wrong about)."""
    if not question:
        return []
    direction = _extract_ranking_direction(question)
    if direction is None:
        return []
    metric = _extract_ranking_metric(question, atomic_facts)
    if metric is None:
        return []

    focus_categories = _extract_focus_categories(question)
    category = next(iter(focus_categories)) if focus_categories else None
    question_state = extract_state_from_question(question)
    question_period = extract_period_from_question(question)

    candidates = [
        f
        for f in atomic_facts
        if f.kind == "distributor"
        and f.metric == metric
        and (category is None or f.category == category)
        and (question_state is None or f.state is None or f.state == question_state)
        and (question_period is None or f.period == question_period)
    ]
    if len(candidates) < 2:
        return []

    extreme_value = max(f.value for f in candidates) if direction == "max" else min(f.value for f in candidates)
    correct_entities = {f.entity for f in candidates if abs(f.value - extreme_value) <= 0.05}

    # Which entity did the answer actually claim? Prefer a structured claim
    # naming this metric; otherwise the first candidate-distributor name
    # mentioned in a sentence that also names this metric or a superlative
    # word (mirrors scan_entity_numeric_claims()'s known-entity scanning
    # pattern, scoped here to just this metric's real candidates).
    #
    # `c.entity in candidate_names` (2026-08-23 stabilization pass): a live
    # SKU-ranking question ("Bottom 3 SKUs by Out-of-Stock Rate in Bihar in
    # October 2025") whose own metric name ("Out-of-Stock Rate") ALSO
    # happens to be a real distributor-level metric name triggered this
    # function even though the question has nothing to do with
    # distributors. The model's structured claim reported entity="Bihar"
    # (a state, describing which SKU had the lowest OOS rate IN Bihar) with
    # metric="Out-of-Stock Rate" -- before this check, that alone was
    # enough for this function to treat "Bihar" as "the claimed distributor
    # entity," which is never one of `candidates`' own distributor names,
    # so it always failed as `unsupported_ranking` no matter how correct
    # the actual (SKU-level) answer was. verify_sku_ranking_claim()'s own
    # structured-claim branch already had the equivalent
    # `c.entity in by_entity` guard -- this brings the distributor-level
    # sibling in line with it, rather than trusting metric-name overlap
    # alone to mean "this claim is about one of MY candidates."
    candidate_names = {f.entity for f in candidates}
    claimed_entity: str | None = None
    for c in claims or []:
        if c.entity and c.entity in candidate_names and c.metric and _metric_words_present(metric, c.metric.lower()):
            claimed_entity = c.entity
            break
    if claimed_entity is None:
        for sentence in _split_sentences(answer_text):
            lower = sentence.lower()
            if metric.lower() not in lower and not _RANKING_SUPERLATIVE_RE.search(sentence):
                continue
            for name in candidate_names:
                if name.lower() in lower:
                    claimed_entity = name
                    break
            if claimed_entity:
                break

    if claimed_entity is None or claimed_entity in correct_entities:
        return []  # nothing confidently claimed, or it's already correct

    correct_list = ", ".join(sorted(correct_entities))
    label = _RANKING_DIRECTION_LABEL[direction]
    return [
        GroundingIssue(
            issue_type="unsupported_ranking",
            sentence=answer_text[:300],
            term=claimed_entity,
            detail=(
                f"The answer names '{claimed_entity}' for the {label} {metric}"
                + (f" ({category})" if category else "")
                + f", but the actual {label} value among the retrieved Atomic Facts is {extreme_value}, "
                f"held by: {correct_list}."
            ),
            source="prose_fallback",
        )
    ]


_SKU_RANK_N_RE = re.compile(r"\b(?:top|bottom)\s+(\d+)\b", re.IGNORECASE)


def _extract_sku_ranking_metric(question: str, atomic_facts: list[AtomicFact]) -> str | None:
    """Sibling of _extract_ranking_metric(), scoped to kind="sku" Atomic
    Facts instead of kind="distributor" -- matched only against metric
    names actually present in the retrieved SKU facts (Units Delivered,
    Revenue, Service Level, Numeric Distribution, Out-of-Stock Rate; see
    fact_structuring.py's SKU_METRIC_FIELDS), longest name first so e.g.
    'Out-of-Stock Rate' is never shadowed by a shorter partial match.
    Also recognizes unambiguous sales-volume synonyms ('most selling',
    'top-selling', 'sold the most') via
    fact_structuring.resolve_sku_ranking_metric_synonym() -- the SAME
    resolution query_requirements.py's ambiguous-ranking gate already
    applies, so a question that skipped clarification for this reason
    still gets its ranking verified here instead of silently returning []
    (only if Units Delivered facts were actually retrieved, matching this
    function's existing "present in atomic_facts" discipline)."""
    metric_names = sorted({f.metric for f in atomic_facts if f.kind == "sku"}, key=len, reverse=True)
    synonym = resolve_sku_ranking_metric_synonym(question)
    if synonym and synonym in metric_names:
        return synonym
    lower_q = question.lower()
    for name in metric_names:
        if name.lower() in lower_q:
            return name
    return None


def _extract_sku_rank_n(question: str) -> int:
    """'top 3'/'bottom 5' -> 3/5. Defaults to 1 when the question asks a
    ranking-shaped SKU question without naming an explicit count (e.g.
    'the highest-selling SKU', 'which SKU has the highest sales') -- a
    single top-1 check is still meaningful verification in that case."""
    m = _SKU_RANK_N_RE.search(question)
    return int(m.group(1)) if m else 1


def verify_sku_ranking_claim(
    answer_text: str,
    claims: list[AnswerClaim] | None,
    atomic_facts: list[AtomicFact],
    question: str | None,
) -> list[GroundingIssue]:
    """SKU-grain sibling of verify_ranking_claim() (Problem 1/4), extended
    to top-N rather than a single extremum, since SKU questions in this
    project's brief commonly ask 'top 3' rather than 'the single highest'
    (see query_requirements.py's rank_n). Same scoping discipline as
    verify_ranking_claim(): only kind="sku" Atomic Facts are ranked, on
    an EXACT metric name the question itself names (never guessed),
    narrowed to the question's own named state/period when it names them.
    Unlike distributor facts (which never carry a period on their own --
    see fact_structuring.py), SKU facts always carry their own state AND
    period (self-contained sentences), so state/period scoping here is a
    plain equality check, not the "None means unscoped" fallback
    verify_ranking_claim() needs for distributor facts.

    Returns [] whenever the question isn't ranking-shaped, names no real
    SKU metric, or fewer real candidates exist than N (nothing to be
    confidently wrong about)."""
    if not question:
        return []
    direction = _extract_ranking_direction(question)
    if direction is None:
        return []
    metric = _extract_sku_ranking_metric(question, atomic_facts)
    if metric is None:
        return []

    question_state = extract_state_from_question(question)
    question_period = extract_period_from_question(question)
    n = _extract_sku_rank_n(question)

    candidates = [
        f
        for f in atomic_facts
        if f.kind == "sku"
        and f.metric == metric
        and (question_state is None or f.state == question_state)
        and (question_period is None or f.period == question_period)
    ]
    # One value per SKU name for this metric/state/period scope -- collapse
    # duplicates in case the same fact was retrieved via more than one
    # source row (e.g. the same month named twice in the question).
    by_entity: dict[str, float] = {}
    for f in candidates:
        by_entity.setdefault(f.entity, f.value)
    if len(by_entity) < n:
        return []

    ranked = sorted(by_entity.items(), key=lambda kv: kv[1], reverse=(direction == "max"))
    correct_entities = {name for name, _ in ranked[:n]}

    claimed_entities: set[str] = set()
    for c in claims or []:
        if c.entity and c.entity in by_entity and c.metric and _metric_words_present(metric, c.metric.lower()):
            claimed_entities.add(c.entity)
    if not claimed_entities:
        for sentence in _split_sentences(answer_text):
            lower = sentence.lower()
            if metric.lower() not in lower and not _RANKING_SUPERLATIVE_RE.search(sentence):
                continue
            for name in by_entity:
                if name.lower() in lower:
                    claimed_entities.add(name)

    wrong = claimed_entities - correct_entities
    if not wrong:
        return []

    label = _RANKING_DIRECTION_LABEL.get(direction, direction)
    correct_list = ", ".join(f"{name} ({value:g})" for name, value in ranked[:n])
    return [
        GroundingIssue(
            issue_type="unsupported_ranking",
            sentence=answer_text[:300],
            term=", ".join(sorted(wrong)),
            detail=(
                f"The answer names {sorted(wrong)} among the top {n} by {label} {metric}, "
                f"but the actual top {n} among the retrieved SKU Atomic Facts is: {correct_list}."
            ),
            source="prose_fallback",
        )
    ]


_ATTENTION_QUESTION_RE = re.compile(
    r"\b(need(?:s)?\s+attention|need(?:s)?\s+a\s+call|follow[\s-]?up|which distributors|underperform\w*)\b",
    re.IGNORECASE,
)


def _question_names_specific_metric(question: str, atomic_facts: list[AtomicFact]) -> bool:
    """True if the question explicitly names one of the real metrics
    present in the retrieved distributor Atomic Facts -- in which case the
    question is legitimately scoped to just that metric, and an answer
    that discusses only that metric is correctly focused, not incomplete."""
    metric_names = {f.metric for f in atomic_facts if f.kind == "distributor"}
    lower_q = question.lower()
    return any(name.lower() in lower_q for name in metric_names)


def verify_thematic_completeness(
    answer_text: str, atomic_facts: list[AtomicFact], question: str | None
) -> list[GroundingIssue]:
    """Requirement L (Dropsize completeness) / G (thematic questions): a
    live investigation found an open-ended, multi-metric thematic question
    ("which distributors need attention?") produce a final answer that
    discussed only Out-of-Stock-Rate-shaped deviations and silently omitted
    every distributor whose only retrieved deviation was Dropsize --
    despite retrieval and fact_structuring.build_atomic_facts_block() both
    correctly, completely surfacing those Dropsize facts in the SAME
    prompt (traced end-to-end against the live pilot index -- root cause is
    LLM selection/completeness bias at generation, not retrieval or fact
    structuring; see this investigation's report for the full trace).

    This is deliberately a WARNING-severity, best-effort completeness
    SIGNAL, not a grounding failure: it does not, by itself, fail the
    check or force a retry (see GroundingCheckResult.status computation in
    check_grounding() -- only error-severity issues do that). A hard
    completeness GUARANTEE would need either a prompt change or a
    mandatory-retry policy for every multi-metric thematic answer, neither
    of which this pass makes (see the final report's "deliberately not
    made" section) -- this is the smallest deterministic, non-disruptive
    signal available: it only fires for a genuinely open-ended
    "attention"/"call"/"follow-up"-shaped question that does NOT already
    name one specific metric (a metric-scoped question correctly
    discussing only that metric is not incomplete), only when the
    retrieved Atomic Facts span 2+ distinct metrics, and only when the
    answer covers at least one of those metrics but entirely omits
    another that has real candidates -- never when the answer is short,
    hedged, or covers none of the metrics at all (those aren't the
    "partially covered, rest silently dropped" shape this targets)."""
    if not question or not _ATTENTION_QUESTION_RE.search(question):
        return []
    if _question_names_specific_metric(question, atomic_facts):
        return []

    by_metric: dict[str, list[AtomicFact]] = {}
    for f in atomic_facts:
        if f.kind == "distributor" and f.entity:
            by_metric.setdefault(f.metric, []).append(f)
    if len(by_metric) < 2:
        return []

    lower_answer = answer_text.lower()
    missing_metrics: list[str] = []
    covered_metrics: list[str] = []
    for metric, facts in by_metric.items():
        entity_names = {f.entity for f in facts}
        if any(name.lower() in lower_answer for name in entity_names):
            covered_metrics.append(metric)
        else:
            missing_metrics.append(metric)

    if not missing_metrics or not covered_metrics:
        return []

    detail_parts = []
    for metric in missing_metrics:
        names = sorted({f.entity for f in by_metric[metric]})
        detail_parts.append(f"{metric} (e.g. {', '.join(names[:3])})")
    return [
        GroundingIssue(
            issue_type="incomplete_coverage",
            sentence=answer_text[:300],
            term=", ".join(missing_metrics),
            detail=(
                "This answer discusses distributors for "
                + ", ".join(covered_metrics)
                + " but the retrieved Atomic Facts also include distributor deviations for "
                + "; ".join(detail_parts)
                + ", none of which are mentioned -- consider whether these should be included too."
            ),
            source="prose_fallback",
            severity="warning",
        )
    ]


def check_grounding(
    answer_text: str,
    context_records: dict[str, pd.DataFrame],
    claims: list[AnswerClaim] | None = None,
    metric_comparisons: list[MetricComparison] | None = None,
    question: str | None = None,
) -> GroundingCheckResult:
    """Main entry point for this stage. Runs claim-level validation first
    (the primary mechanism), then the prose-level fallback scanners
    (secondary safety net) over the whole answer text. Status is 'fail' if
    any ERROR-severity issue was flagged by any mechanism -- a
    "warning"-severity issue (currently only "generic_aggregation": a
    generic group reference like "various distributors" that the cited
    evidence genuinely does back with multiple named mentions) is surfaced
    in flagged_issues for the caller/UI to show, but never by itself fails
    the check. No LLM call.

    `question`, when given, is scanned for category names (see
    _extract_focus_categories()) and forwarded to every validate_claim()
    call so a distributor-deviation claim whose own metric field never
    names a category can't be satisfied by a cited sentence about a
    DIFFERENT category than the one actually asked about -- see
    _primary_fact_supported()'s docstring for the live failure this closes.
    Optional and additive: omitting it (the default) reproduces the exact
    prior behavior, since an empty focus_categories set never rejects a
    sentence on category grounds.

    Also runs scan_entity_numeric_claims() -- the prose-level safety net
    for distributor-deviation sentences that didn't (or didn't correctly)
    become a structured claim, see its own docstring for the Q7 failure
    this closes.

    Also runs verify_ranking_claim() when `question` asks a highest/lowest
    /max/min-shaped question naming a real metric -- see that function's
    docstring for why claim-level/prose-level grounding alone (checking
    only that the claimed entity's own value is real) doesn't verify a
    ranking claim (requirement F). verify_sku_ranking_claim() is its
    top-N-capable sibling for kind="sku" Atomic Facts (Problem 1/4).

    Also runs scan_glossary_term_misuse() -- the GPIL Knowledge Layer's
    citation-independent guard against a known-wrong term expansion (e.g.
    "GPI stands for General Product Inventory") slipping through even when
    it cites a real glossary row; see that function's docstring."""
    evidence_index = _build_evidence_index(context_records)
    focus_categories = _extract_focus_categories(question)
    atomic_facts = extract_atomic_facts(context_records)

    issues: list[GroundingIssue] = []
    for claim in claims or []:
        issues += validate_claim(claim, evidence_index, focus_categories=focus_categories)

    issues += scan_qualitative_language(answer_text, context_records)
    issues += scan_causal_language(answer_text, context_records)
    issues += scan_recommendation_language(answer_text, context_records)
    issues += scan_definition_language(answer_text, context_records)
    issues += scan_glossary_term_misuse(answer_text)
    issues += scan_entity_numeric_claims(answer_text, context_records, claims)
    issues += verify_ranking_claim(answer_text, claims, atomic_facts, question)
    issues += verify_sku_ranking_claim(answer_text, claims, atomic_facts, question)
    issues += verify_thematic_completeness(answer_text, atomic_facts, question)
    if metric_comparisons:
        issues += check_direction_consistency(answer_text, metric_comparisons)

    error_issues = [i for i in issues if i.severity == "error"]
    warning_issues = [i for i in issues if i.severity == "warning"]
    status = "fail" if error_issues else "pass"
    claim_level = sum(1 for i in error_issues if i.source == "claim")
    prose_level = len(error_issues) - claim_level
    explanation = (
        f"{len(error_issues)} issue(s) flagged ({claim_level} claim-level, {prose_level} prose-fallback)."
        if error_issues
        else "No issues found at the claim level or in the prose fallback scan."
    )
    if warning_issues:
        explanation += (
            f" {len(warning_issues)} specificity notice(s): a generic group reference was used where the "
            "evidence supports naming individuals specifically."
        )
    return GroundingCheckResult(status=status, method="deterministic", flagged_issues=issues, explanation=explanation)


def _narrow_evidence_to_claim(evidence_text: str, claim: AnswerClaim) -> str:
    """Narrow a claim's fully-resolved citation text down to just the
    sentence(s) that actually support IT, when the resolved text is
    (potentially) a large multi-fact record -- e.g. a "Sources (sku-
    Punjab-February_2026)" citation resolves to the WHOLE ~59-SKU source
    blob, and without this every claim about that state/period would
    display the SAME multi-thousand-character blob as its "evidence"
    regardless of which one SKU it's actually about.

    WHY THIS EXISTS: a live 15-question SKU validation pass found exactly
    this -- once resolve_claim_sources() started returning real,
    correctly-cited evidence, every displayed SourceCitation for a
    multi-SKU ranking answer showed the identical opening lines of the
    shared source record (whichever SKU happened to render first in it),
    never the specific SKU the claim was actually about. validate_claim()
    itself was never fooled by this (it already does sentence-scoped
    co-occurrence checking via _primary_fact_supported /
    _find_primary_fact_sentence internally) -- but the PROVENANCE DISPLAY
    was still showing the unscoped whole record, which fails Problem 2's
    "evidence must actually support that specific claim" requirement even
    though grounding itself passed correctly.

    Reuses _find_primary_fact_sentence() -- the SAME sentence-matching
    logic validate_claim() already trusts to decide pass/fail -- so the
    sentence shown here is guaranteed to be one validate_claim() itself
    would accept, never a different, looser selection. Falls back to the
    full evidence_text when no single matching sentence can be found
    (e.g. a qualitative/causal claim with no entity+value pair to anchor
    on) -- narrowing is a display refinement, never a reason to show
    less evidence than was actually resolved."""
    parts: list[str] = []
    if claim.entity and claim.value is not None:
        sentence = _find_primary_fact_sentence(evidence_text, claim.entity, claim.value, claim.metric, period=claim.period)
        if sentence:
            parts.append(sentence)
    if claim.comparison_entity and claim.comparison_value is not None:
        sentence = _find_primary_fact_sentence(
            evidence_text, claim.comparison_entity, claim.comparison_value, claim.metric,
            period=claim.comparison_period or claim.period,
        )
        if sentence:
            parts.append(sentence)
    elif claim.comparison_value is not None and claim.entity:
        # Same-entity, same-period comparison shape (e.g. distributor vs.
        # state average) -- comparison_value belongs to the SAME entity
        # named above, just look for it as a second fact in that scope.
        sentence = _find_primary_fact_sentence(evidence_text, claim.entity, claim.comparison_value, claim.metric, period=claim.period)
        if sentence and sentence not in parts:
            parts.append(sentence)
    return " ".join(parts) if parts else evidence_text


def resolve_claim_sources(
    claims: list[AnswerClaim], context_records: dict[str, pd.DataFrame]
) -> list[SourceCitation]:
    """Public provenance facade (Problem 2 -- source traceability).

    Resolves every claim's own `citations` field to the actual evidence
    text those citation ids point to in context_records, reusing the EXACT
    SAME evidence index and citation-group parsing (_build_evidence_index,
    _resolve_citations) check_grounding()/validate_claim() already use to
    VALIDATE claims -- so a source ever displayed here is never fabricated
    or looser than what grounding already treated as real: if a citation
    didn't resolve to real evidence text, it contributes nothing here
    either, exactly like validate_claim() would treat it as unsupported.
    The resolved text is then narrowed to the specific supporting
    sentence(s) via _narrow_evidence_to_claim() -- see that function's
    docstring for why a raw resolved record (which can span many SKUs/
    distributors) is not by itself precise enough evidence to display for
    one specific claim.

    Only claims with at least one resolvable citation are included; a
    claim with no citations, or citations that don't resolve to any known
    record, contributes nothing (never a placeholder or invented source)
    -- callers should not infer anything about whether such a claim PASSED
    grounding from its absence here, since this function does not itself
    judge grounding (that already happened, upstream, in check_grounding()
    -- see pipeline.py, which only calls this on the answer it has already
    committed to showing).
    """
    evidence_index = _build_evidence_index(context_records)
    sources: list[SourceCitation] = []
    for claim in claims:
        if not claim.citations:
            continue
        evidence_text = _resolve_citations(claim.citations, evidence_index)
        if not evidence_text:
            continue
        sources.append(
            SourceCitation(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                citation_ids=list(claim.citations),
                evidence_text=_narrow_evidence_to_claim(evidence_text, claim),
            )
        )
    return sources
