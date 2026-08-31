"""
Phase 7e -- deterministic query-requirements detection (Problem 4:
granularity mismatch).

WHAT THIS FILE IS FOR (plain language):
Before any retrieval-quality judgment is made, a question already implies a
REQUIRED GRANULARITY -- the level of the business hierarchy (State,
Category, Channel, Distributor, SKU) it needs evidence at to be answered
honestly. "Top 3 SKUs in Punjab in 2025" requires SKU-grain evidence; no
amount of State- or Category-grain evidence, however confidently retrieved,
can honestly answer it. Detecting this BEFORE generation is what lets the
pipeline distinguish "the evidence I have doesn't cover what was asked"
from "I searched and found nothing" -- the first is a representation-grain
mismatch, the second is a retrieval failure, and they need different,
honest explanations (see sku_evidence.py's evaluate_sku_evidence()).

This module only detects what the QUESTION requires -- it never looks at
what evidence was actually retrieved (that is evidence_sufficiency's job,
downstream). It reuses premise_check.py's existing state/period extractors
rather than re-implementing them (per the "reuse existing helpers" brief) --
this module's only new logic is SKU-granularity, ranking-count, and
ranking-metric detection.

ranking_metric (added alongside the SKU-in-GraphRAG architecture change):
whether the question itself names one of the real SKU metrics (Units
Delivered, Revenue, Service Level, Numeric Distribution, Out-of-Stock
Rate -- fact_structuring.SKU_METRIC_FIELDS). This project has no confirmed
business-default "top SKU" metric, so a SKU-ranking-shaped question that
names none of these is genuinely ambiguous -- pipeline.py uses
ranking_metric is None (alongside is_ranking and required_granularity ==
"sku") to short-circuit into a clarification request BEFORE generation,
rather than letting the model silently pick a metric.

rank_direction ("top" for a highest/max-shaped ranking question, "bottom"
for a lowest/min-shaped one): computed the same way ranking_metric is, so
sku_evidence.py can compute a ranking DETERMINISTICALLY (sort the real
rows, take N) rather than handing the generation model a pile of SKU facts
and trusting it to rank them correctly -- see that module's
_build_ranking_source_row(). See _extract_rank_direction()'s docstring for
why the word lists mirror grounding_check.py's own post-hoc ranking
verification rather than importing from it.

required_granularity is also set to "sku" when the question names a real
SKU by name (e.g. "the revenue of Marlboro Pack 1") even without the
literal word "SKU" anywhere -- see known_sku_names_param below. Kept
deliberately narrow otherwise: every OTHER granularity (State/Category/
Channel/Distributor) is already the corpus's default representation, so
there is no mismatch to detect for them -- extending this to infer
SKU-shape from a bare product/franchise name mention (with no "Pack N")
would need a fuzzy classifier this project has no evidence it needs yet
(per Problem 4's own "don't overbuild" instruction).

WHY known_sku_names IS A PARAMETER, NOT A FILE READ HERE (SKU-name-only
questions, 2026-08-23 stabilization pass): a literal SKU name ("Marlboro
Pack 1") is data, not question grammar -- recognizing it requires the
actual list of 59 sku_name values from data/kpi_state_month_sku.csv (see
sku_evidence.known_sku_names()). Reading that file INSIDE this module
would break its documented "pure question-text analysis, no I/O" contract
and make every call (including every test) depend on a real data
directory. Instead, detect_query_requirements() takes the already-loaded
name set as an optional parameter -- callers that have it (pipeline.py)
pass it in; callers that don't (existing tests, callers with no interest
in bare-name SKU detection) omit it and get the exact same
literal-"SKU"-word-only behavior as before. Matching is substring-based
against the question's lowercase text -- deliberately NOT a hardcoded name
list in this file's own code: the names come from the single real data
source every other SKU-evidence lookup already reads, so this can never
drift out of sync with what SKUs actually exist.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.inference.fact_structuring import SKU_METRIC_FIELDS, resolve_sku_ranking_metric_synonym
from src.inference.knowledge_layer import detect_glossary_terms
from src.inference.premise_check import (
    INDIAN_STATES,
    extract_all_periods_from_question,
    extract_period_from_question,
    extract_state_from_question,
)
from src.inference.schemas import QueryRequirements

_SKU_WORD_RE = re.compile(r"\bSKUs?\b", re.IGNORECASE)

# Same word lists as grounding_check.py's _RANKING_MAX_WORDS/_RANKING_MIN_WORDS
# -- duplicated, not imported, matching this file's own established
# convention (see _SKU_METRIC_LABELS above) of depending only on fixed
# grammar shared across inference-layer modules, never importing one
# module's internal ranking-verification helpers into another's
# pre-generation requirement detection.
_RANK_MAX_WORDS = {"highest", "maximum", "max", "top", "best", "most"}
_RANK_MIN_WORDS = {"lowest", "minimum", "min", "bottom", "worst", "least"}


def _normalize_sku_name_text(text: str) -> str:
    """Collapse underscores and any run of whitespace to a single space,
    lowercased -- e.g. 'GPI_Franchise_1 Pack 1' and 'GPI Franchise 1 Pack
    1' both normalize to 'gpi franchise 1 pack 1'. Applied to BOTH the
    stored sku_name and the question text before substring matching (see
    _question_names_known_sku()) so a natural, non-underscore phrasing of
    a franchise-derived SKU name (every franchise whose CATEGORY_FRANCHISES
    key uses an underscore, e.g. GPI_Franchise_N/Candy_Franchise_N -- see
    src/data_gen/generate_synthetic_data.py) is still recognized. Live gap
    found 2026-08-23: 'What was the revenue of GPI Franchise 1 Pack 1...'
    (the natural way a person would say the name out loud) failed to match
    the literal stored string 'GPI_Franchise_1 Pack 1' at all, silently
    falling back to required_granularity=None -- a real SKU question
    answered as if it were an ordinary state-level one. 'Marlboro'/'TicTac'
    /'Kinder_Joy' SKU names have no franchise-number suffix in their
    question-facing form, so this was previously invisible for exactly the
    examples this module's own docstring already used."""
    return re.sub(r"[_\s]+", " ", text).strip().lower()


def _question_names_known_sku(question: str, known_sku_names: Iterable[str] | None) -> bool:
    """True if `question` contains, as a substring (after
    _normalize_sku_name_text() normalization on both sides), one of the
    exact sku_name values in `known_sku_names` (e.g. 'Marlboro Pack 1') --
    e.g. 'What was the revenue of Marlboro Pack 1 in Gujarat?' Returns
    False (never raises) when known_sku_names is None/empty, so a caller
    that doesn't have the name list (or doesn't care about bare-name SKU
    detection) gets the exact same behavior as before this parameter
    existed."""
    if not known_sku_names:
        return False
    normalized_q = _normalize_sku_name_text(question)
    return any(_normalize_sku_name_text(name) in normalized_q for name in known_sku_names)

# The canonical list of real SKU metric names, imported (not re-derived)
# from fact_structuring.py -- the module that actually owns the sentence
# grammar these names come from. Longest-name-first so e.g. "Out-of-Stock
# Rate" is never shadowed by a shorter partial match against the question
# text. Used only to detect whether the question ITSELF names a metric --
# never to guess one when it doesn't (see Design Decision: ambiguous SKU
# ranking must ask for clarification, not silently default).
_SKU_METRIC_LABELS = sorted((label for _, label in SKU_METRIC_FIELDS), key=len, reverse=True)

_RANK_N_RE = re.compile(r"\b(?:top|bottom)\s+(\d+)\b", re.IGNORECASE)

# Broader than premise_check.py's _SUPERLATIVE_PERFORMING_RE (which only
# ever matches the narrow, metric-less "<word> performing" shape) -- this
# only feeds QueryRequirements.is_ranking, an informational flag, never a
# hard gate, so it can afford to be more permissive.
_RANKING_WORDS_RE = re.compile(
    r"\b(top|bottom|highest|lowest|best|worst|most|least|rank|ranking|"
    r"performing|poorly|underperform\w*|weak\w*|intervention)\b",
    re.IGNORECASE,
)

_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Signals this project treats as "global_search"-shaped, per the fixed
# three-way classification (basic_search / local_search / global_search)
# -- cross-state, broad-historical, "all"/"every", trend/recurring/pattern
# language, or a multi-month/multi-year scan. Deliberately word-list-based
# (same "grammar, not guessing" discipline every other detector in this
# module already follows), not an LLM classifier -- this only ever
# controls retrieval SCOPE (see pipeline.py's _scope_sources_to_state()),
# never gates which pipeline stage runs, so an imperfect edge case has low
# blast radius by design.
_GLOBAL_SCOPE_WORDS_RE = re.compile(
    r"\b(all|every|across states|across the country|nationwide|country[- ]wide|"
    r"trend|trends|recurring|recurrence|pattern|patterns|history|historical|historically|"
    r"chronic|chronically|unstable|instability|"
    r"how often|how frequently|frequency|commonly|"
    r"getting better|getting worse|over time|over the years|"
    r"multi[- ]year|multi[- ]month|year[- ]over[- ]year|\byoy\b|"
    r"single dominant|dominant cause|independent issues)\b",
    re.IGNORECASE,
)

# Cross-state comparison shape ("X compared to Y", "X vs Y", "X and Y")
# naming two DIFFERENT real states -- also global_search-shaped even
# without hitting a word above, since it spans more than the one state a
# single-state question would otherwise narrow retrieval to. Reuses
# premise_check.INDIAN_STATES (imported above) rather than a third copy of
# the state list.
def _names_multiple_states(question: str) -> bool:
    matched = {name for name in INDIAN_STATES if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE)}
    return len(matched) >= 2


def _extract_target_year(question: str) -> int | None:
    """A bare 4-digit year ('...in 2025') ONLY when the question does not
    already name a specific '<Month> <Year>' period -- an explicit month
    always wins, mirroring premise_check.py's own "the question's own most
    specific statement wins" precedent. Returns the LAST bare year
    mentioned when more than one appears (a single-year question rarely
    names two different bare years; this only matters for pathological
    inputs, and "last" keeps behavior deterministic either way)."""
    if extract_all_periods_from_question(question):
        return None
    matches = _BARE_YEAR_RE.findall(question)
    if not matches:
        return None
    return int(matches[-1])


def _extract_rank_n(question: str) -> int | None:
    """'top 3'/'bottom 5' -> 3/5. None when the question doesn't name an
    explicit count (e.g. 'the highest-selling SKU', 'weak SKUs' -- ranking
    is still implied via is_ranking, just not to a specific N)."""
    m = _RANK_N_RE.search(question)
    return int(m.group(1)) if m else None


def _extract_rank_direction(question: str) -> str | None:
    """'top'/'highest'/'most'/'best' -> "top"; 'bottom'/'lowest'/'least'/
    'worst' -> "bottom". None when neither an unambiguous max- nor
    min-word is present, or both are (a genuinely mixed/ambiguous
    question) -- mirrors extract_direction_claim()'s own "only gate on an
    unambiguous claim" discipline in premise_check.py. This existing
    word-list convention (max/min word -> "highest"/"best"/"most" all mean
    the same "sort descending, take the largest values" direction, exactly
    as grounding_check.py's verify_ranking_claim()/verify_sku_ranking_claim()
    already treat them for post-hoc verification) is deliberately reused
    unchanged here, not redefined more precisely per-metric -- e.g. "best"
    Out-of-Stock Rate sorts the same direction as "highest" Out-of-Stock
    Rate under this convention, even though a lower OOS% is arguably
    "better" business-wise. This project has never defined a per-metric
    good/bad polarity, so keeping the SAME literal max/min-word mapping
    the rest of this codebase already uses is the honest, consistent
    choice -- not a place to quietly invent a new one."""
    tokens = set(re.findall(r"[a-z]+", question.lower()))
    has_max = bool(tokens & _RANK_MAX_WORDS)
    has_min = bool(tokens & _RANK_MIN_WORDS)
    if has_max and not has_min:
        return "top"
    if has_min and not has_max:
        return "bottom"
    return None


def _extract_ranking_metric(question: str) -> str | None:
    """The exact SKU metric name (from _SKU_METRIC_LABELS) the question
    itself names, e.g. 'top 3 SKUs by revenue' -> 'Revenue'. Also
    recognizes unambiguous sales-volume synonyms ('most selling',
    'top-selling', 'sold the most') via
    resolve_sku_ranking_metric_synonym(), checked first since those
    phrases never literally contain a metric label. None when the
    question names no real SKU metric AND no recognized synonym -- e.g. a
    bare 'top 3 SKUs in Punjab in 2025' or 'best-performing SKUs' -- which
    is exactly the genuinely ambiguous case pipeline.py's
    needs_clarification short-circuit exists to catch, never a case this
    function should guess its way out of."""
    synonym = resolve_sku_ranking_metric_synonym(question)
    if synonym:
        return synonym
    lower_q = question.lower()
    for label in _SKU_METRIC_LABELS:
        if label.lower() in lower_q:
            return label
    return None


# Category-scoped-ONLY metrics (fact_structuring.py's own
# _CATEGORY_METRIC_FIELDS naming) -- these are only ever measured per
# product category (GPI/IPM/Ferrero/Candy) in this corpus, never as a
# single state-wide figure, unlike Service Level/Productivity/Dropsize/
# Inventory Turns/Inventory Days (state-level, category-agnostic). Longest
# phrase first so "out of stock" doesn't shadow "range billing" etc. (not
# actually order-sensitive here since none overlap, but matches this
# module's own established convention).
_CATEGORY_SCOPED_METRIC_ALIASES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bout[- ]of[- ]stock(?:\s+rate)?\b", re.IGNORECASE), "Out-of-Stock Rate"),
    (re.compile(r"\boos%?\b", re.IGNORECASE), "Out-of-Stock Rate"),
    (re.compile(r"\bnumeric distribution\b", re.IGNORECASE), "Numeric Distribution"),
    (re.compile(r"\bacv\b", re.IGNORECASE), "ACV"),
    (re.compile(r"\brange billing\b", re.IGNORECASE), "Range Billing"),
)

# The 4 GPIL_GLOSSARY term_ids that name a real product category -- reused
# (not re-derived) from knowledge_layer.py, the single source of truth for
# how "GPI"/"IPM"/"Ferrero"/"Candy" surface forms are recognized in a
# question, so this can never drift out of sync with what the Knowledge
# Layer itself already matches.
_CATEGORY_TERM_IDS = frozenset({"gpi", "ipm", "ferrero", "candy"})


def _extract_category_scoped_metric(question: str) -> str | None:
    """The exact category-scoped metric label the question names, or None
    if it names none of the 4. Checked independently of whether a category
    is also named -- see detect_query_requirements()'s category_ambiguous
    computation for how the two combine."""
    for pattern, label in _CATEGORY_SCOPED_METRIC_ALIASES:
        if pattern.search(question):
            return label
    return None


def _question_names_explicit_category(question: str) -> bool:
    """True if the question names GPI/GPIL, IPM, Ferrero, or Candy by any
    of knowledge_layer.py's own recognized surface forms."""
    return any(entry.term_id in _CATEGORY_TERM_IDS for entry in detect_glossary_terms(question))


def classify_query_intent(question: str, target_state: str | None) -> str:
    """"basic_search" / "local_search" / "global_search" -- deterministic,
    controls retrieval SCOPE only (see QueryRequirements.query_intent's
    docstring and pipeline.py's _scope_sources_to_state()), never gates
    which pipeline stage runs.

    global_search wins over local_search whenever EITHER a global-scope
    word (_GLOBAL_SCOPE_WORDS_RE) appears OR the question names 2+
    different real states -- both are "broader than one state's worth of
    evidence" signals regardless of whether target_state also resolved to
    something (e.g. "Gujarat's Baxter keeps showing up... is a pattern
    developing?" names Gujarat AND uses "developing"/no explicit global
    word, but "Gujarat's history of recurring distributor issues" would
    hit "recurring"/"history" -- global wins in that case even though
    Gujarat is the only state named, since the question is asking for an
    open-ended scan, not a bounded fact/comparison).

    local_search is target_state being resolved (a single named state) AND
    no global signal. basic_search is the residual case -- no state named,
    no global signal -- a plain direct-fact question with nothing to scope
    retrieval to or broaden it for."""
    if _GLOBAL_SCOPE_WORDS_RE.search(question) or _names_multiple_states(question):
        return "global_search"
    if target_state:
        return "local_search"
    return "basic_search"


def detect_query_requirements(
    question: str, known_sku_names: Iterable[str] | None = None
) -> QueryRequirements:
    """The main entry point. Pure question-text analysis -- no I/O of its
    OWN, no context_records, no evidence lookup, safe to call before
    retrieval or on a question that will never touch the index at all.

    `known_sku_names`, when given (see sku_evidence.known_sku_names()),
    additionally recognizes a question that names a real SKU by name (e.g.
    'Marlboro Pack 1') even when the literal word 'SKU' never appears --
    see the module docstring's "WHY known_sku_names IS A PARAMETER" note.
    Omitting it (the default) reproduces the exact prior literal-word-only
    behavior for every existing caller/test."""
    required_granularity = (
        "sku"
        if _SKU_WORD_RE.search(question) or _question_names_known_sku(question, known_sku_names)
        else None
    )

    target_periods = extract_all_periods_from_question(question)
    if not target_periods:
        single = extract_period_from_question(question)
        target_periods = [single] if single else []

    target_state = extract_state_from_question(question)

    # Gated on required_granularity is None: a SKU-ranking question named
    # BY Out-of-Stock Rate ("Bottom 2 SKUs by Out-of-Stock Rate in Gujarat
    # in April 2026") also names this same metric word, but is a
    # completely different, already-handled evidence path
    # (sku_evidence.py's per-SKU ranking) -- it must never be preempted by
    # this state/category-grain short-circuit.
    category_scoped_metric = _extract_category_scoped_metric(question) if required_granularity is None else None
    category_ambiguous = category_scoped_metric is not None and not _question_names_explicit_category(question)

    return QueryRequirements(
        required_granularity=required_granularity,
        target_state=target_state,
        target_periods=target_periods,
        target_year=_extract_target_year(question),
        is_ranking=bool(_RANKING_WORDS_RE.search(question)),
        rank_n=_extract_rank_n(question),
        ranking_metric=_extract_ranking_metric(question) if required_granularity == "sku" else None,
        rank_direction=_extract_rank_direction(question),
        query_intent=classify_query_intent(question, target_state),
        category_ambiguous=category_ambiguous,
        category_scoped_metric=category_scoped_metric if category_ambiguous else None,
    )
