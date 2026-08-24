"""
Phase 7 -- shared data structures for the guarded answer-generation pipeline.

WHAT THIS FILE IS FOR (plain language):
Every stage of the Phase 7 pipeline (premise check -> answer generation ->
grounding check) needs to pass structured results to the next stage and,
eventually, out to a caller (a test, a script, later the Streamlit UI). Instead
of passing around loose dicts, this file defines one small dataclass per
result type so every stage's output is typed and self-documenting. All of
them have a to_dict() method so a whole pipeline run can be dumped as JSON
(matching the schema agreed in the Phase 7 architecture audit) for logging,
debugging, or test assertions.

Nothing in this file calls an LLM or touches GraphRAG -- it is pure data
definitions, safe to import from anywhere without side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Premise verification (Stage 0 -- runs BEFORE any answer-generation LLM call)
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """One metric's value at two periods, plus the direction it moved.

    This is the atomic unit both the premise checker (comparing baseline vs.
    target period) and the grounding checker (checking the drafted answer's
    claims against reality) work with -- it's the deterministic "ground
    truth" a claim gets checked against.
    """

    metric: str
    period_a: str
    value_a: float
    period_b: str
    value_b: float
    direction: Literal["up", "down", "flat"]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PremiseCheckResult:
    """Result of asking: 'does the retrieved evidence actually support the
    directional assumption embedded in the user's question?'

    status meanings:
      - "no_claim": the question doesn't assert a direction (e.g. a plain
        "what was X's performance" question) -- nothing to verify, proceed
        straight to answer generation.
      - "supported": a comparable baseline period was found and every
        checked metric moved in the claimed direction.
      - "contradicted": a comparable baseline was found and every checked
        metric moved the OPPOSITE way from the claim.
      - "unsupported": a comparable baseline was found but the metrics
        disagree with each other (mixed signal) -- the claim isn't clearly
        established either way.
      - "insufficient_data": no comparable baseline period is indexed at
        all, so the claim can't be evaluated in either direction.
      - "undefined_metric": the question asks a superlative/ranking claim
        (e.g. "least performing", "best performing") that this project
        defines no metric or composite score for, and/or the indexed
        corpus contains no comparable aggregate to rank against -- the
        claim can't be computed or verified at all, regardless of what
        evidence retrieval happens to return, so generation is skipped
        the same way "insufficient_data" skips it.
    """

    status: Literal[
        "no_claim", "supported", "contradicted", "unsupported", "insufficient_data", "undefined_metric"
    ]
    method: Literal["deterministic", "llm"]
    claimed_direction: Literal["decline", "improve", "neutral"] | None
    target_state: str | None
    target_period: str | None
    baseline_period: str | None
    metric_comparisons: list[MetricComparison] = field(default_factory=list)
    # Comparisons against whatever OTHER period actually is indexed (e.g. the
    # month after, when the question asks about a decline into the month
    # before) -- informative context even when it can't settle the claim.
    supplementary_comparisons: list[MetricComparison] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Answer generation (Stage 1)
# ---------------------------------------------------------------------------


ClaimType = Literal["factual_numeric", "trend", "comparison", "deviation", "causal", "qualitative"]


@dataclass
class AnswerClaim:
    """One claim the drafted answer makes, extracted from the model's own
    structured output (see src/inference/answer.py) so the grounding
    checker can validate it in plain Python instead of needing a second
    LLM call to re-read the prose.

    No field is required for every claim -- a plain factual_numeric claim
    might only set metric/period/value/citations, while a causal claim
    might set only claim_text/claim_type/citations. claim_text (the exact
    sentence or clause the claim corresponds to) is what lets
    grounding_check.py scope qualitative/causal language scanning to just
    that claim's own cited evidence, instead of the whole retrieved blob --
    see grounding_check.py's module docstring for why that scoping is the
    core fix over the original word-anywhere-in-context approach.

    comparison_entity (Phase 8c): the SECOND entity in a comparison BETWEEN
    TWO DIFFERENT entities (e.g. two states) -- as opposed to
    comparison_period, which compares the SAME entity across two periods.
    Set alongside comparison_value (and comparison_period too, when the two
    entities' figures are also from different periods, e.g. "Maharashtra's
    Ferrero OOS in August 2024 vs. Sikkim's Ferrero OOS in April 2025").
    `entity`/`value`/`period` describe one side, `comparison_entity`/
    `comparison_value`/`comparison_period` describe the other -- never
    combine two entity names into one `entity` string (e.g. do NOT write
    entity="Sikkim, Maharashtra"). See grounding_check.py's Phase 8c notes
    for why this field exists instead of just splitting into two
    factual_numeric claims: a single comparison claim can still state which
    side is higher/lower (direction) and by how much (delta), which two
    independent factual_numeric claims cannot express on their own.
    """

    claim_id: int | str | None = None
    claim_text: str | None = None
    claim_type: ClaimType | None = None
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    comparison_period: str | None = None
    comparison_entity: str | None = None
    value: float | None = None
    comparison_value: float | None = None
    direction: str | None = None
    delta: float | None = None
    citations: list[str] = field(default_factory=list)
    supporting_evidence_hint: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnswerResult:
    """The model's drafted answer plus whatever structured claims it
    reported about itself."""

    text: str
    claims: list[AnswerClaim] = field(default_factory=list)
    llm_calls: int = 1

    def to_dict(self) -> dict:
        return {"text": self.text, "claims": [c.to_dict() for c in self.claims], "llm_calls": self.llm_calls}


# ---------------------------------------------------------------------------
# Grounding / contradiction check (Stage 2 -- runs AFTER answer generation)
# ---------------------------------------------------------------------------


@dataclass
class GroundingIssue:
    """One thing the grounding checker flagged in the drafted answer.

    issue_type meanings:
      - "unsupported_numeric"/"unsupported_entity"/"unsupported_period"/
        "unsupported_delta"/"unsupported_deviation"/"unsupported_causal"/
        "unsupported_recommendation": claim-level or recommendation-scanner
        checks -- a structured AnswerClaim's specific field, or a
        future/prescriptive sentence's own stated number, didn't hold up
        against the evidence its OWN citations point to.
      - "unsupported_comparison": a trend/comparison claim's
        comparison_period and comparison_value don't appear TOGETHER in
        any one sentence of the cited evidence -- each may appear
        separately (a different fact's value, a different fact's period),
        but the cited evidence never actually states this metric's value
        for the comparison period as one fact. Additive to
        "unsupported_numeric"/"unsupported_period", which still fire
        independently for the simpler "not cited at all" cases.
      - "unsupported_qualifier"/"direction_contradiction": the original
        prose-level scanners, now scoped to a sentence's own citations
        when present. Still run over the FULL answer text as a fallback
        safety net, per the claim-level-first design -- see
        grounding_check.py.
      - "generic_aggregation": NOT a failure -- a claim's entity is a
        generic group phrase ("various distributors") that IS backed by
        multiple distinct named mentions in the cited evidence, so the
        aggregation itself is true, just less specific than the evidence
        allows. Always carries severity="warning" (see below); never the
        sole reason a grounding check fails.
      - "unsupported_definition": a sentence asserts what a term or
        abbreviation "stands for"/"means" (e.g. "GPI stands for General
        Product Inventory"), but the cited (or whole, as fallback)
        evidence never uses that same category of definitional language
        itself -- this corpus's source documents never define what any
        category code expands to, so an answer that does is fabricating
        the expansion rather than reporting it. Prose-level scanner only
        (source="prose_fallback"); see scan_definition_language().
      - "unsupported_ranking": the question asked for a highest/lowest/
        max/min/top/bottom entity for a named metric, and the answer names
        an entity that is NOT actually the extremum among the matching
        Atomic Facts retrieved for that metric -- verifying that the
        claimed entity's own value merely exists somewhere in evidence is
        not sufficient for a ranking claim; see
        grounding_check.py:verify_ranking_claim().
      - "incomplete_coverage": for an open-ended, multi-metric thematic
        question (e.g. "which distributors need attention?"), the
        retrieved Atomic Facts span 2+ distinct metrics but the answer's
        own text mentions distributor names for only SOME of them, entirely
        omitting another metric that has real candidates -- a completeness
        signal (see the Dropsize-omission investigation), always
        severity="warning" (best-effort, not a hard failure -- see
        grounding_check.py:verify_thematic_completeness() for why).
      - "unresolved_citation": a claim's `citations` field is non-empty but
        none of those citation ids resolve to any known evidence record
        (e.g. the model cited "Atomic SKU Facts (4)" instead of a real
        "Sources (21)" id -- see answer.py's
        _ATOMIC_FACTS_CITATION_GUIDANCE). Deliberately NOT treated the same
        as citing nothing (which falls back to a whole-blob check) --
        an unresolvable citation looks like provenance but isn't, so
        grounding_check.py:validate_claim() fails closed instead of
        letting the permissive whole-blob fallback wave it through. Always
        severity="error"; source="claim".
      - "wrong_glossary_expansion": GPIL Knowledge Layer guard (see
        knowledge_layer.py). The answer text uses one of a small,
        hand-curated set of generic expansions already KNOWN to be wrong
        for a GPIL term (e.g. "General Product Inventory" for GPI,
        "Integrated Pest Management" for IPM) -- checked against the
        answer's own words directly, independent of citation. This exists
        because "unsupported_definition" above only checks whether SOME
        definitional-style sentence exists in the cited/whole evidence,
        never WHAT it says -- once a real glossary row is in scope (which
        it always is once the question names a known term), a fabricated
        WRONG definition that happens to cite that real, resolvable row
        would otherwise pass "unsupported_definition" cleanly. Prose-level
        scanner only (source="prose_fallback"); see
        scan_glossary_term_misuse(). Always severity="error".

    source distinguishes which mechanism caught it: "claim" (checked a
    specific AnswerClaim against its own citations) vs "prose_fallback"
    (the whole-answer word/direction/recommendation scanners, unscoped or
    citation-scoped to whatever a sentence happens to cite).

    severity distinguishes a real grounding failure ("error", the default --
    counts toward GroundingCheckResult.status="fail") from an informational
    specificity notice ("warning" -- surfaced to the caller/UI but never by
    itself flips status to "fail"). Every pre-existing issue_type keeps
    severity="error" so old behavior is unchanged; only "generic_aggregation"
    currently uses "warning".
    """

    issue_type: Literal[
        "unsupported_qualifier",
        "direction_contradiction",
        "unsupported_numeric",
        "unsupported_entity",
        "unsupported_period",
        "unsupported_delta",
        "unsupported_deviation",
        "unsupported_causal",
        "unsupported_recommendation",
        "unsupported_comparison",
        "generic_aggregation",
        "unsupported_definition",
        "unsupported_ranking",
        "incomplete_coverage",
        "unresolved_citation",
        "wrong_glossary_expansion",
    ]
    sentence: str
    term: str | None = None
    detail: str = ""
    source: Literal["claim", "prose_fallback"] = "prose_fallback"
    claim_id: int | str | None = None
    severity: Literal["error", "warning"] = "error"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingCheckResult:
    status: Literal["pass", "fail"]
    method: Literal["deterministic", "llm"]
    flagged_issues: list[GroundingIssue] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Query requirements / evidence sufficiency (Stage -1 -- runs BEFORE premise
# verification, deterministic, no I/O). See src/inference/query_requirements.py
# and src/inference/sku_evidence.py.
#
# WHAT THESE ARE FOR (plain language):
# PremiseCheckResult only ever answers "does the evidence support the
# DIRECTION this question assumes" -- it says nothing about whether the
# evidence is even at the right GRANULARITY to answer the question at all
# (e.g. a question asking to rank SKUs when the retrieved evidence is all
# State/Category-grain). QueryRequirements captures what the question ITSELF
# needs (computed from the question text alone, before any evidence is
# consulted); EvidenceSufficiencyResult captures whether evidence actually
# matching that requirement was found. Keeping these separate from
# PremiseCheckResult/GroundingCheckResult is deliberate -- see the module
# docstrings of query_requirements.py/sku_evidence.py.
# ---------------------------------------------------------------------------


@dataclass
class QueryRequirements:
    """What a question requires, computed by deterministic question-text
    analysis alone (query_requirements.py) -- no retrieval, no evidence
    lookup. required_granularity is None for the common case (a question
    shaped like the corpus's own default State/Category/Distributor grain,
    which every existing question already gets evidence for); it is "sku"
    when the question explicitly asks about SKUs, the one granularity this
    project's default document corpus does not expose (see sku_evidence.py).

    ranking_metric is the exact SKU metric name (one of
    fact_structuring.SKU_METRIC_FIELDS' labels -- Units Delivered, Revenue,
    Service Level, Numeric Distribution, Out-of-Stock Rate) the question
    itself names, or None when a SKU-ranking-shaped question (is_ranking
    True, required_granularity "sku") does not name one. This is what lets
    pipeline.py distinguish an answerable ranking question from an
    ambiguous one ("top 3 SKUs" with no metric named) BEFORE ever
    generating an answer -- see Design Decision: ranking must be
    deterministic, and an ambiguous ranking metric must be asked about,
    never silently guessed.

    rank_direction ("top" for a highest/max-shaped question, "bottom" for
    a lowest/min-shaped one, None when neither an unambiguous max- nor
    min-word is present) -- alongside ranking_metric and rank_n, this is
    what lets sku_evidence.evaluate_sku_evidence() compute the ranking
    DETERMINISTICALLY (sort real rows, take N) instead of handing the LLM
    a pile of SKU facts and trusting it to rank them correctly itself. See
    query_requirements.py's _extract_rank_direction() for the exact word
    lists (mirrors grounding_check.py's own post-hoc ranking-verification
    word sets, so both stages agree on what "top"/"bottom" mean)."""

    required_granularity: Literal["sku", None]
    target_state: str | None
    target_periods: list[str] = field(default_factory=list)
    target_year: int | None = None
    is_ranking: bool = False
    rank_n: int | None = None
    ranking_metric: str | None = None
    rank_direction: Literal["top", "bottom", None] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceSufficiencyResult:
    """Result of asking: 'was evidence actually found AT the granularity
    the question requires?'

    status meanings:
      - "not_applicable": required_granularity is None -- nothing extra to
        check; the ordinary premise/grounding path handles the question.
      - "sufficient": matching SKU evidence was found and resolved for the
        question's own state/period(s)/year. sku_source_rows carries the
        synthetic Sources-shaped records (id/text) to merge into
        context_records before generation -- see pipeline.py. When the
        question is also ranking-shaped with a known metric AND direction
        (QueryRequirements.is_ranking/ranking_metric/rank_direction all
        set), sku_source_rows additionally includes one authoritative
        "Deterministic SKU Ranking" record per resolved period (see
        sku_evidence.py's _build_ranking_source_row()), and ranked_skus
        carries the same ranking as plain structured data (rank/sku_name/
        metric/value per entry) for callers/tests/logging that want it
        without re-parsing prompt text.
      - "insufficient_scope": SKU granularity IS a real, supported
        representation in this project's data, but no matching rows exist
        for this question's own specific state/period/year (state not
        named/recognized, or the named period/year falls outside the
        indexed range). explanation distinguishes "the raw dataset has no
        SKU data" (never true in this project) from "the current
        aggregated representation has no rows for this specific
        state/period" (the actual, common case) -- see sku_evidence.py.
    """

    status: Literal["not_applicable", "sufficient", "insufficient_scope"]
    requirements: QueryRequirements
    sku_source_rows: list[dict] = field(default_factory=list)
    explanation: str = ""
    ranked_skus: list[dict] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("sku_source_rows", None)  # raw evidence text, not useful in a JSON log dump
        return d


@dataclass
class SourceCitation:
    """One resolved claim -> evidence mapping, for display (Problem 2 --
    source traceability): which citation id(s) a specific answer claim
    relied on, and the actual retrieved/available evidence text those ids
    point to. Built by grounding_check.resolve_claim_sources() AFTER
    grounding has already run, reusing the exact same citation-resolution
    machinery grounding_check.py uses to validate claims -- so a source
    shown here is never invented: every citation_id really is one this
    claim reported, and every evidence_text really is what context_records
    holds for it.
    """

    claim_id: int | str | None
    claim_text: str | None
    citation_ids: list[str]
    evidence_text: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Whole-pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """final_decision meanings:
      - "pass_through": premise ok, first draft passed grounding untouched.
      - "hedged": premise contradicted/unsupported -- no generation call made.
      - "insufficient_evidence": premise insufficient_data (no generation
        call made), OR grounding still failed after the one allowed retry
        (a generation call WAS made -- see retry_attempted/llm_calls_made).
      - "regenerated": first draft failed grounding, the one bounded retry
        fixed it, and the REVISED answer is what final_text holds.
      - "rejected": reserved for a grounding failure with no retry
        attempted. Not currently emitted by run_pipeline (every grounding
        failure gets exactly one retry per the Phase 7 bounded-retry
        design) -- kept in case a future caller wants to represent that
        state explicitly.
      - "needs_clarification": the question is SKU-ranking-shaped ("top 3
        SKUs...") but names no ranking metric, and this project has no
        confirmed business-default metric to guess -- short-circuits
        before premise/generation, zero LLM calls made, per the explicit
        "ask rather than guess" ranking-determinism requirement.

    grounding_check always holds the FIRST draft's grounding result.
    retry_grounding_check holds the retry's result, only when
    retry_attempted is True.
    """

    question: str
    premise_check: PremiseCheckResult
    answer: AnswerResult | None
    grounding_check: GroundingCheckResult | None
    final_decision: Literal[
        "pass_through", "hedged", "insufficient_evidence", "rejected", "regenerated",
        "needs_clarification",
    ]
    final_text: str
    # Completion calls made by run_pipeline itself (0, 1, or 2 -- never
    # more, per the "retry exactly once" rule). answer_question() adds +1
    # on top of this for the retrieval embedding call, which run_pipeline
    # doesn't know about.
    llm_calls_made: int
    retry_attempted: bool = False
    retry_answer: AnswerResult | None = None
    retry_grounding_check: GroundingCheckResult | None = None
    # Evidence-sufficiency/granularity check (Problems 3/4) -- None when the
    # question never triggered a granularity requirement (the common case).
    evidence_sufficiency: EvidenceSufficiencyResult | None = None
    # Resolved claim -> evidence provenance for the COMMITTED final answer
    # (Problem 2) -- always [] when no generation call was made (hedged/
    # insufficient_evidence with no draft), never fabricated.
    sources: list[SourceCitation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "premise_check": self.premise_check.to_dict(),
            "answer": self.answer.to_dict() if self.answer else None,
            "grounding_check": self.grounding_check.to_dict() if self.grounding_check else None,
            "retry_attempted": self.retry_attempted,
            "retry_answer": self.retry_answer.to_dict() if self.retry_answer else None,
            "retry_grounding_check": self.retry_grounding_check.to_dict() if self.retry_grounding_check else None,
            "final_decision": self.final_decision,
            "final_text": self.final_text,
            "llm_calls_made": self.llm_calls_made,
            "evidence_sufficiency": self.evidence_sufficiency.to_dict() if self.evidence_sufficiency else None,
            "sources": [s.to_dict() for s in self.sources],
        }
