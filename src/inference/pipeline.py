"""
Phase 7 -- pipeline orchestrator.

WHAT THIS FILE IS FOR (plain language):
This wires the individual Phase 7 stages together in the order the
architecture audit specified, now including the bounded single-retry step:

    Question -> GraphRAG retrieval -> Premise Verification
             -> (only if the premise passes) Answer Generation
             -> Grounding/Contradiction Check
             -> (only if that check fails) ONE Regeneration Attempt
             -> Grounding/Contradiction Check (again, on the revised answer)
             -> Final Answer

`run_pipeline()` is the pure orchestration logic -- it takes an
already-built QueryContext (see context.py) and injectable answer_fn /
retry_fn callables, and contains no I/O of its own. This is what the test
suite exercises directly with a stub QueryContext and mocked functions, so
the orchestration logic (does it skip generation when the premise fails?
does it retry exactly once, never more, when grounding fails? does it fail
closed if the retry doesn't fix things?) is fully testable without
touching GraphRAG, the index, or any API.

`answer_question()` is the thin real-world wrapper: it calls
context.build_query_context() (the one function that touches the index and
makes the embedding API call) and then hands off to run_pipeline().

WHY THE RETRY IS BOUNDED, NOT A LOOP:
A live test found a real, accurate, well-cited answer withheld entirely
over one unsupported adjective in its closing sentence. The fix is ONE
revise-and-recheck pass -- run_pipeline calls retry_fn at most once per
query, checks the result, and then commits to either the revised answer
(if it now passes) or a fail-closed "insufficient evidence" message (if it
still doesn't). There is no loop back into retry_fn under any
circumstance, keeping worst-case cost at exactly 2 completion calls.

WHAT'S STILL NOT HERE (see the Phase 7 architecture audit):
No LLM-escalation path for premise/grounding checks that come back
ambiguous -- both stages remain fully deterministic. That's a separate,
not-yet-built Phase 7b follow-up, not something this retry addresses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd
from graphrag.query.context_builder.builders import ContextBuilderResult

from src.inference.answer import generate_answer, regenerate_answer
from src.inference.context import QueryContext, build_query_context
from src.inference.fact_structuring import SKU_METRIC_FIELDS, extract_atomic_facts
from src.inference.grounding_check import check_grounding, resolve_claim_sources
from src.inference.knowledge_layer import build_glossary_source_rows
from src.inference.premise_check import check_premise
from src.inference.query_requirements import detect_query_requirements
from src.inference.schemas import (
    AnswerResult,
    EvidenceSufficiencyResult,
    GroundingCheckResult,
    PipelineResult,
    PremiseCheckResult,
    QueryRequirements,
)
from src.inference.sku_evidence import evaluate_sku_evidence, known_sku_names

AnswerFn = Callable[..., Awaitable[AnswerResult]]
RetryFn = Callable[..., Awaitable[AnswerResult]]


def _render_hedge(premise: PremiseCheckResult) -> str:
    """Deterministic, zero-cost text for when the premise check didn't
    pass -- states what the evidence DOES show instead of ever asking the
    model to explain a claim the evidence doesn't establish."""
    lines: list[str] = []
    if premise.status == "undefined_metric":
        lines.append(premise.explanation)
        return "\n".join(lines)
    if premise.status == "insufficient_data":
        lines.append(
            f"The indexed evidence does not include a {premise.baseline_period} document "
            f"for {premise.target_state}, so a claim that performance {premise.claimed_direction}d "
            f"in {premise.target_period} cannot be evaluated against its actual prior period."
        )
        if premise.supplementary_comparisons:
            lines.append("For reference, the periods that ARE indexed show:")
            for c in premise.supplementary_comparisons:
                lines.append(f"- {c.metric}: {c.value_a} ({c.period_a}) -> {c.value_b} ({c.period_b}), i.e. {c.direction}.")
    else:  # "contradicted" or "unsupported"
        lines.append(
            f"The indexed evidence does not clearly establish that {premise.target_state}'s "
            f"performance {premise.claimed_direction}d in {premise.target_period}."
        )
        lines.append(f"Comparing {premise.baseline_period} to {premise.target_period}, the evidence instead shows:")
        for c in premise.metric_comparisons:
            lines.append(f"- {c.metric}: {c.value_a} ({c.period_a}) -> {c.value_b} ({c.period_b}), i.e. {c.direction}.")
    return "\n".join(lines)


def _augment_context_with_sku_evidence(qctx: QueryContext, sku_source_rows: list[dict]) -> QueryContext:
    """Return a NEW QueryContext whose context_records["sources"] includes
    the given synthetic SKU evidence rows (from sku_evidence.py) appended
    to whatever GraphRAG itself retrieved -- never mutates qctx in place,
    so a caller holding the original object (e.g. a test) is unaffected.
    context_chunks (GraphRAG's own formatted prompt text) is left
    untouched -- the SKU facts reach the generation prompt entirely
    through fact_structuring.build_atomic_facts_block() reading the
    augmented context_records["sources"], the SAME additive mechanism the
    existing distributor/state/category Atomic Facts already use (see
    answer.py's _context_data_with_facts()). This is also why
    premise_check/grounding_check/resolve_claim_sources all keep working
    unmodified on the augmented context: they only ever read
    context_records["sources"], never context_chunks directly."""
    original_records = qctx.context_records
    original_sources = original_records.get("sources")
    new_rows = pd.DataFrame(sku_source_rows)
    if original_sources is not None and not original_sources.empty:
        merged_sources = pd.concat([original_sources, new_rows], ignore_index=True)
    else:
        merged_sources = new_rows
    merged_records = {**original_records, "sources": merged_sources}
    new_context_result = ContextBuilderResult(
        context_chunks=qctx.context_result.context_chunks,
        context_records=merged_records,
    )
    return QueryContext(engine=qctx.engine, context_result=new_context_result)


def _augment_context_with_glossary(qctx: QueryContext, glossary_rows: list[dict]) -> QueryContext:
    """Same additive merge as _augment_context_with_sku_evidence() above,
    duplicated (not shared) for the GPIL Knowledge Layer's glossary rows --
    matching this project's own established convention of small,
    separately-named additive-merge helpers rather than one shared helper
    multiple callers depend on (see e.g. query_requirements.py's duplicated
    ranking word-lists). glossary_rows use a disjoint "glossary-"-prefixed
    id namespace (knowledge_layer.py) from every other synthetic Sources
    row this pipeline creates ("sku-...", "sku-ranking-..."), so this never
    collides with or overwrites SKU evidence merged separately below."""
    original_records = qctx.context_records
    original_sources = original_records.get("sources")
    new_rows = pd.DataFrame(glossary_rows)
    if original_sources is not None and not original_sources.empty:
        merged_sources = pd.concat([original_sources, new_rows], ignore_index=True)
    else:
        merged_sources = new_rows
    merged_records = {**original_records, "sources": merged_sources}
    new_context_result = ContextBuilderResult(
        context_chunks=qctx.context_result.context_chunks,
        context_records=merged_records,
    )
    return QueryContext(engine=qctx.engine, context_result=new_context_result)


_SOURCE_STATE_HEADER_RE = re.compile(r"^State:\s*(?P<state>[^\n]+)")


def _scope_sources_to_state(qctx: QueryContext, target_state: str) -> QueryContext:
    """Drop retrieved Sources rows whose OWN 'State: X' header names a
    DIFFERENT state than `target_state` -- local_search query-scope
    narrowing (see QueryRequirements.query_intent's docstring). A live run
    asking about one specific Gujarat distributor had GraphRAG return 10
    Sources rows, only 3 of them about Gujarat -- the other 7 (Odisha,
    Chhattisgarh, Uttarakhand) sat in the same prompt as harmless-looking
    but structurally-identical distributor-deviation sentences ("...showed
    a significant deviation on Out-of-Stock Rate...: 20.0% vs. the state
    average of 8.0%..."), and the model borrowed one of THEIR numbers for
    a Gujarat distributor. Narrowing to the named state before generation
    removes that distractor pool entirely, rather than relying only on
    grounding to catch the misattribution after the fact.

    Never drops a row with NO parseable state header at all (glossary-/
    sku-/named-period-doc- synthetic rows, or a malformed/chunked row that
    lost its header) -- those aren't real competing-state documents, and
    dropping them would silently break the SKU/glossary/named-period-doc
    evidence mechanisms, which already scope themselves correctly on their
    own. Never narrows to zero rows: if filtering would remove every row
    (the named state genuinely wasn't retrieved at all), this returns qctx
    UNCHANGED -- premise_check's own existing "no evidence for this state"
    handling already covers that case correctly; silently emptying
    Sources here would just turn a normal insufficient-data outcome into a
    confusing crash-adjacent state."""
    sources = qctx.context_records.get("sources")
    if sources is None or sources.empty or "text" not in sources.columns:
        return qctx

    def _keep(text: str) -> bool:
        m = _SOURCE_STATE_HEADER_RE.match(str(text))
        return m is None or m.group("state").strip() == target_state

    mask = sources["text"].map(_keep)
    if not mask.any():
        return qctx
    filtered_sources = sources[mask].reset_index(drop=True)
    if len(filtered_sources) == len(sources):
        return qctx

    merged_records = {**qctx.context_records, "sources": filtered_sources}
    new_context_result = ContextBuilderResult(
        context_chunks=qctx.context_result.context_chunks,
        context_records=merged_records,
    )
    return QueryContext(engine=qctx.engine, context_result=new_context_result)


def _render_category_breakdown(
    qctx: QueryContext, state: str, period: str, metric: str
) -> str | None:
    """Deterministic, zero-cost text for a category-scoped-ONLY metric
    (Out-of-Stock Rate/Numeric Distribution/ACV/Range Billing) asked about
    with NO category named -- e.g. "What was the out of stock in Goa in
    June 2025?". This metric genuinely has no single state-wide value in
    this corpus (it's measured per GPI/IPM/Ferrero/Candy category, always)
    -- per the "Missing categories are NEVER guessed from retrieved
    evidence" rule, this reports every category's ACTUAL retrieved value
    side by side instead of the model arbitrarily leading with whichever
    category happened to be nearest in the retrieved text (a live run did
    exactly that -- silently answered with just Candy's figure).

    Returns None (never an empty/misleading string) when no matching
    category-level Atomic Facts exist for this exact state+period+metric
    in the retrieved evidence -- callers must fall through to the normal
    pipeline (premise/generation/grounding) in that case, since this isn't
    a "nothing exists" situation this function can speak to; it only
    handles "evidence exists but is split across categories"."""
    facts = [
        f
        for f in extract_atomic_facts(qctx.context_records)
        if f.kind == "category" and f.state == state and f.period == period and f.metric == metric
    ]
    if not facts:
        return None
    lines = [
        f"{metric} is only measured per product category in this system, not as a single "
        f"state-wide figure -- the question named no category, so every category's actual "
        f"retrieved value for {state} in {period} is reported below rather than picking one:",
    ]
    for f in sorted(facts, key=lambda x: x.category or ""):
        lines.append(f"- {f.category}: {f.value}%")
    return "\n".join(lines)


def _render_clarification(requirements: QueryRequirements) -> str:
    """Deterministic, zero-cost text for an ambiguous SKU-ranking question
    ("top 3 SKUs in Punjab in 2025", no metric named) -- this project has
    no confirmed business-default "top SKU" metric (Revenue vs. Units vs.
    Service Level are all real, equally legitimate SKU KPIs), so per the
    explicit "ask rather than guess" ranking-determinism requirement, the
    pipeline asks which metric to rank by instead of letting the
    generation model pick one silently. Names the exact metric strings a
    follow-up question can use, taken directly from
    fact_structuring.SKU_METRIC_FIELDS -- the same names ranking_metric
    detection (query_requirements.py) and ranking verification
    (grounding_check.py:verify_sku_ranking_claim()) already recognize."""
    metric_names = ", ".join(label for _, label in SKU_METRIC_FIELDS)
    rank_desc = f"top {requirements.rank_n}" if requirements.rank_n else "ranking"
    return (
        f"This is a SKU {rank_desc} question, but no ranking metric was named, and this "
        f"project has no single confirmed default for \"top SKU\" (Revenue, Units Delivered, "
        f"Service Level, Numeric Distribution, and Out-of-Stock Rate are all real, distinct "
        f"SKU-level metrics that could rank differently). Please specify which metric to rank "
        f"by -- one of: {metric_names}."
    )


def _placeholder_premise_for_clarification() -> PremiseCheckResult:
    """A "no_claim"-shaped PremiseCheckResult for the needs_clarification
    short-circuit, which returns BEFORE premise checking ever runs (the
    question's ranking ambiguity is a pure question-shape issue, unrelated
    to whether any directional claim needs verifying). Mirrors
    premise_check.py's own "no_claim" construction exactly, so this isn't
    a new shape PipelineResult consumers have to special-case."""
    return PremiseCheckResult(
        status="no_claim",
        method="deterministic",
        claimed_direction=None,
        target_state=None,
        target_period=None,
        baseline_period=None,
        explanation="Not evaluated -- the question was short-circuited for ranking-metric clarification before premise checking.",
    )


def _render_granularity_gap(evidence: EvidenceSufficiencyResult) -> str:
    """Deterministic, zero-cost text for a granularity/scope mismatch
    (Problems 3+4): the question requires SKU-grain evidence -- this
    project's data genuinely supports that (see sku_evidence.py) -- but no
    matching rows exist for exactly what this question asked (state not
    named/recognized, or the named period/year is outside the indexed
    range). Deliberately just returns evidence.explanation rather than
    re-deriving a message here: sku_evidence.evaluate_sku_evidence()
    already writes that explanation with the required distinction built in
    (raw dataset HAS SKU data vs. this specific state/period has no rows),
    per Problem 3's explicit instruction not to conflate the two."""
    return evidence.explanation


def _render_retry_failure(initial: GroundingCheckResult, retry: GroundingCheckResult) -> str:
    """Deterministic text for when the ONE allowed retry still didn't
    produce a clean answer -- fail closed rather than ever showing an
    answer that didn't pass the grounding check, per the project's
    'insufficient evidence over invented explanation' rule."""
    lines = [
        "A drafted answer was produced and revised once to fix grounding issues, "
        "but the revised answer still did not pass the grounding check, so no answer is being returned.",
        f"Initial draft had {len(initial.flagged_issues)} issue(s); after one revision, "
        f"{len(retry.flagged_issues)} issue(s) remain:",
    ]
    for issue in retry.flagged_issues:
        lines.append(f"- [{issue.issue_type}] {issue.detail}")
    return "\n".join(lines)


def run_pipeline(
    question: str,
    qctx: QueryContext,
    answer_fn: AnswerFn = generate_answer,
    retry_fn: RetryFn = regenerate_answer,
    data_dir: Path | None = None,
) -> PipelineResult:
    """Pure orchestration:

        (maybe) GPIL Knowledge Layer glossary merge -> query requirements
            -> (maybe) SKU evidence sufficiency check
            -> (maybe) ranking-clarification short-circuit
            -> premise check -> (maybe) answer generation
            -> grounding check -> (maybe, at most once) regeneration
            -> grounding check again -> provenance resolution
            -> final decision.

    GPIL KNOWLEDGE LAYER (2026-08-23): before anything else,
    knowledge_layer.build_glossary_source_rows(question) deterministically
    detects any known GPIL business term (GPI/GPIL, IPM, Ferrero, Candy,
    Distributor/WD, Dealer, Hero SKU, and a handful of established KPI
    names) named in the question and, if any matched, merges their
    definitions into qctx as additional synthetic Sources evidence (see
    _augment_context_with_glossary()) -- the SAME additive mechanism SKU
    evidence uses below, so premise_check/generation/grounding all see it
    as ordinary, citable evidence with no special-casing. This runs
    unconditionally (cheap, pure question-text matching, no I/O) and
    independently of required_granularity -- a glossary term and a
    SKU/analytical requirement can both be present in the same question
    (e.g. "What is GPI and how did it perform in Gujarat in April 2026?").

    Takes an already-built QueryContext so this function makes no API
    calls of its own beyond whatever `answer_fn`/`retry_fn` do -- fully
    testable with a stub QueryContext and mocked functions. `data_dir`
    (default: sku_evidence.DEFAULT_DATA_DIR, i.e. this project's own
    data/ directory) is read via known_sku_names() on EVERY call now
    (2026-08-23 stabilization pass, SKU-name-only queries fix) -- one
    small (~59-row) CSV read, so a bare-name SKU question ("the revenue of
    Marlboro Pack 1...") can be recognized as SKU-grain even without the
    literal word "SKU"; see query_requirements.py's module docstring for
    why this couldn't stay a pure question-text check. known_sku_names()
    degrades to an empty set (never raises) when data_dir has no SKU CSV,
    so this stays safe to call with a data_dir that doesn't have SKU data
    at all -- the rest of evaluate_sku_evidence() below is UNCHANGED and
    still only runs (and only then reads the full table) when
    required_granularity actually resolves to "sku".

    QUERY REQUIREMENTS / EVIDENCE SUFFICIENCY (Problems 1/3/4): computed
    BEFORE premise verification, deterministically, from the question text
    alone -- see query_requirements.py. When the question requires SKU-grain
    evidence, sku_evidence.evaluate_sku_evidence() resolves it against
    data/kpi_state_month_sku.csv:
      - no matching rows found ("insufficient_scope") -> short-circuit with
        an EXPLICIT explanation of the scope gap (never a generic "I don't
        have information" refusal, never a hallucinated answer, and never
        conflated with "the raw dataset has no SKU data" -- see
        sku_evidence.py) -- zero generation calls made.
      - matching rows found ("sufficient") -> merged into qctx as
        additional Sources evidence (see _augment_context_with_sku_evidence)
        BEFORE premise/generation/grounding run, so the rest of this
        function proceeds completely unchanged, just with real SKU
        evidence now present to cite. This keeps premise_check() focused
        on premise verification and grounding_check() focused on
        grounding -- neither function was changed to know anything special
        about SKU questions; they just see more Sources rows than before.

    PROVENANCE (Problem 2): once a final answer is committed to (whether
    the first draft or the retry), its claims' citations are resolved to
    real evidence text via grounding_check.resolve_claim_sources() and
    attached as PipelineResult.sources -- never fabricated, never attached
    when no generation happened (hedged/insufficient_evidence with no
    draft always get sources=[], the dataclass default).

    EVIDENCE-TRIGGERED GLOSSARY MERGE (2026-08-23, SKU live validation
    pass): the GPIL Knowledge Layer glossary merge above only ever scans
    the QUESTION text -- so a question like "Did the top-selling SKU in
    Gujarat change between April 2026 and June 2026?" never mentions "IPM"
    or "GPI" and gets no glossary rows at all. But EVERY SKU Atomic Fact
    sentence names its own category (GPI/IPM/Ferrero/Candy), and a live
    run found the model spontaneously re-expanding "IPM" as "Integrated
    Pest Management" while discussing a Marlboro SKU's IPM category label
    -- caught correctly by grounding (scan_glossary_term_misuse()), but
    only after wasting the one bounded retry, since the SAME gap applied
    to both attempts (the retry gets the exact same merged context). Once
    real SKU evidence is resolved, this second pass re-runs the SAME
    build_glossary_source_rows() detection against the EVIDENCE TEXT
    itself (not the question), catching any category name the evidence
    will surface even when the question itself never said it -- purely
    additive, deduplicated against whatever the question-text pass already
    matched, and using the exact same knowledge_layer.py detection/
    rendering this whole feature already relies on (no new glossary
    mechanism, no question-specific hardcoding).
    """
    glossary_rows = build_glossary_source_rows(question)
    glossary_row_ids = {row["id"] for row in glossary_rows}
    if glossary_rows:
        qctx = _augment_context_with_glossary(qctx, glossary_rows)

    requirements = detect_query_requirements(question, known_sku_names=known_sku_names(data_dir))

    # LOCAL_SEARCH QUERY-SCOPE NARROWING (basic/local/global routing):
    # only when the question is local_search-shaped (one named state, not
    # a broad/cross-state/trend-shaped question -- see
    # QueryRequirements.query_intent's docstring) do we narrow retrieved
    # Sources evidence down to that one state. global_search/basic_search
    # questions are left exactly as GraphRAG's own retrieval returned them
    # -- a broad question needs that natural breadth, not a single state's
    # worth of evidence. See _scope_sources_to_state()'s own docstring for
    # the live misattribution failure this fixes.
    if requirements.query_intent == "local_search" and requirements.target_state:
        qctx = _scope_sources_to_state(qctx, requirements.target_state)

    # CATEGORY-AMBIGUITY SHORT-CIRCUIT (Missing categories are NEVER
    # guessed): a category-scoped-ONLY metric (Out-of-Stock Rate/Numeric
    # Distribution/ACV/Range Billing) named with NO explicit category, but
    # WITH a specific state+period to look up -- deterministically report
    # every category's real retrieved value instead of letting generation
    # arbitrarily lead with whichever category's sentence sits nearest in
    # the retrieved text. Zero LLM calls. Requires target_state AND at
    # least one target_period to even attempt a lookup (matches
    # _render_category_breakdown()'s own state+period-scoped contract);
    # without both, this falls straight through to the normal pipeline
    # unchanged -- there's no single state+period to break down.
    if requirements.category_ambiguous and requirements.target_state and requirements.target_periods:
        breakdown = _render_category_breakdown(
            qctx, requirements.target_state, requirements.target_periods[0], requirements.category_scoped_metric
        )
        if breakdown is not None:
            return PipelineResult(
                question=question,
                premise_check=_placeholder_premise_for_clarification(),
                answer=None,
                grounding_check=None,
                final_decision="needs_clarification",
                final_text=breakdown,
                llm_calls_made=0,
                query_requirements=requirements,
            )

    evidence_sufficiency: EvidenceSufficiencyResult | None = None

    if requirements.required_granularity == "sku":
        evidence_sufficiency = evaluate_sku_evidence(requirements, data_dir=data_dir)
        if evidence_sufficiency.status == "sufficient":
            qctx = _augment_context_with_sku_evidence(qctx, evidence_sufficiency.sku_source_rows)
            evidence_glossary_rows = [
                row
                for row in build_glossary_source_rows(
                    " ".join(r["text"] for r in evidence_sufficiency.sku_source_rows)
                )
                if row["id"] not in glossary_row_ids
            ]
            if evidence_glossary_rows:
                qctx = _augment_context_with_glossary(qctx, evidence_glossary_rows)

    # Ambiguous SKU-ranking short-circuit -- checked AFTER evidence
    # sufficiency (above) but BEFORE premise/generation, so it costs zero
    # LLM calls. Deliberately ordered behind the insufficient_scope check
    # below (not in front of it): if the state/period named has no SKU
    # data at all, that is the more fundamental problem, and asking "which
    # metric?" first would just be a wasted round trip before the user
    # hits the same missing-evidence wall regardless of their answer. Only
    # once real evidence is confirmed to exist does an unnamed ranking
    # metric become the actual blocker worth asking about. See
    # _render_clarification()'s docstring for why this project asks rather
    # than guesses a default metric.
    if (
        requirements.required_granularity == "sku"
        and requirements.is_ranking
        and requirements.ranking_metric is None
        and (evidence_sufficiency is None or evidence_sufficiency.status == "sufficient")
    ):
        return PipelineResult(
            question=question,
            premise_check=_placeholder_premise_for_clarification(),
            answer=None,
            grounding_check=None,
            final_decision="needs_clarification",
            final_text=_render_clarification(requirements),
            llm_calls_made=0,
            evidence_sufficiency=evidence_sufficiency,
            query_requirements=requirements,
        )

    premise = check_premise(question, qctx.context_records)

    if evidence_sufficiency is not None and evidence_sufficiency.status == "insufficient_scope":
        return PipelineResult(
            question=question,
            premise_check=premise,
            answer=None,
            grounding_check=None,
            final_decision="insufficient_evidence",
            final_text=_render_granularity_gap(evidence_sufficiency),
            llm_calls_made=0,
            evidence_sufficiency=evidence_sufficiency,
            query_requirements=requirements,
        )

    if premise.status in ("contradicted", "unsupported", "insufficient_data", "undefined_metric"):
        decision = "insufficient_evidence" if premise.status in ("insufficient_data", "undefined_metric") else "hedged"
        return PipelineResult(
            question=question,
            premise_check=premise,
            answer=None,
            grounding_check=None,
            final_decision=decision,
            final_text=_render_hedge(premise),
            llm_calls_made=0,
            evidence_sufficiency=evidence_sufficiency,
            query_requirements=requirements,
        )

    # premise.status is "no_claim" or "supported" -- safe to generate.
    answer = asyncio.run(answer_fn(qctx.engine, question, qctx.context_result))
    grounding = check_grounding(
        answer.text, qctx.context_records, answer.claims, premise.metric_comparisons or None, question
    )

    if grounding.status == "pass":
        return PipelineResult(
            question=question,
            premise_check=premise,
            answer=answer,
            grounding_check=grounding,
            final_decision="pass_through",
            final_text=answer.text,
            llm_calls_made=answer.llm_calls,
            evidence_sufficiency=evidence_sufficiency,
            query_requirements=requirements,
            sources=resolve_claim_sources(answer.claims, qctx.context_records),
        )

    # Grounding failed -- exactly ONE retry, using the SAME retrieved
    # context (no second build_context() call, no re-embedding).
    retry_answer = asyncio.run(
        retry_fn(qctx.engine, question, qctx.context_result, answer, grounding)
    )
    retry_grounding = check_grounding(
        retry_answer.text, qctx.context_records, retry_answer.claims, premise.metric_comparisons or None, question
    )
    completion_calls = answer.llm_calls + retry_answer.llm_calls  # == 2

    if retry_grounding.status == "pass":
        return PipelineResult(
            question=question,
            premise_check=premise,
            answer=answer,
            grounding_check=grounding,
            retry_attempted=True,
            retry_answer=retry_answer,
            retry_grounding_check=retry_grounding,
            final_decision="regenerated",
            final_text=retry_answer.text,
            llm_calls_made=completion_calls,
            evidence_sufficiency=evidence_sufficiency,
            query_requirements=requirements,
            sources=resolve_claim_sources(retry_answer.claims, qctx.context_records),
        )

    # Retry still failed -- fail closed. Never loop back into retry_fn.
    return PipelineResult(
        question=question,
        premise_check=premise,
        answer=answer,
        grounding_check=grounding,
        retry_attempted=True,
        retry_answer=retry_answer,
        retry_grounding_check=retry_grounding,
        final_decision="insufficient_evidence",
        final_text=_render_retry_failure(grounding, retry_grounding),
        llm_calls_made=completion_calls,
        evidence_sufficiency=evidence_sufficiency,
        query_requirements=requirements,
    )


def answer_question(
    question: str,
    index_root: Path,
    output_dir: Path,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
    reporting_dir: Path | None = None,
    answer_fn: AnswerFn = generate_answer,
    retry_fn: RetryFn = regenerate_answer,
    data_dir: Path | None = None,
) -> PipelineResult:
    """Real-world entry point: builds retrieval context against a live
    index (1 embedding API call) and runs the guard pipeline on top of it.
    This is what scripts/CLI callers should use; tests should use
    run_pipeline() directly with a stub QueryContext instead, to avoid
    hitting the index/API at all. `data_dir` is forwarded to run_pipeline()
    (default: sku_evidence.DEFAULT_DATA_DIR) -- see that function's
    docstring; unrelated to index_root/output_dir, which are GraphRAG's own
    paths, not this project's data/ directory.
    """
    qctx = build_query_context(
        index_root=index_root,
        output_dir=output_dir,
        query=question,
        community_level=community_level,
        response_type=response_type,
        reporting_dir=reporting_dir,
    )
    result = run_pipeline(question, qctx, answer_fn=answer_fn, retry_fn=retry_fn, data_dir=data_dir)
    # +1 for the embedding call build_query_context always makes, which
    # run_pipeline() doesn't know about since it only sees the QueryContext
    # after the fact.
    result.llm_calls_made += 1
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one question through the Phase 7 guarded pipeline against a GraphRAG index."
    )
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--index-root", type=Path, required=True, help="Directory containing settings.yaml.")
    parser.add_argument("--data", type=Path, required=True, help="Index output directory (parquet files).")
    parser.add_argument("--community-level", type=int, default=2)
    parser.add_argument("--reporting-dir", type=Path, default=None, help="Where to write query.log (default: index's own logs/).")
    args = parser.parse_args()

    result = answer_question(
        question=args.query,
        index_root=args.index_root,
        output_dir=args.data,
        community_level=args.community_level,
        reporting_dir=args.reporting_dir,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    _main()
