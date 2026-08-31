"""
Phase 7, Stage 0 -- deterministic premise verification.

WHAT THIS FILE IS FOR (plain language):
This is the guard that would have caught the "Why did Bihar's performance
decline in October 2025?" failure: before we ever let an LLM write a "why"
answer, we first ask a narrower, checkable question -- does the retrieved
evidence actually establish the direction ("decline") the user's question
assumes? If it doesn't, we never call the answer-generation model at all;
we hand back what the evidence DOES show instead.

WHY THIS CAN BE DONE WITHOUT AN LLM:
Phase 4's document generator (src/graph/build_documents.py) writes every
state x month document from a fixed sentence template -- "Productivity for
Bihar in October 2025 was 85.5%." -- not free-form prose. That means the
exact numbers for each metric/period are regex-extractable, verbatim, from
the retrieved source text units (context_records["sources"]). No model call
is needed to read a number back out of a sentence it was written into by a
template. This is the "prefer deterministic checks whenever the evidence is
structured" principle from the Phase 7 architecture audit.

WHAT THIS FILE DELIBERATELY DOES NOT DO YET:
There's no LLM-escalation path implemented in this pass -- per instruction,
this is "starting with deterministic premise verification." A case the
deterministic logic can't resolve currently just comes back as
status="unsupported" or "insufficient_data" (the safe default), rather than
falling back to a model call. See the pipeline.py docstring for where an
LLM fallback would plug in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.inference.schemas import MetricComparison, PremiseCheckResult

# The six state-level metrics Phase 4 writes into every document's opening
# paragraph, in a fixed "<Metric> for <State> in <Period> was <value>"
# sentence (Dropsize additionally carries a parenthetical aside, handled by
# the regex below).
STATE_METRIC_NAMES = [
    "Productivity",
    "Service Level",
    "SKUs per Transaction",
    "Dropsize",
    "Inventory Turns",
    "Inventory Days",
]

# When a question talks about "performance" generically instead of naming a
# specific metric, these are the two headline indicators to check -- the
# same two the LLM itself led with in both of our live test runs.
HEADLINE_METRICS = ["Productivity", "Service Level"]

DECLINE_WORDS = {
    "decline", "declined", "declining", "declines",
    "drop", "dropped", "dropping", "drops",
    "fall", "fell", "falling", "falls",
    "worse", "worsen", "worsened", "worsening",
    "deteriorate", "deteriorated", "deterioration", "deteriorating",
    "down", "weak", "weaker", "poor", "poorer",
    "underperform", "underperformed", "underperforming",
    "slump", "slumped", "slip", "slipped", "slipping",
}
IMPROVE_WORDS = {
    "improve", "improved", "improving", "improves",
    "grow", "grew", "growing", "grows",
    "better", "increase", "increased", "increasing", "increases",
    "up", "strong", "stronger", "gain", "gained", "gains",
    "rise", "rose", "rising", "rises",
    "outperform", "outperformed", "outperforming",
    "surge", "surged", "surging",
}

# Superlative/ranking phrasing this project defines no metric or composite
# score for, and no full-population comparable aggregate to compute one
# from (see check_premise()'s "undefined_metric" branch) -- "<superlative>
# performing" only, deliberately narrow so this never fires on a question
# that names a real metric explicitly (e.g. "highest Productivity", which
# this pattern does not match at all).
_SUPERLATIVE_PERFORMING_RE = re.compile(
    r"\b(least|worst|lowest|best|highest)\s+performing\b", re.IGNORECASE
)


def extract_superlative_ranking_claim(question: str) -> str | None:
    """Return the matched superlative phrase (e.g. 'least performing') if
    the question asks a ranking/superlative question across an unnamed
    group ("which state was the X performing"), else None. Only matches
    the literal "<word> performing" shape -- a question naming an actual
    metric (e.g. "which state had the highest Productivity") is a
    different, answerable question shape and must NOT match here."""
    m = _SUPERLATIVE_PERFORMING_RE.search(question)
    if not m:
        return None
    return f"{m.group(1).lower()} performing"


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
_PERIOD_IN_QUESTION_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})\b", re.IGNORECASE
)

# 'January and February 2026' / 'January to February 2026' -- a common
# two-month-range phrasing where the year is written only once, after the
# SECOND month. _PERIOD_IN_QUESTION_RE requires an explicit year
# immediately after EACH month, so without recognizing this shape the
# first month is silently dropped entirely -- live-caught in the 15-query
# SKU validation pass ("Compare the top-selling SKU in Punjab between
# January and February 2026" only ever resolved "February 2026", so the
# SKU evidence lookup never got January's data and the answer wrongly
# claimed January was unavailable, even though it's indexed).
_SHARED_YEAR_PERIOD_RANGE_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(and|to|through)\s+(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)


def _expand_shared_year_period_ranges(question: str) -> str:
    """'January and February 2026' -> 'January 2026 and February 2026' --
    injects the year after the first month too, so _PERIOD_IN_QUESTION_RE
    (which requires each month to have its OWN adjacent year) picks up
    both. Purely a preprocessing step used only by
    extract_all_periods_from_question() -- never changes what's returned,
    only makes an implicit shared year explicit; a question with no such
    range phrase passes through completely unchanged."""
    return _SHARED_YEAR_PERIOD_RANGE_RE.sub(
        lambda m: f"{m.group(1)} {m.group(4)} {m.group(2)} {m.group(3)} {m.group(4)}",
        question,
    )

# Matches the fixed "State: X\nPeriod: Y" header every Phase 4 document
# opens with (see src/graph/build_documents.py).
_SOURCE_HEADER_RE = re.compile(r"State:\s*(?P<state>[^\n]+)\s*\n\s*Period:\s*(?P<period>[^\n]+)")


# A live multi-period question ("...what new deviations did Gamble-Wright
# PICK UP instead?") was misclassified as an "improve" claim: bare token
# matching saw the word "up" (a real IMPROVE_WORDS entry, needed for
# legitimate uses like "Service Level was up") inside the idiom "pick up"
# ("acquire/gain", nothing to do with a metric direction) and gated
# generation on a decline/improve premise the question never actually
# asserted. A second live question ("...show up with issues...") hit the
# SAME class of bug via a DIFFERENT idiom ("show up", not "pick up") --
# proving the original fix (one idiom, hardcoded) was too narrow: bare
# "up"/"down" token matching is fragile against ANY phrasal verb built on
# those words, not just the one instance that happened to be caught live.
# Fix, widened but still a closed, curated list (never a general NLP
# solution -- this project's own "grammar, not guessing" discipline):
# every common English phrasal verb built on "up"/"down" that has NOTHING
# to do with a metric rising or falling, neutralized before token
# matching. Real direction language ("Service Level was up", "Productivity
# fell down to 70%") is completely unaffected, since none of these exact
# phrases appear in genuine metric-direction sentences.
_PHRASAL_VERB_NEUTRALIZE_RE = re.compile(
    r"\b(?:"
    r"pick(?:ed|ing)?|show(?:ed|ing|s)?|turn(?:ed|ing|s)?|end(?:ed|ing|s)?|wind(?:ing)?|wound|"
    r"sum(?:med|ming|s)?|catch(?:ing|es)?|caught|back(?:ed|ing|s)?|wrap(?:ped|ping|s)?|"
    r"step(?:ped|ping|s)?|keep(?:ing|s)?|kept|hold(?:ing|s)?|held|line(?:d|ing|s)?|"
    r"set(?:ting|s)?|open(?:ed|ing|s)?|come(?:s|ing)?|came"
    r")\s+up\b"
    r"|"
    r"\b(?:"
    r"turn(?:ed|ing|s)?|let(?:ting|s)?|calm(?:ed|ing|s)?|settle(?:d|ing|s)?|shut(?:ting|s)?|"
    r"close(?:d|ing|s)?|write(?:s|ing)?|wrote|break(?:ing|s)?|broke"
    r")\s+down\b",
    re.IGNORECASE,
)


def extract_direction_claim(question: str) -> str:
    """Return 'decline', 'improve', or 'neutral' depending on which
    direction words (if any) appear in the question. If BOTH decline-words
    and improve-words appear (a genuinely mixed/ambiguous question), or
    neither appears, this returns 'neutral' -- we only ever gate on a claim
    we can unambiguously identify."""
    neutralized = _PHRASAL_VERB_NEUTRALIZE_RE.sub(" ", question)
    tokens = set(re.findall(r"[a-z]+", neutralized.lower()))
    has_decline = bool(tokens & DECLINE_WORDS)
    has_improve = bool(tokens & IMPROVE_WORDS)
    if has_decline and not has_improve:
        return "decline"
    if has_improve and not has_decline:
        return "improve"
    return "neutral"


def extract_metric_claim(question: str) -> str | None:
    """If the question names one of the known state-level metrics
    explicitly (e.g. 'why did Service Level decline'), return that metric
    name so the check narrows to it instead of the generic headline pair."""
    lower = question.lower()
    for name in STATE_METRIC_NAMES:
        if name.lower() in lower:
            return name
    return None


# The 28 Indian states this project's synthetic data covers (see
# src/data_gen/generate_synthetic_data.py's INDIAN_STATES) -- duplicated
# here as a plain constant, not an import, so this module's only dependency
# stays on the fixed set of state names every document uses verbatim,
# rather than pulling the data-generation module (and its heavier
# faker/numpy import chain) into the live inference path.
INDIAN_STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
)
# Longest names first so a state name that happens to be a substring of
# another (none currently collide, e.g. "Uttar Pradesh" vs. "Uttarakhand"
# don't overlap as whole words) can never be shadowed by a shorter match.
_INDIAN_STATES_BY_LENGTH = sorted(INDIAN_STATES, key=len, reverse=True)


def extract_state_from_question(question: str) -> str | None:
    """If the question explicitly names one of the 28 Indian states this
    project's data covers, return that state's canonical name -- trusted
    over whatever state the top-ranked retrieved document happens to be,
    the same way extract_period_from_question() below is already trusted
    over retrieval for the period. Whole-word match (via \\b) so a state
    name can't be matched as a bare substring of an unrelated word."""
    for name in _INDIAN_STATES_BY_LENGTH:
        if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE):
            return name
    return None


def extract_period_from_question(question: str) -> str | None:
    """Pull an explicit '<Month> <Year>' out of the question text, e.g.
    'October 2025'. Trusted over whatever period the top-ranked retrieved
    document happens to be, since the user's question is the authoritative
    statement of which period they're asking about. Returns only the FIRST
    match -- see extract_all_periods_from_question() below for a question
    that names more than one period itself."""
    m = _PERIOD_IN_QUESTION_RE.search(question)
    if not m:
        return None
    return f"{m.group(1).capitalize()} {m.group(2)}"


def extract_all_periods_from_question(question: str) -> list[str]:
    """Every distinct '<Month> <Year>' the question names, in order of first
    appearance, deduplicated.

    WHY THIS EXISTS: a live question -- "Gamble-Wright Distributors ... had
    a Dropsize deviation ... in February 2025. By August 2025, did those
    issues persist...?" -- explicitly supplies its OWN baseline period
    (February 2025) as well as its target period (August 2025). Before this,
    check_premise() only ever looked at extract_period_from_question()'s
    single (first) match, treated it as the TARGET period, and always
    computed the baseline as `_previous_period(target)` ("previous calendar
    month") -- so a question naming Feb 2025 first and Aug 2025 second had
    its Feb 2025 mention silently treated as the target, with an
    unindexed "January 2025" then computed as its baseline, producing a
    false "insufficient_data" even though the question's own real baseline
    (Feb 2025) and target (Aug 2025) were BOTH actually indexed. See
    check_premise()'s use of this function for how baseline/target are
    picked apart once 2+ periods are found.

    Also recognizes the shared-year range phrasing 'January and February
    2026' (year stated once, after the second month) via
    _expand_shared_year_period_ranges() -- see that function's docstring."""
    seen: list[str] = []
    for m in _PERIOD_IN_QUESTION_RE.finditer(_expand_shared_year_period_ranges(question)):
        period = f"{m.group(1).capitalize()} {m.group(2)}"
        if period not in seen:
            seen.append(period)
    return seen


def _period_key(period: str) -> tuple[int, int] | None:
    """'October 2025' -> (2025, 10), for chronological ordering -- comparing
    period strings lexicographically is wrong ('November 2025' sorts before
    'October 2025' alphabetically despite coming after it in time)."""
    try:
        dt = datetime.strptime(period, "%B %Y")
    except ValueError:
        return None
    return (dt.year, dt.month)


def _previous_period(period: str) -> str | None:
    """'October 2025' -> 'September 2025' (handles the December->January
    year rollover). Returns None if `period` isn't a parseable 'Month
    YYYY' string."""
    try:
        dt = datetime.strptime(period, "%B %Y")
    except ValueError:
        return None
    if dt.month == 1:
        prev = dt.replace(year=dt.year - 1, month=12)
    else:
        prev = dt.replace(month=dt.month - 1)
    return prev.strftime("%B %Y")


@dataclass
class _SourceDocument:
    """One retrieved source text unit, parsed into its state/period header
    plus whatever state-level metric values we could regex out of its body.
    Internal to this module -- callers get MetricComparison objects, not
    this."""

    source_id: str
    state: str
    period: str
    text: str
    metrics: dict[str, float]


def _extract_state_metrics(text: str, state: str, period: str) -> dict[str, float]:
    """Regex out every STATE_METRIC_NAMES value from one document's body,
    matching the exact 'Productivity for Bihar in October 2025 was 85.5%.'
    template build_documents.py writes. The optional non-capturing group
    handles Dropsize's parenthetical definition aside ('Dropsize (units
    ordered per productive visit -- definition pending GPIL confirmation)
    for Bihar in October 2025 was 174.76.')."""
    metrics: dict[str, float] = {}
    for name in STATE_METRIC_NAMES:
        pattern = (
            rf"{re.escape(name)}(?:\s*\([^)]*\))?\s+for\s+{re.escape(state)}\s+in\s+"
            rf"{re.escape(period)}\s+was\s+([\d,]+\.?\d*)"
        )
        m = re.search(pattern, text)
        if m:
            metrics[name] = float(m.group(1).replace(",", ""))
    return metrics


def _parse_source_documents(sources_df: pd.DataFrame) -> list[_SourceDocument]:
    """Parse every retrieved source text unit's 'State: X / Period: Y'
    header and state-level metric sentences. Rows that don't start with
    that header (shouldn't happen for this project's documents, but we
    don't want a malformed row to crash the whole pipeline) are skipped."""
    docs: list[_SourceDocument] = []
    if sources_df is None or sources_df.empty:
        return docs
    for _, row in sources_df.iterrows():
        text = str(row["text"])
        m = _SOURCE_HEADER_RE.search(text)
        if not m:
            continue
        state = m.group("state").strip()
        period = m.group("period").strip()
        docs.append(
            _SourceDocument(
                source_id=str(row["id"]),
                state=state,
                period=period,
                text=text,
                metrics=_extract_state_metrics(text, state, period),
            )
        )
    return docs


def _find_doc(docs: list[_SourceDocument], state: str, period: str) -> _SourceDocument | None:
    for d in docs:
        if d.state == state and d.period == period:
            return d
    return None


def _compare_metric(
    baseline_doc: _SourceDocument, target_doc: _SourceDocument, metric: str
) -> MetricComparison | None:
    if metric not in baseline_doc.metrics or metric not in target_doc.metrics:
        return None
    a, b = baseline_doc.metrics[metric], target_doc.metrics[metric]
    direction = "up" if b > a else "down" if b < a else "flat"
    return MetricComparison(
        metric=metric,
        period_a=baseline_doc.period,
        value_a=a,
        period_b=target_doc.period,
        value_b=b,
        direction=direction,
    )


def check_premise(question: str, context_records: dict[str, pd.DataFrame]) -> PremiseCheckResult:
    """The main entry point for this stage.

    Looks at the question for a directional claim ("decline"/"improve"),
    figures out which state/period the retrieved evidence is actually
    about, finds the ACTUAL prior calendar month's document (not just
    whatever other period happens to be indexed), and compares the
    relevant metric(s) between the two periods.
    """
    superlative = extract_superlative_ranking_claim(question)
    if superlative is not None:
        # Checked BEFORE the direction-claim gate below and independent of
        # context_records: this project defines no performance/ranking
        # metric or composite score anywhere (see docs/), and the pilot
        # corpus has no full-population, comparable-period aggregate to
        # rank states/entities against even if one were defined -- every
        # document is one state x one month. A ranking claim like this
        # cannot be computed or verified from any evidence this index
        # could return, so it is rejected deterministically, before any
        # retrieval-dependent reasoning, rather than left for the LLM to
        # invent a ranking from whatever partial evidence happens to be
        # retrieved.
        return PremiseCheckResult(
            status="undefined_metric",
            method="deterministic",
            claimed_direction=None,
            target_state=None,
            target_period=None,
            baseline_period=None,
            explanation=(
                f"The question asks which entity was '{superlative}', but this project defines no "
                "performance/ranking metric or composite score, and the indexed corpus has no "
                "state-by-state comparable aggregate to rank against -- each retrieved document "
                "covers one state for a single month only, not a full-population summary for any "
                "period. A ranking claim cannot be computed or verified from this evidence."
            ),
        )

    claimed_direction = extract_direction_claim(question)
    if claimed_direction == "neutral":
        # Plain descriptive question ("what was Bihar's performance") --
        # there's no directional assumption to check, so don't block
        # generation on anything.
        return PremiseCheckResult(
            status="no_claim",
            method="deterministic",
            claimed_direction="neutral",
            target_state=None,
            target_period=None,
            baseline_period=None,
            explanation="The question does not assert a direction (decline/improve), so there is no premise to verify.",
        )

    sources = context_records.get("sources")
    docs = _parse_source_documents(sources)
    if not docs:
        return PremiseCheckResult(
            status="insufficient_data",
            method="deterministic",
            claimed_direction=claimed_direction,
            target_state=None,
            target_period=None,
            baseline_period=None,
            explanation="No parseable state/period source documents were retrieved for this question.",
        )

    # The question's own explicitly-named state wins if present -- trusted
    # over retrieval the same way the question's own period already is
    # below -- rather than silently defaulting to whatever state the
    # top-ranked retrieved document happens to be about.
    question_state = extract_state_from_question(question)
    state_docs = [d for d in docs if d.state == question_state] if question_state else docs

    if question_state and not state_docs:
        # The question names a specific state, but nothing retrieved is
        # actually about it -- there is no evidence to validate the claim
        # against, and evidence for a DIFFERENT state must never be
        # substituted in silently.
        return PremiseCheckResult(
            status="insufficient_data",
            method="deterministic",
            claimed_direction=claimed_direction,
            target_state=question_state,
            target_period=extract_period_from_question(question),
            baseline_period=None,
            explanation=(
                f"The question asks about {question_state}, but no retrieved evidence is about "
                f"{question_state}, so a {claimed_direction} claim for it cannot be evaluated."
            ),
        )

    # Multi-period questions (2+ distinct "<Month> <Year>" mentions):
    # chronologically EARLIEST named period is treated as an explicit,
    # user-supplied baseline (never overwritten by `_previous_period()`'s
    # "previous calendar month" assumption), chronologically LATEST as the
    # target -- sorted by actual calendar order, not text order, so this
    # also covers a target-named-first phrasing ("did X, which declined in
    # October 2025, recover by September 2025?" -- deliberately absurd
    # example, but the point is calendar order decides, not sentence
    # order). A single-period (or no-period) question is completely
    # unaffected -- explicit_baseline_period stays None and every line
    # below behaves exactly as it did before this fix.
    all_periods = extract_all_periods_from_question(question)
    explicit_baseline_period: str | None = None
    explicit_target_period: str | None = None
    if len(all_periods) >= 2:
        keyed = sorted((p for p in all_periods if _period_key(p) is not None), key=_period_key)
        if len(keyed) >= 2:
            explicit_baseline_period, explicit_target_period = keyed[0], keyed[-1]

    # The question's own stated period wins if present; otherwise trust the
    # top-ranked retrieved source for the target state (GraphRAG's context
    # builder returns sources already ranked by relevance to the query, so
    # state_docs[0] is its best match -- for that state specifically when
    # the question names one, or overall when it doesn't, identical to the
    # original docs[0] behavior).
    target_period = explicit_target_period or extract_period_from_question(question) or state_docs[0].period
    target_state = question_state or state_docs[0].state
    target_doc = _find_doc(docs, target_state, target_period) or state_docs[0]
    # Re-derive target_period from whichever doc we actually landed on, in
    # case the question named a period that isn't the top match. When the
    # question explicitly named a state, target_state stays authoritative --
    # never silently substituted for a different retrieved state; otherwise
    # (unchanged from before) it's re-derived from whichever doc was found.
    target_state = target_state if question_state else target_doc.state
    target_period = target_doc.period

    if explicit_baseline_period is not None:
        baseline_period = explicit_baseline_period
    else:
        baseline_period = _previous_period(target_period)
    baseline_doc = _find_doc(docs, target_state, baseline_period) if baseline_period else None

    # Whatever OTHER periods for this state ARE indexed, even though they
    # aren't the real baseline -- surfaced as supplementary context in the
    # hedge message rather than silently dropped.
    other_docs = [d for d in docs if d.state == target_state and d.period != target_period]
    metric_names = [extract_metric_claim(question) or ""] if extract_metric_claim(question) else HEADLINE_METRICS
    metric_names = [m for m in metric_names if m]

    if baseline_doc is None:
        supplementary = []
        target_key = _period_key(target_doc.period)
        for d in other_docs:
            d_key = _period_key(d.period)
            # Order the comparison chronologically (earlier period as
            # period_a) whenever both periods parse; otherwise fall back to
            # target-then-other so the comparison still gets reported.
            earlier_first = d_key is not None and target_key is not None and d_key < target_key
            for m in metric_names:
                comp = _compare_metric(d, target_doc, m) if earlier_first else _compare_metric(target_doc, d, m)
                if comp:
                    supplementary.append(comp)
        explanation = (
            f"No {baseline_period} document is indexed for {target_state}, so a "
            f"{claimed_direction} claim for {target_period} cannot be evaluated "
            "against its actual prior period."
        )
        return PremiseCheckResult(
            status="insufficient_data",
            method="deterministic",
            claimed_direction=claimed_direction,
            target_state=target_state,
            target_period=target_period,
            baseline_period=baseline_period,
            supplementary_comparisons=supplementary,
            explanation=explanation,
        )

    comparisons = [c for c in (_compare_metric(baseline_doc, target_doc, m) for m in metric_names) if c]
    if not comparisons:
        return PremiseCheckResult(
            status="insufficient_data",
            method="deterministic",
            claimed_direction=claimed_direction,
            target_state=target_state,
            target_period=target_period,
            baseline_period=baseline_period,
            explanation=f"Neither document reports a value for {', '.join(metric_names)}.",
        )

    directions = {c.direction for c in comparisons}
    if claimed_direction == "decline":
        status = "supported" if directions == {"down"} else "contradicted" if directions == {"up"} else "unsupported"
    else:  # "improve"
        status = "supported" if directions == {"up"} else "contradicted" if directions == {"down"} else "unsupported"

    metric_summary = "; ".join(
        f"{c.metric} {c.value_a}->{c.value_b} ({c.direction})" for c in comparisons
    )
    explanation = (
        f"Comparing {baseline_period} to {target_period} for {target_state}: {metric_summary}. "
        f"Question claimed '{claimed_direction}'; evidence is '{status}'."
    )

    return PremiseCheckResult(
        status=status,
        method="deterministic",
        claimed_direction=claimed_direction,
        target_state=target_state,
        target_period=target_period,
        baseline_period=baseline_period,
        metric_comparisons=comparisons,
        explanation=explanation,
    )
