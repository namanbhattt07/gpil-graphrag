"""
Phase 7, Stage 1 -- evidence-grounded answer generation, plus the bounded
single retry used when the first draft fails the grounding check.

WHAT THIS FILE IS FOR (plain language):
This is the ONLY stage in the whole Phase 7 pipeline that calls the
completion model. generate_answer() only ever runs after premise_check.py
has already decided the question's directional claim (if any) is actually
supported by the retrieved evidence -- pipeline.py is responsible for that
gate, not this file. regenerate_answer() only ever runs after
grounding_check.py has flagged something wrong with generate_answer()'s
draft -- also pipeline.py's call to make, not this file's.

WHY IT ASKS FOR A STRUCTURED CLAIMS BLOCK:
The Phase 7 architecture audit's cost analysis found the cheapest way to
get a grounding check is to have the SAME completion call that writes the
prose answer also report, in a small trailing JSON block, the concrete
metric/period/value/direction claims that prose makes. grounding_check.py
can then validate those claims in plain Python against context_records --
no second LLM call needed in the common case. Both generate_answer() and
regenerate_answer() ask for this same block, so the SAME deterministic
grounding_check() can validate either one's output identically.

WHY THE RETRY IS BOUNDED TO EXACTLY ONE ATTEMPT:
A live test found a real, well-cited, factually accurate answer got
withheld entirely over one unsupported adjective ("concerning") in its
closing sentence -- correct behavior (nothing unsupported reached the
user) but blunt (a 95%-good answer thrown away instead of fixed). The
targeted fix is a single revise-and-recheck pass, not an open-ended retry
loop: pipeline.py calls regenerate_answer() at most once per query and
never loops back into it, keeping worst-case cost at exactly 2 completion
calls.

Reuses GraphRAG's own LocalSearch engine object (built by
context.build_query_context()) for the model client, system prompt, and
call parameters -- it does not create its own OpenAI client or reimplement
GraphRAG's model configuration.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from graphrag_llm.utils import CompletionMessagesBuilder

from src.inference.fact_structuring import build_atomic_facts_block, build_deterministic_ranking_block
from src.inference.knowledge_layer import build_glossary_block, build_glossary_reminder_block
from src.inference.schemas import AnswerClaim, AnswerResult, GroundingCheckResult, GroundingIssue

if TYPE_CHECKING:
    from graphrag.query.context_builder.builders import ContextBuilderResult
    from graphrag.query.structured_search.local_search.search import LocalSearch

# Appended to the index's own local_search_system_prompt.txt (after its
# {context_data}/{response_type} placeholders are already filled in) so the
# model reports its own claims in a machine-checkable shape, in the same
# call that writes the prose. This is the CLAIM-LEVEL grounding design's
# input: grounding_check.py validates each claim against ONLY the specific
# context records its own `citations` point to, not the whole context blob
# -- so citations here matter as much as the values do.
_CLAIM_EXTRACTION_SUFFIX = """


---Structured Claims---

After the prose answer, add a fenced code block labeled json containing a
list of EVERY claim your answer makes that goes beyond a single isolated
fact -- numeric trends, comparisons between periods, distributor/entity
deviations, qualitative characterizations, and any causal or contributing-
factor language. One object per claim, in this shape (omit fields that
don't apply -- do not force-fill unrelated fields):

```json
[
  {
    "claim_id": 0,
    "claim_text": "the exact sentence or clause this claim corresponds to",
    "claim_type": "trend",
    "entity": "Bihar",
    "metric": "Service Level",
    "period": "November 2025",
    "comparison_period": "October 2025",
    "value": 92.9,
    "comparison_value": 94.3,
    "direction": "declined",
    "delta": 1.4,
    "citations": ["Entities (147)", "Sources (3)"]
  },
  {
    "claim_id": 1,
    "claim_text": "Sikkim's Ferrero Out-of-Stock Rate (6.6%) was higher than Maharashtra's (6.5%)",
    "claim_type": "comparison",
    "entity": "Sikkim",
    "comparison_entity": "Maharashtra",
    "metric": "Ferrero Out-of-Stock Rate",
    "period": "April 2025",
    "comparison_period": "August 2024",
    "value": 6.6,
    "comparison_value": 6.5,
    "direction": "higher",
    "delta": 0.1,
    "citations": ["Sources (5)", "Sources (12)"]
  }
]
```

claim_type must be one of: factual_numeric, trend, comparison, deviation,
causal, qualitative.
- factual_numeric: a single value stated for an entity/metric/period.
- trend: a metric's value changed between two periods FOR THE SAME ENTITY
  (set value, comparison_value, direction, delta -- do NOT set
  comparison_entity).
- comparison: the SAME metric compared between TWO DIFFERENT entities
  (states, distributors, categories, ...) -- set comparison_entity to name
  the second entity (see "comparison_entity" below); comparison_period is
  also needed if the two entities' figures come from different periods.
- deviation: an entity's value deviates from a state/group average.
- causal: your claim_text says or implies one thing caused, contributed
  to, led to, resulted in, or was due to another (even hedged language
  like "may have contributed to" counts -- report it as causal so it can
  be checked, don't leave it out because it's hedged).
- qualitative: your claim_text characterizes a fact with an evaluative
  word (e.g. "alarming", "significant", "concerning", "critical").

"entity" (and "comparison_entity", when set) must each name exactly ONE
thing. Never combine multiple named entities into a single field (e.g. do
NOT write entity="Bennett-Webster, Casey, Nguyen and Ramirez,
Weaver-Sherman Distributors" as one entity, and do NOT write
entity="Sikkim, Maharashtra" to represent a comparison between the two --
use "comparison_entity" for that, see below) -- if several named entities
are each individually relevant but NOT being directly compared to one
another, report a SEPARATE claim per entity so each can cite and be
checked against its own evidence. Only use a plural/generic entity phrase
(e.g. "several distributors") when the evidence genuinely only supports an
aggregate statement, not as a way to avoid writing multiple claims.

Before finalizing each claim's "value"/"comparison_value", re-confirm it
is the exact figure the evidence states for THIS claim's own entity,
metric, and period (see Value Attribution above) -- not a value belonging
to a neighboring entity, category, metric, or period in the same cited
evidence.

For a comparison BETWEEN TWO DIFFERENT entities/states/distributors/
categories (not the same entity across two periods), use
"comparison_entity" to name the second entity -- "entity"/"value"/"period"
describe the first side, "comparison_entity"/"comparison_value"/
"comparison_period" describe the second -- and citations must support BOTH
sides of the comparison. For a same-entity period comparison (a trend --
one entity, two periods), leave "comparison_entity" unset; the existing
entity/comparison_period semantics are unchanged. But never assume that a
citation supporting the primary entity's value also supports
comparison_value when comparison_value belongs to a DIFFERENT entity --
that other entity's supporting evidence must be cited too. For example, a
claim comparing Maharashtra's Ferrero Out-of-Stock Rate (6.5%) to Sikkim's
Ferrero Out-of-Stock Rate (6.6%) -- reported for different periods
(Maharashtra: August 2024, Sikkim: April 2025) -- should set entity=
"Sikkim", value=6.6, period="April 2025", comparison_entity="Maharashtra",
comparison_value=6.5, comparison_period="August 2024", and must cite
evidence for Maharashtra's 6.5% AND evidence for Sikkim's 6.6% -- citing
only one side leaves the other half of the comparison unsupported. If a
single claim cannot cleanly cite both sides, report the two values as
separate factual_numeric claims (one per entity) instead of forcing an
under-cited comparison claim -- the prose can still state the comparison
directly once both underlying facts are individually supported.

citations must list every [Data: ...] reference that specific claim relies
on -- this is what the checker validates the claim against, not the whole
answer's evidence. For each claim, choose the record that most directly
and completely supports THAT claim (see Citation Selection above) --
not merely a record that happens to mention the right entities. If a
claim has no real supporting citation, still report it (with an empty
citations list) rather than omitting it -- an unlisted claim can't be
checked at all.

If your answer makes no claims beyond simple isolated facts, return an
empty list: []
"""

# Phase 7 hardening (Part 1): appended alongside _CLAIM_EXTRACTION_SUFFIX to
# fix a live failure mode where the model compressed specific, useful
# evidence ("Landry Ltd Distributors", "Zhang, Brooks and Miles
# Distributors", their individual OOS values) into a vague phrase ("various
# distributors") that then couldn't be checked against any one citation,
# AND blurred historical observations together with forward-looking
# recommendations ("...and improve service levels moving forward" reading,
# to the deterministic direction scanner, as a claim that the KPI already
# went up). Neither fix is a word-specific patch: this asks for a general
# writing discipline (name what evidence names; mark recommendations as
# recommendations), which is what makes it a durable fix rather than a
# one-off for "various distributors" or "improve" specifically -- the
# actual enforcement of both still happens deterministically afterward in
# grounding_check.py, this text only makes a good first draft more likely.
_SPECIFICITY_AND_FRAMING_GUIDANCE = """


---Evidence Specificity And Framing---

When the retrieved evidence names specific entities (distributors,
categories, franchises, zones) together with their own metrics or
deviations that are directly relevant to the question asked, name them
specifically instead of collapsing them into a vague group phrase such as
"various distributors" or "several factors" -- state which entity, which
metric, and which value(s), exactly as the evidence reports them. Only
include a detail if it is actually relevant to explaining the question
asked; do not list every tangential fact the evidence contains just because
it is available.

Report each named entity's observed value/issue exactly as the evidence
states it -- never invent or infer an entity-level fact (a value, a
direction, a cause) that isn't explicitly present in what you're citing,
even if it seems like a reasonable estimate. A generic group phrase
("several distributors showed OOS deviations") is acceptable ONLY when
either the evidence itself speaks in aggregate (no individual figures to
report) or the individually-relevant entities are too numerous for naming
each one to add diagnostic value -- it is not a substitute for the effort
of naming the two or three entities that actually explain the question
asked.

Make clear, through how each sentence is phrased, which of these three
kinds of statement it is:
- an OBSERVED FACT or HISTORICAL COMPARISON: something the data tables
  state already happened, in the past tense, about a specific period
  (e.g. "Service Level fell from 94.3% to 92.9% between October and
  November 2025.").
- an EVIDENCE-BASED INTERPRETATION: a plausible, hedged reading of what
  the observed facts may indicate, still grounded in cited evidence, never
  asserting causation the evidence doesn't state explicitly.
- a FUTURE RECOMMENDATION: an explicitly forward-looking suggestion,
  marked with clear future/prescriptive language ("going forward",
  "moving forward", "should", "consider monitoring") -- never phrase a
  recommendation as if it were a historical observation ("Service Level
  improved") and never phrase a historical observation as if it were
  merely a suggestion.

A future recommendation must stay tied to the entities/metrics/periods the
evidence actually supports (e.g. "monitor the OOS levels of the
distributors identified above") -- do not recommend actions, strategies, or
a scope of change that isn't grounded in what the evidence shows.

Where it fits naturally, let the answer's shape follow: what the data shows
happened; what evidence-based factors may be associated with it; and what
to monitor or consider going forward -- but do not force rigid section
headers if that breaks the natural flow of the requested response format.
"""

# Phase 7d hardening: appended alongside _SPECIFICITY_AND_FRAMING_GUIDANCE
# to fix a live failure mode where a claim was factually correct and the
# fact WAS stated explicitly in a retrieved Sources document, but the
# model cited a much terser Relationships record instead (just an entity-
# to-entity edge with a short description) -- grounding_check.py correctly
# rejected the claim because the cited Relationships text didn't literally
# contain the distributor's full name, the word "deviation", or the
# value, even though a Sources record in the SAME retrieved context did.
# This is a citation-CHOICE problem, not a retrieval or grounding-logic
# problem -- the fix is telling the model which record to reach for, not
# changing what grounding_check.py accepts (grounding_check.py is
# untouched by this pass).
_CITATION_SELECTION_GUIDANCE = """


---Citation Selection---

Multiple data tables may all technically relate to the same claim -- when
they do, cite the one that states the claimed fact most directly and
completely, not just any table that happens to connect the right entities.

1. Prefer Sources when the source document explicitly contains the
   entity, metric, period, value, deviation, or other factual detail you
   are claiming -- Sources documents are full statements of fact and give
   the strongest possible evidence.
2. Reports are also preferred when they state the claimed fact directly
   and in sufficient detail.
3. Entities are acceptable when the entity's own description directly
   contains the claimed fact.
4. Relationships should be cited primarily when the relationship itself
   IS the fact being asserted (e.g. that two things are connected or
   associated), not merely because it happens to connect the entities
   your claim is about. If a Sources or Reports record in the retrieved
   context already states your claimed fact directly, cite that instead
   of a Relationships record that only connects the same entities without
   restating the fact.

Example: if the retrieved context contains a Sources record stating
"Landry Ltd Distributors showed an Out-of-Stock deviation of 28.6% in
November 2025..." and also a Relationships record merely connecting
"Landry Ltd Distributors" to "Bihar", then the claim "Landry Ltd
Distributors showed an Out-of-Stock deviation of 28.6%" should cite the
Sources record, not the Relationships record. A claim like "Landry Ltd
Distributors is associated with Bihar" -- where the connection itself is
what's being claimed -- may legitimately cite the Relationships record.

This is not a rule to always cite Sources -- some claims (e.g. a stated
relationship, or a fact only present in a community report) are genuinely
best supported by a Reports, Entities, or Relationships record. Choose
whichever cited record actually, literally supports your specific claim;
never cite a record that doesn't directly support what you're claiming,
and never invent a citation id that doesn't exist in the data tables.
"""

# Phase 9 hardening: appended alongside _CITATION_SELECTION_GUIDANCE to fix
# a live Q5 failure where the model correctly bound each fact's own
# period, value, and direction (Gujarat Productivity 74.0% in April 2026
# -> 74.3% in June 2026 -- entirely correct arithmetic and framing) but
# cited the fact by its ATOMIC FACTS SECTION NAME AND NUMBER ("Atomic
# State KPI Facts (1)") instead of that fact's own embedded Source:
# identifier ("Sources (21)"). grounding_check.py's citation resolver only
# recognizes the five real GraphRAG dataset names (Sources/Entities/
# Reports/Relationships/Claims); an "Atomic ... Facts (N)"-shaped citation
# silently resolves to no text at all, which correctly (but confusingly,
# from outside) fails the claim as if the fact were unsupported -- even
# though the fact itself, and the genuine Sources citation the model also
# included, were both entirely correct. This is a citation-FORMAT problem,
# not a fact-finding or grounding-strictness problem, so the fix belongs
# here (telling the model which identifier to actually write), not in
# grounding_check.py -- nothing about what grounding accepts changes.
_ATOMIC_FACTS_CITATION_GUIDANCE = """


---Citation Rules For Atomic Facts Sections---

When you use an Atomic Facts section (any section titled "Atomic State KPI
Facts", "Atomic Distributor Facts", "Atomic Category KPI Facts", or
"Atomic SKU Facts") to support a claim, do NOT cite the Atomic Facts
section itself -- these sections are contextual fact helpers, not valid
citation dataset names. "Atomic State KPI Facts (1)" is never a resolvable
citation.

The SAME rule applies to a "Deterministic SKU Ranking" section, when
present: it is an authoritative, pre-computed answer to a ranking
question, not a citable dataset name. Do NOT cite "Deterministic SKU
Ranking (1)" or similar. Each ranked line already ends with its own
"Source: Sources (id)" field (no brackets, exactly like every other
Atomic Facts section's "Source:" field) -- use that exact identifier as
your citation. Do NOT copy that "Source: Sources (id)" text verbatim into
your prose answer -- your prose must use the real citation tag format
"[Data: Sources (id)]" instead, exactly as instructed above.

Each Atomic Fact contains its own "Source:" field. Always use the exact
citation identifier from that Source: field when citing that fact --
copy it from THAT SAME fact block, never from a different, nearby fact.
A SKU-level fact's own Source: field always has the shape "sku-<State>-
<Month_Year>" (e.g. "Sources (sku-Gujarat-April_2026)") -- a plain
number in parentheses (e.g. "Sources (22)") is NEVER a SKU fact's own
citation, even if a numbered source happens to also mention the same
state or period; if you are citing a claim about a specific named SKU,
double-check that your citation string literally starts with "sku-".

For example, if the context contains:

FACT 1
State: Gujarat
Metric: Productivity
Period: June 2026
Value: 74.3%
Source: Sources (21)

then the correct citation is "Sources (21)" -- and NOT "Atomic State KPI
Facts (1)". The Source: field inside an Atomic Fact is the authoritative
citation to use for that fact.

Multi-period / multi-source citation rule: when a claim combines facts
from different periods or source records, cite the source corresponding
to each fact/value. For example:

    Productivity was 74.0% in April 2026 and 74.3% in June 2026.

If:

    April fact -> Source: Sources (22)
    June fact  -> Source: Sources (21)

then the claim must cite both: ["Sources (22)", "Sources (21)"]. Do not
use the Atomic Facts fact number as a citation.

Before returning a structured claim, verify:
1. Every cited identifier is a valid resolvable dataset/source identifier.
2. Every numeric value in the claim is supported by its cited source.
3. Every period in the claim is supported by its cited source.
4. If two periods are compared, cite evidence for both periods.
5. Never invent, transform, or abbreviate a citation identifier.
6. Prefer the exact Source: identifier embedded inside the Atomic Fact.
"""

# GPIL Knowledge Layer (2026-08-23): fixes a live failure where "What is
# GPI?" was answered "General Product Inventory" and "What is IPM?" was
# answered "Integrated Pest Management" -- both plausible-sounding, both
# wrong for this project, both drawn from the model's general world
# knowledge because the indexed corpus never defines these codes and
# GraphRAG's own system prompt explicitly invites "relevant general
# knowledge" as a fallback. See knowledge_layer.py for the glossary and
# detection logic, and pipeline.py's _augment_context_with_glossary() for
# how a matched term's definition reaches context_records["sources"]
# (and therefore this prompt) as real, citable evidence in the first place.
_GLOSSARY_CITATION_GUIDANCE = """


---GPIL Knowledge Layer Guidance---

When the context includes a "-----GPIL Knowledge Layer-----" section, it
contains this project's OWN authoritative business-specific meaning for one
or more terms named in the question. For any term that section defines, you
MUST use that GPIL-specific meaning -- and that section OVERRIDES BOTH (a) a
generic/world-knowledge meaning of the same word or acronym, AND (b) any
OTHER description of that same term appearing elsewhere in these data
tables, including an Entity or Community Report description auto-generated
from the corpus at indexing time. An auto-generated Entity/Report
description for a Knowledge Layer term can be vague or misleading (it was
written without ever seeing a real definition either) -- do not treat it as
confirming, or as license to fall back to, a generic meaning; the Knowledge
Layer entry wins whenever the two seem to differ. For example, if the
Knowledge Layer defines "GPI", never answer with a generic expansion like
"General Product Inventory" even if a retrieved Entity description for
"GPI" reads as generic or ambiguous; if it defines "IPM", never answer with
"Integrated Pest Management" even if a retrieved Entity/Report description
uses agriculture-adjacent language. This override applies ONLY to the exact
terms the Knowledge Layer section defines -- for any other term, retrieval
and your own knowledge work exactly as they otherwise would. If the
question asks what a term "is" or "stands for" and no Knowledge Layer entry
is present for it, say you don't have a confirmed definition for it rather
than guessing one.

Do NOT cite the "-----GPIL Knowledge Layer-----" section name itself --
like the Atomic Facts and Deterministic SKU Ranking sections, it is not a
valid citation dataset name. Each entry ends with its own "Source: Sources
(id)" field (e.g. "Sources (glossary-gpi)") -- when your prose states a
term's meaning, cite that exact identifier in a real "[Data: Sources (id)]"
tag, exactly like every other Atomic Facts section's Source: field.
"""

# Phase 8 hardening: appended alongside _SPECIFICITY_AND_FRAMING_GUIDANCE and
# _CITATION_SELECTION_GUIDANCE to fix a live failure mode where a real,
# evidence-present value was copied onto the WRONG neighboring entity --
# e.g. Candy's ACV (27.7%) restated as Ferrero's ACV, because the two
# categories' ACV figures sit in adjacent, near-identically-shaped
# sentences in the same source document. This is a value-to-label BINDING
# error, not a specificity or citation-choice problem, so it gets its own
# instruction rather than folding into either existing block. Deliberately
# generic (not a Ferrero/Candy special case) since the same adjacent-
# similar-sentence shape recurs throughout the corpus: GPI/IPM/Ferrero/
# Candy category blocks, October-vs-November state summaries, and
# per-distributor deviation bullets all repeat this pattern.
_VALUE_ATTRIBUTION_GUIDANCE = """


---Value Attribution---

Evidence documents often report several similar metrics for different
entities, categories, periods, or comparison groups in immediately
adjacent sentences. Before stating any numeric value, confirm it is the
exact value the evidence attaches to the SAME entity/category, the SAME
metric, and the SAME period being named for that value -- not a value
copied from a neighboring sentence about a different entity, category,
metric, or period merely because it appears nearby.

This does NOT prohibit legitimate comparisons. You should still freely
report comparisons such as:
- Ferrero ACV 30.0% versus Candy ACV 27.7%
- Service Level 94.3% in October versus 92.9% in November
- a distributor's value versus the state average

The requirement is that each individual value in a comparison must be
independently attributed to its own correct entity/category, metric, and
period -- never assumed correct just because a nearby sentence uses the
same metric name or a similar-looking number.

This applies just as much to adjacent DISTRIBUTOR-level deviation
statements as to adjacent category/metric summaries. Evidence often lists
several distributors' deviations back-to-back in the same shape (e.g.
"Distributor A showed a deviation on Metric/Category X: N1% vs. the state
average of M1%... Distributor B showed a deviation on Metric/Category Y:
N2% vs. the state average of M2%..."). When you see this pattern,
independently verify each distributor's own category, metric, value,
period, and comparison/state-average value from ITS OWN statement -- never
transfer a category, metric, or value from one distributor's statement to
a neighboring distributor's statement merely because the two statements
share the same sentence structure.
"""

# Phase 8d hardening: appended alongside the other guidance blocks to fix a
# live failure mode where the model named THREE OR FOUR distributors with a
# supposed deviation, all sharing the exact same value, when the evidence
# only actually stated a deviation for ONE or TWO of them -- the rest were
# never mentioned with any figure at all, just listed alongside the real
# ones in the state's zone roster (e.g. "Zone 2 distributors: Landry Ltd
# Distributors, Porter Group Distributors, ... Garcia-Perry Distributors,
# Edwards and Sons Distributors, Weaver-Sherman Distributors."). This is a
# different failure mode from _VALUE_ATTRIBUTION_GUIDANCE above (copying a
# REAL value onto the WRONG neighboring entity): here the model invents a
# deviation for an entity that has NO deviation statement in evidence AT
# ALL, apparently by pattern-completing "other distributors in this same
# list probably have it too." grounding_check.py's P0 co-occurrence check
# already catches and rejects this deterministically, but rejection alone
# just throws the whole answer away -- this guidance is aimed at the
# generation step itself, so the fabricated claim is never drafted in the
# first place.
_ENTITY_EXISTENCE_GUIDANCE = """


---Entity Existence---

A distributor or entity's appearance in a roster or listing (e.g. "Zone 2
distributors: A Distributors, B Distributors, C Distributors...") only
establishes that the entity EXISTS and operates in that state/zone -- it is
NOT evidence of any deviation, out-of-stock rate, dropsize figure, or other
metric value for that entity. Only report a specific value or deviation for
a named entity when a SEPARATE sentence explicitly states that entity's own
figure (e.g. "Distributor B showed a significant deviation on Out-of-Stock
Rate ...: N% vs. the state average of M%, a gap of G percentage points").

Never assume that because one or two distributors in a list have a stated
deviation, other distributors nearby in the same roster -- even ones with
similar-sounding names, listed in the same zone, or appearing right next to
a named distributor -- share that same or a similar value. Count how many
distinct distributor-deviation sentences the evidence actually contains for
the metric/category in question, and report exactly that many named
distributors -- no more. If a distributor is listed in the state/zone
roster but never given its own deviation sentence, do not invent a value
for it: either state plainly that no deviation was reported for it, or omit
it entirely from the answer.
"""

# Phase 8 hardening: appended alongside the other guidance blocks to fix a
# live failure mode where a later period's evidence explicitly stated no
# distributor had a deviation on a metric, but the model treated this as
# "no information available" instead of using it to answer whether an
# EARLIER period's named deviation persisted. The underlying GraphRAG
# entity/relationship graph has no representation for negative/absence
# facts at all (only positive DEVIATES_FROM-style relationships are
# extracted) -- this is NOT fixed by changing extraction here; the raw
# source text still carries the negative statement, so this is a
# generation-side reasoning gap, not a retrieval or extraction gap.
_RESOLUTION_REASONING_GUIDANCE = """


---Resolution And Absence Reasoning---

Explicit negative statements in the retrieved evidence are still evidence,
not missing information. A later period's evidence may explicitly state
that no distributor had a deviation on a given metric that month (e.g. "No
distributor deviation this month for: Service Level"). This kind of
statement directly answers whether an EARLIER period's named
distributor-level deviation on that same metric persisted into the later
period.

If an earlier period's evidence shows a specific distributor deviating on a
metric, and a later period's evidence explicitly states there was no
distributor deviation for that metric, conclude that the earlier deviation
did NOT persist -- it was resolved by the later period. Do not treat this
as "no information available" or hedge the answer; the explicit negative
statement is a direct, citable answer.

This rule applies ONLY when the evidence explicitly states the absence
(e.g. "No distributor deviation this month for: <metric>"). Do not infer
that a deviation was resolved merely because you did not retrieve a
positive deviation relationship for it -- absence of a positive statement
is not the same as an explicit statement of absence, and only the latter
supports this conclusion.
"""

_JSON_BLOCK_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.DOTALL)


def _coerce_numeric_field(raw: object) -> tuple[float | None, bool]:
    """Coerce one JSON-decoded claim field (value/comparison_value/delta)
    into a float, since the model's JSON claims block isn't guaranteed to
    keep a number typed as a number -- a live failure showed a claim with
    "value": "92.9" (quoted) crash grounding_check.py's arithmetic with a
    bare TypeError. Returns (coerced_value_or_None, was_malformed):
    was_malformed=True means the field was PRESENT but not usable as a
    number (a non-numeric string, a bool, a list/dict, ...) -- distinct
    from raw being None, which just means the model didn't report this
    field at all (the normal, expected case for an optional field).

    Only plain float() parsing is used for strings -- no stripping of
    "%"/","/etc. and no guessing at intent, so a numeric string like
    "92.9" is accepted (identical downstream behavior to a native JSON
    number) while anything ambiguous ("92.9%", "~93", "N/A", "") is
    correctly rejected as malformed rather than silently reinterpreted.
    """
    if raw is None:
        return None, False
    if isinstance(raw, bool):  # bool is an int subclass -- never a real metric value
        return None, True
    if isinstance(raw, (int, float)):
        return float(raw), False
    if isinstance(raw, str):
        try:
            return float(raw), False
        except ValueError:
            return None, True
    return None, True  # list/dict/other -- never a real metric value


def _coerce_string_field(raw: object) -> str | None:
    """Coerce one JSON-decoded claim field (entity/comparison_entity/
    metric/period/comparison_period/direction/claim_text/claim_type) into
    a plain string or None. Live-caught crash (2026-08-23 stabilization
    pass): a two-SKU comparison question had the model emit
    "entity": ["Marlboro Pack 1", "GPI_Franchise_1 Pack 1"] (a JSON array
    of both names, instead of using comparison_entity for the second one
    as instructed) -- every downstream string operation in
    grounding_check.py (starting with _mentions()'s `term.strip()`)
    assumed `claim.entity` is always a plain string and crashed with a
    bare AttributeError, which propagated all the way up through
    run_diagnostic_query() as an unhandled DiagnosticQueryError instead
    of a normal (if imperfect) PipelineResult. Mirrors
    _coerce_numeric_field()'s "fail safe, never crash downstream" pattern:
    a non-string value (list/dict/int/bool/...) is treated as ABSENT
    (None), not as a value to guess at or stringify -- an entity field
    the model couldn't even express as one string is not a value grounding
    can meaningfully check anyway, and None correctly makes every
    downstream check that depends on this field a no-op, exactly like a
    genuinely-omitted field. An empty string is likewise treated as None
    (an empty entity/metric/period name is never meaningfully "present")."""
    if isinstance(raw, str) and raw.strip():
        return raw
    return None


def _coerce_citations_field(raw: object) -> list[str]:
    """Coerce one JSON-decoded claim's "citations" field into a list of
    plain strings. `list(raw)` alone (the prior behavior) silently
    mis-handles a raw string value -- Python iterates a string
    character-by-character, so "citations": "Sources (21)" would become
    ['S', 'o', 'u', 'r', 'c', 'e', 's', ...], each treated as its own
    (unresolvable) citation, rather than the one real citation the model
    meant. Same "fail safe to absent, never crash or silently corrupt"
    philosophy as _coerce_string_field(): [] for anything that isn't
    genuinely a list already; non-string entries within a real list are
    dropped rather than crashing a later .strip()/.lower() call on them."""
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item.strip()]


def _parse_claims(full_response: str) -> list[AnswerClaim]:
    """Pull the trailing ```json [...]``` block back out of the model's
    response and turn it into AnswerClaim objects. Never raises -- a
    malformed or missing block just means zero structured claims (the
    grounding checker still runs its qualitative-language scan either
    way; it just can't cross-check numeric claims it wasn't told about).

    value/comparison_value/delta are run through _coerce_numeric_field()
    rather than passed through raw -- see that function's docstring. A
    malformed value/comparison_value becomes float("nan"), NOT None:
    nan keeps the field "present" so grounding_check.py's existing
    unsupported_numeric/direction/comparison checks correctly flag it as
    unsupported (fail closed), instead of None silently skipping those
    checks the way a genuinely-absent field is meant to. delta has no
    such gap -- it's a supplementary field grounding_check.py only
    checks when value/comparison_value are both present and valid, so a
    malformed delta safely falls back to None (the same as "not
    reported"), rather than needing the nan treatment."""
    match = _JSON_BLOCK_RE.search(full_response)
    if not match:
        return []
    try:
        raw_claims = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    claims: list[AnswerClaim] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        value, value_malformed = _coerce_numeric_field(item.get("value"))
        if value_malformed:
            value = float("nan")
        comparison_value, comparison_value_malformed = _coerce_numeric_field(item.get("comparison_value"))
        if comparison_value_malformed:
            comparison_value = float("nan")
        delta, _delta_malformed = _coerce_numeric_field(item.get("delta"))
        claims.append(
            AnswerClaim(
                claim_id=item.get("claim_id"),
                claim_text=_coerce_string_field(item.get("claim_text")),
                claim_type=_coerce_string_field(item.get("claim_type")),
                entity=_coerce_string_field(item.get("entity")),
                metric=_coerce_string_field(item.get("metric")),
                period=_coerce_string_field(item.get("period")),
                comparison_period=_coerce_string_field(item.get("comparison_period")),
                comparison_entity=_coerce_string_field(item.get("comparison_entity")),
                value=value,
                comparison_value=comparison_value,
                direction=_coerce_string_field(item.get("direction")),
                delta=delta,
                citations=_coerce_citations_field(item.get("citations")),
                supporting_evidence_hint=_coerce_string_field(item.get("supporting_evidence_hint")),
            )
        )
    return claims


def _strip_claims_block(full_response: str) -> str:
    """Remove the trailing ```json ...``` block from the response text so
    the user-facing answer doesn't show the machine-readable claims list."""
    return _JSON_BLOCK_RE.sub("", full_response).rstrip()


async def _call_model_for_answer(engine: "LocalSearch", system_prompt: str, query: str) -> AnswerResult:
    """Shared plumbing for both generate_answer() and regenerate_answer():
    send one [system, user] message pair to the model, stream the response,
    and split it into (prose, structured claims). Mirrors
    LocalSearch.search()'s own model-call pattern (same model, same call
    params) so neither function asks anything different of the model client
    than the existing pipeline already does."""
    messages = (
        CompletionMessagesBuilder()
        .add_system_message(system_prompt)
        .add_user_message(query)
        .build()
    )

    full_response = ""
    response = await engine.model.completion_async(
        messages=messages,
        stream=True,
        **engine.model_params,
    )
    async for chunk in response:
        full_response += chunk.choices[0].delta.content or ""

    claims = _parse_claims(full_response)
    prose = _strip_claims_block(full_response)
    return AnswerResult(text=prose, claims=claims, llm_calls=1)


def _context_data_with_facts(context_result: "ContextBuilderResult") -> str:
    """The GPIL Knowledge Layer glossary section (see knowledge_layer.py)
    PREPENDED before context_chunks, followed by the deterministic Atomic
    Distributor/State/Category/SKU Facts section AND (Design Decision: the
    LLM must never compute a SKU ranking itself) the Deterministic SKU
    Ranking section appended after -- shared by generate_answer() and
    regenerate_answer() so the first draft and the one bounded retry always
    see identical structured evidence. Purely additive: the original
    context_chunks text is always included unchanged; getattr's default
    handles ContextBuilderResult-like stand-ins (e.g. in tests) that don't
    set context_records, in which case no block is added at all.

    WHY THE GLOSSARY BLOCK IS PREPENDED, NOT APPENDED (2026-08-23, live
    "What is GPI and IPM?" investigation): GraphRAG's OWN index -- built
    from a corpus that never defines GPI/IPM -- contains real, retrievable
    Entity nodes titled "GPI"/"IPM" whose descriptions were auto-written by
    an LLM at INDEX TIME from sparse context. IPM's indexed description
    drifts toward "agricultural practices"/"agricultural cycles" language
    (plausible-sounding, but nothing in this project's data says anything
    about agriculture) -- close enough to reinforce, rather than
    contradict, the model's strong pretrained "Integrated Pest Management"
    prior. Because context_chunks (GraphRAG's own retrieved/formatted text,
    including that Entity description) used to come FIRST and the glossary
    block was only appended at the very end, the model read the vague/
    misleading indexed description as "the real evidence" long before ever
    reaching the correct, authoritative glossary entry -- and grounding
    correctly rejected the resulting wrong answer every time, but the
    right answer was never produced in the first place. Putting the
    glossary block first makes it the FIRST thing the model reads in the
    Data tables section (see also _GLOSSARY_CITATION_GUIDANCE, which now
    explicitly says this block overrides any other, including retrieved
    Entity/Report descriptions -- not just the model's own outside
    knowledge).

    WHY THE GLOSSARY BLOCK IS ALSO REPEATED RIGHT AFTER context_chunks
    (2026-08-23 follow-up, "What is IPM?" SINGLE-TERM investigation): "What
    is GPI and IPM?" was confirmed fixed by the prepend above, but "What is
    IPM?" asked alone still failed live. The real indexed "IPM" Entity
    description turned out to be a long, multi-paragraph narrative that
    repeats agriculture-adjacent phrasing across seven separate states/
    periods -- far longer and more repetitive than GPI's much shorter,
    vaguer description, and a single-term question concentrates GraphRAG's
    entire retrieval budget on that one entity's neighborhood (a two-term
    question splits that budget, which is likely why it wasn't observed to
    fail). A single opening glossary block, however well-positioned or
    strongly worded, is comparatively short next to that competing
    narrative. build_glossary_reminder_block() re-renders the SAME matched
    definitions under a distinct header immediately after context_chunks --
    bookending the competing content on both sides with authoritative,
    identical facts, giving the correct definition the same kind of
    recency the competing narrative would otherwise have to itself, right
    before the rest of the generation guidance. Still purely additive and
    still term-scoped (knowledge_layer.py's own detection, unchanged) --
    no edit to GraphRAG's retrieved text, no new forbidden-phrase entry, no
    hardcoded final answer."""
    context_records = getattr(context_result, "context_records", None) or {}
    glossary_block = build_glossary_block(context_records)
    glossary_reminder = build_glossary_reminder_block(context_records)
    trailing_blocks = [
        block
        for block in (build_atomic_facts_block(context_records), build_deterministic_ranking_block(context_records))
        if block
    ]
    parts = (
        ([glossary_block] if glossary_block else [])
        + [context_result.context_chunks]
        + ([glossary_reminder] if glossary_reminder else [])
        + trailing_blocks
    )
    return "\n\n".join(parts)


async def generate_answer(
    engine: "LocalSearch", query: str, context_result: "ContextBuilderResult"
) -> AnswerResult:
    """Call the completion model once: draft the prose answer AND the
    structured claims block in a single request. Mirrors
    LocalSearch.search()'s own prompt-building pattern (same system prompt
    template) so this asks nothing different of the model than the
    existing pipeline already does, plus the claim-extraction addendum."""
    search_prompt = (
        engine.system_prompt.format(
            context_data=_context_data_with_facts(context_result),
            response_type=engine.response_type,
        )
        + _SPECIFICITY_AND_FRAMING_GUIDANCE
        + _VALUE_ATTRIBUTION_GUIDANCE
        + _ENTITY_EXISTENCE_GUIDANCE
        + _RESOLUTION_REASONING_GUIDANCE
        + _CITATION_SELECTION_GUIDANCE
        + _ATOMIC_FACTS_CITATION_GUIDANCE
        + _GLOSSARY_CITATION_GUIDANCE
        + _CLAIM_EXTRACTION_SUFFIX
    )
    return await _call_model_for_answer(engine, search_prompt, query)


_RETRY_FIX_INSTRUCTIONS: dict[str, str] = {
    "unsupported_qualifier": "Reword or remove this qualifier; state only what the data tables support.",
    "direction_contradiction": (
        "Correct the direction word so it matches the data tables. If this sentence is a future "
        "recommendation rather than a historical statement, either remove the directional word entirely "
        "or rephrase it so it clearly reads as a forward-looking action (e.g. 'monitor Service Level "
        "going forward') rather than a claim about which way the metric already moved."
    ),
    "unsupported_numeric": (
        "This number isn't backed by the evidence this claim cited -- fix the number or its citation. If no "
        "sentence anywhere in the data tables states this metric/value for this exact entity (check: is the "
        "entity only named in a roster/listing, with no deviation sentence of its own?), this claim is "
        "fabricated -- remove it entirely rather than guessing or reusing a nearby entity's value."
    ),
    "unsupported_entity": "The evidence this claim cited doesn't establish it's about this entity -- fix the citation or remove the claim.",
    "unsupported_period": "The evidence this claim cited doesn't establish it's about this period -- fix the citation or remove the claim.",
    "unsupported_delta": "The stated change amount doesn't match the cited values -- correct the number or remove it.",
    "unsupported_deviation": (
        "The cited evidence doesn't explicitly describe a deviation. If the SAME data tables include a Sources "
        "record that does state this fact using the word 'deviation' (the terser record you cited, e.g. a "
        "Relationships or Entities record, may just be missing that word), switch the citation to that Sources "
        "record. Otherwise, reword as a plain fact instead, or remove the deviation framing."
    ),
    "unsupported_recommendation": (
        "This recommendation states a number that isn't backed by the evidence -- correct it to match the "
        "data tables, remove the number, or make the recommendation without stating an unsupported figure."
    ),
    "unsupported_comparison": (
        "This claim's comparison side (comparison_period and/or comparison_entity) and comparison_value are "
        "never stated together as one fact anywhere in the cited evidence (each may appear separately, for a "
        "different period, entity, or metric/category) -- add a citation to the source that actually states "
        "this metric's value for that comparison period/entity, correct whichever field is wrong, or remove "
        "the comparison."
    ),
    "unsupported_causal": (
        "This claim asserts or implies causation, but the cited evidence only states facts, not a causal "
        "relationship. Reword using hedged, non-causal language (e.g. 'coincided with', 'occurred alongside', "
        "'is one of several factors present during') instead of 'caused'/'contributed to'/'led to'/'due to', "
        "or remove the causal claim entirely if it isn't essential."
    ),
    "unsupported_ranking": (
        "The named entity is not actually the highest/lowest value for this metric among the Atomic Facts "
        "in the data tables -- name the entity whose Atomic Fact actually holds the extreme value instead."
    ),
    "unsupported_definition": (
        "This sentence states what a term stands for/means, but its citation doesn't back that up. If the "
        "data tables include a \"-----GPIL Knowledge Layer-----\" section, that term's entry there is the "
        "correct GPIL-specific meaning -- rewrite the sentence to match it exactly and cite that entry's own "
        "\"Source: Sources (glossary-...)\" identifier. Do not cite an Entity or Report description instead, "
        "even if one exists for this term -- it was not written from a confirmed definition."
    ),
    "wrong_glossary_expansion": (
        "This uses a generic/world-knowledge meaning already confirmed WRONG for this term in this system -- "
        "replace it with the exact meaning given in the \"-----GPIL Knowledge Layer-----\" section of the data "
        "tables above (see that term's own Definition: line), and cite its \"Source: Sources (glossary-...)\" "
        "identifier. Do not reuse any other retrieved Entity/Report description of this term as a substitute -- "
        "only the Knowledge Layer entry is confirmed correct."
    ),
}


# A live failure (see grounding_check.py's _detect_entity_fusion()) showed
# the model fuse multiple real, individually-supported distributor names
# into one entity field (e.g. "Mooney, Lamb and Weber, Scott-Norman
# Distributors"), and a bounded retry told only "the evidence doesn't
# establish it's about this entity -- fix the citation or remove the
# claim" did not fix it: the model re-fused the same names on the retry
# pass instead of splitting them. grounding_check.py detects this specific
# shape deterministically (never changing whether the claim passes -- see
# that function's docstring) and marks it with this exact substring in the
# issue's `detail` text; when present, the retry gets this much more
# direct instruction instead of the generic unsupported_entity one.
_ENTITY_FUSION_DETAIL_MARKER = "combines multiple distinct named entities"
_ENTITY_FUSION_FIX_INSTRUCTION = (
    "The claim combines multiple distinct entities into one entity field. Split them into "
    "separate claims. Do not invent a combined entity name unless that exact entity appears "
    "in the evidence."
)


def _format_issue_for_retry(issue: GroundingIssue) -> str:
    """One line per flagged grounding issue, phrased as an instruction the
    model can act on directly. Works for both claim-level issues (which
    name a specific claim_id) and prose-fallback issues (which quote the
    offending sentence)."""
    if issue.issue_type == "unsupported_entity" and _ENTITY_FUSION_DETAIL_MARKER in issue.detail:
        fix = _ENTITY_FUSION_FIX_INSTRUCTION
    else:
        fix = _RETRY_FIX_INSTRUCTIONS.get(issue.issue_type, "Fix this issue using only what the data tables support.")
    origin = f"claim {issue.claim_id}" if issue.source == "claim" and issue.claim_id is not None else "the sentence"
    return f'- [{issue.issue_type}] In {origin}: "{issue.sentence}" -- {issue.detail} {fix}'


_RETRY_SYSTEM_PROMPT_TEMPLATE = """
---Role---

You are revising a previously drafted answer so that every claim in it is
fully grounded in the data tables below. You are NOT answering the
question from scratch.

---Data tables---

{context_data}

---Original Question---

{query}

---Draft Answer (to be revised)---

{draft_answer}

---Problems Found In The Draft (must be fixed)---

{issues_text}

---Revision Instructions---

- Fix ONLY the problems listed above.
- Preserve every verified numeric fact, direction, relationship, and
  citation from the draft exactly as it is, unless the specific flagged
  problem requires changing it.
- Preserve every valid citation exactly as written.
- Remove or reword unsupported causal or qualitative claims flagged
  above. Where the evidence supports a WEAKER statement than the draft
  made, replace the unsupported certainty with explicitly hedged wording
  (e.g. "coincided with", "occurred alongside", "is one of several
  factors present during") instead of causal language like "caused" /
  "contributed to" / "led to" / "due to" -- do not simply delete the
  sentence if a hedged, evidence-supported version is possible.
- Do not change any claim that wasn't flagged above.
- Do not introduce any new factual claims that weren't already in the
  draft or the data tables.
- Keep the same citation format: [Data: <dataset name> (record ids)].
- Return the full revised answer text (not a diff, not just the changed
  sentence).
{specificity_guidance}
{claim_extraction_suffix}
"""


async def regenerate_answer(
    engine: "LocalSearch",
    query: str,
    context_result: "ContextBuilderResult",
    draft: AnswerResult,
    grounding: GroundingCheckResult,
) -> AnswerResult:
    """The one bounded retry: send the original context, the draft answer,
    and exactly what grounding_check() flagged back to the model, asking it
    to fix ONLY those specific problems. Callers (pipeline.py) are
    responsible for calling this AT MOST ONCE per query and for re-running
    grounding_check() on the result -- this function has no looping or
    self-retry logic of its own, by design.

    Only severity="error" issues are listed as "must be fixed" -- a
    severity="warning" issue (currently just "generic_aggregation": a
    supported-but-generic entity reference) isn't a grounding failure, so
    telling the model to "fix" it here would just pressure it to strip out
    the very specificity Part 1/3 of the Phase 7 hardening pass are trying
    to preserve. grounding.status is only ever "fail" (triggering a retry
    at all) because of an error-severity issue, but a fail CAN co-occur
    with a warning on an unrelated part of the same draft -- so the filter
    still matters even though it can never make issues_text empty here.
    """
    error_issues = [issue for issue in grounding.flagged_issues if issue.severity == "error"]
    issues_text = "\n".join(_format_issue_for_retry(issue) for issue in error_issues)
    retry_prompt = _RETRY_SYSTEM_PROMPT_TEMPLATE.format(
        context_data=_context_data_with_facts(context_result),
        query=query,
        draft_answer=draft.text,
        issues_text=issues_text,
        specificity_guidance=(
            _SPECIFICITY_AND_FRAMING_GUIDANCE
            + _VALUE_ATTRIBUTION_GUIDANCE
            + _ENTITY_EXISTENCE_GUIDANCE
            + _RESOLUTION_REASONING_GUIDANCE
            + _CITATION_SELECTION_GUIDANCE
            + _ATOMIC_FACTS_CITATION_GUIDANCE
            + _GLOSSARY_CITATION_GUIDANCE
        ),
        claim_extraction_suffix=_CLAIM_EXTRACTION_SUFFIX,
    )
    return await _call_model_for_answer(engine, retry_prompt, query)
