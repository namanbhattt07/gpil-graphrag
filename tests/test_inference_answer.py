"""
Tests for src/inference/answer.py: generate_answer(), regenerate_answer(),
and their shared claim-parsing helpers.

These mock the LLM one level deeper than test_inference_pipeline.py does --
instead of mocking answer_fn/retry_fn themselves, they mock
engine.model.completion_async() (the actual GraphRAG model client call), so
the real prompt-building and response-parsing code in answer.py gets
exercised. No real API calls anywhere in this file.
"""

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from src.inference.answer import (
    _coerce_numeric_field,
    _format_issue_for_retry,
    _parse_claims,
    _strip_claims_block,
    generate_answer,
    regenerate_answer,
)
from src.inference.grounding_check import check_grounding
from src.inference.schemas import AnswerResult, GroundingCheckResult, GroundingIssue


class _FakeChunk:
    def __init__(self, text: str):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]


class _FakeModel:
    """Stands in for graphrag_llm's completion client. Records every call's
    messages so tests can inspect exactly what was sent to 'the model',
    and streams back a canned response in one chunk."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[list[dict]] = []

    async def completion_async(self, messages, stream=True, **kwargs):
        self.calls.append(messages)

        async def _stream():
            yield _FakeChunk(self.response_text)

        return _stream()


def make_fake_engine(response_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=_FakeModel(response_text),
        system_prompt="---Role---\n...\n{context_data}\n...\n{response_type}\n...",
        response_type="Multiple Paragraphs",
        model_params={},
    )


# ---------------------------------------------------------------------------
# _parse_claims / _strip_claims_block
# ---------------------------------------------------------------------------


def test_parse_claims_extracts_trailing_json_block():
    text = (
        "Productivity climbed from 85.5% to 89.0%.\n\n"
        '```json\n[{"metric": "Productivity", "period": "November 2025", '
        '"value": 89.0, "direction": "climbed", "citations": ["Entities (4)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert len(claims) == 1
    assert claims[0].metric == "Productivity"
    assert claims[0].direction == "climbed"
    assert claims[0].citations == ["Entities (4)"]


def test_parse_claims_extracts_comparison_entity():
    """Phase 8c: comparison_entity (the second entity in a cross-entity
    comparison claim) must round-trip through the JSON claims block the
    same way every other field does."""
    text = (
        "Sikkim's Ferrero OOS (6.6%) was higher than Maharashtra's (6.5%).\n\n"
        '```json\n[{"claim_type": "comparison", "entity": "Sikkim", '
        '"comparison_entity": "Maharashtra", "metric": "Out-of-Stock Rate", '
        '"value": 6.6, "comparison_value": 6.5, "citations": ["Sources (1)", "Sources (15)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert len(claims) == 1
    assert claims[0].entity == "Sikkim"
    assert claims[0].comparison_entity == "Maharashtra"


def test_parse_claims_returns_empty_list_when_no_block():
    assert _parse_claims("Just prose, no claims block.") == []


def test_parse_claims_returns_empty_list_on_malformed_json():
    text = "Prose.\n```json\n[not valid json]\n```"
    assert _parse_claims(text) == []


# ---------------------------------------------------------------------------
# Live crash fix (2026-08-23 stabilization pass): a real two-SKU comparison
# question had the model emit "entity": ["Marlboro Pack 1",
# "GPI_Franchise_1 Pack 1"] -- a JSON array of BOTH names, instead of using
# comparison_entity for the second one as the system prompt instructs.
# Every string-typed claim field downstream (starting with
# grounding_check.py's _mentions()) assumed a plain string and crashed with
# a bare AttributeError ('list' object has no attribute 'strip'), which
# propagated all the way up through run_diagnostic_query() as an unhandled
# DiagnosticQueryError instead of a normal PipelineResult. _parse_claims()
# must never construct an AnswerClaim whose string fields could still crash
# a downstream .strip()/.lower() call, no matter how the model malforms its
# JSON output.
# ---------------------------------------------------------------------------


def test_parse_claims_entity_as_list_does_not_crash_and_becomes_none():
    text = (
        "Comparing two SKUs.\n\n"
        '```json\n[{"claim_type": "comparison", '
        '"entity": ["Marlboro Pack 1", "GPI_Franchise_1 Pack 1"], '
        '"metric": "Revenue", "value": 100.0, "citations": ["Sources (1)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert len(claims) == 1
    assert claims[0].entity is None  # malformed (a list) -- coerced to absent, never left as a list
    assert claims[0].value == 100.0  # sibling fields on the SAME claim are unaffected


def test_parse_claims_entity_as_list_claim_survives_full_grounding_check_without_crashing():
    """End-to-end: the exact live failure shape must produce a normal
    (if unsupported) GroundingCheckResult, never raise."""
    from src.inference.grounding_check import check_grounding

    text = (
        "Comparing two SKUs.\n\n"
        '```json\n[{"claim_type": "comparison", '
        '"entity": ["Marlboro Pack 1", "GPI_Franchise_1 Pack 1"], '
        '"metric": "Revenue", "value": 100.0, "citations": ["Sources (1)"]}]\n```'
    )
    claims = _parse_claims(text)
    result = check_grounding("Comparing two SKUs.", {}, claims)  # must not raise
    assert result.status in ("pass", "fail")


@pytest.mark.parametrize(
    "field",
    ["claim_text", "claim_type", "metric", "period", "comparison_period", "comparison_entity", "direction"],
)
def test_parse_claims_every_string_field_rejects_a_list_value(field):
    """Regression guard: the SAME malformed-list crash risk applies to
    every string-typed AnswerClaim field, not just `entity` -- each must
    independently coerce to None rather than passing a list through."""
    text = (
        "Some claim.\n\n"
        f'```json\n[{{"{field}": ["a", "b"], "value": 1.0, "citations": []}}]\n```'
    )
    claims = _parse_claims(text)
    assert len(claims) == 1
    assert getattr(claims[0], field) is None


def test_parse_claims_citations_as_bare_string_does_not_iterate_characters():
    """Live-adjacent defensive fix: `list("Sources (21)")` would silently
    explode a single real citation string into one 'citation' per
    character. citations must stay [] (never a garbled per-character
    list) when the model writes a bare string instead of a JSON array."""
    text = 'Prose.\n\n```json\n[{"metric": "Revenue", "value": 1.0, "citations": "Sources (21)"}]\n```'
    claims = _parse_claims(text)
    assert claims[0].citations == []


def test_parse_claims_citations_drops_non_string_entries_but_keeps_real_ones():
    text = 'Prose.\n\n```json\n[{"metric": "Revenue", "value": 1.0, "citations": ["Sources (21)", 5, null]}]\n```'
    claims = _parse_claims(text)
    assert claims[0].citations == ["Sources (21)"]


def test_parse_claims_empty_string_fields_become_none():
    text = 'Prose.\n\n```json\n[{"entity": "", "metric": "Revenue", "value": 1.0, "citations": []}]\n```'
    claims = _parse_claims(text)
    assert claims[0].entity is None


def test_strip_claims_block_removes_json_leaving_prose():
    text = 'Prose here.\n\n```json\n[{"metric": "X"}]\n```'
    assert _strip_claims_block(text) == "Prose here."


# ---------------------------------------------------------------------------
# _format_issue_for_retry
# ---------------------------------------------------------------------------


def test_format_issue_for_retry_qualifier_names_the_term():
    issue = GroundingIssue(
        issue_type="unsupported_qualifier", sentence="Rates were alarming.", term="alarming", detail="not in evidence"
    )
    formatted = _format_issue_for_retry(issue)
    assert "alarming" in formatted
    assert "Reword or remove" in formatted


def test_format_issue_for_retry_direction_contradiction_names_the_fix():
    issue = GroundingIssue(
        issue_type="direction_contradiction", sentence="Productivity climbed.", term="up", detail="evidence shows down"
    )
    formatted = _format_issue_for_retry(issue)
    assert "Correct the direction word" in formatted


def test_format_issue_for_retry_entity_fusion_gets_explicit_split_instruction():
    """When grounding_check.py's detail carries the entity-fusion marker
    (see _detect_entity_fusion()), the retry instruction must be the
    specific split-into-separate-claims wording, not the generic
    unsupported_entity fallback."""
    issue = GroundingIssue(
        issue_type="unsupported_entity",
        sentence="Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors reported 20.0% OOS.",
        term="Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors",
        detail=(
            "The claim entity 'Kowalski, Ortiz and Reyes, Vance-Whitfield Distributors' combines "
            "multiple distinct named entities (Kowalski, Ortiz and Reyes, Vance-Whitfield) into one "
            "field -- the cited evidence never states them as one combined entity."
        ),
        source="claim",
        claim_id=2,
    )
    formatted = _format_issue_for_retry(issue)
    assert "Split them into separate claims" in formatted
    assert "Do not invent a combined entity name unless that exact entity appears in the evidence" in formatted


def test_format_issue_for_retry_plain_unsupported_entity_keeps_generic_instruction():
    """A plain (non-fused) unsupported_entity issue must be completely
    unaffected -- same generic fix instruction as before this change."""
    issue = GroundingIssue(
        issue_type="unsupported_entity",
        sentence="Bihar's productivity was 89.0%.",
        term="Bihar",
        detail="The cited evidence does not mention 'Bihar'.",
        source="claim",
        claim_id=0,
    )
    formatted = _format_issue_for_retry(issue)
    assert "fix the citation or remove the claim" in formatted
    assert "Split them into separate claims" not in formatted


# ---------------------------------------------------------------------------
# generate_answer() -- exactly one model call, correct parsing
# ---------------------------------------------------------------------------


def test_generate_answer_calls_model_once_and_parses_response():
    response_text = (
        'Bihar\'s Productivity was 85.5% in October 2025 [Data: Sources (0)].\n\n'
        '```json\n[{"metric": "Productivity", "period": "October 2025", '
        '"value": 85.5, "direction_word": null, "citations": ["Sources (0)"]}]\n```'
    )
    engine = make_fake_engine(response_text)
    context_result = SimpleNamespace(context_chunks="<fake context>")

    result = asyncio.run(generate_answer(engine, "What was Bihar's Productivity?", context_result))

    assert isinstance(result, AnswerResult)
    assert "85.5%" in result.text
    assert "```json" not in result.text  # claims block stripped from user-facing text
    assert len(result.claims) == 1
    assert result.llm_calls == 1
    assert len(engine.model.calls) == 1  # exactly one completion call


# ---------------------------------------------------------------------------
# regenerate_answer() -- sends draft + flagged issues, exactly one model call
# ---------------------------------------------------------------------------


def test_regenerate_answer_sends_draft_and_flagged_issues_in_the_prompt():
    draft = AnswerResult(
        text="Bihar's out-of-stock rate was an alarming 5.2% [Data: Sources (0)].",
        claims=[],
        llm_calls=1,
    )
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_qualifier",
                sentence="Bihar's out-of-stock rate was an alarming 5.2%.",
                term="alarming",
                detail="'alarming' does not appear in the retrieved evidence.",
            )
        ],
    )
    corrected_text = "Bihar's out-of-stock rate was 5.2% [Data: Sources (0)]."
    engine = make_fake_engine(corrected_text + "\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with the real evidence>")

    result = asyncio.run(
        regenerate_answer(engine, "Why did Bihar's out-of-stock rate rise?", context_result, draft, grounding)
    )

    assert len(engine.model.calls) == 1  # exactly one retry call
    sent_system_message = engine.model.calls[0][0]["content"]
    # The retry prompt must actually include the original context, the
    # draft answer, and the specific flagged term -- per requirement 3.
    assert "<fake context with the real evidence>" in sent_system_message
    assert draft.text in sent_system_message
    assert "alarming" in sent_system_message
    assert result.text == corrected_text


def test_regenerate_answer_parses_corrected_claims():
    draft = AnswerResult(text="draft", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(status="fail", method="deterministic", flagged_issues=[])
    response_text = (
        'Productivity fell from 92.0% to 85.5% [Data: Entities (4)].\n\n'
        '```json\n[{"metric": "Productivity", "period": "October 2025", '
        '"value": 85.5, "direction": "fell", "citations": ["Entities (4)"]}]\n```'
    )
    engine = make_fake_engine(response_text)
    context_result = SimpleNamespace(context_chunks="<ctx>")

    result = asyncio.run(regenerate_answer(engine, "Why did it decline?", context_result, draft, grounding))
    assert len(result.claims) == 1
    assert result.claims[0].direction == "fell"


# ---------------------------------------------------------------------------
# Phase 7d: citation-selection hardening (prompt-side only -- see
# grounding_check.py's own tests for the behavioral proof that a
# Relationships citation still validates when it's the genuine support).
#
# Test-to-requirement map:
#   A: test_prompt_contains_citation_selection_hierarchy
#   B: test_prompt_citation_hierarchy_does_not_mandate_always_citing_sources
#      (+ claim-extraction requirement) test_claim_extraction_suffix_instructs_most_direct_citation
#   wiring: test_generate_answer_prompt_includes_citation_selection_guidance
#           test_regenerate_answer_prompt_includes_citation_selection_guidance
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def test_prompt_contains_citation_selection_hierarchy():
    """A: the guidance must state the ordered preference (Sources/Reports
    with a direct statement > Entities with a direct statement >
    Relationships only when the relationship itself is the fact), using
    the exact Landry Ltd worked example from the Phase 7d spec."""
    from src.inference.answer import _CITATION_SELECTION_GUIDANCE

    normalized = _normalize_whitespace(_CITATION_SELECTION_GUIDANCE)
    assert "Prefer Sources when the source document explicitly contains" in normalized
    assert "Relationships should be cited primarily when the relationship itself" in normalized
    assert "Landry Ltd Distributors showed an Out-of-Stock deviation of 28.6%" in normalized
    assert "Landry Ltd Distributors is associated with Bihar" in normalized


def test_prompt_citation_hierarchy_does_not_mandate_always_citing_sources():
    """B: the hierarchy must not collapse into a blanket 'always cite
    Sources' rule -- Reports/Entities/Relationships must still be
    presented as legitimate choices for claims they genuinely support."""
    from src.inference.answer import _CITATION_SELECTION_GUIDANCE

    normalized = _normalize_whitespace(_CITATION_SELECTION_GUIDANCE).lower()
    assert "not a rule to always cite sources" in normalized
    assert "reports" in normalized and "entities" in normalized and "relationships" in normalized


def test_claim_extraction_suffix_instructs_most_direct_citation():
    """Claim extraction must tell the model to pick, for EACH claim, the
    record that most directly and completely supports THAT claim -- not
    just any record that happens to mention the right entities."""
    from src.inference.answer import _CLAIM_EXTRACTION_SUFFIX

    normalized = _normalize_whitespace(_CLAIM_EXTRACTION_SUFFIX).lower()
    assert "most directly and completely" in normalized


def test_generate_answer_prompt_includes_citation_selection_guidance():
    """Wiring check: the citation-selection guidance must actually reach
    the model in generate_answer()'s prompt, not just exist as an unused
    constant."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "Which distributors deviated?", context_result))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Citation Selection" in sent_system_message
    assert "Landry Ltd Distributors showed an Out-of-Stock deviation of 28.6%" in sent_system_message


# ---------------------------------------------------------------------------
# Atomic-facts citation-format hardening -- fixes a live Q5 failure where
# the model correctly bound each fact's own period/value/direction but
# cited the fact by its ATOMIC FACTS SECTION NAME AND NUMBER ("Atomic
# State KPI Facts (1)") instead of that fact's own embedded Source:
# identifier ("Sources (21)") -- an unresolvable citation that silently
# drops out in grounding_check.py's _resolve_citations(), leaving the
# claim looking unsupported even though the fact and a genuine Sources
# citation for it were both correct. Prompt-only fix; grounding_check.py
# is untouched.
# ---------------------------------------------------------------------------


def test_atomic_facts_citation_guidance_forbids_citing_the_section_itself():
    """The guidance must explicitly say NOT to cite the Atomic Facts
    section name/number, and must name all three real section titles."""
    from src.inference.answer import _ATOMIC_FACTS_CITATION_GUIDANCE

    normalized = _normalize_whitespace(_ATOMIC_FACTS_CITATION_GUIDANCE)
    assert "do NOT cite the Atomic Facts section itself" in normalized
    assert "Atomic State KPI Facts" in normalized
    assert "Atomic Distributor Facts" in normalized
    assert "Atomic Category KPI Facts" in normalized
    assert '"Atomic State KPI Facts (1)" is never a resolvable citation' in normalized


def test_atomic_facts_citation_guidance_uses_the_embedded_source_field():
    """The guidance must tell the model to use the FACT's own embedded
    Source: identifier instead, with the exact real worked example from
    the live Q5 failure (Gujarat Productivity, June 2026, Sources (21))."""
    from src.inference.answer import _ATOMIC_FACTS_CITATION_GUIDANCE

    normalized = _normalize_whitespace(_ATOMIC_FACTS_CITATION_GUIDANCE)
    assert "Source: Sources (21)" in normalized
    assert 'the correct citation is "Sources (21)"' in normalized
    assert "authoritative citation" in normalized.lower()


def test_atomic_facts_citation_guidance_states_multi_period_citation_rule():
    """A trend/comparison claim spanning two periods must be told to cite
    BOTH periods' sources, not rely on a single (possibly wrong-period)
    Sources citation plus an unresolvable Atomic Facts reference -- the
    exact mechanism of the live Q5 failure."""
    from src.inference.answer import _ATOMIC_FACTS_CITATION_GUIDANCE

    normalized = _normalize_whitespace(_ATOMIC_FACTS_CITATION_GUIDANCE)
    assert '["Sources (22)", "Sources (21)"]' in normalized
    assert "cite evidence for both periods" in normalized.lower()


def test_generate_answer_prompt_includes_atomic_facts_citation_guidance():
    """Wiring check: the guidance must actually reach the model in
    generate_answer()'s prompt, not just exist as an unused constant."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "How did Gujarat's Productivity change?", context_result))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Citation Rules For Atomic Facts Sections" in sent_system_message
    assert "Source: Sources (21)" in sent_system_message


def test_regenerate_answer_prompt_includes_atomic_facts_citation_guidance():
    """Wiring check: the same guidance must also reach the model on the
    retry path, since this is exactly the case where the FIRST draft's
    retry inherited the same broken citation unchanged in the live Q5
    failure (the generic retry-fix instructions gave it no reason to
    change a citation it believed was already correct)."""
    draft = AnswerResult(
        text="draft citing Atomic State KPI Facts (1) only",
        claims=[],
        llm_calls=1,
    )
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_period",
                sentence="Productivity rose to 74.3% in June 2026.",
                term="June 2026",
                detail="The cited evidence does not mention the period 'June 2026'.",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with the real Sources records>")

    asyncio.run(regenerate_answer(engine, "How did Gujarat's Productivity change?", context_result, draft, grounding))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Citation Rules For Atomic Facts Sections" in sent_system_message
    assert "Source: Sources (21)" in sent_system_message


# ---------------------------------------------------------------------------
# Phase 8: value-attribution hardening (prompt-side only -- fixes a live
# failure where a real, evidence-present value was copied onto the WRONG
# neighboring entity, e.g. Candy's ACV restated as Ferrero's ACV, because
# the two categories' figures sit in adjacent, near-identically-shaped
# sentences in the same source document). grounding_check.py is untouched
# by this pass -- see the Phase 8 audit for why the deterministic checker
# has a separate, not-yet-addressed blind spot for this exact shape.
#
# Test-to-requirement map:
#   A: test_generate_answer_prompt_includes_value_attribution_guidance
#   B: test_regenerate_answer_prompt_includes_value_attribution_guidance
#   C: test_claim_extraction_suffix_references_value_attribution
#   D: test_value_attribution_guidance_preserves_legitimate_comparisons
#   E: test_value_attribution_guidance_is_generic_not_ferrero_candy_special_cased
# ---------------------------------------------------------------------------


def test_generate_answer_prompt_includes_value_attribution_guidance():
    """A: wiring check -- the guidance must actually reach the model in
    generate_answer()'s prompt, not just exist as an unused constant."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "Why did Bihar's Service Level decline?", context_result))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Value Attribution" in sent_system_message
    assert "SAME entity/category" in sent_system_message


def test_regenerate_answer_prompt_includes_value_attribution_guidance():
    """B: the same guidance must also reach the model on the retry path --
    otherwise a retry could reintroduce the same attribution error while
    fixing something else, since a retry is a second full drafting call."""
    draft = AnswerResult(text="Ferrero ACV was 27.7% in November 2025.", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_numeric",
                sentence="Ferrero ACV was 27.7% in November 2025.",
                term="27.7",
                detail="Claimed value 27.7 does not appear in the cited evidence for Ferrero.",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with Ferrero and Candy ACV figures>")

    asyncio.run(
        regenerate_answer(engine, "Why did Bihar's Service Level decline?", context_result, draft, grounding)
    )

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Value Attribution" in sent_system_message
    assert "SAME entity/category" in sent_system_message


def test_claim_extraction_suffix_references_value_attribution():
    """C: the structured-claims instructions must tell the model to
    re-confirm each claim's value/comparison_value against that claim's
    own entity, metric, and period before finalizing it."""
    from src.inference.answer import _CLAIM_EXTRACTION_SUFFIX

    normalized = _normalize_whitespace(_CLAIM_EXTRACTION_SUFFIX)
    assert "see Value Attribution above" in normalized
    assert "re-confirm it is the exact figure the evidence states for THIS claim" in normalized


def test_value_attribution_guidance_preserves_legitimate_comparisons():
    """D: the guidance must explicitly say it does NOT forbid reporting
    multiple similar values / legitimate comparisons -- only that each one
    must be independently, correctly attributed."""
    from src.inference.answer import _VALUE_ATTRIBUTION_GUIDANCE

    normalized = _normalize_whitespace(_VALUE_ATTRIBUTION_GUIDANCE)
    assert "does NOT prohibit legitimate comparisons" in normalized
    assert "Service Level 94.3% in October versus 92.9% in November" in normalized
    assert "distributor's value versus the state average" in normalized


def test_value_attribution_guidance_is_generic_not_ferrero_candy_special_cased():
    """E: the RULE itself must be phrased generically (same entity/
    category + same metric + same period) -- Ferrero/Candy may appear only
    as an illustrative example inside the 'legitimate comparisons' list,
    never as a hardcoded special case or conditional logic singling them
    out from any other entity/category pair."""
    from src.inference.answer import _VALUE_ATTRIBUTION_GUIDANCE

    normalized = _normalize_whitespace(_VALUE_ATTRIBUTION_GUIDANCE)
    # The governing rule is stated in fully generic terms.
    assert "SAME entity/category, the SAME metric, and the SAME period" in normalized
    # No conditional/special-case language singling out Ferrero or Candy.
    assert "if entity is ferrero" not in normalized.lower()
    assert "always treat ferrero" not in normalized.lower()
    assert "ferrero" not in normalized.lower().split("legitimate comparisons")[0]
    # Ferrero/Candy appear at most once each, only inside the one illustrative example.
    assert normalized.count("Ferrero") <= 1
    assert normalized.count("Candy") <= 1


def test_regenerate_answer_prompt_includes_citation_selection_guidance():
    """Wiring check: the same guidance must also reach the model on the
    retry path, since a retry is a second full drafting call."""
    draft = AnswerResult(text="draft naming Relationships (10) only", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_entity",
                sentence="Landry Ltd Distributors showed a deviation.",
                term="Landry Ltd Distributors",
                detail="The cited evidence does not mention 'Landry Ltd Distributors'.",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with the real Sources record>")

    asyncio.run(regenerate_answer(engine, "Which distributors deviated?", context_result, draft, grounding))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Citation Selection" in sent_system_message
    assert "Landry Ltd Distributors showed an Out-of-Stock deviation of 28.6%" in sent_system_message


# ---------------------------------------------------------------------------
# Phase 8 Step 9: numeric claim-field coercion. Fixes a live
# "TypeError: unsupported operand type(s) for -: 'str' and 'float'" crash
# in grounding_check.py's arithmetic, caused by a model JSON claims block
# reporting value/comparison_value/delta as something other than a bare
# JSON number (e.g. a quoted "92.9", or "N/A"). _parse_claims() is the
# ONLY thing that changed -- grounding_check.py is untouched; a malformed
# value/comparison_value becomes float("nan") (not None) specifically so
# the EXISTING grounding checks still see the field as "present" and
# correctly flag it, rather than silently skipping it the way a
# genuinely-absent field is meant to.
# ---------------------------------------------------------------------------


def test_coerce_numeric_field_accepts_native_numbers():
    assert _coerce_numeric_field(92.9) == (92.9, False)
    assert _coerce_numeric_field(93) == (93.0, False)


def test_coerce_numeric_field_accepts_clean_numeric_string():
    assert _coerce_numeric_field("92.9") == (92.9, False)
    assert _coerce_numeric_field("-3.5") == (-3.5, False)


def test_coerce_numeric_field_absent_is_none_and_not_malformed():
    assert _coerce_numeric_field(None) == (None, False)


def test_coerce_numeric_field_rejects_bool():
    """bool is an int subclass in Python -- must not silently become 1.0/0.0."""
    assert _coerce_numeric_field(True) == (None, True)
    assert _coerce_numeric_field(False) == (None, True)


def test_coerce_numeric_field_rejects_arbitrary_text():
    assert _coerce_numeric_field("N/A") == (None, True)
    assert _coerce_numeric_field("") == (None, True)
    assert _coerce_numeric_field("approximately 92.9 percent") == (None, True)


def test_coerce_numeric_field_rejects_list_and_dict():
    assert _coerce_numeric_field([92.9]) == (None, True)
    assert _coerce_numeric_field({"value": 92.9}) == (None, True)


def test_parse_claims_clean_numeric_string_value_coerces_to_float():
    text = (
        'Prose.\n\n```json\n[{"claim_id": 0, "entity": "Bihar", "metric": "Service Level", '
        '"value": "92.9", "citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert claims[0].value == 92.9
    assert isinstance(claims[0].value, float)


def test_parse_claims_malformed_value_becomes_nan_not_none():
    """value must stay 'present' (nan) so downstream grounding checks
    still run and correctly flag it, rather than silently being skipped
    the way a genuinely-absent field would be."""
    text = (
        'Prose.\n\n```json\n[{"claim_id": 0, "entity": "Bihar", "metric": "Service Level", '
        '"value": "N/A", "comparison_value": 94.3, "citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert len(claims) == 1
    assert claims[0].value is not None
    assert claims[0].value != claims[0].value  # the standard "is this nan" check
    assert claims[0].comparison_value == 94.3


def test_parse_claims_malformed_comparison_value_becomes_nan():
    text = (
        'Prose.\n\n```json\n[{"claim_id": 0, "entity": "Bihar", "metric": "Service Level", '
        '"value": 92.9, "comparison_value": "", "citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert claims[0].value == 92.9
    assert claims[0].comparison_value is not None
    assert claims[0].comparison_value != claims[0].comparison_value  # nan


def test_parse_claims_malformed_delta_becomes_none_not_nan():
    text = (
        'Prose.\n\n```json\n[{"claim_id": 0, "entity": "Bihar", "metric": "Service Level", '
        '"value": 92.9, "comparison_value": 94.3, "delta": "a lot", "citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert claims[0].value == 92.9
    assert claims[0].comparison_value == 94.3
    assert claims[0].delta is None


def test_parse_claims_absent_numeric_fields_stay_none():
    """A qualitative claim with no numeric fields at all must be
    unaffected by the coercion -- still plain None, not nan."""
    text = (
        'Prose.\n\n```json\n[{"claim_id": 0, "claim_type": "qualitative", "entity": "Bihar", '
        '"citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(text)
    assert claims[0].value is None
    assert claims[0].comparison_value is None
    assert claims[0].delta is None


def test_end_to_end_malformed_value_does_not_crash_check_grounding_and_fails():
    """The exact live failure mode, reproduced through the real parse ->
    grounding pipeline: a claim with a non-numeric value must not crash
    check_grounding() -- it must come back as a normal, visible fail."""
    answer_text = "Bihar's Service Level was N/A in November 2025 [Data: Sources (3)]."
    full_response = (
        answer_text + '\n\n```json\n[{"claim_id": 0, "entity": "Bihar", "metric": "Service Level", '
        '"period": "November 2025", "value": "N/A", "citations": ["Sources (3)"]}]\n```'
    )
    claims = _parse_claims(full_response)
    context_records = {
        "sources": pd.DataFrame([{"id": "3", "text": "Service Level for Bihar in November 2025 was 92.9%."}]),
        "entities": pd.DataFrame(columns=["id", "entity", "description"]),
        "reports": pd.DataFrame(columns=["id", "title", "content"]),
        "relationships": pd.DataFrame(columns=["id", "source", "target", "description"]),
        "claims": pd.DataFrame(columns=["id", "description"]),
    }

    result = check_grounding(answer_text, context_records, claims, metric_comparisons=None)  # must not raise

    assert result.status == "fail"
    assert any(issue.issue_type == "unsupported_numeric" for issue in result.flagged_issues)


# ---------------------------------------------------------------------------
# Phase 8 (P1 x3): resolution/absence reasoning guidance, cross-entity
# comparison citation guidance, and the distributor-bullet-list
# reinforcement to _VALUE_ATTRIBUTION_GUIDANCE.
#
# Test-to-requirement map:
#   3: test_generate_answer_prompt_includes_resolution_reasoning_guidance
#   4: test_regenerate_answer_prompt_includes_resolution_reasoning_guidance
#   5: test_claim_extraction_suffix_includes_cross_entity_comparison_guidance
#   6: test_cross_entity_comparison_guidance_does_not_prohibit_comparisons
#   7: test_value_attribution_guidance_covers_distributor_bullet_lists
# ---------------------------------------------------------------------------


def test_generate_answer_prompt_includes_resolution_reasoning_guidance():
    """3: wiring check -- the resolution/absence reasoning guidance must
    actually reach the model in generate_answer()'s prompt."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "Did the deviation persist into July?", context_result))

    sent_system_message = _normalize_whitespace(engine.model.calls[0][0]["content"])
    assert "Resolution And Absence Reasoning" in sent_system_message
    assert "No distributor deviation this month for: Service Level" in sent_system_message
    assert "did NOT persist" in sent_system_message


def test_regenerate_answer_prompt_includes_resolution_reasoning_guidance():
    """4: the same guidance must also reach the model on the retry path."""
    draft = AnswerResult(text="There is no specific information available.", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_qualifier",
                sentence="There is no specific information available.",
                term="no information",
                detail="hedge instead of using the retrieved evidence",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with a 'No distributor deviation' sentence>")

    asyncio.run(
        regenerate_answer(engine, "Did the deviation persist into July?", context_result, draft, grounding)
    )

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "Resolution And Absence Reasoning" in sent_system_message
    assert "did NOT persist" in sent_system_message


def test_generate_answer_prompt_includes_entity_existence_guidance():
    """Phase 8d wiring check: the anti-fabrication guidance (a roster
    listing is not evidence of a deviation) must reach the model in
    generate_answer()'s prompt."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "Which distributors showed a deviation?", context_result))

    sent_system_message = _normalize_whitespace(engine.model.calls[0][0]["content"])
    assert "Entity Existence" in sent_system_message
    assert "is NOT evidence of any deviation" in sent_system_message


def test_regenerate_answer_prompt_includes_entity_existence_guidance():
    """The same anti-fabrication guidance must also reach the model on the
    retry path, not just the first draft."""
    draft = AnswerResult(text="Several distributors showed a deviation.", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_numeric",
                sentence="Garcia-Perry Distributors showed a deviation of 28.6%.",
                term="28.6",
                detail="entity and value never stated together",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context with a zone roster listing>")

    asyncio.run(
        regenerate_answer(engine, "Which distributors showed a deviation?", context_result, draft, grounding)
    )

    sent_system_message = _normalize_whitespace(engine.model.calls[0][0]["content"])
    assert "Entity Existence" in sent_system_message
    assert "is NOT evidence of any deviation" in sent_system_message


def test_claim_extraction_suffix_includes_cross_entity_comparison_guidance():
    """5: the claim-extraction instructions must explicitly cover
    comparisons between two DIFFERENT entities, requiring citations for
    both sides -- using the exact Maharashtra/Sikkim worked example."""
    from src.inference.answer import _CLAIM_EXTRACTION_SUFFIX

    normalized = _normalize_whitespace(_CLAIM_EXTRACTION_SUFFIX)
    assert "BETWEEN TWO DIFFERENT entities/states/distributors/" in normalized
    assert "citations must support BOTH sides of the comparison" in normalized
    assert "Maharashtra's Ferrero Out-of-Stock Rate (6.5%)" in normalized
    assert "Sikkim's Ferrero Out-of-Stock Rate (6.6%)" in normalized


def test_cross_entity_comparison_guidance_does_not_prohibit_comparisons():
    """6: the guidance must not read as "don't compare different entities"
    -- it must explicitly say the comparison itself is fine once both
    sides are supported, and offer the two-separate-claims fallback."""
    from src.inference.answer import _CLAIM_EXTRACTION_SUFFIX

    normalized = _normalize_whitespace(_CLAIM_EXTRACTION_SUFFIX)
    assert "the prose can still state the comparison directly once both underlying facts are individually supported" in normalized
    assert "report the two values as separate factual_numeric claims" in normalized


def test_generate_answer_appends_atomic_facts_block_to_context_data():
    """Wiring check: generate_answer() must append the deterministic Atomic
    Distributor Facts section (built from context_records["sources"]) onto
    context_chunks in the prompt actually sent to the model, while leaving
    the original context_chunks text fully intact (additive, not a
    replacement) -- see fact_structuring.py."""
    text = (
        "Distributor Okafor Distributors showed a significant deviation on "
        "Dropsize in May 2027: 200.00 vs. the state average of 180.00, a gap "
        "of 11.1 percent."
    )
    context_records = {"sources": pd.DataFrame([{"id": "3", "text": text}])}
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>", context_records=context_records)

    asyncio.run(generate_answer(engine, "Which distributors deviated?", context_result))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "<fake context>" in sent_system_message  # original context preserved
    assert "-----Atomic Distributor Facts-----" in sent_system_message
    assert "Distributor: Okafor Distributors" in sent_system_message
    assert "Source: Sources (3)" in sent_system_message


def test_generate_answer_with_no_context_records_omits_facts_block_and_still_works():
    """When context_result has no context_records at all (e.g. a bare
    ContextBuilderResult stand-in), generate_answer() must not crash and
    must fall back to sending context_chunks unchanged -- matches every
    other pre-existing test in this file that constructs context_result
    without context_records."""
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>")

    asyncio.run(generate_answer(engine, "Which distributors deviated?", context_result))

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "<fake context>" in sent_system_message
    # Checks for the actual FACTS BLOCK section header (with its
    # distinctive dashes), not the bare phrase "Atomic Distributor Facts"
    # -- that bare phrase now also appears, unconditionally, inside
    # _ATOMIC_FACTS_CITATION_GUIDANCE's own instructional text (as an
    # example of a section name never to cite), so it's no longer a valid
    # proxy for "no facts block was appended".
    assert "-----Atomic Distributor Facts-----" not in sent_system_message


def test_regenerate_answer_appends_same_atomic_facts_block_as_generate_answer():
    """Requirement: the retry must receive EXACTLY the same structured
    evidence as the first answer attempt -- same context_records in,
    same Atomic Distributor Facts section out."""
    text = (
        "Distributor Ibarra Distributors showed a significant deviation on "
        "Out-of-Stock Rate for the Sundin category in May 2027: 22.0% vs. the "
        "state average of 9.0%, a gap of 13.0 percentage points."
    )
    context_records = {"sources": pd.DataFrame([{"id": "8", "text": text}])}
    draft = AnswerResult(text="draft naming the wrong distributor", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="unsupported_entity",
                sentence="draft naming the wrong distributor",
                term="wrong distributor",
                detail="entity/value never stated together",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")
    context_result = SimpleNamespace(context_chunks="<fake context>", context_records=context_records)

    asyncio.run(
        regenerate_answer(engine, "Which distributors deviated?", context_result, draft, grounding)
    )

    sent_system_message = engine.model.calls[0][0]["content"]
    assert "<fake context>" in sent_system_message
    assert "-----Atomic Distributor Facts-----" in sent_system_message
    assert "Distributor: Ibarra Distributors" in sent_system_message
    assert "Source: Sources (8)" in sent_system_message


def test_generate_answer_and_regenerate_answer_produce_identical_facts_block():
    """Both entry points must derive the exact same Atomic Distributor Facts
    text from the same context_records -- not two independently-built
    (and potentially divergent) representations."""
    from src.inference.fact_structuring import build_atomic_facts_block

    text = (
        "Distributor Larkspur Distributors showed a significant deviation on "
        "Dropsize in June 2028: 199.00 vs. the state average of 180.00, a gap "
        "of 10.6 percent."
    )
    context_records = {"sources": pd.DataFrame([{"id": "15", "text": text}])}
    expected_block = build_atomic_facts_block(context_records)
    assert expected_block  # sanity: the fixture actually produces a fact

    context_result = SimpleNamespace(context_chunks="<fake context>", context_records=context_records)

    engine_a = make_fake_engine("Answer A.\n\n```json\n[]\n```")
    asyncio.run(generate_answer(engine_a, "Which distributors deviated?", context_result))

    draft = AnswerResult(text="draft", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(status="fail", method="deterministic", flagged_issues=[])
    engine_b = make_fake_engine("Answer B.\n\n```json\n[]\n```")
    asyncio.run(regenerate_answer(engine_b, "Which distributors deviated?", context_result, draft, grounding))

    sent_a = engine_a.model.calls[0][0]["content"]
    sent_b = engine_b.model.calls[0][0]["content"]
    assert expected_block in sent_a
    assert expected_block in sent_b


def test_value_attribution_guidance_covers_distributor_bullet_lists():
    """7: _VALUE_ATTRIBUTION_GUIDANCE must explicitly extend to adjacent
    distributor-level deviation bullets, not just category/metric
    summaries -- using a generic Distributor A/B example, not a
    Bihar-specific hardcoded rule."""
    from src.inference.answer import _VALUE_ATTRIBUTION_GUIDANCE

    normalized = _normalize_whitespace(_VALUE_ATTRIBUTION_GUIDANCE)
    assert "DISTRIBUTOR-level deviation statements" in normalized
    assert "Distributor A showed a deviation" in normalized
    assert "Distributor B showed a deviation" in normalized
    assert "Bihar" not in normalized
    # Still generic overall -- the earlier requirement (no Ferrero/Candy
    # special-casing) must continue to hold after this addition.
    assert normalized.count("Ferrero") <= 1
    assert normalized.count("Candy") <= 1


# ---------------------------------------------------------------------------
# GPIL Knowledge Layer (2026-08-23): "What is GPI and IPM?" live investigation.
#
# Root cause: GraphRAG's OWN index contains real, retrievable Entity nodes
# titled "GPI"/"IPM" whose descriptions were auto-written by an LLM at INDEX
# TIME from sparse corpus context -- IPM's drifts toward "agricultural
# practices" language, close enough to reinforce (not contradict) the
# model's strong pretrained "Integrated Pest Management" prior. The
# glossary block used to be APPENDED after context_chunks (GraphRAG's own
# retrieved text, including that Entity description), so the model read the
# vague/misleading indexed description as "the real evidence" before ever
# reaching the correct glossary entry, and the guidance only said to prefer
# the Knowledge Layer over "generic/world knowledge" -- never over other
# RETRIEVED evidence for the same term. These tests reproduce the failure
# shape and prove the fix: the glossary block is now prepended before
# context_chunks, and the guidance explicitly overrides conflicting
# Entity/Report descriptions too.
# ---------------------------------------------------------------------------


def make_glossary_context_result(question: str, extra_chunks: str = "") -> SimpleNamespace:
    """A context_result whose context_records["sources"] already has the
    glossary rows pipeline.py would have merged in for `question`, plus
    context_chunks standing in for GraphRAG's own retrieval -- by default a
    fabricated Entity description of the exact shape the live investigation
    found (vague, agriculture-adjacent for IPM, no explicit wrong acronym
    but nothing correcting the model's prior either)."""
    from src.inference.knowledge_layer import build_glossary_source_rows

    rows = build_glossary_source_rows(question)
    context_records = {"sources": pd.DataFrame(rows)}
    chunks = extra_chunks or (
        "-----Entities-----\n"
        "id,entity,description\n"
        '5,GPI,"GPI has emerged as a significant product category across various regions..."\n'
        '6,IPM,"IPM is a product category... reflecting its relevance and implementation in '
        'Indian agriculture and market systems."\n'
    )
    return SimpleNamespace(context_chunks=chunks, context_records=context_records)


def test_generate_answer_places_glossary_block_before_graphrag_entity_description():
    """The glossary block must be the FIRST thing in context_data -- ahead
    of GraphRAG's own retrieved Entity descriptions, which is what the live
    "IPM" investigation found the model reading (and being misled by)
    first."""
    context_result = make_glossary_context_result("What is GPI and IPM?")
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")

    asyncio.run(generate_answer(engine, "What is GPI and IPM?", context_result))

    sent = engine.model.calls[0][0]["content"]
    glossary_pos = sent.find("-----GPIL Knowledge Layer-----")
    entity_pos = sent.find("Indian agriculture and market systems")
    assert glossary_pos != -1
    assert entity_pos != -1
    assert glossary_pos < entity_pos


def test_glossary_guidance_explicitly_overrides_entity_and_report_descriptions():
    """The guidance text itself (not just the block's own preface) must
    name Entity/Report descriptions as something the Knowledge Layer
    overrides -- not only 'generic/world knowledge', which is what let the
    original bug slip through even with the guidance already in place."""
    from src.inference.answer import _GLOSSARY_CITATION_GUIDANCE

    normalized = _normalize_whitespace(_GLOSSARY_CITATION_GUIDANCE)
    assert "OVERRIDES" in normalized
    assert "Entity" in normalized and "Report" in normalized
    assert "agriculture" in normalized.lower()


def test_glossary_block_preface_states_it_is_authoritative_only_for_its_own_terms():
    """Requirement: the Knowledge Layer must be authoritative ONLY for
    terms it actually defines -- unrelated terms must be explicitly
    excluded from the override, in the same text the model reads."""
    from src.inference.knowledge_layer import build_glossary_block

    context_records = {"sources": pd.DataFrame([{"id": "glossary-gpi", "text": "Term: GPI / GPIL\nDefinition: x\nSource: Sources (glossary-gpi)"}])}
    block = build_glossary_block(context_records)
    assert "authoritative ONLY for the exact terms" in block


def test_regenerate_answer_also_places_glossary_block_before_entity_description():
    """Requirement 9: the retry must not lose the glossary context or its
    positioning -- same ordering guarantee as generate_answer()."""
    context_result = make_glossary_context_result("What is GPI and IPM?")
    draft = AnswerResult(text="GPI stands for General Product Inventory.", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="wrong_glossary_expansion",
                sentence="GPI stands for General Product Inventory.",
                term="GPI / GPIL",
                detail='The answer uses "general product inventory", a generic meaning explicitly known to be WRONG for "GPI / GPIL" in this system.',
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")

    asyncio.run(regenerate_answer(engine, "What is GPI and IPM?", context_result, draft, grounding))

    sent = engine.model.calls[0][0]["content"]
    glossary_pos = sent.find("-----GPIL Knowledge Layer-----")
    entity_pos = sent.find("Indian agriculture and market systems")
    assert glossary_pos != -1
    assert entity_pos != -1
    assert glossary_pos < entity_pos
    # Requirement 9: the same override guidance must also reach the retry prompt.
    assert "OVERRIDES" in sent
    assert "Entity" in sent


def test_regenerate_answer_gives_specific_fix_instruction_for_wrong_glossary_expansion():
    """Before this fix, "wrong_glossary_expansion" (and
    "unsupported_definition") fell through to the generic "Fix this issue
    using only what the data tables support" -- too vague to reliably
    steer a model away from a confidently-held pretrained prior. The retry
    prompt must now name the Knowledge Layer section explicitly."""
    issue = GroundingIssue(
        issue_type="wrong_glossary_expansion",
        sentence="GPI stands for General Product Inventory.",
        term="GPI / GPIL",
        detail="known wrong",
    )
    formatted = _format_issue_for_retry(issue)
    assert "GPIL Knowledge Layer" in formatted
    assert "Fix this issue using only what the data tables support." not in formatted


def test_regenerate_answer_gives_specific_fix_instruction_for_unsupported_definition():
    issue = GroundingIssue(
        issue_type="unsupported_definition",
        sentence="IPM stands for Integrated Pest Management.",
        term=None,
        detail="not backed by cited evidence",
    )
    formatted = _format_issue_for_retry(issue)
    assert "GPIL Knowledge Layer" in formatted


def test_end_to_end_generic_draft_rejected_then_regenerated_draft_passes():
    """Reproduces the exact reported failure shape end to end within this
    file's real prompt-building + response-parsing code (mocked completion
    call only): a first draft using the generic GPI/IPM meanings must fail
    check_grounding(); a regenerated draft using the GPIL-specific meanings,
    built from the SAME glossary-augmented context, must pass."""
    context_result = make_glossary_context_result("What is GPI and IPM?")

    bad_text = (
        "GPI stands for General Product Inventory [Data: Sources (glossary-gpi)]. "
        "IPM stands for Integrated Pest Management [Data: Sources (glossary-ipm)]."
    )
    engine_bad = make_fake_engine(bad_text + "\n\n```json\n[]\n```")
    bad_answer = asyncio.run(generate_answer(engine_bad, "What is GPI and IPM?", context_result))
    bad_grounding = check_grounding(bad_answer.text, context_result.context_records, bad_answer.claims, question="What is GPI and IPM?")
    assert bad_grounding.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in bad_grounding.flagged_issues)

    good_text = (
        "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]. "
        "IPM stands for the Marlboro product category GPIL sells under license [Data: Sources (glossary-ipm)]."
    )
    engine_good = make_fake_engine(good_text + "\n\n```json\n[]\n```")
    good_answer = asyncio.run(
        regenerate_answer(engine_good, "What is GPI and IPM?", context_result, bad_answer, bad_grounding)
    )
    good_grounding = check_grounding(good_answer.text, context_result.context_records, good_answer.claims, question="What is GPI and IPM?")
    assert good_grounding.status == "pass"

    # The retry prompt must have carried the SAME glossary evidence as the
    # first attempt -- not a stripped-down or re-derived version of it.
    retry_sent = engine_good.model.calls[0][0]["content"]
    assert "-----GPIL Knowledge Layer-----" in retry_sent
    assert "Term: GPI / GPIL" in retry_sent
    assert "Term: IPM" in retry_sent


def test_unrelated_term_is_unaffected_by_glossary_override_guidance():
    """Requirement 7: the override must be scoped to terms the Knowledge
    Layer actually defines -- a question naming no known GPIL term must
    produce no glossary block content (the static citation-guidance
    paragraph always mentions the section name/format regardless, the same
    way _ATOMIC_FACTS_CITATION_GUIDANCE always mentions "Atomic State KPI
    Facts" whether or not that section is present -- so the real signal is
    the absence of any rendered Term:/Definition: entry, not the header
    string)."""
    context_result = make_glossary_context_result("What was Bihar's Productivity in November 2025?")
    context_result.context_records = {"sources": pd.DataFrame(columns=["id", "text"])}
    engine = make_fake_engine("Bihar's Productivity in November 2025 was 89.0% [Data: Sources (0)].\n\n```json\n[]\n```")

    asyncio.run(generate_answer(engine, "What was Bihar's Productivity in November 2025?", context_result))

    sent = engine.model.calls[0][0]["content"]
    assert "Term: GPI" not in sent
    assert "Term: IPM" not in sent
    assert "Godfrey Phillips" not in sent


# ---------------------------------------------------------------------------
# GPIL Knowledge Layer follow-up (2026-08-23): "What is IPM?" asked ALONE
# still failed live even after the GPI+IPM multi-term fix above.
#
# Root cause: data/graphrag_index/pilot_run/output/entities.parquet's real
# "IPM" Entity node has a long, multi-paragraph, auto-generated description
# that repeats agriculture-adjacent phrasing across seven separate states/
# periods -- far longer and more repetitive than GPI's much shorter, vaguer
# one. A single-term question concentrates GraphRAG's whole retrieval
# budget on that one entity's neighborhood, surfacing this long narrative
# prominently; a two-term question splits that budget (diluting the
# effect), which is consistent with "What is GPI and IPM?" not failing the
# same way. The fix: build_glossary_reminder_block() (knowledge_layer.py)
# re-renders the SAME matched definitions immediately AFTER context_chunks,
# bookending the competing narrative on both sides -- see
# _context_data_with_facts()'s docstring.
# ---------------------------------------------------------------------------

# Verbatim (truncated for test-file size) from the real indexed "IPM"
# Entity description that reproduced the live failure -- copied here, not
# regenerated, so this test is anchored to the ACTUAL observed competing
# text, not a stand-in guess at what it might look like.
_REAL_INDEXED_IPM_ENTITY_DESCRIPTION = (
    "IPM is a product category that has been reported across various states in India over several "
    "years. The emergence and reporting of IPM as a product category spans different regions and "
    "times, reflecting its relevance and implementation in Indian agriculture and market systems.\n\n"
    "In August 2024, IPM was noted as a significant product category in Maharashtra and Rajasthan. "
    "This suggests a strategic adoption during this period in these states, potentially aligning with "
    "specific agricultural cycles or market demands prevalent there.\n\n"
    "Shortly thereafter, in September 2024, Kerala also reported IPM as a product category. This "
    "indicates that IPM was gaining traction in the southern part of India, possibly due to its "
    "adaptability to the regional agricultural practices or the market's growing interest in "
    "sustainable and innovative agricultural solutions.\n\n"
    "Continuing this trend, several months later, in March 2025, Karnataka recognized IPM as an "
    "important product category. Karnataka, known for its diverse agricultural matrix and "
    "forward-thinking approaches, brought IPM into focus.\n\n"
    "Overall, IPM's recognition across multiple Indian states from 2024 to 2026 highlights its "
    "growing importance and implementation in agricultural and market strategies, demonstrating "
    "IPM's versatility and its potential role in supporting sustainable agriculture and promoting "
    "enhanced agricultural productivity across India."
)


def make_single_term_ipm_context_result() -> SimpleNamespace:
    """Reproduces the exact live failure shape: a single-term "What is
    IPM?" question, with context_chunks standing in for GraphRAG's own
    retrieval concentrated entirely on the real, long, agriculture-flavored
    "IPM" Entity description -- no GPI entity competing for retrieval
    budget, matching the observed asymmetry between the two live queries."""
    from src.inference.knowledge_layer import build_glossary_source_rows

    rows = build_glossary_source_rows("What is ipm?")
    context_records = {"sources": pd.DataFrame(rows)}
    chunks = (
        "-----Entities-----\n"
        "id,entity,type,description\n"
        f'6,IPM,CATEGORY,"{_REAL_INDEXED_IPM_ENTITY_DESCRIPTION}"\n'
    )
    return SimpleNamespace(context_chunks=chunks, context_records=context_records)


def test_generate_answer_bookends_the_real_long_ipm_entity_description():
    """The glossary block must appear BOTH before AND after the long
    competing IPM narrative -- proving the bookend fix actually reaches the
    real prompt for the exact single-term query that failed live."""
    context_result = make_single_term_ipm_context_result()
    engine = make_fake_engine("Some answer.\n\n```json\n[]\n```")

    asyncio.run(generate_answer(engine, "What is ipm?", context_result))

    sent = engine.model.calls[0][0]["content"]
    opening_pos = sent.find("-----GPIL Knowledge Layer-----")
    entity_pos = sent.find("agricultural cycles")
    reminder_pos = sent.find("GPIL Knowledge Layer (Reminder")
    assert opening_pos != -1
    assert entity_pos != -1
    assert reminder_pos != -1
    assert opening_pos < entity_pos < reminder_pos
    assert "Term: IPM" in sent[:opening_pos + 2000]  # the opening block itself carries the term
    assert sent[reminder_pos:].count("Term: IPM") == 1  # the reminder repeats it once more


def test_single_term_gpi_definition_end_to_end():
    context_result = make_single_term_ipm_context_result()  # entity content irrelevant to GPI's own check
    from src.inference.knowledge_layer import build_glossary_source_rows

    context_result.context_records = {"sources": pd.DataFrame(build_glossary_source_rows("What is gpi?"))}
    good_text = "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]."
    engine = make_fake_engine(good_text + "\n\n```json\n[]\n```")

    answer = asyncio.run(generate_answer(engine, "What is gpi?", context_result))
    grounding = check_grounding(answer.text, context_result.context_records, answer.claims, question="What is gpi?")
    assert grounding.status == "pass"


def test_single_term_ipm_bad_generic_answer_is_rejected():
    context_result = make_single_term_ipm_context_result()
    bad_text = "IPM stands for Integrated Pest Management [Data: Sources (glossary-ipm)]."
    engine = make_fake_engine(bad_text + "\n\n```json\n[]\n```")

    answer = asyncio.run(generate_answer(engine, "What is ipm?", context_result))
    grounding = check_grounding(answer.text, context_result.context_records, answer.claims, question="What is ipm?")
    assert grounding.status == "fail"
    assert any(i.issue_type == "wrong_glossary_expansion" for i in grounding.flagged_issues)


def test_single_term_ipm_correct_gpil_answer_passes():
    context_result = make_single_term_ipm_context_result()
    good_text = "IPM stands for the Marlboro product category GPIL sells under license [Data: Sources (glossary-ipm)]."
    engine = make_fake_engine(good_text + "\n\n```json\n[]\n```")

    answer = asyncio.run(generate_answer(engine, "What is ipm?", context_result))
    grounding = check_grounding(answer.text, context_result.context_records, answer.claims, question="What is ipm?")
    assert grounding.status == "pass"


def test_multi_term_gpi_and_ipm_definition_still_passes_after_bookend_fix():
    """Regression: the bookend fix must not break the multi-term case that
    already worked."""
    from src.inference.knowledge_layer import build_glossary_source_rows

    rows = build_glossary_source_rows("What are GPI and IPM?")
    context_result = SimpleNamespace(
        context_chunks="-----Entities-----\nid,entity,description\n5,GPI,\"generic\"\n6,IPM,\"generic\"\n",
        context_records={"sources": pd.DataFrame(rows)},
    )
    good_text = (
        "GPI stands for GPIL's own cigarette brand portfolio [Data: Sources (glossary-gpi)]. "
        "IPM stands for the Marlboro product category GPIL sells under license [Data: Sources (glossary-ipm)]."
    )
    engine = make_fake_engine(good_text + "\n\n```json\n[]\n```")

    answer = asyncio.run(generate_answer(engine, "What are GPI and IPM?", context_result))
    grounding = check_grounding(answer.text, context_result.context_records, answer.claims, question="What are GPI and IPM?")
    assert grounding.status == "pass"


def test_regenerate_answer_bookends_the_real_long_ipm_entity_description_too():
    """Requirement: regeneration must preserve the SAME Knowledge Layer
    context and bookending as the first attempt for the single-term case,
    not just the multi-term one."""
    context_result = make_single_term_ipm_context_result()
    draft = AnswerResult(text="IPM stands for Integrated Pest Management.", claims=[], llm_calls=1)
    grounding = GroundingCheckResult(
        status="fail",
        method="deterministic",
        flagged_issues=[
            GroundingIssue(
                issue_type="wrong_glossary_expansion",
                sentence="IPM stands for Integrated Pest Management.",
                term="IPM",
                detail="known wrong",
            )
        ],
    )
    engine = make_fake_engine("Revised answer.\n\n```json\n[]\n```")

    asyncio.run(regenerate_answer(engine, "What is ipm?", context_result, draft, grounding))

    sent = engine.model.calls[0][0]["content"]
    opening_pos = sent.find("-----GPIL Knowledge Layer-----")
    entity_pos = sent.find("agricultural cycles")
    reminder_pos = sent.find("GPIL Knowledge Layer (Reminder")
    assert opening_pos != -1 and entity_pos != -1 and reminder_pos != -1
    assert opening_pos < entity_pos < reminder_pos


def test_unrelated_term_gets_no_reminder_block_either():
    context_result = make_glossary_context_result("What was Bihar's Productivity in November 2025?")
    context_result.context_records = {"sources": pd.DataFrame(columns=["id", "text"])}
    engine = make_fake_engine("Bihar's Productivity in November 2025 was 89.0% [Data: Sources (0)].\n\n```json\n[]\n```")

    asyncio.run(generate_answer(engine, "What was Bihar's Productivity in November 2025?", context_result))

    sent = engine.model.calls[0][0]["content"]
    assert "GPIL Knowledge Layer (Reminder" not in sent
