"""
inference: the reasoning layer that turns evidence into grounded answers (Phase 7).

Modules, in pipeline order (see pipeline.py for how they're wired together):
- context.py:         GraphRAG retrieval wrapper -- loads the index, builds
                       local-search context for a question, stops before
                       generation.
- premise_check.py:   Stage 0 -- deterministic check of whether the
                       retrieved evidence supports a directional claim
                       ("decline"/"improve") embedded in the question,
                       BEFORE any answer-generation LLM call.
- answer.py:          Stage 1 -- the one stage that calls the completion
                       model. Drafts the prose answer plus a structured
                       claims block in the same call.
- grounding_check.py: Stage 2 -- deterministic check of the drafted
                       answer's claims against the same retrieved evidence
                       (unsupported qualifiers, direction contradictions).
- pipeline.py:         orchestrates the above + a CLI entry point
                       (`python -m src.inference.pipeline "<question>" ...`).
- schemas.py:          shared dataclasses for every stage's result.

Current status: premise_check and grounding_check are fully deterministic
(no LLM call, no LLM-escalation path yet -- see each module's docstring for
what that means for ambiguous cases). LLM-escalation for both, plus a
bounded regeneration retry when grounding fails, are Phase 7b follow-ups,
not implemented yet.
"""
