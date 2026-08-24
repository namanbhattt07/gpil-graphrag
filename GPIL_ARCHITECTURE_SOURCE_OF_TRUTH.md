# GPIL GraphRAG S&D Chat System — Architecture Source of Truth

Verified against the actual repository at `/Users/namanbhatt/Downloads/GPIL/code` (branch `main`, plus uncommitted working-tree files). Every claim below is backed by a specific file/function. Nothing here is aspirational — items that exist only as docs/plans are called out explicitly in Section 9 ("Things NOT to show").

---

## 1. SYSTEM FLOW (end to end, in build order)

```
generate_synthetic_data.py  →  compute_kpis.py  →  build_documents.py  →  GraphRAG indexing (external `graphrag index` CLI, config-driven)
    →  canonicalize_index.py (optional cleanup)  →  validate_index.py (optional check)
    →  [QUERY TIME] context.py → premise_check.py → answer.py → grounding_check.py → pipeline.py → ui_adapter.py / cli.py
```

Step by step:

1. **Data starts** as nothing (no real GPIL data available) — `src/data_gen/generate_synthetic_data.py::generate_all()` invents a full synthetic S&D dataset with Faker + numpy from a fixed random seed (`config.settings.get_settings().random_seed`, default 42), writing 6 CSVs to `data/`: `geography.csv`, `outlets.csv`, `products.csv`, `visits.csv`, `orders.csv`, `inventory_snapshots.csv`.
2. **KPIs are calculated** by `src/kpis/compute_kpis.py::compute_all()`, which reads those 6 CSVs and rolls raw rows up into KPI tables at multiple grains (State×month, State×month×Category, State×month×Channel, WD×month, WD×month×Category), written as 5 CSVs.
3. **Documents are created** by `src/graph/build_documents.py::build_all_documents()`, which reads the KPI CSVs + `geography.csv` + `products.csv` and renders one narrative `.txt` file per (State, month) — 672 files (28 states × 24 months) — into `data/graphrag_input/`, plus a manifest CSV.
4. **GraphRAG indexing** happens by running the `graphrag index` CLI (Microsoft GraphRAG library, not custom code) against `data/graphrag_index/settings.yaml`, which points `input_storage.base_dir` at `../graphrag_input` (the 672 documents). This chunks documents, runs LLM entity/relationship extraction with a **custom domain-specific prompt** (`prompts/extract_graph.txt`), summarizes descriptions, embeds entities, detects communities (Leiden), and writes community reports — all native GraphRAG workflow steps, config-driven.
5. **Knowledge graph** = the parquet tables GraphRAG writes to `output/`: `entities.parquet`, `relationships.parquet`, `communities.parquet`, `community_reports.parquet`, `text_units.parquet`, `documents.parquet`.
6. Optionally, `src/graph/validate_index.py::validate()` checks this output for data-quality problems (misspelled Category/Franchise entities, missing State entities, referential integrity), and `src/graph/canonicalize_index.py::canonicalize()` deterministically fixes the fixable subset (merges duplicate entities, remaps relationships/text-units/communities, re-embeds only changed entities) — producing a cleaned copy (e.g. `output_fixed_clean/`).
7. **A user query enters the system** via `src/inference/ui_adapter.py::run_diagnostic_query(question)` (the one public boundary function) or the terminal chatbot `src/cli.py`.
8. **Retrieval**: `src/inference/context.py::build_query_context()` loads the cleaned index's parquet tables, builds GraphRAG's own `LocalSearch` engine (`graphrag.query.factory.get_local_search_engine`), and calls `engine.context_builder.build_context(query=...)` — this is GraphRAG's native local-search retrieval: it embeds the query, vector-matches it against entity description embeddings, and assembles ranked Entities/Relationships/Reports/Sources/Claims into a `ContextBuilderResult`. **This is the only place an embedding API call happens.**
9. **Premise verification (Stage 0)**: `src/inference/premise_check.py::check_premise()` — before any answer-generation LLM call, deterministically checks whether the question's directional claim ("decline"/"improve") is actually supported by the retrieved evidence, by regexing exact metric values back out of the retrieved Sources text (which is template-generated, not free prose) and comparing the correct calendar baseline period to the target period.
10. **Query is classified** only in this narrow premise-check sense — there is no separate "query classifier" module; `check_premise()` itself extracts direction words, named state, named period(s), named metric, or detects an unanswerable "superlative ranking" question shape (`extract_superlative_ranking_claim`).
11. If the premise fails (contradicted/unsupported/insufficient_data/undefined_metric), **the pipeline short-circuits — zero completion-model calls are made** and a deterministic hedge message is returned (`pipeline.py::_render_hedge()`).
12. **Facts are structured**: `src/inference/fact_structuring.py::build_atomic_facts_block()` deterministically regexes the SAME retrieved Sources text into an explicit "Atomic Facts" prompt section (Distributor / State KPI / Category KPI facts), which `answer.py` appends to GraphRAG's own context text before sending it to the model — this is a custom addition, not part of GraphRAG.
13. **LLM is called** in exactly one place with retrieval-time context: `src/inference/answer.py::generate_answer()`, which reuses the already-built `LocalSearch` engine's model client (`engine.model.completion_async`) and system prompt, appends heavy custom prompt-engineering guidance blocks (specificity, value-attribution, entity-existence, citation-selection, atomic-facts-citation rules) plus a request for a trailing structured JSON "claims" block, and streams one completion.
14. **Grounding validation**: `src/inference/grounding_check.py::check_grounding()` — fully deterministic (no LLM), validates every self-reported claim against ONLY the specific evidence rows its own citations point to (claim-level check), then runs prose-level fallback scanners (qualitative language, causal language, recommendation language, definition language, entity-numeric co-occurrence, ranking verification, thematic completeness, direction-consistency).
15. If grounding fails, `src/inference/pipeline.py::run_pipeline()` calls `answer.py::regenerate_answer()` **exactly once** (bounded retry, never a loop) with the specific flagged issues, re-checks grounding, and commits to either the revised answer or a fail-closed "insufficient evidence" message.
16. **Final answer** is a `PipelineResult` (`src/inference/schemas.py`) with `final_text`, `final_decision` (`pass_through` / `hedged` / `insufficient_evidence` / `regenerated`), and full stage-by-stage detail — returned to `ui_adapter.py::run_diagnostic_query()`, printed by `src/cli.py`'s terminal loop.

---

## 2. INDEX-TIME ARCHITECTURE

**Input documents**: 672 plain-text files in `data/graphrag_input/` (one per State×month), each written by a fixed Python f-string template in `build_documents.py::render_document()` — never free LLM prose. Each document opens with a machine-parseable `State: X\nPeriod: Y` header, then a zones sentence, a state-level KPI paragraph (6 metrics), a WD roster grouped by zone, distributor-level deviation sentences (or explicit no-deviation sentences), channel-level performance sentences, and one paragraph per product category (KPIs + franchise roster).

**Text units/chunks**: GraphRAG's native `chunking` workflow — `type: tokens`, `size: 1200`, `overlap: 100`, `o200k_base` encoding (`settings.yaml`). A single ~1200-word document may split into 2+ chunks; a chunk mid-document loses the document's opening `State:`/`Period:` header (this is a documented, real failure mode the code works around — see `fact_structuring.py`'s category-block extractor, which deliberately does NOT depend on the header).

**Entity extraction**: GraphRAG's native `extract_graph` workflow, but with a **fully custom prompt** (`data/graphrag_index/prompts/extract_graph.txt`) that replaces GraphRAG's generic default entity types (organization/person/geo/event) with 8 GPIL-specific types: **State, Period, Distributor, Category, Franchise, Observation, Metric, Document**. `max_gleanings: 2` (extra extraction passes for recall). `Observation` is the entity type that carries every actual KPI number (e.g. `"Maharashtra Productivity August 2024"`), constructed via 4 fixed naming templates.

**Relationship extraction**: same custom prompt defines exactly 8 allowed relationship types (never invented by the model): `HAS_OBSERVATION`, `SCOPED_TO_CATEGORY`, `OF_METRIC`, `OBSERVED_IN`, `SERVES`, `CONTAINS`, `EVIDENCED_BY`, `DEVIATES_FROM`. `DEVIATES_FROM` is called out in the prompt as "the highest business-value relationship in this schema" (supports "which distributor caused the decline" questions) and gets relationship_strength 10.

**Claims**: `extract_claims.enabled: false` in `settings.yaml` — claim extraction is explicitly OFF for indexing. (Note: `context_records["claims"]` is still read at query time as an empty/unused table — GraphRAG's query API always exposes this slot.)

**Community detection**: GraphRAG's native Leiden clustering (`cluster_graph.max_cluster_size: 10`) — entirely native GraphRAG code, no custom logic.

**Community reports**: GraphRAG's native `community_reports` workflow — LLM-summarized per community, `max_length: 2000`, custom prompts at `prompts/community_report_graph.txt` / `community_report_text.txt` (present but not verified to differ substantively from GraphRAG defaults in this review).

**Embeddings**: `text-embedding-3-small` (OpenAI), used for entity description embeddings, stored in a **LanceDB** vector store (`vector_store: {type: lancedb, db_uri: output/lancedb}`) — native GraphRAG.

**Storage/output artifacts**: parquet files under `output/` (`entities`, `relationships`, `communities`, `community_reports`, `text_units`, `documents`) + `output/lancedb/` (entity_description, community_full_content, text_unit_text vector tables) + `cache/` (JSON completion/embedding cache, keyed for reruns) + `logs/` (reporting).

**Models/providers**: completion = `gpt-4o` (production `settings.yaml`) or `gpt-4o-mini` (the mini-pilot config actually used live, see Section 9); embedding = `text-embedding-3-small`. Both via OpenAI's API directly (`api_base: https://api.openai.com/v1`), read from `.env`'s `OPENAI_API_KEY`.

**Custom vs. native — explicit split**:
| Custom (this project's code) | Native GraphRAG |
|---|---|
| Document text templates (`build_documents.py`) | Chunking |
| `extract_graph.txt` prompt (8 entity types, 8 relationship types) | Extraction *mechanism* (LLM call loop, gleaning) |
| `canonicalize_index.py`, `validate_index.py` (post-index cleanup/QA) | Community detection (Leiden) |
| — | Community report generation |
| — | Embedding generation & LanceDB storage |
| — | `LocalSearch` retrieval engine |

---

## 3. QUERY-TIME ARCHITECTURE (actual runtime sequence)

```
Question
  ↓  (ui_adapter.run_diagnostic_query / cli.py)
Engine load + Retrieval           — context.build_query_context() / _get_query_context()
  ↓
Premise Verification (Stage 0)    — premise_check.check_premise()
  ↓ (only if premise passes: "no_claim" or "supported")
Answer Generation (Stage 1)       — answer.generate_answer()  [Atomic Facts injected here]
  ↓
Grounding Check (Stage 2)         — grounding_check.check_grounding()
  ↓ (only if grounding fails)
ONE Regeneration Attempt          — answer.regenerate_answer()
  ↓
Grounding Check again (on retry)  — grounding_check.check_grounding()
  ↓
Final Answer                      — pipeline.run_pipeline() assembles PipelineResult
```

This differs from a naive "classify → retrieve → infer" mental model: **there is no separate query-classification step or reasoning/chain-of-thought stage** distinct from premise-check + single-shot generation. (An earlier design doc/memory described a 3-stage "structure → guarded CoT → grounding check" inference layer; the actual implementation collapsed structuring into premise-check + fact_structuring, and CoT into one prompted completion call — see Section 9.)

Per-arrow detail:

- **Question → Retrieval**: `src/inference/ui_adapter.py::run_diagnostic_query()` → `_get_query_context()` → (first call only) `src/inference/context.py::build_query_context()`. Input: question string, fixed `INDEX_ROOT`/`OUTPUT_DIR` constants (`data/graphrag_index/pilot_run_mini`, `output_fixed_clean`). Output: `QueryContext(engine, context_result)` where `context_result.context_records` is a dict of DataFrames (`entities`, `relationships`, `reports`, `sources`, `claims`). Why: separates GraphRAG's context-building from its answer-generation so the guard pipeline can inspect evidence before any prose is written — GraphRAG's own `graphrag.api.local_search()` does not expose this seam.
- **Retrieval → Premise Verification**: `src/inference/pipeline.py::run_pipeline()` calls `premise_check.check_premise(question, qctx.context_records)`. Input: raw question text + `context_records["sources"]`. Output: `PremiseCheckResult` with `status` ∈ {no_claim, supported, contradicted, unsupported, insufficient_data, undefined_metric}. Why: catches the specific live failure where the model was asked to explain a decline that never happened in the evidence (Bihar Oct 2025 case, no Sept 2025 doc indexed).
- **Premise Verification → Answer Generation**: gated in `run_pipeline()` — only proceeds if `status` is `no_claim` or `supported`; every other status short-circuits with zero completion calls.
- **Answer Generation**: `src/inference/answer.py::generate_answer(engine, question, context_result)`. Input: GraphRAG's own formatted `context_chunks` text + `fact_structuring.build_atomic_facts_block()`'s deterministic regex-extracted facts appended to it, plus ~6 stacked prompt-guidance blocks. Output: `AnswerResult(text, claims, llm_calls=1)` — prose with the trailing JSON claims block stripped out and parsed separately. Why: a single call produces both the human-readable answer AND a machine-checkable claims list, avoiding a second LLM call for grounding.
- **Answer Generation → Grounding Check**: `grounding_check.check_grounding(answer.text, context_records, answer.claims, premise.metric_comparisons, question)`. Input: drafted prose, retrieved evidence, self-reported claims, premise-check's own metric comparisons (for direction-consistency cross-check), the original question (for category/ranking scoping). Output: `GroundingCheckResult(status, flagged_issues)`. Why: verifies the model didn't invent numbers/entities/causation/qualifiers beyond what evidence supports.
- **Grounding Check → Regeneration (conditional)**: only on `status == "fail"`. `answer.regenerate_answer(engine, query, context_result, draft, grounding)` — same context, draft answer, and the specific flagged issues sent back, asking the model to fix ONLY those. Why: a live test found a fully correct, well-cited answer discarded entirely over one unsupported adjective — bounded retry salvages it instead of always failing closed.
- **Regeneration → Grounding Check (again) → Final Answer**: `run_pipeline()` re-runs `check_grounding()` on the retry once, then commits — never loops further. `final_decision` is one of `pass_through` / `hedged` / `insufficient_evidence` / `regenerated`.

---

## 4. KNOWLEDGE GRAPH

### Entities (exactly 8 types, per `extract_graph.txt`)
- **State** — one per document (28 total possible), e.g. "Maharashtra".
- **Period** — one per document, "Month YYYY", e.g. "August 2024".
- **Distributor** — named in a zone roster or a deviation sentence, e.g. "Boyd-White Distributors". Always the word "Distributor", never "WD".
- **Category** — GPI / IPM / Ferrero / Candy (4 fixed values).
- **Franchise** — e.g. "Marlboro", "TicTac", "Kinder_Joy", "GPI_Franchise_1..6", "Candy_Franchise_1/2".
- **Observation** — the entity type carrying every numeric KPI value, named via 4 templates: `{State} {Metric} {Period}`, `{State} {Category} {Metric} {Period}`, `{Distributor} {Metric} {Period}`, `{Distributor} {Category} {Metric} {Period}`.
- **Metric** — Productivity, Service Level, SKUs per Transaction, Dropsize, Inventory Turns, Inventory Days, Numeric Distribution, ACV, Out-of-Stock Rate, Range Billing.
- **Document** — one per source file, e.g. "Maharashtra_2024-08.txt".

Explicitly forbidden/skipped entity types (per prompt instruction): Zone, Channel — these are mentioned in document text but the extraction prompt tells the model NOT to create entities for them. `validate_index.py::FORBIDDEN_ENTITY_TYPES = {"ZONE", "CHANNEL"}` exists specifically to catch cases where the model creates them anyway (an observed real extraction error in the GPT-4o-mini pilot).

### Relationships (exactly 8 types)
`HAS_OBSERVATION` (State/Distributor → Observation), `SCOPED_TO_CATEGORY` (Observation → Category), `OF_METRIC` (Observation → Metric), `OBSERVED_IN` (Observation → Period), `SERVES` (Distributor → State), `CONTAINS` (Category → Franchise), `EVIDENCED_BY` (Observation → Document), `DEVIATES_FROM` (Distributor → the state-level Observation for the same metric/category/period — highest strength, 10/10).

`PRECEDES` (period-to-period sequencing) is explicitly named in the prompt as something to NOT extract — chronological ordering is computed in code (`premise_check.py`'s `_period_key`/`_previous_period`), not represented as a graph edge.

### Attributes
- `entity_description` carries the actual numeric value for Observation entities (e.g. "79.0%"), free text for other types.
- `relationship_strength` 1–10 (10 for DEVIATES_FROM, 9 for HAS_OBSERVATION/SCOPED_TO_CATEGORY/SERVES/CONTAINS, 8 for OF_METRIC/OBSERVED_IN/EVIDENCED_BY).
- `human_readable_id` — stable per-entity/relationship citation number used in `[Data: Entities (147)]`-style prompt citations; never renumbered by `canonicalize_index.py` (would break existing citations).

### Hierarchy — verified, not assumed
- **Confirmed in the graph**: State → Distributor (`SERVES`), Category → Franchise (`CONTAINS`).
- **NOT in the graph**: Zone and Sales Executive/Outlet are NOT extracted entities at all — they exist only in the underlying synthetic CSVs (`geography.csv`) and Phase-4 document prose (zone rosters), never as GraphRAG nodes. So the graph does **not** contain `State → Zone → Distributor → Sales Executive → Outlet` as a chain of extracted entities — only `State → Distributor` is a real edge; Zone is text-only context, and SE/Outlet never appear in documents at all (Phase 4 KPIs stop at State grain).
- Product hierarchy `Category → Franchise` IS in the graph via `CONTAINS`; SKU-level entities do NOT exist (KPIs never go below Franchise grain in the documents).
- The graph's real organizing hierarchy is actually **State/Distributor → Observation → Metric/Category/Period/Document** (a KPI-fact-centric star shape), not a pure org-chart hierarchy.

### Use during retrieval/query answering
GraphRAG's native `LocalSearch` uses the entity-description vector embeddings to find entities semantically close to the query, then walks the graph (relationships) and pulls in connected community reports and source text chunks — assembling `context_records` (entities/relationships/reports/sources/claims DataFrames). The custom inference layer then treats `context_records["sources"]` (raw retrieved document chunks) as authoritative for premise-checking and atomic-fact extraction, and treats `context_records["entities"/"relationships"/"reports"]` as citable evidence the grounding checker resolves claim citations against.

---

## 5. GROUNDING ARCHITECTURE

```
Retrieved Evidence (context_records: sources/entities/relationships/reports)
  ↓
Fact/Claim Extraction — TWO parallel, independent mechanisms:
  (a) answer.py's prompted self-report: the model emits a trailing JSON "claims" block
      describing its own answer's claims (entity/metric/period/value/citations/claim_type)
  (b) fact_structuring.py's deterministic regex extraction of "Atomic Facts" (distributor
      deviation / state KPI / category KPI) directly from the SAME Sources text — independent
      of what the model says about itself, used both to seed the generation prompt AND as an
      independent ground truth for grounding_check.py's numeric co-occurrence scanner
  ↓
Evidence Matching — validate_claim() resolves each claim's own `citations` list to the
  specific context_records rows they name (never the whole context blob)
  ↓
Validation — a battery of deterministic checks (below)
  ↓
LLM (never re-invoked for grounding itself — grounding is 100% deterministic Python)
```

Checks performed (all in `src/inference/grounding_check.py`, all deterministic, zero LLM calls):

- **Entity matching** (`_resolve_entity_reference`, `_detect_entity_fusion`): does the cited evidence actually mention this claim's named entity? Also catches "entity fusion" — the model merging several real distributor names into one field (e.g. "Mooney, Lamb and Weber, Scott-Norman Distributors").
- **Distributor matching** (`_known_distributor_core_names`, `scan_entity_numeric_claims`): a prose-level safety net specifically for distributor-deviation sentences that never became a structured claim.
- **Category matching** (`_extract_focus_categories`, `_sentence_categories`): a claim's cited sentence must be about the SAME category the question (or the claim's own metric field) actually asked about — prevents citing a correct-looking but wrong-category sentence.
- **Metric matching** (`_metric_words_present`): the cited text must actually name the claimed metric.
- **Value matching** (`_value_in_evidence`, tolerance 0.05): the claimed numeric value must appear (within tolerance) in the cited scope.
- **Period matching** (`extract_period_from_question`/period co-occurrence in `_primary_fact_supported`): entity+metric+value+period must appear TOGETHER, not just independently present somewhere in a multi-record citation.
- **Source matching / citation resolution** (`_resolve_citations`, `_build_evidence_index`): only the five real GraphRAG dataset names (Sources/Entities/Reports/Relationships/Claims) resolve; a fabricated citation (e.g. "Atomic State KPI Facts (1)") resolves to nothing and fails the claim closed.
- **Attribution checks** (`_VALUE_ATTRIBUTION_GUIDANCE` prompt text + `validate_claim`'s co-occurrence requirement): guards against a real value from one entity/category/period being pasted onto a neighboring one in adjacent, similarly-shaped sentences.
- **Premise checks** (Stage 0, separate module `premise_check.py`, runs BEFORE generation, not part of grounding): rejects a question's own directional assumption before the model ever writes an explanation for something that didn't happen.
- **Deviation/causal/qualitative/recommendation/definition/ranking/comparison/completeness checks**: `scan_qualitative_language`, `scan_causal_language`, `scan_recommendation_language`, `scan_definition_language`, `verify_ranking_claim`, `verify_thematic_completeness`, `check_direction_consistency` — each targets one specific, real, previously-observed failure mode (documented inline in the file's extensive comments), not a generic hypothetical.
- **Unsupported-claim handling**: any ERROR-severity issue → `status="fail"` → triggers exactly one regeneration attempt → if still failing, the final answer is withheld and replaced with an explicit "insufficient evidence" message (`pipeline.py::_render_retry_failure`). A "warning"-severity issue (currently only `generic_aggregation` — a true-but-vague generic group reference) is surfaced but never fails the check.
- **Insufficient-data handling**: handled upstream by premise-check (Stage 0), not grounding — `insufficient_data`/`undefined_metric` statuses skip generation entirely and return a deterministic hedge (`_render_hedge`) stating exactly what evidence IS available instead.

---

## 6. DATA FLOW (representation changes)

```
Synthetic CSVs (geography/outlets/products/visits/orders/inventory_snapshots — data/*.csv)
  → KPI CSVs (kpi_state_month.csv, kpi_state_month_category.csv, kpi_state_month_channel.csv,
              kpi_wd_month.csv, kpi_wd_month_category.csv — data/*.csv)
  → Narrative Documents (672 .txt files, data/graphrag_input/*.txt + docs manifest CSV)
  → GraphRAG Index (parquet: entities/relationships/communities/community_reports/text_units/
                     documents + LanceDB vector tables — data/graphrag_index/pilot_run_mini/output*/)
  → Retrieved Evidence (in-memory pandas DataFrames: context_records["sources"/"entities"/
                         "relationships"/"reports"/"claims"] — never persisted to disk)
  → Structured Facts (two parallel in-memory representations: model-self-reported AnswerClaim
                       objects, and deterministically-regexed AtomicFact objects)
  → Validated Evidence (GroundingCheckResult with pass/fail + flagged GroundingIssue list)
  → LLM Context (GraphRAG's context_chunks text + Atomic Facts block + guidance prompt blocks,
                  assembled fresh per call, never persisted)
  → Final Answer (PipelineResult dataclass, `.to_dict()` → JSON; printed to terminal by cli.py)
```

Nothing between "Retrieved Evidence" and "Final Answer" is written to disk in the current implementation — it all lives in-process for the duration of one `run_diagnostic_query()` call. Only the engine object itself (the loaded index + LocalSearch machinery) is cached in-process across questions (`ui_adapter.py`'s `_cached_engine`).

---

## 7. IMPORTANT FILES / MODULES

| File | Responsibility | Key functions/classes | Called by | Calls |
|---|---|---|---|---|
| `config/settings.py` | Central `.env`/config loader | `get_settings()`, `Settings` | almost every module | `python-dotenv` |
| `src/data_gen/generate_synthetic_data.py` | Invents the raw S&D world (hierarchy, products, visits, orders, inventory) | `generate_all()`, `build_geography/outlets/products/visits/orders/inventory()`, `verify_cascade()` | `compute_kpis.py` (imports `CHANNEL_CATEGORY_ELIGIBILITY`), `validate_index.py` (imports `CATEGORY_FRANCHISES`, `INDIAN_STATES`) | Faker, numpy |
| `src/kpis/compute_kpis.py` | Rolls raw rows into State/WD × month (× category/channel) KPI tables | `compute_all()`, `build_kpi_state_month*()`, `build_kpi_wd_month*()` | `build_documents.py` | pandas |
| `src/graph/build_documents.py` | Renders KPI rows into fixed-template narrative `.txt` documents | `build_all_documents()`, `render_document()`, `render_distributor_deviations()`, `render_channel_performance()`, `find_deviating_distributors()` | (produces GraphRAG's input; not called by later code, only its OUTPUT is consumed) | pandas |
| `data/graphrag_index/settings.yaml` + `prompts/extract_graph.txt` | GraphRAG indexing configuration + custom extraction schema | n/a (config/prompt, not code) | `graphrag index` CLI (external) | OpenAI API |
| `src/graph/validate_index.py` | Deterministic QA over a completed GraphRAG output folder | `validate()`, `check_schema/manifest_source_consistency/referential_integrity/...()` | `canonicalize_index.py`, run standalone via `python -m` | reads parquet |
| `src/graph/canonicalize_index.py` | Deterministically merges duplicate/misspelled entities and fixes the graph | `canonicalize()`, `_merge_entity_group()`, `_remap_relationships()` | run standalone via `python -m` | `validate_index.py`, LanceDB, OpenAI embeddings (only for changed entities) |
| `src/inference/schemas.py` | Typed dataclasses for every pipeline stage's result | `PremiseCheckResult`, `AnswerClaim`, `AnswerResult`, `GroundingIssue`, `GroundingCheckResult`, `PipelineResult` | every inference module | none (pure data) |
| `src/inference/context.py` | Loads index + builds GraphRAG retrieval context, stops before generation | `QueryContext`, `build_query_context()` | `pipeline.py`, `ui_adapter.py` | GraphRAG's `get_local_search_engine`, `DataReader`, embedding API |
| `src/inference/premise_check.py` | Stage 0 — deterministic directional-claim verification | `check_premise()`, `extract_direction_claim()`, `_previous_period()` | `pipeline.py` | regex over `context_records["sources"]` |
| `src/inference/fact_structuring.py` | Deterministic regex extraction of KPI/deviation sentences into "Atomic Facts" | `build_atomic_facts_block()`, `extract_atomic_facts()`, `AtomicFact` | `answer.py` (prompt text), `grounding_check.py` (structured facts) | regex over `context_records["sources"]` |
| `src/inference/answer.py` | Stage 1 — the ONLY module that calls the completion LLM | `generate_answer()`, `regenerate_answer()`, `_parse_claims()` | `pipeline.py` | `fact_structuring.py`, GraphRAG's `LocalSearch.model` |
| `src/inference/grounding_check.py` | Stage 2 — deterministic claim + prose validation | `check_grounding()`, `validate_claim()`, `verify_ranking_claim()`, many `scan_*()` | `pipeline.py` | `fact_structuring.py`, `premise_check.py` (period/state extractors) |
| `src/inference/pipeline.py` | Orchestrates Stage 0→1→2→(retry) | `run_pipeline()`, `answer_question()` | `ui_adapter.py`, own CLI (`_main`) | `context.py`, `premise_check.py`, `answer.py`, `grounding_check.py` |
| `src/inference/ui_adapter.py` | Clean UI/backend boundary; caches the GraphRAG engine across questions | `run_diagnostic_query()`, `_get_query_context()` | `src/cli.py` (and any future UI) | `context.py`, `pipeline.py` |
| `src/cli.py` | Minimal terminal chatbot | `main()` | end user | `ui_adapter.py` |

---

## 8. MODELS & TECHNOLOGIES ACTUALLY USED

| Component | Technology | Model | Purpose | Index-time / Query-time |
|---|---|---|---|---|
| GraphRAG | Microsoft `graphrag` Python library (v3.1.0 per pinned requirements) | — | Chunking, extraction orchestration, community detection, local search retrieval | Both |
| Completion LLM (indexing) | OpenAI API | `gpt-4o` (production `settings.yaml`) — the actually-queried mini-pilot index used `gpt-4o-mini` (`pilot_run_mini/settings.yaml`) | Entity/relationship extraction, description summarization, community reports | Index-time |
| Completion LLM (query) | OpenAI API, via GraphRAG's own model client (`engine.model`) | Same model configured in the loaded index's `settings.yaml` (`default_completion_model`) | Answer generation (`answer.py`), retry regeneration | Query-time |
| Embedding model | OpenAI API | `text-embedding-3-small` | Entity description embeddings (index-time); query embedding for vector search (query-time) | Both |
| Vector store | LanceDB (local, file-based) | — | Stores/searches entity description, community full-content, text-unit-text embeddings | Index-time write, query-time read |
| Grounding / premise logic | Pure Python (`re`, `pandas`) | — (no model) | Deterministic validation, zero API cost | Query-time |
| Local Ollama test | Ollama (`llama3` + `nomic-embed-text`) | — | A one-off, abandoned/failed smoke test (`data/graphrag_index_ollama_test/`) — historical artifact only, not part of the live system | n/a |
| Synthetic data generation | Faker, numpy | — | No LLM involved at all | Pre-index-time |

---

## 9. INDEX-TIME VS QUERY-TIME

### INDEX-TIME (happens once, or whenever the corpus/config changes)
1. Run `python -m src.data_gen.generate_synthetic_data` → 6 raw CSVs.
2. Run `python -m src.kpis.compute_kpis` → 5 KPI CSVs.
3. Run `python -m src.graph.build_documents` → 672 `.txt` documents + manifest.
4. Run the external `graphrag index` CLI against `data/graphrag_index/{settings.yaml or pilot_run_mini/settings.yaml}` → parquet output + LanceDB.
5. Optionally run `python -m src.graph.validate_index` then `python -m src.graph.canonicalize_index` → a cleaned output folder (e.g. `output_fixed_clean`).

**Persisted artifacts reused at query time**: the parquet tables (`entities`, `relationships`, `communities`, `community_reports`, `text_units`, `documents`) and the LanceDB vector tables under whichever `output*/` directory `OUTPUT_DIR` in `ui_adapter.py` points to (currently hardcoded to `data/graphrag_index/pilot_run_mini/output_fixed_clean`).

### QUERY-TIME (happens on every user question)
1. `run_diagnostic_query(question)` — validates input, loads `.env`.
2. `_get_query_context(question)` — reuses a process-lifetime-cached `LocalSearch` engine (built once from the persisted index) and always runs a *fresh* `build_context()` retrieval call for this specific question (1 embedding API call, every time).
3. `check_premise()` — deterministic, reads only the already-retrieved `context_records["sources"]`.
4. (conditionally) `generate_answer()` — 1 completion API call.
5. `check_grounding()` — deterministic.
6. (conditionally) `regenerate_answer()` — 1 more completion API call (bounded to exactly one retry).
7. `check_grounding()` again if step 6 ran.
8. Return `PipelineResult`.

**Nothing query-specific is cached or persisted** — every question gets fresh retrieval, fresh premise-check, fresh generation, fresh grounding-check; only the engine object (index-loading machinery) survives across questions in the same process.

---

## 10. ONE CONCRETE QUERY WALKTHROUGH

Question: **"Why did Service Level drop in Gujarat in April 2026?"**

```
Question
  ↓  ui_adapter.run_diagnostic_query("Why did Service Level drop in Gujarat in April 2026?")
_get_query_context() — reuses cached LocalSearch engine (or builds it on first call),
  calls engine.context_builder.build_context(query=question, ...) → ONE embedding API call.
  Returns context_records: DataFrames of entities/relationships/reports/sources ranked by
  relevance to the embedded query — e.g. Sources rows for Gujarat_2026-04.txt and
  Gujarat_2026-03.txt (the actual prior month), Entities rows for "GUJARAT SERVICE LEVEL
  APRIL 2026", Relationships rows connecting distributors DEVIATES_FROM that Observation.
  ↓
run_pipeline(question, qctx)
  ↓
check_premise(question, context_records)
  - extract_direction_claim("...drop...") → "decline"
  - extract_state_from_question(...) → "Gujarat"
  - extract_period_from_question(...) → "April 2026"
  - extract_metric_claim(...) → "Service Level" (named explicitly, so this ONE metric
    is checked, not the generic Productivity+Service Level headline pair)
  - _parse_source_documents(sources_df) → regexes "Service Level for Gujarat in <Period>
    was <value>%." out of every retrieved Sources row's text
  - baseline_period = _previous_period("April 2026") = "March 2026"
  - IF a March 2026 Gujarat document was retrieved and parses a Service Level value:
      compares March→April, direction computed ("down"/"up"/"flat")
      status = "supported" (claim confirmed) / "contradicted" (evidence says it rose)
        / "unsupported" (mixed — only one metric checked here so this is unlikely) 
  - IF no March 2026 Gujarat document is in the retrieved evidence:
      status = "insufficient_data" → PIPELINE STOPS HERE, zero completion calls,
      _render_hedge() returns: "The indexed evidence does not include a March 2026
      document for Gujarat, so a claim that performance declined in April 2026 cannot
      be evaluated against its actual prior period." (+ whatever other periods ARE
      indexed, as supplementary comparisons)
  ↓ (assume status == "supported": March 92.1% → April 88.4%, confirmed down)
generate_answer(engine, question, context_result)
  - build_atomic_facts_block(context_records) regexes every distributor-deviation and
    state/category KPI sentence in the retrieved Sources text into explicit FACT blocks
    (e.g. "FACT 3\nState: Gujarat\nMetric: Service Level\nPeriod: April 2026\nValue: 88.4%
    \nSource: Sources (21)") — appended to GraphRAG's own context_chunks text.
  - The prompt = index's own local_search system prompt (filled with context_data +
    response_type) + 6 stacked guidance blocks (specificity/value-attribution/entity-
    existence/resolution-reasoning/citation-selection/atomic-facts-citation) + the
    structured-claims-JSON request.
  - ONE completion call streams back prose ("Service Level in Gujarat fell from 92.1%
    in March 2026 to 88.4% in April 2026. Distributor Landry Ltd Distributors showed a
    significant Service Level deviation of ... which may have contributed to...") plus
    a trailing ```json claims block self-describing each claim's entity/metric/period/
    value/citations/claim_type.
  ↓
check_grounding(answer.text, context_records, answer.claims, premise.metric_comparisons, question)
  - validate_claim() resolves each claim's citations (e.g. "Sources (21)") to the exact
    retrieved row text and checks: is 88.4% actually in that text? is "Gujarat" in it?
    is "April 2026" in it? does entity+metric+period+value co-occur in ONE sentence?
  - scan_causal_language() checks the "may have contributed to" clause: is this citation's
    text itself causal-flavored, or just a plain fact? (Phase 4 docs never assert
    causation, so a strong causal claim like "caused" would fail here; a hedged "may
    have contributed to" against a merely-correlated fact is the deliberately-tolerated
    boundary case.)
  - check_direction_consistency() cross-checks "fell" against premise_check's own
    computed direction ("down") — must agree.
  - IF any check fails → status="fail" → regenerate_answer() runs ONCE with the exact
    flagged issues → check_grounding() runs again on the revision → commit to either
    the fixed answer (final_decision="regenerated") or a fail-closed message
    (final_decision="insufficient_evidence") if still bad.
  - IF everything passes on the first draft → final_decision="pass_through".
  ↓
Final Answer
  PipelineResult.final_text returned to ui_adapter → printed by cli.py along with a debug
  line: "[final_decision=pass_through premise=supported grounding=pass llm_calls=2]"
  (1 embedding + 1 completion).
```

---

## 11. CURRENT ARCHITECTURE — SOURCE OF TRUTH SUMMARY

### A. High-Level Architecture (major components)
1. Synthetic Data Generator (`generate_synthetic_data.py`)
2. KPI Computation Engine (`compute_kpis.py`)
3. Narrative Document Builder (`build_documents.py`)
4. GraphRAG Indexing Pipeline (external `graphrag index` CLI + custom `settings.yaml`/`extract_graph.txt`)
5. Index QA & Canonicalization (`validate_index.py`, `canonicalize_index.py`)
6. GraphRAG Knowledge Graph Store (parquet + LanceDB, on disk)
7. Retrieval Layer / Query Context Builder (`context.py`, wraps GraphRAG's `LocalSearch`)
8. Premise Verification Guard (`premise_check.py`)
9. Atomic Fact Structuring (`fact_structuring.py`)
10. Answer Generation (`answer.py`) — the one LLM-calling stage
11. Grounding/Contradiction Validator (`grounding_check.py`)
12. Pipeline Orchestrator (`pipeline.py`)
13. UI Boundary / Engine Cache (`ui_adapter.py`)
14. Terminal Chat Client (`cli.py`)

### B. End-to-End Flow (numbered)
1. Generate synthetic S&D CSVs.
2. Compute State/WD × month KPI tables.
3. Render KPI tables into 672 fixed-template narrative documents.
4. Index documents with GraphRAG (custom 8-entity/8-relationship schema) → graph + community reports + embeddings.
5. (Optional) Validate and canonicalize the index output.
6. User asks a question → build retrieval context (1 embedding call).
7. Deterministically verify the question's directional premise against retrieved evidence.
8. If premise fails, return a deterministic hedge — stop.
9. If premise passes, generate one LLM answer with self-reported structured claims, using GraphRAG context + deterministically-extracted Atomic Facts.
10. Deterministically validate every claim and the prose against retrieved evidence.
11. If invalid, regenerate once with the specific flagged issues, re-validate.
12. Return the final, structured `PipelineResult`.

### C. Knowledge Graph Structure (actual)
- Entities: State, Period, Distributor, Category, Franchise, Observation, Metric, Document (8 types only).
- Relationships: HAS_OBSERVATION, SCOPED_TO_CATEGORY, OF_METRIC, OBSERVED_IN, SERVES, CONTAINS, EVIDENCED_BY, DEVIATES_FROM (8 types only).
- Confirmed hierarchy edges: State→Distributor (SERVES), Category→Franchise (CONTAINS).
- NOT in the graph: Zone, Sales Executive, Outlet, SKU — none of these are extracted entities. The requirements-doc-style "State→Zone→Distributor→SE→Outlet" full org chain does NOT exist as graph edges; only State→Distributor does.
- The graph is fact-centric: every KPI number is its own Observation node linked to State/Distributor, Metric, Period, Category (if scoped), and its source Document.

### D. Query Flow (actual runtime sequence)
Question → cached-engine retrieval (1 embedding call) → deterministic premise check → (gate) → LLM answer generation with self-reported claims (1 completion call) → deterministic grounding check → (gate, bounded to 1 retry) → LLM regeneration (0 or 1 more completion call) → deterministic re-check → final structured result.

### E. Grounding Flow (actual validation sequence, inside `check_grounding()`)
1. Per-claim citation resolution + validation (`validate_claim()` — numeric/entity/period/co-occurrence/causal/comparison checks, claim-type-aware).
2. Prose-level fallback scans over the whole answer text: qualitative language, causal language, recommendation language, definition language.
3. Distributor-numeric co-occurrence safety net (`scan_entity_numeric_claims`).
4. Ranking-claim verification against Atomic Facts (`verify_ranking_claim`).
5. Thematic completeness check (`verify_thematic_completeness`, warning-only).
6. Direction-consistency cross-check against premise-check's own computed metric directions.
7. Aggregate: any ERROR-severity issue → fail; only-WARNING issues → pass with notices.

### F. Important Architecture Facts (must-know before building any visual)
1. This is Microsoft GraphRAG (the actual open-source library), not a custom-built graph database.
2. Indexing is a config-driven CLI run (`graphrag index`), not a custom Python indexing script — no `src/graph/build_index.py` exists.
3. The custom work at index-time is entirely in the PROMPT (`extract_graph.txt`) and post-hoc QA (`validate_index.py`/`canonicalize_index.py`), not in the extraction mechanism itself.
4. The knowledge graph has exactly 8 entity types and exactly 8 relationship types — closed vocabularies enforced by prompt instruction and checked by `validate_index.py`.
5. Zone and Channel are explicitly excluded from the graph by prompt instruction — they exist only as prose context.
6. SKU-level and Outlet/Sales-Executive-level entities never exist in the graph — the finest business grain represented is Distributor and (state/distributor)×Category×Metric×Period.
7. There is no vector database beyond GraphRAG's own bundled LanceDB — no separate Pinecone/Weaviate/Chroma.
8. There is exactly ONE embedding API call per user question (the query embedding for retrieval) and 1–2 completion API calls (answer generation, plus at most one bounded retry) — cost is capped and predictable by design.
9. Premise-checking and grounding-checking are 100% deterministic Python (regex + pandas) — zero LLM calls in either stage.
10. The LLM is called in exactly one function, `answer.py`, reused identically for both the first draft and the one allowed retry.
11. The retry is hard-bounded to exactly one attempt — never a loop — by explicit design decision documented in the code.
12. Grounding validation works by resolving each claim's own self-reported citations to specific evidence rows, not by searching the whole retrieved context for a matching word — this is the core fix over an earlier, weaker word-level version.
13. "Atomic Facts" (fact_structuring.py) is a fully custom, additive, regex-based re-structuring of the SAME retrieved text GraphRAG already returned — it adds no new retrieval and does not modify GraphRAG's own output.
14. Causal claims are held to the strictest bar: the source corpus documents never assert causation, so a claim using "caused"/"led to"/"resulted in" language only passes if the CITED text itself contains causal language — otherwise it's flagged and expected to be hedged on retry.
15. If the question's premise (e.g. "declined") isn't supported by the actual prior-period evidence, the system NEVER calls the answer-generation LLM at all for that turn — it returns a deterministic message describing what the evidence does show.
16. The synthetic dataset (Faker/numpy) has NO real GPIL data in it anywhere — it's entirely invented, with deliberately seeded imperfections (stockouts, partial fulfillment, festive-season spikes, per-state performance skew) so diagnostic questions have real signal.
17. Real brand names used: Marlboro, TicTac, Kinder Joy — everything else (GPI_Franchise_1..6, Candy_Franchise_1/2) is a generic placeholder because real GPIL internal names weren't provided.
18. Document granularity is State × month only (672 documents) — there is no per-Zone, per-SE, or per-outlet document, and no SKU-level document.
19. The live/tested index actually queried by `ui_adapter.py` is a 32-document MINI PILOT (`pilot_run_mini`, GPT-4o-mini), not the full 672-document production corpus — `INDEX_ROOT`/`OUTPUT_DIR` are hardcoded constants pointing there.
20. `canonicalize_index.py` never renumbers an existing entity's `human_readable_id` (would break citations already used in generated community reports); it only appends new ids above the current max.
21. `canonicalize_index.py` makes API calls ONLY to re-embed entities whose description text actually changed due to a merge (or opt-in backfill) — never a full re-embed, never touching unaffected entities.
22. The engine (GraphRAG's `LocalSearch` object, built from the loaded index) is cached for the lifetime of the process across repeated questions; nothing question-specific is ever cached.
23. `PremiseCheckResult.status="undefined_metric"` exists for superlative/ranking questions ("which state performed worst") that this project defines no computable metric or comparable aggregate for — rejected before retrieval-dependent reasoning even runs.
24. `GroundingIssue.severity="warning"` (currently only `generic_aggregation`) never fails a grounding check by itself — only `severity="error"` issues do.
25. The system has a Streamlit UI mentioned in project planning docs, but no Streamlit code exists in the repository yet — the only working UI today is the terminal chatbot (`src/cli.py`).
26. There is no separate "query classification" model or module — direction/state/period/metric extraction inside `premise_check.py` is the closest analog, and it's pure regex, not an LLM classifier.
27. `Claims` extraction (GraphRAG's native claim/covariate feature) is explicitly disabled at index time (`extract_claims.enabled: false`) — the `context_records["claims"]` table is always empty in this project.
28. `src/reports/generate_pan_india_summary.py` is a standalone Excel-report generator built from the raw Phase-2 CSVs — it is NOT part of the query/answer pipeline and shares no code with it.

### G. Things NOT to show (incorrect, not implemented, experimental, or superseded)
- Do NOT show a "3-stage inference layer (structure evidence → guarded chain-of-thought reasoning → grounding check)" as three separate LLM calls — that was an early design description; the actual implementation is ONE completion call for structured drafting (with self-reported claims) plus fully deterministic (non-LLM) premise and grounding stages.
- Do NOT show a Streamlit chat UI as an existing, working component — it does not exist in the codebase; `src/ui/__init__.py` is an empty placeholder package.
- Do NOT show State → Zone → Distributor → Sales Executive → Outlet as an actual knowledge-graph relationship chain — only State→Distributor is a real graph edge; Zone/SE/Outlet are never extracted entities.
- Do NOT show SKU-level entities or Category→Franchise→SKU as a graph relationship chain — SKU never appears in documents or the graph; only Category→Franchise (CONTAINS) is real.
- Do NOT show "query classification" as a distinct ML/LLM-based component — it is regex-based direction/entity/period extraction inside premise_check.py, not a classifier model.
- Do NOT show GraphRAG "Claims"/covariates as an active part of the pipeline — disabled at index time, always empty at query time.
- Do NOT show a full 672-document production index as the one being live-queried today — the only index wired into `ui_adapter.py` is a 32-document mini pilot on `gpt-4o-mini`.
- Do NOT show retry as an open-ended/looping mechanism — it is hard-capped at exactly one attempt, by explicit design.
- Do NOT show grounding or premise-checking as LLM-based/AI steps — both are 100% deterministic Python string/regex/arithmetic logic with zero model calls.
- Do NOT show a generic multi-provider LLM setup (Ollama/Groq as live options) — the only live path uses OpenAI directly; the Ollama test is a documented historical failure, not a supported alternate backend.
- Do NOT show the 9-phase project plan's later phases (Phase 8 Streamlit UI, Phase 9 Evaluation & Hardening) as complete — only through Phase 7 (guarded pipeline) plus a Phase-8-labeled CLI/boundary layer exist; formal evaluation harness and UI are not built.
