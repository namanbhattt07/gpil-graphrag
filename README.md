# GPIL — Sales & Distribution GraphRAG Chat System

A chat system over GPIL's Sales & Distribution (S&D) KPI data. You ask a
plain-language question — e.g. *"Why did Bihar's Service Level decline from
October to November 2025?"* — and it returns an evidence-backed answer that
explains the **why**, not just the raw numbers, by retrieving from a
knowledge graph built over the S&D data and reasoning strictly over that
retrieved evidence. Every claim in an answer must trace back to retrieved
evidence; if the evidence doesn't support a claim, the system withholds or
qualifies it rather than guessing.

Full requirements: `GPIL_GraphRAG_Requirements.pdf` (project spec this repo
implements). Deep architecture reference: `GPIL_ARCHITECTURE_SOURCE_OF_TRUTH.md`.
Plain-language phase-by-phase design doc: `docs/TECHNICAL_DESIGN.md`.

## How the pieces fit together

1. **Data + KPIs** (`src/data_gen`, `src/kpis`) — synthetic S&D data across
   GPIL's real hierarchy (State → Zone → Wholesale Distributor → Sales
   Executive → Outlet, and Category → Franchise → SKU) is generated with a
   fixed random seed, then rolled up into State × month KPIs (Numeric
   Distribution, ACV, Out-of-Stock %, Service Level, Inventory Turns/Days,
   ACL, SKUs/Transaction, Range Billing, Strike Rate).
2. **Graph-ready documents** (`src/graph/build_documents.py`) — each
   State × month KPI row becomes a self-contained narrative `.txt` document
   naming its state, month, distributors, categories, and franchises
   verbatim from the source tables (so GraphRAG's entity extraction links
   the same real-world entity across documents instead of drifting).
3. **Knowledge Graph + GraphRAG indexing** — those documents are indexed
   with Microsoft GraphRAG using a domain-specific 8-entity-type schema
   (State, Period, Distributor, Category, Franchise, Observation, Metric,
   Document) and 8 relationship types, instead of GraphRAG's generic
   defaults. Indexing output is deterministically cleaned up afterwards
   by `src/graph/validate_index.py` (read-only QA report) and
   `src/graph/canonicalize_index.py` (merges duplicate entities, backfills
   a known-missing State entity, removes provably-orphaned forbidden
   entities — never renumbers existing entity ids, so community-report
   citations stay valid).
4. **Retrieval + guarded inference** (`src/inference/`) — a question is
   answered through a 3-stage guarded pipeline, not a single opaque LLM
   call:
   - **Premise check** (deterministic, no LLM call): verifies any claim
     baked into the question itself (e.g. "declined") against the actual
     retrieved numbers and the real previous period, before generating
     anything. A question with no real baseline in the index (e.g. asking
     about a month whose prior month isn't indexed) short-circuits here
     with `insufficient_data` instead of getting a fabricated explanation.
   - **Answer generation** (the only stage that calls an LLM): produces
     prose plus a structured, self-reported list of claims (entity,
     metric, period, direction, citations) in the same call.
   - **Grounding check** (deterministic, no LLM call): resolves every
     claim to only the specific evidence rows it cited (never the whole
     retrieved context blob) and checks numeric/entity/period/direction/
     causal/qualitative support. An answer that fails is regenerated
     once (bounded — never a retry loop) with the specific issues flagged;
     if it still fails, the pipeline fails closed with
     `insufficient_evidence` rather than showing an ungrounded answer.
5. **Chat UI** (`src/cli.py`, `src/ui/streamlit_app.py`) — both are thin
   rendering layers over one shared boundary function,
   `src/inference/ui_adapter.py:run_diagnostic_query()`. Neither UI touches
   retrieval, inference, or grounding logic directly — every fact shown
   (answer text, evidence, citations, pass/fail status) comes straight out
   of the `PipelineResult` the backend already validated.

## Pilot corpus and index

The full synthetic corpus is 672 State × month documents (28 states × 24
months), but this repo's committed, queryable index is a **62-document
pilot** — `data/graphrag_index/input_pilot/` — chosen to keep indexing cost
and time reasonable while still exercising every question type the system
supports:

- All 28 states are represented, most with 2 consecutive months (so a
  basic period-over-period question is answerable for any state).
- Six states (Bihar, Gujarat, Uttarakhand, Maharashtra, Karnataka,
  Rajasthan) have 3+ consecutive months, so multi-period trend and
  root-cause ("why") questions have a real prior-period baseline to
  reason over.
- Documents are the exact, unmodified output of `src/graph/build_documents.py`
  run against the full synthetic dataset — the pilot is a *selection*, not
  a separately generated corpus, so it stays consistent with the full
  672-document set if the corpus is ever scaled up later.

The index itself (`data/graphrag_index/pilot_run_mini/output_fixed_clean/`)
is what `src/inference/ui_adapter.py` queries — the path is a hardcoded
constant there by design (a single, known-good pilot index, not a
runtime-configurable target). It is **not** committed to git (regenerable,
large); see "Reproducing the pilot index" below.

An earlier, smaller (32-document) pilot and a separate GPT-4o comparison
run existed during development; both were superseded by this 62-document,
gpt-4o-mini pilot and have been removed as part of finalizing this repo.

## Setup (one-time)

1. **Create a virtual environment** (Python 3.11 — chosen for GraphRAG
   dependency compatibility; a newer system Python will not work):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
   You'll know it worked because your terminal prompt gets a `(venv)`
   prefix. To leave the virtual environment later, run `deactivate`.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your secrets file:**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and fill in your real `OPENAI_API_KEY`. `.env` is
   gitignored, so this never gets committed. The default models
   (`gpt-4o-mini` for chat, `text-embedding-3-small` for embeddings) are
   what the committed pilot index was built and validated with.
4. **Run the tests** to confirm everything is wired up (see below).

## Reproducing the pilot index

The synthetic data and KPI CSVs are gitignored and regenerated with a fixed
seed; the pilot's 62 input documents and its GraphRAG index are gitignored
too (large, and one `graphrag index` API call away from being reproduced
exactly). To rebuild everything from scratch:

```bash
python -m src.data_gen.generate_synthetic_data   # Phase 2 — raw S&D CSVs (~9 min)
python -m src.kpis.compute_kpis                  # Phase 3 — State x month KPI tables
python -m src.graph.build_documents              # Phase 4 — 672 narrative documents
```

The 62 pilot documents already live under `data/graphrag_index/input_pilot/`
and are committed to git, so indexing can be re-run directly without
redoing the steps above:

```bash
graphrag index --root data/graphrag_index/pilot_run_mini
```

Then apply the same deterministic cleanup pass used for the committed
index (each step's `--output-dir` becomes the next step's `--input-dir`):

```bash
python -m src.graph.canonicalize_index \
    --input-dir data/graphrag_index/pilot_run_mini/output \
    --output-dir data/graphrag_index/pilot_run_mini/output_fixed \
    --source-dir data/graphrag_index/input_pilot \
    --manifest data/graphrag_docs_manifest.csv

python -m src.graph.canonicalize_index \
    --input-dir data/graphrag_index/pilot_run_mini/output_fixed \
    --output-dir data/graphrag_index/pilot_run_mini/output_fixed_maharashtra \
    --source-dir data/graphrag_index/input_pilot \
    --manifest data/graphrag_docs_manifest.csv \
    --backfill-missing-entities

python -m src.graph.canonicalize_index \
    --input-dir data/graphrag_index/pilot_run_mini/output_fixed_maharashtra \
    --output-dir data/graphrag_index/pilot_run_mini/output_fixed_clean \
    --source-dir data/graphrag_index/input_pilot \
    --manifest data/graphrag_docs_manifest.csv \
    --remove-orphaned-forbidden-entities
```

`output_fixed_clean/` is the path `ui_adapter.py` reads. You can verify a
freshly built index with:

```bash
python -m src.graph.validate_index \
    --input-dir data/graphrag_index/pilot_run_mini/output_fixed_clean \
    --source-dir data/graphrag_index/input_pilot \
    --manifest data/graphrag_docs_manifest.csv
```

## How to run the chatbot

**Terminal:**
```bash
python -m src.cli
```

**Web UI (Streamlit):**
```bash
streamlit run src/ui/streamlit_app.py
```

Both require `OPENAI_API_KEY` set in `.env` and the pilot index built (see
above).

## Example queries

These cover the question types the pipeline is designed to handle, and are
answerable against the committed 62-document pilot index:

- **KPI lookup:** "What was Bihar's Service Level in October 2025?"
- **Why / diagnostic:** "Why did Bihar's Service Level decline from October
  to November 2025?"
- **Comparison:** "Compare Bihar's Service Level between October 2025 and
  November 2025."
- **Distributor / category attribution:** "Which distributors served
  Maharashtra in August 2024, and how did the GPI category perform there?"
- **Multi-period trend:** "How did Bihar's Service Level trend across
  September, October, and November 2025?"

A question about a state/period combination not in the pilot index (or one
whose required prior period isn't indexed) correctly returns
`insufficient_data`/`insufficient_evidence` rather than a fabricated
answer — this is expected, guarded behavior, not a bug.

## Running the tests

```bash
pytest
```

All tests are mocked/synthetic — the suite makes zero external API calls,
so it's safe and free to run repeatedly. `tests/test_data_gen.py` calls the
real (fixed-seed) data generator at a small scale and is the slowest file.

## Repository structure

```
code/
├── config/                 settings.py (loads .env), settings.py Settings dataclass
├── data/
│   ├── graphrag_index/
│   │   ├── input_pilot/    the 62 curated pilot documents (committed)
│   │   ├── prompts/        domain-tuned GraphRAG extraction/search prompts (committed)
│   │   └── pilot_run_mini/ settings.yaml (committed); cache/output/logs (gitignored, regenerable)
│   └── *.csv, graphrag_input/, data_analysis.ipynb  generated data (gitignored except the notebook)
├── docs/                   design docs, business validation report, entity/relationship schema
├── scripts/                 one-off reporting utilities (presentation preview, Pan-India summary)
├── src/
│   ├── data_gen/            Phase 2 — synthetic S&D data generator
│   ├── kpis/                Phase 3 — State x month KPI computation
│   ├── graph/                Phase 4-5 — document construction, indexing QA/cleanup
│   ├── inference/            Phase 6-7 — retrieval, premise check, answer generation, grounding
│   ├── reports/               ad-hoc Pan-India Excel summary
│   ├── ui/                    Streamlit chat app
│   └── cli.py                 terminal chat app
├── tests/                   pytest suite (mocked — no external API calls)
├── requirements.txt         pinned dependency versions
├── .env.example             template — copy to .env and fill in your API key
└── GIT_GUIDE.md              plain-language git walkthrough
```

## Important limitations

- **Pilot-scale index, not the full corpus.** The committed, queryable
  index covers 62 of the 672 possible State × month documents. Questions
  about states/periods outside the pilot will correctly report
  insufficient evidence rather than answer from the (unindexed) full
  dataset.
- **Synthetic data.** All S&D data (outlets, orders, visits, inventory) is
  generated, not real GPIL data. See `docs/BUSINESS_VALIDATION_REPORT.md`
  for a business-analyst-style review of what's realistic about it and
  what isn't yet (a few findings — flat fulfilment shortfall, static
  prices, no year-over-year growth — remain open).
- **Two KPI definitions need business confirmation, not just engineering
  sign-off:** ACL (currently a volume-per-visit proxy) and Range Billing
  (currently a per-outlet SKU-range-billed fraction) — both flagged in
  code comments in `src/kpis/compute_kpis.py` rather than guessed
  silently.
- **A causal claim can be grounded in an LLM-generated community report**
  rather than only in a raw fact-listing source document, in rare cases —
  a known, tracked limitation of the grounding check, not a silent gap
  (see `GPIL_ARCHITECTURE_SOURCE_OF_TRUTH.md` §5).
- **Single bounded retry.** If a generated answer fails the grounding
  check twice (original + one retry), the pipeline fails closed rather
  than retrying indefinitely.
