# GPIL — Sales & Distribution GraphRAG Chat System

A chat system over Sales & Distribution (S&D) KPI data. You ask a plain-language
question — e.g. *"Why did sales drop in the North region in Q3?"* — and it
returns an evidence-backed answer that explains the **why**, not just the
raw numbers, by reasoning over a knowledge graph built from the data.

Full requirements: `GPIL_GraphRAG_Requirements.pdf` (project spec this repo implements).

## How the pieces fit together

1. **Data + KPIs** — synthetic S&D data (regions, distributors, retailers,
   orders, inventory) → cleaned into region×month KPIs.
2. **Knowledge Graph + GraphRAG** — the data becomes a graph
   (Region → Retailer → Order → SKU, all tied to Time), indexed with
   Microsoft GraphRAG for retrieval.
3. **Inference layer** — a question comes in, GraphRAG retrieves evidence,
   and an LLM reasons over *only* that evidence to produce a grounded,
   cited answer with a confidence tag.

Everything is served through a Streamlit chat app.

## Project layout

```
code/
├── data/               generated CSVs live here (gitignored — see below)
├── src/
│   ├── data_gen/       Phase 2 — synthetic S&D data generator
│   ├── kpis/           Phase 3 — region x month KPI computation
│   ├── graph/          Phases 4-6 — GraphRAG documents, indexing, retrieval
│   ├── inference/       Phase 7 — grounded reasoning over evidence
│   └── ui/             Phase 8 — Streamlit chat app
├── config/
│   ├── settings.py     loads API keys / config from .env (see below)
│   └── settings.yaml   (added in Phase 5) GraphRAG's own config
├── tests/              pytest tests
├── requirements.txt    exact pinned dependency versions
├── .env.example        template you copy to .env and fill in
└── venv/               your local virtual environment (not committed)
```

Each `src/` subfolder currently just has a docstring explaining what will
live there — they get filled in phase by phase.

**Why `data/` and GraphRAG output are gitignored:** the synthetic data and
the GraphRAG index are both *generated* by scripts using a fixed random
seed, so anyone can reproduce them exactly by re-running the phase scripts.
Committing generated CSVs/indexes would bloat the repo and produce noisy
diffs for no benefit — the code that produces them is what's version
controlled.

## Setup (one-time)

1. **Create a virtual environment** (isolates this project's Python
   packages from the rest of your system — already done for you the first
   time, but here's the command for reference):
   ```bash
   python3.11 -m venv venv
   ```
2. **Activate it** (do this every time you open a new terminal to work on
   this project):
   ```bash
   source venv/bin/activate
   ```
   You'll know it worked because your terminal prompt gets a `(venv)`
   prefix. To leave the virtual environment later, run `deactivate`.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your secrets file:**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and fill in your real `OPENAI_API_KEY`. `.env` is
   gitignored, so this never gets committed.
5. **Run the tests** to confirm everything is wired up:
   ```bash
   pytest
   ```

## How to run each phase

- **Phase 1 (this phase):** nothing to "run" yet beyond `pytest` — it
  proves the project skeleton, dependency install, and secret-loading all
  work.
- **Phase 2 onward:** instructions will be added here as each phase is
  built, along with the exact command to run it and what output to expect.

## Working with Git

This project is checkpointed with git after each phase. If you're new to
git, see [`GIT_GUIDE.md`](GIT_GUIDE.md) for a plain-language walkthrough of
every command used in this repo.

## Build phases (status)

| # | Phase | Status |
|---|-------|--------|
| 1 | Project Setup & Scaffolding | ✅ done |
| 2 | Synthetic Data Generation | ⬜ not started |
| 3 | KPI Computation | ⬜ not started |
| 4 | Graph-Ready Document Construction | ⬜ not started |
| 5 | GraphRAG Indexing | ⬜ not started |
| 6 | Retrieval Layer | ⬜ not started |
| 7 | Inference Layer | ⬜ not started |
| 8 | Streamlit UI | ⬜ not started |
| 9 | Evaluation & Hardening | ⬜ not started |
