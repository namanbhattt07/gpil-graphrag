# GPIL S&D — GraphRAG Document Structure Recommendation

**Purpose:** specifies exactly how `data/graphrag_input/*.txt` documents should be written so the entity/relationship schema (`GRAPHRAG_ENTITY_RELATIONSHIP_SCHEMA.md`) can actually be extracted by an LLM. This is the last design step before extraction prompting and the 10-doc pilot.

**Source of the "before" examples:** the three sample documents originally reviewed (`Maharashtra_2024-08.txt`, `Assam_2025-10.txt`, `Karnataka_2025-03.txt`). Per the Business Dictionary's "Inferred, not verified" note, these are from a **stale data run** — their exact KPI numbers (e.g. Maharashtra Dropsize = 65.79) do not match current CSVs. The "after" rewrites below use current, verified numbers where available (Maharashtra state-level and WD-level, confirmed in the sanity-check step) and mark anything not yet computed as `[PLACEHOLDER]` rather than inventing a number.

---

## 1. Entity Tagging Format

**Rule:** every entity mention gets a labeled tag line immediately before its first prose use in a section, in the exact form `EntityType: exact_name`. This directly addresses the confirmed cause of the earlier Ollama extraction failure (casing drift across passes) — a labeled line removes the need for the LLM to infer entity type from context.

```
State: Maharashtra
Period: August 2024
```

For lists (distributors, franchises), tag the type once, then list members consistently:

```
Distributors serving Maharashtra: Boyd-White Distributors, Larson-Hernandez Distributors, ...
```

**Naming rule enforced here:** always write `Distributor`, never `WD`, per the schema's collision warning.

---

## 2. KPI Sentence Format

**Rule:** one KPI = one sentence, subject-predicate-object, with the metric name, scope, period, and value in a fixed order. This is already a genuine strength of the current documents (confirmed in the dictionary) — preserve it, don't redesign it.

```
[Metric] for [Scope] in [Period] was [Value].
```

Example (unchanged pattern, current data):
```
Productivity for Maharashtra in August 2024 was 79.0%.
Service Level for Maharashtra in August 2024 was 92.3%.
```

**New addition — explicit metric-definition anchor**, so `Metric` nodes carry their formula/confirmation-status property (schema Section 3) instead of that living only in code comments:

```
Dropsize (units ordered per productive visit — definition pending GPIL confirmation) for Maharashtra in August 2024 was 200.96.
```

Apply the "(definition pending GPIL confirmation)" qualifier only to the three flagged metrics — Dropsize, ACV, Range Billing — per the dictionary's KPI Model. Do not add it to the other seven; over-qualifying trains the LLM to treat every number as uncertain.

---

## 3. Anomaly / DEVIATES_FROM Sentence Format

**Rule:** this is new content — the current documents have zero causal or comparative sentences. Fires only when a distributor crosses the threshold from Design Decisions #1 / #1a.

```
Distributor [Name] showed a significant deviation on [Metric] in [Period]: [value] vs. the state average of [state_value], a gap of [X] [percentage points | percent].
```

Example, using real Maharashtra August 2024 WD-level data (from the validated `kpi_wd_month.csv`):

```
Distributor Boyd-White Distributors showed a significant deviation on Service Level in August 2024: 82.9% vs. the state average of 92.3%, a gap of 9.4 percentage points.
```

*(This is a real trigger — WD0001 in the sanity-check data showed Service Level 0.8268, which is 9.6pp below the 92.3% state average, comfortably past the 6pp threshold. Verify exact WD name-to-ID mapping before finalizing this as an actual document sentence — the sanity check used `wd_id`, not confirmed `wd_name`.)*

**No-anomaly sentence (mandatory, per Design Decision #2):**

```
No distributor in [State] showed a significant deviation from the state average on [Metric] in [Period].
```

**Format rule:** write one no-anomaly sentence per metric that had zero triggers that month — not one blanket sentence for the whole document. This keeps each `Metric` node's absence-of-anomaly explicit and queryable, rather than bundling multiple metrics into one vague statement.

---

## 4. Causal-Language Guidance (Where the Underlying Mechanism Actually Supports It)

Per the dictionary's Confirmed-vs-Uncertain section: **no document today states any causal chain**, even though `build_orders()` genuinely does check `stockout_flag` to cap fulfilment. Since disruption severity/magnitude isn't persisted (Known Gap #5) and `SupplyDisruption` isn't yet a schema entity, causal sentences should be scoped narrowly for now:

**Do NOT write** (invents a magnitude/mechanism the data can't support):
```
Service Level dropped sharply in Assam because Distributor X had a severe supply disruption on Marlboro Pack 9.
```

**DO write** (states the deviation as fact, lets the graph's DEVIATES_FROM edge carry the reasoning weight, no invented mechanism):
```
Distributor X's Service Level in Assam, October 2025 was 9.6 percentage points below the state average — the largest deviation among Assam's 11 distributors this month.
```

If/when `stockout_flag` sentences are added in a future revision (requires a new document field, out of scope for this pass), *then* a genuine causal sentence becomes possible:
```
Distributor X experienced a supply disruption in October 2025, which coincided with its below-average Service Level that month.
```
Note "coincided with," not "because of" — the data confirms correlation via the shared mechanism, not a document-level proof of causation for that specific instance. Keep this distinction in the sentence itself, not just in this design doc, since the sentence is what the LLM will extract and the chatbot will eventually repeat.

---

## 5. Evidence / Citation Requirements

Every `Observation` node must be traceable back to (a) which document, and (b) which sentence. Two changes from the current format:

1. **Filename stays the primary document-level citation** (already true — `graphrag_docs_manifest.csv` is confirmed as the canonical map, dictionary Section 19 point 7). No change needed here.
2. **New: each KPI sentence should be independently quotable.** Since one-KPI-one-sentence is already the pattern (Section 2 above), this is mostly already satisfied — the recommendation is to keep sentences short enough that a citation can reference "the sentence containing X," not "the paragraph containing X," so `EVIDENCED_BY.source_sentence` (schema Section 4) is a single clean sentence, not an ambiguous multi-KPI block.

---

## 6. Worked Example — Full Rewrite, Maharashtra August 2024

**BEFORE** (original document, stale data, no anomaly detail, no entity tagging):

> This document reports Sales & Distribution performance for Maharashtra in August 2024.
>
> In Maharashtra during August 2024, Productivity (share of sales visits that resulted in an order) was 79.0%, and the average Service Level (share of ordered quantity actually delivered) was 92.4%. Sales executives averaged 4.03 SKUs per transaction and a Dropsize of 65.79 units per productive visit. [...]
>
> Maharashtra is served by 11 Wholesale Distributor(s): Boyd-White Distributors, Larson-Hernandez Distributors, [...]

**AFTER** (new template, current verified data, entity tags, anomaly section):

> **State:** Maharashtra
> **Period:** August 2024
>
> This document reports Sales & Distribution performance for Maharashtra in August 2024.
>
> Productivity for Maharashtra in August 2024 was 79.0%. Service Level for Maharashtra in August 2024 was 92.3%. SKUs per Transaction for Maharashtra in August 2024 was `[current value — verify against kpi_state_month.csv]`. Dropsize (units ordered per productive visit — definition pending GPIL confirmation) for Maharashtra in August 2024 was 200.96.
>
> **Distributors serving Maharashtra:** Boyd-White Distributors, Larson-Hernandez Distributors, Dorsey LLC Distributors, [... unchanged list ...]
>
> **Distributor-level deviations, August 2024:**
> Distributor Boyd-White Distributors showed a significant deviation on Service Level in August 2024: 82.9% vs. the state average of 92.3%, a gap of 9.4 percentage points.
> No distributor in Maharashtra showed a significant deviation from the state average on Productivity in August 2024.
> No distributor in Maharashtra showed a significant deviation from the state average on Dropsize in August 2024.
>
> [... category paragraphs unchanged in structure, per Section 2's rule to preserve what already works ...]

---

## 7. What Changed vs. What Stayed the Same

| Element | Status |
|---|---|
| One-KPI-one-sentence phrasing | **Kept** — dictionary confirmed this already works |
| Category paragraph structure (4 categories, same order) | **Kept** |
| Entity tag lines (State/Period labels) | **New** |
| "Distributor" naming (never "WD") | **New enforcement** |
| Metric confirmation-status qualifier (3 flagged KPIs only) | **New** |
| Distributor-level deviation sentences | **New** — this is the section that makes "which distributor caused the decline" answerable |
| Explicit no-anomaly sentences | **New** |
| Causal ("because") language | **Deliberately still absent** — see Section 4; adding it now would overstate what the data supports |

---

*Next step: this template gets handed to Claude Code to (a) implement in `build_documents.py`, (b) regenerate a small pilot set (5-10 documents, not all 672), and (c) run extraction against the pilot before committing to a full regeneration + index.*
