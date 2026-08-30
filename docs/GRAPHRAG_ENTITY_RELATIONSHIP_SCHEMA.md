# GPIL S&D — GraphRAG Entity/Relationship Schema

**Purpose:** the extraction-ready ontology derived from `BUSINESS_KNOWLEDGE_DICTIONARY.md`. Where the dictionary explains *what exists and why*, this document specifies *exactly what GraphRAG should extract and how* — this is what feeds the custom entity extraction prompt.

**Derivation rule followed throughout:** an entity only appears below as a NODE if the dictionary (a) recommended NODE, and (b) confirmed it currently appears in `graphrag_input/*.txt` document text, OR is planned to appear once the document template is updated (WD-category detail, per Design Decision #1). Entities recommended as PROPERTY, or NODE-but-not-yet-in-text (Zone, SE, Outlet, Channel Type, Outlet Tier, SKU, Hero SKU, Visit, Order, Inventory Snapshot, Supply Disruption-as-event) are excluded from the extraction schema — including them would ask the LLM to extract things that are not written in the text, which cannot work regardless of prompt quality (Known Gap #2).

---

## 1. Entity Types (Nodes)

| # | Entity Type | Identifying Property | Source in Document Text | Status |
|---|---|---|---|---|
| 1 | `State` | `state_name` | Opening sentence of every document | Current |
| 2 | `Period` | `year_month` (e.g. "2024-08") | Every document's opening sentence, in word form ("August 2024") | Current (formalize as node — currently just a prose string) |
| 3 | `Distributor` | `wd_id`, `wd_name` | Named in the "served by N distributors" list | Current (name-only today) |
| 4 | `Category` | `category_name` | One paragraph per category, 4 per document | Current |
| 5 | `Franchise` | `franchise_name` | Listed inside each category's paragraph | Current (name-only, no franchise-level number) |
| 6 | `Observation` | composite: metric + scope + period | Not yet explicit — every KPI number in the document is implicitly one of these | New (to formalize) |
| 7 | `Metric` (KPI type) | `metric_name` | Implicit in every KPI sentence | New (to formalize) |
| 8 | `Document` | `filename` | n/a (the document IS this node) | Current |

**Explicitly renamed:** `WD` → `Distributor` in this schema, per the dictionary's naming-collision warning (WD vs. the "Dealer" channel-type value). Every extraction prompt and schema reference must use `Distributor`, never `WD`, to avoid the LLM conflating it with the Dealer channel.

**Deliberately excluded from node extraction (PROPERTY-only or not-in-text):** Zone, Sales Executive, Outlet, Channel Type, Outlet Tier, SKU, Hero SKU, Visit, Order/Order Line, Inventory Snapshot, Supply Disruption-as-materialized-event. See Known Gap #2 in the dictionary — these have zero current textual presence, so extracting them is not an extraction-prompt problem to solve, it's a document-content problem not yet solved.

---

## 2. Relationship Types (Edges)

Only relationships that are (a) directly stated in current document text, or (b) will be stated once the WD-category anomaly detail is added (Design Decision #1). Relationships that are real in the underlying data but never appear in text (e.g. `SE → Visit`, `Order Line → SKU`) are excluded for the same reason as their entities above.

| # | Relationship | From → To | Business Meaning | Cardinality | Extraction Source |
|---|---|---|---|---|---|
| 1 | `HAS_OBSERVATION` | `State` → `Observation` | A state has a measured KPI fact for a period | 1 : many | Direct — every state-grain KPI sentence |
| 2 | `HAS_OBSERVATION` | `Distributor` → `Observation` | A distributor has a measured KPI fact for a period (**new**, once WD-category anomaly sentences are added) | 1 : many | Direct, once document template updated |
| 3 | `SCOPED_TO_CATEGORY` | `Observation` → `Category` | The observation is specific to one product category | many : 1 | Direct — category-paragraph KPI sentences |
| 4 | `OF_METRIC` | `Observation` → `Metric` | Which KPI this number is | many : 1 | Direct — implicit in every KPI sentence, made explicit via entity tagging (see Document Structure) |
| 5 | `OBSERVED_IN` | `Observation` → `Period` | When the observation applies | many : 1 | Direct — every document is single-period |
| 6 | `SERVES` | `Distributor` → `State` | A distributor operates in this state | many : 1 | Direct — the "served by N distributors" sentence |
| 7 | `CONTAINS` | `Category` → `Franchise` | Category groups franchises | 1 : many | Direct — franchise list inside category paragraph |
| 8 | `EVIDENCED_BY` | `Observation` → `Document` | Citation — this number's exact source | many : 1 | Direct — every document is the source of its own observations |
| 9 | `PRECEDES` | `Period` → `Period` | Enables month-over-month trend traversal | 1 : 1 (chain) | **Derived**, not extracted from text — inject directly at indexing time, not via LLM extraction (see note below) |
| 10 | `DEVIATES_FROM` | `Distributor` → `Observation` (state-level) | A distributor's value differs from its state average beyond the agreed threshold (Design Decision #1, extended below) | many : 1 | Direct, once anomaly-triggered sentences are added — this is the causal-reasoning-critical edge |

**Note on `PRECEDES`:** this is the one relationship type that should **not** be left to LLM extraction. It's a deterministic fact (Period X precedes Period Y) fully derivable from the `year_month` string — computing and inserting it directly during graph construction is more reliable and cheaper than hoping the LLM infers chronological order from 672 independent documents. Flag this for the indexing-pipeline design step, not the extraction-prompt step.

---

## 3. Node Properties

| Node Type | Properties |
|---|---|
| `State` | `state_name` |
| `Period` | `year_month`, `month_word_form` (e.g. "August 2024") |
| `Distributor` | `wd_id`, `wd_name`, `state_name` |
| `Category` | `category_name` |
| `Franchise` | `franchise_name`, `category_name` |
| `Metric` | `metric_name`, `formula_summary`, `confirmation_status` ("confirmed" / "needs GPIL confirmation" — surfacing the dictionary's ⚠️ flags on Dropsize, ACV tier-weights, and Range Billing directly into the graph) |
| `Observation` | `value`, `scope` ("state" or "distributor"), `category` (nullable), `as_of_period`, `source_document_id` |
| `Document` | `filename`, `state_name`, `year_month` |

---

## 4. Edge Properties

| Edge Type | Properties |
|---|---|
| `HAS_OBSERVATION` | `as_of_period` |
| `DEVIATES_FROM` | `direction` ("worse"/"better"), `gap_percentage_points` — directly supports "which distributor caused the decline" questions with a quantified answer, not just a name |
| `EVIDENCED_BY` | `source_document_id`, `source_sentence` (not just filename — supports sentence-level citation) |
| `SCOPED_TO_CATEGORY` | none needed beyond the edge itself |
| `PRECEDES` | none — purely structural |

---

## 5. What This Schema Deliberately Cannot Answer Yet

Carried forward from the dictionary's Known Gaps, stated here in schema terms so it isn't lost when this becomes an extraction prompt:

- **No `SupplyDisruption` node type exists in this schema.** `stockout_flag` is a real, confirmed mechanism (dictionary Known Gap #5) but is not in any document text today. If disruption sentences are added later, this schema will need a `SupplyDisruption` entity type and an `AFFECTS` edge to `Observation` — deferred, not designed here, so it isn't half-specified.
- **No `SKU`-level or `Franchise`-level `Observation` exists** — Franchises are named but never measured (dictionary confirms zero franchise-level KPI numbers in text). A "why did this specific SKU underperform" question cannot be answered by this schema.
- **`DEVIATES_FROM` now covers both WD-category and WD-month KPIs** — extended below (Design Decision #1a).

---

## 6. Design Decision #1a — Extending the Anomaly Threshold to WD-Month KPIs

Design Decision #1 (in the dictionary) set a ±6 percentage-point threshold for the four **category-level** WD KPIs (ND, ACV, OOS%, Range Billing) — all of which are ratios/percentages, so "percentage points" applies directly.

This has now been extended to the three **WD-month** KPIs (Productivity, Service Level, Dropsize), with one adjustment: **Dropsize is not a percentage — it's an absolute unit count** (e.g. 200.96 units/productive visit), so a "percentage-point" threshold doesn't apply to it directly. The rule is therefore:

| Metric | Threshold Type | Trigger |
|---|---|---|
| Productivity | Percentage points (absolute) | Distributor's value differs from state average by more than **6 percentage points** |
| Service Level | Percentage points (absolute) | Distributor's value differs from state average by more than **6 percentage points** |
| Dropsize | **Percent (relative) deviation** | Distributor's value differs from state average by more than **6% relative** (e.g. if state Dropsize is 200, trigger below 188 or above 212) |
| ND / ACV / OOS% / Range Billing | Percentage points (absolute) | Tightened from Design Decision #1's 6pp to 10pp (see `PP_DEVIATION_THRESHOLD` in `src/graph/build_documents.py`) — the 7-document pilot showed too many call-outs per document at 6pp; unfavorable direction only |

Same tunability caveat as Design Decision #1 applies: this is an agreed default, not a code- or statistics-derived value, and should be revisited once pilot documents show how many distributors it flags per state/month.

---

*Next step: Document Structure Recommendation (Section 5 of the original brief) — designs the actual sentence templates that produce the entity tags and relationships specified above.*
