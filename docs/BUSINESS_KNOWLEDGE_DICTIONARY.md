# GPIL S&D — Draft Business Knowledge Dictionary

**Purpose:** foundation reference for the knowledge-graph ontology. Every claim below is traced to a specific file, function, and (where useful) line range in this repository as it exists on disk today. Nothing here describes what a "typical" FMCG/tobacco distributor *should* look like — only what `src/data_gen/generate_synthetic_data.py`, `src/data_gen/reassign_outlet_tiers.py`, `src/kpis/compute_kpis.py`, `src/graph/build_documents.py`, and the CSV/`.txt` files they produce actually implement.

**Files read to build this document:** `src/data_gen/generate_synthetic_data.py` (1079 lines), `src/data_gen/reassign_outlet_tiers.py` (180 lines), `src/kpis/compute_kpis.py` (365 lines), `src/graph/build_documents.py` (285 lines), `config/settings.py`, `src/reports/generate_pan_india_summary.py`, plus the actual CSV headers on disk and one sample document (`data/graphrag_input/Karnataka_2025-03.txt`).

**A finding that shapes almost every recommendation below:** the 672 narrative `.txt` documents in `data/graphrag_input/` — the *only* corpus GraphRAG will ever read — mention exactly five kinds of thing by name: **State**, **Month** (in prose, e.g. "March 2025"), **Distributor (WD) names**, **Category**, and **Franchise names**. Confirmed by reading `render_document()` in `src/graph/build_documents.py:138-210` line by line: it touches `state_row` (6 state-grain KPI values), `wd_lookup` (WD *names* only, no numbers attached), and `category_rows` + `franchise_lookup` (4 category-grain KPI values + franchise names). **Zone, Sales Executive, Outlet, Channel Type, Outlet Tier, SKU, Visit, Order, and Inventory Snapshot are never named or numerically described in any document's text.** They exist as rich, well-engineered structure in the underlying CSVs, but a GraphRAG entity extractor — however well-prompted — can only extract what is written in the text it's given, so none of those entities can become graph nodes from the current corpus without first enriching the documents. This is not a guess; it is a direct reading of the rendering function's field list.

---

## 1. Entity Dictionary

### State
- **Business meaning:** GPIL's top geography level — one row per Indian state.
- **Exact source:** `data/geography.csv`, rows where `unit_type == "State"`. Built by `build_geography()` in `generate_synthetic_data.py:364-412`.
- **Unique identifier:** `unit_id` (format `ST01`…`ST28`); `unit_name`/`state_name` is the human-readable name and is what's actually used as the join key everywhere downstream (outlets, KPIs, documents all key on `state_name`, not `unit_id`).
- **Important properties:** `unit_name` (= state name), `state_name` (redundant self-reference on the State row itself). Real 2024 population and GSDP figures exist as a hard-coded Python dict (`STATE_POPULATION_GDP`, lines 297-326) used only to derive an in-memory `state_factor` — **not written to `geography.csv` or any output table.**
- **In current GraphRAG documents?** Yes — every document opens by naming the state (`build_documents.py:160-163`).
- **Recommendation:** **NODE.** It's the primary grain of both KPI tables and every document; every other entity in the system is reachable from it.

### Zone
- **Business meaning:** A sales-management subdivision of a state (2 fixed per state).
- **Exact source:** `geography.csv`, rows where `unit_type == "Zone"`, `unit_name` format `"<State> Zone <n>"`. Built in the same `build_geography()` loop, lines 382-389.
- **Unique identifier:** `unit_id` (format `ZN0001`…).
- **Important properties:** `parent_unit_id` (→ State's `unit_id`), `state_name` (denormalized).
- **In current GraphRAG documents?** No — never named or referenced in any document.
- **Recommendation:** **PROPERTY on WD** (or a thin NODE kept only for hierarchy completeness). No KPI or document currently targets Zone grain, so as a graph node today it would be a dead end — reachable from nothing, leading nowhere. Worth revisiting only if zone-level KPIs or documents get built later.

### Distributor / WD (Wholesale Distributor)
- **Business meaning:** The entity that physically holds inventory and whose sales force (SEs) covers outlets.
- **Exact source:** `geography.csv`, rows where `unit_type == "WD"`, `unit_name` is a Faker-generated company name + " Distributors" (line 396). Also the join key for `inventory_snapshots.csv.wd_id` and `orders.csv.wd_id`.
- **Unique identifier:** `unit_id` (format `WD0001`…).
- **Important properties:** `parent_unit_id` (→ Zone), `state_name` (denormalized). No onboarding date, credit terms, or contact info exist anywhere.
- **In current GraphRAG documents?** Partially — WD **names** are listed per state (`build_wd_lookup()`, `build_documents.py:109-119`; rendered at lines 180-189), but **no WD-level KPI number is ever attached to a WD name in any document.** The document only says "State X is served by N distributors: A, B, C…" — a bare name list.
- **Recommendation:** **NODE.** Already has a name-level presence in every document, and is the natural target for future WD-level KPIs (see Known Gaps). Rename to `Distributor` in the ontology — see the naming-collision note under Channel Type below.

### Sales Executive (SE)
- **Business meaning:** A field salesperson who owns a fixed set of outlets ("beat").
- **Exact source:** `geography.csv`, rows where `unit_type == "SE"`, `unit_name` = Faker person name (line 406). Also `visits.csv.se_id` and `outlets.csv.se_id`.
- **Unique identifier:** `unit_id` (format `SE00001`…).
- **Important properties:** `parent_unit_id` (→ WD), `state_name` (denormalized). No tenure, target, or performance field exists.
- **In current GraphRAG documents?** No. `compute_productivity()` (`compute_kpis.py:108-124`) groups visits by `(state_name, month)` only — `se_id` is read from `visits.csv` but never used as a KPI grouping key, and no document text ever names an SE.
- **Recommendation:** **PROPERTY** for now (not a node) — with zero textual presence and zero SE-grain KPI, a node would have no edges into anything GraphRAG can retrieve. Reconsider if SE-level KPIs and documents are built (Known Gaps #1).

### Outlet
- **Business meaning:** The retail point of sale.
- **Exact source:** `data/outlets.csv`. Built by `build_outlets()`, `generate_synthetic_data.py:420-496`.
- **Unique identifier:** `outlet_id` (format `OUT000001`…).
- **Important properties:** `se_id`, `wd_id`, `zone_id`, `state_id`, `state_name` (all denormalized — no join needed to find an outlet's full geography path), `channel_type`, `outlet_tier`, `onboarded_date`, `is_active`, `closure_date`.
- **A nuance worth flagging:** `outlet_tier` as it exists in the *final* `outlets.csv` is not the value `build_outlets()` originally assigned. It is **overwritten** by `src/data_gen/reassign_outlet_tiers.py` (see its own entity note below and Known Gaps #6) — the value on disk is a *performance-derived* label, not the birth-time random draw, but nothing in `outlets.csv` records that distinction or preserves the original label.
- **In current GraphRAG documents?** No — no individual outlet is ever named or described in any document.
- **Recommendation:** **PROPERTY** for the current corpus (~120K individual records with zero textual grounding would be pure graph bloat with no retrievable evidence). Would become node-worthy only alongside outlet-level or outlet-tier-level documents.

### Channel Type
- **Business meaning:** The kind of retail business an outlet is — one of `Retail`, `Hawkers`, `Modern Trade`, `Dealer` (`CHANNEL_TYPES`, `generate_synthetic_data.py:81`).
- **Exact source:** `outlets.csv.channel_type`. Assigned once at outlet creation (`build_outlets()`, line 465) by weighted random draw from `CHANNEL_WEIGHTS_BY_COUNT` — never changes afterward.
- **Unique identifier:** the string value itself (no code table — 4 fixed string literals).
- **Important properties:** channel drives (a) which categories an outlet may sell, via the hard-coded `CHANNEL_CATEGORY_ELIGIBILITY` dict (`generate_synthetic_data.py:144-149`, **not stored in any CSV**), and (b) order-quantity size, via `CHANNEL_BASE_MULTIPLIER` (`generate_synthetic_data.py:132-137`, also **code-only, not stored**).
- **Naming collision to flag explicitly:** "WD" (a geography node) and "Dealer" (one of the four `channel_type` values) sound alike but are unrelated concepts — a WD is a distribution *partner*, a Dealer is a large-format *retail shop* at the bottom of the tree. If this distinction isn't made explicit in any future extraction prompt, an LLM is likely to conflate them.
- **In current GraphRAG documents?** **No** — never mentioned in any document, despite being one of the two axes the entire 12-way sales cascade business rule is built around.
- **Recommendation:** **PROPERTY today** given zero textual presence; the underlying business logic (eligibility, multipliers) argues strongly for eventually becoming a **NODE** if channel-level KPIs/documents are ever produced, since "which channel underperforms" is exactly the kind of traversal question a node enables and a property doesn't.

### Outlet Tier
- **Business meaning:** Gold/Silver/Bronze — a *within-channel* performance rank (`OUTLET_TIERS`, line 165).
- **Exact source:** `outlets.csv.outlet_tier`. Originally assigned randomly at outlet birth by `build_outlets()` (`TIER_WEIGHTS_BY_COUNT`, line 466), then **overwritten** post-hoc by `reassign_outlet_tiers.py::assign_tier_by_rank()` (lines 86-108) based on each outlet's actual summed `qty_delivered` from `orders.csv`, ranked within its `channel_type`, cut at the same 20%/35%/45% percentile boundaries (`TIER_WEIGHTS_BY_COUNT`, reused directly from `generate_synthetic_data.py` to avoid drift — line 46 import). Tiers are **never** compared across channels — a Bronze Dealer can still outsell a Gold Retail shop by construction (Rule below).
- **Unique identifier:** the string value itself (`Gold`/`Silver`/`Bronze`).
- **Important properties:** drives ACV's tier weight (`OUTLET_TIER_WEIGHT`, `compute_kpis.py:61`, Gold=3/Silver=2/Bronze=1 — explicitly flagged `NEEDS GPIL CONFIRMATION`) and order quantity via `TIER_BASE_MULTIPLIER` (`generate_synthetic_data.py:185-189`).
- **In current GraphRAG documents?** No — never mentioned in any document text, even though ACV (which uses tier weighting) is reported per category.
- **Recommendation:** **PROPERTY today**, same reasoning as Channel Type — real business signal, zero corpus visibility currently.

### Category
- **Business meaning:** Top product grouping — `GPI`, `IPM`, `Ferrero`, `Candy` (`CATEGORY_FRANCHISES` keys, line 235).
- **Exact source:** `products.csv.category_name`. There is no separate Category table/ID — category is purely a repeated string column on every SKU row.
- **Unique identifier:** the string value itself (no surrogate key exists anywhere in the codebase).
- **Important properties:** target unit-volume share (`CATEGORY_WEIGHTS_BY_VOLUME`, line 251, code-only, not stored) and channel eligibility (see Channel Type).
- **In current GraphRAG documents?** Yes — one full paragraph per category per document, with 4 KPI numbers each (`render_document()`, lines 194-208).
- **Recommendation:** **NODE.** Directly named with attached numbers in every document — a clean extraction target.

### Franchise
- **Business meaning:** A brand family under a category (e.g. Marlboro, TicTac, Kinder_Joy, GPI_Franchise_1…6).
- **Exact source:** `products.csv.franchise_name`, same structure as Category — a repeated string column, no separate table.
- **Unique identifier:** the string value itself.
- **Important properties:** only 3 of 11 franchise names are real brands (`Marlboro`, `TicTac`, `Kinder_Joy` — explicitly named in `CATEGORY_FRANCHISES`, lines 236-239); the rest (`GPI_Franchise_1`–`6`, `Candy_Franchise_1`–`2`) are generic placeholders.
- **In current GraphRAG documents?** Yes — every category paragraph lists its franchises by name (`build_franchise_lookup()`, `build_documents.py:122-135`, rendered line 198-199), but **with no KPI number attached to the franchise itself** — franchises are named, not measured, in the current documents.
- **Recommendation:** **NODE.** Named consistently across every document (verbatim string reuse is explicit project policy — see module docstring lines 31-35), even without its own KPI value yet.

### SKU
- **Business meaning:** One sellable product variant ("pack") under a franchise, with its own price.
- **Exact source:** `products.csv`. Built by `build_products()`, `generate_synthetic_data.py:503-532`.
- **Unique identifier:** `sku_id` (format `SKU0001`…).
- **Important properties:** `sku_name` (= `"{franchise_name} Pack {n}"`), `pack_size` (a text label `"Variant N"`, **not** an actual quantity — stated explicitly, not inferred), `unit_price`, `launch_date`, `is_active`.
- **In current GraphRAG documents?** No — SKUs are never individually named anywhere in the document text; only their parent Franchise is.
- **Recommendation:** **PROPERTY** given zero corpus visibility. Even structurally, 63 SKUs is small enough that a node would be cheap, but there is currently nothing in the text for GraphRAG to attach it to.

### Hero SKU (informal, code-only classification)
- **Business meaning:** 5 of the 63 SKUs (the flagship "Pack 1" of 4 of GPI's 6 franchises, plus Marlboro's flagship) get a 25× pick-weight boost (`HERO_SKU_WEIGHT_BOOST`, line 198) and 1.4× quantity boost (`HERO_SKU_QTY_MULTIPLIER`, line 199) on every order line.
- **Exact source:** **Nowhere in any output file.** Computed at generation time by `select_hero_skus()` (`generate_synthetic_data.py:535-555`) and held only in memory during `build_orders()`. `products.csv` has no `is_hero` column — confirmed by reading the exact column list written in `build_products()` (lines 521-530).
- **Unique identifier:** n/a — not a persisted entity.
- **Important properties:** n/a.
- **In current GraphRAG documents?** No.
- **Recommendation:** **Not currently representable as a node or property at all** — the underlying fact is deterministically re-derivable (rerunning `select_hero_skus()` on the current `products.csv` gives the same 5 SKUs, since it always picks the first-by-`sku_id` pack of the same 5 franchises), but it does not exist as data today. Flagged in Known Gaps.

### Visit
- **Business meaning:** One SE call on one outlet, on one date, with an outcome (`Order Placed` / `No Order` / `Closed`).
- **Exact source:** `data/visits.csv`. Built by `build_visits()`, `generate_synthetic_data.py:615-730`.
- **Unique identifier:** `visit_id` (format `VIS0000001`…).
- **Important properties:** `visit_date`, `se_id`, `outlet_id`, `visit_outcome`.
- **In current GraphRAG documents?** No individual visit is named; only the aggregate Productivity ratio derived from visits appears (`compute_productivity()`, `compute_kpis.py:108-124`).
- **Recommendation:** **PROPERTY** (rolled into the Productivity KPI). No individual-visit granularity survives into the graph-facing layer.

### Order / Order Line
- **Business meaning:** A basket of SKUs an outlet buys on one visit. **Note the physical table is one row per SKU line, not one row per order** — `order_id` repeats across the 1-6 rows of a single order.
- **Exact source:** `data/orders.csv`. Built by `build_orders()`, `generate_synthetic_data.py:739-917`.
- **Unique identifier:** `order_id` identifies the logical order (repeats across lines); `line_id` is the unique row-level key (format `LN0000001`…).
- **Important properties:** `visit_id` (back-reference to the triggering visit), `order_date`, `outlet_id`, `wd_id` (fulfilling distributor), `sku_id`, `qty_ordered`, `qty_delivered`, `unit_price`.
- **In current GraphRAG documents?** No individual order is named; only aggregates (SKUs/Transaction, Service Level, Dropsize, and category-level ND/ACV/Range Billing) survive into document text.
- **Recommendation:** **PROPERTY** (rolled into multiple KPIs). At ~54M rows in the current data, this is also the practical reason it can't become a node set directly.

### Inventory Snapshot
- **Business meaning:** A WD's monthly stock position for one SKU (opening/received/sold/closing).
- **Exact source:** `data/inventory_snapshots.csv`. Built by `build_inventory()`, `generate_synthetic_data.py:564-608`.
- **Unique identifier:** no single ID column — natural key is `(wd_id, sku_id, snapshot_month)`.
- **Important properties:** `opening_stock`, `qty_received`, `qty_sold`, `closing_stock`, `stockout_flag` (boolean).
- **In current GraphRAG documents?** No individual snapshot is named; only aggregates (Inventory Turns/Days at state grain, OOS% at state×category grain) survive into text.
- **Recommendation:** **PROPERTY** (rolled into KPIs).

### Supply Disruption (implicit — not a separate table)
- **Business meaning:** A WD×SKU×month where incoming supply was cut sharply, constraining fulfilment.
- **Exact source:** `inventory_snapshots.csv.stockout_flag == True`. Set in `build_inventory()`, line 604: `True` if either (a) that WD×SKU×month independently rolled a disruption (`STOCKOUT_EVENT_PROB = 0.08`, line 339) or (b) `closing_stock < 5` regardless of cause. **There is no independent record of magnitude** — `qty_received` is stored, but the counterfactual "what it would have been without disruption" (`base_supply`) is never persisted, so severity cannot be reconstructed from data alone, only the boolean fact of disruption.
- **Unique identifier:** n/a — it's a boolean property, not a materialized entity with its own ID.
- **In current GraphRAG documents?** **No.** The document-rendering code never reads `stockout_flag` — confirmed by its absence from `render_document()`'s field list. This means the single mechanism the generator uses to create believable Service-Level/OOS variation is completely invisible to GraphRAG's text corpus today, even though `build_orders()` internally checks this exact flag (`inv_lookup.get(...)`, line 885) to decide fulfilment.
- **Recommendation:** Currently **not representable** in the graph at all from the existing documents. If a future document revision adds a sentence like "Distributor X had a supply disruption on SKU Y this month," this should become an explicit event-type **NODE** (`SupplyDisruption`) connecting a WD, a SKU, and a Period — but that requires writing new document content first, not just a better extraction prompt.

### KPI Record — State × Month
- **Business meaning:** A derived summary of a state's overall S&D health for one month (category-agnostic).
- **Exact source:** `data/kpi_state_month.csv`. Built by `build_kpi_state_month()`, `compute_kpis.py:198-215`.
- **Unique identifier:** natural key `(state_name, month)`.
- **Important properties:** `productivity`, `skus_per_transaction`, `service_level`, `dropsize`, `inventory_turns`, `inventory_days` — see Section 3 for exact formulas.
- **In current GraphRAG documents?** Yes — this is the source of the entire second paragraph of every document.
- **Recommendation:** These six values should become **typed `Observation` nodes** (value + metric-type + state + period), not just numbers inside prose — a plain sentence like "Productivity was 46.2%" is easy to extract as a State attribute but hard to later compare across months/states unless the number itself is a queryable node.

### KPI Record — State × Month × Category
- **Business meaning:** A derived summary of one category's distribution health in a state/month.
- **Exact source:** `data/kpi_state_month_category.csv`. Built by `build_kpi_state_month_category()`, `compute_kpis.py:323-337`.
- **Unique identifier:** natural key `(state_name, month, category_name)`.
- **Important properties:** `numeric_distribution`, `acv`, `oos_pct`, `range_billing`.
- **In current GraphRAG documents?** Yes — source of each category paragraph.
- **Recommendation:** Same as above — `Observation` nodes, scoped additionally to Category.

### KPI Record — WD × Month
- **Business meaning:** A derived summary of one distributor's overall operational health for a month (category-agnostic).
- **Exact source:** `data/kpi_wd_month.csv`. Built by `build_kpi_wd_month()`, `compute_kpis.py:442-465`.
- **Unique identifier:** natural key `(wd_id, month)`.
- **Important properties:** `productivity`, `skus_per_transaction`, `service_level`, `dropsize`, `inventory_turns`, `inventory_days` — same six formulas as the State-grain version (Section 3), computed by the `compute_*_wd()` sibling functions (`compute_kpis.py:357-440`), just grouped by `wd_id` instead of `state_name`. `wd_name` and `state_name` are also carried as descriptive context columns (not part of the grouping grain).
- **In current GraphRAG documents?** No — not yet. This table is new; `build_documents.py` has not been touched to read from it (out of scope for this work).
- **Recommendation:** These six values should become **typed `Observation` nodes**, scoped to **Distributor** instead of State — same reasoning as the State-grain `KPI Record — State × Month` entry above.

### KPI Record — WD × Month × Category
- **Business meaning:** A derived summary of one distributor's distribution health for one category in one month.
- **Exact source:** `data/kpi_wd_month_category.csv`. Built by `build_kpi_wd_month_category()`, `compute_kpis.py:554-573`.
- **Unique identifier:** natural key `(wd_id, month, category_name)`.
- **Important properties:** `numeric_distribution`, `acv`, `oos_pct`, `range_billing` — same four formulas as the State-grain version (Section 3), grouped by `wd_id` instead of `state_name`.
- **In current GraphRAG documents?** No — not yet, same as above.
- **Recommendation:** Same as above — `Observation` nodes, scoped to Distributor and Category.

### Period / Month
- **Business meaning:** The time grain both KPI tables and every document are anchored to.
- **Exact source:** Stored as a `"YYYY-MM"` string (e.g. `"2025-03"`) in both KPI CSVs (`_month_start()`, `compute_kpis.py:84-92`, then cast with `.astype(str)`). Rendered into prose as `"March 2025"` in documents by `_month_to_words()`, `build_documents.py:60-70`, specifically so the LLM extractor sees unambiguous words rather than a string that could be misread as a plain number.
- **Unique identifier:** the `"YYYY-MM"` string itself.
- **Important properties:** none beyond the string value — no explicit `Period` table exists; every table just repeats the string.
- **In current GraphRAG documents?** Yes — every document's opening sentence.
- **Recommendation:** **NODE.** Currently only a string inside prose; formalizing it lets trend/comparison questions ("was Productivity in March higher than February?") traverse a `PRECEDES`/`FOLLOWS` chain directly instead of re-parsing 24 separate files per state.

### Document
- **Business meaning:** One narrative `.txt` file per State×Month — the atomic unit GraphRAG will actually index.
- **Exact source:** `data/graphrag_input/<State>_<YYYY-MM>.txt`, one per row of `kpi_state_month.csv` (672 total, confirmed by counting `data/graphrag_docs_manifest.csv`: 673 lines including header = 672 documents). Built by `build_all_documents()`, `build_documents.py:213-269`.
- **Unique identifier:** `filename` (also the manifest's key column). Filename encodes state+month but uses `YYYY-MM` (not word form) specifically so files sort chronologically (`_safe_filename()`, lines 73-83).
- **Important properties:** `state_name`, `month` (both also recorded in `graphrag_docs_manifest.csv`).
- **Recommendation:** **NODE**, kept specifically so every extracted fact can cite back to the exact source document — this is the mechanism any "never answer without a source" requirement would rely on. `graphrag_docs_manifest.csv` is already the canonical filename→(state, month) lookup and should stay authoritative rather than being replaced.

---

## 2. Relationship Dictionary

| Source → Target | Relationship | Business meaning | Direction | Cardinality | Direct in data, or Derived? |
|---|---|---|---|---|---|
| State → Zone | CONTAINS | State divided into zones | State→Zone | 1 : 2 (fixed, `ZONES_PER_STATE`) | **Direct** — `geography.csv.parent_unit_id` |
| Zone → WD | CONTAINS | Zone served by several distributors | Zone→WD | 1 : 4–7 (`WD_PER_ZONE_RANGE`) | **Direct** — `parent_unit_id` |
| WD → SE | EMPLOYS | Distributor runs a field sales force | WD→SE | 1 : 12–22 (`SE_PER_WD_RANGE`) | **Direct** — `parent_unit_id` |
| SE → Outlet | COVERS | SE has a fixed assigned outlet list | SE→Outlet | 1 : 140–180 (`OUTLETS_PER_SE_RANGE`) | **Direct** — `outlets.csv.se_id` |
| Outlet → WD/Zone/State | LOCATED_IN | Denormalized geography, no join needed | Outlet→parent | many : 1 | **Direct** — `outlets.csv` carries `wd_id`, `zone_id`, `state_id`, `state_name` directly |
| Category → Franchise | CONTAINS | Category groups franchises | Category→Franchise | 1 : 1–6 | **Derived by grouping** — `products.csv` has no standalone Category or Franchise table; the mapping is implicit per-SKU-row (`category_name`, `franchise_name` columns) and must be aggregated to see it at the Category level |
| Franchise → SKU | CONTAINS | Franchise contains pack variants | Franchise→SKU | 1 : 3–8 | **Direct** per row, **derived to group** — same caveat as above |
| Channel Type → Category | ELIGIBLE_FOR | Which categories a channel may sell | Channel→Category | many : many | **Not in data at all** — hard-coded Python dict `CHANNEL_CATEGORY_ELIGIBILITY` (`generate_synthetic_data.py:144-149`), imported into `compute_kpis.py` (line 56) but never written to any CSV. Currently every channel is eligible for every category. |
| SE → Visit | PERFORMED | SE makes a sales call | SE→Visit | 1 : many | **Direct** — `visits.csv.se_id` |
| Visit → Outlet | TARGETS | Call is on a specific shop | Visit→Outlet | many : 1 | **Direct** — `visits.csv.outlet_id` |
| Visit → Order | RESULTS_IN | Only if outcome = "Order Placed" | Visit→Order | 1 : 0 or 1 | **Direct** — `orders.csv.visit_id` back-reference |
| Order → Order Line | CONTAINS | A basket has 1–6 SKU lines | Order→Line | 1 : many | **Direct** — same table, `order_id` repeats across `line_id` rows |
| Order Line → SKU | REFERENCES | Which product was ordered | Line→SKU | many : 1 | **Direct** — `orders.csv.sku_id` |
| Order Line → WD | SOURCED_FROM | Line fulfilled from this WD's stock | Line→WD | many : 1 | **Direct** — `orders.csv.wd_id`, denormalized from the outlet's WD at generation time (not independently validated against `outlets.csv` at query time — it's copied from the same lookup) |
| WD → SKU (via Inventory Snapshot) | HOLDS_STOCK_OF | Monthly stock position | WD→SKU | many : many, time-scoped | **Direct** — `inventory_snapshots.csv` keyed by `(wd_id, sku_id, snapshot_month)` |
| WD×SKU×Month → Disruption | EXPERIENCED | Supply shock that month | — | many : 1 (per month) | **Direct as a flag** (`stockout_flag`), but **not materialized as its own entity/edge with properties** — see Supply Disruption entity note |
| KPI Record (State×Month) → State + Month | DESCRIBES | The fact belongs to one state/month | Record→scope | many : 1 | **Direct** — natural key columns |
| KPI Record (State×Month×Category) → State + Month + Category | SCOPED_TO | Same, also category-scoped | Record→scope | many : 1 | **Direct** — natural key columns |
| WD → KPI Record (WD-grain) | DESCRIBES | The fact belongs to one distributor/month | WD→Record | many : 1 | **Direct** — natural key columns |
| Document → State + Month | REPORTS_ON | Narrative text is the human-readable form of the two KPI rows | Doc→scope | 1 : 1 | **Direct** — `build_all_documents()` iterates `kpi_state_month` rows 1:1 into filenames; confirmed by manifest join |
| Document → WD name(s) | MENTIONS | Document names WDs it has no numeric KPI for | Doc→WD | 1 : many | **Direct** — `build_wd_lookup()`, but note: name-only, no value attached |
| Document → Franchise name(s) | MENTIONS | Document names franchises it has category-level (not franchise-level) numbers for | Doc→Franchise | 1 : many | **Direct** — `build_franchise_lookup()`, name-only |
| Outlet → Outlet Tier | CLASSIFIED_AS | Performance rank, within-channel | Outlet→Tier | many : 1 | **Direct**, but the *value itself* is a **derived** quantity — see `reassign_outlet_tiers.py` note below |
| Outlet (channel, total_units) → Outlet Tier | RANKED_INTO | Tier assigned by percentile rank of summed `qty_delivered`, within channel | — | — | **Fully derived** — computed by `reassign_outlet_tiers.py::assign_tier_by_rank()` (lines 86-108) by aggregating `orders.csv`; the intermediate `total_units` value is never persisted (dropped before the final `outlets.csv` write, line 171) |

---

## 3. KPI Knowledge Model

All ten KPIs live in `src/kpis/compute_kpis.py`. Formulas below are transcribed directly from the code (variable names preserved).

### State × Month grain (`kpi_state_month.csv`, built by `build_kpi_state_month()`, lines 198-215)

**Productivity** — `compute_productivity()`, lines 108-124
```
productivity = productive_visits / total_visits
  where productive_visits = count(visit_outcome == "Order Placed")
  grouped by (state_name, month)
```
- Scope: State × Month.
- Interpretation: of every sales call an SE made in that state that month, what fraction resulted in an order.
- Placeholder flag: **none** — not flagged as unconfirmed in code.

**SKUs / Transaction** — `compute_order_level_kpis()`, lines 127-147
```
skus_per_transaction = total_order_lines / total_orders
  where total_order_lines = row count in orders.csv for that (state, month)
        total_orders      = nunique(order_id) for that (state, month)
```
- Scope: State × Month.
- Interpretation: how wide a basket the average order carries, in distinct SKU lines.
- Placeholder flag: **none.**

**Service Level** — same function, lines 127-147
```
service_level = total_qty_delivered / total_qty_ordered
```
- Scope: State × Month.
- Interpretation: of the quantity outlets asked for, how much actually got delivered.
- Placeholder flag: **none.**

**Dropsize** — `compute_dropsize()`, lines 150-169
```
dropsize = total_qty_ordered / productive_visits
  where productive_visits = count of visits with outcome "Order Placed"
  (total_qty_ordered summed from orders.csv, productive_visits from visits.csv)
```
- Scope: State × Month.
- Interpretation: average units ordered per successful (order-placed) call.
- Placeholder flag: **YES — explicitly "NEEDS GPIL CONFIRMATION" in the module docstring (lines 25-32) and the function's own docstring (lines 150-156).** The code is explicit that this is a volume metric (units), deliberately different from SKUs/Transaction (a line-count metric), and that "average case load" is defined differently by different companies.

**Inventory Turns** — `compute_inventory_kpis()`, lines 172-195
```
inventory_turns = total_qty_sold / total_avg_stock
  where avg_stock (per WD×SKU×month row) = (opening_stock + closing_stock) / 2
  rolled up WD→State via the WD's parent state
```
- Scope: State × Month (rolled up from WD×SKU×month grain).
- Interpretation: how many times distributor stock turned over that month.
- Placeholder flag: **none**, but note `days_in_month` is a fixed default of 30 (see Inventory Days) regardless of actual calendar month length.

**Inventory Days** — same function, lines 172-195
```
inventory_days = days_in_month / inventory_turns   (days_in_month defaults to 30)
```
- Scope: State × Month.
- Interpretation: same underlying signal as Inventory Turns, expressed as days of stock on hand.
- Placeholder flag: **none directly**, but the fixed 30-day assumption is a stated simplification (function signature default, line 172) — real months vary 28-31 days.

### State × Month × Category grain (`kpi_state_month_category.csv`, built by `build_kpi_state_month_category()`, lines 323-337)

**Numeric Distribution (ND)** — `compute_distribution_kpis()`, lines 240-273
```
numeric_distribution = billed_outlets / eligible_outlets
  where eligible_outlets = nunique(outlet_id) among outlets whose channel_type
                            is eligible for this category (CHANNEL_CATEGORY_ELIGIBILITY)
                            AND is_active == True
        billed_outlets   = nunique(outlet_id) that billed >=1 SKU of the
                            category that month
  grouped by (state_name, month, category_name)  [eligible_outlets computed
  per (state_name, category_name), joined in]
```
- Scope: State × Month × Category.
- Interpretation: of the shops that *could* stock this category, what percent actually billed something.
- Placeholder flag: **none directly**, but its eligible-outlet denominator inherits the untested `CHANNEL_CATEGORY_ELIGIBILITY` assumption (currently: every channel eligible for every category).

**ACV** — same function, lines 240-273
```
acv = billed_weight / eligible_weight
  where tier_weight = OUTLET_TIER_WEIGHT[outlet_tier]  (Gold=3, Silver=2, Bronze=1)
        billed_weight   = sum of tier_weight over outlets that billed the category
        eligible_weight = sum of tier_weight over all eligible outlets
```
- Scope: State × Month × Category.
- Interpretation: same ratio as ND, but weighted so bigger (Gold) outlets billing counts for more than small ones.
- Placeholder flag: **YES — explicit code comment, `OUTLET_TIER_WEIGHT` definition, lines 58-61:** "NEEDS GPIL CONFIRMATION: stand-in weights for ACV until we have real sales-volume weights per outlet... a common rough proxy, not a GPIL-confirmed number."

**Out-of-Stock % (OOS)** — `compute_oos_pct()`, lines 276-293
```
oos_pct = stockout_snapshots / total_snapshots
  where stockout_snapshots = count of inventory_snapshots rows with stockout_flag == True
        total_snapshots    = count of all inventory_snapshots rows
  grouped by (state_name, month, category_name), via SKU's category and WD's parent state
```
- Scope: State × Month × Category.
- Interpretation: what share of that category's WD×SKU×month stock positions were flagged as disrupted that month.
- Placeholder flag: **none.**

**Range Billing** — `compute_range_billing()`, lines 296-320
```
outlet_range_ratio = distinct_skus_billed / skus_in_category   (per outlet, per state/month/category)
range_billing = mean(outlet_range_ratio) across all outlets that billed
                anything in that category that month
```
- Scope: State × Month × Category.
- Interpretation: among shops that stock the category at all, how much of the full SKU range do they carry (assortment depth), as opposed to ND which measures how many shops stock it at all (assortment breadth).
- Placeholder flag: **YES — explicit code comment, module docstring lines 33-35 and function docstring lines 296-306:** "NEEDS GPIL CONFIRMATION on definition."

### Summary
Three of the ten KPIs — **Dropsize**, **ACV**'s tier weighting, and **Range Billing** — are explicitly marked `NEEDS GPIL CONFIRMATION` in code comments, not inferred by this analysis. All ten are computed at State×Month/State×Month×Category grain **and now also at WD×Month/WD×Month×Category grain** (`kpi_wd_month.csv`, `kpi_wd_month_category.csv` — see the two new KPI Record entities below); **SE-level KPI still does not exist anywhere in the codebase** (see Known Gaps #1).

`src/reports/generate_pan_india_summary.py` is a **separate script** producing units/revenue rollups (by category, state, franchise, channel×tier) for validation purposes — it is not part of the KPI/GraphRAG pipeline and does not define any of the ten KPIs above. Don't confuse its output with `compute_kpis.py`'s.

---

## 4. Confirmed vs Uncertain

### Confirmed by directly reading the code (file/function cited for each)
- The geography hierarchy is one flat table (`geography.csv`) distinguished by `unit_type`, not four separate tables — `build_geography()`, `generate_synthetic_data.py:364-412`.
- `outlets.csv` denormalizes the full geography path onto every outlet row — `build_outlets()`, lines 420-496.
- `products.csv` has no surrogate Category or Franchise ID — category/franchise are plain string columns — `build_products()`, lines 503-532.
- `CHANNEL_CATEGORY_ELIGIBILITY` currently allows every channel to sell every category — `generate_synthetic_data.py:144-149`.
- `outlet_tier` in the final `outlets.csv` is a performance-rank label computed from realized `qty_delivered`, overwriting the original random birth-time label — `reassign_outlet_tiers.py:86-108, 152-172`.
- Hero SKU status (5 of 63 SKUs) is never written to `products.csv` — confirmed by the exact column list in `build_products()`, lines 521-530, versus the in-memory-only `select_hero_skus()`, lines 535-555.
- `render_document()` (`build_documents.py:138-210`) never references `channel_type`, `outlet_tier`, `se_id`, `sku_id`, or `stockout_flag` — confirmed by reading its full field list.
- Three KPI formulas (Dropsize, ACV tier weights, Range Billing) are explicitly marked `NEEDS GPIL CONFIRMATION` in code comments — `compute_kpis.py:25-40, 58-61, 150-156, 296-306`.
- `state_factor` (population/GSDP-derived) and `dropsize_factor` are computed in `build_visits()`/`build_orders()` (lines 635-666, 786-798) but never written to any output CSV.
- There is no orchestration script (`Makefile`, `run.py`, or similar) sequencing Phase 2 → tier reassignment → Phase 3 → Phase 4 — confirmed by searching the repo root; each phase's `README.md` section documents it as a manually-run `python -m ...` command.
- `data/graphrag_docs_manifest.csv` contains 672 documents (673 lines including header), matching `kpi_state_month.csv`'s 28 states × 24 months grain.

### Inferred, not verified — do not treat as ground truth
- **The mapping to a real company** ("GPI" = Godfrey Phillips India's own cigarette brands, "IPM" = Marlboro under an International Premium license, etc.). Nothing in the code states this outright; it is a plausible reading of the category/franchise names (`GPI`, `IPM`, `Marlboro`, `TicTac`, `Kinder_Joy`) but is not asserted anywhere in comments or docs. Flagged here only because the existing `docs/BUSINESS_KNOWLEDGE_BASE.md` states it more confidently than the code supports.
- **Any causal narrative connecting a specific disruption to a specific KPI drop** (e.g. "Bihar's October Service Level dropped because of a Marlboro Pack 9 disruption at a specific WD"). The underlying *mechanism* is real and confirmed (`build_orders()` does check `stockout_flag` to cap fulfilment, line 885), but no document text states any such causal chain — see the Supply Disruption entity note. Any specific example sentence describing "why" a metric moved in a real run is illustrative, not a verified fact about the actual generated dataset.
- **Whether the current full-scale data run actually satisfies `verify_cascade()`** (the 12-way Gold-Dealer-beats-everything ordering check, `generate_synthetic_data.py:986-1045). The constants (`CHANNEL_BASE_MULTIPLIER`, `TIER_BASE_MULTIPLIER`) were tuned analytically and the code comment documents the reasoning, but this analysis did not re-run `verify_cascade()` against the CSVs currently on disk to confirm pass/fail.
- **Whether `compute_kpis.py` was run before or after `reassign_outlet_tiers.py`** for the KPI/document files currently in `data/`. Both orderings are structurally possible (no code enforces sequencing); which one actually happened for the current `kpi_state_month_category.csv` (and therefore its ACV values) is not verifiable from the code alone.
- **The three sample documents originally reviewed while building this dictionary — `Maharashtra_2024-08.txt`, `Assam_2025-10.txt`, `Karnataka_2025-03.txt` (all in `data/graphrag_input/`) — were generated from a stale/prior data run and their embedded KPI numbers do NOT match the current `kpi_state_month.csv` on disk.** Confirmed directly: `Maharashtra_2024-08.txt` states Dropsize = "65.79 units per productive visit", while the current `kpi_state_month.csv` shows Dropsize = 200.96 for Maharashtra/2024-08 — roughly 3x off. (Productivity and Service Level in the same document, 79.0% and 92.4%, do still happen to be close to the current CSV's 79.01% and 92.32% — so the divergence isn't uniform across all six metrics, which points at a change specifically in order-quantity generation between data runs, not a wholesale regeneration of every KPI.) **Do not use these three documents — or any other pre-existing `.txt` file in `data/graphrag_input/` — as a reference for expected KPI values going forward.** Only the CSVs currently on disk (`kpi_state_month.csv`, `kpi_state_month_category.csv`, `kpi_wd_month.csv`, `kpi_wd_month_category.csv`) are ground truth as of this update; the `.txt` corpus itself has not been regenerated to match and will need a rebuild before it's trustworthy again.

---

## 5. Known Gaps

1. **PARTIALLY RESOLVED — WD-level KPIs now exist; SE-level still does not.** `data/kpi_wd_month.csv` and `data/kpi_wd_month_category.csv` now exist, computed by `build_kpi_wd_month()` (`compute_kpis.py:442-465`) and `build_kpi_wd_month_category()` (`compute_kpis.py:554-573`), grouped by `(wd_id, month)` and `(wd_id, month, category_name)` respectively — using sibling `compute_*_wd()` functions that reuse the exact same formulas as their State-grain counterparts, just grouped by `wd_id` instead of `state_name`. Validated against `kpi_state_month.csv` for Maharashtra, 2024-08: Productivity, Service Level, and Dropsize all matched within a fraction of a percent when averaged across that state's 11 WDs. **SE-level KPIs still do NOT exist anywhere** — only the WD grain was added; `se_id` is still never used as a KPI grouping key (see the Sales Executive entity note above), so "why did this specific SE underperform" remains unanswerable from the data. This gap stays open for its SE half.

2. **The GraphRAG document corpus is far narrower than the underlying data.** Only State, Month, Category, Franchise (names), and WD (names only) appear in document text. Zone, SE, Outlet, Channel Type, Outlet Tier, SKU, individual Visits/Orders, and Supply Disruptions are fully absent from the 672 `.txt` files, even though they're richly modeled in the CSVs. Since GraphRAG extracts only from the text it's given, no amount of prompt engineering recovers entities that were never written into a document — this requires new document content, not a better extraction prompt.

3. **Hero SKU status is not persisted.** It's re-derivable by rerunning `select_hero_skus()` against the current `products.csv` (the logic is deterministic), but does not exist as a stored fact today.

4. **`state_factor` and `dropsize_factor` — the two mechanisms that create state-to-state performance variation — are never persisted.** They exist only transiently inside `build_visits()`/`build_orders()`. There is no stored "why is Karnataka's Productivity higher than Bihar's" signal beyond the numbers themselves; the generative cause is discarded after generation.

5. **Supply disruption severity is not reconstructable.** `stockout_flag` (boolean) is stored; the counterfactual `base_supply` that would show *how bad* a disruption was is not. Even if disruption events were added to future documents, only "disrupted: yes/no" could be stated, not magnitude.

6. **Category and Franchise have no surrogate ID — only name strings.** Any future graph-merge logic keyed on these names is exposed to string-matching fragility (typos, casing, whitespace) with no ID to fall back on.

7. **No Zone-level KPI or document exists**, despite Zone being a real hierarchy level with its own ID.

8. **Pipeline run-order is not enforced by any script.** `reassign_outlet_tiers.py` and `compute_kpis.py` both read/write independently, with no orchestrator sequencing them. Whether `kpi_state_month_category.csv`'s ACV values reflect the pre- or post-reassignment `outlet_tier` labels depends on the order the two scripts were actually run in for the data currently on disk — this analysis could not verify which happened.

9. **`CHANNEL_CATEGORY_ELIGIBILITY` (which categories a channel may sell) is untested against any real-world rule** — currently every channel is eligible for every category, a coded assumption rather than a discovered fact.

10. **No competitor, pricing/promotion, returns/shrinkage, sales-target/quota, beat-plan-change, credit-terms, regulatory-event, or Union-Territory data exists anywhere in the system.** These are absent by design scope, not oversight, but they bound what any "why" question can ultimately explain — the data can describe *what* happened, never *relative to a target or a competitor*, and never a regulatory or promotional cause.

11. **A separate, structurally different narrative document already exists at `docs/BUSINESS_KNOWLEDGE_BASE.md`** (untracked, 436 lines) covering similar ground with more inferred real-world business narrative woven in. It was not modified as part of producing this dictionary; where the two disagree in confidence level (e.g. the real-company mapping), this document is the more conservative, code-grounded source.

---

## 6. Design Decisions

Unlike Sections 1-5 (which describe only what the code currently does), this section records human decisions made about work that has **not** been implemented in code yet — choices that will shape future document-generation logic once WD-level detail is added to documents, but do not yet exist as `compute_kpis.py` or `build_documents.py` behavior. Flagged explicitly as **tunable design choices, not code-derived facts** — do not treat these as verified against any data.

1. **WD-level category-KPI anomaly threshold: ±10 percentage points.** When a future state/month document is extended to call out individual distributors by name using the new `kpi_wd_month_category.csv` (ND, ACV, OOS%, Range Billing), a WD should be named in that document's text only if its value differs from its state's average for that KPI/month by more than 10 percentage points **in the unfavorable direction** (e.g. an OOS% more than 10pp *higher* than the state average, or an ND more than 10pp *lower*). This is an agreed, tunable design choice — not derived from any statistical test, standard deviation, or code default — and should be revisited if it produces too many or too few call-outs once real documents are generated against it.
   **Tightened from ±6pp after the 7-document pilot** (`build_documents.py`, run against `Maharashtra_2024-08`, `Karnataka_2025-03`, `Assam_2025-10`, `Uttarakhand_2026-07`, `Sikkim_2025-04`, `Rajasthan_2024-08`, `Kerala_2024-09`): at 6pp, documents were producing far too many call-outs to read as a narrative — e.g. `Maharashtra_2024-08.txt` had **20 call-outs across its 11 distributors**, and `Sikkim_2025-04.txt` had **28 call-outs across its 13 distributors**. 10pp was chosen as a tighter bar that still leaves every pilot document with at least one real call-out (see the pilot's per-document counts in `build_documents.py`'s test run) while cutting the noisiest documents' counts substantially.
1a. **Anomaly threshold extended to WD-month KPIs (Productivity, Service Level, Dropsize).** Design Decision #1's ±10 percentage-point rule now also applies to Productivity and Service Level at WD-month grain (`kpi_wd_month.csv`), using the same logic: a distributor's value differing from its state's average by more than 10 percentage points in the unfavorable direction triggers a document call-out. Dropsize cannot use a percentage-point threshold since it's an absolute unit count (e.g. 200.96 units/productive visit), not a ratio — so for Dropsize specifically, the threshold is 10% relative deviation from the state average instead (e.g. if state Dropsize is 200, a WD triggers below 180 or above 220). Same tunability caveat as Design Decision #1 applies: agreed default, not code- or statistics-derived, revisit after pilot documents show how many distributors get flagged.
   **Same ±6 → ±10 tightening applies here, for consistency**, made at the same time and for the same reason as Design Decision #1's tightening (too many call-outs per pilot document at the 6pp/6% bar).
2. **The no-anomaly case must be stated explicitly, not omitted.** If no WD in a given state/month crosses the ±10pp threshold for a KPI, the document should say so directly (e.g. "No distributor showed a significant deviation from the state average this month") rather than silently dropping the topic. This avoids an ambiguous document where "no mention" could mean either "nothing to report" or "not yet implemented" — the same anti-silent-omission principle already applied elsewhere in this dictionary (e.g. Known Gap #2's point that GraphRAG can't extract what was never written).
