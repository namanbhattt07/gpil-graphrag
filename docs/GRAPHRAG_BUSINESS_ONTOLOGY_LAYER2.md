# GPIL S&D — Business Ontology & Diagnostic Reasoning Layer (Layer 2)

**Purpose:** this is the complete business world model — not limited to what current documents contain. It exists so the chatbot's *reasoning* is grounded in the real GPIL business hierarchy and KPI mechanics, even for entities not yet extractable.

**How this relates to the other two documents:**
- `BUSINESS_KNOWLEDGE_DICTIONARY.md` = what exists in code/data, validated line-by-line
- `GRAPHRAG_ENTITY_RELATIONSHIP_SCHEMA.md` ("Layer 1") = what the LLM can actually extract from current document text — **this is what goes into the extraction prompt**
- **This document ("Layer 2") = the full business picture, used for reasoning design and future planning — NOT fed directly into the extraction prompt**

**Critical rule:** Layer 2 entities/relationships marked "not yet extractable" must never be added to an extraction prompt as-is. GraphRAG only extracts what's written in the text it's given (confirmed repeatedly in the dictionary). Feeding an LLM an entity type with no textual grounding causes it to hallucinate matches — inventing edges that look plausible but aren't backed by any document sentence. Every item below is tagged so this distinction is never lost.

---

## 1. Complete Entity Hierarchy

### Geography chain
```
State → Zone → Distributor → Sales Executive → Outlet
```
| Entity | In Layer 1 (extractable today)? |
|---|---|
| State | Yes |
| Zone | No — never named in document text |
| Distributor | Yes (name-only; KPI values coming via Design Decision #1) |
| Sales Executive | No — no SE-grain KPI exists, no SE ever named |
| Outlet | No — individual outlets never named; ~120K records with no textual grounding |

### Product chain
```
Category → Franchise → SKU
```
| Entity | In Layer 1 (extractable today)? |
|---|---|
| Category | Yes |
| Franchise | Yes (name-only; no franchise-level KPI number) |
| SKU | No — never individually named in document text |

### Classification axes (not hierarchy, cross-cutting properties of Outlet)
```
Outlet —classified as→ Channel Type (Retail / Hawkers / Modern Trade / Dealer)
Outlet —classified as→ Outlet Tier (Gold / Silver / Bronze)
```
Neither Channel Type nor Outlet Tier is named in any document today, despite both driving real business logic (the 12-way sales cascade, ACV tier-weighting). Not extractable in Layer 1.

---

## 2. Complete Relationship Set

| Relationship | Meaning | Extractable in Layer 1 today? |
|---|---|---|
| State HAS Zone | State divided into zones | No |
| Zone HAS Distributor | Zone served by distributors | No |
| Distributor EMPLOYS SalesExecutive | Distributor runs a field sales force | No |
| SalesExecutive VISITS Outlet | SE makes sales calls on assigned outlets | No |
| Distributor SERVES Outlet | Distributor supplies stock to outlet (via SE) | No — real relationship, never stated in text |
| Outlet PURCHASES SKU | An order line | No |
| SKU BELONGS_TO Franchise | Product hierarchy | No (only Franchise-level, not SKU) |
| Franchise BELONGS_TO Category | Product hierarchy | **Yes** (Category CONTAINS Franchise, same fact, extractable) |
| Distributor SERVES State | Distributor operates in state | **Yes** |
| **Metric IMPACTS Metric** | KPI dependency (see Section 3) | **Derived, not extracted — computed/injected separately, see note below** |
| **Distributor DEVIATES_FROM StateAverage** | Distributor's KPI differs meaningfully from state | **Yes, once anomaly sentences added (Design Decision #1/#1a)** |

**Note on `Metric IMPACTS Metric`:** like `PRECEDES` in Layer 1, this is not something the LLM should try to extract from prose — it's a fixed business-logic fact (Section 3 below), true regardless of what any single document says. It belongs in the graph as a pre-computed/injected edge, or as reasoning logic in the retrieval layer, not as an extraction target.

---

## 3. KPI Dependency Graph

**Grounding rule:** every dependency below is taken from the dictionary's confirmed formulas and the one Causal KPI Relationships section that was code-validated — not invented from generic FMCG knowledge. Where a dependency is a real formula relationship (e.g. Inventory Days = 30/Inventory Turns), it's marked **mechanical**. Where it's a real but indirect business mechanism confirmed in code (e.g. disruption → OOS%), it's marked **confirmed-indirect**. Nothing here is asserted as a marketing-general "these are usually related" claim.

```
Supply Disruption (stockout_flag)
   │ [confirmed-indirect: build_orders() checks this flag to cap fulfilment]
   ▼
Service Level (qty_delivered / qty_ordered)
   │ [confirmed-indirect: same flag is the numerator of OOS% at category grain]
   ▼
Out-of-Stock % (OOS%)
   │ [confirmed-indirect: if a category is often unavailable, fewer outlets can bill it]
   ▼
Numeric Distribution (ND) / ACV
   │ [mechanical: Range Billing is computed only over outlets that already billed something —
   │  ND is a precondition for Range Billing to even be measured]
   ▼
Range Billing
```

**Confirmed discretization artifact (pilot-verified, not a bug):** IPM category's OOS% at WD grain can only ever land on {0%, 20%, 40%, 60%, 80%, 100%} — IPM has exactly 5 SKUs (all Marlboro), and `inventory_snapshots.csv` is one row per WD×SKU×month, so any WD's IPM OOS% denominator is always 5. Multiple distributors independently landing on the same value (e.g. four different Maharashtra distributors all at exactly 20.0% IPM OOS% in August 2024, each from a different stocked-out SKU) is expected coincidence from the coarse denominator, not a shared cause. **Practical consequence:** a reasoning layer should not treat identical OOS% values across distributors as evidence of a common root cause for small-SKU-count categories (IPM specifically) — check the underlying stocked-out SKU per distributor before inferring any shared mechanism. This does not apply to GPI (6 franchises, more SKUs, finer-grained OOS% possible).

**Structural gap — Channel-level OOS% is not computable.** `inventory_snapshots.csv` (the only OOS% source) is keyed by `(wd_id, sku_id, month)` — a distributor's warehouse stock — and carries no `outlet_id` or `channel_type`. A WD's stock isn't attributable to any single downstream channel, so Channel-grain KPIs (`kpi_state_month_channel.csv`) only carry Numeric Distribution, ACV, and Range Billing — OOS% is state/WD-grain only and does not exist at Channel grain. A diagnostic path should not attempt to attribute an OOS% finding to a specific channel.

```
Inventory Turns (qty_sold / avg_stock)
   │ [mechanical: Inventory Days = 30 / Inventory Turns — reciprocal, same signal]
   ▼
Inventory Days
```

```
State Performance Factor (population/GSDP-derived, dictionary-confirmed, NOT persisted to any output file)
   │ [confirmed: drives visit-to-order conversion rate]
   ▼
Productivity (productive_visits / total_visits)
   │ [confirmed: Dropsize's state-scaling factor is a blend of the same state_factor, ~0.3-0.6 correlation — NOT a 1:1 lock]
   ▼
Dropsize (qty_ordered / productive_visits)
```

**What this dependency graph deliberately does NOT claim:**
- It does not claim Service Level "causes" a Sales figure — there is no persisted Sales/Revenue KPI anywhere in the codebase (confirmed absent in the dictionary's KPI Model — the ten KPIs are Productivity, SKUs/Transaction, Service Level, Dropsize, Inventory Turns, Inventory Days, ND, ACV, OOS%, Range Billing; none of them is "Sales" or "Revenue" directly).
- It does not assign a magnitude to the Supply Disruption → Service Level link, since disruption severity isn't persisted (Known Gap #5) — the arrow means "constrains," not "causes an X% drop."

---

## 4. Diagnostic Reasoning Paths

Investigation paths for the issue types requested, each one only using KPIs and relationships confirmed to exist. Each path is a *question the retrieval layer should check, in order* — not a guarantee every step will find something.

### Supply issues
```
Service Level or OOS% looks abnormal for a State/Category/Month
   → Check: did any Distributor in that state cross the DEVIATES_FROM threshold on Service Level or OOS% that month?
   → If yes: name the distributor(s), report the gap size (Edge property: gap_percentage_points)
   → If no: state explicitly that no distributor showed significant deviation (Design Decision #2)
   → [Cannot go deeper into WHY the disruption happened — severity/magnitude not in data, Known Gap #5]
```

### Outlet coverage issues (ND / ACV)
```
Numeric Distribution or ACV looks low for a State/Category/Month
   → Check: did any Distributor cross the DEVIATES_FROM threshold on ND/ACV for that category?
   → If yes: name the distributor(s)
   → Note the caveat: ACV's tier-weighting is a placeholder (NEEDS GPIL CONFIRMATION) — flag this in the answer,
     don't present ACV gaps with full confidence
   → [Cannot break down further by Channel Type or Outlet Tier — not extractable in Layer 1 today]
```

### Productivity issues
```
Productivity looks low for a State/Month
   → Check: did any Distributor cross the DEVIATES_FROM threshold on Productivity that month?
   → If yes: name the distributor(s)
   → [Cannot attribute to a specific Sales Executive or Outlet — no SE/Outlet grain KPI exists, Known Gap #1's open half]
```
**Confirmed data property (pilot-verified, not a bug):** Productivity's cross-distributor spread within a state is extremely tight — std dev ~0.5-0.9pp, full range under 3pp across all 7 pilot states — consistent with sampling noise around one shared state-wide conversion rate (verified against the binomial-noise formula, matched observed values almost exactly). Service Level's spread is 3-7x larger because it's driven by genuine per-distributor fulfilment mechanics (`stockout_flag`), while the generator does not inject material WD-to-WD heterogeneity into visit outcomes. **Practical consequence: Productivity will rarely or never produce a DEVIATES_FROM edge with the current generator, even at a loose threshold.** A "why is Productivity low" question should expect the answer "no distributor-level explanation available — Productivity doesn't vary meaningfully by distributor in this data" more often than a name-a-distributor answer. This is a real property of the synthetic data, not a gap in the ontology or threshold design.

### Assortment issues (Range Billing)
```
Range Billing looks low for a State/Category/Month
   → First check ND for the same scope — if ND is also low, the issue is breadth (few outlets stock it at all),
     not depth (Range Billing only measures outlets that already billed something)
   → If ND is normal but Range Billing is low: outlets stock the category but not the full SKU range —
     this is a genuine assortment-depth signal
   → Check: did any Distributor cross the DEVIATES_FROM threshold on Range Billing?
```
**Confirmed data property (verified against the full 672-document corpus, not a bug):** Range Billing's cross-distributor deviation never crosses the 10pp threshold anywhere in the corpus — max observed gap is 7.2pp. Its values are naturally low and tightly clustered across distributors (unlike ND/ACV/OOS%, which do show real distributor-level spread). **Practical consequence: a Range Billing DEVIATES_FROM edge should not be expected to ever fire with the current generator and threshold.** Lowering the threshold specifically to force Range Billing triggers would manufacture false signal rather than surface a real one — the metric's tight clustering is a property of the data, not an under-tuned threshold. A "why is Range Billing low" question should rely on the ND-first check above (breadth vs. depth) rather than expecting a named-distributor answer.

### Inventory issues
```
Inventory Turns/Days look abnormal for a State/Month
   → These are reciprocal (mechanical relationship) — report both together, not as independent findings
   → Check OOS% for the same state/month — if OOS% is also elevated, the two are likely linked via
     the shared Supply Disruption mechanism (see Section 3)
   → [Cannot identify which specific SKU or WD×SKU×month combination drove it without Distributor-level
     Inventory KPIs — not currently computed at WD grain the way Service Level/Productivity/Dropsize now are]
```

---

## 5. Scenario Example (Worked, Using Real Validated Data)

**Question:** "Why might Service Level in Maharashtra, August 2024 be worth investigating?"

**Reasoning path followed:**
1. Maharashtra state-level Service Level, August 2024 = 92.3% (confirmed, `kpi_state_month.csv`)
2. Check Distributor-level deviations: Distributor Johnson PLC Distributors shows Service Level 82.9% — a gap of 9.4 percentage points below the state average, past the ±6pp threshold (Design Decision #1a). *(Corrected from an earlier draft of this example, which misattributed this same 82.9%/9.4pp deviation to Boyd-White Distributors — verified against the actual regenerated pilot document, `Maharashtra_2024-08.txt`.)*
3. No other Maharashtra distributor crosses the threshold this month (based on the pilot document reviewed)
4. Conclusion the graph can support: "Johnson PLC Distributors was the primary contributor to any Service Level softness in Maharashtra in August 2024 — it delivered 82.9% against orders, compared to the state average of 92.3%."
5. Conclusion the graph CANNOT support without further data: *why* Johnson PLC's Service Level was low — that would require Supply Disruption to be a stated fact in the document, which it isn't yet (Section 3's caveat).

This shows both what Layer 2's reasoning paths make possible today, and where they honestly stop.

---

## 6. What Layer 2 Intentionally Does Not Solve

- It does not make Zone, SE, Outlet, Channel Type, Outlet Tier, or SKU extractable — that requires new document content (a future document-template revision), not a bigger ontology.
- It does not invent a Sales/Revenue KPI or a "Root Cause" entity — neither exists in the codebase; using them here would contradict the dictionary.
- It does not assign confidence or magnitude to any causal claim beyond what Section 3 explicitly marks as confirmed vs. mechanical vs. indirect.

---

*This document informs future document-template decisions (which entities are worth adding to text next) and the retrieval/reasoning layer's query logic. It is not an input to the entity extraction prompt — only `GRAPHRAG_ENTITY_RELATIONSHIP_SCHEMA.md` (Layer 1) is.*