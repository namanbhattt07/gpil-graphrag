# GPIL Sales & Distribution — Business Knowledge Base

**Purpose:** This document reverse-engineers the *business model* implemented by this codebase — not the code itself. It reads the repository the way a Sales & Distribution (S&D) business architect would, to answer: *what company is this pretending to be, how does its business actually work, and what does that mean for building a knowledge graph on top of it?*

**Grounding:** Everything below is derived from `src/data_gen/generate_synthetic_data.py`, `src/data_gen/reassign_outlet_tiers.py`, `src/kpis/compute_kpis.py`, `src/graph/build_documents.py`, `src/reports/generate_pan_india_summary.py`, and the two prior reports in `docs/`. Where the earlier `BUSINESS_VALIDATION_REPORT.md` describes a data run that predates the most recent tier/channel-multiplier fix, that is called out explicitly rather than silently repeated as current fact.

**Who this company looks like:** The category names (GPI, IPM/Marlboro), the confectionery brands (TicTac, Kinder Joy), and the "Category → Franchise → SKU" language all map cleanly onto a real Indian tobacco distributor that also runs a licensed confectionery business — the shape of the real-world company (**G**odfrey **P**hillips **I**ndia) this dataset is standing in for. GPI = the company's own cigarette brands, IPM = Marlboro under license from **I**nternational **P**remium tobacco, Ferrero = the licensed confectionery line, Candy = an in-house sweets business. Nothing in the code says this outright, but it explains every modeling choice below (why cigarettes dominate revenue, why "hero SKUs" matter, why Diwali season matters).

---

## 1. Business Entity Dictionary

| Entity | What it is in the real business | Where it lives | Count (pilot scale) |
|---|---|---|---|
| **State** | GPIL's top geography level — one of the 28 real Indian states GPIL sells in | `geography.csv`, `unit_type="State"` | 28 |
| **Zone** | A sales-management subdivision of a state (e.g. "Andhra Pradesh Zone 1") | `geography.csv`, `unit_type="Zone"` | 56 (2/state) |
| **WD — Wholesale Distributor** | An independent business (not owned by GPIL) that buys stock from GPIL in bulk and resells it onward through its own sales force. This is GPIL's actual customer of record in each territory. | `geography.csv`, `unit_type="WD"` | ~336–392 |
| **SE — Sales Executive** | A GPIL/WD field salesperson who owns a fixed "beat" (a defined list of outlets they visit on a schedule) | `geography.csv`, `unit_type="SE"` | ~4,000–8,600 |
| **Outlet** | A retail shop where a consumer actually buys a cigarette pack or a Kinder Joy — the true point of sale | `outlets.csv` | ~120,000+ |
| **Channel Type** | The *kind* of shop an outlet is (Retail/Paan shop, Hawkers, Modern Trade, Dealer) — a trade-classification attribute of Outlet, not a separate hierarchy level | `outlets.csv.channel_type` | 4 values |
| **Outlet Tier** | Gold/Silver/Bronze — a performance band assigned *within* a channel, meant to reflect how much volume an outlet actually moves | `outlets.csv.outlet_tier` | 3 values |
| **Category** | GPIL's top product grouping (GPI, IPM, Ferrero, Candy) | `products.csv.category_name` | 4 |
| **Franchise** | A brand family under a category (e.g. Marlboro, TicTac, Kinder_Joy, GPI_Franchise_1..6) | `products.csv.franchise_name` | 11 |
| **SKU** | One sellable product variant ("pack") under a franchise, with its own price | `products.csv` | 63 |
| **Hero SKU** | An informal but business-critical sub-type: the flagship SKU of 4 GPI franchises + Marlboro, deliberately weighted to sell far more than the rest (a Pareto/80-20 pattern real tobacco portfolios show) | not a column — a hard-coded set in `select_hero_skus()` | 5 |
| **Visit** | One SE call on one outlet, on one date, with an outcome | `visits.csv` | ~3.9M+ |
| **Order / Order Line** | A basket of SKUs an outlet buys on a given visit (one row per SKU line) | `orders.csv` | ~8.3M+ lines |
| **Inventory Snapshot** | A WD's monthly stock position for one SKU (opening/received/sold/closing) | `inventory_snapshots.csv` | ~214,000+ |
| **Disruption / Stockout Event** | An implicit event: a WD×SKU×month where incoming supply was cut sharply | `inventory_snapshots.csv.stockout_flag` | ~8% of WD×SKU×months |
| **KPI Record (State×Month)** | A derived, computed "fact" summarizing a state's overall health for a month | `kpi_state_month.csv` | 672 |
| **KPI Record (State×Month×Category)** | A derived fact summarizing one product category's distribution health in a state/month | `kpi_state_month_category.csv` | 2,688 |
| **Narrative Document** | A GraphRAG-ready paragraph describing one State×Month — the atomic unit the knowledge graph will actually be built from | `data/graphrag_input/*.txt` | 672 |

**A naming trap worth flagging up front:** "WD" (Wholesale Distributor, a geography-hierarchy node) and "Dealer" (one of the four retail `channel_type` values) are **two unrelated concepts that sound alike**. A WD is GPIL's direct business partner who owns SEs and holds inventory; a "Dealer"-channel outlet is just a large-format retail shop at the bottom of the tree, no different structurally from a Paan shop. If a GraphRAG extraction prompt isn't told this explicitly, it is very likely to merge "Distributor" and "Dealer" into one entity type — which would silently corrupt every downstream distributor-level answer.

---

## 2. Complete Relationship Dictionary

| Relationship | From → To | Real-world meaning | Cardinality |
|---|---|---|---|
| CONTAINS | State → Zone | A state is divided into zones for sales management | 1 : 2 |
| CONTAINS | Zone → WD | A zone is served by several independent distributors | 1 : 4–7 |
| EMPLOYS / MANAGES | WD → SE | A distributor runs a field sales force | 1 : 12–22 |
| OWNS BEAT | SE → Outlet | An SE has a fixed, assigned list of shops they call on | 1 : 140–180 |
| LOCATED_IN | Outlet → State/Zone/WD | Denormalized geography — every outlet row also carries its state/zone/WD id directly, so no join is needed to answer "which state is this shop in" | many : 1 |
| CONTAINS | Category → Franchise | e.g. GPI category contains 6 franchises | 1 : 1–6 |
| CONTAINS | Franchise → SKU | e.g. Marlboro contains 4–6 pack variants | 1 : 3–8 |
| ELIGIBLE_FOR | Outlet(channel_type) → Category | A business rule, not a transaction: which categories a channel is *allowed* to sell (currently: every channel is eligible for every category — see Section 4) | many : many |
| PERFORMED | SE → Visit | An SE makes a sales call | 1 : many |
| TARGETS | Visit → Outlet | The call is on a specific shop | many : 1 |
| RESULTS_IN | Visit → Order | Only if `visit_outcome = "Order Placed"` | 1 : 0 or 1 |
| CONTAINS | Order → Order Line | A basket has 1–6 SKU lines | 1 : many |
| REFERENCES | Order Line → SKU | Which product was ordered | many : 1 |
| SOURCED_FROM | Order Line → WD | The line is fulfilled out of the outlet's WD's stock | many : 1 |
| HOLDS_STOCK_OF | WD → SKU (via Inventory Snapshot) | A distributor's monthly stock position for a product | many : many, time-scoped |
| EXPERIENCED | WD×SKU×Month → Disruption | A supply-chain shock that constrained fulfilment | many : 1 (per month) |
| DESCRIBES | KPI Record → State + Month | The computed fact belongs to exactly one state/month | many : 1 |
| SCOPED_TO | KPI-Category Record → State + Month + Category | Same, but also scoped to a category | many : 1 |
| REPORTS_ON | Document → State + Month | The narrative text is the human-readable form of the two KPI rows above | 1 : 1 |
| MENTIONS | Document → WD name(s), Franchise name(s) | The document names entities it has no numeric KPI for yet (WD, Franchise), purely so GraphRAG can extract a relationship edge even without a number attached | 1 : many |

**Worked FMCG example:** *SE Rohan Mehta* (an SE under *WD "Rodriguez, Figueroa and Sanchez Distributors"* in *Andhra Pradesh Zone 1*) visits *Outlet OUT004521, a Paan shop*. The visit's outcome is "Order Placed." That creates an *Order* with, say, 3 lines: 40 packs of *Marlboro Pack 9* (a Hero SKU), 10 packs of *GPI_Franchise_3 Pack 5*, and 5 *TicTac Pack 1*. Each line draws down against the WD's *Inventory Snapshot* for that SKU that month — if Marlboro Pack 9 had a stockout event at that WD that month, the Marlboro line gets only partially delivered, which then shows up a month later as a lower *Service Level* number for Andhra Pradesh in the *KPI Record*, and finally as a sentence in that state/month's *Narrative Document*.

---

## 3. KPI Dictionary

### State × Month (category-agnostic — describes overall S&D health)

| KPI | Formula | Business question it answers |
|---|---|---|
| **Productivity** (aka "Strike Rate") | Visits with outcome "Order Placed" ÷ total visits | Of every sales call an SE makes, what fraction turns into a sale? |
| **SKUs / Transaction** | Total order lines ÷ total distinct orders | How wide a basket does the average sale carry? |
| **Service Level** | Total qty delivered ÷ total qty ordered | Of what outlets *asked* for, how much did they actually *get*? |
| **Dropsize** ⚠️ *needs GPIL confirmation* | Total qty ordered ÷ number of productive (order-placed) visits | On an average successful call, how many units does the SE sell? |
| **Inventory Turns** | Total qty sold ÷ average stock held (opening+closing)/2 | How fast is distributor stock moving? |
| **Inventory Days** | 30 ÷ Inventory Turns | Same signal, in "days of stock on hand" — the way ops teams actually talk about it |

### State × Month × Category (product-scoped — describes distribution reach and depth)

| KPI | Formula | Business question it answers |
|---|---|---|
| **Numeric Distribution (ND)** | Outlets that billed ≥1 SKU of the category ÷ outlets eligible to sell that category | What % of shops that *could* stock this category actually do? |
| **ACV** ⚠️ *tier weights need confirmation* | Same ratio as ND, but each outlet counted by tier weight (Gold=3, Silver=2, Bronze=1) instead of counted equally | Same question, but weighted so a few big shops billing matters more than many tiny ones |
| **Out-of-Stock % (OOS)** | Share of that category's WD×SKU×month snapshots flagged `stockout_flag=True` | How often was this category simply unavailable at the distributor level? |
| **Range Billing** ⚠️ *needs GPIL confirmation* | Among outlets that billed *anything* in the category, average (distinct SKUs billed ÷ SKUs available in category) | Of the shops that do stock this category, how much of the full range do they actually carry — one SKU or the whole line-up? |

**Real FMCG example:** In Kerala for March 2025, GPI's Numeric Distribution might be 65% (65% of eligible Paan/Kirana/Dealer/MT shops billed at least one GPI cigarette that month) while its Range Billing is only 12% — meaning most of those shops that *do* sell GPI cigarettes are only carrying one or two SKUs out of the category's full range, not the whole portfolio. That's the classic "wide but shallow" distribution pattern a real S&D team worries about.

Three of these ten KPIs (Dropsize, ACV tier weights, Range Billing) are explicitly flagged in code as placeholder definitions — a real S&D team should sign off on the exact formula before treating them as ground truth in a chatbot answer.

---

## 4. Business Rules

1. **Channel–category eligibility.** Every one of the 4 channel types (Retail, Hawkers, Modern Trade, Dealer) is currently allowed to sell all 4 categories — even Ferrero/Candy through a Paan shop, on the theory that Tic Tac/Kinder Joy are impulse-buy counter items. This is a coded assumption (`CHANNEL_CATEGORY_ELIGIBILITY`), not a discovered fact, and is a one-line change if GPIL says otherwise.
2. **Tier is a performance rank, not a pre-assigned status.** `reassign_outlet_tiers.py` re-labels Gold/Silver/Bronze *after the fact*, purely by ranking each outlet's realized delivered volume within its own channel (top 20% = Gold, next 35% = Silver, bottom 45% = Bronze). Tiers are never compared across channels — a Bronze Dealer can still outsell a Gold Retail shop.
3. **The 12-way sales cascade is an enforced business rule, not an emergent pattern.** The generator is required to produce, by construction: Gold Dealer > Silver Dealer > Bronze Dealer > Gold Modern Trade > Silver Modern Trade > Bronze Modern Trade > Gold Hawkers > Silver Hawkers > Bronze Hawkers > Gold Retail > Silver Retail > Bronze Retail, verified by `verify_cascade()` after every regeneration. This encodes a real trade-marketing belief — a wholesale-style Dealer, even a weak one, moves more volume than the best small shop.
4. **Hero SKUs get a demand advantage by design.** 5 of the 63 SKUs (4 GPI flagships + Marlboro's flagship) get a 25× pick-weight boost and 1.4× quantity boost on every order line — modeling the real Pareto pattern where a handful of SKUs carry most of a tobacco portfolio's volume.
5. **Category volume mix is a fixed target, not a free variable.** GPI≈46%, IPM≈45%, Ferrero≈5%, Candy≈4.5% of units sold — cigarettes dominate volume by design (`CATEGORY_WEIGHTS_BY_VOLUME`), consistent with GPIL being a tobacco-led distributor with a small confectionery side business.
6. **Supply disruptions are random but consequential.** Each WD×SKU×month has an 8% independent chance (`STOCKOUT_EVENT_PROB`) of a supply cut (incoming stock drops to 20% of normal). This is the single mechanism that creates believable Service Level/OOS variation.
7. **Fulfilment depends on whether that month was disrupted.** Disrupted lines get 20–70% fulfilled; normal lines get 100% fulfilled 85% of the time, and 60–99% the other 15% (routine minor shortfalls even without a full disruption).
8. **State performance is a fixed, consistent trait, not monthly noise.** Every state gets one `state_factor`, computed once from real 2024 population and GSDP (log-scaled, min-max normalized), that nudges its visit-to-order conversion rate up or down for *all 24 months*. Bigger/richer states convert visits into sales more reliably — a believable macro assumption, though the code is honest that this factor has no real "story" behind it (no strikes, no local events) — it's a statistical tilt, not a narrative one.
9. **Productivity and Dropsize are deliberately correlated, but not perfectly.** Dropsize's state-level scaling factor is a 50/50 blend of the same `state_factor` used for Productivity, plus independent noise — tuned so the two KPIs land at a believable 0.3–0.6 correlation (a state good at converting visits is *usually* also good at bigger baskets, but not mechanically identical).
10. **Seasonality is uniform across all categories.** Oct–Dec (Diwali) and January get a 10–30% demand boost, and Ferrero/Candy get an *extra* 1.6× boost in Oct/Nov specifically — but the base seasonal multiplier is applied to cigarette categories too, which is a known simplification (real cigarette demand is habitual, not gift-seasonal).
11. **An inactive outlet must generate zero transactions inside the data window.** `is_active=False` outlets get a `closure_date` set *before* the 24-month window even starts, and `build_visits()` explicitly stops generating visits for an outlet once past its closure date — so, in the current generator, a closed shop cannot appear in `visits.csv` or `orders.csv` at all.

---

## 5. Business Process Flow

```
Distribution network exists (State→Zone→WD→SE→Outlet, fixed for the whole 24 months)
        │
        ▼
Each month: WD receives stock from GPIL, sells to outlets, tracks Opening/Received/Sold/Closing
        │ (occasionally disrupted — a supply-chain shock)
        ▼
Each month: SE visits each of their outlets 0-2 times
        │
        ▼
Visit outcome: Order Placed / No Order / Closed
   (probability shaped by season + this state's fixed performance factor)
        │  (only if "Order Placed")
        ▼
Outlet picks 1-6 SKUs (weighted by category volume target, festive boost, Hero-SKU boost)
        │
        ▼
Quantity per SKU set (shaped by channel type × outlet tier × Hero-SKU status)
        │
        ▼
Fulfilment checked against that WD's stock position for that SKU/month
   → if disrupted: partial delivery (20-70%)
   → if normal: usually full, occasionally a small shortfall
        │
        ▼
Raw transactional data (visits, orders, inventory) accumulates for 24 months
        │
        ▼
Phase 3: rolled up into State×Month and State×Month×Category KPIs
        │
        ▼
Phase 4: KPIs turned into 672 plain-English narrative documents
        │
        ▼
Phase 5 (not working yet): documents fed to GraphRAG → knowledge graph
        │
        ▼
Phase 6-8 (planned): a user asks "why did X happen in state Y" and gets a grounded, cited answer
```

This is the standard **Visit → Order → Fulfilment → KPI → Insight** pipeline every S&D-led FMCG company runs, just compressed to state-level monthly granularity for the purposes of this project.

---

## 6. Geography Hierarchy

```
State (28 — real Indian states)
 └─ Zone (2 per state, 56 total)
     └─ WD / Wholesale Distributor (4-7 per zone, ~336-392 total)
         └─ SE / Sales Executive (12-22 per WD, ~4,000-8,600 total)
             └─ Outlet (140-180 per SE, ~120,000+ total)
```

- Every level down to Outlet carries a denormalized `state_name`, so answering "which state is this outlet/SE/WD in" never requires walking the parent chain.
- The real GPIL network is closer to ~600,000–850,000 outlets; this pilot dataset deliberately scales SE-per-WD and Outlet-per-SE back *up* toward realistic ratios while keeping Zone/WD counts small, because GraphRAG only ever reads the 672 State×Month documents — the outlet count only affects how long Phase 2/3 take to run, never the graph's size or cost.
- Union Territories (Delhi, Chandigarh, J&K, etc.) are **not** modeled — only the 28 states. A real GPIL network would include these.

---

## 7. Product Hierarchy

```
Category (4: GPI, IPM, Ferrero, Candy)
 └─ Franchise (11 total)
     └─ SKU (63 total — "packs"/variants of a franchise)
```

| Category | Franchises | SKUs/franchise | Price band (₹) | Real-world read |
|---|---|---|---|---|
| GPI | 6 generic franchises | 4-8 | 70-320 | GPIL's own cigarette brands |
| IPM | Marlboro (1 franchise, deliberately more SKUs) | 4-6 | 100-400 | Licensed international premium brand |
| Ferrero | TicTac, Kinder_Joy | 3-5 | 10-150 | Licensed confectionery |
| Candy | 2 generic franchises | 3-5 | 5-100 | In-house confectionery |

- Only 3 franchise names are real-world (Marlboro, TicTac, Kinder_Joy) — everything else is a generic placeholder standing in for brand names GPIL hasn't disclosed.
- `pack_size` is a text label ("Variant 1..N"), not an actual quantity (stick count, grams) — so prices don't follow a logical size ladder (a known data-quality gap, see Section 13).
- 5 SKUs are "Hero SKUs" (informal, code-only classification) that are deliberately weighted to sell far more than the rest.

---

## 8. Sales & Distribution Hierarchy

This is the geography hierarchy (Section 6) *crossed with* two independent classification axes on Outlet:

| Axis | Values | What it represents |
|---|---|---|
| **Channel Type** | Retail (Paan/tobacconist), Hawkers, Modern Trade, Dealer | The *kind* of retail business — fixed at outlet creation, roughly proportional to how common each shop type is in reality (Retail ≈ 94% of outlet count, Modern Trade the rarest at ≈0.3%) |
| **Outlet Tier** | Gold / Silver / Bronze | A *within-channel* performance rank (top 20% / next 35% / bottom 45% by realized sales volume) |

An outlet's full "address" in the S&D hierarchy is therefore: **State → Zone → WD → SE → Outlet(Channel, Tier)**. Two outlets can share every geography node and channel type and still be worlds apart commercially if one is Gold and the other Bronze — that's the whole point of the tier system, and it's why Section 4's cascade rule and Section 13's honesty about "does tier actually change behavior" matter so much.

---

## 9. Inventory Flow

Modeled at **WD × SKU × Month** grain — the distributor is the entity that physically holds stock, not the outlet or the state.

```
Opening Stock (carried from last month's closing)
      +  Qty Received (this month's delivery from GPIL — cut to 20% if disrupted)
      =  Available
      -  Qty Sold (a noisy 50-95% of what's available — demand, not perfectly matched to supply)
      =  Closing Stock  →  becomes next month's Opening Stock
```

A WD×SKU×month is flagged `stockout_flag=True` if either (a) it had a supply disruption that month (an ≈8% independent chance every month, every WD, every SKU) or (b) its closing stock fell below 5 units regardless of cause. This flag is the single source that later becomes **Out-of-Stock %** at State×Month×Category grain (Section 3), and disrupted months are also the mechanism that caps **Service Level**, since `build_orders()` checks this exact flag to decide how much of an order actually gets delivered.

**Real FMCG example:** A WD in Bihar has a supply disruption on Marlboro Pack 9 in October 2025 (peak Diwali demand). That month's `qty_received` drops to 20% of normal. Every outlet under that WD ordering Marlboro Pack 9 that month gets only 20-70% of what they asked for. Multiply that across all outlets under that WD, and Bihar's October Service Level and IPM-category OOS% both take a visible hit — exactly the kind of causal chain a "why did Service Level drop in Bihar in October" question should be able to trace.

---

## 10. Order Flow

```
Visit outcome = "Order Placed"
        │
        ▼
Determine eligible categories (outlet's channel_type → CHANNEL_CATEGORY_ELIGIBILITY)
        │
        ▼
Weight every candidate SKU by:
   category's target volume share  ×  festive-season boost (Ferrero/Candy, Oct-Nov only)  ×  Hero-SKU boost (25x, if applicable)
        │
        ▼
Sample 1-6 SKU lines without replacement (max lines scaled by the state's "dropsize factor")
        │
        ▼
For each chosen SKU: base qty (1-24) × channel multiplier × tier multiplier × Hero-SKU qty multiplier (1.4x if applicable), capped at 2,000 units
        │
        ▼
Check that WD's stock disruption flag for that SKU/month
   → disrupted: deliver 20-70% of ordered qty
   → normal: deliver ~100% most of the time, 60-99% occasionally
        │
        ▼
One Order Line row per SKU: qty_ordered, qty_delivered, unit_price
```

The channel and tier multipliers are fixed numbers chosen specifically so that a Gold Dealer's average order dwarfs a Bronze Retail shop's — see Section 4, Rule 3.

---

## 11. Visit Flow

```
For every outlet, every month (until its closure_date, if any):
   Roll number of visits this month: 0 (5%), 1 (55%), or 2 (40%)
        │
        ▼
   For each visit:
      p(Order Placed) = base rate (58%) × month's seasonality × this state's performance factor,
                         clipped to 15-90%
      p(Closed) = flat 7%
      p(No Order) = whatever's left
        │
        ▼
      Roll outcome: "Order Placed" / "No Order" / "Closed"
```

A visit is the atomic unit of field sales activity — everything else (orders, revenue, KPIs) only exists because a visit happened first. An outlet that has already closed (closure_date in the past) generates **no visits at all** for any month past that date, which is why inactive outlets should show zero activity in the current generator.

---

## 12. Causal KPI Relationships (which KPI influences which, and why)

| Driver KPI/factor | Influences | Why (business mechanism) |
|---|---|---|
| **State performance factor** (population/GSDP-based) | Productivity | Bigger, richer states convert sales visits into orders more reliably — more disposable income, denser retail networks |
| **State performance factor** (blended, diluted) | Dropsize | The same underlying state strength that improves visit conversion also modestly increases basket size per successful call — deliberately blended to ~0.3-0.6 correlation with Productivity, not a 1:1 lock |
| **Supply disruption (stockout event)** | Service Level | A disrupted WD×SKU×month directly caps how much of an order can be delivered that month |
| **Supply disruption (stockout event)** | Out-of-Stock % | The same flag that caps fulfilment is also the numerator of OOS% at category level |
| **Out-of-Stock %** (indirectly) | Numeric Distribution / ACV | If a category is frequently unavailable at the WD, fewer outlets can bill it at all that month, dragging down ND/ACV — though this link isn't independently modeled; it flows through the same order-fulfilment mechanism |
| **Numeric Distribution** | Range Billing | Range Billing is only computed over outlets that billed *something* — an outlet can't have a Range Billing score if it never appears in ND's numerator. ND (breadth) is a precondition for Range Billing (depth) to even be measured |
| **Inventory Turns** | Inventory Days | Purely a reciprocal relationship (Days = 30/Turns) — the same underlying "how fast is stock moving" signal expressed two ways |
| **Festive seasonality (Oct-Nov)** | Productivity, Dropsize, and (extra) Ferrero/Candy SKU pick-rate | A general demand lift raises visit-to-order conversion and basket size everywhere, with an *additional* boost specifically for gifting categories |
| **Hero SKU status** | SKU-level volume concentration → category-level volume mix | Hero SKUs pull disproportionate share into GPI/IPM, which is *part of* why GPI+IPM dominate total category volume (Rule 5) — the category target and the SKU-level mechanism reinforce each other |
| **Channel type × Outlet tier** | Total units per outlet | The enforced 12-way cascade (Section 4, Rule 3) — by construction, not emergent, a Gold Dealer must move more volume than any lower channel/tier combination |
| **Fulfilment shortfall** | Dropsize, Service Level | Since Dropsize is computed on `qty_ordered` (not delivered) while Service Level is `delivered/ordered`, a bad fulfilment month lowers Service Level without moving Dropsize — these two metrics can diverge and that divergence is itself diagnostic ("outlets are still asking for a lot, but we're not getting it to them") |

**Chain example a "why" question should be able to walk:** *Why did Service Level drop in Assam in October?* → Check Assam's WDs for stockout-flagged SKU/months in October → find a disrupted WD×SKU → that WD's disrupted SKU pulls down that month's aggregate Service Level and that category's OOS% → October is also a festive month, so demand (and therefore order volume trying to be fulfilled) is unusually high at the same time the supply side is strained → the "why" is a supply disruption colliding with a seasonal demand peak.

---

## 13. Hidden Assumptions Made in the Code

These are all explicitly acknowledged in code comments or the design doc, not things this analysis is inferring — but they matter enormously for how much a GraphRAG "why" answer can be trusted:

1. **Scale-down, not scale-accuracy.** ~120,000 outlets stand in for GPIL's real ~600,000-850,000 — safe for GraphRAG cost (documents are state/month, not outlet-level), but any outlet-count-based extrapolation to real GPIL economics would be wrong by 5-7x.
2. **Only 3 real brand names exist** (Marlboro, TicTac, Kinder_Joy); the rest (GPI_Franchise_1-6, Candy_Franchise_1-2) are placeholders. A chatbot answer that says "GPI_Franchise_3 underperformed" is not naming a real GPIL brand.
3. **All 4 channels are assumed eligible for all 4 categories** — untested against real GPIL trade policy (e.g., maybe Modern Trade genuinely can't stock loose confectionery, or Dealers don't retail Candy at all).
4. **Three KPI definitions are explicitly unconfirmed**: Dropsize (units/productive-visit vs. some other definition), Range Billing (its exact formula), and the Gold=3/Silver=2/Bronze=1 ACV weighting (a rough proxy, not a real GPIL sales-value weighting).
5. **Inventory Days assumes a flat 30-day month** year-round — a small but real distortion for February vs. July.
6. **Tier is retrospective, not predictive.** "Gold" in this dataset means "turned out to be a top-20%-by-volume outlet within its channel this run" — it is not a business classification GPIL assigned in advance based on footfall, location, or credit limit. This is philosophically different from how a real S&D team uses outlet tiering (to *decide* investment, not just to *label* an outcome after the fact).
7. **One random seed controls everything** — geography, product catalogue, and all transactional noise are coupled to a single seed, so you cannot vary network size independently of the noise pattern.
8. **Festive seasonality applies uniformly across all 4 categories**, including cigarettes — the code comment for `MONTH_SEASONALITY` even flags this as a simplification; real cigarette demand is habitual, not gift-driven, so it shouldn't spike for Diwali the way Ferrero/Candy legitimately do.
9. **No price changes over 24 months.** Every order line uses the exact catalogue price — no excise-driven cigarette price hikes (which happen almost annually in India) and no confectionery trade promotions are modeled.
10. **No macro shocks in outlet onboarding** — no COVID-era dip, no expansion/contraction story; onboarding rate is a flat random draw across 2015-2023.
11. **State "performance factor" has no real narrative behind it.** It's grounded in real population/GDP data (a defensible proxy), but the code is explicit that there's no equivalent to "this state had a distributor strike" — it's a statistical tilt, not an event a "why" answer can point to as a root cause.
12. **The strict 12-way sales cascade is a constraint the generator is forced to satisfy, not a naturally-emerging pattern** — worth remembering when a future user asks the chatbot "why do Gold outlets outperform," because the honest answer is partly "because the dataset was built to guarantee it," not purely "because of real buying behavior."
13. **The earlier `BUSINESS_VALIDATION_REPORT.md` describes a data run predating the current tier/channel-multiplier retune** — several of its flagged issues (flat tier/channel effect, near-identical SKU volumes) were the specific target of the `CHANNEL_BASE_MULTIPLIER` retune and the new `reassign_outlet_tiers.py` post-processing step. Whether the *current* code fully resolves them has not been re-verified against a fresh full-scale run as part of this analysis — treat that report's numeric findings as historical, not current, but its qualitative checklist (Section 15/16 there) as still a useful test plan.

---

## 14. GraphRAG Ontology Recommendations

1. **Don't rely on GraphRAG's generic defaults again.** Phase 5's failed test used the out-of-the-box `organization, person, geo, event` categories — none of which map onto State/WD/SE/Category/SKU. Define a custom entity extraction prompt before the next indexing attempt.
2. **Disambiguate "WD" from "Dealer" explicitly in the extraction prompt** (see Section 1) — this is the single most likely silent-corruption risk given how similar the words are.
3. **Treat KPI metrics as first-class typed nodes, not just numbers inside prose.** A sentence like "Service Level was 92.9%" is easy for an LLM to extract as a description of the State node, but much harder to later compare across time/state unless the metric itself becomes a queryable node with a value and a time property (see Section 15/17).
4. **Model time as a first-class entity (Month/Period), not just a string inside a sentence.** Every document currently names its own month in prose ("August 2024") because nothing else anchors it — formalizing this as a graph node lets trend and comparison questions (Phase 6's stated "Trend" and "Comparison" question types) traverse time directly instead of re-parsing text.
5. **Add explicit causal-language sentences to the source documents before indexing**, not just descriptive ones (see Section 19) — GraphRAG's relationship extraction only finds what's stated, it doesn't compute correlations on its own.
6. **Plan for incremental indexing before scaling past 672 documents.** Today, any upstream change means reprocessing everything from scratch — fine at 672 documents, not fine once WD- or SE-level documents are added.
7. **Keep the manifest (`graphrag_docs_manifest.csv`) as the canonical source-to-document map** — it's already the right mechanism for the "answer must cite its evidence" requirement; extend it (not replace it) as new document types are added.

---

## 15. Suggested Entity Types

| Entity type | Key identifying property | Notes |
|---|---|---|
| `State` | `state_name` | Top geography node |
| `Zone` | `zone_id` | Sales-management subdivision |
| `Distributor` (WD) | `wd_id`, `wd_name` | **Rename from "WD" in the ontology itself** to avoid the Dealer-channel collision |
| `SalesExecutive` (SE) | `se_id`, `se_name` | Field sales person |
| `Outlet` | `outlet_id` | Retail point of sale |
| `Channel` | `channel_type` (Retail/Hawkers/Modern Trade/Dealer) | Modeled as its own node (not just a property) so "which channel underperforms" queries can traverse it |
| `Tier` | `tier_name` (Gold/Silver/Bronze) | Same reasoning — a node, not just an Outlet property, so tier-level aggregation is a graph traversal |
| `Category` | `category_name` | Top product grouping |
| `Franchise` | `franchise_name` | Brand family |
| `SKU` | `sku_id`, `sku_name` | Sellable product variant |
| `Period` (Month) | `year_month` | First-class time node |
| `Metric` (KPI type) | `metric_name` (Productivity, Service Level, ND, ACV, OOS%, Range Billing, Dropsize, Inventory Turns/Days, SKUs/Transaction) | The *definition* of a metric, separate from any one observed value |
| `Observation` | `value`, `unit`, references to State/Period/Category/Metric | The actual measured number — this is what Section 17 calls out in detail |
| `SupplyDisruption` (event) | `wd_id`, `sku_id`, `period` | An explicit event node, not just a boolean flag buried in inventory data |
| `Document` | `filename` | The narrative source text, kept as a node so every Observation can point back to exactly which document it came from |

---

## 16. Suggested Relationship Types

| Relationship | Connects | Direction/meaning |
|---|---|---|
| `PART_OF` | Zone→State, Distributor→Zone, SalesExecutive→Distributor, Franchise→Category, SKU→Franchise | Generic hierarchy containment |
| `OPERATES_IN` | Distributor→State | A WD's territory |
| `COVERS` | SalesExecutive→Outlet | The beat assignment |
| `CLASSIFIED_AS` | Outlet→Channel, Outlet→Tier | Cross-cutting classification, not hierarchy |
| `ELIGIBLE_FOR` | Channel→Category | The business rule from Section 4 |
| `HAS_OBSERVATION` | State→Observation, (State,Category)→Observation | Links a geography/product scope to a measured fact |
| `OF_METRIC` | Observation→Metric | Which KPI this number is |
| `OBSERVED_IN` | Observation→Period | When |
| `EVIDENCED_BY` | Observation→Document | Citation/traceability — critical for the "never answer without a source" requirement |
| `EXPERIENCED` | Distributor→SupplyDisruption | A WD had a supply shock |
| `AFFECTS` | SupplyDisruption→SKU, SupplyDisruption→Observation(Service Level/OOS%) | The causal link a "why" question needs |
| `CORRELATES_WITH` (derived, not extracted) | Metric→Metric (e.g. Productivity↔Dropsize) | A pre-computed statistical relationship worth injecting directly into the graph rather than hoping the LLM infers it from prose |
| `PRECEDES` / `FOLLOWS` | Period→Period | Enables trend traversal ("compared to last month") |

---

## 17. Suggested Node Properties

| Node type | Recommended properties |
|---|---|
| `State` | name, population, gsdp_cr, region (North/South/East/West — not currently modeled, see Section 20) |
| `Distributor` | id, name, zone, state, onboarding info (not currently tracked) |
| `SalesExecutive` | id, name, tenure (not currently tracked) |
| `Outlet` | id, name, channel_type, tier, onboarded_date, is_active, closure_date |
| `SKU` | id, name, franchise, category, unit_price, launch_date, is_hero (currently implicit, should be explicit) |
| `Metric` | name, definition_text, formula, unit, confirmation_status ("GPIL-confirmed" vs "placeholder" — directly surfacing Section 3's ⚠️ flags into the graph itself, not just this document) |
| `Observation` | value, as_of_period, scope (state / state+category), confidence, source_document_id |
| `SupplyDisruption` | wd_id, sku_id, period, severity (e.g. qty_received cut %) |
| `Document` | filename, state, period, generated_date, category_coverage |

---

## 18. Suggested Edge Properties

| Edge type | Recommended properties |
|---|---|
| `HAS_OBSERVATION` | as_of_period, category (nullable) |
| `EVIDENCED_BY` | exact source sentence/snippet (not just filename) — supports Phase 6's "cite which specific number was used" requirement |
| `AFFECTS` (disruption→metric) | magnitude, direction (worsened/improved), lag (same-month vs. following-month effect) |
| `CORRELATES_WITH` | correlation_coefficient, sample_size, computed_on (a date, since correlation could be recomputed as data grows) |
| `ELIGIBLE_FOR` | effective_from (in case eligibility rules ever change) |
| `COVERS` (SE→Outlet beat) | assigned_since (not currently tracked, but real beat plans change) |

---

## 19. Suggested Document Structure for GraphRAG Extraction

The current 672 documents (Section 20 of the design doc's own "what's not perfect yet" list) are structurally sound but intentionally minimal. To get more reliable, richer extraction:

1. **Tag entity mentions explicitly**, e.g. `State: Andhra Pradesh` / `Distributor: Rodriguez, Figueroa and Sanchez Distributors` as a labeled line before the prose paragraph, rather than relying purely on the LLM to infer type from context — this directly addresses why the Ollama test failed (generic categories couldn't map onto unlabeled prose).
2. **Add explicit month-over-month comparison sentences** — e.g. "Productivity in August 2024 (46.2%) was up from July 2024 (44.8%)." Currently every document is a standalone snapshot; trend and "why" questions would benefit enormously from this being stated, not left for the retrieval layer to reconstruct across 24 separate files.
3. **Add explicit causal sentences where the underlying data supports one** — e.g. "Service Level in Assam was lower than usual in October 2025 because Distributor X experienced a supply disruption on Marlboro Pack 9." This is buildable today: the generator already knows exactly which WD×SKU×month was disrupted; that fact is currently discarded rather than written into the document.
4. **Keep one-KPI-one-sentence, consistent subject-predicate-object phrasing** (already true — this is a genuine strength of the current design, worth preserving as new document types are added, e.g. WD-level or SE-level documents later).
5. **Consider splitting into a layered document set**: a compact "headline" document per state/month (current version) plus an optional deeper "distributor detail" or "SKU detail" document, rather than one document trying to cover every grain at once.
6. **Explicitly state what's NOT known**, where relevant (e.g., "No supply disruption was recorded for any distributor in Kerala this month") — this directly supports the stated requirement that the system should be able to say "no, nothing unusual happened" rather than only ever reporting positive findings.

---

## 20. Missing Entities and Relationships for a More Realistic Knowledge Graph

Things a real GPIL S&D knowledge graph would have that this dataset currently doesn't model at all:

1. **Competitors and market share.** There's no representation of rival cigarette/confectionery brands, so no KPI here can ever explain "why did our share drop" — only "what did we sell," never "relative to whom."
2. **Pricing and promotion events.** No excise/tax-driven price changes, no trade discounts or schemes — a huge real-world driver of monthly volume swings in Indian cigarette distribution is completely absent.
3. **Returns, damages, and expiry.** Real FMCG distribution always has some shrinkage; none is modeled here (`qty_sold` only ever goes one direction).
4. **Sales targets/quotas.** There's no "SE was supposed to sell X, sold Y" — Productivity and Dropsize describe what happened, never what was *expected* to happen, which is usually the actual trigger for a "why did we underperform" business question.
5. **Beat plan / route-to-market changes.** SE-to-outlet assignment is static for the whole 24 months; real beats get restructured, outlets get reassigned to new SEs, territories get split.
6. **Trade schemes / incentive programs for SEs or WDs.** A common real lever ("WD got a bonus for hitting X, so pushed extra volume in month Y") that would directly explain some of the order-level noise this dataset currently attributes to pure randomness.
7. **Credit terms / outstanding payments.** Distributor cash-flow health (a real constraint on how much stock a WD can hold) isn't modeled — inventory just materializes.
8. **Regulatory/excise events as explicit nodes.** Cigarette taxation changes are a first-order driver of real GPIL pricing and volume; currently there's no `RegulatoryEvent` entity at all.
9. **Weather/local events.** Real regional sales dips (floods, local elections restricting movement, festivals beyond Diwali) aren't modeled; the only "why" a state underperforms today is its fixed statistical factor.
10. **Union Territories.** Only the 28 states are modeled; a real pan-India network also covers Delhi, Chandigarh, J&K, etc.
11. **Real brand names beyond 3.** Most franchise names are generic placeholders, which limits how believable a demo answer citing "GPI_Franchise_4" will feel to an actual GPIL stakeholder.
12. **Distributor-level and SE-level KPIs.** Everything currently stops at State grain — a real S&D chatbot would need to answer "which specific distributor in Punjab is dragging down the state," which requires KPI computation one or two levels deeper than what Phase 3 currently produces.
13. **A `Region` (North/South/East/West) grouping above State.** Common in real Indian FMCG reporting for comparing broad zones, and currently entirely absent from the geography hierarchy.
14. **An explicit `MarketPotential` or `Whitespace` entity** — the gap between an outlet/state's actual performance and its theoretical potential (based on population/GDP, which the code already computes as `state_factor` but never exposes as a business-facing concept) would directly power "which state is underpenetrated relative to its potential" questions.

---

### One-paragraph summary for a business stakeholder

This dataset simulates a tobacco-and-confectionery distributor (GPIL-shaped: GPI's own cigarette brands, licensed Marlboro, licensed Ferrero confectionery, and an in-house Candy line) selling through a five-level network (State → Zone → Distributor → Sales Executive → Outlet) across all 28 Indian states, over 24 months. The big-picture business shape is realistic — cigarettes dominate revenue on modest volume, big/wealthy states outsell small ones, Diwali lifts sales — because those patterns were deliberately engineered in. The finer-grained realism (does a Gold outlet actually behave differently from a Bronze one, does a supply disruption visibly explain a bad month) is an ongoing engineering effort, not yet independently verified end-to-end, and several core metric definitions (Dropsize, Range Billing, ACV weighting) are explicitly unconfirmed placeholders awaiting real GPIL sign-off. Before this becomes a trustworthy knowledge graph, the main risks are (a) an LLM extractor conflating "Distributor" with "Dealer," (b) causal "why" answers being invented rather than retrieved, since the current documents describe *what* happened but rarely *why*, and (c) treating placeholder metric definitions as settled fact.
