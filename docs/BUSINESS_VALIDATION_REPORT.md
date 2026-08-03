# GPIL Synthetic Data — Business Validation Report

**Prepared for:** GPIL Sales & Distribution GraphRAG project
**Data checked:** `data/geography.csv`, `outlets.csv`, `products.csv`, `visits.csv`, `orders.csv`, `inventory_snapshots.csv` (Phase 2 output)
**Period covered:** August 2024 – July 2026 (24 months)
**Purpose:** Look at the synthetic data the way a GPIL business/sales analyst would — not the way a data engineer would — and flag anything that does not look like real cigarette/FMCG distribution behaviour, before this data is used to build the GraphRAG chat system.

**Note on currency:** The data does not label its currency. Since this mimics GPIL (an Indian company), all money figures below are shown in **₹ Crore** (1 Crore = ₹1,00,00,000) for easy reading, assuming the `unit_price` column is in Rupees.

---

## 1. Total Rows in Each Table

| Table | Rows | What it holds |
|---|---|---|
| `geography.csv` | 1,654 | States, Zones, Wholesale Distributors (WD), Sales Executives (SE) |
| `outlets.csv` | 121,614 | Retail outlets (shops) |
| `products.csv` | 63 | SKUs (products) |
| `visits.csv` | 3,938,483 | Sales Executive visits to shops |
| `orders.csv` | 8,310,317 | Order line items (one row per product per order) |
| `inventory_snapshots.csv` | 214,704 | Monthly stock position per Distributor per SKU |

**First observation:** For a company the size of GPIL, 63 SKUs is a very small product list. A real cigarette + confectionery distributor usually carries 200–500+ SKUs across brands, pack sizes, and regional variants. This is fine for a first synthetic pass, but it limits how rich SKU-level questions can be later.

---

## 2. Category-wise Sales Volume

| Category | Units Delivered | Revenue (₹ Cr) | Revenue Share | Volume Share |
|---|---|---|---|---|
| GPI (cigarettes) | 5.23 Cr units | 1,271.76 | 58.2% | 54.3% |
| IPM (Marlboro) | 2.09 Cr units | 798.44 | 36.6% | 21.7% |
| Ferrero (TicTac, Kinder Joy) | 1.32 Cr units | 66.67 | 3.1% | 13.7% |
| Candy | 0.99 Cr units | 47.17 | 2.2% | 10.3% |

**Reads correctly:** Cigarettes (GPI + IPM) driving ~95% of revenue while being ~76% of volume matches a real tobacco distributor — cigarettes are low-volume-but-high-value compared to candy. This part of the data is believable.

---

## 3. Channel-wise Sales Volume

| Channel | Outlets | Units Delivered | Revenue (₹ Cr) | Revenue Share |
|---|---|---|---|---|
| Kirana/General Store | 48,669 | 3.86 Cr | 874.68 | 40.0% |
| Paan/Tobacconist | 42,513 | 3.36 Cr | 762.85 | 34.9% |
| Modern Trade | 18,236 | 1.45 Cr | 327.54 | 15.0% |
| Dealer | 12,196 | 0.97 Cr | 218.97 | 10.0% |

**Looks reasonable on the surface** — Kirana and Paan shops dominate volume, which matches how cigarettes are actually sold in India (small daily-purchase outlets, not big-box stores). But see Section 11 — the *reason* the shares look this way is not what a real business would expect.

---

## 4. State-wise Sales Volume

All 28 states, ranked by revenue:

| Rank | State | Units Delivered | Revenue (₹ Cr) | % Delivered vs Ordered |
|---|---|---|---|---|
| 1 | Bihar | 54.62 lakh | 123.64 | 92.5% |
| 2 | Himachal Pradesh | 50.38 lakh | 114.15 | 92.8% |
| 3 | Kerala | 46.59 lakh | 105.76 | 92.7% |
| 4 | Meghalaya | 46.64 lakh | 105.56 | 92.8% |
| 5 | Assam | 45.42 lakh | 102.79 | 92.7% |
| 6 | Nagaland | 43.05 lakh | 97.45 | 92.7% |
| 7 | Madhya Pradesh | 42.55 lakh | 96.33 | 92.8% |
| 8 | Haryana | 37.92 lakh | 85.94 | 93.0% |
| 9 | Chhattisgarh | 37.40 lakh | 84.82 | 93.0% |
| 10 | Gujarat | 37.38 lakh | 84.78 | 92.7% |
| 11 | Rajasthan | 36.56 lakh | 82.89 | 92.8% |
| 12 | Uttar Pradesh | 35.66 lakh | 80.84 | 92.7% |
| 13 | Tamil Nadu | 34.91 lakh | 79.23 | 92.8% |
| 14 | Punjab | 34.86 lakh | 79.02 | 92.7% |
| 15 | Manipur | 33.97 lakh | 77.07 | 92.8% |
| 16 | Sikkim | 33.19 lakh | 75.24 | 92.7% |
| 17 | West Bengal | 31.41 lakh | 71.10 | 92.7% |
| 18 | Maharashtra | 31.19 lakh | 70.76 | 93.1% |
| 19 | Odisha | 31.19 lakh | 70.54 | 92.5% |
| 20 | Karnataka | 30.79 lakh | 69.86 | 92.7% |
| 21 | Tripura | 27.64 lakh | 62.74 | 92.9% |
| 22 | Andhra Pradesh | 26.80 lakh | 60.70 | 93.0% |
| 23 | Telangana | 24.63 lakh | 55.90 | 93.0% |
| 24 | Arunachal Pradesh | 24.26 lakh | 55.01 | 93.0% |
| 25 | Mizoram | 24.09 lakh | 54.66 | 92.8% |
| 26 | Jharkhand | 23.73 lakh | 53.68 | 93.0% |
| 27 | Goa | 21.60 lakh | 49.04 | 92.7% |
| 28 | Uttarakhand | 15.23 lakh | 34.51 | 92.7% |

**Total across all 28 states: ₹2,184.04 Cr revenue.**

**Good, believable signal:** The best state (Bihar) sells about **3.6x** more than the worst state (Uttarakhand). This kind of state-to-state gap is realistic — some states genuinely are bigger tobacco markets, and this spread was deliberately built in via a per-state "performance factor." This is one of the better-designed parts of the dataset.

---

## 5. Monthly Trend

| Month | Revenue (₹ Cr) |
|---|---|
| Aug 2024 | 87.29 |
| Sep 2024 | 86.94 |
| **Oct 2024** | **98.90** |
| **Nov 2024** | **103.35** |
| Dec 2024 | 95.74 |
| Jan 2025 | 95.97 |
| Feb – Sep 2025 | ~86–87 each month (flat) |
| **Oct 2025** | **99.21** |
| **Nov 2025** | **103.37** |
| Dec 2025 | 96.08 |
| Jan 2026 | 96.12 |
| Feb – Jul 2026 | ~87–88 each month (flat) |

**Clear festive bump every October–November** (Diwali season), tapering into a smaller Dec–Jan bump (New Year), then flat for the rest of the year. This pattern **repeats identically in both years** — same shape, same size. Real seasonality is close to this but usually has some year-on-year growth or variation; here Year 2 is almost a photocopy of Year 1 (see Section 13).

---

## 6. SKU-wise Volume

This is where the first real problem shows up. Across all 63 SKUs, delivered volume ranges only from **14.83 lakh units to 16.58 lakh units** — every single SKU sells almost the same quantity, regardless of brand, category, or price. The gap between the best-selling and worst-selling SKU is only **1.12x**.

In a real cigarette/FMCG business, a handful of "hero" SKUs (a bestselling Marlboro variant, GPI's top-selling economy brand) would sell **5–10 times more** than a slow-moving SKU. That gap simply does not exist here — volume looks like it was handed out almost equally to every SKU rather than being driven by actual customer demand.

---

## 7. Top 10 SKUs (by revenue)

| SKU | Category | Units Delivered | Revenue (₹ Cr) |
|---|---|---|---|
| Marlboro Pack 9 | IPM | 14.98 lakh | 71.97 |
| Marlboro Pack 10 | IPM | 14.87 lakh | 71.80 |
| Marlboro Pack 14 | IPM | 15.02 lakh | 66.03 |
| Marlboro Pack 4 | IPM | 14.85 lakh | 63.06 |
| Marlboro Pack 8 | IPM | 15.02 lakh | 62.50 |
| Marlboro Pack 11 | IPM | 14.95 lakh | 60.13 |
| GPI_Franchise_6 Pack 1 | GPI | 14.92 lakh | 55.97 |
| Marlboro Pack 12 | IPM | 14.98 lakh | 55.75 |
| GPI_Franchise_3 Pack 5 | GPI | 14.97 lakh | 54.72 |
| Marlboro Pack 2 | IPM | 14.87 lakh | 54.35 |

Notice the "Units Delivered" column barely changes across all 10 rows — these SKUs are on top **only because they happen to be priced higher**, not because more people bought them.

---

## 8. Bottom 10 SKUs (by revenue)

| SKU | Category | Units Delivered | Revenue (₹ Cr) |
|---|---|---|---|
| Kinder_Joy Pack 4 | Ferrero | 16.49 lakh | 9.83 |
| TicTac Pack 1 | Ferrero | 16.43 lakh | 8.98 |
| Candy_Franchise_2 Pack 1 | Candy | 16.38 lakh | 8.18 |
| Candy_Franchise_1 Pack 2 | Candy | 16.45 lakh | 6.06 |
| Kinder_Joy Pack 1 | Ferrero | 16.45 lakh | 4.65 |
| Kinder_Joy Pack 3 | Ferrero | 16.58 lakh | 4.60 |
| Candy_Franchise_1 Pack 1 | Candy | 16.51 lakh | 3.56 |
| Candy_Franchise_1 Pack 3 | Candy | 16.55 lakh | 2.79 |
| Kinder_Joy Pack 2 | Ferrero | 16.46 lakh | 2.55 |
| TicTac Pack 3 | Ferrero | 16.56 lakh | 1.85 |

Same story in reverse — the "bottom" SKUs actually sold **slightly more units** than most of the "top" SKUs. They only rank at the bottom because candy and TicTac are cheap per unit. **There is no real weak-seller in this data — every SKU sells about the same.**

---

## 9. Outlet Tier Contribution

| Tier | Outlets | % of Outlets | Revenue (₹ Cr) | % of Revenue |
|---|---|---|---|---|
| Gold | 18,376 | 15.1% | 330.33 | 15.1% |
| Silver | 42,453 | 34.9% | 762.01 | 34.9% |
| Bronze | 60,785 | 50.0% | 1,091.70 | 50.0% |

**This is the single biggest issue in the whole dataset.** A Gold outlet is supposed to be a bigger, better-performing shop than a Bronze one — that is the entire point of a tier system. But Gold outlets sell **exactly the same average amount per outlet** as Silver and Bronze outlets (revenue share = outlet-count share, to the decimal point). The "Gold/Silver/Bronze" label currently does nothing except decide a weight used in one KPI formula — it has no effect on how much an outlet actually buys.

---

## 10. Channel Contribution

| Channel | Outlets | % of Outlets | Revenue (₹ Cr) | % of Revenue |
|---|---|---|---|---|
| Kirana/General Store | 48,669 | 40.0% | 874.68 | 40.0% |
| Paan/Tobacconist | 42,513 | 35.0% | 762.85 | 35.0% |
| Modern Trade | 18,236 | 15.0% | 327.54 | 15.0% |
| Dealer | 12,196 | 10.0% | 218.97 | 10.0% |

Same problem as Section 9, at the channel level. A Modern Trade store (a supermarket-type outlet) should sell much more per outlet than a small Paan shop, and a Dealer (who resells in bulk to smaller shops) should sell far more per outlet than either. Instead, **average revenue per outlet is the same ₹1.79–1.80 lakh no matter what type of shop it is.** The channel split in Section 3 is real, but it comes purely from *how many* outlets of each type exist — not from any real difference in buying behaviour between channel types.

---

## 11. Product Mix

| Category | Revenue Share | Volume Share | Avg. Realised Price/Unit |
|---|---|---|---|
| GPI | 58.2% | 54.3% | ₹243 |
| IPM (Marlboro) | 36.6% | 21.7% | ₹382 |
| Ferrero | 3.1% | 13.7% | ₹50 |
| Candy | 2.2% | 10.3% | ₹48 |

Cigarettes (GPI + IPM) make up **~94.9%** of revenue on **~76%** of volume — this mix is directionally correct for a tobacco-led distributor with a confectionery side business. This is one of the more believable business signals in the data.

One catalogue oddity: pack prices don't follow a logical size ladder — e.g. among GPI_Franchise_1's 7 SKUs, "Pack 3" (₹199) is cheaper than "Pack 1" (₹232), which is cheaper than "Pack 4" (₹282), with no consistent pattern. Real product catalogues usually price in an ascending ladder by pack size (10s cheaper than 20s, etc.), whereas here `pack_size` is just a label ("Variant 1..N") not an actual quantity, so prices look randomly shuffled.

---

## 12. Seasonality

- **Festive spike (Oct–Nov) is present and repeats every year** — a good sign, this is real Diwali-season behaviour.
- **Problem:** The spike hits **all four categories equally**, including GPI cigarettes and Marlboro. In real life, cigarette buying is a habitual, non-seasonal purchase — it doesn't jump for Diwali the way gifting categories (Ferrero, Candy) do. Right now the model applies one uniform seasonal multiplier to everything.
- **Problem:** Out-of-Stock % stays flat (~7–9%) all year, even during the Oct–Nov demand spike. In a real distribution network, a demand spike without a matching supply push causes *more* stockouts during peak season — here supply seems to scale perfectly with demand with no strain at all, which is unrealistically smooth.
- **Problem:** Year 1 (Aug 2024–Jul 2025) and Year 2 (Aug 2025–Jul 2026) monthly revenue are near carbon copies of each other, month for month. Real businesses show some year-on-year growth, decline, or at least month-to-month noise layered on top of the seasonal shape — this data is a clean repeating template.

---

## 13. Strange Patterns / Anomalies (Summary)

Ranked by how much they'd worry a real business analyst:

1. **Outlet tier has zero effect on sales** (Section 9) — Gold, Silver and Bronze outlets sell identically per outlet on average. This defeats the purpose of having a tier system at all.
2. **Channel type has zero effect on sales** (Section 10) — Dealer, Modern Trade, Kirana, and Paan shops all sell the same average amount per outlet. A wholesale Dealer should move far more stock than a single Paan shop.
3. **No SKU has a real demand advantage** (Sections 6–8) — every one of the 63 SKUs sells within a narrow ±11% band of every other SKU. There is no hero product and no dead stock, which almost never happens in real FMCG/tobacco.
4. **Fulfilment shortfall is exactly the same everywhere** — about 19.4% of ordered quantity is not delivered, and this rate barely moves between states, channels, tiers, or categories (all cluster at 92.5–93.1% fulfilled). In practice, top-tier outlets and core cigarette SKUs are usually prioritised during any shortage and see far better service than the tail — here everyone is short-changed by the same fixed amount.
5. **Prices never move in 24 months** — every order line uses the exact catalogue price from `products.csv`, with zero variation over time. Real cigarette pricing changes with excise/tax hikes (which happen almost every year in India), and confectionery runs frequent trade promotions/discounts. No price ever changes here.
6. **"Inactive" outlets keep transacting** — 6,075 outlets are flagged `is_active = False`, but they still generate visits and orders across the *entire* 2-year window, same as active outlets. An inactive/closed outlet shouldn't be getting sales calls at all.
7. **Outlet onboarding is a flat line** — roughly the same number of outlets (~12,500–12,800) get onboarded every year from 2015 to 2023 with no acceleration, slowdown, or dip (e.g. around COVID-era disruption in 2020–21 would realistically show a dent). It reads as a random date generator, not a business growth story.
8. **Out-of-Stock % is flat across categories** (~7.9–8.2% for all four) — in real tobacco distribution, cigarettes (GPI/IPM) usually get near-zero stockouts because they're the distributor's top priority revenue line, while slower-moving confectionery is more likely to run out. Here all categories are equally (un)stocked.
9. **Numeric Distribution and ACV are almost identical numbers** (e.g. GPI: ND = 0.6184, ACV = 0.6184) — these are meant to be two different metrics (ACV should be sales-weighted by outlet importance, ND is a simple outlet-count measure), but in this data they move together almost exactly, suggesting the tier-weighting in the ACV formula isn't actually creating any separation from ND — consistent with Finding #1 above.

---

## 14. Business Observations

- The **big-picture shape is right**: cigarettes dominate revenue but not volume, confectionery is a small side business, Bihar-type large states outsell smaller states 3-4x, and Diwali season lifts sales. A business reader would recognise this as "roughly GPIL-shaped."
- The **micro-level behaviour is where it breaks down**: nothing that should matter at the outlet or SKU level (tier, channel, brand strength, pricing power) actually changes outcomes. Every outlet of every type behaves like every other outlet; every SKU sells like every other SKU. Demand looks like it was assigned per outlet with a single random draw, not shaped by outlet type or product appeal.
- This matters a lot for the GraphRAG project specifically: if a user asks "why do Gold outlets underperform in Bihar?" or "which SKU should we push harder in Modern Trade?", there is currently **no real signal in the data to answer that kind of diagnostic question truthfully** — the honest answer would always be "there's no difference," which isn't useful for demonstrating the inference layer's value.

---

## 15. Suggested Improvements (to make the data feel like real GPIL business)

1. **Make outlet tier drive volume.** Give Gold outlets a materially higher average order size/frequency than Silver, and Silver higher than Bronze (e.g. Gold ≈ 2.5–3x Bronze), not just a KPI-formula weight.
2. **Make channel type drive volume.** Dealers should move bulk quantities (they resell onward); Modern Trade should have larger basket sizes than Paan shops. Right now all four channels are statistically identical.
3. **Add a real Pareto curve to SKUs.** Pick 2-3 "hero" SKUs per category that consistently outsell the rest by 5-10x, and let a long tail of SKUs sell much less — mirrors real 80/20 sales concentration.
4. **Let fulfilment/service level vary by priority.** Core GPI/IPM SKUs and Gold outlets should see better fulfilment than tail SKUs/Bronze outlets, especially during the festive demand spike.
5. **Restrict the festive multiplier to gifting categories** (Ferrero, Candy) rather than applying it to cigarette categories too.
6. **Let Out-of-Stock % rise during the Oct-Nov demand spike**, instead of staying flat, to show real supply-chain strain.
7. **Introduce at least one price change** over the 24 months (an excise-driven cigarette price hike is very realistic for GPIL) and/or occasional promotional pricing for confectionery.
8. **Stop transactions for inactive outlets** after their (currently missing) closure date, instead of letting them keep ordering.
9. **Add some year-on-year variation** to the seasonal pattern (a few % growth/decline, or added month-to-month noise) so Year 2 isn't a repeat of Year 1.
10. **Grow the SKU catalogue** — even 100-150 SKUs (more GPI pack variants, a couple more IPM/Ferrero packs) would support richer "which SKU" questions later.
11. **Give pack sizes real, ordered meaning** (e.g. actual stick-count or weight) so price naturally increases with pack size, instead of random Variant labels.
12. **Add a small COVID-era dent** or similar real-world disruption to the outlet onboarding timeline for extra realism (optional, lower priority).

---

## 16. 20 Business Questions GraphRAG Should Be Able to Answer

These are the kind of natural questions a GPIL sales/ops manager would actually ask the chat system. They mix simple "what happened" lookups with "why did it happen" diagnostic questions — both need to be answerable once the data has real signal (per Section 15).

1. Which state had the highest cigarette sales in October 2025, and how does that compare to the previous month?
2. Why did Productivity drop in a particular state in a given month — was it more "No Order" visits or more "Closed" outlets?
3. Which SKU has the worst Out-of-Stock % this quarter, and in which state is it worst?
4. How does Numeric Distribution for Marlboro compare across states — where is GPIL underpenetrated?
5. Which channel (Kirana, Paan, Modern Trade, Dealer) contributes the most to Ferrero sales, and has that changed month over month?
6. What is the festive-season (Oct-Nov) sales uplift for Candy versus GPI cigarettes, and is the gap what we'd expect?
7. Which Wholesale Distributor has the most stockout months in the last six months, and for which SKUs?
8. Why is Service Level lower in one state compared to a neighbouring state — is it a fulfilment issue or an ordering issue?
9. Which outlet tier (Gold/Silver/Bronze) is under-contributing relative to its outlet count, and in which states?
10. How many SKUs are being actively billed (Range Billing) at Gold outlets versus Bronze outlets?
11. What is the trend in Inventory Turns for GPI cigarettes over the last 12 months, and is it improving or worsening?
12. Which state shows the biggest gap between Numeric Distribution and Range Billing for IPM — meaning shops stock it but don't sell the full range?
13. Compare ACV between two named states for the Candy category — which is performing better and why might that be?
14. Which months in the year consistently show the lowest Productivity across all states?
15. Is there a state where Out-of-Stock % is rising month-on-month for a specific category — is that a supply issue or a demand spike?
16. Which SKU had the sharpest month-over-month volume decline, and did it happen everywhere or just in specific states?
17. How does Inventory Days for Ferrero compare between a Northern state and a Southern state?
18. What share of total revenue comes from the top 5 states versus the bottom 5 states, and is that gap growing or shrinking?
19. For a given Wholesale Distributor, which SKUs consistently run into stockouts, and could that be linked to low Inventory Turns?
20. If Modern Trade's contribution to Candy sales dropped in a quarter, was that driven by fewer visits, lower Productivity, or lower fulfilment?

---

## Summary for the Team

The synthetic data gets the **big picture right** (cigarette-led revenue mix, state-level spread, festive seasonality) but is **flat at the micro level** — outlet tier, channel type, and individual SKUs currently make no real difference to sales outcomes. Before building the GraphRAG "why" layer on top of this, it's worth deciding whether to patch the generator (Section 15) so diagnostic questions have real answers, or to proceed as-is and treat this as a structural/format proof-of-concept rather than a business-realistic demo dataset.
