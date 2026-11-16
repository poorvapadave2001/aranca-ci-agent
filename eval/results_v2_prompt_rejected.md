# Evaluation results

Labeled sample: **38 items** (dev 38, test 0). Dashboard threshold: relevance >= 1. Intervals are 95% Wilson CIs.

Label sources: {'auto_llm': 23, 'hand': 15}

> WARNING: some labels were produced by an automatic judge, not by hand. Say so when quoting these numbers.

> 7 labeled items have no agent output (cleared or not analysed yet) and are excluded.

## Agent vs baselines

| System | Shown | Precision | Recall | Noise rejected | F1 |
|---|---|---|---|---|---|
| **Agent (LLM triage)** | 31 | 61% (44%-76%) | 83% (63%-93%) | 20% (7%-45%) | 0.70 |
| Rule judge (no LLM) | 29 | 52% (34%-69%) | 65% (45%-81%) | 7% (1%-30%) | 0.58 |
| Show every keyword hint | 38 | 61% (45%-74%) | 100% (86%-100%) | 0% (0%-20%) | 0.75 |

Agent confusion: TP=19 FP=12 FN=4 TN=3

## Relevance threshold sweep (what the dashboard filter does)

| Show rel >= | Shown | Precision | Recall | Noise rejected | F1 |
|---|---|---|---|---|---|
| 1 | 31 | 61% (44%-76%) | 83% (63%-93%) | 20% (7%-45%) | 0.70 |
| 2 | 13 | 77% (50%-92%) | 43% (26%-63%) | 80% (55%-93%) | 0.56 |
| 3 | 11 | 73% (43%-90%) | 35% (19%-55%) | 80% (55%-93%) | 0.47 |
| 4 | 0 | n/a | 0% (0%-14%) | 100% (80%-100%) | 0.00 |

## Dev vs held-out test

| Split | Shown | Precision | Recall | Noise rejected | F1 |
|---|---|---|---|---|---|
| dev (tuned on) n=38 | 31 | 61% (44%-76%) | 83% (63%-93%) | 20% (7%-45%) | 0.70 |
| test (held out) | n=0 (<10, not scored) |  |  |  |  |
| all n=38 | 31 | 61% (44%-76%) | 83% (63%-93%) | 20% (7%-45%) | 0.70 |

## Category accuracy (true positives only)

Agent: 57% (37%-74%)   |   keyword hint: 70% (49%-84%)   (n=23)

| True category | n | Agent accuracy | Note |
|---|---|---|---|
| Digital Presence | 21 | 62% (41%-79%) |  |
| M&A | 2 | 0% (0%-66%) | indicative only (n<5) |

Confusion matrix (rows = your label, columns = agent):

```
agent_cat2        Business Update  Deal  Digital Presence  Noise  All
true_cat                                                             
Digital Presence                1     3                13      4   21
M&A                             0     2                 0      0    2
All                             1     5                13      4   23
```

## Do the automatic judges agree with your hand labels?

- Rule judge: 7% raw agreement, Cohen's kappa 0.00 (n=15)
- LLM judge: 27% raw agreement, Cohen's kappa -0.00 (n=15)

Kappa above ~0.6 means the judge can reasonably extend your labels; below that, keep labeling by hand.

Items with source link and publication date: 100%

## Error analysis

False positives (shown but irrelevant):

- Syntax Partners: エレクトロニクス商社業界M&Aの最新動向と事例 ：特約店網の再編と商社の統合 (agent: Digital Presence, rel 3)
- Integration Consulting: Integration welcomes Marília José as Partner (agent: Deal, rel 1)
- Infomineo: Automated Data Extraction: What It Handles Well, Where It Breaks, and How to Check It (agent: Digital Presence, rel 1)
- Infomineo: Legacy System Integration: How to Connect Old Systems Without Replacing Them (agent: Digital Presence, rel 1)
- MarketBridge: Marketbridge Acquires Channel 99, Launches AI Visibility for B2B Marketers - Demand Gen Re (agent: Deal, rel 3)
- Integration Consulting: M&A Integration Consulting (agent: Business Update, rel 1)
- Avalon Consulting: AI Changing the ICT Landscape: A Business Perspective for CXOs (agent: Digital Presence, rel 1)
- NMS Consulting: How to Hire an AI Governance Consultant: NIST AI RMF, EU AI Act, ISO 42001, Controls, and  (agent: Digital Presence, rel 3)
- Infomineo: Geospatial Mapping: How Businesses Turn Location Data Into Decisions (agent: Digital Presence, rel 1)
- Infomineo: Is AI as Good in Arabic as in English? What the Latest Benchmarks Show (agent: Digital Presence, rel 1)

False negatives (relevant but hidden):

- MarketBridge: From Brand Awareness to Pipeline: How Marketbridge Grew Avalara Share of Voice In One Spri (agent: Noise, rel 0)
- Pnyx Hill: UAE Investment in Greece: The Strategic Logic Beyond the €15 Billion Announcement (agent: Noise, rel 0)
- Syntax Partners: タイ子会社・合弁持分の売却：日本企業の撤退・再編の最新動向と事例 (agent: Noise, rel 0)
- Syntax Partners: 半導体製造装置業界M&Aの最新動向と事例 ：大手グループからの切り出しと外為法の下での資本の選択 (agent: Noise, rel 0)

Wrong category on true positives:

- MarketBridge: From Brand Awareness to Pipeline: How Marketbridge Grew Avalara Share of Voice I (you: Digital Presence | agent: Noise)
- Pnyx Hill: UAE Investment in Greece: The Strategic Logic Beyond the €15 Billion Announcemen (you: Digital Presence | agent: Noise)
- Saudi Market Research Consulting: Saudi Arabia Dorra Gas Field Capex Analysis Report (you: Digital Presence | agent: Deal)
- Syntax Partners: タイ子会社・合弁持分の売却：日本企業の撤退・再編の最新動向と事例 (you: Digital Presence | agent: Noise)
- Saudi Market Research Consulting: Saudi Arabia Data Center Construction Market Guide (you: Digital Presence | agent: Deal)
- Tecnova India Pvt. Ltd.: India Agrochemical Market Entry: Investment and Growth Opportunities - Tecnova (you: Digital Presence | agent: Business Update)
- MarketBridge: Marketbridge Acquires The B2B Advertising And Analytics Startup Channel99 - AdEx (you: M&A | agent: Deal)
- Syntax Partners: 半導体製造装置業界M&Aの最新動向と事例 ：大手グループからの切り出しと外為法の下での資本の選択 (you: Digital Presence | agent: Noise)

## History (compare versions)

| tag | n | min_rel | precision | recall | noise_rejected | f1 | cat_acc |
|---|---|---|---|---|---|---|---|
| baseline_auto_v1 | 45 | 1 | 0.931 | 0.711 | 0.714 | 0.806 | 0.553 |
| baseline_v1 | 45 | 1 | 0.793 | 1.0 | 0.727 | 0.885 | 0.783 |
| v2 | 38 | 1 | 0.613 | 0.826 | 0.2 | 0.704 | 0.565 |