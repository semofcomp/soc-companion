# Exhibit — the economy × domain 2×2 (cross-survival)

The co-primary framing rests on two facts, both shown here: (1) the two censuses *tilt* toward
different halves of the general construct, and (2) the finding survives in **both** halves of
**both** censuses. Fact (2) — the cross-survival — is the engine of the generality claim.

All numbers: general-rubric census, gpt-4o primary, anisotropy-corrected (mean-centered) embeddings.
"comp-vs-sub cosine AUC" = P(complement cosine > substitute cosine); **< 0.50 = the pathology**
(cosine ranks complements *below* substitutes). "probe AUC" = supervised readout recovering the
complement/substitute split; **> 0.50 = recovery**. Transformer = gemini-embedding-001 / text-embedding-3-large.

## 1. Each census tilts toward a different domain

Labels are the **cross-vendor consensus** (gpt-4o ∩ gemini-2.5-pro): consumer-final = both vendors
agree consumer-final; production = both agree non-consumer; disagreements = REVIEW (dropped from both).
Cross-vendor agreement is high — **CA 93.1%, US 86.8%** — which itself validates the labeling.

| Census | Consumer-final | Production | REVIEW (dropped) |
|---|---|---|---|
| **Canada** (NAPCS 2022, consumption-framed names) | 908 (30%) | 1,931 (63%) | 209 (7%) |
| **United States** (NAPCS collection, ~66% "Manufacturing of X") | 449 (13%) | 2,608 (74%) | 463 (13%) |

Both censuses are production-majority in absolute terms — so the claim is *relative*: Canada is the
**richer consumption exemplar**, the US the **production exemplar**. Under the consensus cut, production
is a **positively-labeled class** (both vendors agree non-consumer), not a residual. The looser
single-model gpt-4o cut gives the same tilt (970/2,079 CA; 748/2,772 US).

## 2. The finding survives in every cell (economy × domain) — cross-vendor consensus cut

| Census (role) | Domain subset | n cats | complement pairs | comp-vs-sub cosine AUC (gem / ope) | gemini probe AUC |
|---|---|---|---|---|---|
| Canada (consumption exemplar) | consumer | 908 | 2,309 | 0.069 / 0.071 | 0.84 |
| Canada | production | 1,931 | 8,859 | 0.051 / 0.059 | 0.90 |
| United States (production exemplar) | consumer | 449 | 884 | 0.114 / 0.110 | 0.82 |
| United States | production | 2,608 | 12,004 | 0.116 / 0.123 | 0.85 |

Reading: in all four cells the transformer cosine AUC is far **below 0.50** (0.05–0.12) — complements
sit below substitutes under cosine, the pathology — and the supervised probe is far **above 0.50**
(0.82–0.90) — the learned readout recovers complementarity. **Robust to the labeling method:** the
single-model gpt-4o cut gives the same pattern (CA consumer 0.07/0.07, CA production 0.05/0.06,
US consumer 0.12/0.13, US production 0.12/0.13), so the result does not depend on how the consumer/
production line is drawn. TF-IDF/GloVe show the same direction (cosine AUC 0.13–0.20).

## 3. Why this drives the thesis home

The two off-diagonal cells are the payoff. Canada — the consumption exemplar — still shows the
finding on its **production** subset (cosine 0.05, probe 0.90). The US — the production exemplar —
still shows it on its **consumption** subset (cosine 0.11, probe 0.82), and under the consensus cut
consumption is only ~13% of the US census — so this is the finding holding where the domain is a small
*minority*. Complementarity
being off the cosine axis, and recoverable by a learned readout, is therefore not an artifact of one
classification's framing: it reproduces across consumption- and production-framed taxonomies and
across two national classifications built by different statistical agencies.

## 4. Honest seams (state these in the text, don't bury)

- **Exemplar ≠ validation source.** The US is the production *exemplar census*, but its production
  *validation* (BEA 2017) is the softer one (Claim B near chance from a concentrated NAPCS↔BEA
  crosswalk). The clean production validation is Canada's (StatCan I-O 2023). Frame production
  complementarity as validated economy-natively in Canada and *reproduced* in the US with documented
  crosswalk limits.
- **US consumption-side concordance is the one grey cell** (Amazon relation-concordance κ ≈ 0.01–0.02),
  measurably caused by US Amazon-crosswalk collapse (44% of US categories share a node vs 15% in
  Canada; on the clean subset US κ = 1.0 on n=6). It is a *consumption*-side test for the
  *production* exemplar — off its primary role — and Canada's consumption concordance is clean (0.31).
- **One thesis spine.** The two co-primary censuses are two instruments for a single contribution
  (complementarity is off-axis and general across consumption and production), not two parallel studies.

## 5. Method notes (how the cells are built)

- **How a category is labeled.** LLM consumer-final classifier (criterion: acquired by households for
  final use vs intermediate/capital/wholesale/business) run under TWO vendors, gpt-4o and gemini-2.5-pro;
  the cut is the **cross-vendor consensus** (consumer = both agree consumer-final; production = both agree
  non-consumer, a positively-labeled class; disagreements dropped). A complement-neighborhood audit (label
  a focal from its complements' labels) was considered and **deliberately not applied** — it would entangle
  the split with the complement graph it partitions, and complementarity legitimately crosses the
  consumption/production line. Labels stay a per-category economic construct.
- **How a cell is built (conservative).** Each cell is restricted to its census's categories with that
  label. A focal→complement pair enters a cell only if **both** endpoints carry the label; substitute
  (taxonomic) and random anchors are **re-derived within each subset**; **cross-domain pairs are excluded**
  from all cells. So "holds within consumption" / "holds within production" are uncontaminated by the
  cross-domain links — the test is deliberately conservative.
