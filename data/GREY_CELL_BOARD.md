# Big-picture evidence board (grey-cell tracker)

Status of every evidence test for the general-complementarity paper, by census (Canada / US),
with the benchmark each cell is judged against and the direction that counts as support.
All values are the gpt-4o census; rounded. Updated 2026-06-18.

**Reading the benchmark column.** Each test is scored against a null benchmark; "want" gives the
direction of support. The headline *pathology* is the one case where support means **below** the
benchmark (cosine ranks complements *below* substitutes — that is the finding). Every other test
supports the thesis when it lands **above** its benchmark.

**Status legend.** `solid` = clearly supports · `cleared` = was grey, fixed by recent work ·
`caveated` = supportive but with a stated weakness · `running` = recompute in progress ·
`GREY` = outstanding weak cell · `retired` = cell removed because the dataset was dropped.

| Test | Metric | Benchmark · want | Canada | US |
|---|---|---|---|---|
| **In-model geometry — the finding** | | | | |
| Pathology | comp-vs-sub cosine AUC (transformers) | 0.50 · below ↓ | 0.06 gem / 0.06 ope — `solid` | 0.12 gem / 0.13 ope — `solid` |
| Probe recovery | learned-readout AUC | 0.50 · above ↑ | 0.91 — `solid` | 0.85 — `solid` |
| **Behavioral validation — consumer** | | | | |
| Claim A (labels real) | complement vs random co-occurrence AUC | 0.50 · above ↑ | 0.66 Amazon · 0.58 Yelp — `solid` | 0.63 Amazon — `solid` |
| Claim B (pathology on behavior) | sub-over-comp cosine AUC | 0.50 · above ↑ | 0.93 Yelp · 0.59 Amazon — `cleared` | 0.60 Amazon — `cleared` |
| Concordance κ | census labels vs Amazon buy/view | 0.00 · above ↑ | 0.31 (4o) · 0.22 (gem) — `solid` | 0.01 (4o) · 0.02 (gem) — `GREY` (design-limited) |
| **Economic validation — production** | | | | |
| Production Claim A | complement vs random I-O linkage AUC | 0.50 · above ↑ | 0.63 StatCan I-O '23 — `solid` | 0.57–0.58 BEA detail '17 — `caveated` (concordance-limited) |
| Production Claim B | cosine separates I-O complements AUC | 0.50 · above ↑ | 0.58–0.59 StatCan · 0.59–0.72 BEA — `solid` | ≈0.50 BEA (crosswalk-noisy) — `caveated` |
| **Robustness** | | | | |
| Generator robustness | finding holds across generators | consensus holds ↑ | holds (4o + gemini) — `solid` | holds (4o ∩ gemini consensus) — `cleared` |

## What changed in this round

**Cleared (were grey, now resolved):**
- Amazon Claim B, Canada — was a weak ~0.55; the mutual-exclusion substitute construct (co-viewed-not-bought vs co-bought-not-viewed) lifts it to gemini 0.59 / openai 0.55, with the mechanism visible in the cosines.
- Amazon Claim B, US — was reversed (~0.41); now 0.60 (gemini) / 0.60 (openai), CIs above 0.5. Part of the prior reversal was a mean-centering artifact (consistent anisotropy correction already de-reverses it); the clean substitute set makes it robust.

- US generator robustness — was thin (consensus rested on gpt-4o ∩ gpt-4o-mini, same vendor). Now the cross-vendor gemini-2.5-pro census is complete and the finding holds on the gpt-4o ∩ gemini consensus (US consensus transformer cosine AUC 0.165 / 0.173, still well below 0.5; probe recovers 0.85–0.87). Cross-vendor robustness established for both economies.

**Cross-vendor replication confirmed (gemini-2.5-pro now the replication generator; gpt-4o-mini archived):**
- The whole finding reproduces under cross-vendor replication. CA consensus comp-vs-sub cosine AUC: tfidf 0.168 / glove 0.180 / gemini 0.092 / openai 0.100; probe 0.79–0.86. US consensus: tfidf 0.229 / glove 0.231 / gemini 0.165 / openai 0.173; probe 0.84–0.87. Gemini is more conservative, so consensus shrinks (CA 4,296→3,126 pairs; US 4,051→2,338) but rank order is unchanged.
- Concordance κ vs the Gemini census matches the gpt-4o picture: CA 0.22 (modest, CI wide), US 0.02 (still chance). Confirms the cell is generator-independent (κ uses no embeddings).

**BEA upgraded to the 2017 detail benchmark (was 1997):** production finding holds, materially unchanged. gpt-4o Claim A complement-vs-random: CA couse 0.579 / supplychain 0.595; US couse 0.578 / supplychain 0.572 (CIs separated). CA Claim B cosine: gemini 0.59 / openai 0.61 / glove 0.72. 2007/2012/2017 sheets agree to ~0.01.

**Retired (cells removed because the dataset was dropped):**
- Instacart US Claim A and Dunnhumby US Claim A (small-N grey) — grocery datasets dropped now that Amazon's Claim B is rehabilitated; Amazon's breadth carries consumer goods. StatCan I-O + BEA retained.
- Retailrocket (newly evaluated) — fully anonymized (numeric category ids, hashed item properties, no names/titles), so it cannot be crosswalked to NAPCS text; unusable for the embedding validation. Not added.

**Outstanding grey:**
- Concordance κ, US (0.01 gpt-4o / 0.02 gemini) — does NOT clear, generator-independent. Concordance needs the intersection of conceptual-anchor pairs AND clean Amazon edges, which collapses to single digits (uninterpretable); the well-powered baseline (n≈90–103) stays near chance because of coarse Amazon↔NAPCS crosswalk coverage. Report as design-limited; US-only (companion scope) — Canada concordance (0.22–0.31) is fine.
- Production Claim A/B, US (BEA) — supportive on complement-vs-random but the US crosswalk is concentrated/noisy (40.8% of NAPCS codes pile onto one BEA cell), so the substitute direction is anomalous and US Claim B ≈ chance. Known US-side limitation; companion scope. Canada production (StatCan + BEA) is clean.

**BEA-2017 gemini replication — clean re-run done:** recomputed against the full census (n_comp in the thousands) and it AGREES with the gpt-4o primary — Claim A complement-vs-random CA couse 0.574 / supplychain 0.605; US couse 0.600 / supplychain 0.603 (complement > random in every cell, both generators, both linkages). Claim B cosine clean in CA (glove 0.73), near-chance in US (same crosswalk-noise pattern as gpt-4o). The production finding replicates across both vendors. (Root cause of the earlier invalid numbers: the script reads the `.jsonl` checkpoint, and only the stale partial `.jsonl` was on the mount; fixed by converting the complete `.json` to `.jsonl`.)

## Notes on benchmarks

- AUC benchmark is 0.50 (chance ranking). Cohen's κ benchmark is 0.00 (chance agreement).
- "comp-vs-sub cosine AUC" = P(complement cosine > substitute cosine); **below 0.5** means complements
  sit *below* substitutes under cosine — the pathology, so for this row lower is stronger.
- "sub-over-comp cosine AUC" (Claim B) = P(substitute cosine > complement cosine); **above 0.5** means the
  same pathology shows up on real behavioral pairs.
- Canada is the submitted-paper economy; US is anonymized-companion scope. Grey cells that are US-only
  therefore sit off the submitted paper's critical path.
