# General-Rubric Results Note

Embedding analytic suite re-run on the NEW general-complementarity census (consumption AND production), both economies, fully local. gpt-4o = PRIMARY, gemini-2.5-pro = REPLICATION (cross-vendor; supersedes gpt-4o-mini, whose numbers are archived under full-census/_archive_gpt4omini/), CONSENSUS = focal->complement pairs present in BOTH generators (gpt-4o AND gemini-2.5-pro). Disagreements are NOT pairwise-adjudicated. Bootstrap B=2000 (seed 42); GroupKFold-by-focal probe; PCA-300 for transformer probes. Substitute anchor = same 6-digit code prefix (CA) / same trilateral_code (US); random = 10k pairs (seed 42).


## CA

> Replication is now CROSS-VENDOR gemini-2.5-pro (Google); the prior same-vendor gpt-4o-mini replication/consensus numbers are archived under full-census/_archive_gpt4omini/.

**Realized Ns** (full universe = 3049 categories):

- Primary (gpt-4o): 15925 complement pairs (taxonomic-filtered), 2896 focals with >=1 complement; substitute pairs 3689; random 10000.
- Replication (gemini-2.5-pro): 13192 complement pairs (taxonomic-filtered), 2730 focals with >=1 complement.
- Consensus (both generators): 3126 complement pairs, 1591 focals retained.

### CA position spectrum (complement position = (comp-rand)/(sub-rand); 0%=random, 100%=substitute)

| Variant | Model | Position [95% CI] | \|d\| comp-sub | comp-vs-sub cosine AUC | probe AUC (comp-vs-sub) |
|---|---|---|---|---|---|
| Primary | tfidf | 15.6% [14.7,16.5] | 1.62 | 0.147 | 0.832 |
| Primary | glove | 46.5% [45.2,47.8] | 1.14 | 0.137 | 0.826 |
| Primary | gemini | 30.6% [29.9,31.4] | 2.18 | 0.055 | 0.910 |
| Primary | openai | 30.5% [29.8,31.3] | 2.09 | 0.061 | 0.908 |
| Replication | tfidf | 18.4% [17.3,19.4] | 1.46 | 0.158 | 0.793 |
| Replication | glove | 53.7% [52.4,55.1] | 0.99 | 0.156 | 0.791 |
| Replication | gemini | 37.2% [36.3,38.1] | 1.88 | 0.071 | 0.883 |
| Replication | openai | 37.4% [36.6,38.3] | 1.81 | 0.081 | 0.874 |
| Consensus | tfidf | 24.6% [22.7,26.8] | 1.03 | 0.168 | 0.787 |
| Consensus | glove | 66.7% [64.8,68.9] | 0.71 | 0.180 | 0.772 |
| Consensus | gemini | 46.0% [44.6,47.3] | 1.60 | 0.092 | 0.857 |
| Consensus | openai | 45.7% [44.3,47.1] | 1.50 | 0.100 | 0.855 |


## US

> Replication is now CROSS-VENDOR gemini-2.5-pro (Google); the prior same-vendor gpt-4o-mini replication/consensus numbers are archived under full-census/_archive_gpt4omini/.

**Realized Ns** (full universe = 3520 categories):

- Primary (gpt-4o): 18312 complement pairs (taxonomic-filtered), 3321 focals with >=1 complement; substitute pairs 23983; random 10000.
- Replication (gemini-2.5-pro): 13241 complement pairs (taxonomic-filtered), 2689 focals with >=1 complement.
- Consensus (both generators): 2338 complement pairs, 1182 focals retained.

### US position spectrum (complement position = (comp-rand)/(sub-rand); 0%=random, 100%=substitute)

| Variant | Model | Position [95% CI] | \|d\| comp-sub | comp-vs-sub cosine AUC | probe AUC (comp-vs-sub) |
|---|---|---|---|---|---|
| Primary | tfidf | 28.3% [26.8,30.0] | 0.68 | 0.192 | 0.850 |
| Primary | glove | 44.5% [42.8,46.1] | 0.97 | 0.169 | 0.767 |
| Primary | gemini | 41.3% [40.0,42.8] | 1.17 | 0.120 | 0.853 |
| Primary | openai | 40.9% [39.7,42.2] | 1.16 | 0.127 | 0.859 |
| Replication | tfidf | 36.2% [34.1,38.4] | 0.57 | 0.217 | 0.846 |
| Replication | glove | 49.6% [47.6,51.3] | 0.87 | 0.171 | 0.768 |
| Replication | gemini | 49.0% [47.2,50.7] | 0.98 | 0.119 | 0.851 |
| Replication | openai | 47.1% [45.5,48.6] | 1.00 | 0.126 | 0.856 |
| Consensus | tfidf | 57.3% [52.6,62.2] | 0.35 | 0.229 | 0.840 |
| Consensus | glove | 79.2% [76.1,82.2] | 0.37 | 0.231 | 0.783 |
| Consensus | gemini | 76.5% [73.6,79.1] | 0.47 | 0.165 | 0.871 |
| Consensus | openai | 74.3% [71.7,77.0] | 0.48 | 0.173 | 0.852 |


## Consumer-vs-Production split — the economy × domain 2×2 (cross-survival)

**Labels (how a category is consumption vs production):** LLM consumer-final classifier (criterion: acquired by households for final use vs intermediate/capital/wholesale/business) run under TWO vendors, gpt-4o AND gemini-2.5-pro. The cut is the **cross-vendor consensus** — consumer = both agree consumer-final; production = both agree non-consumer (a positively-labeled class); disagreements (REVIEW) dropped. Cross-vendor agreement CA 93.1% / US 86.8%. A complement-neighborhood audit was considered and deliberately NOT applied (it would entangle the split with the complement graph it partitions, and complementarity legitimately crosses the consumption/production line). Labels: `analytics-common/consumer-final-labels/consumer_final_xvendor_{CA,US}.csv`.

**Cell construction (conservative):** each of the 4 cells (CA/US × consumption/production) is restricted to that census's categories with that label; a focal→complement pair enters a cell only if BOTH endpoints carry the label; substitute (taxonomic) and random anchors are re-derived WITHIN each subset; cross-domain pairs (e.g. consumption focal × production complement) are excluded from all cells — so "holds within X" is uncontaminated.

### Cross-vendor consensus cut (PRIMARY going forward) — files: `robustness-consumer-final/{CA,US}_consumer_vs_production_xvendor.json`

| Cell | n cats | comp pairs | comp-vs-sub cosine AUC (gem / ope) | gemini probe |
|---|---|---|---|---|
| Canada consumption | 908 | 2,309 | 0.069 / 0.071 | 0.84 |
| Canada production | 1,931 | 8,859 | 0.051 / 0.059 | 0.90 |
| US consumption | 449 | 884 | 0.114 / 0.110 | 0.82 |
| US production | 2,608 | 12,004 | 0.116 / 0.123 | 0.85 |

Finding holds in all four cells (cosine AUC ≪ 0.5 = pathology; probe ≫ 0.5 = recovery). The single-model gpt-4o-label cut below gives the same pattern → robust to the labeling method.

### Single-model gpt-4o-label cut (robustness) — files: `robustness-consumer-final/{CA,US}_consumer_vs_production.json`

Universe restricted to (a) pairs with BOTH endpoints consumer-final, (b) BOTH endpoints production. Substitute/random anchors re-derived WITHIN each subset.


### CA (consumer cats=970, production cats=2079)

Realized Ns -- consumer: N=970, comp=2553, sub=972, rand=10000, focals=803; production: N=2079, comp=9826, sub=2398, rand=10000, focals=1965.

| Model | Consumer pos | Cons cosAUC | Cons probe | Production pos | Prod cosAUC | Prod probe |
|---|---|---|---|---|---|---|
| tfidf | 18.7% | 0.185 | 0.708 | 15.3% | 0.141 | 0.815 |
| glove | 55.7% | 0.162 | 0.759 | 43.9% | 0.131 | 0.822 |
| gemini | 36.1% | 0.068 | 0.862 | 28.9% | 0.052 | 0.902 |
| openai | 35.6% | 0.069 | 0.854 | 29.4% | 0.059 | 0.903 |


### US (consumer cats=748, production cats=2772)

Realized Ns -- consumer: N=748, comp=1428, sub=1462, rand=10000, focals=541; production: N=2772, comp=13261, sub=20879, rand=10000, focals=2610.

| Model | Consumer pos | Cons cosAUC | Cons probe | Production pos | Prod cosAUC | Prod probe |
|---|---|---|---|---|---|---|
| tfidf | 41.7% | 0.196 | 0.700 | 27.3% | 0.193 | 0.821 |
| glove | 49.3% | 0.162 | 0.619 | 42.9% | 0.176 | 0.750 |
| gemini | 51.3% | 0.121 | 0.826 | 39.1% | 0.121 | 0.824 |
| openai | 49.5% | 0.127 | 0.829 | 38.6% | 0.128 | 0.801 |


## Comparison to prior CONSUMER-rubric numbers

Prior consumer-rubric legacy (for reference):

- Canada full-census: position tfidf 17 / glove 48 / gemini 32 / openai 32; comp-vs-sub cosine AUC ~0.15 / 0.14 / 0.06 / 0.06; probe ~0.90.
- US legacy: position tfidf 33 / gemini 44.7 / openai 44.7 / glove 51.7; AUC 0.209 / 0.193 / 0.129 / 0.137; probe 0.75-0.84.

General-rubric PRIMARY results reproduce the SAME qualitative pattern in both economies:

- CA primary: tfidf 15.6 / glove 46.5 / gemini 30.6 / openai 30.5; cosAUC 0.147 / 0.137 / 0.055 / 0.061; probe gemini/openai ~0.91, glove/tfidf ~0.83. Essentially identical to the prior consumer-rubric Canada numbers.
- US primary: tfidf 28.3 / glove 44.5 / gemini 41.3 / openai 40.9; cosAUC 0.192 / 0.169 / 0.120 / 0.127; probe 0.77-0.86. Same shape, slightly lower transformer position than legacy US.

Conclusion: under the general (consumption + production) rubric the core finding survives -- complements sit BELOW taxonomic substitutes under cosine similarity (position well under the substitute = 100% mark), contextual transformers are WORST at separating complements from substitutes (lowest cos AUC), yet a supervised linear probe RECOVERS complementarity (AUC 0.7-0.9). The cross-vendor replication (gemini-2.5-pro) and consensus subsets shift positions upward modestly but preserve every ordering; Gemini is more conservative than gpt-4o-mini was (fewer complement objects), so the consensus pair counts drop (CA 4296->3126, US 4051->2338). The consumer-vs-production split confirms the generality claim: the pattern holds SEPARATELY in both the consumer-only and production-only subsets, in both economies.
