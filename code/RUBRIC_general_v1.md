# General-complementarity rubric — redline (v1, draft for review)

> **Status note (2026-07):** this memo is the historical rubric-design record. The generator pairing it proposes (gpt-4o-mini) is LEGACY; the paper's census uses **GPT-4o (primary)** and **Gemini-2.5-Pro (replication)** — see `regenerate_census_general.py` and `regenerate_census_gemini.py`.


*Goal: replace the consumer-perspective complement-generation prompt with one that treats complementarity as a **unified phenomenon across consumption and production**, and that fixes the diagnosed failure mode — raw-material / intermediate / B2B focals pulling in unrelated consumer goods (e.g. "Green coffee beans" → "Vines, Golf equipment") because the consumer frame gives them no anchor.*

The three places where "consumer" is baked in are the RUBRIC string and the two pass prompts. The substitute exclusion, the product/service type tag, the two-pass structure, and the rapidfuzz resolution are **unchanged**.

---

## 1. RUBRIC (the definition)

**CURRENT (consumer-only):**
> COMPLEMENTARY categories are realistically used together **by a CONSUMER** as part of the same activity, occasion, or demand relationship, while serving DISTINCT roles (one increases the value or need for the other). They are NOT substitutes (same need, interchangeable) and NOT near-duplicates. **Judge from the consumer's perspective (what someone buying the focal would also seek), not the provider's.**

**PROPOSED (general):**
> COMPLEMENTARY categories are realistically **USED TOGETHER** as part of the same activity, process, **experience**, or demand relationship, while serving DISTINCT, mutually reinforcing roles (having or using one increases the value or need for the other). **Complementarity is GENERAL — it arises in CONSUMPTION (goods/services a household uses together in a consumer activity) AND in PRODUCTION/BUSINESS (raw materials, components, equipment, or services used together in a production process, project, or supply-chain stage). FIRST identify the primary activity or process in which the FOCAL is used; THEN find categories used ALONGSIDE it in that SAME context. If the focal is a raw material, intermediate good, component, capital equipment, wholesale offering, or business service, stay within its production/industrial context — do NOT pull in unrelated consumer goods. When in doubt whether a candidate genuinely shares the focal's activity or production context, exclude it rather than guess.** Complements are NOT substitutes (same need, interchangeable) and NOT near-duplicates of the focal. **Judge by genuine joint use, from whichever side — consumer or producer — the focal actually belongs to.**

*Why:* the bolded additions are the fix. "First identify the activity/process… stay within its context… do NOT pull unrelated consumer goods" directly targets the failure mode; the consumption/production framing makes the construct general.

---

## 2. Pass 1 (free-generation of complement phrases)

**CURRENT:**
> List up to 8 DISTINCT complementary product or service categories **a consumer in the market for the focal category** would also realistically need or seek. Short category-name noun phrases. Exclude substitutes and the focal itself. Return ONLY a JSON array of strings.

**PROPOSED:**
> List up to 8 DISTINCT complementary product or service categories **that are genuinely used together with the focal in its primary activity or production process — what a user of the focal (a household consumer, OR a producer/business, whichever the focal belongs to) would also need or use alongside it.** Short category-name noun phrases. Exclude substitutes and the focal itself. Return ONLY a JSON array of strings.

---

## 3. Pass 2 (select & rank from real candidates; tag type + dimensions)

**CURRENT:**
> From the REAL NAPCS categories below, choose the {N} BEST genuine complements, ranked best first. Exclude substitutes/near-duplicates. For each, give its code, its type ('product' or 'service'), and its dimensions (any of: functional, sequential, occasion, demand_spillover). Return ONLY JSON: …

**PROPOSED:**
> From the REAL NAPCS categories below, choose the {N} BEST genuine complements, ranked best first. **Each must belong to the focal's OWN activity or production context and be genuinely used together with it.** Exclude substitutes and near-duplicates. For each, give its code, its type ('product' or 'service'), and its dimensions (any of: **functional** [used together to perform a task or process step], **sequential** [one precedes or follows the other in a usage or production chain], **occasion** [shared activity, occasion, or process context], **demand_spillover** [demand for one raises demand for the other]). Return ONLY JSON: …

*Why:* the four dimension tags are kept (so the output schema is unchanged) but their definitions are broadened to be context-neutral — "occasion" now also covers a production-process context. Tag reliability was already weak (κ 0.16–0.47), so these stay descriptive.

---

## Unchanged
- Substitute exclusion: drop self + same-subclass (Canada 6-digit) / same-trilateral (US) candidates.
- Type tag: product / service.
- Two-pass structure; rapidfuzz high-recall resolution (partial_ratio ≥82, token_set_ratio ≥75).
- Temperature 0.4; both generators (gpt-4o-mini primary, gpt-4o replication).

## Generator decision & disagreement policy (locked)
- **Primary census = gpt-4o** (the hard-case validation showed materially cleaner production labels). **Replication = gpt-4o-mini** (its residual noise is now a *feature*: if the finding survives a noisier generator, it isn't a labeling artifact).
- **Disagreements are NOT adjudicated pair-by-pair.** Report the headline three ways — primary (gpt-4o), replication (gpt-4o-mini), and the **consensus subset** (pairs both produced) — and the claim is whatever holds across all three. Never hand-pick one generator's pair over the other's (that would convert generated labels into curated ones). Inter-generator κ stays a descriptive reliability statistic.

## Validation before any full regeneration
Run this rubric on the 18 hard-case focals in `hard_cases.csv` (12 failure-mode + 6 consumer controls, both economies) and eyeball: does "Green coffee beans" now return roasting/packaging/storage/brewing-equipment instead of vines/golf? Do "Textile components" return weaving/finishing/garment-manufacturing? Do the consumer controls still look sensible? Only after this passes do we regenerate the full census.
