#!/usr/bin/env python3
"""
Rebuild the RECIP data object embedded in companion_site_v2/reciprocity.html from the
CANONICAL gpt-4o complement graph (data/graph_CA.json), with cross-generator consensus
measured against the Gemini-2.5-Pro census.  Replaces the older one-off RECIP (stale graph
counts 16,536/486/2.9%).  Definitions (documented so the page is reproducible):

  base graph        = canonical gpt-4o directed complement links (graph_CA.json edges)
  mutual pair (a,b) = both a->b and b->a present in the base graph
  g, v              = anisotropy-corrected (mean-centred + L2) Gemini / GloVe cosine(a,b)
  cab / cba         = direction a->b / b->a also attested in the Gemini census (consensus)
  *_cons stats      = share of directed links (mutual vs one-way) also in the Gemini census
Outputs RECIP as compact JSON to stdout (and to reciprocity_RECIP.json) for injection.
"""
import os, json, re
import numpy as np, pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)                      # companion_site_v2
PROJ = os.path.dirname(SITE)
DATA = os.path.join(PROJ, "data")
CEN = os.path.join(PROJ, "analytics-general", "census")

graph = json.load(open(os.path.join(SITE, "data", "graph_CA.json")))
edges = set((a, b) for a, b in graph["edges"])           # directed
gp = json.load(open(os.path.join(CEN, "census_complements_CA_gpt-4o.json")))
gm = json.load(open(os.path.join(CEN, "census_complements_CA_gemini-2.5-pro.json")))
gem_links = set((f, x["code"]) for f, lst in gm.items() for x in lst)   # gemini directed links
# type + per-link dimensions from the gpt-4o census
typ = {}; dims = {}
for f, lst in gp.items():
    for x in lst:
        dims[(f, x["code"])] = x.get("dimensions", [])
        typ[x["code"]] = x.get("type", "")
        typ.setdefault(f, "")

meta = pd.read_csv(os.path.join(DATA, "categories_metadata.csv"), dtype={"code": str})
codes = meta.code.tolist(); titles = dict(zip(meta.code, meta.title)); idx = {c: i for i, c in enumerate(codes)}
tbl = pq.read_table(os.path.join(DATA, "embeddings.parquet"), columns=["gemini_embedding"])
la = tbl.column("gemini_embedding").combine_chunks()
gemv = la.values.to_numpy(zero_copy_only=False).reshape(len(la), len(la[0])).astype("float64")
gemv = gemv - gemv.mean(0, keepdims=True); gemv /= (np.linalg.norm(gemv, axis=1, keepdims=True) + 1e-12)
glv = np.load(os.path.join(DATA, "glove_embeddings.npy")).astype("float64")
glv = glv - glv.mean(0, keepdims=True); glv /= (np.linalg.norm(glv, axis=1, keepdims=True) + 1e-12)
def gcos(a, b): return float(gemv[idx[a]] @ gemv[idx[b]])
def vcos(a, b): return float(glv[idx[a]] @ glv[idx[b]])

mutual = set((a, b) for (a, b) in edges if (b, a) in edges)
pairs_fs = sorted({tuple(sorted((a, b))) for (a, b) in mutual})
one_dir = [(a, b) for (a, b) in edges if (b, a) not in edges]

n_directed = len(edges); n_mutual = len(mutual); n_pairs = len(pairs_fs)
frac = round(n_mutual / n_directed, 4)
def mean_g(es): return round(float(np.mean([gcos(a, b) for a, b in es])), 3) if es else 0.0
def mean_v(es): return round(float(np.mean([vcos(a, b) for a, b in es])), 3) if es else 0.0
mut_gem, one_gem = mean_g(mutual), mean_g(one_dir)
mut_glo, one_glo = mean_v(mutual), mean_v(one_dir)
mut_cons = round(float(np.mean([1.0 if (a, b) in gem_links else 0.0 for a, b in mutual])), 3)
one_cons = round(float(np.mean([1.0 if (a, b) in gem_links else 0.0 for a, b in one_dir])), 3)
pairs_any = sum(1 for a, b in pairs_fs if (a, b) in gem_links or (b, a) in gem_links)
pairs_both = sum(1 for a, b in pairs_fs if (a, b) in gem_links and (b, a) in gem_links)

pairs = []
for a, b in pairs_fs:
    pairs.append({"a": a, "b": b, "at": titles.get(a, a), "bt": titles.get(b, b),
                  "g": round(gcos(a, b), 3), "v": round(vcos(a, b), 3),
                  "cab": 1 if (a, b) in gem_links else 0, "cba": 1 if (b, a) in gem_links else 0,
                  "tab": [typ.get(a, ""), dims.get((b, a), [])],
                  "tba": [typ.get(b, ""), dims.get((a, b), [])]})
pairs.sort(key=lambda p: (p["at"], p["bt"]))

RECIP = {"stats": {"n_directed": n_directed, "n_mutual_directed": n_mutual, "n_pairs": n_pairs,
                   "frac": frac, "mut_gem": mut_gem, "one_gem": one_gem, "mut_glo": mut_glo,
                   "one_glo": one_glo, "mut_cons": mut_cons, "one_cons": one_cons,
                   "pairs_cons_any": pairs_any, "pairs_cons_both": pairs_both}, "pairs": pairs}
json.dump(RECIP, open(os.path.join(HERE, "reciprocity_RECIP.json"), "w"), ensure_ascii=False, separators=(",", ":"))

S = RECIP["stats"]
print("STATS:", json.dumps(S))
print(f"CARDS: frac={S['frac']*100:.1f}%  n_directed={S['n_directed']:,}  mutual={S['n_mutual_directed']}  pairs={S['n_pairs']}")
print(f"  gem {S['mut_gem']} vs {S['one_gem']} | glove {S['mut_glo']} vs {S['one_glo']}")
print(f"  cons {S['mut_cons']*100:.1f}% vs {S['one_cons']*100:.1f}% | pairs_any {S['pairs_cons_any']}/{S['n_pairs']} both {S['pairs_cons_both']}")
# directedness illustration check: nature-park visits (8325115) -> photography one-way?
np_focal = "8325115"
for c in [x["code"] for x in gp.get(np_focal, [])]:
    asym = (np_focal, c) in edges and (c, np_focal) not in edges
    print(f"  dir-check {np_focal}->{c} ({titles.get(c,'')[:30]}): one-way={asym}")
