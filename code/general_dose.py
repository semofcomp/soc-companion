#!/usr/bin/env python3
"""Dose-response: mean complement percentile by GENERATION RANK (1..6).
Replicates the percentile method in general_rank_only.py exactly.
Usage: general_dose.py <CA|US>
Writes analytics-general/full-census/<CTRY>_dose.json
"""
import os, sys, json, numpy as np, pandas as pd
from collections import defaultdict

CTRY = sys.argv[1].upper()
ROOT = "/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
CENS = os.path.join(ROOT, "analytics-general", "census")
MAXRANK = 6

if CTRY == "US":
    DATA = os.path.join(ROOT, "data", "US NAPCS"); META = os.path.join(DATA, "us_categories_metadata.csv")
    EMB = os.path.join(DATA, "embeddings.parquet"); GLV = os.path.join(DATA, "us_glove_embeddings.npy")
    CP = os.path.join(CENS, "census_complements_US_gpt-4o.json")
else:
    DATA = os.path.join(ROOT, "data"); META = os.path.join(DATA, "categories_metadata.csv")
    EMB = os.path.join(DATA, "embeddings.parquet"); GLV = os.path.join(DATA, "glove_embeddings.npy")
    CP = os.path.join(CENS, "census_complements_CA_gpt-4o.json")

meta = pd.read_csv(META, dtype=str).fillna("")
codes = meta["code"].tolist(); idx = {c: i for i, c in enumerate(codes)}; N = len(codes)
if CTRY == "US":
    tri = dict(zip(codes, meta["trilateral_code"].tolist()))
    def same_sub(c, f):
        a, b = tri.get(c, ""), tri.get(f, ""); return bool(a) and a == b
else:
    def same_sub(c, f): return c[:6] == f[:6]

src = json.load(open(CP))

trip = []
for f, lst in src.items():
    if f not in idx: continue
    r = 0
    for o in lst:
        c = o.get("code")
        if c in idx and c != f and not same_sub(c, f):
            r += 1
            if r > MAXRANK: break
            trip.append((idx[f], idx[c], r))
trip = np.array(trip)
fi, ci, rk_gen = trip[:, 0], trip[:, 1], trip[:, 2]

df = pd.read_parquet(EMB); df["code"] = df["code"].astype(str)
di = {c: i for i, c in enumerate(df["code"])}; order = np.array([di[c] for c in codes])
def normed(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n < 1e-9] = 1; return (M / n).astype(np.float32)
G = np.vstack([np.asarray(v, np.float32) for v in df["gemini_embedding"]])[order]
O = np.vstack([np.asarray(v, np.float32) for v in df["openai_embedding"]])[order]
GL = np.load(GLV).astype(np.float32)
CEN = {"gemini": normed(G - G.mean(0)), "openai": normed(O - O.mean(0)), "glove": normed(GL - GL.mean(0))}
from sklearn.feature_extraction.text import TfidfVectorizer
TX = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(meta["text"].fillna("").tolist())

MODELS = ["tfidf", "glove", "gemini", "openai"]
S = {}
for k in ["gemini", "openai", "glove"]: S[k] = (CEN[k] @ CEN[k].T).astype(np.float32)
S["tfidf"] = np.asarray((TX @ TX.T).todense(), dtype=np.float32)

PCT = {}
for m in MODELS:
    A = S[m].copy(); np.fill_diagonal(A, -np.inf)
    rkm = np.argsort(np.argsort(A, axis=1), axis=1)
    PCT[m] = ((rkm - 1) / (N - 2)).astype(np.float32)

res = {"country": CTRY, "n_cats": N, "n_pairs": int(len(fi)), "max_rank": MAXRANK,
       "models": MODELS, "per_model": {}}
for m in MODELS:
    pp = PCT[m][fi, ci]
    rowm = {}
    for r in range(1, MAXRANK + 1):
        mask = rk_gen == r
        rowm[str(r)] = {"mean_pct": float(pp[mask].mean()), "n": int(mask.sum())}
    rowm["overall_mean_pct"] = float(pp.mean())
    res["per_model"][m] = rowm
    print("DOSE", CTRY, m, "r1", round(rowm["1"]["mean_pct"], 4),
          "r6", round(rowm["6"]["mean_pct"], 4), "overall", round(rowm["overall_mean_pct"], 4))

out = os.path.join(ROOT, "analytics-general", "full-census", f"{CTRY}_dose.json")
json.dump(res, open(out, "w"), indent=1)
print("saved", out)
