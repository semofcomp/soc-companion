#!/usr/bin/env python3
"""
General-rubric percentile ranks + linear probe, full census.
Usage: python3 general_rank_probe.py <CA|US> <primary|replication|consensus>
Outputs: /tmp/<CA|US>_rank[_variant].json , /tmp/<CA|US>_probe[_variant].json
"""
import os, sys, json, numpy as np, pandas as pd
from collections import defaultdict

CTRY = sys.argv[1].upper()
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "primary"
ROOT = "/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
CENS_DIR = os.path.join(ROOT, "analytics-general", "census")
rng = np.random.default_rng(42)
REPL_MODEL = os.environ.get("REPL_MODEL", "gemini-2.5-pro")
SUFFIX = {"primary": "", "replication": "_replication", "consensus": "_consensus"}[VARIANT]

if CTRY == "US":
    DATA = os.path.join(ROOT, "data", "US NAPCS")
    META_CSV = os.path.join(DATA, "us_categories_metadata.csv")
    EMB_PARQUET = os.path.join(DATA, "embeddings.parquet")
    GLOVE_NPY = os.path.join(DATA, "us_glove_embeddings.npy")
    CEN_P = os.path.join(CENS_DIR, "census_complements_US_gpt-4o.json")
    CEN_R = os.path.join(CENS_DIR, f"census_complements_US_{REPL_MODEL}.json")
else:
    DATA = os.path.join(ROOT, "data")
    META_CSV = os.path.join(DATA, "categories_metadata.csv")
    EMB_PARQUET = os.path.join(DATA, "embeddings.parquet")
    GLOVE_NPY = os.path.join(DATA, "glove_embeddings.npy")
    CEN_P = os.path.join(CENS_DIR, "census_complements_CA_gpt-4o.json")
    CEN_R = os.path.join(CENS_DIR, f"census_complements_CA_{REPL_MODEL}.json")

meta = pd.read_csv(META_CSV, dtype=str).fillna("")
codes = meta["code"].tolist(); idx = {c: i for i, c in enumerate(codes)}; N = len(codes)

if CTRY == "US":
    tri = dict(zip(codes, meta["trilateral_code"].tolist()))
    def same_sub(c, f):
        a, b = tri.get(c, ""), tri.get(f, ""); return bool(a) and a == b
else:
    def same_sub(c, f): return c[:6] == f[:6]

def census_dict(path):
    d = json.load(open(path, encoding="utf-8")); return d

prim = census_dict(CEN_P)
if VARIANT == "primary":
    src = prim
elif VARIANT == "replication":
    src = census_dict(CEN_R)
else:
    repl = census_dict(CEN_R)
    repl_sets = {f: set(o.get("code") if isinstance(o, dict) else o for o in lst) for f, lst in repl.items()}
    src = {}
    for f, lst in prim.items():
        rs = repl_sets.get(f, set())
        kept = [o for o in lst if (o.get("code") if isinstance(o, dict) else o) in rs]
        if kept: src[f] = kept

# complement pairs with rank order
fi, ci, rank = [], [], []
for f, lst in src.items():
    if f not in idx: continue
    for r, o in enumerate(lst):
        c = o.get("code") if isinstance(o, dict) else o
        if c in idx and c != f and not same_sub(c, f):
            fi.append(idx[f]); ci.append(idx[c]); rank.append(r + 1)
fi = np.array(fi); ci = np.array(ci); rank = np.array(rank)

def pref_pairs():
    g = defaultdict(list)
    if CTRY == "US":
        for c in codes:
            k = tri.get(c, "")
            if k: g[k].append(idx[c])
    else:
        for c in codes: g[c[:6]].append(idx[c])
    a, b = [], []
    for v in g.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a.append(v[i]); b.append(v[j])
    return np.array(a), np.array(b)
sub_a, sub_b = pref_pairs()

max_pairs = N * (N - 1) // 2; target = min(10000, max_pairs)
seen = set(); ra, rb = [], []; attempts = 0
while len(ra) < target and attempts < target * 50 + 200000:
    i = int(rng.integers(0, N)); j = int(rng.integers(0, N)); attempts += 1
    if i == j: continue
    key = (i, j) if i < j else (j, i)
    if key in seen: continue
    seen.add(key); ra.append(key[0]); rb.append(key[1])
rnd_a = np.array(ra); rnd_b = np.array(rb)

df = pd.read_parquet(EMB_PARQUET); df["code"] = df["code"].astype(str)
di = {c: i for i, c in enumerate(df["code"])}; order = np.array([di[c] for c in codes])
G = np.vstack([np.asarray(v, np.float32) for v in df["gemini_embedding"]])[order]
O = np.vstack([np.asarray(v, np.float32) for v in df["openai_embedding"]])[order]
GL = np.load(GLOVE_NPY).astype(np.float32)
def normed(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n < 1e-9] = 1
    return (M / n).astype(np.float32)
CEN = {"gemini": normed(G - G.mean(0)), "openai": normed(O - O.mean(0)), "glove": normed(GL - GL.mean(0))}
from sklearn.feature_extraction.text import TfidfVectorizer
TX = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(meta["text"].fillna("").tolist())

MODELS = ["tfidf", "glove", "gemini", "openai"]
S = {}
for k in ["gemini", "openai", "glove"]:
    S[k] = (CEN[k] @ CEN[k].T).astype(np.float32)
S["tfidf"] = np.asarray((TX @ TX.T).todense(), dtype=np.float32)

PCT = {}
for m in MODELS:
    A = S[m].copy(); np.fill_diagonal(A, -np.inf)
    rk = np.argsort(np.argsort(A, axis=1), axis=1)
    PCT[m] = ((rk - 1) / (N - 2)).astype(np.float32)

pos = defaultdict(set)
for a, b in zip(fi, ci): pos[a].add(b)
sibs = defaultdict(set)
for a, b in zip(sub_a, sub_b): sibs[a].add(b); sibs[b].add(a)

res = {"country": CTRY, "variant": VARIANT, "n_cats": N,
       "n_complement_pairs": int(len(fi)), "models": MODELS, "per_model": {}}
for m in MODELS:
    Pm = PCT[m]; pp = Pm[fi, ci]
    aucs, p10, r50, cs_auc, cs_win = [], [], [], [], []
    for f, ps in pos.items():
        psl = np.array(sorted(ps)); pct_pos = Pm[f, psl]
        npos = len(psl); nneg = N - 1 - npos
        ranks = pct_pos * (N - 2) + 1
        auc = (ranks.sum() - npos * (npos + 1) / 2) / (npos * nneg); aucs.append(auc)
        row = Pm[f]; kk = min(50, N - 2)
        topk = np.argpartition(-row, kk)[:kk + 1]; topk = topk[topk != f]; topk = topk[np.argsort(-row[topk])]
        p10.append(len(set(topk[:10]) & ps) / 10)
        r50.append(len(set(topk[:50]) & ps) / npos)
        sb = sibs.get(f)
        if sb:
            sbl = np.array(sorted(sb)); pcs = Pm[f, psl]; scs = Pm[f, sbl]
            u = (pcs[:, None] > scs[None, :]).sum() + 0.5 * (pcs[:, None] == scs[None, :]).sum()
            cs_auc.append(u / (len(pcs) * len(scs))); cs_win.append(float(pcs.mean() > scs.mean()))
    res["per_model"][m] = {
        "mean_pct": float(pp.mean()), "median_pct": float(np.median(pp)),
        "frac_pct_ge_90": float((pp >= 0.90).mean()), "frac_pct_ge_99": float((pp >= 0.99).mean()),
        "macro_auc_comp_vs_all": float(np.mean(aucs)),
        "precision_at_10": float(np.mean(p10)), "recall_at_50": float(np.mean(r50)),
        "comp_vs_sub_auc": float(np.mean(cs_auc)) if cs_auc else None,
        "n_focals_with_subs": len(cs_auc)}
    print("RANK", m, {k: round(v, 4) for k, v in res["per_model"][m].items() if isinstance(v, float)})
json.dump(res, open(f"/tmp/{CTRY}_rank{SUFFIX}.json", "w"), indent=1)

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def probe_binary(X0, g0, X1, g1):
    X = np.vstack([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]; g = np.r_[g0, g1]
    nsplit = min(5, len(np.unique(g))); aucs = []
    for tr, te in GroupKFold(nsplit).split(X, y, g):
        if len(np.unique(y[te])) < 2: continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs)), float(np.std(aucs))

probe_out = {"country": CTRY, "variant": VARIANT, "n_cats": N, "per_model": {}}
for model in ["glove", "gemini", "openai", "tfidf"]:
    if model == "tfidf":
        Mfull = np.asarray(TX.todense(), dtype=np.float32); pca_used = False
        if Mfull.shape[1] > 300:
            Mfull = PCA(n_components=300, random_state=42).fit_transform(Mfull).astype(np.float32); pca_used = True
    else:
        Mfull = CEN[model]; pca_used = False
        if Mfull.shape[1] > 300:
            Mfull = PCA(n_components=300, random_state=42).fit_transform(Mfull).astype(np.float32); pca_used = True
    def feats(a, b):
        A, B = Mfull[a], Mfull[b]
        return np.hstack([A * B, np.abs(A - B)]).astype(np.float32)
    Xc = feats(fi, ci); gc = np.array(["F" + codes[i] for i in fi])
    Xs = feats(sub_a, sub_b)
    if CTRY == "US":
        gs = np.array(["S" + tri.get(codes[i], "") for i in sub_a])
    else:
        gs = np.array(["S" + codes[i][:6] for i in sub_a])
    Xr = feats(rnd_a, rnd_b); gr = np.array([f"R{i}" for i in rnd_a])
    d = {"pca300": pca_used, "n_comp": int(len(Xc)), "n_sub": int(len(Xs)), "n_rand": int(len(Xr))}
    if len(Xs) >= 10:
        m_, s_ = probe_binary(Xs, gs, Xc, gc); d["auc_comp_vs_sub"] = m_; d["auc_comp_vs_sub_sd"] = s_
    else:
        d["auc_comp_vs_sub"] = None
    m_, s_ = probe_binary(Xr, gr, Xc, gc); d["auc_comp_vs_rand"] = m_; d["auc_comp_vs_rand_sd"] = s_
    Smat = S[model]; cos_c = Smat[fi, ci]; cos_s = Smat[sub_a, sub_b]
    if len(cos_s) > 0:
        y = np.r_[np.zeros(len(cos_s)), np.ones(len(cos_c))]; x = np.r_[cos_s, cos_c]
        d["cosine_only_auc_comp_vs_sub"] = float(roc_auc_score(y, x))
    y2 = np.r_[np.zeros(len(rnd_a)), np.ones(len(cos_c))]
    d["cosine_only_auc_comp_vs_rand"] = float(roc_auc_score(y2, np.r_[Smat[rnd_a, rnd_b], cos_c]))
    probe_out["per_model"][model] = d
    print("PROBE", model, "comp_vs_sub", d.get("auc_comp_vs_sub"), "comp_vs_rand", round(d["auc_comp_vs_rand"], 3))
json.dump(probe_out, open(f"/tmp/{CTRY}_probe{SUFFIX}.json", "w"), indent=1)
print("saved rank+probe", CTRY, VARIANT)
