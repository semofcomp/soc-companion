#!/usr/bin/env python3
"""
General-rubric embedding analytic suite: position spectrum + Cohen's d + by_type +
comp-vs-sub cosine AUC, for ONE economy and ONE census variant.

Usage: python3 general_analyze.py <CA|US> <primary|replication|consensus>
Census variants:
  primary     = gpt-4o
  replication = REPL_MODEL env (default gemini-2.5-pro; was gpt-4o-mini)
  consensus   = focal->complement pairs present in BOTH generators
Output: /tmp/<CA|US>_general_stats[_replication|_consensus].json
"""
import os, sys, json, numpy as np, pandas as pd
from collections import defaultdict
from scipy import stats

CTRY = sys.argv[1].upper()
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "primary"
ROOT = "/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
CENS_DIR = os.path.join(ROOT, "analytics-general", "census")
rng = np.random.default_rng(42)
REPL_MODEL = os.environ.get("REPL_MODEL", "gemini-2.5-pro")

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

SUFFIX = {"primary": "", "replication": "_replication", "consensus": "_consensus"}[VARIANT]

meta = pd.read_csv(META_CSV, dtype=str).fillna("")
codes = meta["code"].tolist()
idx = {c: i for i, c in enumerate(codes)}
N = len(codes)

if CTRY == "US":
    tri = dict(zip(meta["code"], meta["trilateral_code"]))
    def same_sub(c, f):
        a, b = tri.get(c, ""), tri.get(f, "")
        return bool(a) and a == b
else:
    def same_sub(c, f):
        return c[:6] == f[:6]

def census_pairset(path):
    """Return dict focal-> set of complement codes (raw, before filters)."""
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    typ = {}
    for f, lst in d.items():
        s = set()
        for o in lst:
            c = o.get("code") if isinstance(o, dict) else o
            s.add(c)
            typ[(f, c)] = o.get("type", "") if isinstance(o, dict) else ""
        out[f] = s
    return out, typ

prim, prim_typ = census_pairset(CEN_P)
if VARIANT == "primary":
    cens, typ = prim, prim_typ
elif VARIANT == "replication":
    cens, typ = census_pairset(CEN_R)
else:  # consensus: intersection of (focal,complement) pairs
    repl, _ = census_pairset(CEN_R)
    cens = {}
    typ = prim_typ
    for f, s in prim.items():
        if f in repl:
            inter = s & repl[f]
            if inter:
                cens[f] = inter

# Build complement pairs (apply taxonomic filter: not same substitute anchor)
cpairs, cfocal, ctype = [], [], []
for f, s in cens.items():
    if f not in idx: continue
    for c in s:
        if c in idx and c != f and not same_sub(c, f):
            cpairs.append((idx[f], idx[c])); cfocal.append(f)
            ctype.append(typ.get((f, c), ""))
cpairs = np.array(cpairs); cfocal = np.array(cfocal)
n_raw_focal_comp = len([1 for f in cens for c in cens[f] if c in idx and c != f])
print(f"[{CTRY}/{VARIANT}] cats:{N} focals_in_census:{len(cens)} comp_pairs(taxon-filtered):{len(cpairs)} focals_with_comp:{len(set(cfocal))}")

# substitute pairs (within full universe)
def pref_pairs():
    g = defaultdict(list)
    if CTRY == "US":
        for c in codes:
            k = tri.get(c, "")
            if k: g[k].append(idx[c])
    else:
        for c in codes: g[c[:6]].append(idx[c])
    out = []
    for v in g.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                out.append((v[i], v[j]))
    return out
sub_pairs = pref_pairs()

# random pairs 10k seed42
max_pairs = N * (N - 1) // 2
target = min(10000, max_pairs)
seen = set(); ra, rb = [], []; attempts = 0
while len(ra) < target and attempts < target * 50 + 200000:
    i = int(rng.integers(0, N)); j = int(rng.integers(0, N)); attempts += 1
    if i == j: continue
    key = (i, j) if i < j else (j, i)
    if key in seen: continue
    seen.add(key); ra.append(key[0]); rb.append(key[1])
randp = list(zip(ra, rb))
print(f"[{CTRY}/{VARIANT}] sub_pairs:{len(sub_pairs)} rand_pairs:{len(randp)}")

PAIRSETS = {"random": randp, "complements": [tuple(x) for x in cpairs], "substitute": sub_pairs}

df = pd.read_parquet(EMB_PARQUET); df["code"] = df["code"].astype(str)
di = {c: i for i, c in enumerate(df["code"])}
order = np.array([di[c] for c in codes])
G = np.vstack([np.asarray(v, np.float32) for v in df["gemini_embedding"]])[order]
O = np.vstack([np.asarray(v, np.float32) for v in df["openai_embedding"]])[order]
EMB = {"gemini": G, "openai": O}
glove_full = np.load(GLOVE_NPY).astype(np.float32)
EMB["glove"] = glove_full  # rows already aligned to meta order (same N)

from sklearn.feature_extraction.text import TfidfVectorizer
TX = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(meta["text"].fillna("").tolist())

def normed(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n < 1e-9] = 1
    return (M / n).astype(np.float32)
CEN = {k: normed(M - M.mean(0)) for k, M in EMB.items()}

def cos_dense(Mn, pp):
    if not pp: return np.array([])
    a = np.array([x[0] for x in pp]); b = np.array([x[1] for x in pp]); out = np.empty(len(a))
    for k in range(0, len(a), 20000):
        s = slice(k, k + 20000)
        out[s] = np.sum(Mn[a[s]].astype(np.float64) * Mn[b[s]].astype(np.float64), 1)
    return out
def cos_tfidf(pp):
    if not pp: return np.array([])
    a = np.array([x[0] for x in pp]); b = np.array([x[1] for x in pp]); out = np.empty(len(a))
    for k in range(0, len(a), 5000):
        s = slice(k, k + 5000)
        out[k:k + 5000] = np.asarray(TX[a[s]].multiply(TX[b[s]]).sum(1)).ravel()
    return out

COS = {}
for k in EMB: COS[k] = {ps: cos_dense(CEN[k], PAIRSETS[ps]) for ps in PAIRSETS}
COS["tfidf"] = {ps: cos_tfidf(PAIRSETS[ps]) for ps in PAIRSETS}
MODS = ["tfidf", "glove", "gemini", "openai"]

def cohend(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp)

def cluster_ci(m, subset=None):
    comp = COS[m]["complements"]; r = COS[m]["random"].mean(); s = COS[m]["substitute"].mean()
    sub = np.arange(len(comp)) if subset is None else np.array(subset)
    base = (comp[sub].mean() - r) / (s - r)
    f2sub = defaultdict(list)
    for i in sub: f2sub[cfocal[i]].append(i)
    fl = list(f2sub); K = len(fl); B = 2000
    csum = np.array([comp[f2sub[fl[j]]].sum() for j in range(K)], dtype=np.float64)
    ccnt = np.array([len(f2sub[fl[j]]) for j in range(K)], dtype=np.float64)
    R = rng.integers(0, K, (B, K))
    num = csum[R].sum(1); den = ccnt[R].sum(1)
    bo = ((num / den) - r) / (s - r)
    return float(base), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))

stats_out = {"country": CTRY, "variant": VARIANT, "n_cats": N,
             "n_complement_pairs": int(len(cpairs)),
             "n_focal_complement_raw_in_universe": int(n_raw_focal_comp),
             "n_focals_in_census": len(cens),
             "n_focals_with_complement": len(set(cfocal)),
             "Ns": {k: len(v) for k, v in PAIRSETS.items()}, "models": MODS}
spec = {}
for m in MODS:
    row = {ps: float(COS[m][ps].mean()) for ps in PAIRSETS}
    r, s = row["random"], row["substitute"]; denom = (s - r)
    row["position"] = (row["complements"] - r) / denom if denom != 0 else None
    b, lo, hi = cluster_ci(m); row["position_lo"], row["position_hi"] = lo, hi
    row["cohend_comp_sub"] = cohend(COS[m]["complements"], COS[m]["substitute"])
    row["cohend_comp_rand"] = cohend(COS[m]["complements"], COS[m]["random"])
    spec[m] = row
stats_out["spectrum"] = spec

conv = {}; dd = [m for m in MODS if m != "tfidf"]
for i in range(len(dd)):
    for j in range(i + 1, len(dd)):
        conv[f"{dd[i]}_{dd[j]}"] = float(stats.pearsonr(COS[dd[i]]["complements"], COS[dd[j]]["complements"])[0])
stats_out["cross_model_pearson"] = conv

sub = {}
for grp in ["product", "service"]:
    sel = [i for i, t in enumerate(ctype) if str(t).lower() == grp]
    if sel:
        d = {"n": len(sel)}
        for m in MODS:
            b, lo, hi = cluster_ci(m, sel)
            d[m] = {"position": b, "lo": lo, "hi": hi, "mean": float(COS[m]['complements'][np.array(sel)].mean())}
        sub[grp] = d
stats_out["by_type"] = sub

# comp-vs-sub cosine AUC (per focal)
sibs = defaultdict(set)
for a, b in sub_pairs: sibs[a].add(b); sibs[b].add(a)
pos = defaultdict(set)
for a, b in cpairs: pos[a].add(b)
cosauc = {}
for m in MODS:
    if m == "tfidf":
        Smat = None  # compute row-wise to save memory
    else:
        Smat = (CEN[m] @ CEN[m].T).astype(np.float32)
    cs_auc, cs_win = [], []
    for f, ps in pos.items():
        sb = sibs.get(f)
        if not sb: continue
        psl = np.array(sorted(ps)); sbl = np.array(sorted(sb))
        if m == "tfidf":
            rowf = np.asarray((TX[f] @ TX.T).todense()).ravel()
            pcs = rowf[psl]; scs = rowf[sbl]
        else:
            pcs = Smat[f, psl]; scs = Smat[f, sbl]
        u = (pcs[:, None] > scs[None, :]).sum() + 0.5 * (pcs[:, None] == scs[None, :]).sum()
        cs_auc.append(u / (len(pcs) * len(scs)))
        cs_win.append(float(pcs.mean() > scs.mean()))
    cosauc[m] = {"comp_vs_sub_auc": float(np.mean(cs_auc)) if cs_auc else None,
                 "comp_vs_sub_focalwin": float(np.mean(cs_win)) if cs_win else None,
                 "n_focals_with_subs": len(cs_auc)}
stats_out["comp_vs_sub_cosine_auc"] = cosauc
stats_out["n_focals_with_comp_and_sub"] = max((v["n_focals_with_subs"] for v in cosauc.values()), default=0)

out_path = f"/tmp/{CTRY}_general_stats{SUFFIX}.json"
json.dump(stats_out, open(out_path, "w"), indent=1)
print("saved", out_path)
for m in MODS:
    r = spec[m]; p = r["position"]; ps = f"{p*100:5.1f}%" if p is not None else "  n/a"
    print(f"  {m:8s} pos={ps} [{r['position_lo']*100:.1f},{r['position_hi']*100:.1f}]  |d_cs|={abs(r['cohend_comp_sub']):.2f}  cos_auc={cosauc[m]['comp_vs_sub_auc']}")
