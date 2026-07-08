#!/usr/bin/env python3
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict

ROOT = "/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
GEN = os.path.join(ROOT, "analytics-general")
CENSUS = os.path.join(GEN, "full-census")
CENS_DIR = os.path.join(GEN, "census")
FIGS = os.path.join(ROOT, "documentation", "figures")
os.makedirs(FIGS, exist_ok=True)
CACHE = "/tmp/r2_cos_cache"
os.makedirs(CACHE, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.labelsize": 9, "legend.fontsize": 8,
})
TEAL = "#1D9E75"; CORAL = "#D85A30"
MODELS = ["tfidf", "glove", "gemini", "openai"]
MNAME = {"tfidf": "TF-IDF", "glove": "GloVe", "gemini": "Gemini", "openai": "OpenAI"}
MCOL = {"tfidf": "#999999", "glove": "#E69F00", "gemini": "#0072B2", "openai": "#CC79A7"}
LEVELS = ["random", "complements", "group", "class", "substitute"]
LEVLAB = {"random": "random", "complements": "complements", "group": "group",
          "class": "class", "substitute": "substitutes"}

def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    plt.close(fig)

def compute_cosines(econ):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = np.random.default_rng(42)
    if econ == "US":
        DATA = os.path.join(ROOT, "data", "US NAPCS")
        META_CSV = os.path.join(DATA, "us_categories_metadata.csv")
        EMB_PARQUET = os.path.join(DATA, "embeddings.parquet")
        GLOVE_NPY = os.path.join(DATA, "us_glove_embeddings.npy")
        CEN_FILE = os.path.join(CENS_DIR, "census_complements_US_gpt-4o.json")
    else:
        DATA = os.path.join(ROOT, "data")
        META_CSV = os.path.join(DATA, "categories_metadata.csv")
        EMB_PARQUET = os.path.join(DATA, "embeddings.parquet")
        GLOVE_NPY = os.path.join(DATA, "glove_embeddings.npy")
        CEN_FILE = os.path.join(CENS_DIR, "census_complements_CA_gpt-4o.json")
    meta = pd.read_csv(META_CSV, dtype=str).fillna("")
    codes = meta["code"].tolist(); idx = {c: i for i, c in enumerate(codes)}; N = len(codes)
    if econ == "US":
        tri = dict(zip(meta["code"], meta["trilateral_code"]))
        def keyfun(level):
            if level == "substitute": return lambda c: tri.get(c, "") if len(tri.get(c, "")) == 11 else None
            if level == "class":      return lambda c: tri.get(c, "")[:10] if len(tri.get(c, "")) == 11 else None
            if level == "group":      return lambda c: tri.get(c, "")[:9] if len(tri.get(c, "")) == 11 else None
        def same_sub(c, f):
            a, b = tri.get(c, ""), tri.get(f, ""); return bool(a) and a == b
    else:
        def keyfun(level):
            k = {"substitute": 6, "class": 5, "group": 4}[level]; return lambda c: c[:k]
        def same_sub(c, f): return c[:6] == f[:6]
    cens = json.load(open(CEN_FILE, encoding="utf-8"))
    cpairs, cfocal = [], []
    for f, lst in cens.items():
        if f not in idx: continue
        for o in lst:
            c = o.get("code") if isinstance(o, dict) else o
            if c in idx and c != f and not same_sub(c, f):
                cpairs.append((idx[f], idx[c])); cfocal.append(f)
    cpairs = np.array(cpairs)
    def pref_pairs(level):
        g = defaultdict(list); kf = keyfun(level)
        for c in codes:
            k = kf(c)
            if k: g[k].append(idx[c])
        out = []
        for v in g.values():
            for i in range(len(v)):
                for j in range(i + 1, len(v)): out.append((v[i], v[j]))
        return out
    sub_pairs = pref_pairs("substitute"); class_pairs = pref_pairs("class"); group_pairs = pref_pairs("group")
    target = min(10000, N * (N - 1) // 2)
    seen = set(); ra, rb = [], []; attempts = 0
    while len(ra) < target and attempts < target * 50 + 200000:
        i = int(rng.integers(0, N)); j = int(rng.integers(0, N)); attempts += 1
        if i == j: continue
        key = (i, j) if i < j else (j, i)
        if key in seen: continue
        seen.add(key); ra.append(key[0]); rb.append(key[1])
    randp = list(zip(ra, rb))
    PAIRSETS = {"random": randp, "complements": [tuple(x) for x in cpairs],
                "group": group_pairs, "class": class_pairs, "substitute": sub_pairs}
    df = pd.read_parquet(EMB_PARQUET); df["code"] = df["code"].astype(str)
    di = {c: i for i, c in enumerate(df["code"])}; order = np.array([di[c] for c in codes])
    G = np.vstack([np.asarray(v, np.float32) for v in df["gemini_embedding"]])[order]
    O = np.vstack([np.asarray(v, np.float32) for v in df["openai_embedding"]])[order]
    EMB = {"gemini": G, "openai": O}
    EMB["glove"] = np.load(GLOVE_NPY).astype(np.float32)
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
    np.savez_compressed(os.path.join(CACHE, econ + ".npz"),
        **{m + "__" + ps: COS[m][ps] for m in MODELS for ps in LEVELS})
    means = {m: {ps: float(COS[m][ps].mean()) for ps in LEVELS} for m in MODELS}
    Ns = {ps: len(PAIRSETS[ps]) for ps in LEVELS}
    json.dump({"econ": econ, "means": means, "Ns": Ns},
              open(os.path.join(CACHE, econ + "_means.json"), "w"), indent=1)
    return means, Ns

def load_cos(econ): return np.load(os.path.join(CACHE, econ + ".npz"))
def load_means(econ):
    j = json.load(open(os.path.join(CACHE, econ + "_means.json"))); return j["means"], j["Ns"]
def load_stats(econ):
    stats = json.load(open(os.path.join(CENSUS, econ + "_general_stats.json")))
    rank = json.load(open(os.path.join(CENSUS, econ + "_rank.json")))
    probe = json.load(open(os.path.join(CENSUS, econ + "_probe.json")))
    return stats, rank, probe

def fig_B2a(econ):
    _, rank, _ = load_stats(econ)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    vals = [rank["per_model"][m]["mean_pct"] * 100 for m in MODELS]
    ax.bar(range(4), vals, width=0.62, color=[MCOL[m] for m in MODELS])
    for i, v in enumerate(vals): ax.annotate(f"{v:.1f}", (i, v + 1.2), ha="center", fontsize=7.8)
    ax.axhline(50, color="#888888", lw=0.9, ls="--")
    ax.annotate("chance = 50", (3.45, 51.5), ha="right", fontsize=6.8, color="#888888")
    ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("Mean complement percentile rank"); ax.set_ylim(0, 100)
    ax.text(-0.18, 1.02, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    save(fig, "fig_B2a_percentile_" + econ)
    return [round(v, 1) for v in vals]

def fig_B2b(econ):
    stats, _, probe = load_stats(econ)
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); x = np.arange(4); w = 0.36
    cos_auc = [stats["comp_vs_sub_cosine_auc"][m]["comp_vs_sub_auc"] for m in MODELS]
    probe_auc = [probe["per_model"][m]["auc_comp_vs_sub"] for m in MODELS]
    ax.bar(x - w / 2, cos_auc, width=w, color="#BBBBBB", label="Cosine similarity")
    ax.bar(x + w / 2, probe_auc, width=w, color="#0072B2", label="Linear probe (held-out)")
    for i, v in enumerate(cos_auc): ax.annotate(f"{v:.2f}", (i - w / 2, v + 0.015), ha="center", fontsize=7.2)
    for i, v in enumerate(probe_auc): ax.annotate(f"{v:.2f}", (i + w / 2, v + 0.015), ha="center", fontsize=7.2)
    ax.axhline(0.5, color="#888888", lw=0.9, ls="--")
    ax.annotate("chance = 0.5", (3.62, 0.515), ha="right", fontsize=6.8, color="#888888")
    ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("AUC: complements vs. substitutes"); ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", frameon=False, fontsize=7.4)
    ax.text(-0.15, 1.02, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    save(fig, "fig_B2b_probe_" + econ)
    return [round(v, 3) for v in cos_auc], [round(v, 3) for v in probe_auc]

def fig_spectrum_means(econ):
    means, Ns = load_means(econ)
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); xs = np.arange(len(LEVELS))
    for m in MODELS:
        ys = [means[m][lv] for lv in LEVELS]
        ax.plot(xs, ys, marker="o", ms=5, lw=1.7, color=MCOL[m], label=MNAME[m], mec="white", mew=0.6, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([LEVLAB[lv] for lv in LEVELS], fontsize=8)
    ax.set_xlabel("Relationship type (random -> substitute)")
    ax.set_ylabel("Mean cosine similarity\n(anisotropy-corrected)")
    ax.axvline(1, color="#DDDDDD", lw=8, zorder=0)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.6)
    ax.set_ylim(bottom=min(0, min(means[m]["random"] for m in MODELS) - 0.01))
    save(fig, "fig_spectrum_means_" + econ)
    return {m: {lv: round(means[m][lv], 4) for lv in LEVELS} for m in MODELS}

def fig_distributions(econ):
    cos = load_cos(econ)
    setcols = {"random": "#999999", "complements": TEAL, "substitute": CORAL}
    sets = ["random", "complements", "substitute"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, m in zip(axes.ravel(), MODELS):
        allv = np.concatenate([cos[m + "__" + s] for s in sets])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.7)
        bins = np.linspace(lo, hi, 60)
        for s in sets:
            v = cos[m + "__" + s]
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=setcols[s], alpha=0.42, lw=0)
            ax.hist(v, bins=bins, density=True, histtype="step", color=setcols[s], lw=1.2)
            ax.axvline(v.mean(), color=setcols[s], lw=1.4, ls="--", alpha=0.95)
        ax.set_title(MNAME[m], fontsize=9, fontweight="bold", pad=3)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlabel("cosine similarity", fontsize=7.6)
    handles = [Patch(facecolor=setcols[s], alpha=0.55, label=LEVLAB[s]) for s in sets]
    handles.append(Line2D([], [], color="#555555", lw=1.4, ls="--", label="mean"))
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_distributions_" + econ)
    return {m: {s: round(float(cos[m + "__" + s].mean()), 4) for s in sets} for m in MODELS}

def fig_B4_prodservice(econ):
    stats, _, _ = load_stats(econ); bt = stats["by_type"]
    np_n = bt["product"]["n"]; ns_n = bt["service"]["n"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); x = np.arange(4); w = 0.38
    prod = [bt["product"][m]["position"] * 100 for m in MODELS]
    serv = [bt["service"][m]["position"] * 100 for m in MODELS]
    ax.bar(x - w / 2, prod, width=w, color="#4C72B0", label=f"Product (n={np_n:,})")
    ax.bar(x + w / 2, serv, width=w, color="#DD8452", label=f"Service (n={ns_n:,})")
    for i, v in enumerate(prod): ax.annotate(f"{v:.1f}", (i - w / 2, v + 0.8), ha="center", fontsize=7.0)
    for i, v in enumerate(serv): ax.annotate(f"{v:.1f}", (i + w / 2, v + 0.8), ha="center", fontsize=7.0)
    ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("Normalized complement position (%)\n(random 0 -> substitute 100)")
    ax.set_ylim(0, max(max(prod), max(serv)) * 1.18)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    save(fig, "fig_B4_prodservice_" + econ)
    return ([round(v, 1) for v in prod], [round(v, 1) for v in serv], np_n, ns_n)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "compute":
        econ = sys.argv[2]; means, Ns = compute_cosines(econ)
        print(json.dumps({"econ": econ, "Ns": Ns,
            "means": {m: {lv: round(means[m][lv], 4) for lv in LEVELS} for m in MODELS}}, indent=1))
    elif mode == "figs":
        out = {}
        for econ in ["CA", "US"]:
            out["Fig7_percentile_" + econ] = fig_B2a(econ)
            out["Fig8_AUC_" + econ] = fig_B2b(econ)
            out["Fig3_gradient_" + econ] = fig_spectrum_means(econ)
            out["Fig5_dist_means_" + econ] = fig_distributions(econ)
            out["Fig9_prodservice_" + econ] = fig_B4_prodservice(econ)
        print(json.dumps(out, indent=1)); print("R2 figures done")
