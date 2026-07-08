# DEPRECATED for Figure 2/6: superseded by analytics-general/render_exemplar_figures.py (locked exemplars: Nature-park visits 8325115 + Women's clothing at wholesale 5511412). Only fig_A1 (the universe map) here remains current.
# make_umap_figures.py — regenerate the projection/example figures with REAL UMAP.
#
# Renders (300-dpi PNG + vector PDF) into ../documentation/figures/ :
#   Figure 1 : fig_A1_map2d_CA, fig_A1_map2d_US  — 2-D UMAP of the whole category
#              universe, colored by NAPCS section.
#   Figure 2 : fig_A2_focal_map  — two panels on the dimmed CA UMAP map:
#              (a) Nature-park visits 8325115 [consumption],
#              (b) Women's clothing at wholesale 5511412 [production];
#              focal + complements (teal) + same-6-digit substitutes (coral).
#   Figure 6 : fig_A3_ego  — two panels (Nature-park visits / Women's clothing) radial ego maps,
#              "two rulers": Gemini vs GloVe cosine.
#
# RUN in the same Python env you used for the earlier runs, after installing UMAP:
#     pip install umap-learn            # (pulls numba/llvmlite; one-time)
#     python make_umap_figures.py       # computes UMAP for CA+US, then renders all
#   Staged alternative (if you want to cache coords first):
#     python make_umap_figures.py umap CA
#     python make_umap_figures.py umap US
#     python make_umap_figures.py fig A1
#     python make_umap_figures.py fig A2
#     python make_umap_figures.py fig A3
#
# It looks for data relative to this file: ../data, ../data/US NAPCS, ./census.
# UMAP params: cosine metric, n_neighbors=15, min_dist=0.1, seed 42. Coords are
# cached to umap2_CA.npy / umap2_US.npy next to this script (Fig 2 reuses CA coords).
# If umap-learn is unavailable it falls back to t-SNE and writes a ".method" marker.
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(PROJ, "data")
USDIR = os.path.join(DATA, "US NAPCS")
CENSUS = os.path.join(ROOT, "census")
FIGS = os.path.join(PROJ, "documentation", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.labelsize": 9, "legend.fontsize": 8,
})

TEAL = "#1D9E75"     # complements
CORAL = "#D85A30"    # substitutes
WONG = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"]

CA_SEC_COLORS = {str(i + 1): WONG[i] for i in range(8)}
CA_SEC_LABELS = {
    "1": "1  Agriculture, food & energy products",
    "2": "2  Textiles, chemicals & materials",
    "3": "3  Metals, machinery & electronics",
    "4": "4  Vehicles, building products & finished goods",
    "5": "5  Trade, transport & logistics",
    "6": "6  Buildings, construction & contract services",
    "7": "7  Professional, financial & information services",
    "8": "8  Health, education, personal & public services",
}
US_SEC_LABELS_FALLBACK = {
    "10": "Agriculture, mining & energy",
    "20": "Processed foods, textiles & chemicals",
    "30": "Metals, machinery & electronics",
    "40": "Vehicles, furniture & finished goods",
    "50": "Trade, transport & logistics",
    "60": "Accommodation, food & utilities",
    "70": "Professional, financial & information svcs",
    "80": "Health, education & personal svcs",
    "90": "Government & special categories",
}

# ---- focal examples + short labels for their complements/substitutes ----
FOCALS = [("8325115", "Nature-park visits", "a"),
          ("5511412", "Women's clothing, wholesale", "b")]
ABBREV = {
    # Coffee shops (cafés) — complements
    "1731131": "Fluid milk", "1731141": "Fluid cream", "1831411": "Cookies/cones",
    "1731311": "Cheese", "1836131": "Syrups", "5511364": "Baked goods, whsl.",
    # Coffee shops — substitutes (same 6-digit 622124x: food-service venues)
    "6221241": "Full-svc restaurants", "6221242": "Fast food", "6221243": "Bakeries",
    "6221244": "Bars/nightclubs", "6221246": "Food courts", "6221247": "Concessions",
    "6221249": "Other restaurants",
    # Printing presses — complements
    "2721351": "Toner cartridges", "2721321": "Inkjet cartridges",
    "5511921": "Printing paper, whsl.", "2512211": "Coated printing paper",
    "3511211": "Office-machine parts", "6812131": "Pre-/post-press svcs",
    # Printing presses — substitutes (same 6-digit 343115x: industrial machinery)
    "3431152": "Printing machinery, other", "3431153": "Chemical-ind. machinery",
    "3431154": "Glass/ceramics machinery", "3431155": "Semiconductor machinery",
    "3431156": "Specialized mfg machinery",
}

# ---------- data ----------
def load_gemini(econ):
    meta_path = os.path.join(DATA, "categories_metadata.csv") if econ == "CA" \
        else os.path.join(USDIR, "us_categories_metadata.csv")
    pq_path = os.path.join(DATA, "embeddings.parquet") if econ == "CA" \
        else os.path.join(USDIR, "embeddings.parquet")
    meta = pd.read_csv(meta_path, dtype={"code": str})
    codes = meta.code.tolist(); idx = {c: i for i, c in enumerate(codes)}
    df = pd.read_parquet(pq_path, columns=["code", "gemini_embedding"])
    df["code"] = df["code"].astype(str)
    emap = dict(zip(df["code"], df["gemini_embedding"]))
    dim = len(np.asarray(emap[codes[0]]))
    gem = np.zeros((len(codes), dim), dtype="float32")
    for c in codes:
        gem[idx[c]] = np.asarray(emap[c], dtype="float32")
    gem /= (np.linalg.norm(gem, axis=1, keepdims=True) + 1e-12)
    return codes, dict(zip(codes, meta.title)), idx, gem

def load_meta(econ):
    meta_path = os.path.join(DATA, "categories_metadata.csv") if econ == "CA" \
        else os.path.join(USDIR, "us_categories_metadata.csv")
    meta = pd.read_csv(meta_path, dtype={"code": str})
    codes = meta.code.tolist()
    return codes, dict(zip(codes, meta.title)), {c: i for i, c in enumerate(codes)}

def section_key(econ, code):
    return code[0] if econ == "CA" else code[:2]

# ---------- UMAP (cached) ----------
def compute_umap(econ):
    codes, titles, idx, gem = load_gemini(econ)
    method = "UMAP"
    try:
        import umap
        u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                       random_state=42).fit_transform(gem)
    except Exception as e:
        sys.stderr.write(f"[WARN] UMAP unavailable ({e!r}); falling back to t-SNE\n")
        from sklearn.manifold import TSNE
        u2 = TSNE(n_components=2, init="pca", metric="cosine",
                  random_state=42, perplexity=30).fit_transform(gem)
        method = "TSNE"
    u2 = np.asarray(u2, dtype="float32")
    np.save(os.path.join(ROOT, f"umap2_{econ}.npy"), u2)
    open(os.path.join(ROOT, f"umap2_{econ}.method"), "w").write(method)
    print(f"{econ}: {method} coords saved {u2.shape}")
    return u2, method

def load_coords(econ):
    p = os.path.join(ROOT, f"umap2_{econ}.npy")
    if not os.path.exists(p):
        return compute_umap(econ)
    mp = os.path.join(ROOT, f"umap2_{econ}.method")
    return np.load(p), (open(mp).read().strip() if os.path.exists(mp) else "UMAP")

def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    plt.close(fig)

# ============ Figure 1: 2-D map per economy ============
def fig_A1(econ):
    codes, titles, idx = load_meta(econ)
    u2, method = load_coords(econ)
    assert u2.shape[0] == len(codes), (u2.shape, len(codes))
    secs = np.array([section_key(econ, c) for c in codes])
    if econ == "CA":
        colors, labels, order = CA_SEC_COLORS, CA_SEC_LABELS, sorted(CA_SEC_COLORS)
        title = "NAPCS section (first code digit)"
    else:
        uniq = sorted(np.unique(secs))
        colors = {s: WONG[i % len(WONG)] for i, s in enumerate(uniq)}
        labels = {s: f"{s}  {US_SEC_LABELS_FALLBACK.get(s, 'Section ' + s)}" for s in uniq}
        order, title = uniq, "NAPCS section (first two code digits)"
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for s in order:
        m = secs == s
        ax.scatter(u2[m, 0], u2[m, 1], s=4.5, c=colors[s], alpha=0.7,
                   linewidths=0, label=labels[s], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=6.6, markerscale=2.6,
              title=title, title_fontsize=7.4, handletextpad=0.2,
              borderaxespad=0.0, labelspacing=0.30, bbox_to_anchor=(-0.02, -0.02))
    save(fig, f"fig_A1_map2d_{econ}")
    print(f"fig_A1_map2d_{econ} done ({method})")

# ============ Figure 2: focal maps on dimmed CA map ============
def _short(titles, code, n=28):
    if code in ABBREV:
        return ABBREV[code]
    t = titles.get(code, code)
    return t if len(t) <= n else t[:n - 2] + "…"

def fig_A2():
    codes, titles, idx = load_meta("CA")
    u2, method = load_coords("CA")
    comps = json.load(open(os.path.join(CENSUS, "census_complements_CA_gpt-4o.json")))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for ax, (focal, fname, lab) in zip(axes, FOCALS):
        comp_codes = [x["code"] for x in comps.get(focal, []) if x["code"] in idx]
        sub_codes = [c for c in codes if c[:6] == focal[:6] and c != focal]
        ax.scatter(u2[:, 0], u2[:, 1], s=3.5, c="#DCDCDC", alpha=0.55,
                   linewidths=0, rasterized=True, zorder=1)
        fx, fy = u2[idx[focal]]
        def draw(lst, color, leg, label_some):
            first = True
            for j, c in enumerate(lst):
                x, y = u2[idx[c]]
                ax.plot([fx, x], [fy, y], color=color, lw=0.9, alpha=0.7, zorder=3)
                ax.scatter([x], [y], s=42, c=color, edgecolors="white",
                           linewidths=0.6, zorder=4, label=(leg if first else None))
                first = False
                if label_some and j < label_some:
                    ax.annotate(_short(titles, c), (x, y), xytext=(4, 4),
                                textcoords="offset points", ha="left", va="bottom",
                                fontsize=6.2, color=color, zorder=5, path_effects=halo)
        draw(comp_codes, TEAL, "Complements (LLM-generated)", 6)
        draw(sub_codes, CORAL, "Substitutes (same 6-digit class)", 6)
        ax.scatter([fx], [fy], marker="*", s=420, c="#222222", edgecolors="white",
                   linewidths=1.0, zorder=6, label=f"Focal: {fname}")
        ax.annotate(f"{fname}\n{focal}", (fx, fy), xytext=(-6, -8),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=7.5, fontweight="bold", zorder=6, path_effects=halo)
        ax.legend(loc="upper left", frameon=False, fontsize=7.2, borderaxespad=0.0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.01, 0.99, f"({lab})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        print(f"A2 ({lab}) {focal}: {len(comp_codes)} comps, {len(sub_codes)} subs")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_A2_focal_map")
    print(f"fig_A2_focal_map done ({method} coords)")

# ============ Figure 6: radial ego map, Gemini vs GloVe ============
def _centered_norm(M):
    M = M - M.mean(axis=0, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

def fig_A3():
    meta = pd.read_csv(os.path.join(DATA, "categories_metadata.csv"), dtype={"code": str})
    codes = meta.code.tolist(); titles = dict(zip(codes, meta.title))
    idx = {c: i for i, c in enumerate(codes)}
    import pyarrow.parquet as pq
    pc = pq.read_table(os.path.join(DATA, "embeddings.parquet"), columns=["code"]).column("code").to_pylist()
    assert list(map(str, pc)) == codes, "parquet/meta order mismatch"
    tbl = pq.read_table(os.path.join(DATA, "embeddings.parquet"), columns=["gemini_embedding"])
    la = tbl.column("gemini_embedding").combine_chunks()
    gem = la.values.to_numpy(zero_copy_only=False).reshape(len(la), len(la[0])).astype("float32")
    glv = np.load(os.path.join(DATA, "glove_embeddings.npy")).astype("float32")
    gemC, glvC = _centered_norm(gem), _centered_norm(glv)
    comps = json.load(open(os.path.join(CENSUS, "census_complements_CA_gpt-4o.json")))

    def short(c):
        if c in ABBREV: return ABBREV[c]
        t = titles.get(c, c); return t if len(t) <= 22 else t[:20] + "…"

    egofocals = [("8325115", "Nature-park\nvisits", "a"),
                 ("5511412", "Women's clothing,\nwholesale", "b")]
    rings = [0.80, 0.60, 0.40, 0.20, 0.00]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), subplot_kw={"aspect": "equal"})
    for ax, (focal, fname, lab) in zip(axes, egofocals):
        comp_codes = [x["code"] for x in comps.get(focal, []) if x["code"] in idx]
        sub_codes = [c for c in codes if c[:6] == focal[:6] and c != focal]
        neighbors = [(c, TEAL) for c in comp_codes] + [(c, CORAL) for c in sub_codes]
        n = len(neighbors)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        fg, fv = gemC[idx[focal]], glvC[idx[focal]]
        cos_g = {c: float(gemC[idx[c]] @ fg) for c, _ in neighbors}
        cos_v = {c: float(glvC[idx[c]] @ fv) for c, _ in neighbors}
        ring_ang = np.deg2rad(353)
        for rc in rings:
            r = 1 - rc
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#E0E0E0", lw=0.8, zorder=1))
            ax.annotate(f"cos {rc:.2f}", (r * np.cos(ring_ang), r * np.sin(ring_ang)),
                        ha="center", va="center", fontsize=5.2, color="#A0A0A0", zorder=1.5,
                        rotation=90, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        for (c, col), ang in zip(neighbors, angles):
            rg = 1 - cos_g[c]; xg, yg = rg * np.cos(ang), rg * np.sin(ang)
            rv = 1 - max(cos_v[c], 0.0); xv, yv = rv * np.cos(ang), rv * np.sin(ang)
            ax.plot([0, max(rg, rv) * np.cos(ang)], [0, max(rg, rv) * np.sin(ang)],
                    color=col, lw=0.7, alpha=0.35, zorder=2)
            ax.scatter([xv], [yv], s=26, facecolors="none", edgecolors=col,
                       linewidths=0.9, alpha=0.65, zorder=3)
            ax.scatter([xg], [yg], s=42, c=col, edgecolors="white", linewidths=0.6, zorder=4)
            rl = max(1.34, rg + 0.16); lx, ly = rl * np.cos(ang), rl * np.sin(ang)
            ha = "left" if np.cos(ang) > 0.15 else ("right" if np.cos(ang) < -0.15 else "center")
            ax.annotate(f"{short(c)}\nGem {cos_g[c]:.2f} / GloVe {cos_v[c]:.2f}", (lx, ly),
                        ha=ha, va="center", fontsize=5.8, color=col, zorder=5, linespacing=1.05,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        ax.scatter([0], [0], marker="*", s=380, c="#222222", edgecolors="white", linewidths=0.9, zorder=6)
        ax.annotate(fname, (0, 0), xytext=(0, 13), textcoords="offset points", ha="center",
                    fontsize=7.5, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_xlim(-2.25, 2.25); ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.01, 0.99, f"({lab})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        print(f"A3 ({lab}) {focal}: {len(comp_codes)} comps, {len(sub_codes)} subs")
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TEAL, label="Complements (LLM-generated)"),
        Line2D([], [], marker="o", ls="", color=CORAL, label="Substitutes (same 6-digit class)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#555555",
               label="Open marker = GloVe radius; filled = Gemini radius")],
        loc="lower center", ncol=3, frameon=False, fontsize=8.0, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.18, bottom=0.10)
    save(fig, "fig_A3_ego")
    print("fig_A3_ego done")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        for ec in ["CA", "US"]:
            compute_umap(ec)   # ALWAYS (re)compute so stale cached coords never shadow UMAP
        fig_A1("CA"); fig_A1("US"); fig_A2(); fig_A3()
    elif args[0] == "umap":
        compute_umap(args[1])
    elif args[0] == "fig":
        {"A1": lambda: (fig_A1("CA"), fig_A1("US")), "A2": fig_A2, "A3": fig_A3}[args[1]]()
    print("[done]")
