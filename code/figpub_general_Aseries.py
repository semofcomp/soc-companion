# DEPRECATED: superseded by analytics-general/render_exemplar_figures.py (exemplars: Nature-park visits 8325115 + Women's clothing at wholesale 5511412). Kept for reference only; do not use to render Figure 2/6.
# figpub_general_Aseries.py — A-series publication figures, GENERAL-CONSTRUCT.
#
# Renders:
#   Figure 1  : fig_A1_map2d_CA, fig_A1_map2d_US  — 2-D UMAP of the whole
#               category universe, colored by NAPCS section.
#   Figure 2  : fig_A2_focal_map  — two panels (Nature-park visits / Women's clothing) on the
#               dimmed CA universe map; focal + complements (teal) + same-6-digit
#               substitutes (coral).
#   Figure 6  : fig_A3_ego        — two panels (Nature-park visits / Women's clothing) radial ego maps,
#               "two rulers, one neighbourhood": Gemini vs GloVe cosine.
#
# House style copied from analytics-legacy/figpub_partA.py and
# analytics-general/figpub_general.py. Saves BOTH 300-dpi PNG + vector PDF to
# ../documentation/figures/. Does NOT touch legacy figures.
#
# Embeddings: GEMINI column from the parquet files. UMAP (cosine, n_neighbors=15,
# min_dist=0.1, seed 42). 2-D coords are cached to analytics-general/umap2_CA.npy
# and umap2_US.npy so Fig 2 reuses the CA coords.
#
# NAPCS "section" colouring:
#   - CA codes are 7 digits; the section is the FIRST DIGIT (1..8), matching the
#     legacy figure. (For CA, first-two-digits is a sub-section with ~40 values,
#     too many to colour distinctly, so we use the canonical first-digit section.)
#   - US codes are 10 digits; the section is the FIRST TWO DIGITS (10,20,..,90).
#   Both use the Wong colorblind-safe palette.
#
# Usage (staged, to respect the sandbox 45s cap):
#   python figpub_general_Aseries.py umap CA      # compute + cache CA coords
#   python figpub_general_Aseries.py umap US      # compute + cache US coords
#   python figpub_general_Aseries.py fig A1       # render A1 from cached coords
#   python figpub_general_Aseries.py fig A2       # render A2
#   python figpub_general_Aseries.py fig A3       # render A3 (recomputes vecs)
#   python figpub_general_Aseries.py all          # everything in one process
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

TEAL = "#1D9E75"    # complements
CORAL = "#D85A30"   # substitutes

# Wong colorblind-safe palette (8 colours) used for NAPCS sections.
WONG = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"]

# CA: section = first digit (1..8). Honest short labels per first digit.
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

# ---------- data loaders (robust to row order: map by code) ----------
def load_economy(econ):
    """Return (codes, titles, idx, gem) aligned to the metadata CSV order.
    gem is L2-normalized gemini embeddings (for cosine via dot product)."""
    if econ == "CA":
        meta_path = os.path.join(DATA, "categories_metadata.csv")
        pq_path = os.path.join(DATA, "embeddings.parquet")
    else:
        meta_path = os.path.join(USDIR, "us_categories_metadata.csv")
        pq_path = os.path.join(USDIR, "embeddings.parquet")
    meta = pd.read_csv(meta_path, dtype={"code": str})
    codes = meta.code.tolist()
    titles = dict(zip(codes, meta.title))
    idx = {c: i for i, c in enumerate(codes)}
    df = pd.read_parquet(pq_path, columns=["code", "gemini_embedding"])
    df["code"] = df["code"].astype(str)
    emap = dict(zip(df["code"], df["gemini_embedding"]))
    dim = len(np.asarray(emap[codes[0]]))
    gem = np.zeros((len(codes), dim), dtype="float32")
    for c in codes:
        gem[idx[c]] = np.asarray(emap[c], dtype="float32")
    gem /= (np.linalg.norm(gem, axis=1, keepdims=True) + 1e-12)
    return codes, titles, idx, gem


def section_key(econ, code):
    return code[0] if econ == "CA" else code[:2]


# ---------- UMAP (cached) ----------
def compute_umap(econ):
    codes, titles, idx, gem = load_economy(econ)
    method = "UMAP"
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=42, verbose=True)
        u2 = reducer.fit_transform(gem)
    except Exception as e:
        # Fallback: t-SNE (init='pca', cosine, seed 42). FLAGGED.
        sys.stderr.write(f"[WARN] UMAP failed ({e!r}); falling back to t-SNE\n")
        from sklearn.manifold import TSNE
        u2 = TSNE(n_components=2, init="pca", metric="cosine",
                  random_state=42, perplexity=30).fit_transform(gem)
        method = "TSNE"
    u2 = np.asarray(u2, dtype="float32")
    np.save(os.path.join(ROOT, f"umap2_{econ}.npy"), u2)
    with open(os.path.join(ROOT, f"umap2_{econ}.method"), "w") as f:
        f.write(method)
    print(f"{econ}: {method} coords saved, shape {u2.shape}")
    return u2, method


def load_coords(econ):
    u2 = np.load(os.path.join(ROOT, f"umap2_{econ}.npy"))
    mp = os.path.join(ROOT, f"umap2_{econ}.method")
    method = open(mp).read().strip() if os.path.exists(mp) else "UMAP"
    return u2, method


def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    plt.close(fig)


# ============ A1: 2D UMAP overview, per economy ============
def fig_A1(econ):
    codes, titles, idx = load_economy_meta(econ)
    u2, method = load_coords(econ)
    assert u2.shape[0] == len(codes), (u2.shape, len(codes))
    secs = np.array([section_key(econ, c) for c in codes])
    if econ == "CA":
        colors, labels = CA_SEC_COLORS, CA_SEC_LABELS
        title = "NAPCS section (first code digit)"
        order = sorted(colors)
    else:
        uniq = sorted(np.unique(secs))
        colors = {s: WONG[i % len(WONG)] for i, s in enumerate(uniq)}
        labels = {s: f"{s}  {us_section_label(s, codes, titles)}" for s in uniq}
        title = "NAPCS section (first two code digits)"
        order = uniq
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
    print(f"fig_A1_map2d_{econ} done ({method}), {len(order)} sections")
    return method


# US section labels: most common leading words of member titles per section.
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


def us_section_label(s, codes, titles):
    return US_SEC_LABELS_FALLBACK.get(s, "Section " + s)


def load_economy_meta(econ):
    """Metadata only (no parquet) — fast path for figures using cached coords."""
    if econ == "CA":
        meta = pd.read_csv(os.path.join(DATA, "categories_metadata.csv"), dtype={"code": str})
    else:
        meta = pd.read_csv(os.path.join(USDIR, "us_categories_metadata.csv"), dtype={"code": str})
    codes = meta.code.tolist()
    titles = dict(zip(codes, meta.title))
    idx = {c: i for i, c in enumerate(codes)}
    return codes, titles, idx


# ============ A2: focal maps on dimmed CA universe ============
def _short(titles, code, abbrev, n=30):
    if code in abbrev:
        return abbrev[code]
    t = titles.get(code, code)
    return t if len(t) <= n else t[:n - 2] + "…"


def fig_A2():
    codes, titles, idx = load_economy_meta("CA")
    u2, method = load_coords("CA")
    comps = json.load(open(os.path.join(CENSUS, "census_complements_CA_gpt-4o.json")))

    FOCALS = [
        ("8325115", "Nature-park visits", "a"),
        ("5511412", "Women's clothing, wholesale", "b"),
    ]
    ABBREV = {
        # coffee complements / substitutes
        "5511364": "Baked goods, at wholesale",
        "1731131": "Packaged fluid milk",
        "1731141": "Packaged fluid cream",
        "1831411": "Cookies, wafers & cones",
        "1731311": "Cheese",
        "1836131": "Liquid sweeteners & syrups",
        "6221241": "Full-service restaurants",
        "6221242": "Fast food restaurants",
        "6221243": "Bakeries",
        "6221244": "Bars or nightclubs",
        "6221246": "Food courts",
        "6221247": "Concession stands",
        "6221249": "Other restaurants",
        # wheat complements / substitutes
        "5411111": "Grain storage",
        "1821295": "Milled grain (animal feed)",
        "5131111": "Rail freight",
        "5132111": "Water freight",
        "5121121": "Road freight (general)",
        "1471142": "Natural-gas deliveries",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8))
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    for ax, (focal, fname, lab) in zip(axes, FOCALS):
        comp_codes = [x["code"] for x in comps.get(focal, []) if x["code"] in idx]
        sub_codes = [c for c in codes if c[:6] == focal[:6] and c != focal]
        ax.scatter(u2[:, 0], u2[:, 1], s=3.5, c="#DCDCDC", alpha=0.55,
                   linewidths=0, rasterized=True, zorder=1)
        fx, fy = u2[idx[focal]]

        def draw(code_list, color, legend_label, label_some):
            first = True
            for j, c in enumerate(code_list):
                x, y = u2[idx[c]]
                ax.plot([fx, x], [fy, y], color=color, lw=0.9, alpha=0.7, zorder=3)
                ax.scatter([x], [y], s=42, c=color, edgecolors="white",
                           linewidths=0.6, zorder=4,
                           label=(legend_label if first else None))
                first = False
                if label_some and j < label_some:
                    ax.annotate(_short(titles, c, ABBREV, 26), (x, y),
                                xytext=(4, 4), textcoords="offset points",
                                ha="left", va="bottom", fontsize=6.2,
                                color=color, zorder=5, path_effects=halo)

        draw(comp_codes, TEAL, "Complements (LLM-generated)", label_some=6)
        draw(sub_codes, CORAL, "Substitutes (same 6-digit class)", label_some=0)
        ax.scatter([fx], [fy], marker="*", s=420, c="#222222",
                   edgecolors="white", linewidths=1.0, zorder=6,
                   label=f"Focal: {fname}")
        ax.annotate(f"{fname}\n{focal}", (fx, fy), xytext=(-6, -8),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=7.5, fontweight="bold", zorder=6, path_effects=halo)
        ax.legend(loc="upper left", frameon=False, fontsize=7.2, borderaxespad=0.0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.01, 0.99, f"({lab})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        print(f"A2 panel ({lab}) {focal}: {len(comp_codes)} complements, "
              f"{len(sub_codes)} substitutes")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_A2_focal_map")
    print(f"fig_A2_focal_map done ({method} coords)")
    return method


# ============ A3: radial ego map, Gemini vs GloVe ============
def fig_A3():
    # Recompute mean-centered + normalized Gemini and GloVe vectors directly,
    # aligned to categories_metadata.csv order (safer than the legacy caches,
    # which were ordered for the legacy census).
    meta = pd.read_csv(os.path.join(DATA, "categories_metadata.csv"), dtype={"code": str})
    codes = meta.code.tolist()
    titles = dict(zip(codes, meta.title))
    idx = {c: i for i, c in enumerate(codes)}

    # Raw Gemini matrix aligned to categories_metadata.csv order. Prefer a fast
    # /tmp cache (built directly from the contiguous Arrow values) if present;
    # otherwise read the parquet via the same zero-copy path.
    cache = "/tmp/_gemraw_CA.npy"
    if os.path.exists(cache):
        gem = np.load(cache).astype("float32")
    else:
        import pyarrow.parquet as pq
        pc = pq.read_table(os.path.join(DATA, "embeddings.parquet"),
                           columns=["code"]).column("code").to_pylist()
        assert list(map(str, pc)) == codes, "parquet/meta order mismatch"
        tbl = pq.read_table(os.path.join(DATA, "embeddings.parquet"),
                            columns=["gemini_embedding"])
        la = tbl.column("gemini_embedding").combine_chunks()
        n, dim = len(la), len(la[0])
        gem = la.values.to_numpy(zero_copy_only=False).reshape(n, dim).astype("float32")
    assert gem.shape[0] == len(codes), (gem.shape, len(codes))

    glv = np.load(os.path.join(DATA, "glove_embeddings.npy")).astype("float32")
    assert glv.shape[0] == len(codes), (glv.shape, len(codes))

    def centered_norm(M):
        M = M - M.mean(axis=0, keepdims=True)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return M

    gemC = centered_norm(gem)
    glvC = centered_norm(glv)

    comps = json.load(open(os.path.join(CENSUS, "census_complements_CA_gpt-4o.json")))
    ABBREV = {
        "5511364": "Baked goods, whsl.",
        "1731131": "Fluid milk", "1731141": "Fluid cream",
        "1831411": "Cookies/cones", "1731311": "Cheese",
        "1836131": "Syrups",
        "6221241": "Full-svc restaurants", "6221242": "Fast food",
        "6221243": "Bakeries", "6221244": "Bars/nightclubs",
        "6221246": "Food courts", "6221247": "Concessions",
        "6221249": "Other restaurants",
        "5411111": "Grain storage", "1821295": "Milled feed grain",
        "5131111": "Rail freight", "5132111": "Water freight",
        "5121121": "Road freight", "1471142": "Natural-gas delivery",
    }

    def short(c):
        if c in ABBREV:
            return ABBREV[c]
        t = titles.get(c, c)
        return t if len(t) <= 22 else t[:20] + "…"

    FOCALS = [("8325115", "Nature-park\nvisits", "a"),
              ("5511412", "Women's clothing,\nwholesale", "b")]
    rings = [0.80, 0.60, 0.40, 0.20, 0.00]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2),
                             subplot_kw={"aspect": "equal"})
    for ax, (focal, fname, lab) in zip(axes, FOCALS):
        comp_codes = [x["code"] for x in comps.get(focal, []) if x["code"] in idx]
        sub_codes = [c for c in codes if c[:6] == focal[:6] and c != focal]
        neighbors = [(c, TEAL) for c in comp_codes] + [(c, CORAL) for c in sub_codes]
        n = len(neighbors)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2

        fg = gemC[idx[focal]]
        cos_g = {c: float(gemC[idx[c]] @ fg) for c, _ in neighbors}

        # GloVe ruler. Clip negative cosines to 0 so they sit on the outer ring
        # (GloVe routinely scores complements ~0 / slightly negative).
        fv = glvC[idx[focal]]
        cos_v = {c: float(glvC[idx[c]] @ fv) for c, _ in neighbors}

        # Place each neighbour at radius (1 - cos_gemini) [the "Gemini ruler"];
        # annotate BOTH the Gemini and GloVe cosine so the reader sees the two
        # rulers disagree (complements far under GloVe, pulled in under Gemini).
        ring_ang = np.deg2rad(353)
        for rc in rings:
            r = 1 - rc
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#E0E0E0",
                                    lw=0.8, zorder=1))
            ax.annotate(f"cos {rc:.2f}", (r * np.cos(ring_ang), r * np.sin(ring_ang)),
                        ha="center", va="center", fontsize=5.2, color="#A0A0A0",
                        zorder=1.5, rotation=90,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        for (c, col), ang in zip(neighbors, angles):
            rg = 1 - cos_g[c]
            xg, yg = rg * np.cos(ang), rg * np.sin(ang)
            rv = 1 - max(cos_v[c], 0.0)
            xv, yv = rv * np.cos(ang), rv * np.sin(ang)
            # faint GloVe marker (open) at its own radius along the same spoke
            ax.plot([0, max(rg, rv) * np.cos(ang)], [0, max(rg, rv) * np.sin(ang)],
                    color=col, lw=0.7, alpha=0.35, zorder=2)
            ax.scatter([xv], [yv], s=26, facecolors="none", edgecolors=col,
                       linewidths=0.9, alpha=0.65, zorder=3)
            # filled Gemini marker
            ax.scatter([xg], [yg], s=42, c=col, edgecolors="white",
                       linewidths=0.6, zorder=4)
            rl = max(1.34, rg + 0.16)
            lx, ly = rl * np.cos(ang), rl * np.sin(ang)
            ha = "left" if np.cos(ang) > 0.15 else ("right" if np.cos(ang) < -0.15 else "center")
            ax.annotate(f"{short(c)}\nGem {cos_g[c]:.2f} / GloVe {cos_v[c]:.2f}",
                        (lx, ly), ha=ha, va="center", fontsize=5.8, color=col,
                        zorder=5, linespacing=1.05,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        ax.scatter([0], [0], marker="*", s=380, c="#222222",
                   edgecolors="white", linewidths=0.9, zorder=6)
        ax.annotate(fname, (0, 0), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=7.5, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_xlim(-2.25, 2.25); ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.01, 0.99, f"({lab})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
        print(f"A3 panel ({lab}) {focal}: {len(comp_codes)} complements, "
              f"{len(sub_codes)} substitutes")

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TEAL, label="Complements (LLM-generated)"),
        Line2D([], [], marker="o", ls="", color=CORAL, label="Substitutes (same 6-digit class)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#555555",
               label="Open marker = GloVe radius; filled = Gemini radius")],
        loc="lower center", ncol=3, frameon=False, fontsize=8.0,
        bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.18, bottom=0.10)
    save(fig, "fig_A3_ego")
    print("fig_A3_ego done")


# ---------- CLI ----------
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        for ec in ["CA", "US"]:
            if not os.path.exists(os.path.join(ROOT, f"umap2_{ec}.npy")):
                compute_umap(ec)
        fig_A1("CA"); fig_A1("US"); fig_A2(); fig_A3()
    elif args[0] == "umap":
        compute_umap(args[1])
    elif args[0] == "fig":
        which = args[1]
        if which == "A1":
            fig_A1("CA"); fig_A1("US")
        elif which == "A1CA":
            fig_A1("CA")
        elif which == "A1US":
            fig_A1("US")
        elif which == "A2":
            fig_A2()
        elif which == "A3":
            fig_A3()
    print("[done]")
