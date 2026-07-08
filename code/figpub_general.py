# figpub_general.py — publication figures from the GENERAL-CONSTRUCT analytics.
# Renders B1 (anchored spectrum, CA+US), B2 (rank/AUC, CA+US), and the NEW
# 2x2 cross-vendor survival figure. 300-dpi PNG + vector PDF. House style copied
# from analytics-legacy/figpub_partB.py. Does NOT touch legacy figures.
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
CENSUS = os.path.join(ROOT, "full-census")
XVEND = os.path.join(ROOT, "robustness-consumer-final")
FIGS = os.path.join(PROJ, "documentation", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.labelsize": 9, "legend.fontsize": 8,
})

TEAL = "#1D9E75"
CORAL = "#D85A30"
MODELS = ["tfidf", "glove", "gemini", "openai"]
MNAME = {"tfidf": "TF-IDF", "glove": "GloVe", "gemini": "Gemini", "openai": "OpenAI"}
MCOL = {"tfidf": "#999999", "glove": "#E69F00", "gemini": "#0072B2", "openai": "#CC79A7"}

def load(econ):
    stats = json.load(open(os.path.join(CENSUS, f"{econ}_general_stats.json")))
    rank = json.load(open(os.path.join(CENSUS, f"{econ}_rank.json")))
    probe = json.load(open(os.path.join(CENSUS, f"{econ}_probe.json")))
    return stats, rank, probe

def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    plt.close(fig)

# ============ B1: anchored similarity gradient (per economy) ============
# General spectrum: NO group/class intermediate anchors, a SINGLE CI.
# Draw random(0) -> substitute(1) line, complement diamond at `position`
# with CI bar [position_lo, position_hi], and the position value label.
def fig_B1(econ):
    stats, _, _ = load(econ)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ys = np.arange(len(MODELS))[::-1]
    for m, y in zip(MODELS, ys):
        sp = stats["spectrum"][m]
        pos, lo, hi = sp["position"], sp["position_lo"], sp["position_hi"]
        ax.plot([0, 1], [y, y], color="#CCCCCC", lw=1.4, zorder=1)
        ax.plot([0], [y], marker="o", ms=5, color="#777777", zorder=3)
        ax.plot([1], [y], marker="o", ms=5, color=CORAL, zorder=3)
        ax.plot([lo, hi], [y, y], color=TEAL, lw=3.4, alpha=0.95,
                solid_capstyle="butt", zorder=5)
        ax.plot([pos], [y], marker="D", ms=6.5, color=TEAL,
                mec="white", mew=0.7, zorder=6)
        ax.annotate(f"{pos:.3f}", (pos, y - 0.22), ha="center", va="top",
                    fontsize=7.2, color=TEAL)
    ax.set_yticks(ys)
    ax.set_yticklabels([MNAME[m] for m in MODELS])
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.65, len(MODELS) - 0.25)
    ax.set_xlabel("Normalized position on the random (0) → substitute (1) similarity scale")
    ax.annotate("random", (0, ys[0] + 0.21), ha="center", fontsize=6.6, color="#777777")
    ax.annotate("substitutes", (1, ys[0] + 0.21), ha="center", fontsize=6.6, color=CORAL)
    fig.legend(handles=[
        Line2D([], [], marker="D", ls="-", lw=3.0, color=TEAL, mec="white",
               label="Complements (diamond = mean position; bar = 95% CI)")],
        loc="lower center", frameon=False, fontsize=7.4, handlelength=2.4,
        bbox_to_anchor=(0.5, -0.14))
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    save(fig, f"fig_B1_spectrum_{econ}")
    return {m: round(stats["spectrum"][m]["position"] * 100, 1) for m in MODELS}

# ============ B2: rank + AUC panels (per economy) ============
def fig_B2(econ):
    stats, rank, probe = load(econ)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    # (a) mean percentile rank
    ax = axes[0]
    vals = [rank["per_model"][m]["mean_pct"] * 100 for m in MODELS]
    ax.bar(range(4), vals, width=0.62, color=[MCOL[m] for m in MODELS])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}", (i, v + 1.2), ha="center", fontsize=7.6)
    ax.axhline(50, color="#888888", lw=0.9, ls="--")
    ax.annotate("chance = 50", (3.45, 51.5), ha="right", fontsize=6.8, color="#888888")
    ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("Mean complement percentile rank")
    ax.set_ylim(0, 100)
    ax.text(-0.16, 1.02, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    # (b) comp-vs-sub AUC: cosine (source-of-truth) vs probe
    ax = axes[1]
    x = np.arange(4)
    w = 0.36
    cos_auc = [stats["comp_vs_sub_cosine_auc"][m]["comp_vs_sub_auc"] for m in MODELS]
    probe_auc = [probe["per_model"][m]["auc_comp_vs_sub"] for m in MODELS]
    ax.bar(x - w / 2, cos_auc, width=w, color="#BBBBBB", label="Cosine similarity")
    ax.bar(x + w / 2, probe_auc, width=w, color="#0072B2", label="Linear probe (held-out)")
    for i, v in enumerate(cos_auc):
        ax.annotate(f"{v:.2f}", (i - w / 2, v + 0.015), ha="center", fontsize=7.2)
    for i, v in enumerate(probe_auc):
        ax.annotate(f"{v:.2f}", (i + w / 2, v + 0.015), ha="center", fontsize=7.2)
    ax.axhline(0.5, color="#888888", lw=0.9, ls="--")
    ax.annotate("chance = 0.5", (3.62, 0.515), ha="right", fontsize=6.8, color="#888888")
    ax.set_xticks(x); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=8)
    ax.set_ylabel("AUC: complements vs. substitutes")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", frameon=False, fontsize=7.4)
    ax.text(-0.13, 1.02, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.subplots_adjust(wspace=0.32)
    save(fig, f"fig_B2_rank_auc_{econ}")
    return ([round(v, 1) for v in vals],
            [round(v, 3) for v in cos_auc],
            [round(v, 3) for v in probe_auc])

# ============ 2x2: economy x domain cross-vendor survival ============
# rows = Canada / United States ; cols = Consumption / Production.
# Each cell: two cosine-AUC bars (Gemini, OpenAI) far below the dashed 0.5
# chance line (pathology), plus the Gemini probe AUC marked above 0.5 (recovery).
def fig_2x2():
    ECON = [("CA", "Canada"), ("US", "United States")]
    DOM = [("consumer", "Consumption"), ("production", "Production")]
    NCATS = {("CA", "consumer"): 908, ("CA", "production"): 1931,
             ("US", "consumer"): 449, ("US", "production"): 2608}
    data = {}
    for ec, _ in ECON:
        x = json.load(open(os.path.join(XVEND, f"{ec}_consumer_vs_production_xvendor.json")))
        for sub, _ in DOM:
            pm = x["subsets"][sub]["per_model"]
            data[(ec, sub)] = {
                "gemini": pm["gemini"]["comp_vs_sub_cosine_auc"],
                "openai": pm["openai"]["comp_vs_sub_cosine_auc"],
                "probe": pm["gemini"]["probe_auc_comp_vs_sub"],
            }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharey=True)
    printed = {}
    for ri, (ec, eclab) in enumerate(ECON):
        for ci, (sub, domlab) in enumerate(DOM):
            ax = axes[ri][ci]
            d = data[(ec, sub)]
            # cosine bars: Gemini, OpenAI (pathology, far below 0.5)
            ax.bar([0], [d["gemini"]], width=0.55, color=MCOL["gemini"], label="Gemini cosine")
            ax.bar([1], [d["openai"]], width=0.55, color=MCOL["openai"], label="OpenAI cosine")
            # Gemini probe: teal marker/short bar (recovery, above 0.5)
            ax.bar([2], [d["probe"]], width=0.55, color=TEAL, label="Gemini probe")
            ax.plot([2], [d["probe"]], marker="D", ms=7, color=TEAL,
                    mec="white", mew=0.8, zorder=6)
            ax.axhline(0.5, color="#888888", lw=0.9, ls="--", zorder=2)
            ax.annotate(f"{d['gemini']:.3f}", (0, d["gemini"] + 0.02), ha="center", fontsize=7.4)
            ax.annotate(f"{d['openai']:.3f}", (1, d["openai"] + 0.02), ha="center", fontsize=7.4)
            ax.annotate(f"{d['probe']:.2f}", (2, d["probe"] + 0.02), ha="center",
                        fontsize=7.8, color=TEAL, fontweight="bold")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["Gem\ncos", "OpenAI\ncos", "Gem\nprobe"], fontsize=7.2)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-0.6, 2.6)
            ax.annotate(f"n = {NCATS[(ec, sub)]:,}", (0.97, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=7.0, color="#555555")
            if ci == 0:
                ax.set_ylabel(f"{eclab}\nAUC: comp. vs. sub.", fontsize=8.5)
            if ri == 0:
                ax.set_title(domlab, fontsize=10, fontweight="bold", pad=6)
            printed[f"{ec}_{sub}"] = (round(d["gemini"], 3), round(d["openai"], 3), round(d["probe"], 2))
    # chance-line legend handle + element legend
    handles = [
        Patch(facecolor=MCOL["gemini"], label="Gemini cosine AUC (pathology)"),
        Patch(facecolor=MCOL["openai"], label="OpenAI cosine AUC (pathology)"),
        Patch(facecolor=TEAL, label="Gemini probe AUC (recovery)"),
        Line2D([], [], color="#888888", lw=0.9, ls="--", label="chance = 0.5"),
    ]
    fig.legend(handles=handles, loc="lower center", frameon=False, fontsize=8,
               ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig_2x2_cross_survival")
    return printed

if __name__ == "__main__":
    out = {}
    for econ in ["CA", "US"]:
        out[f"B1_{econ}_pos%"] = fig_B1(econ)
        out[f"B2_{econ}"] = fig_B2(econ)
    out["2x2"] = fig_2x2()
    print(json.dumps(out, indent=2))
    print("General-construct figures done")
