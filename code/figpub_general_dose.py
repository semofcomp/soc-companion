#!/usr/bin/env python3
"""Figure EC.3 / B3: dose-response (mean complement percentile vs generation rank)
on the GENERAL-construct full census, both economies. House style mirrors
analytics-legacy/figpub_partB.py fig_B3. Renders fig_B3_dose_CA.{png,pdf} and
fig_B3_dose_US.{png,pdf} from analytics-general/full-census/{CA,US}_dose.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
FC = os.path.join(ROOT, "analytics-general", "full-census")
FIGS = os.path.join(ROOT, "documentation", "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.labelsize": 9, "legend.fontsize": 8,
})

MODELS = ["tfidf", "glove", "gemini", "openai"]
MNAME = {"tfidf": "TF-IDF", "glove": "GloVe", "gemini": "Gemini", "openai": "OpenAI"}
MCOL = {"tfidf": "#999999", "glove": "#E69F00", "gemini": "#0072B2", "openai": "#CC79A7"}
ranks = [1, 2, 3, 4, 5, 6]

def save(fig, stem):
    fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGS, stem + ".pdf"))
    plt.close(fig)

for ctry in ["CA", "US"]:
    d = json.load(open(os.path.join(FC, f"{ctry}_dose.json")))
    pm = d["per_model"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for m in MODELS:
        ys = [pm[m][str(r)]["mean_pct"] * 100 for r in ranks]
        ax.plot(ranks, ys, marker="o", ms=4.5, lw=1.6, color=MCOL[m],
                label=MNAME[m], mec="white", mew=0.5)
    ax.set_xlabel("Generator rank of complement (1 = listed first)")
    ax.set_ylabel("Mean percentile rank")
    ax.set_xticks(ranks)
    ax.set_ylim(50, 95)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right", ncol=2,
              columnspacing=1.0, handlelength=1.6)
    save(fig, f"fig_B3_dose_{ctry}")
    print("saved", f"fig_B3_dose_{ctry}")
