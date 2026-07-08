#!/usr/bin/env python3
"""Build companion_site_v2/data/coords3d_{CA,US}.json — a true 3-D UMAP of the
Gemini embeddings for each economy, normalized to [0,1]^3, keyed by NAPCS code.

Both economies use IDENTICAL parameters so the two 3-D universes match:
  mean-center -> L2-normalize -> UMAP(n_components=3, n_neighbors=15,
  min_dist=0.1, metric="cosine", random_state=42) -> min-max to [0,1].

Source: data/embeddings.parquet (Canada) and data/US NAPCS/embeddings.parquet
(United States) — each has columns: code, gemini_embedding (3072-d).

Run locally (no time limit) from anywhere:
    pip install umap-learn pandas pyarrow scikit-learn numpy
    python companion_site_v2/code/build_coords3d.py
It autodetects the project root from this file's location and overwrites the two
coords3d_*.json files under companion_site_v2/data/. Takes ~1-3 min total.
"""
import os, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # project root
OUT  = os.path.join(ROOT, "companion_site_v2", "data")

ECON = {
    "CA": os.path.join(ROOT, "data", "embeddings.parquet"),
    "US": os.path.join(ROOT, "data", "US NAPCS", "embeddings.parquet"),
}

def build(ec, parquet):
    t0 = time.time()
    df = pd.read_parquet(parquet)
    codes = [str(c).strip() for c in df["code"].tolist()]
    X = np.vstack(df["gemini_embedding"].apply(
        lambda v: np.asarray(v, dtype=np.float32)).tolist())
    # anisotropy correction: mean-center, then L2-normalize rows
    Xc = X - X.mean(0, keepdims=True)
    Xc = Xc / np.clip(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-9, None)
    import umap
    emb = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=42,
                    verbose=True).fit_transform(Xc)
    lo, hi = emb.min(0, keepdims=True), emb.max(0, keepdims=True)
    emb = (emb - lo) / np.where(hi - lo == 0, 1, hi - lo)
    out = {c: [round(float(emb[i, 0]), 4),
               round(float(emb[i, 1]), 4),
               round(float(emb[i, 2]), 4)] for i, c in enumerate(codes)}
    path = os.path.join(OUT, f"coords3d_{ec}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"  {ec}: wrote {len(out)} coords -> {path}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    for ec, pq in ECON.items():
        assert os.path.exists(pq), f"missing embeddings parquet for {ec}: {pq}"
        build(ec, pq)
    print("done — coords3d_CA.json and coords3d_US.json regenerated (matched UMAP).")
