# dimension_agreement.py — cross-generator reliability of the complementarity-dimension
# tags (functional / sequential / occasion / demand_spillover) and the product/service
# type tag, computed on the SHARED pairs of the general census (gpt-4o primary vs
# gemini-2.5-pro replication). Deterministic: pure counting, sorted iteration, no RNG.
#
# A shared pair = same focal -> same complement code present in BOTH generators'
# censuses. For each dimension, each generator gives a binary label (dimension
# present in its tag list); agreement = share of shared pairs with equal labels;
# kappa = Cohen's kappa on the 2x2 table; prevalence per generator = share tagged.
# The type tag (product/service) is a binary product-vs-service comparison — the
# same construction RESULTS_CANONICAL.json § robustness_6_7 reports.
#
# Output: analytics-general/dimension_agreement.json  (CA, US, pooled)
# Usage:  python dimension_agreement.py
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
DIMS = ["functional", "sequential", "occasion", "demand_spillover"]


def load(country, gen):
    p = os.path.join(ROOT, "census", f"census_complements_{country}_{gen}.json")
    with open(p, encoding="utf8") as f:
        return json.load(f)


def pairs(census):
    out = {}
    for focal in sorted(census):
        for rec in census[focal]:
            out[(focal, rec["code"])] = {
                "type": rec.get("type"),
                "dims": set(rec.get("dimensions") or []),
            }
    return out


def kappa(a11, a10, a01, a00):
    n = a11 + a10 + a01 + a00
    if n == 0:
        return None, None
    po = (a11 + a00) / n
    p1 = (a11 + a10) / n * (a11 + a01) / n
    p0 = (a00 + a10) / n * (a00 + a01) / n
    pe = p1 + p0
    return po, (po - pe) / (1 - pe) if pe < 1 else None


def analyze(P, G):
    shared = sorted(set(P) & set(G))
    res = {"n_shared_pairs": len(shared)}
    # type tag
    t11 = sum(1 for k in shared if P[k]["type"] == "product" and G[k]["type"] == "product")
    t00 = sum(1 for k in shared if P[k]["type"] == "service" and G[k]["type"] == "service")
    t10 = sum(1 for k in shared if P[k]["type"] == "product" and G[k]["type"] == "service")
    t01 = sum(1 for k in shared if P[k]["type"] == "service" and G[k]["type"] == "product")
    po, ka = kappa(t11, t10, t01, t00)
    res["type"] = {"agreement": po, "kappa": ka}
    # dimensions
    for d in DIMS:
        a11 = sum(1 for k in shared if d in P[k]["dims"] and d in G[k]["dims"])
        a10 = sum(1 for k in shared if d in P[k]["dims"] and d not in G[k]["dims"])
        a01 = sum(1 for k in shared if d not in P[k]["dims"] and d in G[k]["dims"])
        a00 = len(shared) - a11 - a10 - a01
        po, ka = kappa(a11, a10, a01, a00)
        res[d] = {
            "agreement": po,
            "kappa": ka,
            "prevalence_primary": (a11 + a10) / len(shared),
            "prevalence_replication": (a11 + a01) / len(shared),
        }
    return res


def main():
    out = {"_meta": {
        "primary": "gpt-4o", "replication": "gemini-2.5-pro",
        "unit": "shared focal->complement pairs (both generators)",
        "generated_by": "analytics-general/dimension_agreement.py",
    }}
    pooledP, pooledG = {}, {}
    for c in ["CA", "US"]:
        P, G = pairs(load(c, "gpt-4o")), pairs(load(c, "gemini-2.5-pro"))
        out[c] = analyze(P, G)
        pooledP.update({(c,) + k: v for k, v in P.items()})
        pooledG.update({(c,) + k: v for k, v in G.items()})
    out["pooled"] = analyze(pooledP, pooledG)
    dst = os.path.join(ROOT, "dimension_agreement.json")
    with open(dst, "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    print("wrote", dst)
    for c in ["CA", "US", "pooled"]:
        r = out[c]
        print(f"\n{c}: shared pairs {r['n_shared_pairs']}, type agr {r['type']['agreement']:.3f} kappa {r['type']['kappa']:.3f}")
        for d in DIMS:
            x = r[d]
            print(f"  {d:16s} agr {x['agreement']:.3f} kappa {x['kappa']:.3f} prev {x['prevalence_primary']:.3f}/{x['prevalence_replication']:.3f}")


if __name__ == "__main__":
    main()
