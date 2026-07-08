#!/usr/bin/env python3
"""Complement-graph analysis on the GENERAL gpt-4o census, per economy (CA, US).

Replicates the legacy methodology (analytics-legacy/analytics_02_graph.py):
hubs (in-degree), reciprocity (mutual pairs), and Louvain communities on the
undirected projection (weight 1 one-way, 2 reciprocal) with modularity and
per-community NAPCS-section themes / section entropy.

Source: analytics-general/census/census_complements_{CA,US}_gpt-4o.json
        focal_code -> [ {code,name,type,dimensions}, ... ]
Titles/sections: data/categories_metadata.csv (CA, 7-digit, section = 1st digit)
                 data/US NAPCS/us_categories_metadata.csv (US, 10-digit, section = 1st 2 digits)
Writes: companion_site_v2/data/graph_{CA,US}.json
"""
import os, json, csv, math
from collections import defaultdict, Counter
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
def P(*a): return os.path.join(ROOT, *a)

ECON = {
    "CA": {
        "census": P("analytics-general", "census", "census_complements_CA_gpt-4o.json"),
        "meta":   P("data", "categories_metadata.csv"),
        "sec_len": 1,
        "out":    P("companion_site_v2", "data", "graph_CA.json"),
    },
    "US": {
        "census": P("analytics-general", "census", "census_complements_US_gpt-4o.json"),
        "meta":   P("data", "US NAPCS", "us_categories_metadata.csv"),
        "sec_len": 2,
        "out":    P("companion_site_v2", "data", "graph_US.json"),
    },
}

CA_SECTION = {
    "1": "Agriculture / mining / utilities goods",
    "2": "Processed foods, textiles, wood & paper",
    "3": "Chemicals, plastics, minerals",
    "4": "Metals, machinery, equipment, vehicles",
    "5": "Transportation, postal & warehousing services",
    "6": "Construction, real estate, prof. services",
    "7": "Business, education, health & personal services",
    "8": "Government & misc. services",
    "9": "Other / residual",
}


def load_meta(path):
    title = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title[str(row["code"]).strip()] = (row.get("title") or "").strip()
    return title


def section_of(code, sec_len):
    return str(code)[:sec_len]


def entropy(counter):
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    p = [v / tot for v in counter.values()]
    return float(-sum(x * math.log2(x) for x in p if x > 0))


def section_label(sec, econ):
    if econ == "CA":
        return CA_SECTION.get(sec, f"Section {sec}")
    return f"Section {sec}"


def theme_for(econ, top_sections, rep_titles):
    if not top_sections:
        return "; ".join(t for t in rep_titles[:2] if t)[:80]
    sec = top_sections[0][0]
    base = section_label(sec, econ)
    reps = [t for t in rep_titles[:2] if t]
    if reps:
        return f"{base} — e.g. {', '.join(t[:30] for t in reps)}"
    return base


def build(econ, cfg):
    census = json.load(open(cfg["census"], encoding="utf-8"))
    title = load_meta(cfg["meta"])
    sec_len = cfg["sec_len"]

    # fallback names + type tags from census complement entries
    name_of, type_of = {}, {}
    for comps in census.values():
        for c in comps:
            cc = str(c.get("code", "")).strip()
            if cc:
                name_of.setdefault(cc, c.get("name", ""))
                type_of.setdefault(cc, c.get("type", ""))

    def ttl(code):
        return title.get(code) or name_of.get(code) or code

    # directed edges focal -> complement (dedup, no self-loops)
    edges = set()
    for focal, comps in census.items():
        f = str(focal).strip()
        for c in comps:
            cc = str(c.get("code", "")).strip()
            if cc and cc != f:
                edges.add((f, cc))

    nodes = set()
    for a, b in edges:
        nodes.add(a); nodes.add(b)
    n_nodes, n_edges = len(nodes), len(edges)
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    # hubs: in-degree
    indeg = Counter(b for _, b in edges)
    hubs = [{"code": code, "title": ttl(code), "in_degree": indeg.get(code, 0),
             "section": section_of(code, sec_len)}
            for code in sorted(nodes, key=lambda n: (-indeg.get(n, 0), n))]

    # reciprocity
    mutual_dir = set((a, b) for a, b in edges if (b, a) in edges)
    unique_pairs = set(frozenset((a, b)) for a, b in mutual_dir)
    pairs_out = []
    for fs in unique_pairs:
        a, b = sorted(fs)
        pairs_out.append({"a": a, "b": b, "title_a": ttl(a), "title_b": ttl(b),
                          "type_a": type_of.get(a, ""), "type_b": type_of.get(b, "")})
    pairs_out.sort(key=lambda p: (p["title_a"], p["title_b"]))
    reciprocity = {
        "n_directed_mutual": len(mutual_dir),
        "n_unique_pairs": len(unique_pairs),
        "pct_of_edges": (len(mutual_dir) / n_edges) if n_edges else 0.0,
        "pairs": pairs_out,
    }

    # communities: Louvain on undirected projection (weight 1, 2 if reciprocal)
    # Deterministic: insert nodes/edges in sorted order so the partition is
    # reproducible across runs (independent of PYTHONHASHSEED) with random_state=42.
    GU = nx.Graph()
    GU.add_nodes_from(sorted(nodes))
    for a, b in sorted(edges):
        if GU.has_edge(a, b):
            GU[a][b]["weight"] = 2
        else:
            GU.add_edge(a, b, weight=1)

    try:
        import community as community_louvain
        part = community_louvain.best_partition(GU, weight="weight", random_state=42)
        modularity = float(community_louvain.modularity(part, GU, weight="weight"))
        method = "louvain (python-louvain)"
    except Exception:
        from networkx.algorithms.community import louvain_communities, modularity as nx_mod
        comms = louvain_communities(GU, weight="weight", seed=42)
        part = {n: i for i, cm in enumerate(comms) for n in cm}
        modularity = float(nx_mod(GU, comms, weight="weight"))
        method = "louvain (networkx)"

    cm2n = defaultdict(list)
    for n, c in part.items():
        cm2n[c].append(n)
    sizes = sorted(cm2n.items(), key=lambda kv: -len(kv[1]))

    communities, node_community = [], {}
    for new_id, (cid, members) in enumerate(sizes):
        for m in members:
            node_community[m] = new_id
        secs = Counter(section_of(m, sec_len) for m in members)
        sub = GU.subgraph(members)
        ranked = sorted(members, key=lambda n: -sub.degree(n, weight="weight"))
        rep_titles = [ttl(m) for m in ranked[:3]]
        top_sections = [[s, n] for s, n in secs.most_common(5)]
        communities.append({
            "id": new_id,
            "size": len(members),
            "theme": theme_for(econ, top_sections, rep_titles),
            "top_sections": top_sections,
            "section_entropy_bits": round(entropy(secs), 3),
            "members": ranked,
        })

    result = {
        "economy": econ, "model": "gpt-4o", "census": "general",
        "n_nodes": n_nodes, "n_edges": n_edges, "density": round(density, 6),
        "modularity": round(modularity, 4), "n_communities": len(communities),
        "community_method": method,
        "node_community": node_community,
        "communities": communities,
        "hubs": hubs,
        "reciprocity": reciprocity,
        "edges": [[a, b] for a, b in sorted(edges)],
    }
    json.dump(result, open(cfg["out"], "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    return result


def main():
    for econ, cfg in ECON.items():
        r = build(econ, cfg)
        print(f"\n=== {econ} ===")
        print(f"nodes={r['n_nodes']} edges={r['n_edges']} density={r['density']}")
        print(f"communities={r['n_communities']} modularity={r['modularity']} ({r['community_method']})")
        print("top-5 hubs:")
        for h in r["hubs"][:5]:
            print(f"   {h['in_degree']:4d}  {h['code']}  {h['title']}")
        rc = r["reciprocity"]
        print(f"reciprocity: {rc['n_unique_pairs']} unique pairs, {rc['n_directed_mutual']} directed, {rc['pct_of_edges']*100:.2f}% of edges")
        print(f"wrote {cfg['out']}")


if __name__ == "__main__":
    main()
