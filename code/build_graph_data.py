#!/usr/bin/env python3
"""Canonical companion graph-data generator (deterministic; supersedes graph_general.py).
Emits companion_site_v2/data/graph_{CA,US}.json with:
  - communities  : figure-facing format (size, theme, top_sections, section_entropy_bits, members=codes)
  - ecosystems_view : ecosystems.html-facing format (id,color,label,n,entropy,n_sections,sections,edges_in,
                      members=[[code,title,links_in_eco],...])
  - plus n_nodes,n_edges,density,modularity,n_communities,node_community,hubs,reciprocity,edges
Deterministic: sorted node/edge insertion + python-louvain random_state=42 (run with PYTHONHASHSEED=0)."""
import os, json, csv, math
from collections import defaultdict, Counter
import networkx as nx, community as cl
ROOT="/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
def P(*a): return os.path.join(ROOT,*a)
PALETTE=["#5DCAA5","#F0997B","#7FB5E6","#E6C84F","#B58BD6","#7 F0","#E67FA8","#8FD17A",
         "#D98B5F","#6FB0A0","#C2C24F","#9A8CE0","#E69B9B","#6FC2C2","#C98BD6","#A0B85D"]
PALETTE=[c for c in PALETTE if " " not in c]  # guard
ECON={"CA":{"census":P("analytics-general","census","census_complements_CA_gpt-4o.json"),
            "meta":P("data","categories_metadata.csv"),"sec_len":1,"out":P("companion_site_v2","data","graph_CA.json")},
      "US":{"census":P("analytics-general","census","census_complements_US_gpt-4o.json"),
            "meta":P("data","US NAPCS","us_categories_metadata.csv"),"sec_len":2,"out":P("companion_site_v2","data","graph_US.json")}}
CA_SECTION={"1":"Agriculture / mining / utilities goods","2":"Processed foods, textiles, wood & paper",
 "3":"Chemicals, plastics, minerals","4":"Metals, machinery, equipment, vehicles",
 "5":"Transportation, postal & warehousing services","6":"Construction, real estate, prof. services",
 "7":"Business, education, health & personal services","8":"Government & misc. services","9":"Other / residual"}
US_SECTION={"10":"Agriculture, mining & extraction goods","20":"Manufactured goods",
 "30":"Construction & buildings","40":"Wholesale trade services","50":"Retail trade services",
 "60":"Utilities & energy services","70":"Business, personal & other services",
 "80":"Finance, insurance & government services","90":"Other / residual"}
def load_meta(p):
    t={}
    with open(p,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f): t[str(row["code"]).strip()]=(row.get("title") or "").strip()
    return t
def secof(c,n): return str(c)[:n]
def entropy(c):
    tot=sum(c.values())
    return 0.0 if not tot else float(-sum((v/tot)*math.log2(v/tot) for v in c.values() if v>0))
def slabel(s,e): return (CA_SECTION if e=="CA" else US_SECTION).get(s,f"Section {s}")
def theme(e,top,reps):
    if not top: return "; ".join(t for t in reps[:2] if t)[:80]
    base=slabel(top[0][0],e); r=[t for t in reps[:2] if t]
    return f"{base} — e.g. {', '.join(t[:30] for t in r)}" if r else base
def build(e,cfg):
    census=json.load(open(cfg["census"],encoding="utf-8")); title=load_meta(cfg["meta"]); sl=cfg["sec_len"]
    name,typ={},{}
    for cs in census.values():
        for c in cs:
            cc=str(c.get("code","")).strip()
            if cc: name.setdefault(cc,c.get("name","")); typ.setdefault(cc,c.get("type",""))
    def ttl(c): return title.get(c) or name.get(c) or c
    edges=set((str(f).strip(),str(o.get("code","")).strip()) for f,cs in census.items() for o in cs
              if str(o.get("code","")).strip() and str(o.get("code","")).strip()!=str(f).strip())
    nodes=set(n for ed in edges for n in ed); nn,ne=len(nodes),len(edges)
    density=ne/(nn*(nn-1)) if nn>1 else 0.0
    indeg=Counter(b for _,b in edges)
    hubs=[{"code":c,"title":ttl(c),"in_degree":indeg.get(c,0),"section":secof(c,sl)} for c in sorted(nodes,key=lambda n:(-indeg.get(n,0),n))]
    mutual=set((a,b) for a,b in edges if (b,a) in edges); uniq=set(frozenset((a,b)) for a,b in mutual)
    pairs=[]
    for fs in uniq:
        a,b=sorted(fs); pairs.append({"a":a,"b":b,"title_a":ttl(a),"title_b":ttl(b),"type_a":typ.get(a,""),"type_b":typ.get(b,"")})
    pairs.sort(key=lambda p:(p["title_a"],p["title_b"]))
    reciprocity={"n_directed_mutual":len(mutual),"n_unique_pairs":len(uniq),"pct_of_edges":(len(mutual)/ne) if ne else 0.0,"pairs":pairs}
    GU=nx.Graph(); GU.add_nodes_from(sorted(nodes))
    for a,b in sorted(edges):
        if GU.has_edge(a,b): GU[a][b]["weight"]=2
        else: GU.add_edge(a,b,weight=1)
    part=cl.best_partition(GU,weight="weight",random_state=42); mod=float(cl.modularity(part,GU,weight="weight"))
    cm2n=defaultdict(list)
    for n,c in part.items(): cm2n[c].append(n)
    sizes=sorted(cm2n.items(),key=lambda kv:-len(kv[1]))
    communities,eco,node_comm=[],[],{}
    for nid,(cid,members) in enumerate(sizes):
        for m in members: node_comm[m]=nid
        secs=Counter(secof(m,sl) for m in members); sub=GU.subgraph(members)
        ranked=sorted(members,key=lambda n:(-sub.degree(n,weight="weight"),n))
        reps=[ttl(m) for m in ranked[:3]]; top=[[s,n] for s,n in secs.most_common(5)]
        th=theme(e,top,reps)
        communities.append({"id":nid,"size":len(members),"theme":th,"top_sections":top,
                            "section_entropy_bits":round(entropy(secs),3),"members":ranked})
        eco.append({"id":nid,"color":PALETTE[nid%len(PALETTE)],"label":th,"n":len(members),
                    "entropy":round(entropy(secs),2),"n_sections":len(secs),
                    "sections":[[s,n] for s,n in secs.most_common(6)],
                    "edges_in":sub.number_of_edges(),
                    "members":[[m,ttl(m),int(sub.degree(m,weight="weight"))] for m in ranked]})
    import numpy as np
    rng=np.random.default_rng(42); alln=sorted(nodes)
    def sente(ns): return entropy(Counter(secof(m,sl) for m in ns))
    mce=sum(c["size"]*c["section_entropy_bits"] for c in communities)/sum(c["size"] for c in communities)
    reb=float(np.mean([sente(list(rng.choice(alln,size=c["size"],replace=False))) for c in communities]))
    res={"economy":e,"model":"gpt-4o","census":"general","n_nodes":nn,"n_edges":ne,"density":round(density,6),
         "modularity":round(mod,4),"n_communities":len(communities),
         "community_method":"weighted Louvain (python-louvain), deterministic sorted, seed 42",
         "node_community":node_comm,"communities":communities,"ecosystems_view":eco,"hubs":hubs,
         "mean_community_entropy":round(mce,2),"random_entropy_baseline":round(reb,2),"reciprocity":reciprocity,"edges":[[a,b] for a,b in sorted(edges)]}
    json.dump(res,open(cfg["out"],"w",encoding="utf-8"),ensure_ascii=False,separators=(",",":"))
    return res
for e,cfg in ECON.items():
    r=build(e,cfg)
    print(f"{e}: {r['n_communities']} communities mod {r['modularity']}; ecosystems_view[0] keys={list(r['ecosystems_view'][0].keys())}; member0={r['ecosystems_view'][0]['members'][0]}")
