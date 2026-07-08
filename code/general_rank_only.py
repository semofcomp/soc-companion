#!/usr/bin/env python3
"""Rank only. Usage: <CA|US> <variant>"""
import os,sys,json,numpy as np,pandas as pd
from collections import defaultdict
CTRY=sys.argv[1].upper(); VARIANT=sys.argv[2] if len(sys.argv)>2 else "primary"
ROOT="/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
CENS=os.path.join(ROOT,"analytics-general","census"); rng=np.random.default_rng(42)
REPL_MODEL=os.environ.get("REPL_MODEL","gemini-2.5-pro")
SUF={"primary":"","replication":"_replication","consensus":"_consensus"}[VARIANT]
if CTRY=="US":
    DATA=os.path.join(ROOT,"data","US NAPCS"); META=os.path.join(DATA,"us_categories_metadata.csv")
    EMB=os.path.join(DATA,"embeddings.parquet"); GLV=os.path.join(DATA,"us_glove_embeddings.npy")
    CP=os.path.join(CENS,"census_complements_US_gpt-4o.json"); CR=os.path.join(CENS,f"census_complements_US_{REPL_MODEL}.json")
else:
    DATA=os.path.join(ROOT,"data"); META=os.path.join(DATA,"categories_metadata.csv")
    EMB=os.path.join(DATA,"embeddings.parquet"); GLV=os.path.join(DATA,"glove_embeddings.npy")
    CP=os.path.join(CENS,"census_complements_CA_gpt-4o.json"); CR=os.path.join(CENS,f"census_complements_CA_{REPL_MODEL}.json")
meta=pd.read_csv(META,dtype=str).fillna(""); codes=meta["code"].tolist(); idx={c:i for i,c in enumerate(codes)}; N=len(codes)
if CTRY=="US":
    tri=dict(zip(codes,meta["trilateral_code"].tolist()))
    def same_sub(c,f): a,b=tri.get(c,""),tri.get(f,""); return bool(a) and a==b
else:
    def same_sub(c,f): return c[:6]==f[:6]
prim=json.load(open(CP))
if VARIANT=="primary": src=prim
elif VARIANT=="replication": src=json.load(open(CR))
else:
    repl=json.load(open(CR)); rs={f:set(o.get("code") for o in lst) for f,lst in repl.items()}
    src={}
    for f,lst in prim.items():
        k=[o for o in lst if o.get("code") in rs.get(f,set())]
        if k: src[f]=k
fi,ci=[],[]
for f,lst in src.items():
    if f not in idx: continue
    for o in lst:
        c=o.get("code")
        if c in idx and c!=f and not same_sub(c,f): fi.append(idx[f]); ci.append(idx[c])
fi=np.array(fi); ci=np.array(ci)
g=defaultdict(list)
if CTRY=="US":
    for c in codes:
        k=tri.get(c,"")
        if k: g[k].append(idx[c])
else:
    for c in codes: g[c[:6]].append(idx[c])
sa,sb=[],[]
for v in g.values():
    for i in range(len(v)):
        for j in range(i+1,len(v)): sa.append(v[i]); sb.append(v[j])
sub_a=np.array(sa); sub_b=np.array(sb)
df=pd.read_parquet(EMB); df["code"]=df["code"].astype(str); di={c:i for i,c in enumerate(df["code"])}; order=np.array([di[c] for c in codes])
def normed(M):
    n=np.linalg.norm(M,axis=1,keepdims=True); n[n<1e-9]=1; return (M/n).astype(np.float32)
G=np.vstack([np.asarray(v,np.float32) for v in df["gemini_embedding"]])[order]
O=np.vstack([np.asarray(v,np.float32) for v in df["openai_embedding"]])[order]
GL=np.load(GLV).astype(np.float32)
CEN={"gemini":normed(G-G.mean(0)),"openai":normed(O-O.mean(0)),"glove":normed(GL-GL.mean(0))}
from sklearn.feature_extraction.text import TfidfVectorizer
TX=TfidfVectorizer(stop_words="english",min_df=2).fit_transform(meta["text"].fillna("").tolist())
MODELS=["tfidf","glove","gemini","openai"]; S={}
for k in ["gemini","openai","glove"]: S[k]=(CEN[k]@CEN[k].T).astype(np.float32)
S["tfidf"]=np.asarray((TX@TX.T).todense(),dtype=np.float32)
PCT={}
for m in MODELS:
    A=S[m].copy(); np.fill_diagonal(A,-np.inf); rk=np.argsort(np.argsort(A,axis=1),axis=1); PCT[m]=((rk-1)/(N-2)).astype(np.float32)
pos=defaultdict(set)
for a,b in zip(fi,ci): pos[a].add(b)
sibs=defaultdict(set)
for a,b in zip(sub_a,sub_b): sibs[a].add(b); sibs[b].add(a)
res={"country":CTRY,"variant":VARIANT,"n_cats":N,"n_complement_pairs":int(len(fi)),"models":MODELS,"per_model":{}}
for m in MODELS:
    Pm=PCT[m]; pp=Pm[fi,ci]; aucs,p10,r50,cs=[],[],[],[]
    for f,ps in pos.items():
        psl=np.array(sorted(ps)); pct_pos=Pm[f,psl]; npos=len(psl); nneg=N-1-npos
        ranks=pct_pos*(N-2)+1; aucs.append((ranks.sum()-npos*(npos+1)/2)/(npos*nneg))
        row=Pm[f]; kk=min(50,N-2); topk=np.argpartition(-row,kk)[:kk+1]; topk=topk[topk!=f]; topk=topk[np.argsort(-row[topk])]
        p10.append(len(set(topk[:10])&ps)/10); r50.append(len(set(topk[:50])&ps)/npos)
        sbb=sibs.get(f)
        if sbb:
            sbl=np.array(sorted(sbb)); pcs=Pm[f,psl]; scs=Pm[f,sbl]
            u=(pcs[:,None]>scs[None,:]).sum()+0.5*(pcs[:,None]==scs[None,:]).sum(); cs.append(u/(len(pcs)*len(scs)))
    res["per_model"][m]={"mean_pct":float(pp.mean()),"median_pct":float(np.median(pp)),
        "frac_pct_ge_90":float((pp>=0.90).mean()),"frac_pct_ge_99":float((pp>=0.99).mean()),
        "macro_auc_comp_vs_all":float(np.mean(aucs)),"precision_at_10":float(np.mean(p10)),
        "recall_at_50":float(np.mean(r50)),"comp_vs_sub_auc":float(np.mean(cs)) if cs else None,"n_focals_with_subs":len(cs)}
    print("RANK",m,round(res["per_model"][m]["macro_auc_comp_vs_all"],3),"cvs",res["per_model"][m]["comp_vs_sub_auc"])
json.dump(res,open(f"/tmp/{CTRY}_rank{SUF}.json","w"),indent=1); print("saved rank",CTRY,VARIANT)
