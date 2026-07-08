#!/usr/bin/env python3
"""Probe only, incremental per-model checkpoint. Usage: <CA|US> <variant>"""
import os, sys, json, numpy as np, pandas as pd
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
mp=N*(N-1)//2; tgt=min(10000,mp); seen=set(); ra,rb=[],[]; at=0
while len(ra)<tgt and at<tgt*50+200000:
    i=int(rng.integers(0,N)); j=int(rng.integers(0,N)); at+=1
    if i==j: continue
    key=(i,j) if i<j else (j,i)
    if key in seen: continue
    seen.add(key); ra.append(key[0]); rb.append(key[1])
rnd_a=np.array(ra); rnd_b=np.array(rb)
df=pd.read_parquet(EMB); df["code"]=df["code"].astype(str); di={c:i for i,c in enumerate(df["code"])}; order=np.array([di[c] for c in codes])
def normed(M):
    n=np.linalg.norm(M,axis=1,keepdims=True); n[n<1e-9]=1; return (M/n).astype(np.float32)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
def probe_binary(X0,g0,X1,g1):
    X=np.vstack([X0,X1]); y=np.r_[np.zeros(len(X0)),np.ones(len(X1))]; gg=np.r_[g0,g1]
    ns=min(5,len(np.unique(gg))); aucs=[]
    for tr,te in GroupKFold(ns).split(X,y,gg):
        if len(np.unique(y[te]))<2: continue
        sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(max_iter=300,C=1.0,class_weight="balanced")
        clf.fit(sc.transform(X[tr]),y[tr]); aucs.append(roc_auc_score(y[te],clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs)),float(np.std(aucs))
OUT=f"/tmp/{CTRY}_probe{SUF}.json"
if os.path.exists(OUT): probe_out=json.load(open(OUT))
else: probe_out={"country":CTRY,"variant":VARIANT,"n_cats":N,"per_model":{}}
ONLY=sys.argv[3] if len(sys.argv)>3 else None
MODELS=[ONLY] if ONLY else ["glove","gemini","openai","tfidf"]
for model in MODELS:
    if model in probe_out["per_model"]:
        print("skip",model); continue
    if model=="tfidf":
        TX=TfidfVectorizer(stop_words="english",min_df=2).fit_transform(meta["text"].fillna("").tolist())
        Mfull=np.asarray(TX.todense(),dtype=np.float32); pca=False
        if Mfull.shape[1]>300: Mfull=PCA(300,random_state=42).fit_transform(Mfull).astype(np.float32); pca=True
        Smat=np.asarray((TX@TX.T).todense(),dtype=np.float32)
    else:
        col="gemini_embedding" if model=="gemini" else ("openai_embedding" if model=="openai" else None)
        if model=="glove": M=np.load(GLV).astype(np.float32)
        else: M=np.vstack([np.asarray(v,np.float32) for v in df[col]])[order]
        CENm=normed(M-M.mean(0)); Mfull=CENm; pca=False
        if Mfull.shape[1]>300: Mfull=PCA(300,random_state=42).fit_transform(Mfull).astype(np.float32); pca=True
        Smat=(CENm@CENm.T).astype(np.float32)
    def feats(a,b):
        A,B=Mfull[a],Mfull[b]; return np.hstack([A*B,np.abs(A-B)]).astype(np.float32)
    Xc=feats(fi,ci); gc=np.array(["F"+codes[i] for i in fi]); Xs=feats(sub_a,sub_b)
    gs=np.array(["S"+(tri.get(codes[i],"") if CTRY=="US" else codes[i][:6]) for i in sub_a])
    Xr=feats(rnd_a,rnd_b); gr=np.array([f"R{i}" for i in rnd_a])
    d={"pca300":pca,"n_comp":int(len(Xc)),"n_sub":int(len(Xs)),"n_rand":int(len(Xr))}
    if len(Xs)>=10:
        m_,s_=probe_binary(Xs,gs,Xc,gc); d["auc_comp_vs_sub"]=m_; d["auc_comp_vs_sub_sd"]=s_
    else: d["auc_comp_vs_sub"]=None
    m_,s_=probe_binary(Xr,gr,Xc,gc); d["auc_comp_vs_rand"]=m_; d["auc_comp_vs_rand_sd"]=s_
    cos_c=Smat[fi,ci]; cos_s=Smat[sub_a,sub_b]
    if len(cos_s)>0:
        y=np.r_[np.zeros(len(cos_s)),np.ones(len(cos_c))]; d["cosine_only_auc_comp_vs_sub"]=float(roc_auc_score(y,np.r_[cos_s,cos_c]))
    y2=np.r_[np.zeros(len(rnd_a)),np.ones(len(cos_c))]; d["cosine_only_auc_comp_vs_rand"]=float(roc_auc_score(y2,np.r_[Smat[rnd_a,rnd_b],cos_c]))
    probe_out["per_model"][model]=d; json.dump(probe_out,open(OUT,"w"),indent=1)
    print("PROBE",model,"comp_vs_sub",d.get("auc_comp_vs_sub"),"comp_vs_rand",round(d["auc_comp_vs_rand"],3))
print("done",CTRY,VARIANT,"models in file:",list(probe_out["per_model"]))
