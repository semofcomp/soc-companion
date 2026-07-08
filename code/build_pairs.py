import os,sys,json,numpy as np,pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

ROOT="/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
OUT=os.path.join(ROOT,"companion_site_v2","data")
CENS=os.path.join(ROOT,"analytics-general","census")
REPL_MODEL="gemini-2.5-pro"
SUB_CAP=12

def build(ctry):
    if ctry=="US":
        DATA=os.path.join(ROOT,"data","US NAPCS"); META=os.path.join(DATA,"us_categories_metadata.csv")
        EMB=os.path.join(DATA,"embeddings.parquet"); GLV=os.path.join(DATA,"us_glove_embeddings.npy")
        CP=os.path.join(CENS,"census_complements_US_gpt-4o.json")
        CR=os.path.join(CENS,f"census_complements_US_{REPL_MODEL}.json")
    else:
        DATA=os.path.join(ROOT,"data"); META=os.path.join(DATA,"categories_metadata.csv")
        EMB=os.path.join(DATA,"embeddings.parquet"); GLV=os.path.join(DATA,"glove_embeddings.npy")
        CP=os.path.join(CENS,"census_complements_CA_gpt-4o.json")
        CR=os.path.join(CENS,f"census_complements_CA_{REPL_MODEL}.json")

    meta=pd.read_csv(META,dtype=str).fillna(""); codes=meta["code"].tolist()
    idx={c:i for i,c in enumerate(codes)}; N=len(codes)

    if ctry=="US":
        tri=dict(zip(codes,meta["trilateral_code"].tolist()))
        def samekey(c): return tri.get(c,"")
    else:
        def samekey(c): return c[:6]

    # ---- embeddings, mean-centered + L2 normalized over universe (exactly general_rank_only.py) ----
    df=pd.read_parquet(EMB); df["code"]=df["code"].astype(str)
    di={c:i for i,c in enumerate(df["code"])}; order=np.array([di[c] for c in codes])
    def normed(M):
        n=np.linalg.norm(M,axis=1,keepdims=True); n[n<1e-9]=1; return (M/n).astype(np.float32)
    G=np.vstack([np.asarray(v,np.float32) for v in df["gemini_embedding"]])[order]
    O=np.vstack([np.asarray(v,np.float32) for v in df["openai_embedding"]])[order]
    GL=np.load(GLV).astype(np.float32)
    CEN={"gemini":normed(G-G.mean(0)),"openai":normed(O-O.mean(0)),"glove":normed(GL-GL.mean(0))}
    # tfidf: fit on meta text, rows already L2-normalized by sklearn default norm='l2'
    TX=TfidfVectorizer(stop_words="english",min_df=2).fit_transform(meta["text"].fillna("").tolist())
    TX=csr_matrix(TX)

    def cos_pair(a,b):
        # a,b are row indices; returns dict of 4 cosines
        out={}
        for k in ("tfidf","glove","gemini","openai"):
            if k=="tfidf":
                v=float(TX[a].multiply(TX[b]).sum())
            else:
                M=CEN[k]; v=float(np.dot(M[a],M[b]))
            out[k]=round(v,4)
        return out

    # ---- complements from a census json ----
    def comp_list(path,full_cos=True):
        src=json.load(open(path))
        result={}
        for f,lst in src.items():
            if f not in idx: continue
            fi=idx[f]; entries=[]
            for rank,o in enumerate(lst,1):
                c=o.get("code")
                if c not in idx: continue
                e={"code":c,"rank":rank,"type":o.get("type",""),"dimensions":o.get("dimensions",[])}
                if full_cos:
                    e["cos"]=cos_pair(fi,idx[c])
                else:
                    # gemini list: include at least gemini+glove
                    e["cos"]={"gemini":round(float(np.dot(CEN["gemini"][fi],CEN["gemini"][idx[c]])),4),
                              "glove":round(float(np.dot(CEN["glove"][fi],CEN["glove"][idx[c]])),4)}
                entries.append(e)
            result[f]=entries
        return result

    comp_gpt=comp_list(CP,full_cos=True)
    comp_gem=comp_list(CR,full_cos=True)   # all 4 cosines for gemini complements too

    # ---- substitutes: same-key siblings in universe, capped ----
    groups=defaultdict(list)
    for c in codes:
        k=samekey(c)
        if k: groups[k].append(c)

    focals=set(comp_gpt)|set(comp_gem)
    rng=np.random.default_rng(42)
    pairs={}
    n_pairs=0
    for f in codes:
        fi=idx[f]
        cg=comp_gpt.get(f,[]); ce=comp_gem.get(f,[])
        k=samekey(f); sibs=[s for s in groups.get(k,[]) if s!=f]
        subs=[]
        if sibs:
            if len(sibs)>SUB_CAP:
                # keep closest by gemini cosine to be deterministic & informative
                gsim=[(s,float(np.dot(CEN["gemini"][fi],CEN["gemini"][idx[s]]))) for s in sibs]
                gsim.sort(key=lambda x:-x[1]); sibs=[s for s,_ in gsim[:SUB_CAP]]
            for s in sibs:
                subs.append({"code":s,"cos":cos_pair(fi,idx[s])})
        # only emit focal if it has any complements or substitutes
        if not cg and not ce and not subs: continue
        pairs[f]={"complements_gpt4o":cg,"complements_gemini":ce,"substitutes":subs}
        n_pairs+=len(cg)+len(subs)

    json.dump(pairs,open(os.path.join(OUT,f"pairs_{ctry}.json"),"w"))
    n_focals_comp=sum(1 for v in pairs.values() if v["complements_gpt4o"])
    print(ctry,"pairs written N=",N,"focals_in_file=",len(pairs),
          "focals_with_gpt4o_complements=",n_focals_comp,"n_pairs=",n_pairs)
    return {"n_categories":N,"n_focals_with_complements":n_focals_comp,
            "n_focals_in_file":len(pairs),"n_pairs":n_pairs}

if __name__=="__main__":
    targets=sys.argv[1:] if len(sys.argv)>1 else ["CA","US"]
    stats={}
    for c in targets:
        stats[c]=build(c)
    json.dump(stats,open(os.path.join(OUT,f"_pairs_stats_{'_'.join(targets)}.json"),"w"))
    print("STATS",json.dumps(stats))
