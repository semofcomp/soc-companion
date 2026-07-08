import os,json,numpy as np,pandas as pd
ROOT="/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
OUT=os.path.join(ROOT,"companion_site_v2","data")
os.makedirs(OUT,exist_ok=True)
AG=os.path.join(ROOT,"analytics-general")
XV=os.path.join(ROOT,"analytics-common","consumer-final-labels")

def build(ctry):
    if ctry=="US":
        META=os.path.join(ROOT,"data","US NAPCS","us_categories_metadata.csv")
        NPY=os.path.join(AG,"umap2_US.npy")
        XVF=os.path.join(XV,"consumer_final_xvendor_US.csv")
        seclen=2
    else:
        META=os.path.join(ROOT,"data","categories_metadata.csv")
        NPY=os.path.join(AG,"umap2_CA.npy")
        XVF=os.path.join(XV,"consumer_final_xvendor_CA.csv")
        seclen=1
    meta=pd.read_csv(META,dtype=str).fillna("")
    codes=meta["code"].tolist(); N=len(codes)
    title=dict(zip(codes,meta["title"]))
    xv=pd.read_csv(XVF,dtype=str).fillna("")
    dom=dict(zip(xv["code"],xv["final_label_xv"]))
    gtype=dict(zip(xv["code"],xv["gemini_type"])) if "gemini_type" in xv.columns else {}
    coords=np.load(NPY)
    assert coords.shape[0]==N, f"{ctry} coord len {coords.shape[0]} != N {N}"
    meta_obj={}; coord_obj={}
    for i,c in enumerate(codes):
        o={"title":title.get(c,""),"section":c[:seclen],"domain":dom.get(c,"REVIEW")}
        th=gtype.get(c,"")
        if th: o["type_hint"]=th
        meta_obj[c]=o
        coord_obj[c]=[round(float(coords[i,0]),4),round(float(coords[i,1]),4)]
    json.dump(meta_obj,open(os.path.join(OUT,f"meta_{ctry}.json"),"w"))
    json.dump(coord_obj,open(os.path.join(OUT,f"coords2d_{ctry}.json"),"w"))
    print(ctry,"meta+coords written N=",N,"coord_len_match=",coords.shape[0]==N)
    return N

for c in ["CA","US"]:
    build(c)
