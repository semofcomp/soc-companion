import os,json
ROOT="/sessions/elegant-compassionate-ramanujan/mnt/SemanticsOfComplementarity"
OUT=os.path.join(ROOT,"companion_site_v2","data")

def jload(f): return json.load(open(os.path.join(OUT,f)))
def size(f): return os.path.getsize(os.path.join(OUT,f))

man={"description":"Static-site data bundles for SemanticsOfComplementarity companion website v2 (general construct). Two economies: CA (NAPCS Canada, 7-digit) and US (NAPCS US, 10-digit).",
     "cosine_method":"Dense models (glove/gemini/openai) mean-centered over universe then L2-normalized; cosine=dot. TF-IDF: TfidfVectorizer(stop_words='english',min_df=2) on metadata 'text', rows L2-normalized, cosine=dot. Matches analytics-general/full-census/general_rank_only.py exactly.",
     "files":[]}

for ctry in ["CA","US"]:
    meta=jload(f"meta_{ctry}.json"); coords=jload(f"coords2d_{ctry}.json")
    pairs=jload(f"pairs_{ctry}.json")
    nfoc=sum(1 for v in pairs.values() if v["complements_gpt4o"])
    npairs=sum(len(v["complements_gpt4o"])+len(v["substitutes"]) for v in pairs.values())
    man["files"].append({"file":f"meta_{ctry}.json","economy":ctry,"bytes":size(f"meta_{ctry}.json"),
        "n_categories":len(meta),"description":"Per-category metadata keyed by code: title, section (NAPCS section digit(s)), domain (consumer_final|non_consumer|REVIEW xvendor consensus), type_hint."})
    man["files"].append({"file":f"coords2d_{ctry}.json","economy":ctry,"bytes":size(f"coords2d_{ctry}.json"),
        "n_categories":len(coords),"description":"UMAP 2-D coordinates [x,y] (4 dp) keyed by code, aligned to metadata CSV row order."})
    man["files"].append({"file":f"pairs_{ctry}.json","economy":ctry,"bytes":size(f"pairs_{ctry}.json"),
        "n_categories":len(meta),"n_focals_in_file":len(pairs),"n_focals_with_complements":nfoc,"n_pairs":npairs,
        "description":"Per-focal complements (gpt-4o & gemini-2.5-pro, ranked, with per-representation cosines) and same-sibling substitutes (capped 12, closest by gemini cosine)."})

json.dump(man,open(os.path.join(OUT,"bundles_manifest.json"),"w"),indent=1)
print(json.dumps(man,indent=1))
