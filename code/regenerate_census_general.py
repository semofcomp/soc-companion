#!/usr/bin/env python3
"""
FULL CENSUS REGENERATION under the GENERAL-complementarity rubric (see RUBRIC_general_v1.md).

One resumable run regenerates the complement census for BOTH economies and BOTH models:
  economies: CA (Canada NAPCS level-4, 3,049 focals) and US (US NAPCS collection, 3,520 focals)
  models:    gpt-4o (PRIMARY) and gpt-4o-mini (LEGACY; unused in the paper) -- the
             paper's replication generator is Gemini-2.5-Pro via regenerate_census_gemini.py
The two generators are kept independent (no pairwise adjudication); the analysis later reports
primary / replication / consensus. Same two-pass pipeline + rapidfuzz resolution as before;
ONLY the rubric/prompts are the general version, and the substitute exclusion is economy-native
(CA = same 6-digit prefix; US = same trilateral_code).

Run LOCALLY (api.openai.com reachable). Resumable; safe to stop/restart (checkpoints per
economy x model). This is a LONG run (~6,569 focals x 2 passes x 2 models).
  pip install openai pandas rapidfuzz
  OPENAI_API_KEY=...  python3 regenerate_census_general.py
  # optional (env or project-root .env):  WORKERS=8   LIMIT=0 (>0 = first N focals/econ, for a test)
.env (with OPENAI_API_KEY) is read from the PROJECT ROOT.

Output (this folder, analytics-general/census/):
  census_complements_<ECON>_<model>.jsonl   checkpoint, one JSON/line {focal, complements:[...] }
  census_complements_<ECON>_<model>.json    consolidated {focal_code: [ {code,name,type,dimensions}, ... ]}
  ECON in {CA, US}; model in {gpt-4o, gpt-4o-mini}.
"""
import os, re, json, time, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rapidfuzz import process, fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # analytics-general/census -> project root
def O(*a): return os.path.join(HERE, *a)

# ---- config: shell env -> project-root .env -> default ----
env = {}
envpath = os.path.join(ROOT, ".env")
if os.path.exists(envpath):
    for line in open(envpath, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
def cfg(name, default=None):
    val = os.environ.get(name)
    if val is None or val == "": val = env.get(name)
    return default if (val is None or val == "") else val

API_KEY = cfg("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY (env var or project-root .env).")
MODELS  = ["gpt-4o", "gpt-4o-mini"]   # gpt-4o PRIMARY first, then mini REPLICATION
WORKERS = int(cfg("WORKERS", "8"))
LIMIT   = int(cfg("LIMIT", "0"))
NUM_COMPLEMENTS = 6
from openai import OpenAI
client = OpenAI(api_key=API_KEY)
ABORT = threading.Event()   # set when the API quota is exhausted -> stop fast, resume after topping up

# ---- economies ----
def universe(csv, sub_kind):
    df = pd.read_csv(csv, dtype=str).fillna("")
    title = dict(zip(df["code"], df["title"]))
    defn  = dict(zip(df["code"], df["definition"])) if "definition" in df.columns else {}
    tri   = dict(zip(df["code"], df["trilateral_code"])) if "trilateral_code" in df.columns else {}
    return dict(codes=df["code"].tolist(), titles=df["title"].tolist(),
                title=title, defn=defn, tri=tri, code_set=set(df["code"]), sub_kind=sub_kind)
UNI = {
    "CA": universe(os.path.join(ROOT, "data", "categories_metadata.csv"), "prefix6"),
    "US": universe(os.path.join(ROOT, "data", "US NAPCS", "us_categories_metadata.csv"), "trilateral"),
}
def same_substitute(econ, c, fc):
    if UNI[econ]["sub_kind"] == "prefix6":
        return c[:6] == fc[:6]
    a, b = UNI[econ]["tri"].get(c, ""), UNI[econ]["tri"].get(fc, "")
    return bool(a) and a == b

# ---- GENERAL rubric (guarded) -- identical to validate_rubric.py / RUBRIC_general_v1.md ----
RUBRIC = ("COMPLEMENTARY categories are realistically USED TOGETHER as part of the same activity, process, "
 "experience, or demand relationship, while serving DISTINCT, mutually reinforcing roles (having or using one "
 "increases the value or need for the other). Complementarity is GENERAL - it arises in CONSUMPTION (goods/services "
 "a household uses together in a consumer activity or experience) AND in PRODUCTION/BUSINESS (raw materials, "
 "components, equipment, or services used together in a production process, project, or supply-chain stage). "
 "FIRST identify the primary activity or process in which the FOCAL is used; THEN find categories used ALONGSIDE it "
 "in that SAME context. If the focal is a raw material, intermediate good, component, capital equipment, wholesale "
 "offering, or business service, stay within its production/industrial context - do NOT pull in unrelated consumer "
 "goods. When in doubt whether a candidate genuinely shares the focal's activity or production context, EXCLUDE it "
 "rather than guess. Complements are NOT substitutes (same need, interchangeable) and NOT near-duplicates of the "
 "focal. Judge by genuine joint use, from whichever side - consumer or producer - the focal actually belongs to.")

def chat(model, messages, temperature=0.4, retries=5):
    for a in range(retries):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=temperature).choices[0].message.content
        except Exception as e:
            m = str(e)
            if "insufficient_quota" in m or "exceeded your current quota" in m:
                ABORT.set(); raise   # billing cap: retrying won't help -> abort fast, resume later
            if a == retries - 1: raise
            time.sleep(2 * (a + 1))

def parse_json(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m: return []
    try: return json.loads(m.group(0))
    except Exception: return []

def pass1(model, U, fc):
    msg = [{"role": "user", "content":
        f"Focal category: {U['title'].get(fc,'')}\nDefinition: {U['defn'].get(fc,'')}\n\n{RUBRIC}\n\n"
        f"List up to 8 DISTINCT complementary product or service categories that are genuinely used together with the "
        f"focal in its primary activity or production process - what a user of the focal (a household consumer, OR a "
        f"producer/business, whichever the focal belongs to) would also need or use alongside it. Short category-name "
        f"noun phrases. Exclude substitutes and the focal itself. Return ONLY a JSON array of strings."}]
    return [str(x) for x in parse_json(chat(model, msg))][:8]

def resolve(econ, U, phrases, fc):
    pool = {}
    for ph in phrases:
        if len(ph.strip()) < 4: continue
        cand  = [(i, s) for t, s, i in process.extract(ph, U["titles"], scorer=fuzz.partial_ratio, limit=5) if s >= 82]
        cand += [(i, s) for t, s, i in process.extract(ph, U["titles"], scorer=fuzz.token_set_ratio, limit=3) if s >= 75]
        for i, _ in cand:
            c = U["codes"][i]
            if c == fc or same_substitute(econ, c, fc): continue
            pool[c] = U["title"][c]
    return pool

def pass2(model, econ, U, fc, pool):
    if not pool: return []
    items = list(pool.items())
    listing = "\n".join(f"{i}. [{c}] {t}" for i, (c, t) in enumerate(items))
    msg = [{"role": "user", "content":
        f"Focal category: {U['title'].get(fc,'')}\nDefinition: {U['defn'].get(fc,'')}\n\n{RUBRIC}\n\n"
        f"From the REAL NAPCS categories below, choose the {NUM_COMPLEMENTS} BEST genuine complements, ranked best first. "
        f"Each must belong to the focal's OWN activity or production context and be genuinely used together with it. "
        f"Exclude substitutes and near-duplicates. For each, give its code, its type ('product' or 'service'), and its "
        f"dimensions (any of: functional [used together to perform a task or process step], sequential [one precedes or "
        f"follows the other in a usage or production chain], occasion [shared activity, occasion, or process context], "
        f"demand_spillover [demand for one raises demand for the other]). "
        f"Return ONLY JSON: [{{\"code\":\"...\",\"type\":\"...\",\"dimensions\":[\"...\"]}}]\n\nCandidates:\n{listing}"}]
    out = parse_json(chat(model, msg)); res = []
    for o in out:
        c = str(o.get("code", "")).strip()
        if c in U["code_set"] and c != fc and not same_substitute(econ, c, fc):
            res.append({"code": c, "name": U["title"].get(c, ""), "type": o.get("type", ""), "dimensions": o.get("dimensions", [])})
        if len(res) >= NUM_COMPLEMENTS: break
    return res

def do_focal(model, econ, U, fc):
    if ABORT.is_set():
        return fc, None, "__ABORT__"          # quota hit: short-circuit, leave focal as todo
    try:
        return fc, pass2(model, econ, U, fc, resolve(econ, U, pass1(model, U, fc), fc)), None
    except Exception as e:
        return fc, None, str(e)

def run(model, econ):
    U = UNI[econ]; codes = U["codes"]
    ckpt = O(f"census_complements_{econ}_{model}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try: o = json.loads(line)
            except Exception: continue
            if not o.get("error"):            # errored focals are NOT done -> retried on resume
                done.add(o["focal"])
    todo = [c for c in codes if c not in done]
    if LIMIT > 0: todo = todo[:LIMIT]
    print(f"[{econ}/{model}] universe={len(codes)} done={len(done)} todo={len(todo)}")
    lock = threading.Lock(); n = [0]; errs = [0]
    with open(ckpt, "a", encoding="utf-8") as ck, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(do_focal, model, econ, U, fc): fc for fc in todo}
        for fut in as_completed(futs):
            fc, sel, err = fut.result()
            with lock:
                if err == "__ABORT__":         # quota: do NOT write (stays todo), skip
                    continue
                rec = {"focal": fc, "complements": [], "error": err} if err else {"focal": fc, "complements": sel}
                if err: errs[0] += 1
                ck.write(json.dumps(rec) + "\n"); ck.flush(); n[0] += 1
                if n[0] % 100 == 0: print(f"  [{econ}/{model}] {n[0]}/{len(todo)} (errors={errs[0]})")
    # consolidate
    out = {}
    for line in open(ckpt, encoding="utf-8"):
        try:
            o = json.loads(line); out[o["focal"]] = o.get("complements", [])
        except Exception: pass
    outpath = O(f"census_complements_{econ}_{model}.json")
    json.dump(out, open(outpath, "w"), indent=0)
    npairs = sum(len(v) for v in out.values())
    print(f"[{econ}/{model}] DONE focals={len(out)} pairs={npairs} -> {os.path.basename(outpath)}")

def main():
    for model in MODELS:          # gpt-4o (primary) first, then mini (replication)
        for econ in ["CA", "US"]:
            run(model, econ)
            if ABORT.is_set():
                print("\n*** API QUOTA EXHAUSTED (insufficient_quota). Stopping. ***\n"
                      "Top up credits / raise the usage limit in the OpenAI billing dashboard,\n"
                      "then re-run this script -- it resumes and retries only the missing/errored focals.")
                return
    print("\nALL DONE. 4 censuses written to analytics-general/census/ (CA/US x gpt-4o/gpt-4o-mini).")

if __name__ == "__main__":
    main()
