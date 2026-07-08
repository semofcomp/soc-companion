#!/usr/bin/env python3
"""
GEMINI replication census under the GENERAL-complementarity rubric.

Generates the *replication* census with a CROSS-VENDOR generator (Google Gemini), to pair
with the gpt-4o PRIMARY census from regenerate_census_general.py. Same guarded general rubric,
same two-pass pipeline + rapidfuzz resolution, same economy-native substitute exclusion
(CA = same 6-digit prefix; US = same trilateral_code) — ONLY the LLM backend changes
(OpenAI -> Google Gemini). This upgrades the robustness design from cross-SIZE/same-vendor
(gpt-4o vs gpt-4o-mini) to cross-VENDOR (OpenAI gpt-4o vs Google Gemini).

Run LOCALLY (generativelanguage.googleapis.com reachable). Resumable; safe to stop/restart.
  pip install google-genai pandas rapidfuzz
  python3 regenerate_census_gemini.py
.env (PROJECT ROOT) must provide:
  GEMINI_API_KEY=...                 (same key used for gemini-embedding-001)
  GEMINI_GEN_MODEL=gemini-2.5-pro    <-- SET THE MODEL HERE (any strong current Gemini model;
                                         e.g. gemini-2.5-pro / gemini-2.0-pro / gemini-1.5-pro).
                                         Falls back to gemini-1.5-pro if unset.
  # optional:  WORKERS=4   LIMIT=0 (>0 = first N focals/econ, for a test)

Output (this folder, analytics-general/census/), parallel to the gpt-4o files:
  census_complements_<ECON>_<MODELTAG>.jsonl   checkpoint
  census_complements_<ECON>_<MODELTAG>.json    consolidated {focal: [ {code,name,type,dimensions} ]}
  ECON in {CA, US}; MODELTAG = the Gemini model id (e.g., gemini-2.5-pro).

NOTE for downstream: the analytics scripts currently treat the replication census as
"gpt-4o-mini". Point the replication model at <MODELTAG> (this Gemini id) when you recompute
consensus / replication stats.
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

API_KEY = cfg("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY (env var or project-root .env).")
MODEL    = cfg("GEMINI_GEN_MODEL", "gemini-1.5-pro")   # SET in .env; default is a safe fallback
MODELTAG = MODEL.split("/")[-1]                          # filename-safe id
WORKERS  = int(cfg("WORKERS", "12"))                     # latency-bound (reasoning model): more workers = throughput
LIMIT    = int(cfg("LIMIT", "0"))
# Thinking budget: gemini-2.5-pro is a REASONING model (~30s/call) -> minimize thinking to speed up.
#   2.5-pro: min 128 (cannot fully disable); 2.5-flash: 0 disables; -1 = dynamic (model decides).
# Override via env GEMINI_THINK_BUDGET. Default 128 = minimal thinking for 2.5-pro. Set 0 for flash.
THINK_BUDGET = int(cfg("GEMINI_THINK_BUDGET", "128"))
NUM_COMPLEMENTS = 6

from google import genai
from google.genai import types
client = genai.Client(api_key=API_KEY)
_THINK_OK = [True]   # flips off if the API rejects thinking_config at call time

def _gen_config(temperature):
    """Config with a (low) thinking budget when supported; falls back to plain config."""
    if _THINK_OK[0]:
        try:
            return types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET))
        except Exception:
            _THINK_OK[0] = False
    return types.GenerateContentConfig(temperature=temperature)

# ---- economies (identical to the gpt-4o script) ----
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

# ---- GENERAL rubric (guarded) -- IDENTICAL to regenerate_census_general.py / RUBRIC_general_v1.md ----
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

def chat(prompt, temperature=0.4, retries=6):
    """Gemini call. Transient 429/rate-limit -> backoff & retry (Gemini limits recover, unlike a hard
    OpenAI insufficient_quota wall), so no global abort; exhausted focals are retried on resume."""
    for a in range(retries):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=prompt, config=_gen_config(temperature))
            return r.text or ""
        except Exception as e:
            m = str(e).lower()
            if _THINK_OK[0] and ("thinking" in m or "thinking_budget" in m):
                _THINK_OK[0] = False   # this model/SDK won't take a thinking budget -> retry plain
                continue
            transient = ("429" in m or "resource_exhausted" in m or "rate" in m or
                         "503" in m or "unavailable" in m or "timeout" in m or "500" in m)
            if a == retries - 1: raise
            time.sleep((6 if transient else 2) * (a + 1))   # longer backoff on rate limits

def parse_json(txt):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m: return []
    try: return json.loads(m.group(0))
    except Exception: return []

def pass1(U, fc):
    prompt = (f"Focal category: {U['title'].get(fc,'')}\nDefinition: {U['defn'].get(fc,'')}\n\n{RUBRIC}\n\n"
        f"List up to 8 DISTINCT complementary product or service categories that are genuinely used together with the "
        f"focal in its primary activity or production process - what a user of the focal (a household consumer, OR a "
        f"producer/business, whichever the focal belongs to) would also need or use alongside it. Short category-name "
        f"noun phrases. Exclude substitutes and the focal itself. Return ONLY a JSON array of strings.")
    return [str(x) for x in parse_json(chat(prompt))][:8]

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

def pass2(econ, U, fc, pool):
    if not pool: return []
    items = list(pool.items())
    listing = "\n".join(f"{i}. [{c}] {t}" for i, (c, t) in enumerate(items))
    prompt = (f"Focal category: {U['title'].get(fc,'')}\nDefinition: {U['defn'].get(fc,'')}\n\n{RUBRIC}\n\n"
        f"From the REAL NAPCS categories below, choose the {NUM_COMPLEMENTS} BEST genuine complements, ranked best first. "
        f"Each must belong to the focal's OWN activity or production context and be genuinely used together with it. "
        f"Exclude substitutes and near-duplicates. For each, give its code, its type ('product' or 'service'), and its "
        f"dimensions (any of: functional [used together to perform a task or process step], sequential [one precedes or "
        f"follows the other in a usage or production chain], occasion [shared activity, occasion, or process context], "
        f"demand_spillover [demand for one raises demand for the other]). "
        f"Return ONLY JSON: [{{\"code\":\"...\",\"type\":\"...\",\"dimensions\":[\"...\"]}}]\n\nCandidates:\n{listing}")
    out = parse_json(chat(prompt)); res = []
    for o in out:
        c = str(o.get("code", "")).strip()
        if c in U["code_set"] and c != fc and not same_substitute(econ, c, fc):
            res.append({"code": c, "name": U["title"].get(c, ""), "type": o.get("type", ""), "dimensions": o.get("dimensions", [])})
        if len(res) >= NUM_COMPLEMENTS: break
    return res

def do_focal(econ, U, fc):
    try:
        return fc, pass2(econ, U, fc, resolve(econ, U, pass1(U, fc), fc)), None
    except Exception as e:
        return fc, None, str(e)

def run(econ):
    U = UNI[econ]; codes = U["codes"]
    ckpt = O(f"census_complements_{econ}_{MODELTAG}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try: o = json.loads(line)
            except Exception: continue
            if not o.get("error"): done.add(o["focal"])   # errored focals retried on resume
    todo = [c for c in codes if c not in done]
    if LIMIT > 0: todo = todo[:LIMIT]
    print(f"[{econ}/{MODELTAG}] universe={len(codes)} done={len(done)} todo={len(todo)} workers={WORKERS}")
    lock = threading.Lock(); n = [0]; errs = [0]
    with open(ckpt, "a", encoding="utf-8") as ck, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(do_focal, econ, U, fc): fc for fc in todo}
        for fut in as_completed(futs):
            fc, sel, err = fut.result()
            with lock:
                rec = {"focal": fc, "complements": [], "error": err} if err else {"focal": fc, "complements": sel}
                if err: errs[0] += 1
                ck.write(json.dumps(rec) + "\n"); ck.flush(); n[0] += 1
                if n[0] % 100 == 0: print(f"  [{econ}/{MODELTAG}] {n[0]}/{len(todo)} (errors={errs[0]})")
    out = {}
    for line in open(ckpt, encoding="utf-8"):
        try:
            o = json.loads(line); out[o["focal"]] = o.get("complements", [])
        except Exception: pass
    outpath = O(f"census_complements_{econ}_{MODELTAG}.json")
    json.dump(out, open(outpath, "w"), indent=0)
    npairs = sum(len(v) for v in out.values())
    print(f"[{econ}/{MODELTAG}] DONE focals={len(out)} pairs={npairs} -> {os.path.basename(outpath)}")

def main():
    print(f"Gemini replication census | model={MODEL} (tag={MODELTAG})")
    for econ in ["CA", "US"]:
        run(econ)
    print(f"\nALL DONE. Gemini censuses -> analytics-general/census/census_complements_{{CA,US}}_{MODELTAG}.json")

if __name__ == "__main__":
    main()
