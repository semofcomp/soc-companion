#!/usr/bin/env python3
"""
CROSS-VENDOR consumer-final classifier (GEMINI backend).

Companion to classify_consumer_final.py: re-labels EVERY category in BOTH censuses as
consumer-final vs not, using Google **gemini-2.5-pro**, with a BYTE-IDENTICAL criterion/prompt,
so the consumer-final cut can be made on a CROSS-VENDOR consensus (OpenAI gpt-4o vs Google gemini)
instead of cross-size/same-vendor (gpt-4o vs gpt-4o-mini). Only the model backend changes.

Run LOCALLY (generativelanguage.googleapis.com reachable). Resumable; safe to stop/restart.
  pip install google-genai pandas
  python3 classify_consumer_final_gemini.py
.env (PROJECT ROOT) must provide GEMINI_API_KEY (same key used for the Gemini census/embeddings).
  optional: GEMINI_GEN_MODEL=gemini-2.5-pro (default)  GEMINI_THINK_BUDGET=128  WORKERS=8  BATCH=30  LIMIT=0

Outputs (next to this script, analytics-common/consumer-final-labels/):
  raw_<census>_<MODELTAG>.jsonl   checkpoint; one JSON/line {code,consumer_final,type,confidence}
  <census> in {US, CA}; MODELTAG = gemini-2.5-pro.
(The downstream merge into a cross-vendor consensus CSV is done separately, in-repo.)
"""
import os, json, time, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # consumer-final-labels -> project root
def O(*a): return os.path.join(HERE, *a)

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
MODEL    = cfg("GEMINI_GEN_MODEL", "gemini-2.5-pro")
MODELTAG = MODEL.split("/")[-1]
WORKERS  = int(cfg("WORKERS", "8"))
BATCH    = int(cfg("BATCH", "30"))
LIMIT    = int(cfg("LIMIT", "0"))
THINK_BUDGET = int(cfg("GEMINI_THINK_BUDGET", "128"))   # minimal thinking -> faster (2.5-pro floor=128)

from google import genai
from google.genai import types
client = genai.Client(api_key=API_KEY)
_THINK_OK = [True]

CENSUSES = {
    "US": os.path.join(ROOT, "data", "US NAPCS", "us_categories_metadata.csv"),
    "CA": os.path.join(ROOT, "data", "categories_metadata.csv"),
}

# ---- criterion: BYTE-IDENTICAL to classify_consumer_final.py SYSTEM ----
SYSTEM = """You are an expert economic statistician classifying categories from the North American Product Classification System (NAPCS). For each category, decide whether it is CONSUMER-FINAL, judging by the category's typical/primary use.

CONSUMER-FINAL = offerings typically acquired by households or individual consumers for their own final use (final consumption): consumer goods at retail; consumer services (personal care, dining, recreation, travel/accommodation, repair of consumer goods, education and health care for persons); residential housing/rents; consumer financial and insurance products.

NOT CONSUMER-FINAL = offerings primarily intermediate, capital, wholesale, or business/government: raw materials; agricultural/mining/extractive outputs for processing; manufacturing supplies, components, parts; machinery/equipment for production; goods sold at wholesale; non-residential construction; contract manufacturing; business/professional services bought mainly by firms (advertising, management/administrative consulting, scientific/technical services, commercial finance, freight/logistics, labor supply).

DUAL-USE RULE: if bought by both households and businesses, label CONSUMER-FINAL when a typical household buys it for its own use (e.g., passenger cars, laptops, cleaning products). A wholesale version is NOT consumer-final even if the retail version is.

Examples:
- "Iron ore, unprocessed" -> false (raw_material)
- "Plastic bottles and containers, manufacturing output" -> false (component_or_part)
- "Children's bicycles, at retail" -> true (consumer_good)
- "Compressed air, at wholesale" -> false (wholesale)
- "Haircare services" -> true (consumer_service)
- "Management consulting services" -> false (b2b_service)
- "Residential rents" -> true (residential_housing)
- "Non-residential building construction" -> false (construction)
- "Passenger cars" -> true (consumer_good)

Judge each category on its own merits. Return STRICT JSON only: an object {"results":[ ... ]} with one element per input category, each {"code":"<code>","consumer_final":true|false,"type":"<consumer_good|consumer_service|residential_housing|consumer_finance|raw_material|intermediate_good|component_or_part|capital_equipment|wholesale|construction|b2b_service|other>","confidence":<0.0-1.0>}."""

TYPES = {"consumer_good","consumer_service","residential_housing","consumer_finance",
         "raw_material","intermediate_good","component_or_part","capital_equipment",
         "wholesale","construction","b2b_service","other"}

def _config():
    base = dict(system_instruction=SYSTEM, temperature=0, response_mime_type="application/json")
    if _THINK_OK[0]:
        try:
            return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=THINK_BUDGET), **base)
        except Exception:
            _THINK_OK[0] = False
    return types.GenerateContentConfig(**base)

def classify_batch(items):
    user = "Classify each NAPCS category below.\n\nCategories:\n" + json.dumps(
        [{"code": it["code"], "text": it["text"][:600]} for it in items], ensure_ascii=False)
    for attempt in range(5):
        try:
            r = client.models.generate_content(model=MODEL, contents=user, config=_config())
            data = json.loads(r.text or "{}")
            out = {}
            for o in data.get("results", []):
                c = str(o.get("code", "")).strip()
                if not c: continue
                cf = o.get("consumer_final")
                if isinstance(cf, str): cf = cf.strip().lower() in ("true", "yes", "1")
                out[c] = {"consumer_final": bool(cf),
                          "type": o.get("type") if o.get("type") in TYPES else "other",
                          "confidence": float(o.get("confidence", 0.0) or 0.0)}
            return out
        except Exception as e:
            m = str(e).lower()
            if _THINK_OK[0] and ("thinking" in m or "thinking_budget" in m):
                _THINK_OK[0] = False; continue
            if attempt == 4: return {}
            time.sleep((6 if any(t in m for t in ("429","resource_exhausted","rate","503","unavailable","timeout","500")) else 2) * (attempt + 1))
    return {}

def load_census(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    txt = df["text"] if "text" in df.columns else (df["title"] + ": " + df.get("definition", ""))
    return list(zip(df["code"].tolist(), txt.tolist()))

def run_one(census, cats):
    ckpt = O(f"raw_{census}_{MODELTAG}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try: done.add(json.loads(line)["code"])
            except Exception: pass
    todo = [{"code": c, "text": t} for c, t in cats if c not in done]
    if LIMIT > 0: todo = todo[:LIMIT]
    batches = [todo[i:i+BATCH] for i in range(0, len(todo), BATCH)]
    print(f"[{census}/{MODELTAG}] universe={len(cats)} done={len(done)} todo={len(todo)} batches={len(batches)} workers={WORKERS}")
    lock = threading.Lock(); n = [0]
    with open(ckpt, "a", encoding="utf-8") as ck, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(classify_batch, b): b for b in batches}
        for fut in as_completed(futs):
            b = futs[fut]; res = fut.result()
            with lock:
                for it in b:
                    o = res.get(it["code"])
                    if o is None: continue
                    ck.write(json.dumps({"code": it["code"], **o}) + "\n")
                ck.flush(); n[0] += 1
                if n[0] % 20 == 0: print(f"  [{census}/{MODELTAG}] {n[0]}/{len(batches)} batches")
    got = sum(1 for _ in open(ckpt, encoding="utf-8"))
    print(f"[{census}/{MODELTAG}] DONE rows={got} -> {os.path.basename(ckpt)}")

def main():
    print(f"Gemini consumer-final classifier | model={MODEL}")
    for census, path in CENSUSES.items():
        run_one(census, load_census(path))
    print(f"\nALL DONE. Gemini labels -> raw_{{US,CA}}_{MODELTAG}.jsonl  (then run the in-repo merge).")

if __name__ == "__main__":
    main()
