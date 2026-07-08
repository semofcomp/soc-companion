#!/usr/bin/env python3
"""
Consumer-final classifier for the Semantics of Complementarity study.

Labels EVERY category in BOTH censuses as consumer-final vs not, under TWO models
(gpt-4o-mini AND gpt-4o) in a SINGLE run. Used to truncate each census to its
consumer-final subset -- the domain of the consumer-experience complementarity
construct -- before re-running the analyses. The two censuses are processed
INDEPENDENTLY (no matching); the only shared thing is the criterion below.

Run LOCALLY (api.openai.com must be reachable). Resumable; safe to stop/restart.
  pip install openai pandas
  OPENAI_API_KEY=...  python3 classify_consumer_final.py
  # optional knobs (env or .env):  WORKERS=8  BATCH=30  LIMIT=0
  #   LIMIT>0 = classify only the first N categories per census (quick test)

.env (containing OPENAI_API_KEY=...) is read from the PROJECT ROOT, as established.

Outputs (written next to this script, in analytics/consumer-final/):
  raw_<census>_<model>.jsonl    checkpoint; one JSON/line {code,consumer_final,type,confidence}
  consumer_final_<census>.csv   merged per category: BOTH models' labels + agreement + final_label
  classify_summary.json         per-census counts + model agreement rate
where <census> in {US, CA}; <model> in {gpt-4o-mini, gpt-4o}.

final_label rule: consumer_final / non_consumer when the two models AGREE; REVIEW when they disagree.
The truncation uses final_label; REVIEW rows are eyeballed before the cut.
"""
import os, json, time, threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # analytics/consumer-final -> project root
def O(*a): return os.path.join(HERE, *a)

# ---- config resolution: shell env var first, then project-root .env, then default ----
env = {}
envpath = os.path.join(ROOT, ".env")
if os.path.exists(envpath):
    for line in open(envpath, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")

def cfg(name, default=None):
    val = os.environ.get(name)
    if val is None or val == "":
        val = env.get(name)
    return default if (val is None or val == "") else val

API_KEY = cfg("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY (env var or project-root .env).")

MODELS  = ["gpt-4o-mini", "gpt-4o"]          # BOTH models, hardwired -- one run does both
WORKERS = int(cfg("WORKERS", "8"))
BATCH   = int(cfg("BATCH", "30"))
LIMIT   = int(cfg("LIMIT", "0"))
SEED    = 42

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# ---- the two censuses (processed independently); both CSVs have a 'text' column ----
CENSUSES = {
    "US": os.path.join(ROOT, "data", "US NAPCS", "us_categories_metadata.csv"),  # text = collection-code description
    "CA": os.path.join(ROOT, "data", "categories_metadata.csv"),                 # text = "title: definition"
}

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

def classify_batch(model, items):
    """items: [{code,text}]. Returns {code: {consumer_final,type,confidence}} for well-formed rows."""
    user = "Classify each NAPCS category below.\n\nCategories:\n" + json.dumps(
        [{"code": it["code"], "text": it["text"][:600]} for it in items], ensure_ascii=False)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, seed=SEED,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}])
            data = json.loads(r.choices[0].message.content)
            out = {}
            for o in data.get("results", []):
                c = str(o.get("code", "")).strip()
                if not c:
                    continue
                cf = o.get("consumer_final")
                if isinstance(cf, str):
                    cf = cf.strip().lower() in ("true", "yes", "1")
                out[c] = {"consumer_final": bool(cf),
                          "type": o.get("type") if o.get("type") in TYPES else "other",
                          "confidence": float(o.get("confidence", 0.0) or 0.0)}
            return out
        except Exception:
            if attempt == 3:
                return {}
            time.sleep(2 * (attempt + 1))
    return {}

def load_census(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    if "text" in df.columns:
        txt = df["text"]
    else:
        txt = df["title"] + ": " + df.get("definition", "")
    return list(zip(df["code"].tolist(), txt.tolist()))   # [(code, text), ...]

def run_one(census, model, cats):
    ckpt = O(f"raw_{census}_{model}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt, encoding="utf-8"):
            try:
                o = json.loads(line); done[o["code"]] = o
            except Exception:
                pass
    todo = [(c, t) for (c, t) in cats if c not in done]
    if LIMIT > 0:
        todo = todo[:LIMIT]
    print(f"[{census}/{model}] total={len(cats)} done={len(done)} todo={len(todo)}")
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    lock = threading.Lock(); n = [0]

    def work(batch):
        items = [{"code": c, "text": t} for c, t in batch]
        res = classify_batch(model, items)
        missing = [it for it in items if it["code"] not in res]   # one retry on stragglers
        if missing:
            res.update(classify_batch(model, missing))
        recs = []
        for c, t in batch:
            if c in res:
                recs.append({"code": c, **res[c]})
            else:
                recs.append({"code": c, "consumer_final": None, "type": "error", "confidence": 0.0})
        return recs

    with open(ckpt, "a", encoding="utf-8") as ck, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, b) for b in batches]
        for fut in as_completed(futs):
            for rec in fut.result():
                with lock:
                    ck.write(json.dumps(rec) + "\n"); ck.flush(); n[0] += 1
            with lock:
                if n[0] % 300 < BATCH:
                    print(f"  [{census}/{model}] ~{n[0]}/{len(todo)}")

    out = {}
    for line in open(ckpt, encoding="utf-8"):
        try:
            o = json.loads(line); out[o["code"]] = o
        except Exception:
            pass
    return out

def consolidate(census, cats, by_model):
    code2text = dict(cats)
    mini = by_model["gpt-4o-mini"]; fo = by_model["gpt-4o"]
    rows = []
    for c, _ in cats:
        m = mini.get(c, {}); f = fo.get(c, {})
        ml = m.get("consumer_final"); fl = f.get("consumer_final")
        agree = (ml is not None and ml == fl)
        final = ("consumer_final" if (agree and ml) else
                 "non_consumer" if (agree and ml is False) else "REVIEW")
        rows.append({"code": c, "text": code2text.get(c, "")[:140],
                     "mini_consumer_final": ml, "mini_type": m.get("type"), "mini_conf": m.get("confidence"),
                     "gpt4o_consumer_final": fl, "gpt4o_type": f.get("type"), "gpt4o_conf": f.get("confidence"),
                     "agree": agree, "final_label": final})
    df = pd.DataFrame(rows)
    df.to_csv(O(f"consumer_final_{census}.csv"), index=False)
    return df

def main():
    summary = {}
    for census, path in CENSUSES.items():
        if not os.path.exists(path):
            print(f"[{census}] SKIP -- not found: {path}"); continue
        cats = load_census(path)
        by_model = {model: run_one(census, model, cats) for model in MODELS}
        df = consolidate(census, cats, by_model)
        summary[census] = {
            "n_categories": len(df),
            "mini_consumer_final": int((df["mini_consumer_final"] == True).sum()),
            "gpt4o_consumer_final": int((df["gpt4o_consumer_final"] == True).sum()),
            "both_agree": int(df["agree"].sum()),
            "agreement_rate": round(float(df["agree"].mean()), 3),
            "final_consumer_final": int((df["final_label"] == "consumer_final").sum()),
            "final_non_consumer": int((df["final_label"] == "non_consumer").sum()),
            "final_review": int((df["final_label"] == "REVIEW").sum()),
        }
        print(f"\n[{census}] {summary[census]}")
    json.dump(summary, open(O("classify_summary.json"), "w"), indent=1)
    print("\nSaved consumer_final_US.csv, consumer_final_CA.csv, classify_summary.json")

if __name__ == "__main__":
    main()
