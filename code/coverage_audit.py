# coverage_audit.py — audit of focal categories WITHOUT complements in the general
# census (gpt-4o primary), for Appendix A: are coverage failures systematic
# (concentrated in particular NAPCS sections) or idiosyncratic?
#
# Consumes: census/census_complements_{C}_gpt-4o.json,
#           ../companion_site_v2/data/meta_{C}.json (titles + section labels).
# Deterministic: pure counting, sorted iteration.
# Output: analytics-general/coverage_audit.json + console table.
# Usage:  python coverage_audit.py
import json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)


def main():
    out = {"_meta": {"generator": "gpt-4o (primary census)",
                     "generated_by": "analytics-general/coverage_audit.py"}}
    for c in ["CA", "US"]:
        census = json.load(open(os.path.join(ROOT, "census", f"census_complements_{c}_gpt-4o.json"), encoding="utf8"))
        meta = json.load(open(os.path.join(PROJ, "companion_site_v2", "data", f"meta_{c}.json"), encoding="utf8"))
        # meta: {code: {title, section, ...}} or list — normalize
        if isinstance(meta, list):
            meta = {str(m["code"]): m for m in meta}
        sec_tot, sec_miss, miss_titles = {}, {}, []
        nec = 0
        for code in sorted(census):
            m = meta.get(str(code), {})
            sec = str(m.get("section", m.get("section_label", "?")))
            sec_tot[sec] = sec_tot.get(sec, 0) + 1
            if not census[code]:
                sec_miss[sec] = sec_miss.get(sec, 0) + 1
                t = m.get("title", m.get("name", str(code)))
                miss_titles.append(t)
                if re.search(r"n\.e\.c\.|not elsewhere", str(t), re.I):
                    nec += 1
        n_missing = sum(sec_miss.values())
        rows = []
        for sec in sorted(sec_tot):
            miss = sec_miss.get(sec, 0)
            rows.append({"section": sec, "n_categories": sec_tot[sec],
                         "n_without_complements": miss,
                         "share_without": round(miss / sec_tot[sec], 4)})
        out[c] = {
            "n_focals": len(census),
            "n_with_complements": len(census) - n_missing,
            "n_without_complements": n_missing,
            "coverage": round((len(census) - n_missing) / len(census), 4),
            "n_missing_nec_titled": nec,
            "max_section_share_without": max(r["share_without"] for r in rows),
            "by_section": rows,
            "missing_titles_sample": sorted(miss_titles)[:15],
        }
        print(f"\n{c}: {out[c]['n_with_complements']}/{out[c]['n_focals']} covered "
              f"({out[c]['coverage']:.1%}); {n_missing} without; "
              f"{nec} of those titled n.e.c./not-elsewhere; "
              f"max section share without = {out[c]['max_section_share_without']:.1%}")
        for r in rows:
            print(f"  sec {r['section']:>3}: {r['n_without_complements']:>3}/{r['n_categories']:<4} ({r['share_without']:.1%})")
    dst = os.path.join(ROOT, "coverage_audit.json")
    json.dump(out, open(dst, "w", encoding="utf8"), indent=1)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
