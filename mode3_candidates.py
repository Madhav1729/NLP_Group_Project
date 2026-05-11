"""
Surface failing queries with at least one OOV token. Splits the OOV cases
by severity (catastrophic = all content tokens are OOV; partial = some
survive) and shows what the surviving query degrades to.
"""

import csv
import json
from collections import Counter


def load():
    processed = json.load(open("output/stopword_removed_docs.txt"))
    docs = {int(d["id"]): d for d in json.load(open("cranfield/cran_docs.json"))}
    qrels = json.load(open("cranfield/cran_qrels.json"))
    rows = list(csv.DictReader(open("part4_output/per_query_diagnostics.csv")))
    return processed, docs, qrels, rows


def main():
    processed, docs, qrels, rows = load()
    df = Counter()
    for doc in processed:
        for t in {t for s in doc for t in s}:
            df[t] += 1
    vocab = set(df.keys())

    candidates = []
    for r in rows:
        tokens = [t for t in r["tokens"].split() if t not in (".", ",") and len(t) >= 2]
        if not tokens:
            continue
        oov = [t for t in tokens if t not in vocab]
        if not oov:
            continue
        in_vocab = [t for t in tokens if t in vocab]
        candidates.append({
            "qid": int(r["qid"]),
            "r": r,
            "oov": oov,
            "in_vocab": in_vocab,
            "severity": "CATASTROPHIC" if not in_vocab else "PARTIAL",
            "p10": float(r["P@10"]),
        })

    candidates.sort(key=lambda c: (c["severity"] != "CATASTROPHIC", c["p10"]))

    n_cat = sum(1 for c in candidates if c["severity"] == "CATASTROPHIC")
    print(f"Found {len(candidates)} OOV queries  "
          f"(catastrophic: {n_cat}, partial: {len(candidates) - n_cat})")
    print("=" * 70)

    for c in candidates[:15]:
        qid = c["qid"]
        r = c["r"]
        rel = {int(e["id"]) for e in qrels if int(e["query_num"]) == qid}
        t10 = [int(x) for x in r["top10_docs"].split(";")]
        missed = [d for d in list(rel) if d not in t10]

        print(f"\n[{c['severity']}]  QID={qid}  P@10={c['p10']:.2f}  "
              f"n_rel={r['n_relevant']}  rank1st={r['rank_first_relevant']}")
        print(f"QUERY: {r['raw_query'].strip()}")
        print(f"  OOV tokens     : {c['oov']}")
        print(f"  Surviving stems: {c['in_vocab']}")
        if c["in_vocab"]:
            print("  Surviving DFs  : " +
                  "  ".join(f"{t}={df.get(t, 0)}" for t in c["in_vocab"]))
        print("  Retrieved (top 3):")
        for did in t10[:3]:
            mark = "*" if did in rel else " "
            print(f"    {mark} Doc {did}: {docs[did]['title'].strip()[:70]}")
        print("  Missed relevant (first 2):")
        for did in missed[:2]:
            print(f"    - Doc {did}: {docs[did]['title'].strip()[:70]}")
            print(f"      \"{docs[did]['body'][:160].replace(chr(10), ' ')}\"")


if __name__ == "__main__":
    main()
