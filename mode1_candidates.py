"""
Surface failing queries that look like synonymy gaps: low P@10 with
enough relevant docs to be meaningful, and no OOV tokens. Run after
part4_analysis.py.
"""

import csv
import json


def load():
    docs  = {int(d["id"]): d for d in json.load(open("cranfield/cran_docs.json"))}
    qrels = json.load(open("cranfield/cran_qrels.json"))
    rows  = list(csv.DictReader(open("part4_output/per_query_diagnostics.csv")))
    return docs, qrels, rows


def main():
    docs, qrels, rows = load()

    candidates = [r for r in rows
                  if r["oov"].strip() == ""
                  and float(r["P@10"]) < 0.30
                  and int(r["n_relevant"]) >= 5]
    candidates.sort(key=lambda r: float(r["P@10"]))

    def rel_for(qid):
        return {int(e["id"]) for e in qrels if int(e["query_num"]) == qid}

    def top10(r):
        return [int(x) for x in r["top10_docs"].split(";")]

    print(f"Found {len(candidates)} candidates\n" + "=" * 70 + "\n")

    for r in candidates[:12]:
        qid = int(r["qid"])
        rel = rel_for(qid)
        t10 = top10(r)
        missed = [d for d in list(rel)[:8] if d not in t10]

        print(f"QID={qid}  P@10={r['P@10']}  n_rel={r['n_relevant']}  "
              f"rank1st={r['rank_first_relevant']}")
        print(f"QUERY: {r['raw_query'].strip()}")
        print("  Retrieved (top 3):")
        for did in t10[:3]:
            mark = "*" if did in rel else " "
            print(f"    {mark} Doc {did}: {docs[did]['title'].strip()[:80]}")
        print("  Missed relevant docs:")
        for did in missed[:4]:
            print(f"    - Doc {did}: {docs[did]['title'].strip()[:80]}")
            print(f"      \"{docs[did]['body'][:200].replace(chr(10), ' ')}\"")
        print()


if __name__ == "__main__":
    main()
