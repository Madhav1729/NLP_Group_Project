"""
Surface failing queries that have explicit negation/exclusion markers,
or where the retrieved-vs-missed split lines up on different high-IDF
query terms (focus drift). Two passes, one per pattern.
"""

import csv
import json
import re
from math import log


NEGATION_RE = re.compile(
    r"\bnon[- ]\w+|\bnot\b|\bwithout\b|\bneglect\w*|\bother than\b|"
    r"\brather than\b|\bexcept\b|\bexclud\w*|\bno\b(?!\w)",
    re.IGNORECASE)


def load():
    docs = {int(d["id"]): d for d in json.load(open("cranfield/cran_docs.json"))}
    qrels = json.load(open("cranfield/cran_qrels.json"))
    queries = json.load(open("cranfield/cran_queries.json"))
    rows = {int(r["qid"]): r
            for r in csv.DictReader(open("part4_output/per_query_diagnostics.csv"))}
    processed = json.load(open("output/stopword_removed_docs.txt"))
    return docs, qrels, queries, rows, processed


def doc_frequency_map(processed):
    df = {}
    for doc in processed:
        for t in {t for s in doc for t in s}:
            df[t] = df.get(t, 0) + 1
    return df, len(processed)


def negation_pass(docs, qrels, queries, rows):
    print("=" * 70)
    print("PASS 1 -- Queries with negation / exclusion markers")
    print("=" * 70)

    hits = []
    for q in queries:
        qid = int(q["query number"])
        if qid not in rows:
            continue
        markers = NEGATION_RE.findall(q["query"])
        if not markers or float(rows[qid]["P@10"]) >= 0.40:
            continue
        hits.append((qid, markers, q["query"], rows[qid]))
    hits.sort(key=lambda h: float(h[3]["P@10"]))

    for qid, markers, text, r in hits[:10]:
        rel = {int(e["id"]) for e in qrels if int(e["query_num"]) == qid}
        t10 = [int(x) for x in r["top10_docs"].split(";")]
        missed = [d for d in list(rel) if d not in t10]
        print(f"\nQID={qid}  P@10={r['P@10']}  n_rel={r['n_relevant']}  "
              f"markers={markers}")
        print(f"QUERY: {text.strip()}")
        print("  Retrieved (top 3):")
        for did in t10[:3]:
            mark = "*" if did in rel else " "
            print(f"    {mark} Doc {did}: {docs[did]['title'].strip()[:70]}")
        print("  Missed relevant (first 2):")
        for did in missed[:2]:
            print(f"    - Doc {did}: {docs[did]['title'].strip()[:70]}")
            print(f"      \"{docs[did]['body'][:160].replace(chr(10), ' ')}\"")


def focus_drift_pass(docs, qrels, rows, df_map, N):
    print("\n" + "=" * 70)
    print("PASS 2 -- Focus drift: retrieved-vs-missed lock onto different tokens")
    print("=" * 70)

    def idf(t):
        return log((N + 1) / (df_map.get(t, 0) + 1))

    def tokens_in_doc(did):
        if did not in docs:
            return set()
        return set(re.findall(r"[a-z]+", docs[did]["body"].lower()))

    hits = []
    for qid_str, r in rows.items():
        qid = int(qid_str)
        if float(r["P@10"]) >= 0.30 or int(r["n_relevant"]) < 3 or r["oov"].strip():
            continue
        q_tokens = [t for t in r["tokens"].split() if t not in (".", ",") and len(t) >= 4]
        q_tokens = sorted(set(q_tokens), key=lambda t: -idf(t))[:6]
        if len(q_tokens) < 3:
            continue

        rel = {int(e["id"]) for e in qrels if int(e["query_num"]) == qid}
        t10 = [int(x) for x in r["top10_docs"].split(";")]
        non_rel = [d for d in t10 if d not in rel][:5]
        missed = [d for d in list(rel) if d not in t10][:5]
        if not non_rel or not missed:
            continue

        def overlap(tok, docs_):
            prefix = tok[:5]
            return sum(1 for d in docs_
                       if any(w.startswith(prefix) for w in tokens_in_doc(d)))

        nr_scores = {t: overlap(t, non_rel) for t in q_tokens}
        ms_scores = {t: overlap(t, missed) for t in q_tokens}
        nr_top = max(nr_scores, key=nr_scores.get)
        ms_top = max(ms_scores, key=ms_scores.get)

        if nr_top != ms_top:
            hits.append((qid, r, q_tokens, nr_top, nr_scores[nr_top],
                         ms_top, ms_scores[ms_top]))

    hits.sort(key=lambda h: float(h[1]["P@10"]))

    for qid, r, q_tokens, nr_top, nr_n, ms_top, ms_n in hits[:10]:
        rel = {int(e["id"]) for e in qrels if int(e["query_num"]) == qid}
        t10 = [int(x) for x in r["top10_docs"].split(";")]
        missed = [d for d in list(rel) if d not in t10]
        print(f"\nQID={qid}  P@10={r['P@10']}  n_rel={r['n_relevant']}")
        print(f"QUERY: {r['raw_query'].strip()}")
        print("  Top query terms by IDF: " +
              "  ".join(f"{t}(IDF={idf(t):.2f})" for t in q_tokens))
        print(f"  System latched onto : '{nr_top}' in {nr_n}/5 non-relevant top docs")
        print(f"  Relevant docs share : '{ms_top}' in {ms_n}/5 missed relevant docs")
        print("  Retrieved (top 3):")
        for did in t10[:3]:
            mark = "*" if did in rel else " "
            print(f"    {mark} Doc {did}: {docs[did]['title'].strip()[:70]}")
        print("  Missed relevant (first 2):")
        for did in missed[:2]:
            print(f"    - Doc {did}: {docs[did]['title'].strip()[:70]}")
            print(f"      \"{docs[did]['body'][:160].replace(chr(10), ' ')}\"")


def main():
    docs, qrels, queries, rows, processed = load()
    df_map, N = doc_frequency_map(processed)
    negation_pass(docs, qrels, queries, rows)
    focus_drift_pass(docs, qrels, rows, df_map, N)


if __name__ == "__main__":
    main()
