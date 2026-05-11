"""
failure_compare.py — compare baseline VSM with the three Part-5 systems
on a chosen set of "failure-case" queries.

Picks up the systems already implemented in part5.py (no changes to it).
Builds each index once, then ranks only the chosen queries, and prints a
per-query comparison.

Usage:
    python failure_compare.py 1 6 13 14 22 24
"""

import json
import os
import sys

from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from part5 import (
    preprocess, load_cranfield,
    BM25Retrieval, LSARetrieval, QueryExpansionRetrieval,
)

# Use the k that the teammate's k-sweep picked; fall back to 200 if missing.
RESULTS_JSON = os.path.join("output_part5", "results.json")
if os.path.exists(RESULTS_JSON):
    LSA_K = json.load(open(RESULTS_JSON))["lsa"]["best_k"]
else:
    LSA_K = 200


def main():
    if len(sys.argv) < 2:
        print("usage: python failure_compare.py <qid> [<qid> ...]")
        sys.exit(1)
    target = [int(x) for x in sys.argv[1:]]

    doc_ids, doc_texts, query_ids, query_texts, qrels = load_cranfield()
    docs_proc, _              = preprocess(doc_texts)
    queries_proc, queries_raw = preprocess(query_texts)

    # Build each index once.
    print(f"Building indices (LSA k={LSA_K}) ...")
    vsm  = InformationRetrieval();           vsm.buildIndex(docs_proc, doc_ids)
    bm25 = BM25Retrieval();                  bm25.buildIndex(docs_proc, doc_ids)
    lsa  = LSARetrieval(n_components=LSA_K); lsa.buildIndex(docs_proc, doc_ids)
    qe   = QueryExpansionRetrieval();        qe.buildIndex(docs_proc, doc_ids)

    ev = Evaluation()
    qid_to_idx = {int(q): i for i, q in enumerate(query_ids)}

    for qid in target:
        if qid not in qid_to_idx:
            print(f"\nQ{qid}: not found in cran_queries.json")
            continue
        i        = qid_to_idx[qid]
        true_set = ev._get_true_doc_IDs(qid, qrels)
        q_proc   = queries_proc[i]
        q_raw    = queries_raw[i]
        tokens   = [t for s in q_proc for t in s]
        oov      = [t for t in tokens if t not in vsm.idf]

        ranks = {
            "VSM":  vsm.rank([q_proc])[0],
            "BM25": bm25.rank([q_proc])[0],
            "LSA":  lsa.rank([q_proc])[0],
            "QE":   qe.rank([q_proc], [q_raw])[0],
        }

        print(f"\n=== Q{qid} ===")
        print(f"  text       : {query_texts[i].strip()}")
        print(f"  tokens     : {tokens}")
        print(f"  oov        : {oov}")
        print(f"  n_relevant : {len(true_set)}    relevant: {sorted(true_set)}")
        for name, ranked in ranks.items():
            top10 = ranked[:10]
            marks = " ".join(f"{d}{'*' if int(d) in true_set else ''}" for d in top10)
            p10   = ev.queryPrecision(ranked, qid, true_set, 10)
            ap10  = ev.queryAveragePrecision(ranked, qid, true_set, 10)
            rank1 = next((r for r, d in enumerate(ranked, 1) if int(d) in true_set), None)
            print(f"  {name:>4} P@10={p10:.2f} AP@10={ap10:.2f} rank1stRel={rank1}")
            print(f"         top10: {marks}")


if __name__ == "__main__":
    main()