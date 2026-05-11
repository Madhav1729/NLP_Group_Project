"""
Part 4 analysis script. Runs the same VSM pipeline as main.py over the
Cranfield collection and writes aggregate metrics, per-query diagnostics,
an OOV summary, failure cases, and document statistics to part4_output/.

Usage:
    python part4_analysis.py
    python part4_analysis.py --stem inversion
    python part4_analysis.py --df flow
    python part4_analysis.py --query 22
"""

import argparse
import csv
import json
import math
import os
import time
from collections import Counter

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation


def build_pipeline():
    seg = SentenceSegmentation()
    tok = Tokenization()
    inf = InflectionReduction()
    sw = StopwordRemoval()

    def run(text):
        return sw.fromList(inf.reduce(tok.pennTreeBank(seg.punkt(text))))

    return run


def cache_is_valid(path, expected_n):
    if not os.path.exists(path):
        return False
    try:
        data = json.load(open(path))
    except Exception:
        return False
    return (isinstance(data, list)
            and len(data) == expected_n
            and all(x is not None for x in data))


def preprocess_corpus(dataset_dir, cache_dir, force=False):
    with open(os.path.join(dataset_dir, "cran_docs.json")) as f:
        docs_json = json.load(f)
    with open(os.path.join(dataset_dir, "cran_queries.json")) as f:
        queries_json = json.load(f)

    doc_ids = [int(d["id"]) for d in docs_json]
    query_ids = [int(q["query number"]) for q in queries_json]
    raw_queries = [q["query"] for q in queries_json]

    docs_cache = os.path.join(cache_dir, "stopword_removed_docs.txt")
    qs_cache = os.path.join(cache_dir, "stopword_removed_queries.txt")

    if (not force
            and cache_is_valid(docs_cache, len(docs_json))
            and cache_is_valid(qs_cache, len(queries_json))):
        docs_proc = json.load(open(docs_cache))
        queries_proc = json.load(open(qs_cache))
        print(f"[preprocess] using cached {cache_dir}/")
    else:
        os.makedirs(cache_dir, exist_ok=True)
        run = build_pipeline()
        t0 = time.time()
        docs_proc = [run(d["body"]) for d in docs_json]
        queries_proc = [run(q["query"]) for q in queries_json]
        print(f"[preprocess] {time.time() - t0:.1f}s")
        json.dump(docs_proc, open(docs_cache, "w"))
        json.dump(queries_proc, open(qs_cache, "w"))

    return docs_proc, doc_ids, queries_proc, query_ids, raw_queries


def build_and_rank(docs_proc, doc_ids, queries_proc):
    ir = InformationRetrieval()
    t0 = time.time()
    ir.buildIndex(docs_proc, doc_ids)
    t_index = time.time() - t0
    t0 = time.time()
    ranked = ir.rank(queries_proc)
    t_rank = time.time() - t0
    print(f"[index] {t_index:.2f}s, |V| = {len(ir.idf)}")
    print(f"[rank]  {len(ranked)} queries in {t_rank:.2f}s")
    return ir, ranked, t_index, t_rank


def aggregate_metrics(ranked, query_ids, qrels, ks=range(1, 11)):
    e = Evaluation()
    rows = []
    for k in ks:
        rows.append({
            "k": k,
            "P@k":    e.meanPrecision(ranked, query_ids, qrels, k),
            "R@k":    e.meanRecall(ranked, query_ids, qrels, k),
            "F0.5@k": e.meanFscore(ranked, query_ids, qrels, k),
            "AP@k":   e.meanAveragePrecision(ranked, query_ids, qrels, k),
            "nDCG@k": e.meanNDCG(ranked, query_ids, qrels, k),
        })
    full = len(ranked[0])
    return rows, e.meanAveragePrecision(ranked, query_ids, qrels, full), \
           e.meanReciprocalRank(ranked, query_ids, qrels, full)


# ir.rank() drops the cosine scores, so we recompute them for the top-k
# we want to report.
def cosine_scores(ir, query_proc, doc_id_subset):
    terms = [t for sent in query_proc for t in sent]
    q_tf = Counter(terms)
    q_vec, q_norm_sq = {}, 0.0
    for t, c in q_tf.items():
        if t in ir.idf:
            v = c * ir.idf[t]
            q_vec[t] = v
            q_norm_sq += v * v
    q_norm = math.sqrt(q_norm_sq) or 1.0

    id_to_idx = {d: i for i, d in enumerate(ir.doc_ids)}
    out = []
    for did in doc_id_subset:
        idx = id_to_idx[int(did)]
        dvec = ir.doc_vectors[idx]
        dnorm = ir.doc_norms[idx] or 1.0
        dot = sum(q_vec[t] * dvec.get(t, 0.0) for t in q_vec)
        out.append(round(dot / (q_norm * dnorm), 4))
    return out


def per_query_diagnostics(ir, ranked, queries_proc, raw_queries, query_ids, qrels):
    e = Evaluation()
    rows = []
    for i, qid in enumerate(query_ids):
        terms = [t for sent in queries_proc[i] for t in sent]
        oov = [t for t in terms if t not in ir.idf]
        true_set = e._get_true_doc_IDs(qid, qrels)

        rank_first = None
        for r, did in enumerate(ranked[i], start=1):
            if int(did) in true_set:
                rank_first = r
                break

        top10 = ranked[i][:10]
        rows.append({
            "qid": qid,
            "raw_query": raw_queries[i],
            "tokens": terms,
            "oov": oov,
            "n_relevant": len(true_set),
            "P@10": e.queryPrecision(ranked[i], qid, true_set, 10),
            "R@10": e.queryRecall(ranked[i], qid, true_set, 10),
            "rank_first_relevant": rank_first,
            "top10_docs": top10,
            "top10_scores": cosine_scores(ir, queries_proc[i], top10),
            "top10_relevant_flag": [int(int(d) in true_set) for d in top10],
        })
    return rows


def oov_summary(diagnostics):
    counter = Counter()
    for r in diagnostics:
        counter.update(r["oov"])
    n_with_oov = sum(1 for r in diagnostics if r["oov"])
    return n_with_oov, len(diagnostics), counter


def failure_cases(diagnostics, top_n=15):
    by_p = sorted(diagnostics,
                  key=lambda r: (r["P@10"], -r["n_relevant"]))[:top_n]
    by_rank = sorted(
        diagnostics,
        key=lambda r: (r["rank_first_relevant"] is None,
                       -(r["rank_first_relevant"] or 0)))[:top_n]
    return by_p, by_rank


def doc_stats(docs_proc, doc_ids):
    pairs = [(doc_ids[i], sum(len(s) for s in d))
             for i, d in enumerate(docs_proc)]
    lens = sorted(L for _, L in pairs)
    return {
        "n_docs": len(pairs),
        "min_len": lens[0],
        "max_len": lens[-1],
        "mean_len": sum(lens) / len(lens),
        "median_len": lens[len(lens) // 2],
        "empty_docs": [did for did, L in pairs if L == 0],
    }


def write_outputs(out_dir, rows, diagnostics, oov_info, failures, ds):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "aggregate_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    with open(os.path.join(out_dir, "per_query_diagnostics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "raw_query", "tokens", "oov", "n_relevant",
                    "P@10", "R@10", "rank_first_relevant",
                    "top10_docs", "top10_scores", "top10_relevant_flag"])
        for r in diagnostics:
            w.writerow([
                r["qid"], r["raw_query"],
                " ".join(r["tokens"]),
                " ".join(r["oov"]),
                r["n_relevant"],
                f"{r['P@10']:.3f}", f"{r['R@10']:.3f}",
                r["rank_first_relevant"],
                ";".join(map(str, r["top10_docs"])),
                ";".join(map(str, r["top10_scores"])),
                ";".join(map(str, r["top10_relevant_flag"])),
            ])

    n_with_oov, n_total, oov_terms = oov_info
    with open(os.path.join(out_dir, "oov_summary.txt"), "w") as f:
        f.write(f"Queries with >=1 OOV token after preprocessing: "
                f"{n_with_oov}/{n_total} ({n_with_oov/n_total:.1%})\n\n")
        f.write("Most frequent OOV tokens (across queries):\n")
        for t, c in oov_terms.most_common(50):
            f.write(f"  {t:<20} {c}\n")

    by_p, by_rank = failures
    with open(os.path.join(out_dir, "failure_cases.txt"), "w") as f:
        f.write("=== Worst by P@10 (ties broken by larger n_relevant) ===\n")
        for r in by_p:
            f.write(f"qid={r['qid']:<4} P@10={r['P@10']:.3f} "
                    f"R@10={r['R@10']:.3f} "
                    f"rank1stRel={r['rank_first_relevant']} "
                    f"n_relevant={r['n_relevant']} oov={r['oov']}\n"
                    f"    {r['raw_query']}\n")
        f.write("\n=== Worst by rank-of-first-relevant ===\n")
        for r in by_rank:
            f.write(f"qid={r['qid']:<4} "
                    f"rank1stRel={r['rank_first_relevant']} "
                    f"P@10={r['P@10']:.3f} oov={r['oov']}\n"
                    f"    {r['raw_query']}\n")

    with open(os.path.join(out_dir, "doc_stats.txt"), "w") as f:
        for k, v in ds.items():
            f.write(f"{k}: {v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cranfield")
    ap.add_argument("--out", default="part4_output")
    ap.add_argument("--cache", default="output")
    ap.add_argument("--force_preproc", action="store_true")
    ap.add_argument("--stem")
    ap.add_argument("--df")
    ap.add_argument("--query", type=int)
    args = ap.parse_args()

    if args.stem and not (args.df or args.query):
        from nltk.stem import PorterStemmer
        print(f"stem({args.stem!r}) = {PorterStemmer().stem(args.stem)!r}")
        return

    docs_proc, doc_ids, queries_proc, query_ids, raw_queries = preprocess_corpus(
        args.dataset, args.cache, force=args.force_preproc
    )

    ir, ranked, t_index, t_rank = build_and_rank(docs_proc, doc_ids, queries_proc)

    if args.df:
        from nltk.stem import PorterStemmer
        stem = PorterStemmer().stem(args.df)
        df = sum(1 for v in ir.doc_vectors if stem in v)
        N = len(doc_ids)
        print(f"{args.df!r} -> stem={stem!r}  DF={df}/{N} ({df/N:.2%})")
        return

    with open(os.path.join(args.dataset, "cran_qrels.json")) as f:
        qrels = json.load(f)

    diagnostics = per_query_diagnostics(
        ir, ranked, queries_proc, raw_queries, query_ids, qrels
    )

    if args.query:
        try:
            r = next(x for x in diagnostics if x["qid"] == args.query)
        except StopIteration:
            raise SystemExit(f"query id {args.query} not found")
        for k, v in r.items():
            print(f"{k:>22} : {v}")
        return

    rows, map_full, mrr_full = aggregate_metrics(ranked, query_ids, qrels)
    write_outputs(args.out,
                  rows, diagnostics,
                  oov_summary(diagnostics),
                  failure_cases(diagnostics),
                  doc_stats(docs_proc, doc_ids))

    print()
    print(f"  MAP        = {map_full:.4f}")
    print(f"  MRR        = {mrr_full:.4f}")
    print(f"  P@10       = {rows[9]['P@k']:.4f}")
    print(f"  R@10       = {rows[9]['R@k']:.4f}")
    print(f"  nDCG@10    = {rows[9]['nDCG@k']:.4f}")
    print(f"  index/rank = {t_index:.2f}s / {t_rank:.2f}s")
    print(f"  outputs    -> {args.out}/")


if __name__ == "__main__":
    main()
