# 8. Single-term lookups (CLI helpers)
# =========================================================================
# These let you answer questions like:
#   "what is the stem of `inversion`?"          --stem inversion
#   "how many docs contain the stem of `flow`?"  --df flow
# without re-running the entire eval.
# -------------------------------------------------------------------------

def lookup_df(ir, term, stemmer):
    stem = stemmer.stem(term)
    df = sum(1 for v in ir.doc_vectors if stem in v)
    return stem, df, df / len(ir.doc_vectors)


# =========================================================================
# main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cranfield")
    ap.add_argument("--out", default="part4_output")
    ap.add_argument("--cache", default="output")
    ap.add_argument("--force_preproc", action="store_true")
    ap.add_argument("--stem", help="show Porter stem of WORD and exit")
    ap.add_argument("--df", help="show document frequency of stem(WORD) and exit")
    ap.add_argument("--query", type=int, help="dump diagnostics for one query id")
    args = ap.parse_args()

    # --stem is a freebie: it doesn't need the index.
    if args.stem and not (args.df or args.query):
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        print(f"stem({args.stem!r}) = {ps.stem(args.stem)!r}")
        return

    docs_proc, doc_ids, queries_proc, query_ids, raw_queries = preprocess_corpus(
        args.dataset, args.cache, force=args.force_preproc
    )

    ir, ranked, t_index, t_rank = build_and_rank(docs_proc, doc_ids, queries_proc)

    # --df: show DF for one term, then exit.
    if args.df:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        stem, df, frac = lookup_df(ir, args.df, ps)
        print(f"{args.df!r}  ->  stem={stem!r}  ->  DF={df}/{len(doc_ids)}  ({frac:.2%})")
        return

    with open(os.path.join(args.dataset, "cran_qrels.json")) as f:
        qrels = json.load(f)

    diagnostics = per_query_diagnostics(
        ir, ranked, queries_proc, raw_queries, query_ids, qrels
    )

    # --query: dump full diagnostics for a single query, then exit.
    if args.query:
        try:
            r = next(x for x in diagnostics if x["qid"] == args.query)
        except StopIteration:
            raise SystemExit(f"query id {args.query} not found")
        for k, v in r.items():
            print(f"{k:>22} : {v}")
        return

    # ---- otherwise: full analysis ----
    os.makedirs(args.out, exist_ok=True)

    rows, map_full, mrr_full = aggregate_metrics(ranked, query_ids, qrels)
    n_with_oov, n_total, oov_terms = oov_summary(diagnostics)
    by_p, by_rank = failure_cases(diagnostics)
    ds = doc_stats(docs_proc, doc_ids)

    # --- aggregate metrics CSV ---
    with open(os.path.join(args.out, "aggregate_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # --- per-query diagnostics CSV ---
    with open(os.path.join(args.out, "per_query_diagnostics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "qid", "raw_query", "tokens", "oov", "n_relevant",
            "P@10", "R@10", "rank_first_relevant",
            "top10_docs", "top10_scores", "top10_relevant_flag",
        ])
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

    # --- OOV summary ---
    with open(os.path.join(args.out, "oov_summary.txt"), "w") as f:
        pct = n_with_oov / n_total
        f.write(f"Queries with >=1 OOV token after preprocessing: "
                f"{n_with_oov}/{n_total} ({pct:.1%})\n\n")
        f.write("Most frequent OOV tokens (across queries):\n")
        for t, c in oov_terms.most_common(50):
            f.write(f"  {t:<20} {c}\n")

    # --- failure cases ---
    with open(os.path.join(args.out, "failure_cases.txt"), "w") as f:
        f.write("=== Worst by P@10 (ties broken by larger n_relevant) ===\n")
        for r in by_p:
            f.write(
                f"qid={r['qid']:<4} P@10={r['P@10']:.3f} R@10={r['R@10']:.3f} "
                f"rank1stRel={r['rank_first_relevant']} "
                f"n_relevant={r['n_relevant']} oov={r['oov']}\n"
                f"    {r['raw_query']}\n"
            )
        f.write("\n=== Worst by rank-of-first-relevant ===\n")
        for r in by_rank:
            f.write(
                f"qid={r['qid']:<4} rank1stRel={r['rank_first_relevant']} "
                f"P@10={r['P@10']:.3f} oov={r['oov']}\n"
                f"    {r['raw_query']}\n"
            )

    # --- doc stats ---
    with open(os.path.join(args.out, "doc_stats.txt"), "w") as f:
        for k, v in ds.items():
            f.write(f"{k}: {v}\n")

    # --- console summary (the headline numbers) ---
    print()
    print(f"  MAP        = {map_full:.4f}")
    print(f"  MRR        = {mrr_full:.4f}")
    print(f"  P@10       = {rows[9]['P@k']:.4f}")
    print(f"  R@10       = {rows[9]['R@k']:.4f}")
    print(f"  nDCG@10    = {rows[9]['nDCG@k']:.4f}")
    print(f"  index_time = {t_index:.2f}s    rank_time = {t_rank:.2f}s")
    print(f"  OOV queries: {n_with_oov}/{n_total} ({n_with_oov/n_total:.1%})")
    print(f"  empty docs : {ds['empty_docs']}")
    print(f"  outputs    -> {args.out}/")


if __name__ == "__main__":
    main()