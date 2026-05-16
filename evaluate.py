"""
evaluate.py

Benchmarks Normal RAG vs Temporal RAG on the MMM/marketing corpus.

Metrics computed per retrieval run:
    Precision@1  — fraction of queries where the top-ranked document is correct
    MRR          — Mean Reciprocal Rank (1/rank of first correct document)
    NDCG@3       — Normalised Discounted Cumulative Gain at cutoff 3

Both retrievers use the same TF-IDF vectoriser and corpus. The only
difference is the presence or absence of the temporal decay factor.

Usage:
    python evaluate.py --corpus corpus/ --benchmark benchmark.json --half_life 365
"""

import json
import math
import argparse
import numpy as np
from temporal_rag import TemporalRAG


def precision_at_k(retrieved_ids: list, relevant_id: str, k: int = 1) -> float:
    return 1.0 if retrieved_ids[:k] and retrieved_ids[0] == relevant_id else 0.0


def reciprocal_rank(retrieved_ids: list, relevant_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id == relevant_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_id: str, k: int = 3) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == relevant_id:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_dcg = 1.0 / math.log2(2)  # ideal: correct doc at rank 1
    return dcg / ideal_dcg


def run_evaluation(corpus_dir: str, benchmark_path: str, half_life: int):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    rag = TemporalRAG(half_life_days=half_life)
    n_docs = rag.load_corpus_from_directory(corpus_dir)

    queries = benchmark["queries"]
    n = len(queries)

    results = {
        "normal": {"p1": [], "mrr": [], "ndcg3": [], "failures": []},
        "temporal": {"p1": [], "mrr": [], "ndcg3": [], "failures": []}
    }

    col_w = 52
    print(f"\nCorpus: {n_docs} documents  |  Queries: {n}  |  Half-life: {half_life} days\n")
    print(f"{'Query':<6}  {'Normal RAG':<{col_w}}  {'Temporal RAG':<{col_w}}")
    print(f"{'':─<6}  {'':─<{col_w}}  {'':─<{col_w}}")

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        query_date = q["query_date"]
        ground_truth = q["ground_truth_doc"].replace(".txt", "")

        normal_hits = rag.retrieve(query_text, query_date, top_k=3, temporal=False)
        temporal_hits = rag.retrieve(query_text, query_date, top_k=3, temporal=True)

        normal_ids = [r.doc_id for r in normal_hits]
        temporal_ids = [r.doc_id for r in temporal_hits]

        for mode, ids, hits in [("normal", normal_ids, normal_hits), ("temporal", temporal_ids, temporal_hits)]:
            p1 = precision_at_k(ids, ground_truth, k=1)
            mrr = reciprocal_rank(ids, ground_truth)
            ndcg = ndcg_at_k(ids, ground_truth, k=3)
            results[mode]["p1"].append(p1)
            results[mode]["mrr"].append(mrr)
            results[mode]["ndcg3"].append(ndcg)
            if p1 == 0:
                results[mode]["failures"].append({
                    "query_id": qid,
                    "query": query_text[:60],
                    "ground_truth": ground_truth,
                    "retrieved_rank1": ids[0] if ids else "none"
                })

        def fmt(ids, hits, gt):
            rank1 = ids[0] if ids else "—"
            marker = "✓" if rank1 == gt else "✗"
            score = f"{hits[0].final_score:.3f}" if hits else "—"
            return f"{marker} [{hits[0].date}] {rank1[:28]} ({score})"

        print(f"{qid:<6}  {fmt(normal_ids, normal_hits, ground_truth):<{col_w}}  {fmt(temporal_ids, temporal_hits, ground_truth):<{col_w}}")

    print(f"\n{'─'*120}")
    print(f"\n{'Metric':<20}  {'Normal RAG':>14}  {'Temporal RAG':>14}  {'Delta':>10}")
    print(f"{'':─<20}  {'':─>14}  {'':─>14}  {'':─>10}")

    for label, key in [("Precision@1", "p1"), ("MRR", "mrr"), ("NDCG@3", "ndcg3")]:
        n_val = np.mean(results["normal"][key])
        t_val = np.mean(results["temporal"][key])
        delta = t_val - n_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{label:<20}  {n_val:>14.4f}  {t_val:>14.4f}  {delta_str:>10}")

    print(f"\nNormal RAG failures ({len(results['normal']['failures'])}/{n}):")
    for f in results["normal"]["failures"]:
        print(f"  {f['query_id']}  gt={f['ground_truth']:<35}  retrieved={f['retrieved_rank1']}")

    print(f"\nTemporal RAG failures ({len(results['temporal']['failures'])}/{n}):")
    for f in results["temporal"]["failures"]:
        print(f"  {f['query_id']}  gt={f['ground_truth']:<35}  retrieved={f['retrieved_rank1']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal RAG evaluation")
    parser.add_argument("--corpus", default="corpus/", help="Directory containing .txt corpus files")
    parser.add_argument("--benchmark", default="benchmark.json", help="Benchmark JSON file")
    parser.add_argument("--half_life", type=int, default=365, help="Temporal decay half-life in days")
    args = parser.parse_args()

    run_evaluation(args.corpus, args.benchmark, args.half_life)
