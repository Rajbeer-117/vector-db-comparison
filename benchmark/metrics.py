"""
Metric helpers for the benchmark harness: latency percentiles, throughput,
and retrieval-quality metrics (recall@k against a brute-force ground truth,
Jaccard overlap between backends for sanity-checking hybrid filter
semantics agree across systems).
"""

import numpy as np


def latency_stats(latencies_ms: "list[float]") -> dict:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "throughput_qps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else float("inf"),
    }


def recall_at_k(retrieved_ids: "list[str]", ground_truth_ids: "list[str]") -> float:
    """Fraction of ground-truth top-k ids present in the retrieved set."""
    if not ground_truth_ids:
        return float("nan")
    gt = set(ground_truth_ids)
    hit = sum(1 for i in retrieved_ids if i in gt)
    return hit / len(gt)


def brute_force_top_k(
    query_embedding: "np.ndarray", corpus_embeddings: "np.ndarray", corpus_ids: "list[str]", k: int
) -> "list[str]":
    """Exact cosine-similarity nearest neighbors, used as ground truth for
    recall@k. Embeddings are assumed L2-normalized (both backends and
    embed.py normalize embeddings, so dot product == cosine similarity)."""
    sims = corpus_embeddings @ query_embedding
    top_idx = np.argsort(-sims)[:k]
    return [corpus_ids[i] for i in top_idx]
