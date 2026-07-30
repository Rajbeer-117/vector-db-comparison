"""
Main benchmark orchestrator.

For each configured backend and each dataset size in config.DATASET_SIZES,
this script:
  1. creates a fresh schema/collection,
  2. times bulk insertion,
  3. times ANN index construction,
  4. runs each hybrid query workload config.QUERY_REPEATS times and records
     latency percentiles,
  5. computes recall@k against a brute-force ground truth computed once
     in-memory over the same embeddings,
  6. writes all results to results/benchmark_results.csv.

Usage:
    python run_benchmark.py --data ../data/cve_embeddings.parquet --backends postgres milvus weaviate
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.append("..")
import config  # noqa: E402
from benchmark.metrics import brute_force_top_k, latency_stats, recall_at_k  # noqa: E402
from benchmark.queries import build_query_workloads, to_hybrid_query  # noqa: E402
from db_backends.base import as_list  # noqa: E402
from db_backends.postgres_backend import PostgresBackend  # noqa: E402
from db_backends.milvus_backend import MilvusBackend  # noqa: E402
from db_backends.weaviate_backend import WeaviateBackend  # noqa: E402

BACKEND_REGISTRY = {
    "postgres": PostgresBackend,
    "milvus": MilvusBackend,
    "weaviate": WeaviateBackend,
}


def _apply_structured_filters(df: pd.DataFrame, workload: dict) -> pd.DataFrame:
    """Mirrors each backend's structured filters in pandas so recall@k has a
    ground truth that respects the same constraints, not just raw
    similarity."""
    f = workload["filters"]
    mask = pd.Series(True, index=df.index)
    if "severity_in" in f:
        mask &= df["severity"].isin(f["severity_in"])
    if "min_cvss" in f:
        mask &= df["cvss_score"] >= f["min_cvss"]
    if "max_cvss" in f:
        mask &= df["cvss_score"] <= f["max_cvss"]
    if "published_after" in f:
        mask &= df["published_date"] >= f["published_after"]
    if "published_before" in f:
        mask &= df["published_date"] <= f["published_before"]
    if "cwe_in" in f:
        wanted = set(f["cwe_in"])
        mask &= df["cwe_ids"].apply(lambda ids: bool(wanted.intersection(as_list(ids))))
    return df[mask]


def run_for_backend(backend_name: str, df: pd.DataFrame, embeddings: np.ndarray) -> "list[dict]":
    backend_cls = BACKEND_REGISTRY[backend_name]
    results = []
    workloads = build_query_workloads()
    records = df.to_dict("records")
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb.tolist()

    with backend_cls() as backend:
        backend.create_schema(dim=embeddings.shape[1], drop_existing=True)

        # --- Indexing phase -------------------------------------------------
        insert_start = time.perf_counter()
        for i in range(0, len(records), config.BATCH_SIZE):
            backend.insert_batch(records[i:i + config.BATCH_SIZE])
        insert_time_s = time.perf_counter() - insert_start
        index_build_time_s = backend.build_index()

        results.append({
            "backend": backend_name,
            "phase": "indexing",
            "n_records": len(records),
            "insert_time_s": insert_time_s,
            "index_build_time_s": index_build_time_s,
            "total_setup_time_s": insert_time_s + index_build_time_s,
        })

        # --- Query phase ------------------------------------------------
        for workload in workloads:
            gt_pool = _apply_structured_filters(df, workload)
            gt_pool_idx = gt_pool.index.to_numpy()
            gt_embeddings = embeddings[gt_pool_idx]
            gt_ids = gt_pool["id"].tolist()

            # use the first record's embedding as a stand-in query vector so
            # runs are deterministic and reproducible without calling the
            # embedding model again at query time
            query_embedding = embeddings[0]
            ground_truth = (
                brute_force_top_k(query_embedding, gt_embeddings, gt_ids, config.TOP_K)
                if len(gt_ids) > 0 else []
            )

            hq = to_hybrid_query(workload, query_embedding.tolist(), config.TOP_K)

            latencies, recalls = [], []
            for _ in range(config.QUERY_REPEATS):
                qr = backend.hybrid_query(hq)
                latencies.append(qr.latency_ms)
                recalls.append(recall_at_k(qr.ids, ground_truth))

            stats = latency_stats(latencies)
            results.append({
                "backend": backend_name,
                "phase": "query",
                "workload": workload["name"],
                "n_records": len(records),
                "recall_at_k_mean": float(np.nanmean(recalls)),
                **stats,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.EMBEDDED_DATA_PATH)
    parser.add_argument("--backends", nargs="+", default=list(BACKEND_REGISTRY.keys()),
                         choices=list(BACKEND_REGISTRY.keys()))
    parser.add_argument("--sizes", nargs="+", type=int, default=config.DATASET_SIZES)
    parser.add_argument("--out", default=f"{config.RESULTS_DIR}/benchmark_results.csv")
    args = parser.parse_args()

    df_full = pd.read_parquet(args.data)
    embeddings_full = np.stack(df_full["embedding"].to_numpy())

    all_results = []
    for size in args.sizes:
        n = min(size, len(df_full))
        df = df_full.iloc[:n].reset_index(drop=True)
        embeddings = embeddings_full[:n]
        print(f"\n=== Dataset size: {n} ===")
        for backend_name in args.backends:
            print(f"--- Backend: {backend_name} ---")
            try:
                all_results.extend(run_for_backend(backend_name, df, embeddings))
            except Exception as e:  # noqa: BLE001
                print(f"[run_benchmark.py] {backend_name} failed at size {n}: {e!r}")
                all_results.append({
                    "backend": backend_name, "phase": "error",
                    "n_records": n, "error": str(e),
                })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df)} result rows -> {args.out}")


if __name__ == "__main__":
    main()
