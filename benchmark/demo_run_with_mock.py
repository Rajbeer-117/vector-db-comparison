"""
DEMO ONLY -- produces a results CSV in the exact schema run_benchmark.py
outputs, but using MockBackend (in-memory numpy) three times under
different "backend" labels, with small artificial latency offsets added so
the charts render three distinguishable series.

This exists purely so analysis/plot_results.py has something real to plot
in environments without live PostgreSQL/Milvus/Weaviate servers (e.g. this
sandbox). It is NOT a substitute for real benchmark numbers -- swap this
out for the real run_benchmark.py output before drawing any conclusions.

Usage:
    python demo_run_with_mock.py
"""

import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.append("..")
import config  # noqa: E402
from benchmark.metrics import brute_force_top_k, latency_stats, recall_at_k  # noqa: E402
from benchmark.queries import build_query_workloads, to_hybrid_query  # noqa: E402
from db_backends.mock_backend import MockBackend  # noqa: E402

random.seed(7)
np.random.seed(7)

# artificial per-backend latency multipliers/offsets, purely to make the
# demo charts show three distinguishable (fake) series -- NOT measured
# characteristics of the real systems.
DEMO_BACKEND_PROFILES = {
    "postgres": {"query_ms_offset": 0.15, "query_ms_scale": 1.35, "setup_scale": 1.6},
    "milvus": {"query_ms_offset": 0.05, "query_ms_scale": 0.9, "setup_scale": 1.0},
    "weaviate": {"query_ms_offset": 0.08, "query_ms_scale": 1.05, "setup_scale": 0.75},
}


def run_for_demo_backend(backend_label: str, df: pd.DataFrame, embeddings: np.ndarray) -> "list[dict]":
    profile = DEMO_BACKEND_PROFILES[backend_label]
    results = []
    workloads = build_query_workloads()
    records = df.to_dict("records")
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb.tolist()

    with MockBackend() as backend:
        backend.create_schema(dim=embeddings.shape[1])

        insert_start = time.perf_counter()
        for i in range(0, len(records), config.BATCH_SIZE):
            backend.insert_batch(records[i:i + config.BATCH_SIZE])
        insert_time_s = (time.perf_counter() - insert_start) * profile["setup_scale"]
        index_build_time_s = backend.build_index() * profile["setup_scale"]

        results.append({
            "backend": backend_label, "phase": "indexing", "n_records": len(records),
            "insert_time_s": insert_time_s, "index_build_time_s": index_build_time_s,
            "total_setup_time_s": insert_time_s + index_build_time_s,
        })

        for workload in workloads:
            gt_pool_idx = df.index.to_numpy()
            query_embedding = embeddings[0]
            ground_truth = brute_force_top_k(
                query_embedding, embeddings[gt_pool_idx], df["id"].tolist(), config.TOP_K
            )
            hq = to_hybrid_query(workload, query_embedding.tolist(), config.TOP_K)

            latencies, recalls = [], []
            for _ in range(config.QUERY_REPEATS):
                qr = backend.hybrid_query(hq)
                fake_latency = (
                    qr.latency_ms * profile["query_ms_scale"]
                    + profile["query_ms_offset"]
                    + abs(np.random.normal(0, 0.05))
                )
                latencies.append(fake_latency)
                recalls.append(recall_at_k(qr.ids, ground_truth))

            stats = latency_stats(latencies)
            results.append({
                "backend": backend_label, "phase": "query", "workload": workload["name"],
                "n_records": len(records), "recall_at_k_mean": float(np.nanmean(recalls)),
                **stats,
            })

    return results


def main():
    df_full = pd.read_parquet(config.EMBEDDED_DATA_PATH)
    embeddings_full = np.stack(df_full["embedding"].to_numpy())

    all_results = []
    for size in [100, 250, 500]:
        n = min(size, len(df_full))
        df = df_full.iloc[:n].reset_index(drop=True)
        embeddings = embeddings_full[:n]
        for backend_label in DEMO_BACKEND_PROFILES:
            all_results.extend(run_for_demo_backend(backend_label, df, embeddings))

    out_df = pd.DataFrame(all_results)
    out_path = f"{config.RESULTS_DIR}/benchmark_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[DEMO DATA -- not real backend measurements] Saved {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
