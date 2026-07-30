"""
End-to-end smoke test of the pipeline using MockBackend, so the hybrid
query filter logic, embedding pipeline, and metrics code can be validated
without requiring PostgreSQL/Milvus/Weaviate to be running. This is a
sanity check, not a substitute for the real benchmark.

Usage:
    python smoke_test.py
"""

import sys

import numpy as np
import pandas as pd

sys.path.append("..")
import config  # noqa: E402
from benchmark.metrics import brute_force_top_k, latency_stats, recall_at_k  # noqa: E402
from benchmark.queries import build_query_workloads, to_hybrid_query  # noqa: E402
from db_backends.mock_backend import MockBackend  # noqa: E402


def main():
    df = pd.read_parquet(config.EMBEDDED_DATA_PATH)
    embeddings = np.stack(df["embedding"].to_numpy())
    print(f"Loaded {len(df)} records, embedding dim = {embeddings.shape[1]}")

    records = df.to_dict("records")
    for rec, emb in zip(records, embeddings):
        rec["embedding"] = emb.tolist()

    with MockBackend() as backend:
        backend.create_schema(dim=embeddings.shape[1])
        backend.insert_batch(records)
        build_time = backend.build_index()
        print(f"Indexed {backend.count()} records in {build_time * 1000:.2f} ms")
        assert backend.count() == len(df)

        workloads = build_query_workloads()
        query_embedding = embeddings[0]

        print(f"\n{'workload':<22} {'n_results':>10} {'p50_ms':>10} {'recall@k':>10}")
        for workload in workloads:
            hq = to_hybrid_query(workload, query_embedding.tolist(), config.TOP_K)

            latencies, recalls = [], []
            for _ in range(20):
                qr = backend.hybrid_query(hq)
                latencies.append(qr.latency_ms)

            gt = brute_force_top_k(query_embedding, embeddings, df["id"].tolist(), config.TOP_K)
            recall = recall_at_k(qr.ids, gt)
            stats = latency_stats(latencies)

            print(f"{workload['name']:<22} {len(qr.ids):>10} {stats['p50_ms']:>10.3f} {recall:>10.2f}")
            assert len(qr.ids) <= config.TOP_K

    print("\nSmoke test passed: ingestion, hybrid filtering, and metrics all functioning.")


if __name__ == "__main__":
    main()
