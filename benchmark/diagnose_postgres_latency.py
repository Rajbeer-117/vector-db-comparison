"""
Diagnostic: isolates whether PostgresBackend's ~45ms measured query latency
is happening inside Postgres itself, or on the Python/psycopg2 client side
(parameter serialization, network round trip, result fetching).

It runs the *exact* hybrid_query() code path used by the real benchmark,
but wraps the SQL in EXPLAIN ANALYZE so Postgres reports its own server-side
timing for the same parameter-passing style (a 384-float Python list bound
as a query parameter) -- not the simplified subquery version used earlier
in manual psql testing.

It also times each Python-side step separately (connect reuse, execute,
fetchall) so we can see where the wall-clock time is actually going.

Usage:
    python diagnose_postgres_latency.py --data ../data/cve_embeddings.parquet
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector

sys.path.append("..")
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.EMBEDDED_DATA_PATH)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    embeddings = np.stack(df["embedding"].to_numpy())
    query_embedding = embeddings[0].tolist()  # exactly what run_benchmark.py sends

    conn = psycopg2.connect(**config.POSTGRES_CONFIG)
    conn.autocommit = True

    # Same TCP_NODELAY fix as PostgresBackend.connect() -- disable Nagle's
    # algorithm, which on Windows + Docker Desktop commonly causes a very
    # consistent ~40ms stall per round trip. socket.fromfd() is Unix-only
    # and fails silently on Windows; socket.socket(fileno=...) is the
    # portable equivalent, but must be detach()'d afterward so it doesn't
    # close the real connection when garbage-collected.
    try:
        import socket
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=conn.fileno())
        try:
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("[diagnose] TCP_NODELAY set successfully on connection socket.\n")
        finally:
            raw.detach()
    except (OSError, ValueError) as e:
        print(f"[diagnose] Could not set TCP_NODELAY: {e!r}\n")

    register_vector(conn)

    table = config.POSTGRES_TABLE

    # --- 1. Server-side EXPLAIN ANALYZE using the real bound parameter ----
    explain_sql = f"""
        EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
        SELECT id, 1 - (embedding <=> %s::vector) AS score
        FROM {table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(explain_sql, [query_embedding, query_embedding, config.TOP_K])
        plan_lines = cur.fetchall()
    print("=" * 70)
    print("SERVER-SIDE EXPLAIN ANALYZE (same parameter style as the benchmark)")
    print("=" * 70)
    for row in plan_lines:
        print(row[0])
    print()

    # --- 2. Python-side step timing (10 repeats) ---------------------------
    real_sql = f"""
        SELECT id, 1 - (embedding <=> %s::vector) AS score
        FROM {table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    # 2a. Trivial query (tiny payload, no embedding). If this is ALSO
    # ~45ms, the overhead is a fixed per-round-trip network cost (e.g.
    # Docker Desktop/WSL2 port-forwarding proxy latency) unrelated to the
    # embedding payload. If this is fast (~1ms) while the real query stays
    # slow, the overhead scales with payload size instead.
    trivial_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchall()
        trivial_times.append((time.perf_counter() - t0) * 1000)

    # 2b. The real query, embedding passed as a PLAIN PYTHON LIST -- this is
    # exactly what HybridQuery.query_embedding contains today (run_benchmark.py
    # calls .tolist() before building the query), so this reproduces the
    # benchmark's current behavior exactly.
    execute_times, fetch_times, total_times = [], [], []
    for _ in range(10):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            t1 = time.perf_counter()
            cur.execute(real_sql, [query_embedding, query_embedding, config.TOP_K])
            t2 = time.perf_counter()
            rows = cur.fetchall()
            t3 = time.perf_counter()
        execute_times.append((t2 - t1) * 1000)
        fetch_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t0) * 1000)

    # 2c. SAME query, but embedding passed as a NUMPY ARRAY instead of a
    # plain list. pgvector's register_vector() may only register its fast
    # serializer for numpy.ndarray, silently falling back to psycopg2's slow
    # generic per-element list adapter for plain Python lists. If this is
    # much faster than 2b, that confirms the theory and points to the fix.
    query_embedding_np = embeddings[0]  # numpy array, no .tolist()
    np_execute_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(real_sql, [query_embedding_np, query_embedding_np, config.TOP_K])
            rows = cur.fetchall()
        np_execute_times.append((time.perf_counter() - t0) * 1000)

    print("=" * 70)
    print("PYTHON-SIDE STEP TIMING (10 repeats, milliseconds)")
    print("=" * 70)
    print(f"{'step':<28}{'mean':>10}{'min':>10}{'max':>10}")
    trivial_arr = np.array(trivial_times)
    print(f"{'trivial SELECT 1 (total)':<28}{trivial_arr.mean():>10.3f}"
          f"{trivial_arr.min():>10.3f}{trivial_arr.max():>10.3f}")
    for label, arr in [("real query: execute()", execute_times),
                        ("real query: fetchall()", fetch_times),
                        ("real query: total (list param)", total_times)]:
        a = np.array(arr)
        print(f"{label:<28}{a.mean():>10.3f}{a.min():>10.3f}{a.max():>10.3f}")
    np_arr = np.array(np_execute_times)
    print(f"{'real query: total (ndarray param)':<28}{np_arr.mean():>10.3f}"
          f"{np_arr.min():>10.3f}{np_arr.max():>10.3f}")

    conn.close()


if __name__ == "__main__":
    main()
