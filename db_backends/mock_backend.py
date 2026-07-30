"""
Pure-Python, in-memory backend implementing the same VectorDBBackend
interface as the three real systems. This is NOT one of the systems under
comparison -- it exists purely so the ingestion pipeline, hybrid-query
filter logic, and benchmark harness can be exercised end-to-end with unit
tests / in this repo's CI, without requiring live PostgreSQL, Milvus, or
Weaviate instances.

Run `python benchmark/smoke_test.py` to see it in action.
"""

import time

import numpy as np

from db_backends.base import HybridQuery, QueryResult, VectorDBBackend


class MockBackend(VectorDBBackend):
    name = "mock"

    def __init__(self):
        self._rows = []
        self._dim = None

    def connect(self) -> None:
        pass

    def create_schema(self, dim: int, drop_existing: bool = True) -> None:
        self._dim = dim
        self._rows = []

    def insert_batch(self, records: "list[dict]") -> None:
        self._rows.extend(records)

    def build_index(self) -> float:
        start = time.perf_counter()
        self._emb_matrix = np.array([r["embedding"] for r in self._rows], dtype=np.float32)
        return time.perf_counter() - start

    def hybrid_query(self, query: HybridQuery) -> QueryResult:
        start = time.perf_counter()

        candidates = self._rows
        if query.severity_in:
            candidates = [r for r in candidates if r.get("severity") in query.severity_in]
        if query.min_cvss is not None:
            candidates = [r for r in candidates if (r.get("cvss_score") or -1) >= query.min_cvss]
        if query.max_cvss is not None:
            candidates = [r for r in candidates if (r.get("cvss_score") or -1) <= query.max_cvss]
        if query.published_after:
            candidates = [r for r in candidates
                          if (r.get("published_date") or "") >= query.published_after]
        if query.published_before:
            candidates = [r for r in candidates
                          if (r.get("published_date") or "9999") <= query.published_before]
        if query.cwe_in:
            wanted = set(query.cwe_in)
            candidates = [r for r in candidates
                          if wanted.intersection(r.get("cwe_ids") or [])]

        if not candidates:
            return QueryResult(ids=[], scores=[], latency_ms=(time.perf_counter() - start) * 1000)

        q = np.asarray(query.query_embedding, dtype=np.float32)
        embs = np.array([r["embedding"] for r in candidates], dtype=np.float32)
        sims = embs @ q
        top_idx = np.argsort(-sims)[:query.top_k]

        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(
            ids=[candidates[i]["id"] for i in top_idx],
            scores=[float(sims[i]) for i in top_idx],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        pass
