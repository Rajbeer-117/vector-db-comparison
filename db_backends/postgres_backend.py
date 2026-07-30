"""
PostgreSQL + pgvector backend.

Hybrid query strategy: a single SQL statement combines standard WHERE
clauses (severity, cvss range, date range, CWE membership) with an
ORDER BY <-> (pgvector cosine-distance operator) LIMIT k clause, so
filtering and ANN search are pushed down into one planner-optimized query.

Requires PostgreSQL with the `pgvector` extension installed (see
docker-compose.yml, which uses the pgvector/pgvector:pg16 image).
"""

import sys
import time

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

sys.path.append("..")
import config  # noqa: E402
from db_backends.base import HybridQuery, QueryResult, VectorDBBackend, as_list  # noqa: E402


class PostgresBackend(VectorDBBackend):
    name = "postgres"

    def __init__(self, cfg: dict = None, table: str = None):
        self.cfg = cfg or config.POSTGRES_CONFIG
        self.table = table or config.POSTGRES_TABLE
        self.conn = None
        self._dim = config.EMBEDDING_DIM

    def connect(self) -> None:
        self.conn = psycopg2.connect(**self.cfg)
        self.conn.autocommit = True

        # Disable Nagle's algorithm on the connection socket. Without this,
        # some environments (notably Windows + Docker Desktop) exhibit a
        # very consistent ~40ms stall per query round trip, caused by
        # Nagle's algorithm on the client interacting with delayed ACKs on
        # the server. libpq/psycopg2 don't set TCP_NODELAY themselves, so
        # we set it explicitly on the underlying socket.
        #
        # NOTE: socket.fromfd() is Unix-only and silently fails on Windows.
        # socket.socket(..., fileno=fd) is the portable equivalent (Python
        # 3.3+, works on Windows too). It wraps the existing OS socket
        # without duplicating it, so we must .detach() afterward -- otherwise
        # this temporary wrapper gets garbage-collected and closes the
        # underlying connection out from under psycopg2.
        try:
            import socket
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=self.conn.fileno())
            try:
                raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            finally:
                raw.detach()
        except (OSError, ValueError):
            pass  # best-effort; harmless if unsupported on this platform

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(self.conn)

    def create_schema(self, dim: int, drop_existing: bool = True) -> None:
        self._dim = dim
        with self.conn.cursor() as cur:
            if drop_existing:
                cur.execute(f"DROP TABLE IF EXISTS {self.table};")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    embedding VECTOR({dim}) NOT NULL,
                    severity TEXT,
                    cvss_score REAL,
                    published_date DATE,
                    cwe_ids TEXT[],
                    vendor_products TEXT[]
                );
                """
            )
            # structured-filter indexes -- these are what let the planner
            # cheaply narrow rows before/alongside the vector distance scan
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_severity "
                f"ON {self.table} (severity);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_cvss "
                f"ON {self.table} (cvss_score);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_date "
                f"ON {self.table} (published_date);"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_cwe "
                f"ON {self.table} USING GIN (cwe_ids);"
            )

    def insert_batch(self, records: "list[dict]") -> None:
        rows = [
            (
                r["id"],
                r["description"],
                np.asarray(r["embedding"], dtype=np.float32),
                r.get("severity"),
                r.get("cvss_score"),
                r.get("published_date") or None,
                as_list(r.get("cwe_ids")),
                as_list(r.get("vendor_products")),
            )
            for r in records
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {self.table}
                    (id, description, embedding, severity, cvss_score,
                     published_date, cwe_ids, vendor_products)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s, %s)",
            )

    def build_index(self) -> float:
        """HNSW (or IVFFlat) ANN index over the embedding column. This is
        typically the most expensive setup step and is timed separately
        from raw insertion so the two costs can be compared across
        backends."""
        start = time.perf_counter()
        with self.conn.cursor() as cur:
            if config.POSTGRES_INDEX_TYPE == "hnsw":
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ann "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops);"
                )
            else:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ann "
                    f"ON {self.table} USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = 100);"
                )
            cur.execute(f"ANALYZE {self.table};")
        return time.perf_counter() - start

    def hybrid_query(self, query: HybridQuery) -> QueryResult:
        where_clauses, params = [], []

        if query.severity_in:
            where_clauses.append("severity = ANY(%s)")
            params.append(query.severity_in)
        if query.min_cvss is not None:
            where_clauses.append("cvss_score >= %s")
            params.append(query.min_cvss)
        if query.max_cvss is not None:
            where_clauses.append("cvss_score <= %s")
            params.append(query.max_cvss)
        if query.published_after:
            where_clauses.append("published_date >= %s")
            params.append(query.published_after)
        if query.published_before:
            where_clauses.append("published_date <= %s")
            params.append(query.published_before)
        if query.cwe_in:
            where_clauses.append("cwe_ids && %s")
            params.append(query.cwe_in)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT id, 1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        # Bind the embedding as a numpy array, not a plain Python list.
        # pgvector's register_vector() registers its fast serializer for
        # numpy.ndarray; a plain list falls back to psycopg2's slow generic
        # per-element adapter, which measured ~45ms/query overhead here for
        # a 384-dim vector used twice (see benchmark/diagnose_postgres_latency.py).
        qemb = np.asarray(query.query_embedding, dtype=np.float32)
        # embedding param needed twice: once for scoring, once for ORDER BY
        full_params = [qemb] + params + [qemb, query.top_k]

        start = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute(sql, full_params)
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000

        return QueryResult(
            ids=[r[0] for r in rows],
            scores=[float(r[1]) for r in rows],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table};")
            return cur.fetchone()[0]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
