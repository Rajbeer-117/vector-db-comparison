"""
Weaviate backend (v4 client).

Hybrid query strategy: Weaviate's `near_vector` search accepts a `filters`
object (its own structured-filter DSL, roughly analogous to a WHERE clause)
that is applied by the underlying HNSW-aware filtered search, so structured
and semantic conditions are again evaluated in a single request rather than
a two-phase filter-then-rerank.
"""

import sys
import time

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, MetadataQuery

sys.path.append("..")
import config  # noqa: E402
from db_backends.base import HybridQuery, QueryResult, VectorDBBackend, as_list  # noqa: E402


class WeaviateBackend(VectorDBBackend):
    name = "weaviate"

    def __init__(self, cfg: dict = None, class_name: str = None):
        self.cfg = cfg or config.WEAVIATE_CONFIG
        self.class_name = class_name or config.WEAVIATE_CLASS
        self.client: "weaviate.WeaviateClient | None" = None

    def connect(self) -> None:
        self.client = weaviate.connect_to_local(
            host=self.cfg["host"],
            port=self.cfg["port"],
            grpc_port=self.cfg["grpc_port"],
        )

    def create_schema(self, dim: int, drop_existing: bool = True) -> None:
        if drop_existing and self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)

        self.client.collections.create(
            name=self.class_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
                ef_construction=200,
                max_connections=16,
            ),
            properties=[
                wvc.config.Property(name="cve_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="severity", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="cvss_score", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="published_date", data_type=wvc.config.DataType.DATE),
                wvc.config.Property(name="cwe_ids", data_type=wvc.config.DataType.TEXT_ARRAY),
                wvc.config.Property(name="vendor_products", data_type=wvc.config.DataType.TEXT_ARRAY),
            ],
        )

    def _collection(self):
        return self.client.collections.get(self.class_name)

    @staticmethod
    def _to_rfc3339(date_str: "str | None") -> "str | None":
        return f"{date_str}T00:00:00Z" if date_str else None

    def insert_batch(self, records: "list[dict]") -> None:
        col = self._collection()
        with col.batch.dynamic() as batch:
            for r in records:
                batch.add_object(
                    properties={
                        "cve_id": r["id"],
                        "description": r["description"],
                        "severity": r.get("severity") or "UNKNOWN",
                        "cvss_score": float(r.get("cvss_score", -1.0)),
                        "published_date": self._to_rfc3339(r.get("published_date")),
                        "cwe_ids": as_list(r.get("cwe_ids")),
                        "vendor_products": as_list(r.get("vendor_products")),
                    },
                    vector=r["embedding"],
                    uuid=weaviate.util.generate_uuid5(r["id"]),
                )

    def build_index(self) -> float:
        # Weaviate builds/maintains the HNSW index incrementally during
        # insertion (configured in create_schema), so there is no separate
        # "build" phase like Postgres/Milvus. We report near-zero here and
        # note this asymmetry explicitly when reporting indexing-time
        # comparisons.
        return 0.0

    def hybrid_query(self, query: HybridQuery) -> QueryResult:
        col = self._collection()

        filters = []
        if query.severity_in:
            filters.append(Filter.by_property("severity").contains_any(query.severity_in))
        if query.min_cvss is not None:
            filters.append(Filter.by_property("cvss_score").greater_or_equal(query.min_cvss))
        if query.max_cvss is not None:
            filters.append(Filter.by_property("cvss_score").less_or_equal(query.max_cvss))
        if query.published_after:
            filters.append(
                Filter.by_property("published_date").greater_or_equal(
                    self._to_rfc3339(query.published_after)
                )
            )
        if query.published_before:
            filters.append(
                Filter.by_property("published_date").less_or_equal(
                    self._to_rfc3339(query.published_before)
                )
            )
        if query.cwe_in:
            filters.append(Filter.by_property("cwe_ids").contains_any(query.cwe_in))

        combined_filter = None
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f

        start = time.perf_counter()
        response = col.query.near_vector(
            near_vector=query.query_embedding,
            limit=query.top_k,
            filters=combined_filter,
            return_properties=["cve_id"],
            return_metadata=MetadataQuery(distance=True),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        ids = [o.properties["cve_id"] for o in response.objects]
        scores = [
            1 - o.metadata.distance if o.metadata.distance is not None else 0.0
            for o in response.objects
        ]
        return QueryResult(ids=ids, scores=scores, latency_ms=latency_ms)

    def count(self) -> int:
        col = self._collection()
        return col.aggregate.over_all(total_count=True).total_count

    def close(self) -> None:
        if self.client:
            self.client.close()
