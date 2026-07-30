"""
Milvus backend.

Hybrid query strategy: Milvus supports boolean scalar filter expressions
passed alongside an ANN search request, evaluated by the query node before/
during the vector scan (filtered search), rather than as a separate SQL
join. Filter syntax is a restricted boolean-expression language over
scalar fields.
"""

import sys
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

sys.path.append("..")
import config  # noqa: E402
from db_backends.base import HybridQuery, QueryResult, VectorDBBackend, as_list  # noqa: E402

MAX_DESC_LEN = 4096
MAX_ID_LEN = 64
MAX_CWE_LEN = 32
MAX_CWES = 10
MAX_VP_LEN = 128
MAX_VPS = 10


class MilvusBackend(VectorDBBackend):
    name = "milvus"

    def __init__(self, cfg: dict = None, collection_name: str = None):
        self.cfg = cfg or config.MILVUS_CONFIG
        self.collection_name = collection_name or config.MILVUS_COLLECTION
        self.collection: "Collection | None" = None
        self._alias = "vectorbench"

    def connect(self) -> None:
        connections.connect(
            alias=self._alias, host=self.cfg["host"], port=self.cfg["port"]
        )

    def create_schema(self, dim: int, drop_existing: bool = True) -> None:
        if drop_existing and utility.has_collection(
            self.collection_name, using=self._alias
        ):
            utility.drop_collection(self.collection_name, using=self._alias)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=MAX_ID_LEN,
                        is_primary=True),
            FieldSchema(name="description", dtype=DataType.VARCHAR,
                        max_length=MAX_DESC_LEN),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="severity", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="cvss_score", dtype=DataType.FLOAT),
            # stored as YYYYMMDD int for cheap scalar range filtering
            FieldSchema(name="published_date_int", dtype=DataType.INT64),
            FieldSchema(name="cwe_ids", dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR, max_capacity=MAX_CWES,
                        max_length=MAX_CWE_LEN),
            FieldSchema(name="vendor_products", dtype=DataType.ARRAY,
                        element_type=DataType.VARCHAR, max_capacity=MAX_VPS,
                        max_length=MAX_VP_LEN),
        ]
        schema = CollectionSchema(fields, description="CVE threat intel records")
        self.collection = Collection(
            self.collection_name, schema=schema, using=self._alias
        )

    def _get_collection(self) -> Collection:
        if self.collection is None:
            self.collection = Collection(self.collection_name, using=self._alias)
        return self.collection

    @staticmethod
    def _date_to_int(date_str: "str | None") -> int:
        if not date_str:
            return 0
        return int(date_str.replace("-", ""))

    def insert_batch(self, records: "list[dict]") -> None:
        col = self._get_collection()
        data = [
            [r["id"] for r in records],
            [r["description"][:MAX_DESC_LEN] for r in records],
            [r["embedding"] for r in records],
            [r.get("severity") or "UNKNOWN" for r in records],
            [float(r.get("cvss_score", -1.0)) for r in records],
            [self._date_to_int(r.get("published_date")) for r in records],
            [as_list(r.get("cwe_ids"))[:MAX_CWES] for r in records],
            [as_list(r.get("vendor_products"))[:MAX_VPS] for r in records],
        ]
        col.insert(data)

    def build_index(self) -> float:
        col = self._get_collection()
        start = time.perf_counter()
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": config.MILVUS_INDEX_TYPE,
                "metric_type": config.MILVUS_METRIC_TYPE,
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        col.flush()
        col.load()
        return time.perf_counter() - start

    def hybrid_query(self, query: HybridQuery) -> QueryResult:
        col = self._get_collection()

        expr_parts = []
        if query.severity_in:
            in_list = ", ".join(f'"{s}"' for s in query.severity_in)
            expr_parts.append(f"severity in [{in_list}]")
        if query.min_cvss is not None:
            expr_parts.append(f"cvss_score >= {query.min_cvss}")
        if query.max_cvss is not None:
            expr_parts.append(f"cvss_score <= {query.max_cvss}")
        if query.published_after:
            expr_parts.append(
                f"published_date_int >= {self._date_to_int(query.published_after)}"
            )
        if query.published_before:
            expr_parts.append(
                f"published_date_int <= {self._date_to_int(query.published_before)}"
            )
        if query.cwe_in:
            expr_parts.append(
                "(" + " or ".join(
                    f'array_contains(cwe_ids, "{c}")' for c in query.cwe_in
                ) + ")"
            )
        expr = " and ".join(expr_parts) if expr_parts else None

        start = time.perf_counter()
        results = col.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": config.MILVUS_METRIC_TYPE, "params": {"ef": 128}},
            limit=query.top_k,
            expr=expr,
            output_fields=["id"],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        hits = results[0]
        return QueryResult(
            ids=[h.id for h in hits],
            scores=[float(h.distance) for h in hits],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        col = self._get_collection()
        return col.num_entities

    def close(self) -> None:
        connections.disconnect(self._alias)
