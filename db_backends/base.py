"""
Abstract interface that every vector database backend implements. Keeping a
single shared contract lets benchmark/run_benchmark.py drive PostgreSQL,
Milvus, and Weaviate through identical code paths, which is what makes the
latency/recall/indexing comparisons apples-to-apples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def as_list(value) -> list:
    """Coerces a field that should be a list into an actual Python list,
    safely handling None, plain lists/tuples, and numpy arrays.

    This matters because pandas/pyarrow deserialize parquet list-type
    columns (e.g. cwe_ids, vendor_products) as numpy arrays rather than
    plain lists. The old pattern `value or []` breaks for numpy arrays --
    Python can't evaluate the truthiness of a multi-element array and
    raises `ValueError: The truth value of an array with more than one
    element is ambiguous`. Always route these fields through this helper
    instead of using `or []` directly.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return list(value)


@dataclass
class HybridQuery:
    """A single hybrid query: semantic similarity + structured filters."""

    query_text: str
    query_embedding: "list[float] | None" = None
    severity_in: "list[str] | None" = None       # e.g. ["CRITICAL", "HIGH"]
    min_cvss: "float | None" = None
    max_cvss: "float | None" = None
    published_after: "str | None" = None          # "YYYY-MM-DD"
    published_before: "str | None" = None
    cwe_in: "list[str] | None" = None
    top_k: int = 10


@dataclass
class QueryResult:
    ids: "list[str]" = field(default_factory=list)
    scores: "list[float]" = field(default_factory=list)
    latency_ms: float = 0.0


class VectorDBBackend(ABC):
    """Common lifecycle: connect -> create_schema -> insert -> query -> close."""

    name: str = "base"

    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def create_schema(self, dim: int, drop_existing: bool = True) -> None:
        ...

    @abstractmethod
    def insert_batch(self, records: "list[dict]") -> None:
        """Each record must contain: id, description, embedding, severity,
        cvss_score, published_date, cwe_ids, vendor_products."""
        ...

    @abstractmethod
    def build_index(self) -> float:
        """Builds/finalizes the ANN index. Returns wall-clock seconds taken."""
        ...

    @abstractmethod
    def hybrid_query(self, query: HybridQuery) -> QueryResult:
        """Executes a single hybrid (structured filter + vector similarity)
        query and returns ranked results with measured latency."""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
