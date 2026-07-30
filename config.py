"""
Central configuration for the vector database comparison project.

Edit the connection settings below to match your local / docker-compose
deployment (see docker-compose.yml). All three backends are optional --
you can run the benchmark against any subset of them by editing
benchmark/run_benchmark.py.
"""

import os

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_CVE_PATH = os.path.join(DATA_DIR, "raw_cve_data.json")
SAMPLE_CVE_PATH = os.path.join(DATA_DIR, "sample_cve_data.json")
EMBEDDED_DATA_PATH = os.path.join(DATA_DIR, "cve_embeddings.parquet")

# NVD API v2.0 (public, no key required for low request volumes -- request an
# API key from https://nvd.nist.gov/developers/request-an-api-key for higher
# rate limits when pulling large amounts of data)
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = os.environ.get("NVD_API_KEY", "")  # optional

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good baseline
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# PostgreSQL / pgvector
# ---------------------------------------------------------------------------
POSTGRES_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "dbname": os.environ.get("PG_DB", "threat_intel"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "postgres"),
}
POSTGRES_TABLE = "cve_records"
POSTGRES_INDEX_TYPE = "hnsw"  # 'hnsw' or 'ivfflat'

# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------
MILVUS_CONFIG = {
    "host": os.environ.get("MILVUS_HOST", "localhost"),
    "port": os.environ.get("MILVUS_PORT", "19530"),
}
MILVUS_COLLECTION = "cve_records"
MILVUS_INDEX_TYPE = "HNSW"
MILVUS_METRIC_TYPE = "COSINE"

# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------
WEAVIATE_CONFIG = {
    "host": os.environ.get("WEAVIATE_HOST", "localhost"),
    "port": int(os.environ.get("WEAVIATE_PORT", 8080)),
    "grpc_port": int(os.environ.get("WEAVIATE_GRPC_PORT", 50051)),
}
WEAVIATE_CLASS = "CveRecord"

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
TOP_K = 10
QUERY_REPEATS = 30          # repetitions per query to compute stable latency stats
DATASET_SIZES = [1000, 5000, 20000]  # scalability sweep; trimmed to available data
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
