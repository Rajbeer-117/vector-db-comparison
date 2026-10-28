# Comparative Analysis of Vector Database Systems for Hybrid Cybersecurity Threat Intelligence Retrieval
##Results: **https://rajbeer-117.github.io/report-site/**
Implementation companion to the project proposal. Benchmarks **PostgreSQL
(pgvector)**, **Milvus**, and **Weaviate** on hybrid queries (structured
metadata filters + semantic vector similarity) over CVE/NVD vulnerability
data.

## What "hybrid" means here

Every query workload combines:
- **Structured filters**: severity, CVSS score range, publication date range,
  CWE weakness category
- **Semantic similarity**: nearest-neighbor search over description
  embeddings

Each backend implements this differently under the hood (Postgres: single
SQL statement with `WHERE` + `ORDER BY <=>`; Milvus: boolean filter
expression passed to filtered ANN search; Weaviate: `filters` object passed
to `near_vector`) — see the docstrings in `db_backends/*.py` for the
specifics of each. The benchmark harness drives all three through the same
abstract interface (`db_backends/base.py`) so results are directly
comparable.

## Project layout

```
config.py                    # all connection settings & experiment params
data/
  fetch_nvd_data.py          # pulls real CVEs from the NVD REST API
  generate_sample_data.py    # synthetic CVE-style data for offline dev/testing
embeddings/
  embed.py                   # sentence-transformers embeddings (+TF-IDF fallback)
db_backends/
  base.py                    # shared interface (HybridQuery, QueryResult, ABC)
  postgres_backend.py
  milvus_backend.py
  weaviate_backend.py
  mock_backend.py            # in-memory backend for pipeline testing w/o servers
benchmark/
  queries.py                 # the 6 hybrid query workloads used for comparison
  metrics.py                 # latency percentiles, recall@k, brute-force ground truth
  run_benchmark.py           # main orchestrator: indexing + query benchmarks
  smoke_test.py              # end-to-end sanity check using MockBackend
analysis/
  plot_results.py            # latency / indexing-time / recall comparison charts
docker-compose.yml           # local Postgres+pgvector, Milvus, Weaviate stack
results/                     # CSV + PNG output lands here
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# start the three databases locally
docker compose up -d
```

## 1. Get data

Real NVD data (needs outbound internet; NVD rate-limits unauthenticated
clients to ~5 req/30s — request a free API key at
https://nvd.nist.gov/developers/request-an-api-key and set `NVD_API_KEY`
to go faster):

```bash
cd data
python fetch_nvd_data.py --limit 20000 --out raw_cve_data.json
```

Or, for quick local development without hitting the live API:

```bash
cd data
python generate_sample_data.py --n 5000 --out sample_cve_data.json
```

## 2. Generate embeddings

```bash
cd embeddings
python embed.py --in ../data/raw_cve_data.json --out ../data/cve_embeddings.parquet
```

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim). If the model can't
be downloaded (no internet), this automatically falls back to a TF-IDF+SVD
embedding so the rest of the pipeline still runs — **only use the real
transformer embeddings for any results you report**, the fallback is for
offline development only.

## 3. Run the benchmark

```bash
cd benchmark
python run_benchmark.py \
  --data ../data/cve_embeddings.parquet \
  --backends postgres milvus weaviate \
  --sizes 1000 5000 20000
```

This sweeps dataset size (`config.DATASET_SIZES`), and for each
(backend, size) pair measures:
- bulk insertion time
- ANN index build time
- per-workload query latency (mean/p50/p95/p99, over `config.QUERY_REPEATS`
  repetitions)
- recall@k against an in-memory brute-force ground truth

Results are written to `results/benchmark_results.csv`.

## 4. Sanity-check the pipeline without live databases

```bash
cd benchmark
python smoke_test.py
```

Runs the full ingestion → hybrid-filter → metrics pipeline against
`MockBackend`, an in-memory numpy implementation of the same interface.
Useful for validating changes to `queries.py` or `metrics.py` without
needing Postgres/Milvus/Weaviate running, and for CI.

## 5. Generate comparison charts

```bash
cd analysis
python plot_results.py --results ../results/benchmark_results.csv --out ../results
```

Produces:
- `query_latency_p50.png` — latency by workload × backend
- `indexing_time_scalability.png` — setup time vs dataset size
- `recall_at_k.png` — retrieval quality by workload × backend

**Note:** `results/` currently contains charts generated from
`benchmark/demo_run_with_mock.py`, a synthetic run using `MockBackend`
with fake per-backend latency offsets, included only to confirm the
plotting code works end-to-end without live database servers. Delete
`results/benchmark_results.csv` and the PNGs and re-run
`run_benchmark.py` against real Postgres/Milvus/Weaviate instances before
drawing any conclusions — the demo numbers are not measurements of the
real systems.

## Query workloads (`benchmark/queries.py`)

| Workload | Structured filter(s) |
|---|---|
| `pure_semantic` | none (baseline) |
| `severity_filtered` | severity ∈ {CRITICAL, HIGH} |
| `cvss_range_filtered` | 7.0 ≤ CVSS ≤ 9.0 |
| `temporal_filtered` | published 2023–2024 |
| `cwe_filtered` | CWE-22 (path traversal) |
| `compound_filtered` | severity + CVSS + date, combined (realistic analyst query) |

## Notes on fairness / known asymmetries

- Weaviate builds its HNSW index incrementally during insertion, so there's
  no separate index-build phase the way there is for Postgres
  (`CREATE INDEX ... USING hnsw`) and Milvus (`create_index` after bulk
  load). `WeaviateBackend.build_index()` returns 0 and this is called out
  explicitly rather than hidden, since directly comparing "total setup
  time" across backends needs this caveat.
- All three backends are configured with cosine similarity and HNSW
  (or HNSW-equivalent) ANN indexes with comparable `M`/`efConstruction`
  settings, but each system's HNSW implementation and default tuning
  differs — see each backend file for the exact parameters used, and treat
  latency comparisons as characterizing *these configurations*, not the
  theoretical ceiling of each system.
- Recall@k ground truth is computed by brute-force cosine similarity over
  the same embeddings used to populate each backend, filtered in pandas
  using the same predicate as the corresponding hybrid query, so recall
  differences reflect real ANN/filter-interaction behavior rather than
  ground-truth mismatch.
