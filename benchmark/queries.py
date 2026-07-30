"""
Defines the set of hybrid query workloads used to compare backends. Each
workload pairs a natural-language "analyst intent" with structured filter
constraints, mirroring realistic threat-intelligence lookups (e.g. "find
SQL injection issues in critical severity products discovered this year").
"""

import sys

sys.path.append("..")
from db_backends.base import HybridQuery  # noqa: E402


def build_query_workloads() -> "list[dict]":
    """Returns a list of {name, description, query_text, filters} dicts.
    query_embedding is filled in at run time by the benchmark script."""
    return [
        {
            "name": "pure_semantic",
            "description": "No structured filters -- baseline semantic-only search.",
            "query_text": "remote code execution vulnerability in a web server",
            "filters": {},
        },
        {
            "name": "severity_filtered",
            "description": "Semantic search restricted to CRITICAL/HIGH severity.",
            "query_text": "authentication bypass allowing administrative access",
            "filters": {"severity_in": ["CRITICAL", "HIGH"]},
        },
        {
            "name": "cvss_range_filtered",
            "description": "Semantic search within a CVSS score band.",
            "query_text": "buffer overflow leading to arbitrary code execution",
            "filters": {"min_cvss": 7.0, "max_cvss": 9.0},
        },
        {
            "name": "temporal_filtered",
            "description": "Semantic search restricted to a recent publication window.",
            "query_text": "SQL injection in a database management system",
            "filters": {"published_after": "2023-01-01", "published_before": "2024-12-31"},
        },
        {
            "name": "cwe_filtered",
            "description": "Semantic search constrained to a specific weakness category.",
            "query_text": "attacker able to read files outside the intended directory",
            "filters": {"cwe_in": ["CWE-22"]},
        },
        {
            "name": "compound_filtered",
            "description": "Semantic search with multiple simultaneous structured constraints "
                            "(severity + CVSS range + date window) -- the realistic analyst case.",
            "query_text": "server-side request forgery exposing internal services",
            "filters": {
                "severity_in": ["CRITICAL", "HIGH"],
                "min_cvss": 7.5,
                "published_after": "2022-01-01",
            },
        },
    ]


def to_hybrid_query(workload: dict, query_embedding: "list[float]", top_k: int) -> HybridQuery:
    return HybridQuery(
        query_text=workload["query_text"],
        query_embedding=query_embedding,
        top_k=top_k,
        **workload["filters"],
    )
