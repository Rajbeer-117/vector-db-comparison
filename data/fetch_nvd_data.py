"""
Fetches real-world vulnerability records from the NVD (National Vulnerability
Database) REST API v2.0 and stores them as flat JSON records suitable for
embedding + hybrid indexing.

Each record captures the fields needed for the project's hybrid queries:
  - id (CVE ID)              -> exact match / join key
  - description (text)       -> embedded for semantic search
  - severity (str)           -> structured filter (CRITICAL/HIGH/MEDIUM/LOW)
  - cvss_score (float)       -> structured range filter
  - published_date (str)     -> structured temporal filter
  - cwe_ids (list[str])      -> structured categorical filter
  - vendor_products (list)   -> structured filter

Usage:
    python fetch_nvd_data.py --limit 20000 --out data/raw_cve_data.json

NOTE: This script needs outbound access to services.nvd.nist.gov. NVD
rate-limits unauthenticated clients to ~5 requests / 30s; set NVD_API_KEY
in the environment (see config.py) to raise that to ~50 requests / 30s.
"""

import argparse
import json
import sys
import time

import requests

sys.path.append("..")
import config  # noqa: E402


def _extract_record(vuln: dict) -> dict | None:
    cve = vuln.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        return None

    descriptions = cve.get("descriptions", [])
    description = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"), None
    )
    if not description:
        return None

    metrics = cve.get("metrics", {})
    severity, cvss_score = None, None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            m = metrics[key][0]
            cvss_score = m.get("cvssData", {}).get("baseScore")
            severity = m.get("baseSeverity") or m.get("cvssData", {}).get(
                "baseSeverity"
            )
            break

    cwe_ids = []
    for weakness in cve.get("weaknesses", []):
        for desc in weakness.get("description", []):
            if desc.get("value", "").startswith("CWE-"):
                cwe_ids.append(desc["value"])

    vendor_products = set()
    for config_block in cve.get("configurations", []):
        for node in config_block.get("nodes", []):
            for match in node.get("cpeMatch", []):
                crit = match.get("criteria", "")
                parts = crit.split(":")
                if len(parts) > 4:
                    vendor_products.add(f"{parts[3]}:{parts[4]}")

    return {
        "id": cve_id,
        "description": description,
        "severity": severity or "UNKNOWN",
        "cvss_score": float(cvss_score) if cvss_score is not None else -1.0,
        "published_date": cve.get("published", "")[:10],
        "cwe_ids": cwe_ids,
        "vendor_products": sorted(vendor_products),
    }


def fetch_cves(limit: int, page_size: int = 2000) -> list[dict]:
    records = []
    start_index = 0
    headers = {"apiKey": config.NVD_API_KEY} if config.NVD_API_KEY else {}
    # unauthenticated requests must stay under 5 req/30s
    delay = 6.5 if not config.NVD_API_KEY else 0.7

    while len(records) < limit:
        params = {"startIndex": start_index, "resultsPerPage": page_size}
        resp = requests.get(
            config.NVD_API_URL, params=params, headers=headers, timeout=30
        )
        resp.raise_for_status()
        payload = resp.json()

        vulns = payload.get("vulnerabilities", [])
        if not vulns:
            break

        for v in vulns:
            rec = _extract_record(v)
            if rec:
                records.append(rec)

        total_results = payload.get("totalResults", len(records))
        start_index += page_size
        print(f"Fetched {len(records)}/{min(limit, total_results)} records...")

        if start_index >= total_results:
            break
        time.sleep(delay)

    return records[:limit]


def main():
    parser = argparse.ArgumentParser(description="Fetch CVE data from NVD")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--out", type=str, default=config.RAW_CVE_PATH)
    args = parser.parse_args()

    records = fetch_cves(args.limit)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
