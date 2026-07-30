"""
Generates a small, realistic synthetic CVE-style dataset for local
development, unit testing, and CI -- for when you don't want to (or can't)
hit the live NVD API. Schema-compatible with fetch_nvd_data.py output, so
every downstream script (embed.py, db backends, benchmark) works unchanged
against either source.

Usage:
    python generate_sample_data.py --n 500 --out data/sample_cve_data.json
"""

import argparse
import json
import random
from datetime import datetime, timedelta

random.seed(42)

VENDORS_PRODUCTS = [
    "apache:http_server", "microsoft:windows_server", "openssl:openssl",
    "linux:linux_kernel", "cisco:ios_xe", "vmware:esxi", "oracle:mysql",
    "nginx:nginx", "wordpress:wordpress", "adobe:acrobat_reader",
    "google:chrome", "mozilla:firefox", "fortinet:fortios", "sap:netweaver",
    "atlassian:confluence", "docker:docker_engine", "kubernetes:kubernetes",
    "redis:redis", "elastic:elasticsearch", "jenkins:jenkins",
]

CWE_CATALOG = [
    ("CWE-79", "cross-site scripting"),
    ("CWE-89", "SQL injection"),
    ("CWE-119", "buffer overflow"),
    ("CWE-352", "cross-site request forgery"),
    ("CWE-287", "improper authentication"),
    ("CWE-22", "path traversal"),
    ("CWE-502", "insecure deserialization"),
    ("CWE-798", "hard-coded credentials"),
    ("CWE-918", "server-side request forgery"),
    ("CWE-611", "XML external entity injection"),
]

TEMPLATES = [
    "A {cwe_name} vulnerability in {vp} allows a remote unauthenticated "
    "attacker to {impact} via {vector}.",
    "{vp} is vulnerable to {cwe_name}, which could allow a local attacker "
    "to {impact} by supplying a crafted {vector}.",
    "An issue was discovered in {vp} where {cwe_name} enables an attacker "
    "with low privileges to {impact} through {vector}.",
    "{vp} contains a {cwe_name} flaw that permits {impact} when processing "
    "a malicious {vector} sent over the network.",
]

IMPACTS = [
    "execute arbitrary code", "escalate privileges to root",
    "bypass authentication", "read sensitive configuration files",
    "cause a denial of service", "exfiltrate credentials",
    "modify application data", "gain unauthorized administrative access",
]

VECTORS = [
    "HTTP request", "API payload", "configuration file", "uploaded document",
    "network packet", "authentication token", "serialized object",
]

SEVERITIES = [
    ("CRITICAL", (9.0, 10.0)),
    ("HIGH", (7.0, 8.9)),
    ("MEDIUM", (4.0, 6.9)),
    ("LOW", (0.1, 3.9)),
]


def _rand_date(start_year=2019, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime(
        "%Y-%m-%d"
    )


def generate(n: int) -> list[dict]:
    records = []
    for i in range(n):
        cwe_id, cwe_name = random.choice(CWE_CATALOG)
        vp = random.choice(VENDORS_PRODUCTS)
        template = random.choice(TEMPLATES)
        description = template.format(
            cwe_name=cwe_name,
            vp=vp.replace(":", " "),
            impact=random.choice(IMPACTS),
            vector=random.choice(VECTORS),
        )
        severity, (lo, hi) = random.choice(SEVERITIES)
        cvss = round(random.uniform(lo, hi), 1)

        n_extra_vp = random.randint(0, 2)
        vendor_products = sorted(
            set([vp] + random.sample(VENDORS_PRODUCTS, n_extra_vp))
        )

        records.append(
            {
                "id": f"CVE-{random.randint(2019, 2025)}-{10000 + i}",
                "description": description,
                "severity": severity,
                "cvss_score": cvss,
                "published_date": _rand_date(),
                "cwe_ids": [cwe_id],
                "vendor_products": vendor_products,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", type=str, default="sample_cve_data.json")
    args = parser.parse_args()

    records = generate(args.n)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} synthetic CVE-style records -> {args.out}")


if __name__ == "__main__":
    main()
