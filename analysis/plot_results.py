"""
Generates comparison plots from results/benchmark_results.csv:
  1. Query latency (p50/p95) by backend and workload
  2. Indexing time (insert + index build) vs dataset size, by backend
  3. Recall@k by backend and workload

Usage:
    python plot_results.py --results ../results/benchmark_results.csv --out ../results
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("..")


def plot_query_latency(df: pd.DataFrame, out_dir: str):
    q = df[df["phase"] == "query"]
    if q.empty:
        return
    pivot = q.pivot_table(index="workload", columns="backend", values="p50_ms")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Hybrid Query Latency (p50) by Workload and Backend")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/query_latency_p50.png", dpi=150)
    plt.close()


def plot_indexing_time(df: pd.DataFrame, out_dir: str):
    idx = df[df["phase"] == "indexing"]
    if idx.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for backend, sub in idx.groupby("backend"):
        sub = sub.sort_values("n_records")
        ax.plot(sub["n_records"], sub["total_setup_time_s"], marker="o", label=backend)
    ax.set_xlabel("Dataset size (records)")
    ax.set_ylabel("Total setup time (s)")
    ax.set_title("Indexing Time vs Dataset Size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/indexing_time_scalability.png", dpi=150)
    plt.close()


def plot_recall(df: pd.DataFrame, out_dir: str):
    q = df[df["phase"] == "query"]
    if q.empty:
        return
    pivot = q.pivot_table(index="workload", columns="backend", values="recall_at_k_mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Recall@k by Workload and Backend")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/recall_at_k.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="../results/benchmark_results.csv")
    parser.add_argument("--out", default="../results")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    plot_query_latency(df, args.out)
    plot_indexing_time(df, args.out)
    plot_recall(df, args.out)
    print(f"Plots saved to {args.out}/")


if __name__ == "__main__":
    main()
