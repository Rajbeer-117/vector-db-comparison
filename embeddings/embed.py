"""
Generates dense embeddings for CVE descriptions.

Primary path: sentence-transformers (all-MiniLM-L6-v2, 384-dim). This
requires downloading model weights from huggingface.co the first time it
runs, so it needs outbound internet access.

Fallback path: if sentence-transformers / the model download is unavailable
(e.g. offline dev environment, CI sandbox), a deterministic TF-IDF + SVD
pipeline produces fixed-size vectors of the same dimensionality so the rest
of the pipeline (indexing, hybrid queries, benchmarking) is unaffected. This
is for local development only -- use the real transformer embeddings for
any reported results, since TF-IDF embeddings do not capture the same
semantic relationships.

Usage:
    python embed.py --in ../data/sample_cve_data.json --out ../data/cve_embeddings.parquet
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.append("..")
import config  # noqa: E402


def _embed_with_sentence_transformers(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(embeddings, dtype=np.float32)


def _embed_with_tfidf_fallback(texts: list[str]) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    print(
        "[embed.py] sentence-transformers unavailable -- falling back to "
        "TF-IDF + SVD. This is for offline development only; re-run with "
        "network access for real semantic embeddings before benchmarking."
    )
    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    n_components = min(config.EMBEDDING_DIM, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf)

    if reduced.shape[1] < config.EMBEDDING_DIM:
        pad = np.zeros((reduced.shape[0], config.EMBEDDING_DIM - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])

    return normalize(reduced).astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        return _embed_with_sentence_transformers(texts)
    except Exception as e:  # noqa: BLE001
        print(f"[embed.py] sentence-transformers failed ({e!r}); using fallback.")
        return _embed_with_tfidf_fallback(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=config.SAMPLE_CVE_PATH)
    parser.add_argument("--out", dest="out_path", default=config.EMBEDDED_DATA_PATH)
    args = parser.parse_args()

    with open(args.in_path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    embeddings = embed_texts(df["description"].tolist())
    df["embedding"] = list(embeddings)

    df.to_parquet(args.out_path, index=False)
    print(f"Embedded {len(df)} records ({embeddings.shape[1]}-dim) -> {args.out_path}")


if __name__ == "__main__":
    main()
