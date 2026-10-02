from __future__ import annotations

import json
import pickle
import re

from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "as",
    "do", "does", "did", "can", "could", "would", "should",
    "what", "when", "where", "who", "why", "how", "which",
    "this", "that", "these", "those", "it", "its",
    "me", "my", "your", "our", "their", "them",
}


def load_index(
    chunks_path: str = "kb/chunks.jsonl",
    index_path: str  = "kb/bm25_index.pkl",
) -> tuple[list[dict], BM25Okapi]:
    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    with open(index_path, "rb") as f:
        bm25: BM25Okapi = pickle.load(f)

    return chunks, bm25


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def retrieve(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": float(scores[idx]),
        })

    return results
