"""
build_index.py — Chunk clean documents and build the BM25 index.

Reads:  kb/clean/*.txt
Writes: kb/chunks.jsonl       — one JSON chunk per line
        kb/bm25_index.pkl     — serialized BM25Okapi index

Run:
    python scripts/build_index.py
"""

from __future__ import annotations

import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi

CLEAN_DIR    = "kb/clean"
CHUNKS_PATH  = "kb/chunks.jsonl"
INDEX_PATH   = "kb/bm25_index.pkl"

CHUNK_WORDS  = 150
STRIDE_WORDS = 75

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "as",
    "do", "does", "did", "can", "could", "would", "should",
    "what", "when", "where", "who", "why", "how", "which",
    "this", "that", "these", "those", "it", "its",
    "me", "my", "your", "our", "their", "them",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]

def chunk_document(doc_id: str, text: str) -> list[dict]:
    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"{doc_id}_{idx:03d}",
            "doc_id":   doc_id,
            "text":     chunk_text,
        })
        idx   += 1
        start += STRIDE_WORDS

        if len(words) - start < CHUNK_WORDS * 0.25:
            break

    return chunks

def main() -> None:
    txt_files = sorted(
        f for f in os.listdir(CLEAN_DIR) if f.endswith(".txt")
    )

    if not txt_files:
        print(f"ERROR: No .txt files found in {CLEAN_DIR}. Run collect_data.py first.")
        return

    all_chunks: list[dict] = []

    print(f"\n  Building index from {len(txt_files)} documents in '{CLEAN_DIR}/'")
    print(f"  Chunk size: {CHUNK_WORDS} words | Stride: {STRIDE_WORDS} words\n")

    for fname in txt_files:
        doc_id   = fname.replace(".txt", "")
        fpath    = os.path.join(CLEAN_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            text = f.read().strip()

        doc_chunks = chunk_document(doc_id, text)
        all_chunks.extend(doc_chunks)
        print(f"  {doc_id:<32} {len(text.split()):>5} words → {len(doc_chunks)} chunk(s)")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n  Saved {len(all_chunks)} chunks → {CHUNKS_PATH}")

    tokenized = [tokenize(chunk["text"]) for chunk in all_chunks]
    bm25      = BM25Okapi(tokenized)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"  Saved BM25 index     → {INDEX_PATH}")
    print(f"\n  Done. Index ready for retrieval.\n")


if __name__ == "__main__":
    main()
