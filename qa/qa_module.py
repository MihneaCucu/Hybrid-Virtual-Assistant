"""
qa_module.py — Public API for the Knowledge/QA module.

Usage:
    from qa.qa_module import load_qa_system, answer_question

    load_qa_system()   # call ONCE at startup
    result = answer_question("What are the opening hours of the Louvre?")
    # result = {
    #     "status": "answered",
    #     "reason_code": None,
    #     "answer": "The Louvre is open from 9am to 6pm.",
    #     "source_doc": "louvre_museum",
    #     "sources": [{"doc_id": "louvre_museum", "chunk_id": "louvre_museum_001"}],
    #     "confidence": 0.83,
    #     "fallback": False,
    # }
"""

from __future__ import annotations

import re

from qa.retrieval import load_index, retrieve
from qa.reader import load_reader, extract_answer
from qa.fallback import (
    make_answer_response,
    make_fallback_response,
    make_handoff_response,
    should_fallback_reader,
    should_fallback_retrieval,
)

_system_loaded = False
_chunks = []
_bm25 = None


def _looks_like_question(text: str) -> bool:
    lowered = text.lower().strip()
    if "?" in lowered:
        return True
    return re.search(r"\b(what|when|where|who|why|how|which|tell me|is|are|can)\b", lowered) is not None


def _looks_like_command(text: str) -> bool:
    lowered = text.lower().strip()
    return re.search(r"\b(set|book|schedule|create|add|cancel|remind)\b", lowered) is not None


def load_qa_system(
    kb_path: str = "kb/chunks.jsonl",
    index_path: str = "kb/bm25_index.pkl",
) -> None:
    """
    Load the knowledge base index and QA model into memory.
    Must be called ONCE before any calls to answer_question().

    Args:
        kb_path:    Path to the chunked knowledge base (JSONL).
        index_path: Path to the serialized BM25 index (pickle).
    """
    global _system_loaded, _chunks, _bm25
    if _system_loaded:
        return

    _chunks, _bm25 = load_index(chunks_path=kb_path, index_path=index_path)
    load_reader()
    _system_loaded = True


def answer_question(query: str) -> dict:
    """
    Answer a factual question using the loaded knowledge base.

    Args:
        query: The user's natural-language question.

    Returns:
        A dict with the following keys:
            status      (str)         - answered | fallback | handoff
            reason_code (str | None)  - machine-readable fallback/handoff reason
            answer      (str | None)  - extracted answer span, or None on fallback
            source_doc  (str | None)  - document slug the answer came from
            sources     (list[dict])  - evidence chunks used
            confidence  (float)       - confidence score in [0, 1]
            fallback    (bool)        - True if no reliable answer was found
    """
    if not _system_loaded:
        raise RuntimeError("Must call load_qa_system() before answering questions.")

    query = query.strip()
    if not query:
        return make_fallback_response(reason_code="EMPTY_QUERY")

    if _looks_like_question(query) and _looks_like_command(query):
        return make_handoff_response(reason_code="MIXED_COMMAND_QUERY")

    # 1. Retrieve top passages
    results = retrieve(query, _bm25, _chunks, top_k=5)

    if not results:
        return make_fallback_response(reason_code="NO_RELEVANT_DOC")

    best_result = results[0]
    best_chunk = best_result["chunk"]
    bm25_score = best_result["score"]
    second_score = results[1]["score"] if len(results) > 1 else None

    # 2. Check retrieval fallback threshold
    if should_fallback_retrieval(bm25_score, second_score):
        return make_fallback_response(reason_code="LOW_RETRIEVAL_CONFIDENCE")

    # 3. Extract exact answer span
    extraction = extract_answer(query, best_chunk["text"])

    # 4. Check reader fallback threshold
    if extraction["answer"] is None or should_fallback_reader(extraction["score"]):
        return make_fallback_response(reason_code="LOW_READER_CONFIDENCE")

    return make_answer_response(
        answer=extraction["answer"],
        chunk=best_chunk,
        reader_score=extraction["score"],
    )
