"""
fallback.py - Confidence scoring and fallback logic for the QA module.

Centralizes threshold decisions and structured response builders.
"""

from __future__ import annotations

from qa.types import QAResponse, QASource, QAStatus

BM25_THRESHOLD: float = 2.0

# If top-1 and top-2 retrieval scores are almost identical, confidence is weak.
BM25_MARGIN_THRESHOLD: float = 0.10

READER_THRESHOLD: float = 0.10


def should_fallback_retrieval(top1_score: float, top2_score: float | None = None) -> bool:
    if top1_score < BM25_THRESHOLD:
        return True
    if top2_score is not None and (top1_score - top2_score) < BM25_MARGIN_THRESHOLD:
        return True
    return False


def should_fallback_reader(reader_score: float) -> bool:
    return reader_score < READER_THRESHOLD


def make_fallback_response(reason_code: str = "LOW_CONFIDENCE", answer: str | None = None) -> dict:
    return QAResponse(
        status=QAStatus.FALLBACK,
        reason_code=reason_code,
        answer=answer,
        source_doc=None,
        confidence=0.0,
        fallback=True,
    ).to_dict()


def make_handoff_response(reason_code: str = "MIXED_COMMAND_QUERY") -> dict:
    """Return response instructing the router/dialogue manager to hand off."""
    return QAResponse(
        status=QAStatus.HANDOFF,
        reason_code=reason_code,
        answer=None,
        source_doc=None,
        confidence=0.0,
        fallback=True,
    ).to_dict()


def make_passage_response(chunk: dict, bm25_score: float, reason_code: str = "LOW_READER_CONFIDENCE") -> dict:
    normalized = min(bm25_score / 15.0, 1.0)
    return QAResponse(
        status=QAStatus.ANSWERED,
        reason_code=reason_code,
        answer=chunk["text"],
        source_doc=chunk["doc_id"],
        sources=[QASource(doc_id=chunk["doc_id"], chunk_id=chunk.get("chunk_id"))],
        confidence=normalized,
        fallback=False,
    ).to_dict()


def make_answer_response(answer: str, chunk: dict, reader_score: float) -> dict:
    return QAResponse(
        status=QAStatus.ANSWERED,
        reason_code=None,
        answer=answer,
        source_doc=chunk["doc_id"],
        sources=[QASource(doc_id=chunk["doc_id"], chunk_id=chunk.get("chunk_id"))],
        confidence=reader_score,
        fallback=False,
    ).to_dict()
