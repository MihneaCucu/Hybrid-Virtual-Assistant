from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QAStatus(StrEnum):
    ANSWERED = "answered"
    FALLBACK = "fallback"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class QASource:
    doc_id: str
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id}


@dataclass(frozen=True)
class QAResponse:
    status: QAStatus
    answer: str | None
    source_doc: str | None
    sources: list[QASource] = field(default_factory=list)
    confidence: float = 0.0
    reason_code: str | None = None
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "answer": self.answer,
            "source_doc": self.source_doc,
            "sources": [source.to_dict() for source in self.sources],
            "confidence": round(float(self.confidence), 3),
            "fallback": bool(self.fallback),
        }


REQUIRED_RESPONSE_KEYS = {
    "status",
    "reason_code",
    "answer",
    "source_doc",
    "sources",
    "confidence",
    "fallback",
}


def validate_response_shape(response: dict[str, Any]) -> None:
    missing = REQUIRED_RESPONSE_KEYS - set(response)
    if missing:
        raise ValueError(f"QA response missing keys: {sorted(missing)}")
    if response["status"] not in {status.value for status in QAStatus}:
        raise ValueError(f"Invalid QA status: {response['status']}")
    if not isinstance(response["sources"], list):
        raise ValueError("QA response field 'sources' must be a list")
    if not isinstance(response["fallback"], bool):
        raise ValueError("QA response field 'fallback' must be a bool")
