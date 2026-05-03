from __future__ import annotations

from qa.qa_module import answer_question, load_qa_system
from qa.types import REQUIRED_RESPONSE_KEYS, QAStatus, validate_response_shape


def setup_module() -> None:
    load_qa_system()


def test_answer_response_contract() -> None:
    result = answer_question("What does Arcul de Triumf symbolize?")

    assert set(result) == REQUIRED_RESPONSE_KEYS
    validate_response_shape(result)
    assert result["status"] == QAStatus.ANSWERED.value
    assert result["fallback"] is False
    assert result["answer"]
    assert result["source_doc"]
    assert result["sources"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_fallback_response_contract() -> None:
    result = answer_question("Who is the president of France?")

    assert set(result) == REQUIRED_RESPONSE_KEYS
    validate_response_shape(result)
    assert result["status"] == QAStatus.FALLBACK.value
    assert result["fallback"] is True
    assert result["source_doc"] is None
    assert result["sources"] == []
    assert result["confidence"] == 0.0


def test_handoff_response_contract() -> None:
    result = answer_question("Set a reminder for tomorrow and tell me where University Square is.")

    assert set(result) == REQUIRED_RESPONSE_KEYS
    validate_response_shape(result)
    assert result["status"] == QAStatus.HANDOFF.value
    assert result["fallback"] is True
    assert result["reason_code"] == "MIXED_COMMAND_QUERY"
