from __future__ import annotations

import pytest

from qa.qa_module import answer_question, load_qa_system


def setup_module() -> None:
    load_qa_system()


@pytest.mark.parametrize(
    ("question", "expected_status", "expected_source", "answer_fragment"),
    [
        (
            "What collections does the National Museum of Art of Romania feature?",
            "answered",
            "national_museum_of_art_of_romania",
            "Romanian art",
        ),
        (
            "What metro station is Antipa Museum at?",
            "answered",
            "structured_place_metro_museum_00012",
            "Piața Victoriei",
        ),
        (
            "Show metro as well as STB stations near Romanian Athenaeum",
            "answered",
            "structured_place_metro_osm_place_romanian_athenaeum",
            "Nearby transport",
        ),
        (
            "What season is best to visit Bucharest?",
            "answered",
            "curated_bucharest_travel_guidance",
            "spring",
        ),
    ],
)
def test_golden_answered_questions(question: str, expected_status: str, expected_source: str, answer_fragment: str) -> None:
    result = answer_question(question)

    assert result["status"] == expected_status
    assert result["source_doc"] == expected_source
    assert answer_fragment.lower() in str(result["answer"]).lower()


@pytest.mark.parametrize(
    ("question", "expected_status", "reason_code"),
    [
        ("How much is a metro ticket in Bucharest?", "fallback", "UNSUPPORTED_PRICE_QUERY"),
        ("Who is the president of France?", "fallback", None),
        ("Set a reminder for tomorrow and tell me where University Square is.", "handoff", "MIXED_COMMAND_QUERY"),
    ],
)
def test_golden_fallback_and_handoff(question: str, expected_status: str, reason_code: str | None) -> None:
    result = answer_question(question)

    assert result["status"] == expected_status
    if reason_code is not None:
        assert result["reason_code"] == reason_code
