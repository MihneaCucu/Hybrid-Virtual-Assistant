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
        ("What is the ticket price for the Romanian Athenaeum?", "fallback", "UNSUPPORTED_PRICE_QUERY"),
        ("What are the opening hours of Antipa Museum today?", "fallback", "UNSUPPORTED_LIVE_QUERY"),
        ("Who is the president of France?", "fallback", None),
        ("Set a reminder for tomorrow and tell me where University Square is.", "handoff", "MIXED_COMMAND_QUERY"),
    ],
)
def test_golden_fallback_and_handoff(question: str, expected_status: str, reason_code: str | None) -> None:
    result = answer_question(question)

    assert result["status"] == expected_status
    if reason_code is not None:
        assert result["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("question", "required_fragments"),
    [
        (
            "How do I get to the Palace of the Parliament?",
            ["nearest metro", "not live turn-by-turn navigation"],
        ),
        (
            "How can I reach Romanian Athenaeum?",
            ["Piața Romană", "not live turn-by-turn navigation"],
        ),
        (
            "What transport should I take to Village Museum?",
            ["local transport summary", "not live turn-by-turn navigation"],
        ),
    ],
)
def test_golden_directions_answers(question: str, required_fragments: list[str]) -> None:
    result = answer_question(question)
    answer = str(result["answer"])

    assert result["status"] == "answered"
    assert result["reason_code"] == "RULE_BASED_DIRECTIONS_TRANSPORT_MATCH"
    assert str(result["source_doc"]).startswith("structured_")
    for fragment in required_fragments:
        assert fragment.lower() in answer.lower()


@pytest.mark.parametrize(
    ("question", "expected_source", "required_fragments"),
    [
        (
            "What is the Romanian Athenaeum?",
            "romanian_athenaeum",
            ["concert hall", "1888", "George Enescu", "central Bucharest"],
        ),
        (
            "What can you tell me about the romanian atheneum?",
            "romanian_athenaeum",
            ["concert hall", "1888", "George Enescu", "central Bucharest"],
        ),
        (
            "Tell me about the Palace of the Parliament.",
            "palace_of_the_parliament",
            ["Nicolae Ceaușescu", "1984", "Bucharest"],
        ),
        (
            "What can I see at the Village Museum?",
            "village_museum",
            ["traditional houses", "churches", "Romanian"],
        ),
        (
            "Where is the Romanian Athenaeum?",
            "structured_osm_place_romanian_athenaeum",
            ["Strada Benjamin Franklin 1-3", "Bucharest"],
        ),
        (
            "Why visit Cișmigiu Gardens?",
            "cismigiu_gardens",
            ["walk", "green", "city center"],
        ),
    ],
)
def test_golden_rich_entity_profile_answers(
    question: str,
    expected_source: str,
    required_fragments: list[str],
) -> None:
    result = answer_question(question)
    answer = str(result["answer"])

    assert result["status"] == "answered"
    assert result["source_doc"] == expected_source
    for fragment in required_fragments:
        assert fragment.lower() in answer.lower()
