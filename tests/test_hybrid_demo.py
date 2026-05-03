from __future__ import annotations

from qa.qa_module import load_qa_system
from scripts.hybrid_demo_cli import route_request


def setup_module() -> None:
    load_qa_system()


def test_hybrid_command_route() -> None:
    result = route_request("Set a reminder for 9 AM tomorrow.")

    assert result["route"] == "command"
    assert result["status"] == "simulated"
    assert result["intent"] == "set_reminder"


def test_hybrid_qa_route() -> None:
    result = route_request("What does Arcul de Triumf symbolize?")

    assert result["route"] == "qa"
    assert result["status"] == "answered"
    assert "First World War" in result["answer"]


def test_hybrid_fallback_route() -> None:
    result = route_request("Tell me a joke about computers.")

    assert result["route"] == "fallback"
    assert result["status"] == "fallback"
    assert result["reason_code"] == "UNSUPPORTED_CHITCHAT"


def test_hybrid_handoff_route() -> None:
    result = route_request("Set a reminder for tomorrow and tell me where University Square is.")

    assert result["route"] == "handoff"
    assert result["status"] == "handoff"
    assert result["reason_code"] == "MIXED_COMMAND_QUERY"
