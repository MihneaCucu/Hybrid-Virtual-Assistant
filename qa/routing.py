from __future__ import annotations

import re


DEFAULT_QUESTION_MARKERS = ("what", "when", "where", "who", "why", "how", "which", "tell me", "is", "are", "can")
DEFAULT_COMMAND_MARKERS = ("set", "book", "schedule", "create", "add", "cancel", "remind")
DEFAULT_UNSUPPORTED_CHAT_MARKERS = ("joke", "story", "poem", "chat", "talk to me")


def normalize_for_routing(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def contains_marker(text: str, markers: list[str] | tuple[str, ...]) -> bool:
    normalized = normalize_for_routing(text)
    return any(re.search(rf"\b{re.escape(marker)}\b", normalized) is not None for marker in markers)


def looks_like_question(text: str, markers: list[str] | tuple[str, ...] = DEFAULT_QUESTION_MARKERS) -> bool:
    return "?" in text or contains_marker(text, markers)


def looks_like_command(text: str, markers: list[str] | tuple[str, ...] = DEFAULT_COMMAND_MARKERS) -> bool:
    return contains_marker(text, markers)


def looks_like_unsupported_chat(text: str, markers: list[str] | tuple[str, ...] = DEFAULT_UNSUPPORTED_CHAT_MARKERS) -> bool:
    return contains_marker(text, markers)
