from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system
from qa.routing import looks_like_command, looks_like_question, looks_like_unsupported_chat, normalize_for_routing


QUIT_WORDS = {"quit", "exit", ":q", "/q"}
DEFAULT_QUERIES = Path("data/hybrid_demo_story.jsonl")

def _compact(text: Any, max_len: int = 500) -> str:
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _simulate_command(text: str) -> dict[str, Any]:
    normalized = normalize_for_routing(text)
    if "remind" in normalized or "reminder" in normalized:
        intent = "set_reminder"
        answer = "Command branch: simulated reminder creation."
    elif "book" in normalized:
        intent = "booking_request"
        answer = "Command branch: simulated booking request."
    elif "schedule" in normalized:
        intent = "schedule_event"
        answer = "Command branch: simulated calendar scheduling."
    elif "alarm" in normalized:
        intent = "set_alarm"
        answer = "Command branch: simulated alarm creation."
    else:
        intent = "generic_command"
        answer = "Command branch: simulated command handling."

    return {
        "route": "command",
        "status": "simulated",
        "intent": intent,
        "answer": answer,
        "source_doc": None,
        "confidence": 1.0,
    }


def route_request(text: str) -> dict[str, Any]:
    if not text.strip():
        return {
            "route": "fallback",
            "status": "fallback",
            "reason_code": "EMPTY_QUERY",
            "answer": "Please enter a command or a Bucharest tourist-guide question.",
            "source_doc": None,
            "confidence": 0.0,
        }

    has_command = looks_like_command(text)
    has_question = looks_like_question(text)

    if has_command and has_question:
        qa_result = answer_question(text)
        return {
            "route": "handoff",
            "status": qa_result.get("status", "handoff"),
            "reason_code": qa_result.get("reason_code", "MIXED_COMMAND_QUERY"),
            "answer": "This mixes a command with a factual question, so the dialogue manager should split or reroute it.",
            "source_doc": qa_result.get("source_doc"),
            "confidence": qa_result.get("confidence", 0.0),
        }

    if has_command:
        return _simulate_command(text)

    if looks_like_unsupported_chat(text):
        return {
            "route": "fallback",
            "status": "fallback",
            "reason_code": "UNSUPPORTED_CHITCHAT",
            "answer": "This assistant demo is focused on commands and Bucharest tourist-guide Q&A, not chit-chat.",
            "source_doc": None,
            "confidence": 0.0,
        }

    qa_result = answer_question(text)
    status = str(qa_result.get("status", "fallback"))
    route = "qa" if status == "answered" else status
    return {
        "route": route,
        "status": status,
        "reason_code": qa_result.get("reason_code"),
        "answer": qa_result.get("answer") or "I could not answer reliably from the Bucharest knowledge base.",
        "source_doc": qa_result.get("source_doc"),
        "confidence": qa_result.get("confidence", 0.0),
    }


def _load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("query"):
                rows.append(row)
    return rows


def _print_result(result: dict[str, Any], show_meta: bool, full_answer: bool = False) -> None:
    answer = str(result.get("answer") or "")
    print(f"Assistant: {answer if full_answer else _compact(answer)}")
    if show_meta:
        print(
            "[meta] "
            f"route={result.get('route')} "
            f"status={result.get('status')} "
            f"reason={result.get('reason_code') or result.get('intent') or '-'} "
            f"source={result.get('source_doc') or '-'} "
            f"confidence={result.get('confidence')}"
        )


def _run_scripted(path: Path, show_meta: bool, full_answer: bool = False) -> None:
    rows = _load_queries(path)
    if not rows:
        raise SystemExit(f"No queries loaded from: {path}")

    matched = 0
    expected_count = 0
    print(f"Running hybrid assistant demo with {len(rows)} queries from {path}\n")
    for idx, row in enumerate(rows, start=1):
        query = str(row["query"])
        expected = row.get("expected_route")
        result = route_request(query)
        got = result.get("route")
        if expected:
            expected_count += 1
            matched += int(got == expected)

        print(f"[{idx:02d}] {row.get('id', f'q{idx}')}")
        print(f"You: {query}")
        _print_result(result, show_meta=show_meta, full_answer=full_answer)
        if expected:
            print(f"Expected route: {expected} | Got: {got} | Match: {'yes' if got == expected else 'no'}")
        if row.get("story_point"):
            print(f"Story point: {row['story_point']}")
        print()

    if expected_count:
        print(f"Route match: {matched}/{expected_count} = {matched / expected_count:.3f}")


def _run_interactive(show_meta: bool, full_answer: bool = False) -> None:
    print("Hybrid assistant demo ready.")
    print("Type a command or a Bucharest tourist-guide question. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return

        if not user_text:
            continue
        if user_text.lower() in QUIT_WORDS:
            print("Session closed.")
            return

        _print_result(route_request(user_text), show_meta=show_meta, full_answer=full_answer)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid assistant terminal demo: command simulation, Q&A, fallback, and handoff.")
    parser.add_argument("--queries", help=f"Run a scripted JSONL demo instead of interactive mode, for example {DEFAULT_QUERIES}.")
    parser.add_argument("--story", action="store_true", help=f"Shortcut for --queries {DEFAULT_QUERIES}.")
    parser.add_argument("--show-meta", action="store_true", help="Print route/status/source/confidence metadata.")
    parser.add_argument("--full-answer", action="store_true", help="Print full assistant answers without truncating long text.")
    args = parser.parse_args()

    load_qa_system()

    query_path = Path(args.queries) if args.queries else (DEFAULT_QUERIES if args.story else None)
    if query_path is not None:
        _run_scripted(query_path, show_meta=args.show_meta, full_answer=args.full_answer)
    else:
        _run_interactive(show_meta=args.show_meta, full_answer=args.full_answer)


if __name__ == "__main__":
    main()
