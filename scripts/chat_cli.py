from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system


QUIT_WORDS = {"quit", "exit", ":q", "/q"}


def compact(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def print_result(result: dict, show_meta: bool) -> None:
    status = str(result.get("status", "fallback"))
    answer = compact(result.get("answer"))
    reason = str(result.get("reason_code") or "")
    source = str(result.get("source_doc") or "")
    confidence = result.get("confidence", 0.0)

    if status == "answered" and answer:
        print(f"Assistant: {answer}")
    elif status == "handoff":
        if answer:
            print(f"Assistant: {answer}")
        else:
            print("Assistant: This looks like a command request and should go to the command branch.")
    else:
        if answer:
            print(f"Assistant: {answer}")
        else:
            print("Assistant: I could not answer reliably from the knowledge base.")

    if show_meta:
        print(f"[meta] status={status} reason={reason or '-'} source={source or '-'} confidence={confidence}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive terminal QA chat.")
    parser.add_argument("--show-meta", action="store_true", help="Print status/reason/source/confidence after each answer.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result for each query.")
    args = parser.parse_args()

    load_qa_system()

    print("QA terminal chat ready.")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

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

        result = answer_question(user_text)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result, show_meta=args.show_meta)
        print()


if __name__ == "__main__":
    main()
