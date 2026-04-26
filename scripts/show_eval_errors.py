from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DETAILS_PATH = Path("results/qa_eval_details.jsonl")


def _load_details(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def _short(text: Any, max_len: int = 220) -> str:
    if text is None:
        return "-"
    compact = " ".join(str(text).split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _format_answers(answers: Any) -> str:
    if not isinstance(answers, list) or not answers:
        return "-"
    return "; ".join(_short(answer, 80) for answer in answers)


def _format_top_docs(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, dict):
        return "-"
    docs = retrieval.get("top_docs")
    if not isinstance(docs, list) or not docs:
        return "-"
    formatted = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", "-")
        score = doc.get("score", "-")
        formatted.append(f"{doc_id} ({score})")
    return ", ".join(formatted) if formatted else "-"


def _matches_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.split and row.get("split") != args.split:
        return False
    if args.tag and row.get("error_tag") != args.tag:
        return False
    if not args.show_ok and row.get("error_tag") == "ok":
        return False
    return True


def print_row(row: dict[str, Any], idx: int) -> None:
    expected = row.get("expected_status", "-")
    got = row.get("status", "-")
    print(f"[{idx:02d}] {row.get('id', '-')}")
    print(f"Split/tag: {row.get('split', '-')} / {row.get('error_tag', '-')}")
    print(f"Question: {_short(row.get('question'))}")
    print(f"Status: expected={expected} got={got} reason={row.get('reason_code', '-')}")
    print(f"Gold doc: {row.get('gold_doc', '-') or '-'}")
    print(f"Pred doc: {row.get('source_doc', '-') or '-'}")
    print(f"Accepted: {_format_answers(row.get('accepted_answers'))}")
    print(f"Answer: {_short(row.get('answer'))}")
    print(f"Top docs: {_format_top_docs(row)}")
    if row.get("f1") is not None or row.get("confidence") is not None:
        print(f"F1: {row.get('f1', '-')} | Confidence: {row.get('confidence', '-')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect QA evaluation errors from qa_eval_details.jsonl.")
    parser.add_argument("--details", default=str(DEFAULT_DETAILS_PATH), help=f"Details JSONL path (default: {DEFAULT_DETAILS_PATH})")
    parser.add_argument("--tag", help="Only show rows with this error_tag, for example retrieval_miss or reader_span_bad.")
    parser.add_argument("--split", choices=["positive", "negative"], help="Only show one evaluation split.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to show (0 = no limit).")
    parser.add_argument("--show-ok", action="store_true", help="Include rows whose error_tag is ok.")
    args = parser.parse_args()

    details_path = Path(args.details)
    if not details_path.exists():
        raise SystemExit(f"Details file not found: {details_path}. Run: python scripts/evaluate_qa.py")

    rows = [row for row in _load_details(details_path) if _matches_filters(row, args)]
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        print("No matching QA evaluation rows found.")
        return

    print(f"Showing {len(rows)} QA evaluation row(s) from {details_path}\n")
    for idx, row in enumerate(rows, start=1):
        print_row(row, idx)


if __name__ == "__main__":
    main()
