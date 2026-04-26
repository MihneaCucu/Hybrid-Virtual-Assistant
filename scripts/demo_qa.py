from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system

DEFAULT_QUERIES = Path("data/demo_queries.jsonl")


def load_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "query" not in row:
                continue
            rows.append(row)
    return rows


def short_answer(text: str | None, max_len: int = 180) -> str:
    if not text:
        return "None"
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def format_sources(sources: object) -> str:
    if not isinstance(sources, list) or not sources:
        return "-"
    formatted: list[str] = []
    for source in sources[:5]:
        if not isinstance(source, dict):
            continue
        doc_id = source.get("doc_id") or source.get("source_doc") or source.get("id") or "-"
        score = source.get("score")
        if score is None:
            formatted.append(str(doc_id))
        else:
            try:
                formatted.append(f"{doc_id} ({float(score):.3f})")
            except (TypeError, ValueError):
                formatted.append(f"{doc_id} ({score})")
    return ", ".join(formatted) if formatted else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QA demo using predefined queries.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES), help=f"Path to demo query JSONL (default: {DEFAULT_QUERIES})")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of demo queries to run (0 = all).")
    parser.add_argument("--show-sources", action="store_true", help="Print grounding sources returned by the QA module.")
    args = parser.parse_args()

    query_path = Path(args.queries)
    if not query_path.exists():
        raise SystemExit(f"Demo query file not found: {query_path}")

    queries = load_queries(query_path)
    if args.limit > 0:
        queries = queries[: args.limit]

    if not queries:
        raise SystemExit("No demo queries loaded.")

    load_qa_system()

    matched = 0
    print(f"Running QA demo with {len(queries)} queries from {query_path}\n")
    for idx, row in enumerate(queries, start=1):
        q = row["query"]
        expected = row.get("expected_status", "")
        result = answer_question(q)
        got = result.get("status", "")
        ok = bool(expected and got == expected)
        if ok:
            matched += 1

        print(f"[{idx:02d}] {row.get('id', f'q{idx}')}")
        print(f"Q: {q}")
        print(f"Expected: {expected or '-'} | Got: {got} | Match: {'yes' if ok else 'no'}")
        print(f"Reason: {result.get('reason_code')}")
        print(f"Source: {result.get('source_doc')}")
        if args.show_sources:
            print(f"Sources: {format_sources(result.get('sources'))}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Answer: {short_answer(result.get('answer'))}")
        print()

    expected_count = sum(1 for row in queries if row.get("expected_status"))
    if expected_count > 0:
        score = matched / expected_count
        print(f"Status match: {matched}/{expected_count} = {score:.3f}")
    else:
        print("No expected_status labels found in input.")


if __name__ == "__main__":
    main()
