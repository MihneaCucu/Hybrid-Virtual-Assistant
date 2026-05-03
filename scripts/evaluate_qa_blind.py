from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system
from scripts.evaluate_qa import _best_answer_match, _load_negative, _retrieval_metrics
from qa.retrieval import load_index


BLIND_PATH = Path("data/blind_test_set.jsonl")
OUTPUT_PATH = Path("results/qa_blind_eval_summary.json")
DETAILS_PATH = Path("results/qa_blind_eval_details.jsonl")


def _accepted_answers(ex: dict) -> list[str]:
    values = ex.get("accepted_answers", [])
    if isinstance(values, list):
        return [str(item) for item in values if str(item).strip()]
    return []


def _error_tag(result: dict, ex: dict, match: dict, retrieval: dict) -> str:
    expected_status = str(ex.get("expected_status", "fallback"))
    if result.get("status") != expected_status:
        if expected_status == "answered":
            return "missed_answer"
        if expected_status == "handoff":
            return "missed_handoff"
        return "false_answer_or_wrong_status"
    if expected_status != "answered":
        return "ok"
    gold_doc = str(ex.get("gold_doc", "")).strip()
    if gold_doc and result.get("source_doc") != gold_doc:
        if retrieval.get("recall_at_3") is False:
            return "retrieval_miss"
        return "wrong_source_doc"
    accepted = _accepted_answers(ex)
    if accepted and not match["contains"]:
        return "answer_mismatch"
    return "ok"


def _count_tags(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        tag = str(row.get("error_tag", "unknown"))
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_blind(examples: list[dict], bm25, chunks: list[dict]) -> tuple[dict, list[dict]]:
    details: list[dict] = []
    status_correct = 0
    answer_contains = 0
    source_correct = 0
    answered_count = 0

    for ex in examples:
        result = answer_question(ex["question"])
        expected_status = str(ex.get("expected_status", "fallback"))
        accepted = _accepted_answers(ex)
        match = _best_answer_match(result.get("answer"), accepted)
        gold_doc = str(ex.get("gold_doc", "")).strip()
        retrieval = _retrieval_metrics(ex["question"], gold_doc, bm25, chunks) if gold_doc else {"top_docs": []}

        status_ok = result.get("status") == expected_status
        status_correct += int(status_ok)
        if expected_status == "answered":
            answered_count += 1
            answer_contains += int(bool(match["contains"]))
            source_correct += int(bool(gold_doc and result.get("source_doc") == gold_doc))

        tag = _error_tag(result, ex, match, retrieval)
        details.append(
            {
                "id": ex.get("id"),
                "category": ex.get("category"),
                "question": ex["question"],
                "expected_status": expected_status,
                "status": result.get("status"),
                "reason_code": result.get("reason_code"),
                "gold_doc": gold_doc,
                "source_doc": result.get("source_doc"),
                "accepted_answers": accepted,
                "answer": result.get("answer"),
                "contains": match["contains"],
                "f1": round(float(match["f1"]), 3),
                "retrieval": retrieval,
                "error_tag": tag,
            }
        )

    count = len(examples)
    return {
        "count": count,
        "answered_expected": answered_count,
        "status_accuracy": round(status_correct / count, 3) if count else 0.0,
        "answer_contains_gold_rate": round(answer_contains / answered_count, 3) if answered_count else 0.0,
        "source_doc_accuracy": round(source_correct / answered_count, 3) if answered_count else 0.0,
        "error_tags": _count_tags(details),
    }, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA module on a blind/stress JSONL set.")
    parser.add_argument("--input", default=str(BLIND_PATH), help=f"Blind set JSONL path (default: {BLIND_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Summary JSON path (default: {OUTPUT_PATH})")
    parser.add_argument("--details-output", default=str(DETAILS_PATH), help=f"Details JSONL path (default: {DETAILS_PATH})")
    args = parser.parse_args()

    load_qa_system()
    chunks, bm25 = load_index()
    examples = _load_negative(Path(args.input))
    summary, details = evaluate_blind(examples, bm25, chunks)

    output_path = Path(args.output)
    details_path = Path(args.details_output)
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved blind summary to: {output_path}")
    print(f"Saved blind details to: {details_path}")


if __name__ == "__main__":
    main()
