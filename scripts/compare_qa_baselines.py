from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import load_qa_system
from qa.retrieval import load_index, retrieve
from scripts.evaluate_qa import (
    _accepted_answers,
    _best_answer_match,
    _gold_doc,
    _load_positive,
    _retrieval_metrics,
    evaluate_positive,
)


POSITIVE_PATH = Path("data/test_set.json")
OUTPUT_PATH = Path("results/qa_baseline_comparison.json")


def _evaluate_retrieval_only(examples: list[dict], bm25, chunks: list[dict]) -> dict:
    if not examples:
        return {
            "count": 0,
            "source_doc_accuracy": 0.0,
            "answer_contains_gold_rate": 0.0,
            "token_f1": 0.0,
            "fallback_rate": 0.0,
            "retrieval_recall_at_1": 0.0,
            "retrieval_recall_at_3": 0.0,
        }

    doc_correct = 0
    contains = 0
    f1_total = 0.0
    fallback = 0
    recall_at_1 = 0
    recall_at_3 = 0
    retrieval_count = 0

    for ex in examples:
        question = ex["question"]
        accepted = _accepted_answers(ex)
        gold_doc = _gold_doc(ex)
        top = retrieve(question, bm25, chunks, top_k=1)

        if not top:
            predicted_doc = ""
            answer = ""
            fallback += 1
        else:
            predicted_doc = str(top[0]["chunk"].get("doc_id", ""))
            answer = str(top[0]["chunk"].get("text", ""))

        match = _best_answer_match(answer, accepted)
        retrieval = _retrieval_metrics(question, gold_doc, bm25, chunks)

        if gold_doc and predicted_doc == gold_doc:
            doc_correct += 1
        if match["contains"]:
            contains += 1
        f1_total += float(match["f1"])
        if retrieval["recall_at_1"] is not None:
            retrieval_count += 1
            recall_at_1 += int(bool(retrieval["recall_at_1"]))
            recall_at_3 += int(bool(retrieval["recall_at_3"]))

    count = len(examples)
    return {
        "count": count,
        "source_doc_accuracy": round(doc_correct / count, 3),
        "answer_contains_gold_rate": round(contains / count, 3),
        "token_f1": round(f1_total / count, 3),
        "fallback_rate": round(fallback / count, 3),
        "retrieval_recall_at_1": round(recall_at_1 / retrieval_count, 3) if retrieval_count else 0.0,
        "retrieval_recall_at_3": round(recall_at_3 / retrieval_count, 3) if retrieval_count else 0.0,
    }


def _print_table(comparison: dict) -> None:
    metrics = [
        "source_doc_accuracy",
        "answer_contains_gold_rate",
        "token_f1",
        "fallback_rate",
        "retrieval_recall_at_1",
        "retrieval_recall_at_3",
    ]
    retrieval_only = comparison["retrieval_only"]
    full_qa = comparison["full_qa"]

    print("QA baseline comparison")
    print(f"Positive examples: {comparison['count']}\n")
    print(f"{'Metric':<30} {'Retrieval-only':>16} {'Full Q&A':>12}")
    print("-" * 62)
    for metric in metrics:
        left = retrieval_only.get(metric, "-")
        right = full_qa.get(metric, "-")
        print(f"{metric:<30} {left:>16} {right:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval-only QA with the full Q&A module.")
    parser.add_argument("--positive", default=str(POSITIVE_PATH), help=f"Positive test set path (default: {POSITIVE_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Output JSON path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    positive_path = Path(args.positive)
    output_path = Path(args.output)

    examples = _load_positive(positive_path)
    chunks, bm25 = load_index()

    retrieval_only = _evaluate_retrieval_only(examples, bm25, chunks)

    load_qa_system()
    full_qa, _ = evaluate_positive(examples, bm25, chunks)

    comparison = {
        "count": len(examples),
        "dataset": str(positive_path),
        "retrieval_only": retrieval_only,
        "full_qa": {
            "count": full_qa["count"],
            "source_doc_accuracy": full_qa["source_doc_accuracy"],
            "answer_contains_gold_rate": full_qa["answer_contains_gold_rate"],
            "token_f1": full_qa["token_f1"],
            "fallback_rate": full_qa["fallback_rate"],
            "retrieval_recall_at_1": full_qa["retrieval_recall_at_1"],
            "retrieval_recall_at_3": full_qa["retrieval_recall_at_3"],
        },
        "interpretation": (
            "Retrieval-only returns the top BM25 chunk directly. Full Q&A uses the same retrieval layer "
            "plus structured rules, extractive answer selection, confidence scoring, and safe fallback."
        ),
    }

    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    _print_table(comparison)
    print(f"\nSaved baseline comparison to: {output_path}")


if __name__ == "__main__":
    main()
