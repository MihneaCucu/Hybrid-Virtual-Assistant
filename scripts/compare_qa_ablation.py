from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.fallback import should_fallback_reader
from qa.qa_module import answer_question, load_qa_system
from qa.reader import extract_answer, load_reader
from qa.retrieval import load_index, retrieve
from scripts.evaluate_qa import _accepted_answers, _best_answer_match, _gold_doc, _load_positive


POSITIVE_PATH = Path("data/test_set.json")
OUTPUT_PATH = Path("results/qa_ablation_comparison.json")


def _empty_metrics(count: int = 0) -> dict:
    return {
        "count": count,
        "source_doc_accuracy": 0.0,
        "answer_contains_gold_rate": 0.0,
        "token_f1": 0.0,
        "fallback_rate": 0.0,
    }


def _evaluate_predictions(examples: list[dict], predictions: list[dict]) -> dict:
    if not examples:
        return _empty_metrics()

    doc_correct = 0
    contains = 0
    f1_total = 0.0
    fallback = 0

    for ex, pred in zip(examples, predictions, strict=True):
        gold_doc = _gold_doc(ex)
        accepted = _accepted_answers(ex)
        match = _best_answer_match(pred.get("answer"), accepted)
        if pred.get("status") != "answered":
            fallback += 1
        if gold_doc and pred.get("source_doc") == gold_doc:
            doc_correct += 1
        if match["contains"]:
            contains += 1
        f1_total += float(match["f1"])

    count = len(examples)
    return {
        "count": count,
        "source_doc_accuracy": round(doc_correct / count, 3),
        "answer_contains_gold_rate": round(contains / count, 3),
        "token_f1": round(f1_total / count, 3),
        "fallback_rate": round(fallback / count, 3),
    }


def _retrieval_only_predictions(examples: list[dict], bm25, chunks: list[dict]) -> list[dict]:
    predictions = []
    for ex in examples:
        top = retrieve(ex["question"], bm25, chunks, top_k=1)
        if not top:
            predictions.append({"status": "fallback", "answer": None, "source_doc": None})
            continue
        chunk = top[0]["chunk"]
        predictions.append({"status": "answered", "answer": chunk.get("text"), "source_doc": chunk.get("doc_id")})
    return predictions


def _bm25_reader_predictions(examples: list[dict], bm25, chunks: list[dict], use_threshold: bool) -> list[dict]:
    predictions = []
    for ex in examples:
        results = retrieve(ex["question"], bm25, chunks, top_k=6)
        best: tuple[dict, dict] | None = None
        for item in results:
            chunk = item["chunk"]
            extraction = extract_answer(ex["question"], chunk.get("text", ""))
            if extraction["answer"] is None:
                continue
            if best is None or extraction["score"] > best[0]["score"]:
                best = (extraction, chunk)

        if best is None or (use_threshold and should_fallback_reader(best[0]["score"])):
            predictions.append({"status": "fallback", "answer": None, "source_doc": None})
        else:
            predictions.append({"status": "answered", "answer": best[0]["answer"], "source_doc": best[1].get("doc_id")})
    return predictions


def _full_module_predictions(examples: list[dict]) -> list[dict]:
    predictions = []
    for ex in examples:
        result = answer_question(ex["question"])
        predictions.append({"status": result.get("status"), "answer": result.get("answer"), "source_doc": result.get("source_doc")})
    return predictions


def _print_table(comparison: dict) -> None:
    metrics = ["source_doc_accuracy", "answer_contains_gold_rate", "token_f1", "fallback_rate"]
    labels = [
        ("retrieval_only", "Retrieval-only"),
        ("bm25_reader_no_threshold", "BM25+reader"),
        ("bm25_reader_with_threshold", "BM25+reader+threshold"),
        ("full_module", "Full module"),
    ]
    print("QA ablation comparison")
    print(f"Positive examples: {comparison['count']}\n")
    print(f"{'Metric':<30} " + " ".join(f"{label:>22}" for _, label in labels))
    print("-" * (31 + 23 * len(labels)))
    for metric in metrics:
        values = " ".join(f"{comparison[key].get(metric, '-'):>22}" for key, _ in labels)
        print(f"{metric:<30} {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare QA ablations for report/thesis analysis.")
    parser.add_argument("--positive", default=str(POSITIVE_PATH), help=f"Positive test set path (default: {POSITIVE_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Output JSON path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    examples = _load_positive(Path(args.positive))
    chunks, bm25 = load_index()
    load_reader()
    load_qa_system()

    comparison = {
        "count": len(examples),
        "dataset": args.positive,
        "retrieval_only": _evaluate_predictions(examples, _retrieval_only_predictions(examples, bm25, chunks)),
        "bm25_reader_no_threshold": _evaluate_predictions(examples, _bm25_reader_predictions(examples, bm25, chunks, use_threshold=False)),
        "bm25_reader_with_threshold": _evaluate_predictions(examples, _bm25_reader_predictions(examples, bm25, chunks, use_threshold=True)),
        "full_module": _evaluate_predictions(examples, _full_module_predictions(examples)),
        "interpretation": (
            "The ablation separates top-passage retrieval, extractive reading, reader confidence thresholding, "
            "and the full module with structured rules and handoff/fallback behavior."
        ),
    }

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    _print_table(comparison)
    print(f"\nSaved ablation comparison to: {output_path}")


if __name__ == "__main__":
    main()
