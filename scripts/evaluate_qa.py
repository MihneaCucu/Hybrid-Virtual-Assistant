from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system

POSITIVE_PATH = Path("data/test_set.json")
NEGATIVE_PATH = Path("data/negative_test_set.jsonl")
OUTPUT_PATH = Path("results/qa_eval_summary.json")


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(normalized.lower().split())


def _load_positive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data and isinstance(data["examples"], list):
        return data["examples"]
    raise ValueError(f"Unsupported positive test format in {path}")


def _load_negative(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_positive(examples: list[dict]) -> dict:
    if not examples:
        return {
            "count": 0,
            "fallback_rate": 0.0,
            "em": 0.0,
            "answer_contains_gold_rate": 0.0,
            "source_doc_accuracy": 0.0,
            "avg_confidence": 0.0,
        }

    fallback = 0
    em = 0
    contains = 0
    doc_correct = 0
    confidences = []

    for ex in examples:
        result = answer_question(ex["question"])
        predicted = _normalize(result.get("answer"))
        gold = _normalize(ex.get("answer"))
        if result.get("fallback"):
            fallback += 1
        if predicted and predicted == gold:
            em += 1
        if predicted and gold and gold in predicted:
            contains += 1
        gold_doc = ex.get("source_doc")
        if gold_doc and result.get("source_doc") == gold_doc:
            doc_correct += 1
        confidences.append(float(result.get("confidence", 0.0)))

    count = len(examples)
    return {
        "count": count,
        "fallback_rate": round(fallback / count, 3),
        "em": round(em / count, 3),
        "answer_contains_gold_rate": round(contains / count, 3),
        "source_doc_accuracy": round(doc_correct / count, 3),
        "avg_confidence": round(sum(confidences) / count, 3),
    }


def evaluate_negative(examples: list[dict]) -> dict:
    if not examples:
        return {"count": 0, "correct_fallback_rate": 0.0}

    correct_fallbacks = 0
    for ex in examples:
        result = answer_question(ex["question"])
        if result.get("fallback"):
            correct_fallbacks += 1

    count = len(examples)
    return {
        "count": count,
        "correct_fallback_rate": round(correct_fallbacks / count, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA module on local datasets.")
    parser.add_argument("--positive", default=str(POSITIVE_PATH), help=f"Positive test set path (default: {POSITIVE_PATH})")
    parser.add_argument("--negative", default=str(NEGATIVE_PATH), help=f"Negative test set path (default: {NEGATIVE_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Output JSON path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    positive_path = Path(args.positive)
    negative_path = Path(args.negative)
    output_path = Path(args.output)

    os.makedirs("results", exist_ok=True)
    load_qa_system()

    positive = _load_positive(positive_path)
    negative = _load_negative(negative_path)

    summary = {
        "positive": evaluate_positive(positive),
        "negative": evaluate_negative(negative),
        "paths": {
            "positive": str(positive_path),
            "negative": str(negative_path),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
