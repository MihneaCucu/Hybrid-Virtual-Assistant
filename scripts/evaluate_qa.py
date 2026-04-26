from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qa.qa_module import answer_question, load_qa_system
from qa.retrieval import load_index, retrieve

POSITIVE_PATH = Path("data/test_set.json")
NEGATIVE_PATH = Path("data/negative_test_set.jsonl")
OUTPUT_PATH = Path("results/qa_eval_summary.json")
DETAILS_PATH = Path("results/qa_eval_details.jsonl")


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return " ".join(normalized.split())


def _tokens(text: str | None) -> list[str]:
    normalized = _normalize(text)
    return normalized.split() if normalized else []


def _token_f1(predicted: str | None, gold: str | None) -> float:
    pred_tokens = _tokens(predicted)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = {}
    for token in pred_tokens:
        overlap[token] = overlap.get(token, 0) + 1
    common = 0
    for token in gold_tokens:
        count = overlap.get(token, 0)
        if count:
            common += 1
            overlap[token] = count - 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _accepted_answers(ex: dict) -> list[str]:
    answers: list[str] = []
    if ex.get("answer"):
        answers.append(str(ex["answer"]))
    if ex.get("gold_answer"):
        answers.append(str(ex["gold_answer"]))
    extra = ex.get("accepted_answers", [])
    if isinstance(extra, list):
        answers.extend(str(item) for item in extra if str(item).strip())
    seen: set[str] = set()
    unique: list[str] = []
    for answer in answers:
        norm = _normalize(answer)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(answer)
    return unique


def _gold_doc(ex: dict) -> str:
    return str(ex.get("gold_doc") or ex.get("source_doc") or "").strip()


def _best_answer_match(predicted: str | None, accepted: list[str]) -> dict:
    pred_norm = _normalize(predicted)
    best_f1 = 0.0
    em = False
    contains = False
    for answer in accepted:
        gold_norm = _normalize(answer)
        if not gold_norm:
            continue
        em = em or bool(pred_norm and pred_norm == gold_norm)
        contains = contains or bool(pred_norm and gold_norm in pred_norm)
        best_f1 = max(best_f1, _token_f1(predicted, answer))
    return {"em": em, "contains": contains, "f1": best_f1}


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


def _retrieval_metrics(question: str, gold_doc: str, bm25, chunks: list[dict]) -> dict:
    if not gold_doc:
        return {"recall_at_1": None, "recall_at_3": None, "mrr": None, "top_docs": []}
    results = retrieve(question, bm25, chunks, top_k=5)
    top_docs = []
    seen: set[str] = set()
    rank = None
    for idx, item in enumerate(results, start=1):
        doc_id = item["chunk"].get("doc_id", "")
        if doc_id not in seen:
            top_docs.append({"doc_id": doc_id, "score": round(float(item["score"]), 3)})
            seen.add(doc_id)
        if doc_id == gold_doc and rank is None:
            rank = idx
    return {
        "recall_at_1": rank == 1,
        "recall_at_3": rank is not None and rank <= 3,
        "mrr": (1 / rank) if rank else 0.0,
        "top_docs": top_docs,
    }


def _error_tag_positive(result: dict, match: dict, gold_doc: str, retrieval: dict) -> str:
    if result.get("status") != "answered":
        if retrieval.get("recall_at_3") is False:
            return "retrieval_miss"
        return "false_fallback"
    if gold_doc and result.get("source_doc") != gold_doc:
        if retrieval.get("recall_at_3") is False:
            return "retrieval_miss"
        return "wrong_source_doc"
    if not match["contains"]:
        return "reader_span_bad"
    return "ok"


def _error_tag_negative(result: dict, expected_status: str) -> str:
    if result.get("status") == expected_status:
        return "ok"
    if expected_status == "fallback" and result.get("status") == "answered":
        return "false_answer_instead_of_fallback"
    if expected_status == "handoff":
        return "missed_handoff"
    return "wrong_status"


def _count_tags(details: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in details:
        tag = str(row.get("error_tag", "unknown"))
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_positive(examples: list[dict], bm25, chunks: list[dict]) -> tuple[dict, list[dict]]:
    if not examples:
        return {
            "count": 0,
            "fallback_rate": 0.0,
            "em": 0.0,
            "answer_contains_gold_rate": 0.0,
            "token_f1": 0.0,
            "source_doc_accuracy": 0.0,
            "retrieval_recall_at_1": 0.0,
            "retrieval_recall_at_3": 0.0,
            "retrieval_mrr": 0.0,
            "avg_confidence": 0.0,
        }, []

    fallback = 0
    em = 0
    contains = 0
    f1_total = 0.0
    doc_correct = 0
    recall_at_1 = 0
    recall_at_3 = 0
    mrr_total = 0.0
    retrieval_count = 0
    confidences = []
    details: list[dict] = []

    for ex in examples:
        result = answer_question(ex["question"])
        accepted = _accepted_answers(ex)
        match = _best_answer_match(result.get("answer"), accepted)
        gold_doc = _gold_doc(ex)
        retrieval = _retrieval_metrics(ex["question"], gold_doc, bm25, chunks)
        if result.get("fallback"):
            fallback += 1
        if match["em"]:
            em += 1
        if match["contains"]:
            contains += 1
        if gold_doc and result.get("source_doc") == gold_doc:
            doc_correct += 1
        if retrieval["recall_at_1"] is not None:
            retrieval_count += 1
            recall_at_1 += int(bool(retrieval["recall_at_1"]))
            recall_at_3 += int(bool(retrieval["recall_at_3"]))
            mrr_total += float(retrieval["mrr"])
        f1_total += float(match["f1"])
        confidences.append(float(result.get("confidence", 0.0)))
        error_tag = _error_tag_positive(result, match, gold_doc, retrieval)
        details.append(
            {
                "split": "positive",
                "id": ex.get("id"),
                "question": ex["question"],
                "question_type": ex.get("question_type"),
                "expected_status": ex.get("expected_status", "answered"),
                "status": result.get("status"),
                "reason_code": result.get("reason_code"),
                "gold_doc": gold_doc,
                "source_doc": result.get("source_doc"),
                "accepted_answers": accepted,
                "answer": result.get("answer"),
                "confidence": result.get("confidence"),
                "em": match["em"],
                "contains": match["contains"],
                "f1": round(float(match["f1"]), 3),
                "doc_correct": bool(gold_doc and result.get("source_doc") == gold_doc),
                "retrieval": retrieval,
                "error_tag": error_tag,
            }
        )

    count = len(examples)
    return {
        "count": count,
        "fallback_rate": round(fallback / count, 3),
        "em": round(em / count, 3),
        "answer_contains_gold_rate": round(contains / count, 3),
        "token_f1": round(f1_total / count, 3),
        "source_doc_accuracy": round(doc_correct / count, 3),
        "retrieval_recall_at_1": round(recall_at_1 / retrieval_count, 3) if retrieval_count else 0.0,
        "retrieval_recall_at_3": round(recall_at_3 / retrieval_count, 3) if retrieval_count else 0.0,
        "retrieval_mrr": round(mrr_total / retrieval_count, 3) if retrieval_count else 0.0,
        "avg_confidence": round(sum(confidences) / count, 3),
    }, details


def evaluate_negative(examples: list[dict]) -> tuple[dict, list[dict]]:
    if not examples:
        return {"count": 0, "status_accuracy": 0.0, "correct_fallback_rate": 0.0}, []

    correct_fallbacks = 0
    status_correct = 0
    details: list[dict] = []
    for ex in examples:
        result = answer_question(ex["question"])
        expected_status = str(ex.get("expected_status") or "fallback")
        if result.get("fallback"):
            correct_fallbacks += 1
        if result.get("status") == expected_status:
            status_correct += 1
        error_tag = _error_tag_negative(result, expected_status)
        details.append(
            {
                "split": "negative",
                "id": ex.get("id"),
                "question": ex["question"],
                "expected_status": expected_status,
                "status": result.get("status"),
                "reason_code": result.get("reason_code"),
                "answer": result.get("answer"),
                "source_doc": result.get("source_doc"),
                "confidence": result.get("confidence"),
                "status_correct": result.get("status") == expected_status,
                "error_tag": error_tag,
            }
        )

    count = len(examples)
    return {
        "count": count,
        "status_accuracy": round(status_correct / count, 3),
        "correct_fallback_rate": round(correct_fallbacks / count, 3),
    }, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA module on local datasets.")
    parser.add_argument("--positive", default=str(POSITIVE_PATH), help=f"Positive test set path (default: {POSITIVE_PATH})")
    parser.add_argument("--negative", default=str(NEGATIVE_PATH), help=f"Negative test set path (default: {NEGATIVE_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Output JSON path (default: {OUTPUT_PATH})")
    parser.add_argument("--details-output", default=str(DETAILS_PATH), help=f"Detailed JSONL output path (default: {DETAILS_PATH})")
    args = parser.parse_args()

    positive_path = Path(args.positive)
    negative_path = Path(args.negative)
    output_path = Path(args.output)
    details_path = Path(args.details_output)

    os.makedirs("results", exist_ok=True)
    load_qa_system()
    chunks, bm25 = load_index()

    positive = _load_positive(positive_path)
    negative = _load_negative(negative_path)
    positive_summary, positive_details = evaluate_positive(positive, bm25, chunks)
    negative_summary, negative_details = evaluate_negative(negative)
    all_details = positive_details + negative_details

    summary = {
        "positive": positive_summary,
        "negative": negative_summary,
        "error_tags": {
            "positive": _count_tags(positive_details),
            "negative": _count_tags(negative_details),
            "all": _count_tags(all_details),
        },
        "paths": {
            "positive": str(positive_path),
            "negative": str(negative_path),
            "details": str(details_path),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with details_path.open("w", encoding="utf-8") as f:
        for row in all_details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {output_path}")
    print(f"Saved details to: {details_path}")


if __name__ == "__main__":
    main()
