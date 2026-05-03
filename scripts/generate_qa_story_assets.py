from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVAL_SUMMARY_PATH = Path("results/qa_eval_summary.json")
BASELINE_PATH = Path("results/qa_baseline_comparison.json")
BLIND_PATH = Path("results/qa_blind_eval_summary.json")
ABLATION_PATH = Path("results/qa_ablation_comparison.json")
DEMO_PATH = Path("data/demo_queries_story.jsonl")
OUTPUT_PATH = Path("docs/qa_story_assets.md")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"Expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metric_table(summary: dict[str, Any]) -> str:
    positive = summary.get("positive", {})
    negative = summary.get("negative", {})
    rows = [
        ("Positive examples", positive.get("count", "-")),
        ("Negative/mixed examples", negative.get("count", "-")),
        ("Answer contains gold", positive.get("answer_contains_gold_rate", "-")),
        ("Source document accuracy", positive.get("source_doc_accuracy", "-")),
        ("Token F1", positive.get("token_f1", "-")),
        ("Retrieval Recall@1", positive.get("retrieval_recall_at_1", "-")),
        ("Retrieval Recall@3", positive.get("retrieval_recall_at_3", "-")),
        ("Negative/mixed status accuracy", negative.get("status_accuracy", "-")),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {_fmt(value)} |" for name, value in rows)
    return "\n".join(lines)


def _baseline_table(comparison: dict[str, Any]) -> str:
    retrieval_only = comparison.get("retrieval_only", {})
    full_qa = comparison.get("full_qa", {})
    metrics = [
        ("Source document accuracy", "source_doc_accuracy"),
        ("Answer contains gold", "answer_contains_gold_rate"),
        ("Token F1", "token_f1"),
        ("Fallback rate on positives", "fallback_rate"),
        ("Retrieval Recall@1", "retrieval_recall_at_1"),
        ("Retrieval Recall@3", "retrieval_recall_at_3"),
    ]
    lines = ["| Metric | Retrieval-only | Full Q&A |", "| --- | ---: | ---: |"]
    for label, key in metrics:
        lines.append(f"| {label} | {_fmt(retrieval_only.get(key, '-'))} | {_fmt(full_qa.get(key, '-'))} |")
    return "\n".join(lines)


def _blind_table(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Blind/stress evaluation has not been generated yet."
    rows = [
        ("Examples", summary.get("count", "-")),
        ("Answered expected", summary.get("answered_expected", "-")),
        ("Status accuracy", summary.get("status_accuracy", "-")),
        ("Answer contains gold", summary.get("answer_contains_gold_rate", "-")),
        ("Source document accuracy", summary.get("source_doc_accuracy", "-")),
        ("Error tags", ", ".join(f"{k}={v}" for k, v in summary.get("error_tags", {}).items())),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {name} | {_fmt(value)} |" for name, value in rows)
    return "\n".join(lines)


def _ablation_table(comparison: dict[str, Any] | None) -> str:
    if not comparison:
        return "Ablation comparison has not been generated yet."
    labels = [
        ("retrieval_only", "Retrieval-only"),
        ("bm25_reader_no_threshold", "BM25+reader"),
        ("bm25_reader_with_threshold", "BM25+reader+threshold"),
        ("full_module", "Full module"),
    ]
    metrics = [
        ("Source document accuracy", "source_doc_accuracy"),
        ("Answer contains gold", "answer_contains_gold_rate"),
        ("Token F1", "token_f1"),
        ("Fallback rate", "fallback_rate"),
    ]
    lines = ["| Metric | " + " | ".join(label for _, label in labels) + " |"]
    lines.append("| --- | " + " | ".join("---:" for _ in labels) + " |")
    for metric_label, metric_key in metrics:
        values = [_fmt(comparison.get(key, {}).get(metric_key, "-")) for key, _ in labels]
        lines.append(f"| {metric_label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _demo_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Step | Query | Expected | Story point |", "| ---: | --- | --- | --- |"]
    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).replace("|", "\\|")
        expected = str(row.get("expected_status", "-"))
        story_point = str(row.get("story_point", row.get("notes", ""))).replace("|", "\\|")
        lines.append(f"| {idx} | `{query}` | `{expected}` | {story_point} |")
    return "\n".join(lines)


def _error_tag_text(summary: dict[str, Any]) -> str:
    tags = summary.get("error_tags", {}).get("all", {})
    if not isinstance(tags, dict) or not tags:
        return "no error-tag summary available"
    return ", ".join(f"{key}={value}" for key, value in sorted(tags.items()))


def build_markdown(
    summary: dict[str, Any],
    comparison: dict[str, Any],
    demo_rows: list[dict[str, Any]],
    blind_summary: dict[str, Any] | None = None,
    ablation: dict[str, Any] | None = None,
) -> str:
    positive = summary.get("positive", {})
    negative = summary.get("negative", {})
    retrieval_only = comparison.get("retrieval_only", {})
    full_qa = comparison.get("full_qa", {})

    blind_sentence = ""
    if blind_summary:
        blind_sentence = (
            f" On the separate blind/stress set, status accuracy was {_fmt(blind_summary.get('status_accuracy', '-'))}, "
            f"with answer-containing-gold accuracy {_fmt(blind_summary.get('answer_contains_gold_rate', '-'))} on expected answerable examples."
        )

    return f"""# Knowledge/Q&A Story Assets

This file is generated from the current evaluation and baseline artifacts. Use it as the source of truth for report tables, slide numbers, and the short demo story.

## One-Sentence Claim

The Knowledge/Q&A module answers bounded Bucharest tourist-guide questions from a local knowledge base, grounds answers in source documents, and safely falls back or hands off when a request is unsupported or mixed with a command.

## Current Evaluation Snapshot

{_metric_table(summary)}

The latest error-tag summary is: {_error_tag_text(summary)}.

## Baseline Comparison

{_baseline_table(comparison)}

Interpretation: retrieval-only and full Q&A use the same BM25 retriever, so retrieval recall is unchanged. The full Q&A module improves the final answer through structured rules, extractive span selection, source-aware ranking, and fallback/handoff logic.

## Ablation Comparison

{_ablation_table(ablation)}

## Blind/Stress Evaluation

{_blind_table(blind_summary)}

## Recommended Six-Question Demo

Run:

```bash
python scripts/demo_qa.py --queries data/demo_queries_story.jsonl --show-sources
```

For the full hybrid assistant story, run:

```bash
python scripts/hybrid_demo_cli.py --story --show-meta
```

{_demo_table(demo_rows)}

## Report-Ready Paragraph

The final Knowledge/Q&A module is a closed-domain retrieval-based Question Answering system for a Bucharest tourist-guide assistant. It combines deterministic structured rules for addresses, nearest metro stations, nearby transport, travel guidance, and mixed-query handoff with BM25 retrieval and an extractive reader for narrative factual questions. On the local evaluation set, it handled {positive.get("count", "-")} positive examples and {negative.get("count", "-")} unsupported or mixed examples, with source document accuracy {_fmt(positive.get("source_doc_accuracy", "-"))} and negative/mixed status accuracy {_fmt(negative.get("status_accuracy", "-"))}.{blind_sentence} Compared with a retrieval-only baseline, source document accuracy improved from {_fmt(retrieval_only.get("source_doc_accuracy", "-"))} to {_fmt(full_qa.get("source_doc_accuracy", "-"))}, and answer-containing-gold accuracy improved from {_fmt(retrieval_only.get("answer_contains_gold_rate", "-"))} to {_fmt(full_qa.get("answer_contains_gold_rate", "-"))}.

## Slide Bullets

- Scope: Bucharest tourist-guide Q&A, not open-domain chat.
- Data: curated tourist documents plus structured museums, places, transit, and travel guidance.
- Architecture: structured rules -> BM25 retrieval -> extractive reader -> confidence/fallback.
- Safety: no live facts or prices; unsupported questions fallback; mixed command/question requests hand off.
- Evidence: local positive, negative, and baseline evaluations are reproducible with repository scripts.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Knowledge/Q&A report and demo story assets.")
    parser.add_argument("--summary", default=str(EVAL_SUMMARY_PATH), help=f"QA evaluation summary path (default: {EVAL_SUMMARY_PATH})")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help=f"Baseline comparison path (default: {BASELINE_PATH})")
    parser.add_argument("--blind", default=str(BLIND_PATH), help=f"Blind/stress summary path (default: {BLIND_PATH})")
    parser.add_argument("--ablation", default=str(ABLATION_PATH), help=f"Ablation comparison path (default: {ABLATION_PATH})")
    parser.add_argument("--demo", default=str(DEMO_PATH), help=f"Story demo JSONL path (default: {DEMO_PATH})")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help=f"Markdown output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    summary = _load_json(Path(args.summary))
    baseline = _load_json(Path(args.baseline))
    blind = _load_json(Path(args.blind)) if Path(args.blind).exists() else None
    ablation = _load_json(Path(args.ablation)) if Path(args.ablation).exists() else None
    demo_rows = _load_jsonl(Path(args.demo))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(summary, baseline, demo_rows, blind_summary=blind, ablation=ablation), encoding="utf-8")
    print(f"Generated Knowledge/Q&A story assets: {output_path}")


if __name__ == "__main__":
    main()
