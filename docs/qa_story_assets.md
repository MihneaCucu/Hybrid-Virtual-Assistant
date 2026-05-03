# Knowledge/Q&A Story Assets

This file is generated from the current evaluation and baseline artifacts. Use it as the source of truth for report tables, slide numbers, and the short demo story.

## One-Sentence Claim

The Knowledge/Q&A module answers bounded Bucharest tourist-guide questions from a local knowledge base, grounds answers in source documents, and safely falls back or hands off when a request is unsupported or mixed with a command.

## Current Evaluation Snapshot

| Metric | Value |
| --- | ---: |
| Positive examples | 50 |
| Negative/mixed examples | 14 |
| Answer contains gold | 1.000 |
| Source document accuracy | 1.000 |
| Token F1 | 0.679 |
| Retrieval Recall@1 | 0.540 |
| Retrieval Recall@3 | 0.800 |
| Negative/mixed status accuracy | 1.000 |

The latest error-tag summary is: ok=64.

## Baseline Comparison

| Metric | Retrieval-only | Full Q&A |
| --- | ---: | ---: |
| Source document accuracy | 0.540 | 1.000 |
| Answer contains gold | 0.600 | 1.000 |
| Token F1 | 0.052 | 0.679 |
| Fallback rate on positives | 0.000 | 0.000 |
| Retrieval Recall@1 | 0.540 | 0.540 |
| Retrieval Recall@3 | 0.800 | 0.800 |

Interpretation: retrieval-only and full Q&A use the same BM25 retriever, so retrieval recall is unchanged. The full Q&A module improves the final answer through structured rules, extractive span selection, source-aware ranking, and fallback/handoff logic.

## Ablation Comparison

| Metric | Retrieval-only | BM25+reader | BM25+reader+threshold | Full module |
| --- | ---: | ---: | ---: | ---: |
| Source document accuracy | 0.540 | 0.600 | 0.600 | 1.000 |
| Answer contains gold | 0.600 | 0.620 | 0.620 | 1.000 |
| Token F1 | 0.052 | 0.603 | 0.603 | 0.679 |
| Fallback rate | 0.000 | 0.060 | 0.060 | 0.000 |

## Blind/Stress Evaluation

| Metric | Value |
| --- | ---: |
| Examples | 25 |
| Answered expected | 16 |
| Status accuracy | 0.920 |
| Answer contains gold | 0.812 |
| Source document accuracy | 0.812 |
| Error tags | missed_answer=2, ok=22, retrieval_miss=1 |

## Recommended Six-Question Demo

Run:

```bash
python scripts/demo_qa.py --queries data/demo_queries_story.jsonl --show-sources
```

For the full hybrid assistant story, run:

```bash
python scripts/hybrid_demo_cli.py --story --show-meta
```

| Step | Query | Expected | Story point |
| ---: | --- | --- | --- |
| 1 | `What collections does the National Museum of Art of Romania feature?` | `answered` | Narrative Q&A from a museum document |
| 2 | `What does the National History Museum of Romania contain?` | `answered` | Museum-content Q&A, not just addresses |
| 3 | `What does Arcul de Triumf symbolize?` | `answered` | Short extracted landmark fact |
| 4 | `Show metro as well as STB stations near Romanian Athenaeum` | `answered` | Structured assistant-style local transport answer |
| 5 | `How much is a metro ticket in Bucharest?` | `fallback` | Safe fallback for live price information |
| 6 | `Set a reminder for tomorrow and tell me where University Square is.` | `handoff` | Mixed command plus Q&A is handed off to the router/dialogue manager |

## Report-Ready Paragraph

The final Knowledge/Q&A module is a closed-domain retrieval-based Question Answering system for a Bucharest tourist-guide assistant. It combines deterministic structured rules for addresses, nearest metro stations, nearby transport, travel guidance, and mixed-query handoff with BM25 retrieval and an extractive reader for narrative factual questions. On the local evaluation set, it handled 50 positive examples and 14 unsupported or mixed examples, with source document accuracy 1.000 and negative/mixed status accuracy 1.000. On the separate blind/stress set, status accuracy was 0.920, with answer-containing-gold accuracy 0.812 on expected answerable examples. Compared with a retrieval-only baseline, source document accuracy improved from 0.540 to 1.000, and answer-containing-gold accuracy improved from 0.600 to 1.000.

## Slide Bullets

- Scope: Bucharest tourist-guide Q&A, not open-domain chat.
- Data: curated tourist documents plus structured museums, places, transit, and travel guidance.
- Architecture: structured rules -> BM25 retrieval -> extractive reader -> confidence/fallback.
- Safety: no live facts or prices; unsupported questions fallback; mixed command/question requests hand off.
- Evidence: local positive, negative, and baseline evaluations are reproducible with repository scripts.
