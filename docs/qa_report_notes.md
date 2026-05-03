# Knowledge/Q&A Report Notes

## Scope

The Knowledge/Q&A module is a closed-domain Question Answering component for a Bucharest tourist-guide assistant. It answers factual questions from a curated local knowledge base and returns fallback or handoff responses when the question is outside scope.

Supported examples:
- Landmark and museum facts, such as location, construction dates, architectural style, and short descriptions.
- Place addresses and nearest metro information for curated Bucharest attractions.
- Nearby public transport summaries for selected tourist places.
- Curated travel guidance, such as best season, budget, and suggested visit duration.

Unsupported examples:
- Live weather, live prices, current opening hours, ticket availability, and bookings.
- General world knowledge outside Bucharest tourism.
- Mixed command plus factual requests that should be routed or split by the dialogue manager.

## Data And Knowledge Base

The runtime knowledge base is intentionally small and reproducible. It combines curated Bucharest tourism documents, structured local records, and generated text documents used for retrieval.

Main artifacts:
- `data/kb_sources_bucharest.json`: source registry for Bucharest tourism documents.
- `data/bucharest_travel_guidance.json`: curated travel guidance facts.
- `kb/clean/`: cleaned text documents used by the retriever.
- `kb/structured/`: normalized structured records such as museums, places, transit links, and place-to-metro links.
- `kb/chunks.jsonl` and `kb/bm25_index.pkl`: regenerated retrieval artifacts.

The final report/demo profile is built with:

```bash
python scripts/rebuild_qa_kb.py --no-network-fetch --final-demo
```

This profile keeps the presentation scope focused on Bucharest tourist-guide Q&A and excludes bulk restaurant/coffee data unless explicitly allowlisted.

## Architecture

The selected design is retrieval-based closed-domain Q&A:

1. The router sends information questions to `qa.qa_module.answer_question`.
2. Rule-based handlers answer high-value structured tourist queries such as nearest metro, nearby transport, addresses, and curated travel guidance.
3. BM25 retrieves the most relevant text chunks from the Bucharest KB.
4. An extractive reader selects a short answer span from retrieved context.
5. Confidence and relevance thresholds decide whether to answer, fallback, or hand off.
6. Returned metadata includes `status`, `answer`, `source_doc`, `sources`, `confidence`, and `reason_code`.

This design is explainable, deterministic enough for a semester project, and safer than generative Q&A because answers are grounded in retrieved documents.

## Evaluation

Evaluation uses local positive and negative sets:

```bash
python scripts/evaluate_qa.py
```

Outputs:
- `results/qa_eval_summary.json`: aggregate metrics.
- `results/qa_eval_details.jsonl`: per-question predictions and error tags.

Current evaluation dimensions:
- Positive factual Q&A: exact match, answer-contains-gold rate, token F1, source document accuracy, fallback rate, average confidence.
- Retrieval quality: Recall@1, Recall@3, and MRR.
- Negative and unsupported questions: expected status accuracy and correct fallback behavior.
- Error tags: `retrieval_miss`, `wrong_source_doc`, `reader_span_bad`, `false_fallback`, `missed_handoff`, and `false_answer_instead_of_fallback`.

Current result snapshot:
- Positive test set: 50 examples.
- Negative and mixed test set: 14 examples.
- Answer contains gold: 1.000.
- Source document accuracy: 1.000.
- Token F1: 0.679.
- Negative/mixed status accuracy: 1.000.
- Error tags after the latest run: 64 `ok`, 0 remaining tagged errors.

Blind/stress result snapshot:
- Blind/stress examples: 25.
- Status accuracy: 0.920.
- Answer contains gold on expected answerable examples: 0.812.
- Source document accuracy on expected answerable examples: 0.812.
- Remaining error tags: 2 missed answers and 1 retrieval miss.

Inspect concrete failures with:

```bash
python scripts/show_eval_errors.py --limit 10
```

## Baseline Comparison

The method comparison is intentionally simple:

```bash
python scripts/compare_qa_baselines.py
```

Compared systems:
- Retrieval-only baseline: returns the top BM25 chunk as the answer.
- Full Q&A module: BM25 retrieval plus structured rules, extractive answer selection, confidence scoring, and fallback.

This comparison supports the report argument that retrieval-only is easy to implement and useful for source selection, while the full Q&A pipeline gives shorter answers and safer unsupported-question behavior.

Current baseline comparison:

| Metric | Retrieval-only | Full Q&A |
| --- | ---: | ---: |
| Source document accuracy | 0.540 | 1.000 |
| Answer contains gold | 0.600 | 1.000 |
| Token F1 | 0.052 | 0.679 |
| Fallback rate on positives | 0.000 | 0.000 |
| Retrieval Recall@1 | 0.540 | 0.540 |
| Retrieval Recall@3 | 0.800 | 0.800 |

Important interpretation: the retrieval metrics are the same because both systems use BM25 retrieval. The gain comes from the Q&A layer above retrieval: structured rules, narrative-document preference, extractive answer selection, and confidence/fallback logic.

Current ablation comparison:

| Metric | Retrieval-only | BM25+reader | BM25+reader+threshold | Full module |
| --- | ---: | ---: | ---: | ---: |
| Source document accuracy | 0.540 | 0.600 | 0.600 | 1.000 |
| Answer contains gold | 0.600 | 0.620 | 0.620 | 1.000 |
| Token F1 | 0.052 | 0.603 | 0.603 | 0.679 |
| Fallback rate | 0.000 | 0.060 | 0.060 | 0.000 |

## Demo Commands

Presentation demo:

```bash
python scripts/demo_qa.py --show-sources
```

Recommended Knowledge/Q&A workflow:

```bash
python scripts/rebuild_qa_kb.py --no-network-fetch --final-demo
python scripts/evaluate_qa.py
python scripts/show_eval_errors.py --limit 10
python scripts/compare_qa_baselines.py
python scripts/demo_qa.py --show-sources
```

## Known Limitations

- The module does not use live APIs for prices, tickets, opening hours, or weather.
- The evaluation set is intentionally small and project-specific, so the high score should be presented as local validation, not as general Q&A performance.
- Some unseen questions may still fail when BM25 ranks a related but wrong Bucharest document first.
- Some unseen questions may still fail when the extractive reader selects a plausible but incomplete span.
- The system is scoped to English Bucharest tourist-guide questions unless multilingual support is added later.
