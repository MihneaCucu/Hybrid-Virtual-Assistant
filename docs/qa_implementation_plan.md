# Knowledge/Q&A Implementation Summary (Bucharest Tourist Guide)

## Goal And Current State
The Knowledge/Q&A module delivers the information branch of the hybrid assistant:
- User asks a factual Bucharest tourist-guide question.
- The module retrieves or selects grounded evidence from the local KB.
- The module returns a short answer, source metadata, and confidence.
- Unsupported or mixed requests return `fallback` or `handoff`.
- No open-web answering is used at runtime.

The implementation is complete enough for the final report and demo. Remaining work should focus on presentation, report writing, and optional stress testing rather than another architecture change.

## Final Scope
Primary scope: one city (Bucharest).

Data categories:
- City overview and landmarks.
- Museums and cultural places.
- Transport facts (metro and transit stops/routes).
- Selected places, addresses, nearest metro links, and nearby transport.
- Curated travel guidance for season, budget, and trip length.

## Data Sources
1. `data/kb_sources_bucharest.json` - central source registry.
2. Wikipedia summaries (`wikipedia_targets`) - narrative text for extractive QA.
3. TPBI GTFS feed - structured transport entities (stops/routes/schedules metadata).
4. Romania museums dataset - structured museum entities (filter Bucharest).
5. OSM/Nominatim records - selected place addresses and coordinates.
6. `data/bucharest_travel_guidance.json` - curated travel guidance facts.

## Implemented Components
- Ingestion scripts for Wikipedia, museums, transit, OSM places, restaurants, and coffee shops.
- Structured-to-text conversion through `scripts/build_structured_text_docs.py`.
- One-command rebuild workflow through `scripts/rebuild_qa_kb.py`.
- BM25 index construction through `scripts/build_index.py`.
- Public Q&A API in `qa/qa_module.py`.
- Internal retrieval, reader, and fallback modules in `qa/retrieval.py`, `qa/reader.py`, and `qa/fallback.py`.
- Evaluation and analysis tooling through `scripts/evaluate_qa.py`, `scripts/show_eval_errors.py`, and `scripts/compare_qa_baselines.py`.
- Demo tooling through `scripts/demo_qa.py` and `scripts/chat_cli.py`.

## Final Architecture
The selected design is closed-domain retrieval-based Q&A:

`question -> structured rule handlers -> BM25 retrieval -> extractive reader -> confidence/fallback`

Rule handlers cover high-value deterministic cases:
- exact addresses
- nearest metro station
- nearby transport
- curated travel guidance
- selected museum-content questions
- mixed command/question handoff

BM25 plus the extractive reader covers narrative factual questions from the curated tourist-guide documents.

## Evaluation Artifacts
- `data/test_set.json`: 50 positive answerable Q&A examples.
- `data/negative_test_set.jsonl`: 14 unsupported or mixed examples.
- `results/qa_eval_summary.json`: aggregate metrics.
- `results/qa_eval_details.jsonl`: per-example predictions and error tags.
- `results/qa_baseline_comparison.json`: retrieval-only vs full Q&A comparison.

Current result snapshot:
- Positive examples: 50.
- Negative/mixed examples: 14.
- Source document accuracy: 1.000.
- Answer contains gold: 1.000.
- Token F1: 0.679.
- Negative/mixed status accuracy: 1.000.
- Error tags: 64 `ok`, 0 remaining tagged errors.

## Success Criteria
- Source document accuracy target: `>= 0.80`; current: `1.000`.
- Correct fallback/status target on unanswerables and mixed requests: `>= 0.85`; current: `1.000`.
- All answered examples include source metadata.
- Demo covers `answered`, `fallback`, and `handoff`.

## Team Interface Contract
`qa.answer_question(query) -> dict` must return:
- `status`: `answered | fallback | handoff`
- `reason_code`: machine-readable reason
- `answer`, `source_doc`, `sources`, `confidence`, `fallback`

Router behavior:
- `status=answered` -> speak answer.
- `status=fallback` -> fallback message.
- `status=handoff` -> split or reroute to command branch.

Teammates should import only:

```python
from qa.qa_module import load_qa_system, answer_question
```
