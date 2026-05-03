# Knowledge/Q&A Slide Outline

## Slide 1: Role In The Hybrid Assistant

Title: Knowledge/Q&A Branch

Core message:
- Handles factual information questions.
- Does not execute commands.
- Returns `answered`, `fallback`, or `handoff`.

Visual idea:
- Three-branch assistant diagram: command branch, Q&A branch, fallback branch.
- Highlight the Q&A branch.

Speaker note:
The Knowledge/Q&A module is the assistant's information branch. It answers from a bounded Bucharest tourist-guide knowledge base and refuses or hands off requests that do not belong there.

## Slide 2: Knowledge Base

Title: Bucharest Tourist-Guide KB

Core message:
- Curated city and landmark documents.
- Structured museums, places, transit records.
- Curated travel guidance.
- No open-web answering at runtime.

Speaker note:
The scope is deliberately bounded. This makes the module easier to evaluate and safer than a general chatbot.

## Slide 3: Architecture

Title: Retrieval-Based Q&A Pipeline

Pipeline:

`query -> structured rules -> BM25 retrieval -> extractive reader -> confidence check`

Core message:
- Rules handle deterministic assistant-style cases.
- BM25 finds candidate evidence.
- Extractive reader selects short spans.
- Confidence/fallback avoids unsupported claims.

Speaker note:
The design is explainable and reproducible. We did not use generative Q&A because hallucination would be harder to control in a small semester project.

## Slide 4: Evaluation

Title: Local Evaluation

Use table:

| Metric | Result |
| --- | ---: |
| Positive examples | 50 |
| Negative/mixed examples | 14 |
| Source document accuracy | 1.000 |
| Answer contains gold | 1.000 |
| Token F1 | 0.679 |
| Negative/mixed status accuracy | 1.000 |

Speaker note:
This is local validation for our domain, not a claim of open-domain Q&A performance.

## Slide 5: Baseline Comparison

Title: Retrieval-Only vs Full Q&A

Use table:

| Metric | Retrieval-only | Full Q&A |
| --- | ---: | ---: |
| Source document accuracy | 0.540 | 1.000 |
| Answer contains gold | 0.600 | 1.000 |
| Token F1 | 0.052 | 0.679 |

Speaker note:
BM25 is useful for retrieval, but returning the whole top chunk is not a good assistant answer. The full module adds answer extraction, source-aware selection, structured rules, and fallback behavior.

## Slide 6: Demo

Title: Live Demo

Run:

```bash
python scripts/hybrid_demo_cli.py --story --show-meta
```

Demo sequence:
- Simulated command branch.
- Museum-content factual answer.
- Nearby transport structured answer.
- Live price fallback.
- Mixed command/question handoff.
- Out-of-domain chit-chat fallback.

Speaker note:
The team-level demo shows all three branches. For the Knowledge/Q&A-only demo, use `python scripts/demo_qa.py --queries data/demo_queries_story.jsonl --show-sources`.

## Slide 7: Limitations And Extensions

Title: Limits And Future Work

Limitations:
- Static KB, no live prices/weather/opening hours.
- English-only questions.
- Small local test set.

Extensions:
- Romanian/English support.
- Dense retrieval or reranking.
- UI source citations.
- Larger manually annotated test set.

Speaker note:
The current system is complete for the project scope, but future work would expand coverage and language support.
