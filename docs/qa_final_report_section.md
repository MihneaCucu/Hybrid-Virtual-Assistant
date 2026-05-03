# Knowledge/Q&A Module: Near-Final Report Section

## Role In The Hybrid Assistant

The Knowledge/Q&A module is responsible for the assistant's information-question branch. It does not handle general conversation or command execution. When the router identifies a factual question, the module attempts to answer from a bounded Bucharest tourist-guide knowledge base. If the question is outside the knowledge base or below confidence thresholds, the module returns a fallback response. If a request mixes a command with a factual question, it returns a handoff signal for the dialogue manager.

This separates responsibilities clearly:
- NLU/command branch: detects command intents and slots.
- Dialogue manager/router: decides whether a request is a command, information question, or unsupported request.
- Knowledge/Q&A module: retrieves evidence, extracts or selects an answer, and reports confidence/source metadata.
- App/integration layer: displays or speaks the answer and handles command execution or simulation.

## Problem Framing

The module is framed as closed-domain retrieval-based Question Answering. This is more appropriate than open-domain or generative Q&A for the project because the assistant must behave like a practical Google-Assistant-style system over a known demo domain, not like a general chatbot.

The final design combines:
- BM25 retrieval for explainable document matching.
- Rule-based handlers for structured tourist-guide facts.
- Extractive Q&A for short answers from retrieved passages.
- Confidence thresholds and fallback logic to avoid unsupported answers.

This design is bounded, reproducible, and easy to explain in a semester report. A generative model over the corpus was not selected because it would add hallucination risk and make evaluation harder.

## Dataset Construction

The knowledge base is a Bucharest tourist guide. It contains:
- Narrative landmark and city documents from curated source targets.
- Structured museum records filtered to Bucharest.
- Structured transit data for routes and stops.
- Structured OSM/Nominatim place records for selected landmarks.
- Curated travel guidance for visit season, budget, and trip length.

The runtime system does not query the open web. The KB is rebuilt locally, chunked into passages, and indexed with BM25. Structured records are also converted into short text documents so they can be retrieved alongside narrative documents. The final report/demo profile is generated with:

```bash
python scripts/rebuild_qa_kb.py --no-network-fetch --final-demo
```

The data provenance file separates runtime sources from methodology references. SQuAD is used only as inspiration for extractive Q&A and EM/F1-style metrics, not as runtime data.

## Preprocessing And Chunking

Narrative documents are stored in `kb/clean/`. Structured records are normalized into JSONL files under `kb/structured/`, then converted into factual text snippets. The index builder chunks documents into passages and serializes a BM25 index. This keeps the runtime pipeline reproducible and avoids hidden external calls during answering.

## Architecture

The runtime pipeline is:

`query -> structured rules -> BM25 retrieval -> extractive reader -> confidence check -> answer/fallback/handoff`

Structured rules are used for cases where retrieval plus span extraction is unnecessary or less reliable, such as nearest metro station, nearby transport, exact addresses, curated travel guidance, and selected museum-content questions. BM25 retrieval and the extractive reader handle narrative questions such as construction dates, symbolic meaning, definitions, and landmark facts.

The response contract is fixed: every result contains `status`, `reason_code`, `answer`, `source_doc`, `sources`, `confidence`, and `fallback`. This makes integration predictable for the router and allows evaluation scripts to inspect source grounding.

The public interface is:

```python
from qa.qa_module import load_qa_system, answer_question
```

The output dictionary contains `status`, `reason_code`, `answer`, `source_doc`, `sources`, `confidence`, and `fallback`. This lets the router decide whether to speak an answer, use the fallback branch, or hand the request to another module.

## Evaluation

The evaluation uses a local test set designed for the project domain:
- 50 positive answerable questions.
- 14 negative or mixed examples.
- Positive examples cover landmarks, museums, addresses, nearest metro, transit, and travel guidance.
- Negative examples cover out-of-domain questions, live facts/prices, and mixed command-plus-question requests.

Metrics:
- Source document accuracy: whether the answer was grounded in the expected document.
- Answer contains gold: whether the predicted answer contains an accepted answer string.
- Token F1: overlap between predicted and accepted answers.
- Retrieval Recall@1 and Recall@3: whether BM25 retrieved the gold document near the top.
- Negative/mixed status accuracy: whether unsupported and mixed requests returned the expected status.

Current results:

| Metric | Result |
| --- | ---: |
| Positive examples | 50 |
| Negative/mixed examples | 14 |
| Source document accuracy | 1.000 |
| Answer contains gold | 1.000 |
| Token F1 | 0.679 |
| Retrieval Recall@1 | 0.540 |
| Retrieval Recall@3 | 0.800 |
| Negative/mixed status accuracy | 1.000 |

The relatively lower retrieval Recall@1 shows why answer selection and structured rules are useful: BM25 alone does not always rank the final source first, but the full Q&A pipeline can still select a correct grounded answer.

The project also includes a separate blind/stress set with paraphrases, unsupported live-data requests, out-of-domain questions, ambiguous wording, and mixed command-plus-Q&A requests. These results should be reported separately from the development set because the development set has been used during implementation.

Blind/stress result snapshot:

| Metric | Result |
| --- | ---: |
| Examples | 25 |
| Expected answerable examples | 16 |
| Status accuracy | 0.920 |
| Answer contains gold | 0.812 |
| Source document accuracy | 0.812 |

The remaining blind/stress failures are useful evidence for limitations: two paraphrased answerable questions still fall back, and one underspecified "history museum" question retrieves the wrong museum.

## Baseline Comparison

The main baseline is retrieval-only Q&A: return the top BM25 chunk as the answer. This baseline is simple and explainable, but it often returns long or partially irrelevant passages.

| Metric | Retrieval-only | Full Q&A |
| --- | ---: | ---: |
| Source document accuracy | 0.540 | 1.000 |
| Answer contains gold | 0.600 | 1.000 |
| Token F1 | 0.052 | 0.679 |
| Fallback rate on positives | 0.000 | 0.000 |
| Retrieval Recall@1 | 0.540 | 0.540 |
| Retrieval Recall@3 | 0.800 | 0.800 |

The full Q&A module improves answer quality without changing the underlying retrieval recall because it adds structured rules, better answer extraction, and confidence-aware fallback on top of the same BM25 retriever.

An additional ablation script separates the contribution of retrieval-only answering, BM25 plus extractive reading, reader thresholding, and the full module. This supports a stronger methodological comparison in the final report.

Current ablation result:

| Metric | Retrieval-only | BM25+reader | BM25+reader+threshold | Full module |
| --- | ---: | ---: | ---: | ---: |
| Source document accuracy | 0.540 | 0.600 | 0.600 | 1.000 |
| Answer contains gold | 0.600 | 0.620 | 0.620 | 1.000 |
| Token F1 | 0.052 | 0.603 | 0.603 | 0.679 |
| Fallback rate | 0.000 | 0.060 | 0.060 | 0.000 |

This shows that the reader improves answer compactness substantially, while the full module's structured rules and source-aware selection drive the strongest grounding result.

## Safety And Fallback

The module avoids hallucination by answering only from local evidence or deterministic structured records. Unsupported requests return fallback responses. Examples include live prices, current opening hours, weather, and general world knowledge. Mixed command-plus-question requests return `handoff`, allowing the dialogue manager to split or reroute them.

Fallback examples:
- "How much is a metro ticket in Bucharest?" -> fallback with official transit links.
- "Who is the president of France?" -> fallback because it is outside the Bucharest KB.
- "Set a reminder for tomorrow and tell me where University Square is." -> handoff because it mixes command and Q&A behavior.

The confidence score is an engineering signal, not a calibrated probability. It is used to decide whether retrieved evidence and reader spans are strong enough to return an answer.

## Error Analysis

Per-example evaluation details are written to `results/qa_eval_details.jsonl`. The error-inspection script groups failures into categories such as retrieval miss, wrong source document, reader span error, false fallback, missed handoff, and false answer instead of fallback. This gives the report a concrete error-analysis method even when the current development set has no remaining tagged errors.

## Threats To Validity

The main threat is overfitting to a small local evaluation set. The current development set is useful for validating planned behavior, but it is not enough to claim general Q&A performance. A second blind/stress set is therefore reported separately. Another limitation is that the corpus is English-focused and static; live facts and Romanian-language paraphrases are not fully covered.

## Limitations

The current evaluation is small and domain-specific, so high scores should be interpreted as successful validation for the project demo rather than general Q&A performance. The module does not support live information, booking execution, Romanian-language Q&A, or broad open-domain questions. Future extensions could add multilingual support, dense retrieval, reranking, richer source citation in a graphical interface, and a larger independently annotated test set.
