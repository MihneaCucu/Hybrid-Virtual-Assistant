# Knowledge/Q&A Defense Questions

## Why not use a generative chatbot?

Because the project goal is a Google-Assistant-style factual branch, not open-ended chit-chat. A generative model could produce fluent but unsupported answers. Our module is bounded to a Bucharest tourist-guide KB, returns source metadata, and falls back when evidence is missing.

## Why BM25 instead of dense embeddings?

BM25 is simple, explainable, reproducible, and works well enough for a small curated corpus. It also makes the method comparison easier to explain. Dense retrieval is a reasonable extension, but it would add dependency and tuning cost that is not necessary for the semester scope.

## Why combine rules with retrieval?

Some assistant-style queries are structured, not purely textual. For example, nearest metro station, nearby STB stops, exact addresses, and mixed command/question handoff are more reliable with deterministic rules over structured records. Retrieval and the extractive reader are better for narrative facts such as dates, descriptions, and symbolic meaning.

## Why is retrieval Recall@1 lower than final source accuracy?

The final Q&A module does more than return the top BM25 result. It can inspect multiple retrieved chunks, prefer narrative documents for definition/content questions, use structured rules, and select better answer spans. The baseline comparison shows this clearly: retrieval-only source accuracy is 0.540, while full Q&A source accuracy is 1.000 on the local test set.

## What does fallback mean?

Fallback means the module should not answer from the current KB. This happens for out-of-domain questions, live facts, prices, ambiguous requests, or low-confidence retrieval/reader results.

## What does handoff mean?

Handoff means the request mixes Q&A with a command, so the dialogue manager or router should handle it. Example: "Set a reminder for tomorrow and tell me where University Square is." The Q&A module does not execute reminders.

## Is the evaluation too small?

Yes, it is small by research standards. It is appropriate for a semester implementation because it covers the planned demo domain: landmarks, museums, addresses, transit, travel guidance, unsupported questions, and mixed command/Q&A requests. To reduce overfitting risk, the project separates the tuned development set from a blind/stress set with paraphrases and unsupported examples. The report should present the score as local validation, not general open-domain performance.

## Why are blind/stress results lower than the development results?

The development set was used during implementation, so it measures whether the planned behavior works. The blind/stress set contains paraphrases, ambiguous wording, and unsupported live-data requests that were not tuned as heavily. Lower blind/stress performance is expected and is useful for the limitations section.

## What is the main limitation?

The system cannot answer live or changing information such as ticket prices, weather, current opening hours, availability, or bookings. This is a deliberate safety choice because the runtime KB is static.

## What would you add with more time?

The best next extensions are Romanian/English support, dense retrieval or reranking, richer UI source citations, and a larger manually annotated evaluation set.
