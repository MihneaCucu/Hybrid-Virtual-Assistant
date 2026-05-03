# Knowledge/Q&A Demo Script

## Setup

Use the clean chat demo when presenting live:

```bash
python scripts/chat_cli.py
```

Use metadata only when explaining internals:

```bash
python scripts/chat_cli.py --show-meta
```

Use the scripted source-aware demo if you want a deterministic run:

```bash
python scripts/demo_qa.py --show-sources
```

Use the shorter story demo for the final presentation:

```bash
python scripts/demo_qa.py --queries data/demo_queries_story.jsonl --show-sources
```

Use the hybrid story demo when the team wants to show all three assistant branches in one terminal:

```bash
python scripts/hybrid_demo_cli.py --story --show-meta
```

## Recommended Live Demo Flow

### 1. Narrative factual Q&A

Type:

```text
What collections does the National Museum of Art of Romania feature?
```

Expected behavior: answered from the narrative museum document.

Speaking point: this shows extractive Q&A over the local Bucharest KB, not a hardcoded FAQ answer.

### 2. Museum-content Q&A

Type:

```text
What does the National History Museum of Romania contain?
```

Expected behavior: answered with historical artifacts from the correct source document.

Speaking point: this shows that the system can answer content questions about museums, not only addresses.

### 3. Landmark meaning

Type:

```text
What does Arcul de Triumf symbolize?
```

Expected behavior: answered with Romania's victory in the First World War.

Speaking point: this shows a short extracted fact from a landmark document.

### 4. Structured local assistant behavior

Type:

```text
Show metro as well as STB stations near Romanian Athenaeum
```

Expected behavior: answered with nearby metro and STB stops.

Speaking point: this is closer to Google Assistant/Alexa behavior because it uses structured local records, not just passage retrieval.

### 5. Safe fallback

Type:

```text
How much is a metro ticket in Bucharest?
```

Expected behavior: fallback with official transit links.

Speaking point: the system refuses live price claims because prices are not guaranteed current in the static KB.

### 6. Mixed command plus factual request

Type:

```text
Set a reminder for tomorrow and tell me where University Square is.
```

Expected behavior: handoff to the command/dialogue branch.

Speaking point: the Q&A module detects that the request mixes a command and a factual question, so it does not try to handle everything itself.

## Backup Questions

Use these if one answer is too short during the live demo:

```text
What is King Michael I Park?
On which river does Bucharest stand?
When was Stavropoleos Monastery built?
What style is the Palace of the Parliament built in?
What metro station is the National Museum of Art of Romania at?
What season is best to visit Bucharest?
How many days should I stay in Bucharest?
```

## What To Say About Results

Use this compact summary:

```text
For my Knowledge/Q&A module, I built a bounded Bucharest tourist-guide knowledge base. The final design uses structured rules for deterministic assistant-style queries, BM25 retrieval for finding relevant documents, and an extractive reader for short factual answers. The module returns source metadata and confidence, and it falls back or hands off when the request is unsupported or mixed with a command.
```

Use this metric summary:

```text
On the local evaluation set, the module answered 50/50 positive examples with the expected source and handled 14/14 unsupported or mixed examples with the expected status. Compared with a retrieval-only baseline, the full Q&A module improved source document accuracy from 0.54 to 1.00 and answer-containing-gold accuracy from 0.60 to 1.00.
```
