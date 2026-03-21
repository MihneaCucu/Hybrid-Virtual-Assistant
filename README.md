# Hybrid Virtual Assistant

A hybrid Google-Assistant-style conversational agent built as a university project.

## Architecture

The assistant handles three types of input:
| Branch | Trigger | Owner |
|--------|---------|-------|
| **Command** | User gives an instruction (set alarm, book table) | NLU lead |
| **Information QA** | User asks a factual question | **Knowledge/QA lead** |
| **Fallback** | Request is unsupported | Dialogue Manager lead |

---

## Repository Structure

```
Hybrid-Virtual-Assistant/
├── qa/                    # Knowledge/QA module (public API: qa_module.py)
│   ├── qa_module.py       ← ONLY file teammates should import
│   ├── retrieval.py       ← BM25 indexing and retrieval (internal)
│   ├── reader.py          ← Extractive QA reader (internal)
│   └── fallback.py        ← Confidence scoring and fallback logic (internal)
├── scripts/
│   ├── collect_data.py    ← Wikipedia collection from JSON registry
│   ├── build_index.py     ← Chunking + BM25 index builder
│   ├── validate_kb_sources.py ← Source registry validator
│   └── evaluate_qa.py     ← QA evaluation metrics
├── kb/
│   ├── raw/               ← Raw Wikipedia text (one .txt per document)
│   ├── clean/             ← Cleaned documents (committed to git)
│   ├── chunks.jsonl        ← Chunked documents (gitignored, regenerated)
│   └── bm25_index.pkl     ← BM25 index (gitignored, regenerated)
├── data/
│   ├── kb_sources_bucharest.json ← QA source registry (single-city scope)
│   ├── test_set.json             ← Local positive QA eval set
│   ├── negative_test_set.jsonl   ← Local unanswerable/mixed eval set
│   └── qa_annotation_template_bucharest.jsonl ← Template for creating final eval set
├── docs/
│   ├── qa_implementation_plan.md ← QA lead implementation roadmap
│   └── qa_dataset_research.md    ← Curated dataset/source decisions
├── results/               ← Evaluation output (gitignored, regenerated)
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate source registry (single-city Bucharest)
python scripts/validate_kb_sources.py --path data/kb_sources_bucharest.json

# 4. Collect knowledge base documents (from source registry)
python scripts/collect_data.py

# Optional: use a custom targets file
python scripts/collect_data.py --targets data/kb_sources_bucharest.json --sentences 20

# Recommended when switching domains: clear old KB files first
python scripts/collect_data.py --targets data/kb_sources_bucharest.json --clear-output

# 5. Build BM25 index
python scripts/build_index.py

# 6. Verify the QA module works
python -c "
from qa.qa_module import load_qa_system, answer_question
load_qa_system()
print(answer_question('Where is the Romanian Athenaeum located?'))
"
```

---

## QA Module Integration

Teammates should only import from `qa.qa_module`:

```python
from qa.qa_module import load_qa_system, answer_question

# At startup (call ONCE):
load_qa_system()

# At query time (intent == "information_query"):
result = answer_question(query)

# result shape:
# {
#   "status":     str,   # answered | fallback | handoff
#   "reason_code":str | None,
#   "answer":     str | None,
#   "source_doc": str | None,
#   "sources":    list[dict],
#   "confidence": float, # 0.0 – 1.0
#   "fallback":   bool   # True = no reliable answer found
# }

if result["fallback"]:
    speak("I don't have information on that topic.")
else:
    speak(result["answer"])
```

---

## Evaluation

```bash
# Run module evaluation on local positive/negative sets:
python scripts/evaluate_qa.py

# Optional: override dataset paths
python scripts/evaluate_qa.py \
  --positive data/test_set.json \
  --negative data/negative_test_set.jsonl \
  --output results/qa_eval_summary.json
```

Results are saved to `results/`.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `wikipedia` | Wikipedia API for data collection |
| `rank_bm25` | BM25 sparse retrieval |
| `transformers` | Extractive QA reader model |
| `torch` | Backend for transformers |
| `nltk` | Tokenization utilities |
| `tqdm` | Progress bars |
