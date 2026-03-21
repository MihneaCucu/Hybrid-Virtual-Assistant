from __future__ import annotations

import os

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "deepset/minilm-uncased-squad2"

_tokenizer = None
_model = None


def load_reader() -> None:
    global _tokenizer, _model
    if _model is not None:
        return

    os.environ["TORCH_DYNAMO_DISABLE"] = "1"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    _model.eval()


def extract_answer(question: str, context: str) -> dict:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Reader not loaded. Call load_reader() first.")

    inputs = _tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    with torch.no_grad():
        outputs = _model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    start_prob = torch.softmax(start_logits, dim=0)[start_idx]
    end_prob = torch.softmax(end_logits, dim=0)[end_idx]
    score = float(start_prob * end_prob)

    if start_idx == 0 or end_idx < start_idx:
        return {"answer": None, "score": score, "start": 0, "end": 0}

    start_char = offset_mapping[start_idx][0].item()
    end_char = offset_mapping[end_idx][1].item()
    answer_text = context[start_char:end_char].strip()

    if not answer_text:
        return {"answer": None, "score": score, "start": 0, "end": 0}

    return {
        "answer": answer_text,
        "score": score,
        "start": start_char,
        "end": end_char,
    }
