"""
Integration layer: NLU (Stefan) -> router -> QA (Mihnea) | DM (George)
All three module imports are guarded so the app still runs when branches
haven't been merged yet (stubs return placeholder output).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

# paths
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# intent routing table
DM_INTENTS = {
    "takeaway_query", "takeaway_order",
    "transport_taxi", "transport_query", "transport_ticket",
    "calendar_set", "calendar_query",
    "recommendation_locations",
    "general_greet", "general_quirky",
}
QA_INTENTS = {
    "qa_factoid", "qa_definition", "qa_stock", "qa_currency", "qa_maths",
}

# NLU (Stefan)
_nlu_model = None
_nlu_tokenizer = None
_nlu_id2label: dict = {}
_nlu_intent_names: list = []
_nlu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_nlu_ready = False

def _load_nlu():
    global _nlu_model, _nlu_tokenizer, _nlu_id2label, _nlu_intent_names, _nlu_ready
    try:
        import NLU_model as nlu_mod
        from transformers import AutoTokenizer as _AT

        model_dir = os.path.join(ROOT, "BiLSTM_model_10epochs")
        if not os.path.isdir(model_dir):
            model_dir = os.path.join(ROOT, "saved_model")

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(model_dir, "id2label.json")) as f:
            _nlu_id2label = json.load(f)
        with open(os.path.join(model_dir, "intent_names.json")) as f:
            _nlu_intent_names = json.load(f)

        _nlu_model = nlu_mod.JointModel(cfg["num_intents"], cfg["num_slots"])
        _nlu_model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pt"), map_location=_nlu_device)
        )
        _nlu_model.to(_nlu_device).eval()
        _nlu_tokenizer = _AT.from_pretrained(model_dir)
        _nlu_ready = True
        print("[pipeline] NLU loaded.")
    except Exception as e:
        print(f"[pipeline] NLU not available: {e}")


def _run_nlu(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (intent_label, [(slot_name, slot_value), ...])."""
    if not _nlu_ready:
        return "general_greet", []

    inputs = _nlu_tokenizer(
        text.lower(), return_tensors="pt", truncation=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(_nlu_device)
    attention_mask = inputs["attention_mask"].to(_nlu_device)

    with torch.no_grad():
        _, slot_preds, intent_logits = _nlu_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

    intent_label = _nlu_intent_names[torch.argmax(intent_logits, dim=1).item()]
    tokens = _nlu_tokenizer.convert_ids_to_tokens(input_ids[0])[1:-1]
    raw_labels = slot_preds[0][1:-1]
    slot_labels = [
        _nlu_id2label[str(p)] if str(p) in _nlu_id2label else _nlu_id2label.get(p, "O")
        for p in raw_labels
    ]

    slots: list[tuple[str, str]] = []
    cur_slot, cur_toks = None, []
    for tok, lbl in zip(tokens, slot_labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if lbl.startswith("B-"):
            if cur_slot:
                slots.append((cur_slot, _nlu_tokenizer.convert_tokens_to_string(cur_toks).strip()))
            cur_slot, cur_toks = lbl[2:], [tok]
        elif lbl.startswith("I-") and cur_slot:
            cur_toks.append(tok)
        else:
            if cur_slot:
                slots.append((cur_slot, _nlu_tokenizer.convert_tokens_to_string(cur_toks).strip()))
            cur_slot, cur_toks = None, []
    if cur_slot:
        slots.append((cur_slot, _nlu_tokenizer.convert_tokens_to_string(cur_toks).strip()))

    return intent_label, slots


# Dialogue Manager / NLG (George)
_dm = None
_dm_ready = False

def _load_dm():
    global _dm, _dm_ready
    try:
        from manager import DialogueManager
        from data.vocab import Vocabulary
        from model.seq2seq import Encoder, Decoder, Seq2Seq
        from model.train import EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

        data_dir = os.path.join(ROOT, "data", "processed")
        ckpt = os.path.join(ROOT, "checkpoints", "best_model.pt")

        enc_vocab = Vocabulary()
        dec_vocab = Vocabulary()
        enc_vocab.load(os.path.join(data_dir, "enc_vocab.json"))
        dec_vocab.load(os.path.join(data_dir, "dec_vocab.json"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = Encoder(len(enc_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
        decoder = Decoder(len(dec_vocab), EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, DROPOUT)

        from model.seq2seq import Seq2Seq
        model = Seq2Seq(encoder, decoder, device).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        _dm = DialogueManager(model=model, enc_vocab=enc_vocab, dec_vocab=dec_vocab, device=device)
        _dm_ready = True
        print("[pipeline] Dialogue Manager loaded.")
    except Exception as e:
        print(f"[pipeline] Dialogue Manager not available: {e}")


def _run_dm(user_text: str, intent: str, slots: list[tuple[str, str]]) -> str:
    if not _dm_ready:
        return f"(DM stub) Intent: {intent}. Slots: {dict(slots) if slots else 'none'}."
    kb = {k: v for k, v in slots} if slots else {}
    return _dm.process_turn(user_text=user_text, intent=intent, kb_results=kb)


def reset_dm():
    if _dm_ready and _dm is not None:
        _dm.reset()


# QA module (Mihnea)
_qa_ready = False
_looks_like_question_fn = None

def _load_qa():
    global _qa_ready, _looks_like_question_fn
    try:
        chunks_path = os.path.join(ROOT, "kb", "chunks.jsonl")
        if not os.path.exists(chunks_path):
            print("[pipeline] QA index missing, building now...")
            import scripts.build_index as _bi
            _bi.main()
        from qa.qa_module import load_qa_system
        from qa.routing import looks_like_question
        load_qa_system()
        _looks_like_question_fn = looks_like_question
        _qa_ready = True
        print("[pipeline] QA system loaded.")
    except Exception as e:
        print(f"[pipeline] QA system not available: {e}")


def _run_qa(text: str) -> dict[str, Any]:
    if not _qa_ready:
        return {
            "status": "fallback",
            "answer": "(QA stub) Knowledge base not available yet.",
            "confidence": 0.0,
            "source_doc": None,
        }
    from qa.qa_module import answer_question
    return answer_question(text)


# public API
def load_all():
    """Call once at startup to load all models."""
    _load_nlu()
    _load_qa()
    _load_dm()


def process(user_text: str) -> dict[str, Any]:
    """
    Run the full pipeline on one user turn.

    Returns a dict with keys:
      response   str   - text to show the user
      intent     str   - MASSIVE intent label
      slots      list  - [(slot_name, value), ...]
      route      str   - "dm" | "qa" | "fallback"
      confidence float - QA confidence when route=="qa", else 1.0
      source_doc str | None
    """
    text = user_text.strip()
    if not text:
        return {
            "response": "Please type something.",
            "intent": "", "slots": [], "route": "fallback",
            "confidence": 0.0, "source_doc": None,
        }

    intent, slots = _run_nlu(text)

    # routing - NLU intent takes priority over heuristics
    if intent in DM_INTENTS:
        response = _run_dm(text, intent, slots)
        return {
            "response": response,
            "intent": intent, "slots": slots,
            "route": "dm", "confidence": 1.0, "source_doc": None,
        }

    is_question = (
        intent in QA_INTENTS
        or (_looks_like_question_fn is not None and _looks_like_question_fn(text))
    )

    if is_question:
        qa_result = _run_qa(text)
        answer = qa_result.get("answer") or "I couldn't find a reliable answer for that."
        if qa_result.get("fallback", False) or qa_result.get("status") != "answered":
            answer = "I don't have information on that topic in my knowledge base."
        return {
            "response": answer,
            "intent": intent, "slots": slots,
            "route": "qa",
            "confidence": qa_result.get("confidence", 0.0),
            "source_doc": qa_result.get("source_doc"),
        }

    # fallback
    return {
        "response": "I'm not sure how to help with that. I can answer questions about Bucharest or help with tasks like bookings and transport.",
        "intent": intent, "slots": slots,
        "route": "fallback", "confidence": 0.0, "source_doc": None,
    }
