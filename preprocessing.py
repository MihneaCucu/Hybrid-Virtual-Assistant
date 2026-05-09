from datasets import load_dataset
from transformers import AutoTokenizer
import re

def validate_labels(labels, id2label):
    prev = "O"
    for label_id in labels:
        if label_id == -100:
            continue
        label = id2label[label_id]
        if label.startswith("I-"):
            if not (prev.startswith("B-") or prev.startswith("I-")):
                return False
            if prev[2:] != label[2:]:
                return False
        prev = label
    return True
def is_span_truncated(offsets, spans):
    valid_offsets = [end for start, end in offsets if end > 0]
    max_token_end = max(valid_offsets) if valid_offsets else 0
    for span in spans:
        if span["end"] > max_token_end:
            return True
    return False
def parse_annot_utt(example):
    text = example["annot_utt"].lower()
    clean_text = ""
    spans = []
    idx = 0
    clean_idx = 0

    pattern = re.compile(r"\[(.*?) : (.*?)\]")

    for match in pattern.finditer(text):
        start, end = match.span()
        slot_name, slot_value = match.groups()

        prefix = text[idx:start]
        clean_text += prefix
        clean_idx += len(prefix)

        value_start = clean_idx
        clean_text += slot_value
        value_end = clean_idx + len(slot_value)

        spans.append({
            "start": value_start,
            "end": value_end,
            "label": slot_name.strip()
        })

        clean_idx = value_end
        idx = end

    suffix = text[idx:]
    clean_text += suffix

    return {
        "clean_text": clean_text,
        "spans": spans
    }
def tokenize_and_align(example, tokenizer, label2id, id2label):
    MAX_LEN = 128
    tokenized = tokenizer(
        example["clean_text"],
        truncation=True,
        max_length=MAX_LEN,
        return_offsets_mapping=True
    )

    offsets = tokenized["offset_mapping"]
    spans = example["spans"]

    if is_span_truncated(offsets, spans):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "drop": True
        }

    labels = []

    for start, end in offsets:
        if start == end:
            labels.append(label2id["O"])
            continue

        assigned_label = "O"

        for span in spans:
            span_start = span["start"]
            span_end = span["end"]
            slot_name = span["label"]

            if not (end <= span_start or start >= span_end):
                if start == span_start:
                    assigned_label = "B-" + slot_name
                else:
                    assigned_label = "I-" + slot_name
                break

        labels.append(label2id[assigned_label])

    tokenized["labels"] = labels
    tokenized.pop("offset_mapping")

    if not validate_labels(labels, id2label):
        tokenized["drop"] = True
    else:
        tokenized["drop"] = False

    return tokenized
def build_label_map(unique_slots):
    labels = ["O"]
    for slot in sorted(unique_slots):
        labels.append("B-" + slot)
        labels.append("I-" + slot)

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label

def get_attributes():
    dataset = load_dataset("AmazonScience/massive")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset.filter(lambda x: x["locale"] == "en-US")
    intent_names = dataset["train"].features["intent"].names
    dataset = dataset.map(parse_annot_utt)
    unique_slots = set(
        span["label"]
        for example in dataset["train"]["spans"]
        for span in example
    )
    # label <-> id maps
    label2id, id2label = build_label_map(unique_slots)
    return dataset, tokenizer, intent_names, unique_slots, label2id, id2label

def preprocess_dataset():
    dataset, tokenizer, intent_names, unique_slots, label2id, id2label = get_attributes()
    before = len(dataset["train"])
    dataset = dataset.map(tokenize_and_align,
                          fn_kwargs={
                              "tokenizer": tokenizer,
                              "label2id": label2id,
                              "id2label": id2label
                          }
                          )
    dataset = dataset.filter(lambda x: not x.get("drop", False))
    after = len(dataset["train"])
    print(f"Dropped {before - after} examples")
    #cleaning up before training step
    dataset = dataset.remove_columns([
        col for col in dataset["train"].column_names
        if col not in ["input_ids", "attention_mask", "labels", "intent"]
    ])
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "intent"]
    )
    dataset.save_to_disk("massive_processed")
    return dataset, tokenizer, intent_names, unique_slots, label2id, id2label
