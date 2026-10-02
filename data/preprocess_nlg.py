"""Preprocess MultiWOZ 2.4 into NLG training pairs.

Input:  [USR] ... [SYS] ... (last 4 turns) [INTENT] <massive_intent> [KB] key=value ...
Target: system response text (natural language)

Dialogue acts keys in MultiWOZ are 1-based, so system turn at 0-based log
index i is looked up with key str(i+1).
"""

import json
import os
import random
import zipfile

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

random.seed(42)

TAXI_COLORS = ["black", "white", "red", "yellow", "blue", "grey"]
TAXI_TYPES = ["toyota", "skoda", "bmw", "honda", "ford", "audi",
              "lexus", "volvo", "volkswagen", "tesla"]

EMPTY_VALUES = {"", "not mentioned"}


def resolve_intent(act_names, belief_metadata):
    """Map a set of MultiWOZ act names to a MASSIVE intent. Returns None to skip."""
    if act_names & {"general-greet", "general-welcome"}:
        return "general_greet"
    if act_names & {"general-reqmore", "general-bye"}:
        return "general_quirky"

    if "Taxi-Request" in act_names:
        return "transport_taxi"
    if "Taxi-Inform" in act_names:
        taxi_semi = belief_metadata.get("taxi", {}).get("semi", {})
        filled = any(v for v in taxi_semi.values() if v not in EMPTY_VALUES)
        return "transport_taxi" if filled else "transport_query"

    if act_names & {"Train-OfferBook", "Train-OfferBooked"}:
        return "transport_ticket"
    if "Train-Inform" in act_names:
        train_semi = belief_metadata.get("train", {}).get("semi", {})
        filled = any(v for v in train_semi.values() if v not in EMPTY_VALUES)
        return "transport_ticket" if filled else "qa_factoid"

    has_booking_book = "Booking-Book" in act_names
    has_restaurant = bool(act_names & {"Restaurant-Inform", "Restaurant-Recommend"})
    has_booking_inform = "Booking-Inform" in act_names
    has_booking_request = "Booking-Request" in act_names
    has_booking_nobook = "Booking-NoBook" in act_names

    if has_booking_book and has_restaurant:
        return "takeaway_order"
    if has_booking_book:
        return "calendar_set"
    if has_booking_inform and (has_booking_request or has_booking_nobook):
        return "calendar_query"
    if has_booking_inform:
        return "calendar_set"
    if has_booking_request or has_booking_nobook:
        return "calendar_query"

    if has_restaurant:
        return "takeaway_query"
    if "Attraction-Recommend" in act_names:
        return "recommendation_locations"
    if "Attraction-Inform" in act_names:
        return "qa_factoid"

    return None


def extract_kb(intent, belief_metadata):
    """Extract KB facts as a flat key=value string for the given intent."""
    if intent in ("general_greet", "general_quirky"):
        return "none"

    parts = []

    if intent == "recommendation_locations":
        semi = belief_metadata.get("attraction", {}).get("semi", {})
        for slot in ["name", "type", "area"]:
            v = semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")

    elif intent == "qa_factoid":
        attr_semi = belief_metadata.get("attraction", {}).get("semi", {})
        for slot in ["name", "type", "area"]:
            v = attr_semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")
        train_semi = belief_metadata.get("train", {}).get("semi", {})
        for slot in ["departure", "destination", "day"]:
            v = train_semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")

    elif intent == "takeaway_query":
        semi = belief_metadata.get("restaurant", {}).get("semi", {})
        for slot in ["name", "food", "area", "pricerange"]:
            v = semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")

    elif intent == "takeaway_order":
        semi = belief_metadata.get("restaurant", {}).get("semi", {})
        book = belief_metadata.get("restaurant", {}).get("book", {})
        for slot in ["name", "food", "area", "pricerange"]:
            v = semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")
        for slot in ["time", "day", "people"]:
            v = book.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")

    elif intent in ("calendar_set", "calendar_query"):
        semi = belief_metadata.get("restaurant", {}).get("semi", {})
        book = belief_metadata.get("restaurant", {}).get("book", {})
        name = semi.get("name", "")
        if name not in EMPTY_VALUES:
            parts.append(f"name={name}")
        for slot in ["time", "day", "people"]:
            v = book.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")
        booked = book.get("booked", [])
        if booked and isinstance(booked[0], dict):
            ref = booked[0].get("reference", "")
            if ref:
                parts.append(f"reference={ref}")

    elif intent in ("transport_taxi", "transport_query"):
        semi = belief_metadata.get("taxi", {}).get("semi", {})
        for slot in ["departure", "destination", "leaveAt", "arriveBy"]:
            v = semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")
        parts.append(f"color={random.choice(TAXI_COLORS)}")
        parts.append(f"type={random.choice(TAXI_TYPES)}")

    elif intent == "transport_ticket":
        semi = belief_metadata.get("train", {}).get("semi", {})
        book = belief_metadata.get("train", {}).get("book", {})
        for slot in ["departure", "destination", "day", "leaveAt", "arriveBy"]:
            v = semi.get(slot, "")
            if v not in EMPTY_VALUES:
                parts.append(f"{slot}={v}")
        booked = book.get("booked", [])
        if booked and isinstance(booked[0], dict):
            price = booked[0].get("price", "")
            if price:
                parts.append(f"price={price}")
            train_id = booked[0].get("trainID", "")
            if train_id:
                parts.append(f"trainID={train_id}")

    return " ".join(parts) if parts else "none"


INTENT_CAP = {
    "general_quirky": 1000,
    "general_greet": 1000,
}


def build_nlg_pairs(data, dialogue_acts, dial_ids):
    pairs = []
    intent_counts = {}

    for dial_id in dial_ids:
        dialogue = data[dial_id]
        log = dialogue["log"]
        acts_key = dial_id.replace(".json", "")
        dial_acts = dialogue_acts.get(acts_key, {})

        history_turns = []

        for i, turn in enumerate(log):
            utterance = turn["text"].strip()

            if i % 2 == 0:
                history_turns.append(f"[USR] {utterance}")
                continue

            # System turn — look up acts with 1-based key
            turn_acts = dial_acts.get(str(i + 1), {})
            if not isinstance(turn_acts, dict) or not turn_acts:
                history_turns.append(f"[SYS] {utterance}")
                continue

            belief_metadata = turn.get("metadata", {})
            intent = resolve_intent(set(turn_acts.keys()), belief_metadata)

            if intent is None:
                history_turns.append(f"[SYS] {utterance}")
                continue

            kb_text = extract_kb(intent, belief_metadata)

            # Skip non-general intents with no KB content
            if kb_text == "none" and intent not in ("general_greet", "general_quirky"):
                history_turns.append(f"[SYS] {utterance}")
                continue

            context = history_turns[-4:]
            history_text = " ".join(context)
            input_text = f"{history_text} [INTENT] {intent} [KB] {kb_text}".strip()

            cap = INTENT_CAP.get(intent)
            if cap and intent_counts.get(intent, 0) >= cap:
                history_turns.append(f"[SYS] {utterance}")
                continue

            pairs.append({
                "input": input_text,
                "target": utterance,
                "dialogue_id": dial_id,
                "intent": intent,
            })

            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            history_turns.append(f"[SYS] {utterance}")

    return pairs, intent_counts


def preprocess_nlg():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    zip_path = os.path.join(RAW_DIR, "MultiWOZ2.4-main", "data", "MULTIWOZ2.4.zip")
    print("Loading MultiWOZ 2.4...")
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("MULTIWOZ2.4/data.json"))
        dialogue_acts = json.loads(z.read("MULTIWOZ2.4/dialogue_acts.json"))
        val_ids = set(z.read("MULTIWOZ2.4/valListFile.json").decode().strip().splitlines())
        test_ids = set(z.read("MULTIWOZ2.4/testListFile.json").decode().strip().splitlines())

    print(f"Total dialogues: {len(data)}")

    splits = {"train": [], "val": [], "test": []}
    for dial_id in data:
        if dial_id in val_ids:
            splits["val"].append(dial_id)
        elif dial_id in test_ids:
            splits["test"].append(dial_id)
        else:
            splits["train"].append(dial_id)

    for split_name, dial_ids in splits.items():
        print(f"\nProcessing {split_name}: {len(dial_ids)} dialogues...")
        pairs, intent_counts = build_nlg_pairs(data, dialogue_acts, dial_ids)
        print(f"  {len(pairs)} NLG pairs")
        print(f"  Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"    {intent}: {count}")

        out_path = os.path.join(PROCESSED_DIR, f"nlg_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    preprocess_nlg()
