"""Preprocess MultiWOZ 2.4 into DST training pairs.

Filters to restaurant + attraction domains.
Produces (dialogue_history, belief_state) pairs per turn.
"""

import json
import os
import zipfile

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

TARGET_DOMAINS = {"restaurant", "attraction"}

# Slots we care about per domain
DOMAIN_SLOTS = {
    "restaurant": {
        "semi": ["food", "pricerange", "name", "area"],
        "book": ["time", "day", "people"],
    },
    "attraction": {
        "semi": ["type", "name", "area"],
    },
}


def load_multiwoz():
    """Load all data + split lists from the inner zip."""
    zip_path = os.path.join(RAW_DIR, "MultiWOZ2.4-main", "data", "MULTIWOZ2.4.zip")

    with zipfile.ZipFile(zip_path, "r") as z:
        data = json.loads(z.read("MULTIWOZ2.4/data.json"))

        val_ids = set(
            z.read("MULTIWOZ2.4/valListFile.json").decode().strip().splitlines()
        )
        test_ids = set(
            z.read("MULTIWOZ2.4/testListFile.json").decode().strip().splitlines()
        )

    return data, val_ids, test_ids


def get_split(dial_id, val_ids, test_ids):
    """Determine which split a dialogue belongs to."""
    if dial_id in val_ids:
        return "val"
    if dial_id in test_ids:
        return "test"
    return "train"


def has_target_domain(dialogue):
    """Check if dialogue involves restaurant or attraction."""
    goal = dialogue.get("goal", {})
    for domain in TARGET_DOMAINS:
        domain_goal = goal.get(domain, {})
        if domain_goal and any(v for v in domain_goal.values() if v):
            return True
    return False


def extract_belief_state(metadata):
    """Extract belief state from a system turn's metadata.

    Returns a sorted comma-separated string like:
        restaurant-food=italian, restaurant-area=centre
    """
    slots = []

    for domain in sorted(TARGET_DOMAINS):
        domain_meta = metadata.get(domain, {})
        slot_groups = DOMAIN_SLOTS.get(domain, {})

        for group_name, slot_names in slot_groups.items():
            group_data = domain_meta.get(group_name, {})
            for slot in slot_names:
                value = group_data.get(slot, "")
                if value and value not in ("", "not mentioned"):
                    slots.append(f"{domain}-{slot}={value}")

    return ", ".join(slots) if slots else "none"


def build_training_pairs(data, dial_ids):
    """Build (history, belief_state) pairs from a set of dialogues."""
    pairs = []

    for dial_id in dial_ids:
        dialogue = data[dial_id]
        log = dialogue["log"]
        history = []

        for i, turn in enumerate(log):
            utterance = turn["text"].strip()

            if i % 2 == 0:
                # User turn (even indices)
                history.append(f"[USR] {utterance}")
            else:
                # System turn (odd indices) — has metadata with belief state
                history.append(f"[SYS] {utterance}")

                belief_str = extract_belief_state(turn.get("metadata", {}))

                pairs.append({
                    "input": " ".join(history) + " Generate belief state:",
                    "target": belief_str,
                    "dialogue_id": dial_id,
                    "turn_num": i,
                })

    return pairs


def preprocess_and_save():
    """Process all splits and save."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading MultiWOZ 2.4...")
    data, val_ids, test_ids = load_multiwoz()
    print(f"Total dialogues: {len(data)}")

    # Filter to target domains and split
    splits = {"train": [], "val": [], "test": []}
    for dial_id, dialogue in data.items():
        if has_target_domain(dialogue):
            split = get_split(dial_id, val_ids, test_ids)
            splits[split].append(dial_id)

    for split_name, dial_ids in splits.items():
        print(f"\n{split_name}: {len(dial_ids)} dialogues")
        pairs = build_training_pairs(data, dial_ids)
        print(f"  {len(pairs)} training pairs")

        out_path = os.path.join(PROCESSED_DIR, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Files saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_and_save()
