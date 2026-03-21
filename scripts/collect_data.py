from __future__ import annotations

import argparse
import json
import os
import re
import time

import wikipedia

RAW_DIR   = "kb/raw"
CLEAN_DIR = "kb/clean"
SENTENCES = 20
TARGETS_PATH = "data/kb_sources_bucharest.json"


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[note \d+\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fetch_and_save(title: str, slug: str, sentences: int = SENTENCES) -> dict:
    raw_path   = os.path.join(RAW_DIR,   f"{slug}.txt")
    clean_path = os.path.join(CLEAN_DIR, f"{slug}.txt")

    status = {"slug": slug, "title": title, "words": 0,
               "status": "OK", "error": ""}

    try:
        raw_text = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            raw_text = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
            status["title"] = e.options[0]
        except Exception as inner:
            status["status"] = "FAIL"
            status["error"]  = str(inner)
            return status
    except wikipedia.exceptions.PageError:
        status["status"] = "FAIL"
        status["error"]  = "Page not found"
        return status
    except Exception as e:
        status["status"] = "FAIL"
        status["error"]  = str(e)
        return status

    cleaned = clean_text(raw_text)

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    status["words"] = len(cleaned.split())
    return status


def load_targets(path: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        items = payload.get("wikipedia_targets")
        if items is None:
            items = payload.get("targets")
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError(f"Invalid targets format in {path}: expected list.")

    targets: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid target entry in {path}: expected object, got {type(item)}.")
        title = item.get("title")
        slug = item.get("slug")
        if not title or not slug:
            raise ValueError(f"Invalid target entry in {path}: missing title or slug.")
        targets.append((title, slug))
    return targets


def print_summary(results: list[dict]) -> None:
    ok   = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] == "FAIL"]

    print("\n" + "=" * 64)
    print(f"  Data collection complete: {len(ok)} OK  |  {len(fail)} FAILED")
    print("=" * 64)
    print(f"  {'Slug':<30} {'Words':>6}  Status")
    print(f"  {'-'*30} {'-'*6}  ------")
    for r in results:
        marker = "✓" if r["status"] == "OK" else "✗"
        err    = f"  [{r['error']}]" if r["error"] else ""
        print(f"  {marker} {r['slug']:<30} {r['words']:>6}{err}")
    print("=" * 64)
    total_words = sum(r["words"] for r in ok)
    print(f"  Total words in KB: {total_words:,}")
    print("=" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect KB documents from Wikipedia.")
    parser.add_argument(
        "--targets",
        default=TARGETS_PATH,
        help=f"Path to targets/source-registry JSON file (default: {TARGETS_PATH})",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=SENTENCES,
        help=f"Number of summary sentences per page (default: {SENTENCES})",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove existing .txt files in kb/raw and kb/clean before fetching new targets.",
    )
    args = parser.parse_args()

    os.makedirs(RAW_DIR,   exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    if args.clear_output:
        for dirname in (RAW_DIR, CLEAN_DIR):
            for fname in os.listdir(dirname):
                if fname.endswith(".txt"):
                    os.remove(os.path.join(dirname, fname))
        print("  Cleared previous KB text files from kb/raw and kb/clean.")

    wikipedia.set_lang("en")
    targets = load_targets(args.targets)

    results = []
    for i, (title, slug) in enumerate(targets, 1):
        print(f"  [{i:02d}/{len(targets)}] Fetching: {title} ...", end=" ", flush=True)
        result = fetch_and_save(title, slug, sentences=args.sentences)
        print(f"{result['status']}  ({result['words']} words)")
        results.append(result)

        if i < len(targets):
            time.sleep(0.5)

    print_summary(results)


if __name__ == "__main__":
    main()
