from __future__ import annotations

import argparse
import json


def validate_wikipedia_targets(targets: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_slugs: set[str] = set()
    for i, item in enumerate(targets):
        if not isinstance(item, dict):
            errors.append(f"wikipedia_targets[{i}] must be an object.")
            continue
        title = item.get("title")
        slug = item.get("slug")
        if not title or not isinstance(title, str):
            errors.append(f"wikipedia_targets[{i}] missing valid title.")
        if not slug or not isinstance(slug, str):
            errors.append(f"wikipedia_targets[{i}] missing valid slug.")
        elif slug in seen_slugs:
            errors.append(f"wikipedia_targets[{i}] duplicate slug: {slug}.")
        else:
            seen_slugs.add(slug)
    return errors


def validate_structured_sources(sources: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    required = {"id", "kind", "provider", "entry_url", "enabled"}
    for i, item in enumerate(sources):
        if not isinstance(item, dict):
            errors.append(f"structured_sources[{i}] must be an object.")
            continue
        missing = [key for key in required if key not in item]
        if missing:
            errors.append(f"structured_sources[{i}] missing keys: {', '.join(missing)}.")
        source_id = item.get("id")
        if isinstance(source_id, str):
            if source_id in seen_ids:
                errors.append(f"structured_sources[{i}] duplicate id: {source_id}.")
            else:
                seen_ids.add(source_id)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KB source registry JSON.")
    parser.add_argument("--path", required=True, help="Path to source registry JSON file.")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        payload = json.load(f)

    errors: list[str] = []

    if not isinstance(payload, dict):
        errors.append("Top-level JSON must be an object.")
    else:
        wiki_targets = payload.get("wikipedia_targets", [])
        structured_sources = payload.get("structured_sources", [])
        if not isinstance(wiki_targets, list):
            errors.append("wikipedia_targets must be a list.")
        else:
            errors.extend(validate_wikipedia_targets(wiki_targets))
        if not isinstance(structured_sources, list):
            errors.append("structured_sources must be a list.")
        else:
            errors.extend(validate_structured_sources(structured_sources))

    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    wiki_count = len(payload.get("wikipedia_targets", []))
    structured_count = len(payload.get("structured_sources", []))
    print("Validation PASSED.")
    print(f"- wikipedia_targets: {wiki_count}")
    print(f"- structured_sources: {structured_count}")


if __name__ == "__main__":
    main()
